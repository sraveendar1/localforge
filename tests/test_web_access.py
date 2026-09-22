"""Option A for web access: the orchestrator researches through localforge's
own web_search/fetch_url and passes findings to local models (which have no
internet), and a CLI orchestrator's own agent tools are switched off.

Reported: the orchestrator could reach the internet but local models could
not. The cause was that `claude -p` is a full agent with its own tools --
verified live it also had Bash/Edit/Write in the user's folder and their
Slack/M365/Docs connectors -- while under an API key the orchestrator had
no web access at all.
"""

import os
import socket
from unittest.mock import MagicMock, patch

import httpx
import pytest

from localforge import cli_transport, config, web
from localforge.hardware import HardwareProfile
from localforge.tools import ActivityHooks, Dispatcher, build_tool_schemas


def _hw() -> HardwareProfile:
    return HardwareProfile(os="Linux", arch="x86_64", cpu_cores=8, ram_gb=32, free_disk_gb=100, gpus=[])


def _public_dns(host, port, *a, **kw):
    return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.215.14", port))]


def _mock_client(handler):
    real = httpx.Client

    def factory(*args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(handler)
        return real(*args, **kwargs)

    return factory


# --- the tools are offered and routed -------------------------------------


def test_web_tools_are_always_offered_to_the_orchestrator():
    names = {s["function"]["name"] for s in build_tool_schemas(_hw(), catalog=[])}
    assert {"web_search", "fetch_url"} <= names


def test_dispatcher_runs_web_tools_itself_and_reports_them():
    seen = []
    dispatcher = Dispatcher(_hw(), catalog=[], hooks=ActivityHooks(on_tool=lambda t, a: seen.append((t, a))))
    with patch.object(web, "web_search", lambda q: f"results for {q}"):
        out = dispatcher.dispatch("web_search", {"query": "fastapi lifespan"})
    assert out == "results for fastapi lifespan"
    assert seen == [("web_search", "fastapi lifespan")]
    assert dispatcher.local_tokens_generated == 0  # no local model involved


def test_a_web_failure_goes_back_to_the_orchestrator_as_text():
    def boom(url):
        raise web.WebError("example.invalid returned HTTP 404")

    with patch.object(web, "fetch_url", boom):
        out = Dispatcher(_hw(), catalog=[]).dispatch("fetch_url", {"url": "https://example.invalid"})
    assert out == "fetch_url failed: example.invalid returned HTTP 404"


def test_system_prompt_says_local_models_have_no_internet():
    import localforge.orchestrator as orch

    assert "no internet access" in orch.SYSTEM_PROMPT
    assert "web_search" in orch.SYSTEM_PROMPT and "fetch_url" in orch.SYSTEM_PROMPT


# --- fetch_url --------------------------------------------------------------


@pytest.mark.parametrize(
    "url",
    [
        "http://localhost:11434/api/tags",
        "http://127.0.0.1/",
        "http://192.168.1.1/",
        "http://10.0.0.5/",
        "http://169.254.169.254/latest/meta-data",
        "file:///etc/passwd",
        "ftp://example.com/x",
    ],
)
def test_fetch_refuses_local_network_and_non_http(url):
    with pytest.raises(web.WebError):
        web.fetch_url(url)


def test_fetch_checks_every_redirect_hop():
    def handler(request):
        return httpx.Response(302, headers={"location": "http://127.0.0.1:11434/api/tags"})

    real_dns = socket.getaddrinfo

    def dns(host, port, *a, **kw):
        return _public_dns(host, port) if host == "example.com" else real_dns(host, port, *a, **kw)

    with patch.object(web.socket, "getaddrinfo", dns), patch.object(web.httpx, "Client", _mock_client(handler)):
        with pytest.raises(web.WebError, match="non-public"):
            web.fetch_url("https://example.com/")


def test_fetch_returns_readable_text_without_scripts():
    html = (
        "<html><head><title>Docs  Page</title><style>.x{}</style></head><body>"
        "<script>alert(1)</script><h1>Install</h1><p>pip install thing</p></body></html>"
    )

    def handler(request):
        return httpx.Response(200, headers={"content-type": "text/html; charset=utf-8"}, text=html)

    with patch.object(web.socket, "getaddrinfo", _public_dns), patch.object(web.httpx, "Client", _mock_client(handler)):
        out = web.fetch_url("https://example.com/docs")

    assert "Title: Docs Page" in out
    assert "Install\npip install thing" in out
    assert "alert" not in out and ".x{}" not in out


def test_fetch_truncates_huge_pages_and_says_so():
    def handler(request):
        return httpx.Response(200, headers={"content-type": "text/plain"}, text="a" * (web.MAX_PAGE_CHARS + 500))

    with patch.object(web.socket, "getaddrinfo", _public_dns), patch.object(web.httpx, "Client", _mock_client(handler)):
        out = web.fetch_url("https://example.com/big.txt")
    assert f"[truncated to the first {web.MAX_PAGE_CHARS} characters]" in out


def test_fetch_rejects_binary_content():
    def handler(request):
        return httpx.Response(200, headers={"content-type": "application/zip"}, content=b"PK\x03\x04")

    with patch.object(web.socket, "getaddrinfo", _public_dns), patch.object(web.httpx, "Client", _mock_client(handler)):
        with pytest.raises(web.WebError, match="not a text page"):
            web.fetch_url("https://example.com/a.zip")


# --- web_search ---------------------------------------------------------------

DDG_HTML = """
<div class="result"><h2><a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fpypi.org%2Fproject%2Fhttpx%2F&rut=x">httpx &middot; PyPI</a></h2>
<a class="result__snippet" href="#">The next generation <b>HTTP</b> client.</a></div>
<div class="result"><h2><a class="result__a" href="https://www.python-httpx.org/">HTTPX</a></h2>
<a class="result__snippet" href="#">Docs.</a></div>
"""


def test_search_results_are_parsed_and_redirects_unwrapped():
    results = web.parse_results(DDG_HTML)
    assert results[0] == {"title": "httpx · PyPI", "url": "https://pypi.org/project/httpx/", "snippet": "The next generation HTTP client."}
    assert results[1]["url"] == "https://www.python-httpx.org/"


def test_search_formats_numbered_results():
    with patch.object(web.httpx, "Client", _mock_client(lambda r: httpx.Response(200, text=DDG_HTML))):
        out = web.web_search("httpx")
    assert out.startswith("Search results for 'httpx':\n1. httpx · PyPI\n   https://pypi.org/project/httpx/")


def test_search_with_no_results_explains_instead_of_failing():
    with patch.object(web.httpx, "Client", _mock_client(lambda r: httpx.Response(202, text="<html>anomaly</html>"))):
        out = web.web_search("anything")
    assert "No search results" in out and "fetch_url" in out


# --- the CLI orchestrator is fenced in ------------------------------------------


def _proc(stdout='{"result": "{\\"final_answer\\": \\"ok\\"}", "usage": {}}'):
    return MagicMock(returncode=0, stdout=stdout, stderr="")


def test_claude_runs_with_its_own_tools_and_connectors_switched_off():
    with patch.object(cli_transport, "available", return_value=True), patch.object(
        cli_transport.subprocess, "run", return_value=_proc()
    ) as run:
        cli_transport.complete("anthropic", [{"role": "user", "content": "hi"}], [])

    cmd = run.call_args.args[0]
    i = cmd.index("--tools")
    assert cmd[i + 1] == ""
    # --tools is variadic: the next arg must be an option, never the prompt
    assert cmd[i + 2].startswith("--")
    assert "--strict-mcp-config" in cmd


def test_cli_orchestrator_runs_in_an_empty_scratch_dir_with_the_prompt_on_stdin():
    """A session's prompt grows every turn: it goes over stdin (verified live
    with 340k chars), never as one argv element, and stdin is then closed so
    `claude -p` can't hang waiting on it.
    """
    with patch.object(cli_transport, "available", return_value=True), patch.object(
        cli_transport.subprocess, "run", return_value=_proc()
    ) as run:
        cli_transport.complete("anthropic", [{"role": "user", "content": "hi there"}], [])

    kwargs = run.call_args.kwargs
    assert "hi there" in kwargs["input"]
    assert not any("hi there" in arg for arg in run.call_args.args[0])
    assert kwargs["cwd"] != os.getcwd()
    assert "localforge-orchestrator-" in kwargs["cwd"]
    assert not os.path.exists(kwargs["cwd"])  # cleaned up afterwards


def test_every_cli_provider_declares_isolation_args():
    for provider, spec in config.FRONTIER_CLI_AUTH.items():
        assert "isolation_args" in spec, provider


def test_render_prompt_lists_web_tools_and_says_they_are_the_only_ones():
    tools = build_tool_schemas(_hw(), catalog=[])
    prompt = cli_transport._render_prompt([{"role": "user", "content": "x"}], tools)
    assert "- web_search(query):" in prompt and "- fetch_url(url):" in prompt
    assert "- read_file(path, offset?, limit?):" in prompt
    assert "the only ones you have" in prompt


def test_web_activity_line_renders_markup_and_keeps_brackets_in_urls(monkeypatch, tmp_path):
    from typer.testing import CliRunner

    import localforge.cli as cli_module
    from localforge.orchestrator import RunResult, RunStats

    monkeypatch.setattr(cli_module.config, "CONFIG_FILE", tmp_path / "config.env")
    monkeypatch.delenv("LOCALFORGE_AUTH_METHOD", raising=False)

    def fake_run(task, frontier_model, cli_provider=None, hooks=None, **kwargs):
        hooks.on_tool("fetch_url", "https://example.com/a?x=[1]")
        hooks.on_tool("web_search", "fastapi lifespan")
        return RunResult("done", RunStats())

    with patch.object(cli_module, "run_orchestrator", side_effect=fake_run):
        out = CliRunner().invoke(cli_module.app, ["run", "t", "-m", "gpt-5"]).output
    assert "● Fetch https://example.com/a?x=[1]" in out
    assert "● Web search fastapi lifespan" in out
    assert "[accent]" not in out
