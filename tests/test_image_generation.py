"""Image (and video) as task types: coding/docs/general have a Models row each,
and so do image and video. Image generation goes through a paid cloud image
model the user chose (the orchestrator is only offered `generate_image` then),
is saved as a binary file after the user approves it, and is billed like any
other paid delegate. Video has a row but can't be chosen yet.
"""
import base64
import io
import json
import struct
import zlib
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

import localforge.cli as cli_module
import localforge.tools as tools
from localforge import config, trust
from localforge import delegate_target as dt
from localforge.backends import cloud
from localforge.hardware import HardwareProfile
from localforge.serve import StdioServer
from localforge.tools import Dispatcher, build_tool_schemas
from localforge.workspace import Workspace

runner = CliRunner()


def _png(w=4, h=4):
    raw = b"".join(b"\x00" + bytes((255, 0, 0)) * w for _ in range(h))

    def chunk(t, d):
        return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)

    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b"")


def _hw():
    return HardwareProfile(os="Linux", arch="x86_64", cpu_cores=8, ram_gb=32, free_disk_gb=100, gpus=[])


class FakeImages:
    def __init__(self, data=None, cost=0.04, error=None):
        self.data, self.cost, self.error, self.calls = data if data is not None else _png(), cost, error, []

    def generate(self, model, prompt, provider=None):
        self.calls.append((model, prompt, provider))
        if self.error:
            raise self.error
        return cloud.GeneratedImage(data=self.data, mime_type=cloud.sniff_image_type(self.data) or "image/png", cost_usd=self.cost)


@pytest.fixture
def openai_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")


def _dispatcher(tmp_path, approve=True, fake=None, monkeypatch=None):
    ws = Workspace(tmp_path, approver=lambda kind, title, detail: approve)
    return Dispatcher(_hw(), installed=set(), workspace=ws), ws


# --- the task types ---------------------------------------------------------------


def test_image_and_video_are_task_types_beside_the_text_ones():
    assert dt.ALL_MODALITIES == ("coding", "docs", "general", "image", "video")
    assert dt.MODALITIES == ("coding", "docs", "general")  # text work is unchanged


def test_image_is_off_by_default_and_video_is_not_available():
    assert "off" in dt.describe_modality("image")
    assert dt.describe_modality("video") == "not available yet"


def test_only_a_paid_api_image_model_can_be_chosen(openai_key):
    target = dt.apply("image", "api:openai:gpt-image-1")
    assert target == dt.DelegateTarget(kind="api", provider="openai", model="gpt-image-1")
    assert dt.get("image") == target
    assert "gpt-image-1" in dt.describe_modality("image")


@pytest.mark.parametrize(
    "value, why",
    [
        ("qwen2.5-coder:7b", "can't generate images"),
        ("cli:anthropic:claude-opus-5", "can't generate images"),
        ("api:anthropic:claude-opus-5", "no image generation"),
        ("api:openai:not-a-model", "isn't a known"),
    ],
)
def test_bad_image_targets_are_refused(openai_key, value, why):
    with pytest.raises(dt.InvalidTarget) as info:
        dt.apply("image", value)
    assert why in str(info.value)
    assert dt.get("image") == dt.AUTO


def test_an_image_model_needs_its_providers_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(dt.InvalidTarget, match="No API key"):
        dt.apply("image", "api:openai:gpt-image-1")


def test_image_can_be_turned_off_again(openai_key):
    dt.apply("image", "api:openai:gpt-image-1")
    assert dt.apply("image", "auto") is dt.AUTO


def test_video_cannot_be_chosen_yet(openai_key):
    with pytest.raises(dt.InvalidTarget, match="isn't available yet"):
        dt.apply("video", "api:openai:gpt-image-1")
    assert dt.apply("video", "auto") is dt.AUTO


def test_every_offered_image_model_is_one_litellms_own_list_knows():
    import litellm

    known = {k for k, v in litellm.model_cost.items() if v.get("mode") == "image_generation"}
    for provider, models in config.IMAGE_MODEL_CHOICES.items():
        for model in models:
            assert model in known, f"{model} isn't a LiteLLM image-generation model"


# --- the orchestrator's tool ------------------------------------------------------


def test_the_image_tool_is_only_offered_once_an_image_model_is_chosen(openai_key):
    names = lambda: {t["function"]["name"] for t in build_tool_schemas(_hw(), [], set())}  # noqa: E731
    assert "generate_image" not in names()
    dt.apply("image", "api:openai:gpt-image-1")
    assert "generate_image" in names()
    dt.apply("image", "auto")
    assert "generate_image" not in names()


def test_generating_saves_the_image_after_approval_and_bills_it(tmp_path, openai_key, monkeypatch):
    dt.apply("image", "api:openai:gpt-image-1")
    fake = FakeImages(cost=0.04)
    monkeypatch.setattr(tools, "_image_backend", fake)
    seen = []
    d, ws = _dispatcher(tmp_path)
    ws.approver = lambda kind, title, detail: seen.append((kind, title, detail)) or True
    out = d.dispatch("generate_image", {"instructions": "a red square", "path": "assets/red.png"})
    assert (tmp_path / "assets" / "red.png").read_bytes() == _png()
    assert out.startswith("Created assets/red.png (image/png")
    assert fake.calls == [("gpt-image-1", "a red square", "openai")]
    # asked about the cost first, then about saving the file
    assert [k for k, _t, _d in seen] == ["spend", "write"]
    assert seen[1][0] == "write" and "assets/red.png" in seen[1][1] and "image/png" in seen[1][2]
    assert d.delegate_cost_usd == pytest.approx(0.04) and d.local_tokens_generated == 0
    assert d._last_delegate_model == "gpt-image-1"


def test_a_declined_image_is_not_written_but_the_orchestrator_is_told_it_was_billed(tmp_path, openai_key, monkeypatch):
    dt.apply("image", "api:openai:gpt-image-1")
    monkeypatch.setattr(tools, "_image_backend", FakeImages())
    d, _ = _dispatcher(tmp_path, approve=False)
    out = d.dispatch("generate_image", {"instructions": "x", "path": "a.png"})
    assert not (tmp_path / "a.png").exists()
    assert "declined" in out and "billed" in out


def test_the_file_extension_follows_the_real_image_type(tmp_path, openai_key, monkeypatch):
    dt.apply("image", "api:openai:gpt-image-1")
    monkeypatch.setattr(tools, "_image_backend", FakeImages())
    d, _ = _dispatcher(tmp_path)
    out = d.dispatch("generate_image", {"instructions": "x", "path": "pic.jpg"})
    assert (tmp_path / "pic.png").exists() and not (tmp_path / "pic.jpg").exists()
    assert "saved as pic.png" in out


def test_without_an_image_model_the_call_explains_how_to_choose_one(tmp_path):
    d, _ = _dispatcher(tmp_path)
    out = d.dispatch("generate_image", {"instructions": "x", "path": "a.png"})
    assert "no image model is chosen" in out and "advanced-model image" in out


def test_missing_arguments_and_paths_outside_the_project_are_refused(tmp_path, openai_key, monkeypatch):
    dt.apply("image", "api:openai:gpt-image-1")
    fake = FakeImages()
    monkeypatch.setattr(tools, "_image_backend", fake)
    d, _ = _dispatcher(tmp_path)
    assert "needs `instructions`" in d.dispatch("generate_image", {"instructions": "x"})
    assert "Cannot write" in d.dispatch("generate_image", {"instructions": "x", "path": "../escape.png"})
    assert fake.calls == []  # nothing was billed for either


def test_a_provider_failure_is_a_failed_step_with_the_reason(tmp_path, openai_key, monkeypatch):
    dt.apply("image", "api:openai:gpt-image-1")
    monkeypatch.setattr(tools, "_image_backend", FakeImages(error=RuntimeError("content policy")))
    d, _ = _dispatcher(tmp_path)
    with pytest.raises(RuntimeError, match="gpt-image-1 image request failed: content policy"):
        d.dispatch("generate_image", {"instructions": "x", "path": "a.png"})


# --- the backend itself -----------------------------------------------------------


def test_backend_decodes_base64_and_prices_the_call(monkeypatch, openai_key):
    data = _png()
    resp = SimpleNamespace(data=[{"b64_json": base64.b64encode(data).decode()}])
    monkeypatch.setattr(cloud.litellm, "image_generation", lambda **kw: resp)
    monkeypatch.setattr(cloud.litellm, "completion_cost", lambda **kw: 0.042)
    image = cloud.CloudImageBackend().generate("gpt-image-1", "a cat", provider="openai")
    assert image.data == data and image.mime_type == "image/png" and image.cost_usd == pytest.approx(0.042)


def test_backend_refuses_something_that_is_not_an_image(monkeypatch, openai_key):
    resp = SimpleNamespace(data=[{"b64_json": base64.b64encode(b"<html>hello</html>").decode()}])
    monkeypatch.setattr(cloud.litellm, "image_generation", lambda **kw: resp)
    with pytest.raises(RuntimeError, match="isn't a PNG, JPEG or WebP"):
        cloud.CloudImageBackend().generate("gpt-image-1", "x", provider="openai")


def test_backend_does_not_fetch_a_non_https_link(monkeypatch, openai_key):
    resp = SimpleNamespace(data=[{"url": "http://example.com/x.png"}])
    monkeypatch.setattr(cloud.litellm, "image_generation", lambda **kw: resp)
    with pytest.raises(RuntimeError, match="isn't https"):
        cloud.CloudImageBackend().generate("gpt-image-1", "x", provider="openai")


def test_backend_needs_the_providers_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(cloud.CloudProviderNotConfigured):
        cloud.CloudImageBackend().generate("gpt-image-1", "x", provider="openai")


def test_a_cost_pricing_failure_does_not_fail_the_image(monkeypatch, openai_key):
    resp = SimpleNamespace(data=[{"b64_json": base64.b64encode(_png()).decode()}])
    monkeypatch.setattr(cloud.litellm, "image_generation", lambda **kw: resp)
    monkeypatch.setattr(cloud.litellm, "completion_cost", lambda **kw: (_ for _ in ()).throw(ValueError("unpriced")))
    assert cloud.CloudImageBackend().generate("gpt-image-1", "x", provider="openai").cost_usd == 0.0


def test_image_types_are_sniffed_from_the_bytes_not_trusted():
    assert cloud.sniff_image_type(_png()) == "image/png"
    assert cloud.sniff_image_type(b"\xff\xd8\xff\xe0abc") == "image/jpeg"
    assert cloud.sniff_image_type(b"RIFF\x00\x00\x00\x00WEBPVP8 ") == "image/webp"
    assert cloud.sniff_image_type(b"GIF89a") is None


# --- the workspace write ----------------------------------------------------------


def test_write_bytes_asks_first_and_says_when_it_replaces(tmp_path):
    seen = []
    ws = Workspace(tmp_path, approver=lambda kind, title, detail: seen.append((title, detail)) or True)
    ws.write_bytes("a.png", b"x" * 2500, "image/png")
    ws.write_bytes("a.png", b"y" * 2500, "image/png")
    assert seen[0][0] == "Create a.png" and "2 KB" in seen[0][1]
    assert seen[1][0] == "Update a.png" and "replaces the existing file" in seen[1][1]


def test_write_bytes_is_confined_to_the_project(tmp_path):
    from localforge.workspace import WorkspaceError

    ws = Workspace(tmp_path, approver=lambda *a: True)
    with pytest.raises(WorkspaceError):
        ws.write_bytes("../out.png", b"x", "image/png")


# --- the summary, CLI and desktop -------------------------------------------------


def test_a_generated_image_shows_up_in_the_task_summary():
    from localforge import task_summary as ts

    log = ts.TaskLog()
    log.record("generate_image", "assets/red.png", "ok", "Created assets/red.png (image/png, 1 KB).", model="gpt-image-1")
    s = ts.build(log, "completed")
    assert s["files"] == [{"action": "Created", "path": "assets/red.png", "by": "gpt-image-1"}]
    assert s["delegations"][0]["model"] == "gpt-image-1"


def test_cli_lists_image_and_video_and_sets_an_image_model(openai_key, monkeypatch):
    monkeypatch.setattr(cli_module, "detect_hardware", _hw)
    monkeypatch.setattr(cli_module, "_installed_model_names", lambda ollama: set())
    shown = runner.invoke(cli_module.app, ["advanced-model"])
    assert "image:" in shown.output and "off" in shown.output and "video: not available yet" in shown.output
    ok = runner.invoke(cli_module.app, ["advanced-model", "image", "api:openai:gpt-image-1"])
    assert ok.exit_code == 0 and "image generation now uses gpt-image-1" in ok.output and "costs money" in ok.output
    bad = runner.invoke(cli_module.app, ["advanced-model", "video", "api:openai:gpt-image-1"])
    assert bad.exit_code == 1 and "isn't available yet" in bad.output
    off = runner.invoke(cli_module.app, ["advanced-model", "image", "auto"])
    assert off.exit_code == 0 and "off" in off.output


def _server(tmp_path):
    out = io.StringIO()
    (tmp_path / "scratch").mkdir(exist_ok=True)
    trust.trust(tmp_path)
    return StdioServer(tmp_path, "m", out=out, run_fn=lambda *a, **k: None, scratch_root=tmp_path / "scratch", conversation=object()), out


def _last(out, kind):
    return [e for e in (json.loads(l) for l in out.getvalue().splitlines() if l.strip()) if e["type"] == kind][-1]


def test_desktop_snapshot_has_image_and_video_rows(tmp_path):
    server, out = _server(tmp_path)
    server.handle({"type": "advanced_model_request"})
    targets = _last(out, "advanced_model")["targets"]
    assert set(targets) == set(dt.ALL_MODALITIES)
    assert "off" in targets["image"]["description"] and targets["video"]["description"] == "not available yet"


def test_desktop_image_options_list_models_of_providers_with_a_key(tmp_path, openai_key, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    server, out = _server(tmp_path)
    server.handle({"type": "delegate_options_request", "modality": "image"})
    ev = _last(out, "delegate_options")
    assert ev["local"] == []
    assert {(c["provider"], c["model"]) for c in ev["cloud"]} == {("openai", "gpt-image-1"), ("openai", "gpt-image-1-mini")}
    assert all(c["kind"] == "api" for c in ev["cloud"])


def test_desktop_can_set_and_clear_the_image_model_and_gets_video_refused(tmp_path, openai_key):
    server, out = _server(tmp_path)
    server.handle({"type": "set_delegate_target", "modality": "image", "target": "api:openai:gpt-image-1"})
    assert _last(out, "advanced_model")["targets"]["image"]["target"] == "api:openai:gpt-image-1"
    server.handle({"type": "set_delegate_target", "modality": "video", "target": "api:openai:gpt-image-1"})
    assert "isn't available yet" in _last(out, "error")["message"]
    server.handle({"type": "user_message", "text": "/advanced-model image auto"})
    assert _last(out, "advanced_model")["targets"]["image"]["target"] == "auto"


# --- the whole loop ---------------------------------------------------------------


def test_the_orchestrator_can_generate_an_image_end_to_end(tmp_path, openai_key, monkeypatch):
    """A real run(): the tool is offered because an image model is chosen, the
    orchestrator calls it, the file is written after approval, the paid cost is
    counted as a paid delegate, and the summary lists the image."""
    from unittest.mock import MagicMock, patch

    import localforge.orchestrator as orch
    from localforge import task_summary as ts

    dt.apply("image", "api:openai:gpt-image-1")
    monkeypatch.setattr(tools, "_image_backend", FakeImages(cost=0.04))
    offered = []

    def call(name, args):
        msg = MagicMock()
        msg.tool_calls = [MagicMock(id="c", function=MagicMock(arguments=json.dumps(args)))]
        msg.tool_calls[0].function.name = name
        msg.model_dump.return_value = {"role": "assistant", "content": None}
        return MagicMock(choices=[MagicMock(message=msg)], usage=None)

    def final():
        msg = MagicMock(tool_calls=None, content="Made the hero image.")
        msg.model_dump.return_value = {"role": "assistant", "content": "Made the hero image."}
        return MagicMock(choices=[MagicMock(message=msg)], usage=None)

    replies = iter([call("generate_image", {"instructions": "a red square", "path": "assets/hero.png"}), final(), final(), final()])

    def completion(**kw):
        offered.append({t["function"]["name"] for t in kw["tools"]})
        return next(replies)

    ws = Workspace(tmp_path, approver=lambda *a: True)
    with (
        patch.object(orch, "completion", side_effect=completion),
        patch.object(orch, "_installed_models", return_value=set()),
        patch.object(orch.litellm, "completion_cost", return_value=0.0),
    ):
        result = orch.run("add a hero image", "gpt-5", hardware=_hw(), workspace=ws, conversation=orch.Conversation())
    assert "generate_image" in offered[0]
    assert (tmp_path / "assets" / "hero.png").read_bytes() == _png()
    assert result.stats.delegate_cost_usd == pytest.approx(0.04)
    summary = ts.build(result.stats.log, "completed")
    assert summary["files"] == [{"action": "Created", "path": "assets/hero.png", "by": "gpt-image-1"}]


def test_a_generated_image_is_not_an_unchecked_change():
    """The completion check sends the orchestrator back to test or read back
    changed files; it can't do either for a picture."""
    import localforge.orchestrator as orch

    progress = orch._Progress()
    progress.note("generate_image", {"path": "a.png"}, "Created a.png (image/png, 1 KB).", step=1)
    assert orch._completion_gaps(progress, "done", "") is None
    progress.note("delegate_coding_task", {"path": "a.py"}, "Created a.py (+3 -0)", step=2)
    assert orch._completion_gaps(progress, "done", "") is not None  # code still is


# --- image through a provider's CLI login (Codex on a ChatGPT login) -----------------


@pytest.fixture
def codex_present(monkeypatch):
    monkeypatch.setattr(cloud.cli_transport, "available", lambda provider: provider == "openai")
    monkeypatch.setattr("localforge.cli_transport.available", lambda provider: provider == "openai")


def test_a_codex_login_can_be_chosen_for_images(codex_present):
    target = dt.apply("image", "cli:openai:codex-image")
    assert target == dt.DelegateTarget(kind="cli", provider="openai", model="codex-image")
    assert "your CLI login" in dt.describe_modality("image")


def test_claudes_login_cannot_generate_images_and_the_message_says_why(monkeypatch):
    monkeypatch.setattr("localforge.cli_transport.available", lambda provider: True)
    with pytest.raises(dt.InvalidTarget, match="Claude has no image model"):
        dt.apply("image", "cli:anthropic:claude-opus-5")
    with pytest.raises(dt.InvalidTarget, match="Only OpenAI's Codex login can"):
        dt.apply("image", "cli:gemini:gemini-2.5-pro")


def test_a_codex_target_needs_the_codex_cli_installed(monkeypatch):
    monkeypatch.setattr("localforge.cli_transport.available", lambda provider: False)
    with pytest.raises(dt.InvalidTarget):
        dt.apply("image", "cli:openai:codex-image")
    assert dt.get("image") == dt.AUTO


def _fake_codex(monkeypatch, write=None, returncode=0, stdout="done", codex_home=None, timeout=False):
    """Stands in for `codex exec`: runs in the scratch cwd, may write an image there."""
    calls = []

    def run(cmd, cwd=None, **kw):
        calls.append({"cmd": cmd, "cwd": cwd, **kw})
        if timeout:
            raise cloud.subprocess.TimeoutExpired(cmd, 1)
        if write is not None:
            (cwd / "image.png").write_bytes(write) if hasattr(cwd, "joinpath") else None
        return SimpleNamespace(returncode=returncode, stdout=stdout, stderr="")

    monkeypatch.setattr(cloud.subprocess, "run", run)
    if codex_home is not None:
        monkeypatch.setenv("CODEX_HOME", str(codex_home))
    return calls


def test_the_cli_backend_runs_codex_exec_in_a_scratch_folder_and_returns_the_image(monkeypatch, codex_present, tmp_path):
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "home"))
    calls = _fake_codex(monkeypatch, write=_png())
    image = cloud.CloudImageCliBackend().generate("codex-image", "a red square", provider="openai")
    assert image.data == _png() and image.mime_type == "image/png" and image.cost_usd == 0.0
    cmd = calls[0]["cmd"]
    assert cmd[:2] == ["codex", "exec"] and "workspace-write" in cmd and "a red square" in cmd[-1]
    assert calls[0]["stdin"] == cloud.subprocess.DEVNULL  # never waits on the terminal
    assert not calls[0]["cwd"].exists()  # the scratch folder is cleaned up


def test_the_cli_backend_finds_an_image_codex_saved_under_its_own_folder(monkeypatch, codex_present, tmp_path):
    home = tmp_path / "codexhome"
    calls = _fake_codex(monkeypatch, codex_home=home)

    def run(cmd, cwd=None, **kw):  # saved under $CODEX_HOME rather than in the cwd
        (home / "generated").mkdir(parents=True)
        (home / "generated" / "abc.png").write_bytes(_png())
        return SimpleNamespace(returncode=0, stdout="done", stderr="")

    monkeypatch.setattr(cloud.subprocess, "run", run)
    assert cloud.CloudImageCliBackend().generate("codex-image", "x", provider="openai").data == _png()


def test_an_older_image_in_codexs_folder_is_not_mistaken_for_the_new_one(monkeypatch, codex_present, tmp_path):
    import os

    home = tmp_path / "codexhome"
    (home / "old").mkdir(parents=True)
    old = home / "old" / "previous.png"
    old.write_bytes(_png())
    os.utime(old, (1_000_000, 1_000_000))  # long ago
    _fake_codex(monkeypatch, codex_home=home, stdout="I couldn't do that")
    with pytest.raises(RuntimeError, match="produced no image file"):
        cloud.CloudImageCliBackend().generate("codex-image", "x", provider="openai")


def test_no_image_means_an_error_that_says_what_codex_printed_and_how_to_log_in(monkeypatch, codex_present, tmp_path):
    _fake_codex(monkeypatch, codex_home=tmp_path / "h", returncode=1, stdout="Error: not signed in")
    with pytest.raises(RuntimeError) as info:
        cloud.CloudImageCliBackend().generate("codex-image", "x", provider="openai")
    assert "exit 1" in str(info.value) and "not signed in" in str(info.value) and "codex login" in str(info.value)


def test_a_file_that_isnt_an_image_is_refused(monkeypatch, codex_present, tmp_path):
    _fake_codex(monkeypatch, write=b"<html>oops</html>", codex_home=tmp_path / "h")
    with pytest.raises(RuntimeError, match="isn't a PNG, JPEG or WebP"):
        cloud.CloudImageCliBackend().generate("codex-image", "x", provider="openai")


def test_a_hung_codex_is_stopped(monkeypatch, codex_present, tmp_path):
    _fake_codex(monkeypatch, timeout=True, codex_home=tmp_path / "h")
    with pytest.raises(RuntimeError, match="took longer than"):
        cloud.CloudImageCliBackend().generate("codex-image", "x", provider="openai")


def test_the_cli_backend_needs_the_cli(monkeypatch):
    monkeypatch.setattr(cloud.cli_transport, "available", lambda p: False)
    with pytest.raises(cloud.CloudProviderNotConfigured):
        cloud.CloudImageCliBackend().generate("codex-image", "x", provider="openai")
    with pytest.raises(cloud.CloudProviderNotConfigured, match="can't generate images"):
        cloud.CloudImageCliBackend().generate("m", "x", provider="anthropic")


def test_a_codex_target_routes_to_the_cli_backend_and_bills_nothing(tmp_path, codex_present, monkeypatch):
    dt.apply("image", "cli:openai:codex-image")

    class Cli:
        def __init__(self):
            self.calls = []

        def generate(self, model, prompt, provider=None):
            self.calls.append((model, prompt, provider))
            return cloud.GeneratedImage(data=_png(), mime_type="image/png", cost_usd=0.0)

    cli_backend, api_backend = Cli(), FakeImages()
    monkeypatch.setattr(tools, "_image_cli_backend", cli_backend)
    monkeypatch.setattr(tools, "_image_backend", api_backend)
    assert "generate_image" in {t["function"]["name"] for t in build_tool_schemas(_hw(), [], set())}
    d, _ = _dispatcher(tmp_path)
    out = d.dispatch("generate_image", {"instructions": "a cat", "path": "cat.png"})
    assert cli_backend.calls == [("codex-image", "a cat", "openai")] and api_backend.calls == []
    assert (tmp_path / "cat.png").read_bytes() == _png()
    assert "Included in the user's plan" in out and d.delegate_cost_usd == 0.0
    assert d.delegate_models["codex-image"]["kind"] == "cli" and d.delegate_models["codex-image"]["roles"] == ["image"]


def test_the_desktop_offers_codex_for_images_when_the_cli_is_installed(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.setattr("localforge.serve.cli_transport.available", lambda p: p == "openai")
    server, out = _server(tmp_path)
    server.handle({"type": "delegate_options_request", "modality": "image"})
    cloud_opts = _last(out, "delegate_options")["cloud"]
    assert cloud_opts == [{"kind": "cli", "provider": "openai", "model": "codex-image"}]


def test_cli_says_the_codex_route_is_experimental(codex_present, monkeypatch):
    monkeypatch.setattr(cli_module, "detect_hardware", _hw)
    monkeypatch.setattr(cli_module, "_installed_model_names", lambda ollama: set())
    out = runner.invoke(cli_module.app, ["advanced-model", "image", "cli:openai:codex-image"])
    assert out.exit_code == 0 and "Experimental" in out.output and "plan's image allowance" in out.output


# --- paid usage is tracked per model ----------------------------------------------


def test_each_paid_model_is_counted_under_its_own_name(tmp_path, openai_key, monkeypatch):
    dt.apply("image", "api:openai:gpt-image-1")
    monkeypatch.setattr(tools, "_image_backend", FakeImages(cost=0.04))
    d, _ = _dispatcher(tmp_path)
    d.dispatch("generate_image", {"instructions": "a", "path": "a.png"})
    d.dispatch("generate_image", {"instructions": "b", "path": "b.png"})
    slot = d.delegate_models["gpt-image-1"]
    assert slot["runs"] == 2 and slot["cost_usd"] == pytest.approx(0.08) and slot["roles"] == ["image"] and slot["provider"] == "openai"


def test_paid_text_delegates_are_tracked_by_model_and_reach_the_run_stats(tmp_path, monkeypatch):
    import localforge.orchestrator as orch
    from localforge.backends import BACKENDS

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    dt.apply("coding", "api:anthropic:claude-haiku-4-5")
    dt.apply("docs", "api:anthropic:claude-haiku-4-5")

    class Stub:
        def ensure_available(self, *a, **k):
            pass

        def generate(self, name, prompt, **k):
            return {"type": "text", "content": "```python\nprint(1)\n```\nNOTES: none", "tokens": 50, "cost_usd": 0.01, "notional_cost_usd": 0.0}

    monkeypatch.setitem(BACKENDS, "api", Stub())
    d = Dispatcher(_hw(), installed=set(), workspace=Workspace(tmp_path, approver=lambda *a: True))
    d.dispatch("delegate_coding_task", {"instructions": "x", "path": "a.py"})
    d.dispatch("delegate_docs_task", {"instructions": "y"})
    slot = d.delegate_models["claude-haiku-4-5"]
    assert slot["roles"] == ["coding", "docs"] and slot["runs"] == 2 and slot["tokens"] == 100 and slot["cost_usd"] == pytest.approx(0.02)
    stats = orch.RunStats()
    orch._copy_dispatch_stats(stats, d)
    assert stats.delegate_models["claude-haiku-4-5"]["runs"] == 2
    import dataclasses

    assert dataclasses.asdict(stats)["delegate_models"]["claude-haiku-4-5"]["tokens"] == 100  # reaches the desktop as-is
