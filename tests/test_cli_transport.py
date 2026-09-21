"""Tests for orchestrating via a provider's own logged-in CLI."""

from unittest.mock import MagicMock, patch

import pytest

from localforge import cli_transport, config
from localforge.cli_transport import CLINotAvailableError, CLIResponse
from localforge.orchestrator import RunStats, _record_frontier_usage


def _proc(stdout: str, returncode: int = 0):
    return MagicMock(stdout=stdout, stderr="", returncode=returncode)


# --- parsing a tool-call decision out of CLI text ---


def test_parses_tool_calls_from_json_envelope():
    envelope = (
        '{"result": "{\\"tool_calls\\": [{\\"name\\": \\"delegate_coding_task\\", '
        '\\"instructions\\": \\"write a parser\\"}]}", '
        '"usage": {"input_tokens": 120, "output_tokens": 45}, "total_cost_usd": 0.031}'
    )
    with patch.object(cli_transport, "available", return_value=True), patch.object(
        cli_transport.subprocess, "run", return_value=_proc(envelope)
    ):
        resp = cli_transport.complete("anthropic", [{"role": "user", "content": "hi"}], [])

    call = resp.choices[0].message.tool_calls[0]
    assert call.function.name == "delegate_coding_task"
    assert '"instructions": "write a parser"' in call.function.arguments
    # real usage is carried through, not invented
    assert (resp.usage.prompt_tokens, resp.usage.completion_tokens) == (120, 45)
    assert resp.notional_cost_usd == pytest.approx(0.031)


def test_parses_final_answer():
    envelope = '{"result": "{\\"final_answer\\": \\"all done\\"}", "usage": {}}'
    with patch.object(cli_transport, "available", return_value=True), patch.object(
        cli_transport.subprocess, "run", return_value=_proc(envelope)
    ):
        resp = cli_transport.complete("anthropic", [], [])

    assert resp.choices[0].message.tool_calls is None
    assert resp.choices[0].message.content == "all done"


def test_tolerates_a_code_fence_around_the_json():
    """Models wrap JSON in fences even when told not to."""
    fenced = '{"result": "```json\\n{\\"final_answer\\": \\"fenced reply\\"}\\n```", "usage": {}}'
    with patch.object(cli_transport, "available", return_value=True), patch.object(
        cli_transport.subprocess, "run", return_value=_proc(fenced)
    ):
        resp = cli_transport.complete("anthropic", [], [])
    assert resp.choices[0].message.content == "fenced reply"


def test_unparseable_reply_becomes_the_answer_rather_than_crashing():
    envelope = '{"result": "I could not produce JSON.", "usage": {}}'
    with patch.object(cli_transport, "available", return_value=True), patch.object(
        cli_transport.subprocess, "run", return_value=_proc(envelope)
    ):
        resp = cli_transport.complete("anthropic", [], [])
    assert "could not produce JSON" in resp.choices[0].message.content


# --- failure modes surface actionable errors, not cryptic ones ---


def test_missing_cli_raises_with_install_instructions():
    with patch.object(cli_transport.shutil, "which", return_value=None):
        with pytest.raises(CLINotAvailableError) as exc:
            cli_transport.complete("anthropic", [], [])
    assert "claude" in str(exc.value)
    assert "claude login" in str(exc.value)


def test_nonzero_exit_is_reported():
    with patch.object(cli_transport, "available", return_value=True), patch.object(
        cli_transport.subprocess, "run", return_value=_proc("boom", returncode=2)
    ):
        with pytest.raises(CLINotAvailableError) as exc:
            cli_transport.complete("anthropic", [], [])
    assert "exited 2" in str(exc.value)


def test_unknown_provider_has_no_cli_path():
    assert cli_transport.available("nope") is False
    with pytest.raises(CLINotAvailableError):
        cli_transport.complete("nope", [], [])


# --- billing must never be misreported as separately-charged money ---


def test_cli_usage_is_recorded_as_subscription_not_api_billing():
    stats = RunStats()
    resp = CLIResponse(choices=[], usage=cli_transport._Usage(10, 5), notional_cost_usd=0.25)
    _record_frontier_usage(resp, stats)

    assert stats.frontier_prompt_tokens == 10
    assert stats.frontier_completion_tokens == 5
    assert stats.frontier_cost_usd == pytest.approx(0.25)
    assert stats.frontier_via_subscription is True  # flags it as not separately billed


def test_api_responses_are_not_flagged_as_subscription():
    """A mock/proxy response must not be mistaken for the CLI transport --
    transport is detected by type, not by probing for an attribute.
    """
    stats = RunStats()
    api_resp = MagicMock()  # auto-creates any attribute asked of it
    api_resp.usage.prompt_tokens = 7
    api_resp.usage.completion_tokens = 3
    with patch("localforge.orchestrator.litellm.completion_cost", return_value=0.01):
        _record_frontier_usage(api_resp, stats)

    assert stats.frontier_via_subscription is False
    assert stats.frontier_cost_usd == pytest.approx(0.01)


def test_provider_spec_covers_every_api_key_provider():
    """Every provider that needs a key should have a CLI-login alternative."""
    key_providers = {p for p, v in config.FRONTIER_PROVIDERS.items() if v is not None}
    assert key_providers <= set(config.FRONTIER_CLI_AUTH)


# --- the advisor must also route through the CLI when that's the auth mode ---


def test_advisor_uses_cli_transport_and_validates_the_answer():
    """Under CLI login there's no API key, so the advisor must not go through
    litellm -- and it must still reject a model name outside the catalog.
    """
    from localforge.advisor import recommend_models
    from localforge.catalog import ModelEntry
    from localforge.hardware import HardwareProfile

    catalog = [
        ModelEntry(name="good-coder", modality="coding", runtime="ollama", min_vram_gb=0, min_ram_gb=8, disk_gb=1, quality_tier=2),
        ModelEntry(name="ok-coder", modality="coding", runtime="ollama", min_vram_gb=0, min_ram_gb=8, disk_gb=1, quality_tier=1),
    ]
    hw = HardwareProfile(os="Linux", arch="x86_64", cpu_cores=8, ram_gb=32, free_disk_gb=100, gpus=[])

    picked = CLIResponse(choices=[MagicMock(message=MagicMock(content='{"coding": "ok-coder"}'))])
    with patch.object(cli_transport, "complete", return_value=picked) as mock_complete, patch(
        "localforge.advisor.completion"
    ) as mock_litellm:
        recs = recommend_models(hw, "unused-model", catalog=catalog, cli_provider="anthropic")

    mock_complete.assert_called_once()
    mock_litellm.assert_not_called()  # never touches the API path
    assert recs["coding"].name == "ok-coder"


def test_advisor_falls_back_when_cli_names_a_model_not_in_the_catalog():
    from localforge.advisor import recommend_models
    from localforge.catalog import ModelEntry
    from localforge.hardware import HardwareProfile

    catalog = [
        ModelEntry(name="good-coder", modality="coding", runtime="ollama", min_vram_gb=0, min_ram_gb=8, disk_gb=1, quality_tier=2),
    ]
    hw = HardwareProfile(os="Linux", arch="x86_64", cpu_cores=8, ram_gb=32, free_disk_gb=100, gpus=[])

    hallucinated = CLIResponse(choices=[MagicMock(message=MagicMock(content='{"coding": "not-a-real-model"}'))])
    with patch.object(cli_transport, "complete", return_value=hallucinated):
        recs = recommend_models(hw, "unused", catalog=catalog, cli_provider="anthropic")

    assert recs["coding"].name == "good-coder"  # fell back to the catalog pick
