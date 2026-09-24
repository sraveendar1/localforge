"""Upgrading installed local models (asked for: "when localforge already has
open-weight models installed, it doesn't check if there's a better one ...
replace it, like an upgrade, and remove the older ones to free space").

The user chose: ask once, then automatic; and only models localforge itself
installed are ever removed on its own.
"""

import os
from unittest.mock import patch

import pytest

import localforge.cli as cli_module
from localforge import config, upgrades
from localforge.catalog import load_catalog
from localforge.hardware import GPU, HardwareProfile
from localforge.tools import Dispatcher

GB = 10**9


def _mac(ram, bandwidth, free_disk=200):
    return HardwareProfile(
        os="Darwin", arch="arm64", cpu_cores=12, ram_gb=ram, free_disk_gb=free_disk,
        gpus=[GPU(name="Apple M", vram_gb=ram * 0.75, backend="metal")], memory_bandwidth_gbps=bandwidth,
    )


M4_PRO_48 = _mac(48, 273)
M4_16 = _mac(16, 120)


def _plan(hw, installed, managed=None, keep=frozenset()):
    sizes = {name: int(next((e.disk_gb for e in load_catalog() if e.name == name), 1) * GB) for name in installed}
    return upgrades.plan(hw, sizes, keep=keep, managed_names=set(installed) if managed is None else managed)


# --- the plan ---------------------------------------------------------------------


def test_a_better_model_that_fits_replaces_the_installed_one():
    plan = _plan(M4_PRO_48, {"qwen2.5-coder:7b"})
    assert [(u.old, u.new.name, u.modalities) for u in plan.upgrades] == [("qwen2.5-coder:7b", "qwen3-coder:30b", ["coding"])]
    assert plan.remove == ["qwen2.5-coder:7b"]  # superseded, and localforge installed it
    assert plan.download_gb == pytest.approx(18.6)
    assert plan.freed_gb == pytest.approx(4.7)


def test_a_model_that_no_longer_fits_is_replaced_by_one_that_does():
    # a 14b kept from before the memory-aware rules: on a 16 GB Mac it swaps
    plan = _plan(M4_16, {"qwen2.5-coder:14b"})
    assert [(u.old, u.new.name) for u in plan.upgrades] == [("qwen2.5-coder:14b", "qwen2.5-coder:7b")]
    assert plan.remove == ["qwen2.5-coder:14b"]


def test_the_users_own_models_are_only_mentioned_never_removed():
    plan = _plan(M4_PRO_48, {"qwen2.5-coder:7b"}, managed=set())
    assert plan.upgrades  # still upgraded...
    assert plan.remove == [] and plan.unused_own == ["qwen2.5-coder:7b"]  # ...but theirs isn't touched


def test_models_outside_the_catalog_are_never_touched():
    plan = _plan(M4_PRO_48, {"qwen2.5-coder:7b", "my-finetune:latest"})
    assert "my-finetune:latest" not in plan.remove + plan.unused_own


def test_same_tier_is_not_an_upgrade():
    # llama3.1:8b (tier 1) for docs on a 16 GB Mac: the balanced pick is tier 1 too
    plan = _plan(M4_16, {"qwen2.5-coder:7b", "llama3.1:8b"})
    assert plan.upgrades == [] and plan.remove == []
    assert plan.empty


def test_already_the_best_fit_means_nothing_to_do():
    assert _plan(M4_16, {"qwen2.5-coder:7b"}).empty


def test_short_on_disk_it_picks_a_smaller_upgrade_that_fits():
    plan = _plan(_mac(48, 273, free_disk=15), {"qwen2.5-coder:7b"})
    assert [u.new.name for u in plan.upgrades] == ["qwen2.5-coder:14b"]  # 18.6 GB won't fit; 9 GB will


def test_not_enough_disk_blocks_the_download_and_keeps_what_is_in_use():
    # 20 GB free: the 18.6 GB model technically fits, but would leave the disk nearly full
    plan = _plan(_mac(48, 273, free_disk=20), {"qwen2.5-coder:7b"})
    assert plan.upgrades == [] and plan.blocked and "GB free" in plan.blocked
    assert "qwen2.5-coder:7b" not in plan.remove  # still the model in use


def test_the_local_orchestrator_is_never_removed():
    plan = _plan(M4_PRO_48, {"qwen2.5-coder:7b"}, keep={"qwen2.5-coder:7b"})
    assert plan.remove == []


# --- tracking what localforge installed ------------------------------------------


def test_managed_models_round_trip():
    upgrades.mark_managed("a:1")
    upgrades.mark_managed("b:2")
    upgrades.unmark("a:1")
    assert upgrades.managed() == {"b:2"}


def test_a_model_downloaded_during_a_task_is_marked_managed(tmp_path):
    from localforge.workspace import Workspace

    class Stub:
        def ensure_available(self, name, on_progress=None):
            pass

        def generate(self, name, prompt, on_token=None, **kw):
            return {"type": "text", "content": "some useful output", "tokens": 3}

    entry = next(e for e in load_catalog() if e.name == "qwen2.5-coder:7b")
    d = Dispatcher(M4_16, catalog=[entry], installed=set(), workspace=Workspace(tmp_path, lambda *a: True))
    with patch.dict("localforge.tools.BACKENDS", {"ollama": Stub()}):
        d.dispatch("delegate_coding_task", {"instructions": "write it"})
    assert "qwen2.5-coder:7b" in upgrades.managed()


# --- applying it -----------------------------------------------------------------


class FakeOllama:
    def __init__(self, fail=()):
        self.fail, self.pulled, self.deleted = set(fail), [], []

    def ensure_available(self, name, on_progress=None):
        if name in self.fail:
            raise RuntimeError("network down")
        self.pulled.append(name)

    def delete(self, name):
        self.deleted.append(name)

    def list_installed(self):
        return [{"name": "qwen2.5-coder:7b", "size": int(4.7 * GB)}]


def test_download_first_then_remove_the_old_one():
    ollama = FakeOllama()
    upgrades.mark_managed("qwen2.5-coder:7b")
    plan = _plan(M4_PRO_48, {"qwen2.5-coder:7b"})
    assert cli_module._apply_upgrade(plan, ollama)
    assert ollama.pulled == ["qwen3-coder:30b"] and ollama.deleted == ["qwen2.5-coder:7b"]
    assert upgrades.managed() == {"qwen3-coder:30b"}


def test_a_failed_download_removes_nothing():
    ollama = FakeOllama(fail={"qwen3-coder:30b"})
    assert not cli_module._apply_upgrade(_plan(M4_PRO_48, {"qwen2.5-coder:7b"}), ollama)
    assert ollama.deleted == []  # the old model is still what's in use


def test_old_models_are_removed_only_once_no_task_is_running(monkeypatch):
    class Runner:
        checks = 0

        @property
        def busy(self):
            Runner.checks += 1
            return Runner.checks < 3  # a task is running for the first two checks

    monkeypatch.setattr(cli_module._session, "runner", Runner())
    monkeypatch.setattr(cli_module, "UPGRADE_IDLE_POLL_SECONDS", 0)
    ollama = FakeOllama()
    cli_module._apply_upgrade(_plan(M4_PRO_48, {"qwen2.5-coder:7b"}), ollama, wait_for_idle=True)
    assert Runner.checks == 3 and ollama.deleted == ["qwen2.5-coder:7b"]


def test_a_model_made_the_orchestrator_mid_download_is_kept(monkeypatch):
    plan = _plan(M4_PRO_48, {"qwen2.5-coder:7b"})
    monkeypatch.setenv(config.FRONTIER_MODEL_ENV_VAR, "ollama/qwen2.5-coder:7b")  # /model during the download
    ollama = FakeOllama()
    cli_module._apply_upgrade(plan, ollama)
    assert ollama.deleted == []


def test_background_download_reports_progress_in_steps(monkeypatch):
    lines = []
    monkeypatch.setattr(cli_module.console, "print", lambda text, **kw: lines.append(str(text)))

    class Slow(FakeOllama):
        def ensure_available(self, name, on_progress=None):
            for done in range(0, 101, 5):
                on_progress({"total": 100, "completed": done})
            self.pulled.append(name)

    ollama = Slow()
    thread = cli_module._start_background_upgrade(_plan(M4_PRO_48, {"qwen2.5-coder:7b"}), ollama)
    thread.join(5)
    progress = [line for line in lines if "% of" in line]
    assert [p.split(": ")[1].split("%")[0] for p in progress] == ["25", "50", "75"]


# --- asking once, then automatic ---------------------------------------------------


@pytest.fixture
def machine(monkeypatch):
    ollama = FakeOllama()
    upgrades.mark_managed("qwen2.5-coder:7b")
    monkeypatch.setattr(cli_module, "OllamaBackend", lambda: ollama)
    monkeypatch.setattr(cli_module, "detect_hardware", lambda: M4_PRO_48)
    monkeypatch.setattr(cli_module.sys.stdin, "isatty", lambda: True)
    started = []
    monkeypatch.setattr(cli_module, "_start_background_upgrade", lambda plan, o: started.append(plan))
    return started


def test_first_time_it_asks_and_always_is_remembered(machine, monkeypatch):
    monkeypatch.delenv(upgrades.AUTO_UPGRADE_ENV_VAR, raising=False)
    monkeypatch.setattr(cli_module, "_ask_number", lambda prompt, count: 2)
    cli_module._offer_upgrades_at_start()
    assert len(machine) == 1
    assert os.environ[upgrades.AUTO_UPGRADE_ENV_VAR] == "always"
    assert "LOCALFORGE_AUTO_UPGRADE=always" in config.CONFIG_FILE.read_text()


def test_after_always_it_upgrades_without_asking(machine, monkeypatch):
    monkeypatch.setenv(upgrades.AUTO_UPGRADE_ENV_VAR, "always")
    monkeypatch.setattr(cli_module, "_ask_number", lambda *a: pytest.fail("asked again"))
    cli_module._offer_upgrades_at_start()
    assert len(machine) == 1


def test_never_means_no_prompt_and_no_upgrade(machine, monkeypatch):
    monkeypatch.setenv(upgrades.AUTO_UPGRADE_ENV_VAR, "never")
    monkeypatch.setattr(cli_module, "_ask_number", lambda *a: pytest.fail("asked"))
    cli_module._offer_upgrades_at_start()
    assert machine == []


def test_not_now_changes_nothing(machine, monkeypatch):
    monkeypatch.delenv(upgrades.AUTO_UPGRADE_ENV_VAR, raising=False)
    monkeypatch.setattr(cli_module, "_ask_number", lambda prompt, count: 3)
    cli_module._offer_upgrades_at_start()
    assert machine == [] and upgrades.AUTO_UPGRADE_ENV_VAR not in os.environ


def test_upgrade_command_with_yes_never_removes_the_users_own_models(monkeypatch):
    ollama = FakeOllama()
    monkeypatch.setattr(cli_module, "OllamaBackend", lambda: ollama)
    monkeypatch.setattr(cli_module, "detect_hardware", lambda: M4_PRO_48)
    cli_module.upgrade(yes=True)  # 7b wasn't installed by localforge here
    assert ollama.pulled == ["qwen3-coder:30b"] and ollama.deleted == []


# --- nothing is lost in the handover ----------------------------------------------


def test_memory_and_the_conversation_carry_over_to_the_upgraded_model(tmp_path, monkeypatch):
    """Memory, facts and the conversation are plain text, kept per project,
    never per model: after an upgrade the next task runs on the new model
    with everything the old one did."""
    import json as _json
    from unittest.mock import MagicMock

    import localforge.orchestrator as orch
    from localforge import memory
    from localforge.workspace import Workspace

    root = tmp_path
    memory.save(root, "## Goal\nBuild the status bar (done by the old model).")
    memory.remember(root, "prefers-tabs", "The user wants tabs, not spaces.", type="feedback")

    old = next(e for e in load_catalog() if e.name == "qwen2.5-coder:7b").model_copy(update={"runtime": "stub"})
    new = next(e for e in load_catalog() if e.name == "qwen3-coder:30b").model_copy(update={"runtime": "stub"})
    used, prompts = [], []

    class Local:
        def ensure_available(self, name, on_progress=None):
            pass

        def generate(self, name, prompt, on_token=None, **kw):
            used.append(name)
            return {"type": "text", "content": "done and working", "tokens": 3}

    step = {"n": 0}

    def completion(**kw):
        prompts.append(kw["messages"][0]["content"] + "\n" + _json.dumps(kw["messages"][1:], default=str))
        step["n"] += 1
        msg = MagicMock()
        if step["n"] % 2:  # delegate once, then answer
            call = MagicMock(id=f"c{step['n']}")
            call.function.name, call.function.arguments = "delegate_coding_task", '{"instructions": "next step"}'
            msg.tool_calls, msg.content = [call], None
            msg.model_dump.return_value = {"role": "assistant", "content": None}
        else:
            msg.tool_calls, msg.content = None, "ok"
            msg.model_dump.return_value = {"role": "assistant", "content": "ok"}
        resp = MagicMock()
        resp.choices, resp.usage = [MagicMock(message=msg)], None
        return resp

    conversation = orch.Conversation(memory=memory.load(root), facts=memory.facts_for_prompt(root))
    installed = {"qwen2.5-coder:7b"}
    monkeypatch.setattr(orch, "_installed_models", lambda: set(installed))
    monkeypatch.setattr("localforge.tools.load_catalog", lambda: [old, new])
    monkeypatch.setattr(orch.litellm, "completion_cost", lambda **kw: 0.0)
    with patch.object(orch, "completion", side_effect=completion), patch.dict("localforge.tools.BACKENDS", {"stub": Local()}):
        orch.run("first task", "claude-opus-5", hardware=M4_PRO_48, conversation=conversation, workspace=Workspace(root))
        installed.add("qwen3-coder:30b")  # the background upgrade finished
        installed.discard("qwen2.5-coder:7b")  # ...and removed the old model
        orch.run("second task", "claude-opus-5", hardware=M4_PRO_48, conversation=conversation, workspace=Workspace(root))

    assert used == ["qwen2.5-coder:7b", "qwen3-coder:30b"]  # the new model took over
    last = prompts[-1]
    assert "Build the status bar (done by the old model)" in last  # session memory
    assert "The user wants tabs, not spaces." in last  # remembered facts
    assert "first task" in last  # the earlier conversation
