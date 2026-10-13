"""The Overall goal reads like a goal (not the pasted prompt) and follows the project as it
moves, and "Open items" in the last-session note move to "Completed items" once they're done.
(Reported: the goal was "just like how I wrote the prompt"; finished open items stayed open.)"""
import io
import json

import pytest

from localforge import brief, memory, trust
from localforge.orchestrator import Conversation
from localforge.serve import StdioServer

NOTE = """## Goal
Build the inventory dashboard.

## Open items
* **osquery installation**: User needs to run `brew install osquery` and enter their sudo password.
* **MCP configuration verification**: Confirm the structure of the Claude Desktop configuration file.
  Check the servers key too.
* **Browser Cache**: User must hard-refresh the browser.

## Current state
* Backend runs.
"""


class FakeDispatcher:
    local_tokens_generated = 0

    def __init__(self, reply):
        self.reply, self.installed = reply, {"gemma3:12b"}

    def resolve(self, modality):
        from localforge.catalog import ModelEntry

        return ModelEntry(name="gemma3:12b", modality=modality, runtime="ollama", min_vram_gb=0, min_ram_gb=0, disk_gb=1, quality_tier=1)

    def context_limit(self, entry):
        return 8192


def _keeper_says(monkeypatch, text):
    class B:
        def generate(self, name, prompt, context_limit=None):
            B.prompt = prompt
            return {"content": text, "tokens": 5}

    monkeypatch.setitem(memory.BACKENDS, "ollama", B())
    return B


def test_the_goal_is_cleaned_of_greetings_and_lead_ins():
    raw = "Hi The Overall goal of the project is to use OSQuery to see which AI agents run on the machines and what MCP connections they have. I should be able to see all of it"
    out = brief.clean_goal(raw)
    assert out.startswith("Use OSQuery") and not out.lower().startswith(("hi", "the overall goal"))
    assert out.endswith(".")
    assert brief.clean_goal("I want to build a CLI todo app in python") == "A CLI todo app in python."


def test_the_prompt_is_kept_as_the_source_and_the_shown_goal_is_the_clean_one(tmp_path):
    brief.record_goal(tmp_path, "Hi, the goal of this project is to build a todo app in React with a login page")
    assert brief.request_text(tmp_path).startswith("Hi, the goal")
    assert brief.project_goal(tmp_path) == "Build a todo app in React with a login page."


def test_the_keeper_restates_the_goal_once(tmp_path, monkeypatch):
    brief.record_goal(tmp_path, "hi i want a thing that shows my machines AI agents")
    _keeper_says(monkeypatch, "A dashboard that shows which AI agents are running on each machine.")
    assert brief.refine_goal(tmp_path, FakeDispatcher(None))
    assert brief.project_goal(tmp_path) == "A dashboard that shows which AI agents are running on each machine."
    assert not brief.refine_goal(tmp_path, FakeDispatcher(None))  # only once


def test_the_goal_follows_a_change_of_direction_and_remembers_the_old_one(tmp_path, monkeypatch):
    brief.record_goal(tmp_path, "Build a todo app in React with a login page")
    _keeper_says(monkeypatch, json.dumps({"changed": True, "goal": "A habit tracker mobile app in Flutter."}))
    assert brief.evolve_goal(tmp_path, FakeDispatcher(None), ["Build a todo app in React", "Actually forget React, make it a Flutter habit tracker"])
    assert brief.project_goal(tmp_path) == "A habit tracker mobile app in Flutter."
    assert "todo app" in (memory.project_dir(tmp_path) / brief.HISTORY_FILE).read_text()


def test_a_small_request_does_not_change_the_goal(tmp_path, monkeypatch):
    brief.record_goal(tmp_path, "Build a todo app in React with a login page")
    _keeper_says(monkeypatch, json.dumps({"changed": False, "goal": "whatever"}))
    assert not brief.evolve_goal(tmp_path, FakeDispatcher(None), ["Fix the button colour on the login page"])
    assert brief.project_goal(tmp_path) == "Build a todo app in React with a login page."


def test_without_a_local_model_only_an_explicit_pivot_changes_the_goal(tmp_path, monkeypatch):
    brief.record_goal(tmp_path, "Build a todo app in React with a login page")
    monkeypatch.setattr(memory, "keeper", lambda d: None)
    assert not brief.evolve_goal(tmp_path, None, ["add dark mode to the todo list please"])
    assert brief.evolve_goal(tmp_path, None, ["New goal: build a habit tracker in Flutter with streaks"])
    assert "habit tracker" in brief.project_goal(tmp_path)


def test_an_evolved_goal_beats_an_older_agents_md_section(tmp_path, monkeypatch):
    (tmp_path / "AGENTS.md").write_text("# Brief\n\n## What this project is\n\nA todo app.\n")
    brief.record_goal(tmp_path, "Build a todo app in React with a login page")
    assert brief.project_goal(tmp_path) == "A todo app."
    brief._set_goal(tmp_path, "A habit tracker in Flutter.")
    assert brief.project_goal(tmp_path) == "A habit tracker in Flutter."
    assert "A habit tracker in Flutter." in brief.with_goal_replaced((tmp_path / "AGENTS.md").read_text(), "A habit tracker in Flutter.")


# --- open and completed items -------------------------------------------------------


def test_open_items_are_read_with_their_indented_continuation():
    items = memory.note_items(NOTE, "open")
    assert len(items) == 3 and "Check the servers key too" in items[1]


def test_finished_items_move_to_completed_and_nothing_else_changes():
    new = memory.move_items(NOTE, [0, 2], "done")
    assert [i.split(":")[0] for i in memory.note_items(new, "open")] == ["**MCP configuration verification**"]
    assert len(memory.note_items(new, "done")) == 2
    assert "## Completed items" in new and "## Current state\n* Backend runs." in new
    assert "Build the inventory dashboard." in new
    back = memory.move_items(new, [0], "open")
    assert len(memory.note_items(back, "open")) == 2


def test_the_keeper_decides_which_items_a_finished_task_completed(tmp_path, monkeypatch):
    memory.save(tmp_path, NOTE)
    _keeper_says(monkeypatch, json.dumps({"completed": [1, 3]}))
    moved = memory.reconcile_open_items(tmp_path, "install osquery and refresh", "Installed osquery; hard refresh done", [], FakeDispatcher(None))
    assert moved == 2
    assert len(memory.note_items(memory.load(tmp_path), "open")) == 1


def test_without_a_local_model_a_completed_plan_step_still_closes_its_item(tmp_path, monkeypatch):
    memory.save(tmp_path, NOTE)
    monkeypatch.setattr(memory, "keeper", lambda d: None)
    moved = memory.reconcile_open_items(tmp_path, "t", "a", ["Complete the osquery installation on the machine"], None)
    assert moved == 1 and "osquery" in memory.note_items(memory.load(tmp_path), "done")[0].lower()
    assert memory.reconcile_open_items(tmp_path, "t", "a", ["Write the README"], None) == 0


def test_ticking_an_item_in_the_app_moves_it(tmp_path):
    out = io.StringIO()
    (tmp_path / "scratch").mkdir()
    trust.trust(tmp_path)
    server = StdioServer(tmp_path, "claude-opus-5", None, out=out, run_fn=lambda *a, **k: None, scratch_root=tmp_path / "scratch", conversation=Conversation())
    memory.save(tmp_path, NOTE)
    server.handle({"type": "memory_item", "action": "done", "index": 0})
    events = [json.loads(l) for l in out.getvalue().splitlines() if l.strip()]
    narrative = [e for e in events if e["type"] == "memory"][-1]["narrative"]
    assert len(memory.note_items(narrative, "open")) == 2 and len(memory.note_items(narrative, "done")) == 1
    server.handle({"type": "memory_item", "action": "reopen", "index": 0})
    assert len(memory.note_items(memory.load(tmp_path), "open")) == 3


# --- a question is not a goal (reported: the goal was "Can you summarize what is left") -------------


@pytest.mark.parametrize("text", [
    "Can you summarize what is left",
    "what is the next step . And also please check if the ReadMe is actually accurate",
    "explain this repo to me?", "fix the bug in app.js please", "continue", "thanks that works great",
])
def test_questions_and_chores_are_not_goals(text):
    assert not brief.worth_recording_as_goal(text)


@pytest.mark.parametrize("text", [
    "Can you build a trading signal bot that posts to Slack",
    "I want this packaged up nicely so a user can download it from github",
    "The overall goal of the project is to use osquery to see which AI agents are running",
])
def test_real_requests_are_still_goals(text):
    assert brief.worth_recording_as_goal(text)


def test_a_question_recorded_earlier_is_replaced_by_the_next_real_request(tmp_path):
    from localforge import memory

    folder = memory.ensure_dir(tmp_path)
    (folder / brief.REQUEST_FILE).write_text("Can you summarize what is left\n")
    (folder / brief.GOAL_FILE).write_text("Can you summarize what is left.\n")
    (folder / brief.REFINED_FILE).write_text("1\n")
    assert brief.shown_goal(tmp_path) == ""  # not shown meanwhile
    assert brief.record_goal(tmp_path, "Build a Slack alert bot that posts trading signals every morning")
    assert "Slack alert bot" in brief.shown_goal(tmp_path) and not brief.goal_is_refined(tmp_path)


def test_a_real_goal_is_not_replaced_by_a_later_request(tmp_path):
    assert brief.record_goal(tmp_path, "Build a Slack alert bot that posts trading signals every morning")
    assert not brief.record_goal(tmp_path, "Build something completely different instead of that thing")


def test_agents_md_that_only_copies_the_first_message_does_not_beat_the_rewritten_goal(tmp_path):
    raw = "Hi The Overall goal of the project is to use OSquery to diferent what area the AI agents are running on the machines"
    assert brief.record_goal(tmp_path, raw)
    (tmp_path / "AGENTS.md").write_text(f"# Project brief\n\n## What this project is\n\n{raw}\n")
    from localforge import memory

    (memory.project_dir(tmp_path) / brief.GOAL_FILE).write_text("A tool that uses osquery to find which AI agents are running on a machine.\n")
    (memory.project_dir(tmp_path) / brief.REFINED_FILE).write_text("1\n")
    assert brief.project_goal(tmp_path).startswith("A tool that uses osquery")


def test_agents_md_written_to_describe_the_project_is_still_preferred(tmp_path):
    assert brief.record_goal(tmp_path, "Build a Slack alert bot that posts trading signals every morning")
    (tmp_path / "AGENTS.md").write_text("# Project brief\n\n## What this project is\n\nA scanner that finds trading signals and posts them to Slack.\n")
    assert brief.project_goal(tmp_path).startswith("A scanner that finds")


def test_with_no_goal_the_readme_opening_is_shown(tmp_path):
    (tmp_path / "README.md").write_text("# Tradingbot\n\n![badge](x)\n\nScans the market each morning for momentum signals and sends them to Slack.\n")
    assert brief.project_goal(tmp_path).startswith("Scans the market")


def test_the_user_can_write_the_goal_themselves(tmp_path):
    assert brief.record_goal(tmp_path, "Build a Slack alert bot that posts trading signals every morning")
    assert brief.set_goal_by_user(tmp_path, "A morning trading-signal digest for my Slack.")
    assert brief.project_goal(tmp_path) == "A morning trading-signal digest for my Slack."
    assert not brief.record_goal(tmp_path, "Build a totally different dashboard for something else entirely")
