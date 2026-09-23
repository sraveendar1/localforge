"""Interactive session for `localforge`, in the spirit of Claude Code's own
terminal UI: type a task directly and it runs, or use /commands for
everything else. Started automatically when `localforge` is run with no
subcommand in an interactive terminal (see cli.py's root callback).
"""

from __future__ import annotations

import os
import re
import shlex
from contextlib import nullcontext
import sys
import time
from collections.abc import Callable
from pathlib import Path

import typer
from prompt_toolkit import PromptSession
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.formatted_text import HTML, FormattedText
from prompt_toolkit.history import FileHistory, InMemoryHistory
from prompt_toolkit.patch_stdout import patch_stdout
from prompt_toolkit.styles import Style
from rich.console import Console
from rich.markup import escape

from localforge import banner, config, theme

SLASH_HELP = """[bold]Commands:[/bold]
  /run <task>       work on a task in this folder (or just type it directly)
  /clear            start a fresh conversation (/clear --forget also drops saved memory)
  /compact          have a local model condense the conversation into session memory
  /init             write LOCALFORGE.md: what this project is, for future sessions
  /memory           show this folder's memory (/memory forget <name>, /memory clear)
  /scratch          list this session's scratchpad (/scratch clear to empty it)
  /auto <on|off>    approve file changes and commands without asking
  /stream <on|off>  print local models' output in full while they write
  /why              explain what you're being asked to approve (or just ask in words)
  /model <id>       show or switch the orchestrator (a local Ollama model or a cloud one)
  /summary          what's happening now: which model is doing what, the plan, the queue
  /queue            tasks waiting behind the current one (/queue clear to drop them)
  /stop             stop the running task
  /tell <note>      add a note to the running task
  /usage            token usage for the last task and this session
  /setup            one-time interactive setup
  /wizard           setup as a terminal UI
  /doctor           check everything's configured correctly
  /scan             show detected hardware
  /models           best-fit local model per modality
  /catalog          full model catalog
  /installed        models actually on disk
  /delete <names>   delete installed model(s), omit names to choose interactively
  /theme <name>     show/switch color theme (matrix, dark, light)
  /uninstall        remove localforge and everything it manages
  /help             show this list
  /exit, /quit, /q  leave this session"""

EXIT_COMMANDS = {"/exit", "/quit", "/q"}


def slash_commands() -> list[tuple[str, str]]:
    """(command, description) for every slash command, read from SLASH_HELP
    so the completion menu and /help can never disagree."""
    commands = []
    for line in SLASH_HELP.splitlines()[1:]:
        match = re.match(r"\s+(/\S+(?:, /\S+)*)(?:\s+<[^>]*>)?\s{2,}(.*)", line)
        if not match:
            continue
        for name in match.group(1).split(", "):
            commands.append((name, match.group(2).strip()))
    return commands


class SlashCompleter(Completer):
    """Typing "/" lists every command with its description, narrowing as
    you type; Tab completes -- like Claude Code's slash menu."""

    def get_completions(self, document, complete_event):
        text = document.text_before_cursor
        if not text.startswith("/") or " " in text:
            return
        for name, description in slash_commands():
            if name.startswith(text):
                yield Completion(name, start_position=-len(text), display=name, display_meta=description)


class LineReader:
    """Reads one line of input. In a real terminal it's a full line editor
    (prompt_toolkit): arrow keys move the cursor, text can be edited
    mid-line, Up/Down recall earlier input, and "/" opens the command menu.
    Plain `input()` has none of that -- on macOS the arrow keys just print
    escape codes like ^[[D -- so it's only the fallback for pipes and tests.
    """

    def __init__(self, console: Console, history_file: Path | None = None, runner=None):
        self.console = console
        self.runner = runner
        self.session = None
        if sys.stdin.isatty() and sys.stdout.isatty():
            history = None
            if history_file is not None:
                try:
                    history_file.parent.mkdir(parents=True, exist_ok=True)
                    history_file.touch(mode=0o600, exist_ok=True)
                    history = FileHistory(str(history_file))
                except OSError:
                    history = None
            extra = {}
            if runner is not None:
                # Redraw while a task runs. The status goes *in* the prompt
                # (see _message), not in bottom_toolbar: a pinned bottom bar
                # left the typing line stranded mid-screen with a gap below.
                extra = {"refresh_interval": 0.25, "style": status_style(os.environ.get(config.THEME_ENV_VAR, theme.DEFAULT_THEME))}
            self.session = PromptSession(
                history=history or InMemoryHistory(),
                completer=SlashCompleter(),
                complete_while_typing=True,
                reserve_space_for_menu=8,
                **extra,
            )

    def read(self) -> str:
        if self.session is None:
            return self.console.input("[bold accent]localforge>[/bold accent] ")
        return self.session.prompt(self._message)

    def _message(self):
        """What's drawn at the prompt, top to bottom: the live status and the
        code preview (while a task runs), then the line you type on. Rebuilt
        on every redraw, so it animates."""
        fragments: list[tuple[str, str]] = []
        if self.runner is not None:
            for kind, text in self.runner.status_lines():
                fragments.append((f"class:{kind}", text + "\n"))
        approval = self.runner.approval if self.runner is not None else None
        if approval is not None:
            fragments.append(("class:ask", f"Allow {approval.title}? (y)es / (n)o / (a)lways > "))
        else:
            fragments.append(("class:prompt", "localforge> "))
        return FormattedText(fragments)


def _html(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


# Commands that change the conversation, the settings or the models a running
# task is using -- refused while one runs rather than risking a corrupted turn.
# The prompt block follows the session's theme (localforge theme), so the
# status and the code preview are in the same colors as everything else --
# no grey lines pasted into a green session.
THEME_COLORS = {
    "matrix": {"main": "ansibrightgreen", "soft": "ansigreen", "alert": "ansibrightyellow"},
    "dark": {"main": "ansibrightcyan", "soft": "ansicyan", "alert": "ansibrightyellow"},
    "light": {"main": "ansiblue", "soft": "ansiblue", "alert": "ansibrightred"},
}


def status_style(theme_name: str) -> Style:
    colors = THEME_COLORS.get(theme_name, THEME_COLORS["matrix"])
    return Style.from_dict(
        {
            "status": colors["main"],
            "attention": f"{colors['alert']} bold",
            "preview": colors["soft"],
            "prompt": f"{colors['main']} bold",
            "ask": f"{colors['alert']} bold",
        }
    )

BUSY_BLOCKED = {"clear", "compact", "model", "setup", "wizard", "uninstall", "delete", "run", "init"}
HELP_COMMANDS = {"/help", "/?"}


def _split(line: str) -> tuple[str, str]:
    """A leading "/word" is a command; anything else is shorthand for
    `/run <the whole line>`, matching how you'd just type a task to Claude
    Code directly without a slash.
    """
    if line.startswith("/"):
        cmd, _, remainder = line[1:].partition(" ")
        return cmd, remainder.strip()
    return "run", line


def _to_argv(cmd: str, remainder: str) -> list[str]:
    if cmd == "run":
        return ["run", remainder] if remainder else ["run"]
    return [cmd, *(shlex.split(remainder) if remainder else [])]


def run_repl(
    app: typer.Typer,
    console: Console,
    on_start: Callable[[], bool] | None = None,
    on_exit: Callable[[], None] | None = None,
    history_file: Path | None = None,
    runner=None,
) -> None:
    """`on_start` runs after the banner (cli.py uses it to ask which
    orchestrator to use); returning False ends the session. `on_exit` runs
    when the session ends (cli.py saves session memory there)."""
    try:
        _loop(app, console, on_start, history_file, runner)
    finally:
        if on_exit is not None:
            try:
                on_exit()
            except Exception as exc:  # noqa: BLE001 - never lose the exit over a memory save
                console.print(f"[error]Couldn't save session memory: {exc}[/error]")


def _loop(
    app: typer.Typer, console: Console, on_start: Callable[[], bool] | None, history_file: Path | None = None, runner=None
) -> None:
    banner.render(console)
    console.print(
        f"\n[dim]Project folder: {escape(str(Path.cwd()))}[/dim]\n"
        "Type what you want built or changed, or [accent]/help[/accent] for commands. "
        "[accent]/exit[/accent] to leave.\n",
        highlight=False,
    )
    if on_start is not None and not on_start():
        return
    if Path.cwd().resolve() in (Path.home().resolve(), Path("/")):
        console.print(
            "[warning]You're in your home folder, so localforge can see all of it. "
            "For a project, /exit, cd into its folder, and run localforge there.[/warning]\n",
            highlight=False,
        )

    reader = LineReader(console, history_file, runner)
    if reader.session is None:
        runner = None  # no real terminal: run tasks in the foreground, as before
    with patch_stdout(raw=True) if runner is not None else nullcontext():
        _read_eval(app, console, reader, runner)


def _read_eval(app: typer.Typer, console: Console, reader: LineReader, runner) -> None:
    last_interrupt = 0.0
    while True:
        try:
            line = reader.read()
        except EOFError:  # Ctrl+D
            console.print()
            break
        except KeyboardInterrupt:
            if runner is not None and runner.busy:
                runner.cancel()  # Ctrl+C with a task running stops the task, not the session
                console.print("[warning]Stopping the task…[/warning]")
                continue
            # Like Claude Code: Ctrl+C clears the line; twice in a row leaves.
            now = time.monotonic()
            if now - last_interrupt < 2.0:
                console.print()
                break
            last_interrupt = now
            console.print("[dim](press Ctrl+C again to exit)[/dim]")
            continue
        last_interrupt = 0.0  # only two presses in a row exit

        line = line.strip()
        if runner is not None and runner.approval is not None:
            answer = line.lower()
            if answer in ("y", "yes", "n", "no", "a", "always"):
                runner.answer(answer in ("y", "yes", "a", "always"), always=answer in ("a", "always"))
                continue
            if line and not line.startswith("/") and runner.ask_question(line):
                continue  # a question about the request, answered without deciding it
        if not line:
            continue
        if line in EXIT_COMMANDS:
            if runner is not None and runner.busy:
                runner.shutdown()
                console.print("[warning]Stopped the running task.[/warning]")
            break
        if line in HELP_COMMANDS:
            # highlight=False: Rich's automatic highlighter styles "/word"
            # tokens as if they were filesystem paths, splitting the color
            # oddly mid-command in a way that looks broken, not helpful.
            console.print(SLASH_HELP, highlight=False)
            continue

        cmd, remainder = _split(line)
        if not cmd:
            console.print(SLASH_HELP, highlight=False)
            continue
        if cmd == "run" and not remainder:
            console.print("[warning]Usage: /run <task>, or just type your task directly.[/warning]")
            continue
        if runner is not None:
            if cmd == "run":
                position = runner.submit(remainder)
                if position:
                    console.print(f"[dim]Queued (#{position}) — it starts when the current task finishes. /queue to see.[/dim]")
                continue
            if runner.busy and cmd in BUSY_BLOCKED and not (cmd == "model" and runner.waiting):
                console.print(f"[warning]/{cmd} has to wait until the current task is done[/warning] (/summary to check on it, /stop to end it).")
                continue

        argv = _to_argv(cmd, remainder)
        try:
            app(argv, standalone_mode=False)
        except typer.Exit:
            pass  # the command already reported its own success/failure
        except KeyboardInterrupt:
            console.print("\n[warning]Stopped.[/warning]")  # Ctrl+C stops the command, not the session
        except Exception as exc:  # noqa: BLE001 - keep the session alive on any command failure
            console.print(f"[error]Error:[/error] {exc}")
        console.print()
