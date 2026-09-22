"""Interactive session for `localforge`, in the spirit of Claude Code's own
terminal UI: type a task directly and it runs, or use /commands for
everything else. Started automatically when `localforge` is run with no
subcommand in an interactive terminal (see cli.py's root callback).
"""

from __future__ import annotations

import shlex
from pathlib import Path

import typer
from rich.console import Console
from rich.markup import escape

from localforge import banner

SLASH_HELP = """[bold]Commands:[/bold]
  /run <task>       work on a task in this folder (or just type it directly)
  /clear            start a fresh conversation (/clear --forget also drops saved memory)
  /compact          have a local model condense the conversation into session memory
  /auto <on|off>    approve file changes and commands without asking
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


def run_repl(app: typer.Typer, console: Console) -> None:
    banner.render(console)
    console.print(
        f"\n[dim]Project folder: {escape(str(Path.cwd()))}[/dim]\n"
        "Type what you want built or changed, or [accent]/help[/accent] for commands. "
        "[accent]/exit[/accent] to leave.\n",
        highlight=False,
    )

    while True:
        try:
            line = console.input("[bold accent]localforge>[/bold accent] ")
        except (EOFError, KeyboardInterrupt):
            console.print()
            break

        line = line.strip()
        if not line:
            continue
        if line in EXIT_COMMANDS:
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

        argv = _to_argv(cmd, remainder)
        try:
            app(argv, standalone_mode=False)
        except typer.Exit:
            pass  # the command already reported its own success/failure
        except Exception as exc:  # noqa: BLE001 - keep the session alive on any command failure
            console.print(f"[error]Error:[/error] {exc}")
        console.print()
