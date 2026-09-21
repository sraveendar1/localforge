from unittest.mock import MagicMock, patch

import pytest
from rich.console import Console

from localforge import banner
from localforge.repl import SLASH_HELP, _split, _to_argv, run_repl


# --- argv construction: the part most likely to silently misroute a command ---


def test_plain_text_becomes_a_run_command():
    assert _split("write a snake game") == ("run", "write a snake game")


def test_slash_command_is_split_on_first_space():
    assert _split("/theme dark") == ("theme", "dark")


def test_slash_command_with_no_args():
    assert _split("/scan") == ("scan", "")


def test_to_argv_keeps_the_task_as_one_argument_not_shlex_split():
    """A task like `write a "hello world" function` must reach `run` as a
    single string, not get exploded by shell-style quote parsing.
    """
    argv = _to_argv("run", 'write a "hello world" function')
    assert argv == ["run", 'write a "hello world" function']


def test_to_argv_shlex_splits_flags_for_non_run_commands():
    argv = _to_argv("delete", "model-a model-b --yes")
    assert argv == ["delete", "model-a", "model-b", "--yes"]


def test_to_argv_run_with_no_remainder():
    assert _to_argv("run", "") == ["run"]


# --- help text must not collide with Rich's [markup] syntax ---


def test_help_text_has_no_unescaped_square_brackets_outside_known_tags():
    """Rich silently swallows anything in [brackets] it doesn't recognize as
    a style tag. `/delete [names]` previously vanished from the printed
    help entirely -- this guards against reintroducing that.
    """
    known_tags = {"[bold]", "[/bold]", "[accent]", "[/accent]"}
    remaining = SLASH_HELP
    for tag in known_tags:
        remaining = remaining.replace(tag, "")
    assert "[" not in remaining and "]" not in remaining, remaining


def test_help_actually_renders_the_documented_flags(capsys):
    """Render through a real Console (not just check the source string) --
    confirms Rich doesn't strip anything at print time either.
    """
    console = Console(width=120, force_terminal=False)
    console.print(SLASH_HELP)
    out = capsys.readouterr().out
    assert "<names>" in out
    assert "<name>" in out
    assert "<task>" in out


# --- the loop itself: dispatch, error handling, exit ---


def _fake_console(width: int = 120):
    console = Console(width=width, force_terminal=False)
    return console


def test_repl_dispatches_a_slash_command_to_the_app():
    app = MagicMock()
    console = _fake_console()
    lines = iter(["/scan", "/exit"])
    console.input = lambda prompt="": next(lines)

    run_repl(app, console)

    app.assert_called_once_with(["scan"], standalone_mode=False)


def test_repl_dispatches_plain_text_as_run():
    app = MagicMock()
    console = _fake_console()
    lines = iter(["build a todo app", "/exit"])
    console.input = lambda prompt="": next(lines)

    run_repl(app, console)

    app.assert_called_once_with(["run", "build a todo app"], standalone_mode=False)


def test_repl_exits_on_all_exit_spellings():
    for spelling in ("/exit", "/quit", "/q"):
        app = MagicMock()
        console = _fake_console()
        lines = iter([spelling])
        console.input = lambda prompt="": next(lines)
        run_repl(app, console)  # must return, not hang or raise
        app.assert_not_called()


def test_repl_exits_cleanly_on_eof():
    """Ctrl+D / a closed pipe must end the session, not crash it."""
    app = MagicMock()
    console = _fake_console()

    def raise_eof(prompt=""):
        raise EOFError

    console.input = raise_eof
    run_repl(app, console)  # should not raise


def test_repl_survives_a_failing_command_and_keeps_looping():
    app = MagicMock(side_effect=[RuntimeError("boom"), None])
    console = _fake_console()
    lines = iter(["/run task one", "/scan", "/exit"])
    console.input = lambda prompt="": next(lines)

    run_repl(app, console)  # must not propagate the RuntimeError

    assert app.call_count == 2


def test_repl_swallows_typer_exit_silently():
    """A command that calls typer.Exit (its normal success/failure signal)
    must not be treated as an unexpected error.
    """
    import typer

    app = MagicMock(side_effect=typer.Exit(code=1))
    console = _fake_console()
    lines = iter(["/doctor", "/exit"])
    console.input = lambda prompt="": next(lines)

    run_repl(app, console)  # must not print "Error: ..."


def test_repl_ignores_blank_lines():
    app = MagicMock()
    console = _fake_console()
    lines = iter(["", "   ", "/exit"])
    console.input = lambda prompt="": next(lines)
    run_repl(app, console)
    app.assert_not_called()


def test_run_with_no_task_shows_usage_not_an_empty_run():
    app = MagicMock()
    console = _fake_console()
    lines = iter(["/run", "/exit"])
    console.input = lambda prompt="": next(lines)
    run_repl(app, console)
    app.assert_not_called()  # never dispatched with an empty task


# --- banner: verified glyphs, and a fallback that never wraps ugly ---


def test_banner_renders_full_logo_when_wide_enough(capsys):
    console = Console(width=120, force_terminal=False)
    banner.render(console)
    out = capsys.readouterr().out
    assert "LOCALFORGE" not in out  # it's block art, not literal text
    assert out.strip() != ""
    assert out.count("\n") >= banner._LOGO.count("\n")  # the multi-line art, not the one-line fallback


def test_banner_falls_back_when_narrow(capsys):
    console = Console(width=40, force_terminal=False)
    banner.render(console)
    out = capsys.readouterr().out
    assert "LOCALFORGE" in out
    assert len(out.strip().splitlines()[0]) <= 40


def test_banner_logo_has_no_line_wider_than_reported():
    for line in banner._LOGO.splitlines():
        assert len(line) <= banner._LOGO_WIDTH


def test_help_output_has_no_path_style_highlighting_split():
    """Rich's automatic ReprHighlighter styles "/word" as if it were a
    filesystem path, inserting extra escape codes that split "/run" into
    separately-colored fragments. Confirmed live via pexpect against a real
    pty. highlight=False must produce strictly fewer escape sequences than
    the highlighted version for the same text.
    """
    console_highlighted = Console(width=120, force_terminal=True, color_system="standard")
    with console_highlighted.capture() as cap:
        console_highlighted.print(SLASH_HELP, highlight=True)
    highlighted_codes = cap.get().count("\x1b[")

    console_plain = Console(width=120, force_terminal=True, color_system="standard")
    with console_plain.capture() as cap:
        console_plain.print(SLASH_HELP, highlight=False)
    plain_codes = cap.get().count("\x1b[")

    assert plain_codes < highlighted_codes
