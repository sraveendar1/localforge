"""Color themes for the CLI's Rich output. Every message in cli.py uses
semantic style names (success/error/warning/accent/bold/dim) rather than
literal color words, so switching a theme actually changes what's shown --
see `localforge theme`.
"""

from __future__ import annotations

from rich.theme import Theme

THEMES: dict[str, Theme] = {
    "matrix": Theme(
        {
            "success": "bold bright_green",
            "error": "bold bright_red",
            "warning": "bold bright_yellow",
            "accent": "bold bright_green",
            "panel.border": "bright_green",
        }
    ),
    "dark": Theme(
        {
            "success": "bold green",
            "error": "bold red",
            "warning": "bold yellow",
            "accent": "bold cyan",
            "panel.border": "cyan",
        }
    ),
    "light": Theme(
        {
            "success": "bold dark_green",
            "error": "bold dark_red",
            "warning": "bold dark_orange3",
            "accent": "bold blue",
            "panel.border": "blue",
        }
    ),
}

DEFAULT_THEME = "matrix"


def get_theme(name: str) -> Theme:
    return THEMES.get(name, THEMES[DEFAULT_THEME])
