"""The Matrix-style welcome banner shown when `localforge` starts an
interactive session. Generated once with pyfiglet's "doom" font and baked
in as a plain string here, rather than depending on pyfiglet at runtime for
one static piece of text. Picked over wider fonts (e.g. "ansi_shadow", 85
columns) specifically because 45 columns actually fits an ordinary 80-column
terminal -- a wider banner just silently falls back to plain text on most
people's default terminal size, which defeats the point of having one.
"""

from __future__ import annotations

from rich.console import Console

_LOGO = (
    " _                 _  __                     \n"
    "| |               | |/ _|                    \n"
    "| | ___   ___ __ _| | |_ ___  _ __ __ _  ___ \n"
    "| |/ _ \\ / __/ _` | |  _/ _ \\| '__/ _` |/ _ \\\n"
    "| | (_) | (_| (_| | | || (_) | | | (_| |  __/\n"
    "|_|\\___/ \\___\\__,_|_|_| \\___/|_|  \\__, |\\___|\n"
    "                                   __/ |     \n"
    "                                  |___/      "
)

_LOGO_WIDTH = max(len(line) for line in _LOGO.splitlines())


def render(console: Console) -> None:
    """Print the banner, in the big block-letter form if the terminal is
    wide enough, otherwise a plain bold fallback that never wraps ugly.
    """
    if console.size.width >= _LOGO_WIDTH:
        # highlight=False: otherwise Rich's automatic highlighter fragments
        # the ASCII art's slashes/parens into extra same-color spans -- same
        # visual result, just bloated ANSI output.
        console.print(f"[success]{_LOGO}[/success]", highlight=False)
    else:
        console.print("[bold success]▓▓▓ LOCALFORGE ▓▓▓[/bold success]", highlight=False)
