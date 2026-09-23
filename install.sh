#!/usr/bin/env bash
# One-command install for localforge: installs uv if needed and installs the
# localforge CLI globally via `uv tool install`. That's all -- choosing a
# model, installing Ollama and downloading anything happens the first time
# the user runs `localforge`, where they can see what's going on and skip
# what they don't want. (`./install.sh --setup` does it here instead.)
# Works both from a local checkout and piped straight from GitHub
# (curl ... | bash), in which case it clones the repo into
# ~/.local/share/localforge/src first.
set -euo pipefail

REPO_URL="https://github.com/sraveendar1/localforge.git"
CLONE_DIR="$HOME/.local/share/localforge/src"

RUN_SETUP=0
for arg in "$@"; do
    case "$arg" in
        --setup) RUN_SETUP=1 ;;
    esac
done

echo "localforge installer — this will install:"
if ! command -v uv >/dev/null 2>&1; then
    echo "  - uv (the Python tool installer — not currently installed)"
fi
echo "  - The localforge CLI tool itself"
if [ "$RUN_SETUP" = "1" ]; then
    if [ "$(uname -s)" = "Darwin" ] && ! command -v brew >/dev/null 2>&1; then
        echo "  - Homebrew (macOS package manager — not currently installed)"
    fi
    if ! command -v ollama >/dev/null 2>&1; then
        echo "  - Ollama (runs open-weight models locally — not currently installed)"
    fi
    echo "  - One or more open-weight models matched to this machine's hardware"
    echo "    (a real download, likely several GB — sizes shown before each pull)"
else
    echo
    echo "Nothing else yet: the first time you run 'localforge' it offers to install"
    echo "Ollama and set up a model, so you can see what it's doing and choose."
fi
echo
# Can we actually *open* the controlling terminal? `[ -e /dev/tty ]` is not
# enough: the device node exists in CI/containers/nohup/cron but opening it
# fails ("Device not configured"), and with `set -e` a failed redirect would
# kill the script. Probe by really opening it.
if { : < /dev/tty; } 2>/dev/null; then
    HAVE_TTY=1
else
    HAVE_TTY=0
fi

# Read from the controlling terminal explicitly, not stdin -- when this
# script is run as `curl ... | bash`, stdin is the script itself, not the
# keyboard. Skipped entirely when there's no usable tty, rather than hanging
# or aborting.
if [ "$HAVE_TTY" = "1" ]; then
    read -r -p "Press Enter to continue, or Ctrl+C to cancel: " _ < /dev/tty || true
else
    echo "(non-interactive: continuing without confirmation)"
fi
echo

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-.}")" 2>/dev/null && pwd || true)"

if [ -n "$SCRIPT_DIR" ] && [ -f "$SCRIPT_DIR/pyproject.toml" ]; then
    PROJECT_DIR="$SCRIPT_DIR"
else
    echo "Fetching localforge source from $REPO_URL..."
    if ! command -v git >/dev/null 2>&1; then
        echo "git is required to install localforge this way. Install git and re-run." >&2
        exit 1
    fi
    if [ -d "$CLONE_DIR/.git" ]; then
        git -C "$CLONE_DIR" pull --ff-only
    else
        git clone "$REPO_URL" "$CLONE_DIR"
    fi
    PROJECT_DIR="$CLONE_DIR"
fi

if [ "$RUN_SETUP" = "1" ] && [ "$(uname -s)" = "Darwin" ] && ! command -v brew >/dev/null 2>&1; then
    echo "Homebrew not found — installing it (needed to auto-install Ollama)..."
    NONINTERACTIVE=1 /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    if [ -x /opt/homebrew/bin/brew ]; then
        eval "$(/opt/homebrew/bin/brew shellenv)"
    elif [ -x /usr/local/bin/brew ]; then
        eval "$(/usr/local/bin/brew shellenv)"
    fi
fi

if ! command -v uv >/dev/null 2>&1; then
    echo "uv not found — installing it..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

echo "Installing localforge..."
uv tool install --reinstall "$PROJECT_DIR"

if [ "$RUN_SETUP" = "1" ]; then
    echo
    echo "Running setup (--setup was passed)..."
    # Read prompts from the terminal, not this script's stdin: under
    # `curl ... | bash`, stdin is the script itself, so setup's interactive
    # prompts would hit EOF and abort. Falls back to inherited stdin when
    # there's no tty (non-interactive context).
    if [ "$HAVE_TTY" = "1" ]; then
        "$HOME/.local/bin/localforge" setup < /dev/tty
    else
        "$HOME/.local/bin/localforge" setup
    fi
fi

echo
# Print the "you're set up" panel and exit. stdin comes from /dev/null on
# purpose: a bare `localforge` opens the interactive session whenever stdin
# is a terminal, which would leave the installer sitting in a prompt instead
# of finishing. The panel tells the user to type `localforge` themselves.
"$HOME/.local/bin/localforge" < /dev/null
