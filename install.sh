#!/usr/bin/env bash
# One-command install for localforge: installs uv if needed, installs the
# localforge CLI globally via `uv tool install`, then runs setup (installs
# Ollama + pulls local models automatically; only asks for a frontier model
# API key). Works both from a local checkout (./install.sh) and piped
# straight from GitHub (curl ... | bash), in which case it clones the repo
# into ~/.local/share/localforge/src first.
set -euo pipefail

REPO_URL="https://github.com/sanjayraveendar/localforge.git"
CLONE_DIR="$HOME/.local/share/localforge/src"

echo "localforge installer — this will set up the following on this machine:"
if [ "$(uname -s)" = "Darwin" ] && ! command -v brew >/dev/null 2>&1; then
    echo "  - Homebrew (macOS package manager — not currently installed)"
fi
if ! command -v ollama >/dev/null 2>&1; then
    echo "  - Ollama (runs open-weight models locally — not currently installed)"
fi
echo "  - One or more open-weight models matched to this machine's hardware"
echo "    (a real download, likely several GB — sizes shown before each pull)"
echo "  - The localforge CLI tool itself"
echo
# Read from the controlling terminal explicitly, not stdin -- when this
# script is run as `curl ... | bash`, stdin is the script itself, not the
# keyboard. Falls through automatically if there's no tty (non-interactive
# context) rather than hanging.
read -r -p "Press Enter to continue, or Ctrl+C to cancel: " _ < /dev/tty || true
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

if [ "$(uname -s)" = "Darwin" ] && ! command -v brew >/dev/null 2>&1; then
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

echo
echo "Running setup (you'll only be asked for a frontier model API key)..."
"$HOME/.local/bin/localforge" setup

echo
"$HOME/.local/bin/localforge"
