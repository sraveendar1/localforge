#!/usr/bin/env bash
# One-command install for localforge: installs uv if needed, installs the
# localforge CLI globally via `uv tool install`, then runs the interactive
# setup wizard (Ollama + local models + frontier API key).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if ! command -v uv >/dev/null 2>&1; then
    echo "uv not found — installing it..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

echo "Installing localforge..."
uv tool install --reinstall "$SCRIPT_DIR"

echo
echo "Running interactive setup..."
"$HOME/.local/bin/localforge" setup

echo
echo "Done. Try: localforge run \"Build a todo REST API with docs\""
