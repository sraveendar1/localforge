"""PyInstaller entry point: `localforge.cli:app` is a Typer app object, not
a plain `main()` function, so PyInstaller (which needs a runnable script)
can't target it directly -- this just imports and calls it, exactly like
the `localforge` console-script shim `pyproject.toml` generates for a
regular pip/uv install does.
"""
import os

# A bundled/sidecar binary should start instantly like any other app.
# Without this, litellm's own import-time model-cost-map fetch retries a
# remote URL 3 times (a real delay when offline, and every single startup
# if the network genuinely can't reach it) before falling back to the
# local map it already ships -- so skip straight to that local map. Set
# before importing anything litellm-adjacent; it reads this once at import.
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

from localforge.cli import app

if __name__ == "__main__":
    app()
