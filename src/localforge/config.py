"""Persists user setup choices (API keys, default frontier model) to
~/.config/localforge/config.env so `localforge setup` only has to run once.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

CONFIG_DIR = Path(os.environ.get("LOCALFORGE_CONFIG_DIR", Path.home() / ".config" / "localforge"))
CONFIG_FILE = CONFIG_DIR / "config.env"

FRONTIER_PROVIDERS = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "gemini": "GEMINI_API_KEY",
}
FRONTIER_API_KEY_ENV_VARS = list(FRONTIER_PROVIDERS.values()) + ["AWS_ACCESS_KEY_ID"]  # Bedrock

# Default LiteLLM model string to call for each provider, e.g. when the
# orchestrator/advisor need a concrete model id and only a provider (or an
# already-set API key) is known.
FRONTIER_DEFAULT_MODELS = {
    "anthropic": "claude-opus-5",
    "openai": "gpt-5",
    "gemini": "gemini-2.5-pro",
}


def frontier_model_for_env_var(env_var: str) -> str | None:
    """Map an already-set API key env var back to a default frontier model
    id, e.g. "ANTHROPIC_API_KEY" -> "claude-opus-5". Returns None for
    providers (like Bedrock) with no simple default mapping.
    """
    provider = next((p for p, v in FRONTIER_PROVIDERS.items() if v == env_var), None)
    return FRONTIER_DEFAULT_MODELS.get(provider) if provider else None


def load() -> None:
    """Load saved config into the environment, without overriding vars the
    user already set for this shell session.
    """
    if CONFIG_FILE.exists():
        load_dotenv(CONFIG_FILE, override=False)


def save(values: dict[str, str]) -> None:
    """Merge `values` into the saved config file, creating it if needed."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    existing: dict[str, str] = {}
    if CONFIG_FILE.exists():
        for line in CONFIG_FILE.read_text().splitlines():
            if "=" in line and not line.startswith("#"):
                key, _, val = line.partition("=")
                existing[key] = val

    existing.update(values)
    CONFIG_FILE.write_text("".join(f"{k}={v}\n" for k, v in existing.items()))
    CONFIG_FILE.chmod(0o600)  # contains API keys
