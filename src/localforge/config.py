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

# Persists which frontier model localforge should use by default, so `run`
# doesn't have to re-ask or silently guess from whatever API keys happen to
# be in the environment.
FRONTIER_MODEL_ENV_VAR = "LOCALFORGE_FRONTIER_MODEL"

# Curated model choices per provider, offered during setup so a user can
# pick e.g. Sonnet or Fable instead of always getting the first entry. The
# first entry in each list is the default. The Anthropic list is a full,
# verified lineup; OpenAI/Gemini only get one entry each here because we
# don't have an equally reliable/current lineup to offer without risking a
# stale or invented model id -- "Other" (a free-text model id) is always
# offered alongside these in the UI as the escape hatch.
FRONTIER_MODEL_CHOICES: dict[str, list[str]] = {
    "anthropic": ["claude-opus-5", "claude-sonnet-5", "claude-haiku-4-5-20251001", "claude-fable-5-1"],
    "openai": ["gpt-5"],
    "gemini": ["gemini-2.5-pro"],
}
FRONTIER_DEFAULT_MODELS = {provider: choices[0] for provider, choices in FRONTIER_MODEL_CHOICES.items()}


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
