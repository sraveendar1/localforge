"""Persists user setup choices (API keys, default frontier model) to
~/.config/localforge/config.env so `localforge setup` only has to run once.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

CONFIG_DIR = Path(os.environ.get("LOCALFORGE_CONFIG_DIR", Path.home() / ".config" / "localforge"))
CONFIG_FILE = CONFIG_DIR / "config.env"

FRONTIER_PROVIDERS: dict[str, str | None] = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "gemini": "GEMINI_API_KEY",
    # An open-weight model can itself be the orchestrator -- served locally
    # via Ollama, routed through LiteLLM's "ollama/<model>" convention, and
    # needing no API key at all. This is the only provider with `None` here.
    "local": None,
}
FRONTIER_API_KEY_ENV_VARS = [v for v in FRONTIER_PROVIDERS.values() if v] + ["AWS_ACCESS_KEY_ID"]  # Bedrock

# Persists which frontier model localforge should use by default, so `run`
# doesn't have to re-ask or silently guess from whatever API keys happen to
# be in the environment.
FRONTIER_MODEL_ENV_VAR = "LOCALFORGE_FRONTIER_MODEL"

# Persists the chosen CLI color theme (see theme.py) so it only has to be
# picked once, via `localforge theme`.
THEME_ENV_VAR = "LOCALFORGE_THEME"

# Curated model choices per provider, offered during setup so a user can
# pick e.g. Sonnet or Fable instead of always getting the first entry. The
# first entry in each list is the default. The Anthropic list is a full,
# verified lineup; OpenAI/Gemini only get one entry each here because we
# don't have an equally reliable/current lineup to offer without risking a
# stale or invented model id -- "Other" (a free-text model id) is always
# offered alongside these in the UI as the escape hatch. The "local" list
# suggests large-enough Ollama models known to support tool calling
# reasonably well -- the orchestrator role delegates via tool calls, which
# most small open-weight models handle unreliably; "Other" lets a user try
# a different one anyway.
FRONTIER_MODEL_CHOICES: dict[str, list[str]] = {
    "anthropic": ["claude-opus-5", "claude-sonnet-5", "claude-haiku-4-5-20251001", "claude-fable-5-1"],
    "openai": ["gpt-5"],
    "gemini": ["gemini-2.5-pro"],
    "local": ["ollama/llama3.1:70b", "ollama/qwen2.5:72b"],
}
FRONTIER_DEFAULT_MODELS = {provider: choices[0] for provider, choices in FRONTIER_MODEL_CHOICES.items()}

# How localforge authenticates to the frontier provider. "api_key" is
# pay-per-token billing against a key you paste in; "cli_login" shells out to
# the provider's own already-logged-in CLI, drawing on whatever subscription
# that account has instead of separate API charges.
AUTH_METHOD_ENV_VAR = "LOCALFORGE_AUTH_METHOD"
# Which provider the saved model/auth choice belongs to -- needed because CLI
# login routes by provider, not by a LiteLLM model string.
FRONTIER_PROVIDER_ENV_VAR = "LOCALFORGE_FRONTIER_PROVIDER"
AUTH_API_KEY = "api_key"
AUTH_CLI_LOGIN = "cli_login"

# Per-provider first-party CLI that supports account/subscription login.
#   command       : executable to look for on PATH
#   headless_args : args that make it take a prompt on argv and print a reply
#   json_args     : args that additionally make it emit machine-readable output
#   envelope      : "json"  -> stdout is one JSON object
#                   "jsonl" -> stdout is newline-delimited JSON events
#   result_key    : (json) key holding the reply text
#   usage_key     : (json) key holding token counts
#   isolation_args: args that switch off the CLI's own agent tools (shell,
#                   file edits, web, MCP connectors). The CLI is the
#                   *orchestrator* here: it plans and delegates through
#                   localforge's tools only. Left on, `claude -p` could run
#                   Bash/Edit/Write in the user's folder and reach their
#                   Slack/M365/Docs connectors -- verified live. Web access
#                   comes from localforge's own web_search/fetch_url instead,
#                   so it works the same under an API key.
#   Flags below were checked against each project's own published docs.
#   Anthropic is additionally verified live end to end; the other two are
#   probed at runtime (cli_transport.available/logged_in) rather than
#   assumed, so a wrong guess surfaces as "not installed / not logged in"
#   with install instructions instead of a cryptic failure.
FRONTIER_CLI_AUTH: dict[str, dict] = {
    "anthropic": {
        "command": "claude",
        "headless_args": ["-p"],
        # Verified live (claude 2.1.x): with these the model reports no
        # built-in or MCP tools. `--tools` is variadic, so it must be followed
        # by another option, never directly by the prompt.
        "isolation_args": ["--tools", "", "--strict-mcp-config"],
        "json_args": ["--output-format", "json"],
        "envelope": "json",
        "result_key": "result",
        "usage_key": "usage",
        "install_hint": "https://claude.com/claude-code",
        "login_hint": "run `claude login`",
        "verified": True,
    },
    "openai": {
        # `codex exec --json` emits JSON Lines; the reply arrives as an
        # item.completed event whose item.type is "agent_message".
        "command": "codex",
        "headless_args": ["exec"],
        # Read-only sandbox: no file writes or shell side effects. Codex's web
        # search is off unless --search is passed. --skip-git-repo-check
        # because it runs in an empty scratch directory (see cli_transport).
        # From Codex's docs, not verified live.
        "isolation_args": ["--sandbox", "read-only", "--skip-git-repo-check"],
        "json_args": ["--json"],
        "envelope": "jsonl",
        "result_key": "text",
        "usage_key": "usage",
        "install_hint": "https://developers.openai.com/codex/cli",
        "login_hint": "run `codex login` and sign in with your ChatGPT account",
        "verified": False,
    },
    "gemini": {
        # Single JSON object: {"response": ..., "stats": {...}}
        "command": "gemini",
        "headless_args": ["-p"],
        # No verified flag to switch its tools off; it still runs in an empty
        # scratch directory, so its file tools see nothing of the user's.
        "isolation_args": [],
        "json_args": ["--output-format", "json"],
        "envelope": "json",
        "result_key": "response",
        "usage_key": "stats",
        "install_hint": "https://github.com/google-gemini/gemini-cli",
        "login_hint": "run `gemini` once and sign in with your Google account",
        "verified": False,
    },
}


# Where to create an API key for each provider. There's no public OAuth/
# browser-login flow any of these providers expose for third-party CLI
# tools to authenticate on a user's behalf (unlike e.g. GitHub's device
# flow) -- this is a plain convenience: opened automatically when a key is
# needed and none is found yet, so the user lands on the right page instead
# of navigating there manually, then still pastes the key themselves.
FRONTIER_CONSOLE_URLS: dict[str, str] = {
    "anthropic": "https://console.anthropic.com/settings/keys",
    "openai": "https://platform.openai.com/api-keys",
    "gemini": "https://aistudio.google.com/app/apikey",
}


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
