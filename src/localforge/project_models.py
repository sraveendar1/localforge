"""Which models a project uses, kept in the project.

Asked for: "for every project, a local file that stores what the models are
set to, so if I change the project folder it references it, and if I change it
the file gets updated." That is `<project>/.localforge/models.json`:

    {
      "orchestrator": {"model": "claude-opus-5", "auth_method": "api_key", "provider": "anthropic"},
      "delegates": {"coding": "auto", "docs": "ollama:llama3.1:8b", "general": "auto",
                    "image": "api:openai:gpt-image-1", "video": "auto"}
    }

Precedence, lowest to highest: your global defaults (`localforge setup`,
`config.env`) -> this project's file -> an explicit `--model`. A project with
no file simply uses the defaults, and the first change made in it writes the
file (the whole set, so it says what the project uses). Changing a model in a
project rewrites that project's file only; it doesn't move your defaults, and
other projects don't notice.

It lives in `.localforge/`, which git ignores by itself (see memory.py), so it
stays yours; delete that folder's `.gitignore` to share it with a team.

The file is only read for a folder the user has trusted: a cloned repo could
otherwise choose a paid model to be billed to your keys.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from localforge import config, memory

FILE_NAME = "models.json"

_active_root: Path | None = None  # the project whose file applies (and gets written) in this process


def path(root: Path) -> Path:
    return memory.project_dir(root) / FILE_NAME


def load(root: Path) -> dict | None:
    """The project's saved models, or None if it has none (or the file is
    unreadable -- a bad file must never stop a session from starting)."""
    try:
        data = json.loads(path(root).read_text())
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _env_updates(data: dict) -> dict[str, str]:
    """The environment values a saved file stands for. Anything malformed is
    skipped: a hand-edited or stale value degrades to the defaults."""
    from localforge import delegate_target

    updates: dict[str, str] = {}
    orch = data.get("orchestrator")
    if isinstance(orch, dict):
        model = orch.get("model")
        if isinstance(model, str) and model.strip():
            updates[config.FRONTIER_MODEL_ENV_VAR] = config.litellm_model_id(model.strip())
            auth = orch.get("auth_method")
            if auth in (config.AUTH_API_KEY, config.AUTH_CLI_LOGIN, config.AUTH_LOCAL):
                updates[config.AUTH_METHOD_ENV_VAR] = auth
            provider = orch.get("provider")
            if isinstance(provider, str) and provider:
                updates[config.FRONTIER_PROVIDER_ENV_VAR] = provider
    delegates = data.get("delegates")
    if isinstance(delegates, dict):
        for modality, value in delegates.items():
            if modality in delegate_target.TARGET_ENV_VARS and isinstance(value, str):
                updates[delegate_target.TARGET_ENV_VARS[modality]] = delegate_target.render(delegate_target.parse(value))
    return updates


def activate(root: Path) -> bool:
    """Make `root` this process's project: apply its saved models over the
    defaults (True if it had any), and send later changes to its file. Call
    only for a trusted folder."""
    global _active_root
    _active_root = Path(root).resolve()
    data = load(_active_root)
    if data is None:
        return False
    os.environ.update(_env_updates(data))
    return True


def deactivate() -> None:
    global _active_root
    _active_root = None


def active_root() -> Path | None:
    return _active_root


def source() -> str:
    """"project" when the active project has its own file, else "defaults"."""
    return "project" if _active_root is not None and path(_active_root).is_file() else "defaults"


def snapshot() -> dict:
    """The models in effect right now, in the file's shape."""
    from localforge import delegate_target

    orch: dict[str, str] = {}
    for key, env in (("model", config.FRONTIER_MODEL_ENV_VAR), ("auth_method", config.AUTH_METHOD_ENV_VAR),
                     ("provider", config.FRONTIER_PROVIDER_ENV_VAR)):
        if os.environ.get(env):
            orch[key] = os.environ[env]
    return {
        "orchestrator": orch,
        "delegates": {m: delegate_target.render(delegate_target.get(m)) for m in delegate_target.ALL_MODALITIES},
    }


def save(updates: dict[str, str], scope: str = "project") -> None:
    """Record a change to the models. With a project active it goes to that
    project's file (the whole current set, written after applying the change
    here); otherwise, or with scope="global", to the saved defaults."""
    if scope != "project" or _active_root is None:
        config.save(updates)
        return
    os.environ.update(updates)
    try:
        target = memory.ensure_dir(_active_root) / FILE_NAME
        target.write_text(json.dumps(snapshot(), indent=2) + "\n")
    except OSError:
        config.save(updates)  # can't write into the project (read-only?): keep the choice as a default instead
