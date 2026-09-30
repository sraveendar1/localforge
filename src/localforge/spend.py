"""What paid image generation has cost this month, and an optional monthly limit.

Why: a subscription (say Google AI Pro) doesn't make API calls free, and image
generation through an API key is billed per image. The subscription does include
some monthly cloud credit, and a user who wants to stay inside it -- rather than
find an extra charge -- needs to see what's been spent and have localforge stop
at a limit they set.

Kept globally, in `~/.config/localforge/spend.json`, not per project: the credit
and the bill belong to the provider account, whichever folder spent it. The limit
is `LOCALFORGE_BUDGET_USD_<PROVIDER>` in the same config file as the keys.

The limit is enforced on what's *known*: an estimate from LiteLLM's price list
before a call (Gemini's image models have one; OpenAI's are priced per token, so
they have none) and the recorded cost after. A model with no known price can't be
pre-checked, only counted once LiteLLM reports what it cost.
"""
from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

from localforge import config

ENV_PREFIX = "LOCALFORGE_BUDGET_USD_"


def _ledger() -> Path:
    return config.CONFIG_DIR / "spend.json"  # read at call time: tests point CONFIG_DIR elsewhere


def month_key(now: datetime | None = None) -> str:
    return (now or datetime.now()).strftime("%Y-%m")


def _load() -> dict:
    try:
        data = json.loads(_ledger().read_text())
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def spent(provider: str, month: str | None = None) -> float:
    return float(((_load().get(month or month_key()) or {}).get(provider) or {}).get("usd") or 0.0)


def images(provider: str, month: str | None = None) -> int:
    return int(((_load().get(month or month_key()) or {}).get(provider) or {}).get("images") or 0)


def record(provider: str, usd: float, model: str = "") -> None:
    """Add one paid image to this month's total. Never raises: bookkeeping must
    not fail an image that was already generated and billed."""
    try:
        data = _load()
        slot = data.setdefault(month_key(), {}).setdefault(provider, {"usd": 0.0, "images": 0})
        slot["usd"] = round(float(slot.get("usd") or 0.0) + max(usd, 0.0), 6)
        slot["images"] = int(slot.get("images") or 0) + 1
        if model:
            slot["last_model"] = model
        _ledger().parent.mkdir(parents=True, exist_ok=True)
        tmp = _ledger().with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2))
        tmp.replace(_ledger())
    except Exception:  # noqa: BLE001, S110
        pass


def budget(provider: str) -> float | None:
    """The monthly limit in USD, or None for no limit."""
    raw = os.environ.get(ENV_PREFIX + provider.upper(), "").strip()
    try:
        value = float(raw)
    except ValueError:
        return None
    return value if value > 0 else None


def set_budget(provider: str, usd: float | None) -> None:
    """Set (or, with None, remove) the monthly limit. Saved with the other
    settings in the config file."""
    config.save({ENV_PREFIX + provider.upper(): "" if usd is None else f"{usd:g}"})


def check(provider: str, estimate: float | None = None) -> str | None:
    """Why another image must not be generated now (over the monthly limit), or
    None. Says what it knows: with no known price only the limit already being
    reached can stop a call."""
    cap = budget(provider)
    if cap is None:
        return None
    used = spent(provider)
    if used >= cap:
        return f"this month's {provider} image limit is reached (${used:.2f} of ${cap:.2f})"
    if estimate and used + estimate > cap:
        return (
            f"another image (about ${estimate:.3f}) would go over this month's {provider} limit "
            f"(${used:.2f} of ${cap:.2f} used)"
        )
    return None


def summary(provider: str) -> dict:
    return {"provider": provider, "spent": round(spent(provider), 4), "limit": budget(provider), "images": images(provider)}
