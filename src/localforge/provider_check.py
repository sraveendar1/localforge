"""Checks against a cloud provider, shared by the desktop's first-run setup,
`doctor`, and the model pickers.

- `check_key()` says whether an API key is accepted, so a wrong or expired one
  is caught when it's entered rather than as a raw error on the first task.
- `gemini_image_models()` lists the image models a Google AI Studio key can
  actually use, instead of a list that goes stale.
- `image_price()` is the per-image price where LiteLLM knows it.

Every network call is short and never raises: "couldn't check" is an answer
(the key is then kept, with a note), never a reason to lose what was typed.
"""
from __future__ import annotations

import re

import httpx

from localforge import config

TIMEOUT_SECONDS = 8.0
OK, REJECTED, UNREACHABLE = "ok", "rejected", "unreachable"

_ENDPOINTS = {
    # provider: (url, how the key is sent)
    "anthropic": ("https://api.anthropic.com/v1/models", lambda key: {"x-api-key": key, "anthropic-version": "2023-06-01"}),
    "openai": ("https://api.openai.com/v1/models", lambda key: {"Authorization": f"Bearer {key}"}),
    # a header, never the URL, so the key can't land in a log
    "gemini": ("https://generativelanguage.googleapis.com/v1beta/models", lambda key: {"x-goog-api-key": key}),
}


def clean_key(raw: str) -> str:
    """A pasted key with the whitespace and quotes a copy usually drags along removed."""
    return (raw or "").strip().strip("'\"").strip()


def looks_like_a_key(key: str) -> bool:
    """Not a real check -- just refuses obvious mistakes (empty, spaces inside,
    absurdly short) before anything is sent anywhere."""
    return len(key) >= 20 and not re.search(r"\s", key)


def check_key(provider: str, key: str) -> tuple[str, str]:
    """(status, detail): "ok" if the provider accepts the key, "rejected" if it
    says no (401/403), "unreachable" if it couldn't be asked (offline, blocked,
    a timeout, or any other answer -- not evidence the key is wrong)."""
    if provider not in _ENDPOINTS:
        return UNREACHABLE, f"no check is available for {provider}"
    url, headers = _ENDPOINTS[provider]
    try:
        resp = httpx.get(url, headers=headers(key), timeout=TIMEOUT_SECONDS)
    except Exception as exc:  # noqa: BLE001 - see module docstring
        return UNREACHABLE, f"couldn't reach {provider} to check it ({type(exc).__name__})"
    if resp.status_code == 200:
        return OK, f"{provider} accepted the key"
    if resp.status_code in (400, 401, 403):
        # Google answers a bad key with 400; the others with 401/403.
        return REJECTED, f"{provider} rejected the key (HTTP {resp.status_code})"
    return UNREACHABLE, f"{provider} answered HTTP {resp.status_code}, so the key couldn't be checked"


def image_price(model: str) -> float | None:
    """USD per image where LiteLLM's price list has one (Gemini's image models
    do; OpenAI's are priced per token, so None)."""
    import litellm

    info = litellm.model_cost.get(model) or {}
    price = info.get("output_cost_per_image")
    return float(price) if price else None


def _litellm_image_models() -> set[str]:
    import litellm

    return {name for name, info in litellm.model_cost.items() if info.get("mode") == "image_generation"}


def gemini_image_models(api_key: str) -> list[str]:
    """`gemini/<id>` for every image model this key can use that LiteLLM can
    also call (the intersection is what makes an entry safe to offer), newest
    first. Empty on any failure: the curated list is used instead."""
    try:
        resp = httpx.get(_ENDPOINTS["gemini"][0], params={"pageSize": 1000}, headers=_ENDPOINTS["gemini"][1](api_key), timeout=TIMEOUT_SECONDS)
        resp.raise_for_status()
        listed = resp.json().get("models") or []
    except Exception:  # noqa: BLE001
        return []
    known = _litellm_image_models()
    found = []
    for m in listed:
        name = str(m.get("name") or "").removeprefix("models/")
        ident = f"gemini/{name}"
        if ("image" in name or "imagen" in name) and ident in known:
            found.append(ident)

    def rank(ident: str):
        version = re.search(r"(\d+(?:\.\d+)?)", ident.split("/", 1)[1])
        return (-float(version.group(1)) if version else 0.0, "preview" in ident, ident)

    return sorted(set(found), key=rank)


def image_models_for(provider: str, api_key: str | None = None) -> list[str]:
    """The image models to offer for `provider`: the live list for Gemini when
    a key is set and reachable, else the curated one."""
    curated = list(config.IMAGE_MODEL_CHOICES.get(provider, []))
    if provider == "gemini" and api_key:
        live = gemini_image_models(api_key)
        if live:
            return live
    return curated
