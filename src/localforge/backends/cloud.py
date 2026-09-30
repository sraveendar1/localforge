"""Delegate backends that aren't local: an "advanced" per-modality override
lets coding/docs/general delegate to a paid cloud model instead of the
usual free local Ollama pick -- either metered (an API key, the same
LiteLLM call the orchestrator's own API-key path already makes) or via a
CLI subscription login (reusing `cli_transport`'s existing subprocess/
JSON-protocol machinery, the same one the orchestrator uses under CLI
login). See `delegate_target.py` for the override itself and
`tools.Dispatcher.resolve()` for how it becomes a `ModelEntry` these
backends can run.

Both backends are one-shot: no tool calling, no multi-turn planning -- just
"here are the instructions, write the content" -- which is exactly the
delegate contract every local backend already fulfills. `BackendResult`'s
`cost_usd`/`notional_cost_usd` (see backends/base.py) are how a cloud
delegate's real (or subscription-notional) spend reaches usage accounting,
mirroring `RunStats.frontier_cost_usd`/`frontier_via_subscription`.
"""

from __future__ import annotations

import base64
import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import httpx
import litellm

from localforge import cli_transport, config
from localforge.backends.base import BackendResult


class CloudProviderNotConfigured(RuntimeError):
    """The provider a cloud delegate target names has no usable API
    key/CLI login -- raised instead of letting the underlying call fail
    with a confusing, unrelated error."""


class CloudApiBackend:
    """Runs a delegate task via a plain LiteLLM completion, billed like any
    other API call -- the same mechanism the orchestrator's own API-key
    transport uses, just for a single one-shot generation instead of a
    planning loop."""

    def ensure_available(self, model_name: str, on_progress: object = None, provider: str | None = None) -> None:
        env_var = config.FRONTIER_PROVIDERS.get(provider or "")
        if env_var and not os.environ.get(env_var):
            raise CloudProviderNotConfigured(
                f"No {env_var} is set, so {provider} can't be used as a delegate. "
                "Set it (e.g. via `localforge setup`) or pick a different target with `localforge advanced-model`."
            )

    def generate(
        self, model_name: str, prompt: str, on_token: Callable[[str], None] | None = None,
        provider: str | None = None, **kwargs,
    ) -> BackendResult:
        model_id = config.litellm_model_id(model_name)
        response = litellm.completion(model=model_id, messages=[{"role": "user", "content": prompt}])
        message = response.choices[0].message
        content = message.content or ""
        if on_token is not None and content:
            on_token(content)  # LiteLLM's non-streaming call returns the whole reply at once
        usage = getattr(response, "usage", None)
        tokens = getattr(usage, "completion_tokens", 0) if usage else 0
        try:
            cost = litellm.completion_cost(completion_response=response)
        except Exception:  # noqa: BLE001 - an unpriced/unknown model shouldn't fail the delegation
            cost = 0.0
        return {"type": "text", "content": content, "tokens": tokens, "cost_usd": cost, "notional_cost_usd": 0.0}


class CloudCliBackend:
    """Runs a delegate task through a provider's logged-in CLI (the same
    subscription the orchestrator itself may already be using), reusing
    cli_transport.complete() with an empty tool list -- there's nothing to
    call, just instructions to follow and content to write, exactly the
    JSON-decision "final_answer" path local/CLI orchestrators already use."""

    def ensure_available(self, model_name: str, on_progress: object = None, provider: str | None = None) -> None:
        if provider and not cli_transport.available(provider):
            raise CloudProviderNotConfigured(cli_transport.requirements_message(provider))

    def generate(
        self, model_name: str, prompt: str, on_token: Callable[[str], None] | None = None,
        provider: str | None = None, **kwargs,
    ) -> BackendResult:
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": "Proceed."},
        ]
        response = cli_transport.complete(provider, messages, tools=[], model=model_name, on_text=on_token)
        message = response.choices[0].message
        content = message.content or ""
        return {
            "type": "text",
            "content": content,
            "tokens": response.usage.completion_tokens,
            "cost_usd": 0.0,
            "notional_cost_usd": response.notional_cost_usd,
        }


MAX_IMAGE_BYTES = 30_000_000  # a generated image is a few MB; this is a guard, not a limit anyone should meet
IMAGE_TIMEOUT_SECONDS = 180.0  # image models can take a minute or more


@dataclass
class GeneratedImage:
    data: bytes
    mime_type: str
    cost_usd: float = 0.0


def sniff_image_type(data: bytes) -> str | None:
    """The image format from its first bytes, or None if it isn't one we
    write to a project (never trust a provider's claim about what it sent)."""
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return None


class CloudImageBackend:
    """Generates one image through a provider's image API (LiteLLM's
    `image_generation`), billed per image to the user's API key. Separate from
    the text backends above: the result is bytes for a file, not text, and
    there is no CLI-login route (none of the provider CLIs generate images)."""

    def ensure_available(self, model_name: str, provider: str | None = None) -> None:
        env_var = config.FRONTIER_PROVIDERS.get(provider or "")
        if not env_var or not os.environ.get(env_var):
            raise CloudProviderNotConfigured(
                f"No API key is set for {provider or 'the image provider'}, so it can't generate images. "
                "Set one (e.g. via `localforge setup`) or turn image generation off with "
                "`localforge advanced-model image auto`."
            )

    def generate(self, model_name: str, prompt: str, provider: str | None = None) -> GeneratedImage:
        self.ensure_available(model_name, provider)
        response = litellm.image_generation(model=model_name, prompt=prompt, n=1, timeout=IMAGE_TIMEOUT_SECONDS)
        items = getattr(response, "data", None) or []
        if not items:
            raise RuntimeError(f"{model_name} returned no image")
        item = items[0]
        b64 = item.get("b64_json") if isinstance(item, dict) else getattr(item, "b64_json", None)
        url = item.get("url") if isinstance(item, dict) else getattr(item, "url", None)
        if b64:
            data = base64.b64decode(b64)
        elif url:
            data = self._download(str(url))
        else:
            raise RuntimeError(f"{model_name} returned neither image data nor a link to one")
        mime = sniff_image_type(data)
        if mime is None:
            raise RuntimeError(f"{model_name} returned something that isn't a PNG, JPEG or WebP image")
        try:
            cost = float(litellm.completion_cost(completion_response=response) or 0.0)
        except Exception:  # noqa: BLE001 - an unpriced model shouldn't fail the image
            cost = 0.0
        return GeneratedImage(data=data, mime_type=mime, cost_usd=cost)

    @staticmethod
    def _download(url: str) -> bytes:
        if not url.startswith("https://"):
            raise RuntimeError("the image link isn't https, so it wasn't fetched")
        with httpx.stream("GET", url, timeout=60.0, follow_redirects=True) as resp:
            resp.raise_for_status()
            chunks, size = [], 0
            for chunk in resp.iter_bytes():
                size += len(chunk)
                if size > MAX_IMAGE_BYTES:
                    raise RuntimeError("the generated image is larger than the limit and wasn't saved")
                chunks.append(chunk)
        return b"".join(chunks)


IMAGE_CLI_TIMEOUT_SECONDS = 600.0
_IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp")
_MAX_SCANNED = 5000  # files looked at under the CLI's own folder; it can hold a lot of history


class CloudImageCliBackend:
    """Generates an image through a provider's own logged-in CLI -- OpenAI's
    Codex, whose built-in `image_gen` tool runs on a ChatGPT login with no API
    key -- so it draws on the subscription instead of per-image billing.

    EXPERIMENTAL, and not verified live (no Codex login was available when this
    was written): it runs the official `codex exec` in an empty scratch folder,
    asks it to generate the image and leave it in that folder, and takes
    whatever image file appears -- there, or (Codex saves under $CODEX_HOME by
    default) newly written under it. It does *not* call the ChatGPT backend
    directly with the login token, as some third-party tools do: that endpoint is
    undocumented and outside what the login is offered for. If the CLI produces
    no image the error says what it printed. Claude has no image generation and
    Gemini's CLI only has one through an extension, so neither is offered.
    """

    def ensure_available(self, model_name: str, provider: str | None = None) -> None:
        if provider not in config.IMAGE_CLI_CHOICES:
            raise CloudProviderNotConfigured(f"{provider}'s CLI login can't generate images.")
        if not cli_transport.available(provider):
            raise CloudProviderNotConfigured(cli_transport.requirements_message(provider))

    def generate(self, model_name: str, prompt: str, provider: str | None = None) -> GeneratedImage:
        self.ensure_available(model_name, provider)
        spec = config.FRONTIER_CLI_AUTH[provider]
        work = Path(tempfile.mkdtemp(prefix="localforge-image-"))
        started = time.time()
        instruction = (
            "Use your built-in image generation tool to create exactly one image for this request:\n\n"
            f"{prompt}\n\n"
            "Save the final image as ./image.png in the current folder (copy or move it there if it was saved "
            "somewhere else). Do nothing else: no other files, no commands beyond that. Reply 'done' once the file exists."
        )
        cmd = [spec["command"], *spec["headless_args"], "--sandbox", "workspace-write", "--skip-git-repo-check", instruction]
        try:
            try:
                proc = subprocess.run(
                    cmd, cwd=work, capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=IMAGE_CLI_TIMEOUT_SECONDS,
                )
            except subprocess.TimeoutExpired:
                raise RuntimeError(f"{spec['command']} took longer than {int(IMAGE_CLI_TIMEOUT_SECONDS)}s to make the image") from None
            found = self._find_image(work, started)
            if found is None:
                tail = (proc.stdout + proc.stderr).strip()[-400:] or "(no output)"
                raise RuntimeError(
                    f"{spec['command']} produced no image file (exit {proc.returncode}). It said: {tail}. "
                    f"Check that you're signed in ({spec['login_hint']}) and that your plan includes image generation."
                )
            data = found.read_bytes()
        finally:
            shutil.rmtree(work, ignore_errors=True)
        mime = sniff_image_type(data)
        if mime is None:
            raise RuntimeError(f"{spec['command']} left a file that isn't a PNG, JPEG or WebP image")
        if len(data) > MAX_IMAGE_BYTES:
            raise RuntimeError("the generated image is larger than the limit and wasn't saved")
        return GeneratedImage(data=data, mime_type=mime, cost_usd=0.0)  # a subscription: nothing billed per image

    @staticmethod
    def _find_image(work: Path, started: float) -> Path | None:
        """The image the CLI made: in the scratch folder if it put it there,
        else the newest image written under its own folder during this call."""
        here = [p for p in work.rglob("*") if p.suffix.lower() in _IMAGE_SUFFIXES and p.is_file()]
        if here:
            return max(here, key=lambda p: p.stat().st_mtime)
        home = Path(os.environ.get("CODEX_HOME") or Path.home() / ".codex")
        newest: tuple[float, Path] | None = None
        scanned = 0
        for root, _dirs, files in os.walk(home):
            for name in files:
                scanned += 1
                if scanned > _MAX_SCANNED:
                    return newest[1] if newest else None
                path = Path(root) / name
                if path.suffix.lower() in _IMAGE_SUFFIXES:
                    try:
                        mtime = path.stat().st_mtime
                    except OSError:
                        continue
                    if mtime >= started - 1 and (newest is None or mtime > newest[0]):
                        newest = (mtime, path)
        return newest[1] if newest else None
