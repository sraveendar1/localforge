"""Multimodal message content: a plain string (the overwhelmingly common
case) or a list of content blocks (text + an attached image) in the shape
LiteLLM/OpenAI use, which litellm.completion() translates natively for
Claude/GPT/Gemini vision models without us doing per-provider work.

Kept as its own module, rather than living in orchestrator.py or memory.py,
so every module that touches conversation messages -- orchestrator,
memory, cli_transport, local_transport -- can extract a message's text
without importing each other (memory.py in particular must not import
orchestrator.py, which already imports memory.py).
"""
from __future__ import annotations


def text_of(content) -> str:
    """The text portion of a message's content, whether it's a plain
    string (the common case) or a multimodal content-block list -- an
    attached image block is simply not part of the text rendering, which
    matters for memory compaction and CLI-transport prompt flattening,
    neither of which can do anything with an inline image anyway."""
    if isinstance(content, list):
        return " ".join(
            block.get("text", "") for block in content if isinstance(block, dict) and block.get("type") == "text"
        )
    return str(content or "")


def has_image(content) -> bool:
    return isinstance(content, list) and any(isinstance(b, dict) and b.get("type") == "image_url" for b in content)


def image_url_block(mime_type: str, data_base64: str) -> dict:
    """A LiteLLM/OpenAI-shaped image content block (a data: URI)."""
    return {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{data_base64}"}}


def user_content(text: str, image: dict | None):
    """A user message's `content` value: plain text (unchanged) or a
    multimodal block list when `image` (`{"mime_type", "data"}`, `data`
    base64-encoded) is given."""
    if not image:
        return text
    return [{"type": "text", "text": text}, image_url_block(image["mime_type"], image["data"])]


def extract_image(messages: list[dict]) -> dict | None:
    """The image attached to the latest user message, if any, as
    `{"mime_type", "data"}` (`data` base64-encoded, no `data:` URI prefix)
    -- shared by local_transport.py and cli_transport.py, which each need
    to hand the image to their own backend a different way (Ollama's
    `images` field vs. an Anthropic-shaped content block)."""
    for m in reversed(messages):
        if m.get("role") != "user":
            continue
        content = m.get("content")
        if not has_image(content):
            return None
        for block in content:
            if isinstance(block, dict) and block.get("type") == "image_url":
                url = block.get("image_url", {}).get("url", "")
                if url.startswith("data:") and ";base64," in url:
                    header, data = url.split(",", 1)
                    return {"mime_type": header[len("data:") :].split(";")[0], "data": data}
                return None
        return None
    return None
