"""AI provider abstraction layer for StudyPilot.

Supports Anthropic (Claude) and OpenAI (ChatGPT) via a unified interface.
Settings are stored in settings.json and fall back to .env values.
"""

import base64
import json
import logging
from pathlib import Path

import config

logger = logging.getLogger("tutor")

SETTINGS_FILE = config.BASE_DIR / "settings.json"

_DEFAULTS = {
    "provider": "anthropic",
    "model": "claude-sonnet-4-6",
    "ai_api_key": "",
    "elevenlabs_api_key": "",
}


def load_settings() -> dict:
    """Load settings from settings.json, with .env fallback."""
    settings = dict(_DEFAULTS)
    if SETTINGS_FILE.exists():
        try:
            with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
                stored = json.load(f)
            settings.update(stored)
        except Exception as e:
            logger.warning(f"Kon settings.json niet laden: {e}")
    # .env fallback
    if not settings["ai_api_key"] and config.ANTHROPIC_API_KEY:
        settings["ai_api_key"] = config.ANTHROPIC_API_KEY
        settings["provider"] = "anthropic"
    if not settings["elevenlabs_api_key"] and config.ELEVENLABS_API_KEY:
        settings["elevenlabs_api_key"] = config.ELEVENLABS_API_KEY
    return settings


def save_settings(new_settings: dict) -> None:
    """Persist settings to settings.json."""
    current = load_settings()
    current.update(new_settings)
    with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
        json.dump(current, f, indent=2, ensure_ascii=False)
    logger.info(f"Instellingen opgeslagen: provider={current.get('provider')}, model={current.get('model')}")


def is_configured() -> bool:
    """Return True if an AI API key is present."""
    return bool(load_settings().get("ai_api_key", "").strip())


def get_active_model() -> str:
    return load_settings().get("model", _DEFAULTS["model"])


def get_active_provider() -> str:
    return load_settings().get("provider", _DEFAULTS["provider"])


def create_chat_completion(
    messages: list,
    system: str = "",
    max_tokens: int = None,
) -> str:
    """Unified AI call. Returns the response text as a string.

    Works with both Anthropic and OpenAI. Messages format:
        [{"role": "user"|"assistant", "content": "..."}]
    """
    if max_tokens is None:
        max_tokens = config.MAX_TOKENS

    settings = load_settings()
    provider = settings.get("provider", "anthropic")
    model = settings.get("model", _DEFAULTS["model"])
    api_key = settings.get("ai_api_key", "")

    if not api_key:
        raise RuntimeError(
            "Geen AI API-key geconfigureerd. Ga naar de Setup tab om een key in te stellen."
        )

    if provider == "openai":
        import openai
        client = openai.OpenAI(api_key=api_key)
        full_messages = []
        if system:
            full_messages.append({"role": "system", "content": system})
        # Normalize image content blocks for OpenAI format
        full_messages.extend(_normalize_messages_for_openai(messages))
        response = client.chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            messages=full_messages,
        )
        return response.choices[0].message.content

    else:  # anthropic (default)
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        # Normalize image content blocks for Anthropic format
        normalized = _normalize_messages_for_anthropic(messages)
        kwargs = dict(model=model, max_tokens=max_tokens, messages=normalized)
        if system:
            kwargs["system"] = system
        response = client.messages.create(**kwargs)
        return response.content[0].text


# ── Multimodal helpers ────────────────────────────────────────────────


def _normalize_messages_for_anthropic(messages: list) -> list:
    """Ensure image content blocks use Anthropic format.

    Converts OpenAI-style image_url blocks to Anthropic image blocks if needed.
    Plain string content passes through unchanged.
    """
    result = []
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, str):
            result.append(msg)
            continue
        if isinstance(content, list):
            new_content = []
            for block in content:
                if block.get("type") == "image_url":
                    # Convert OpenAI format → Anthropic format
                    url = block["image_url"]["url"]
                    if url.startswith("data:"):
                        # data:image/png;base64,XXXX
                        header, b64_data = url.split(",", 1)
                        media_type = header.split(":")[1].split(";")[0]
                        new_content.append({
                            "type": "image",
                            "source": {"type": "base64", "media_type": media_type, "data": b64_data},
                        })
                    else:
                        new_content.append({
                            "type": "image",
                            "source": {"type": "url", "url": url},
                        })
                else:
                    new_content.append(block)
            result.append({**msg, "content": new_content})
        else:
            result.append(msg)
    return result


def _normalize_messages_for_openai(messages: list) -> list:
    """Ensure image content blocks use OpenAI format.

    Converts Anthropic-style image blocks to OpenAI image_url blocks if needed.
    Plain string content passes through unchanged.
    """
    result = []
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, str):
            result.append(msg)
            continue
        if isinstance(content, list):
            new_content = []
            for block in content:
                if block.get("type") == "image" and "source" in block:
                    src = block["source"]
                    if src.get("type") == "base64":
                        data_uri = f"data:{src['media_type']};base64,{src['data']}"
                        new_content.append({
                            "type": "image_url",
                            "image_url": {"url": data_uri},
                        })
                    elif src.get("type") == "url":
                        new_content.append({
                            "type": "image_url",
                            "image_url": {"url": src["url"]},
                        })
                    else:
                        new_content.append(block)
                else:
                    new_content.append(block)
            result.append({**msg, "content": new_content})
        else:
            result.append(msg)
    return result


def create_vision_completion(
    prompt: str,
    image_bytes: bytes,
    mime_type: str = "image/png",
    max_tokens: int = 512,
) -> str:
    """Send an image to the vision API and return the text description.

    Works with both Anthropic (Claude) and OpenAI (GPT-4o) vision.
    """
    settings = load_settings()
    provider = settings.get("provider", "anthropic")
    model = settings.get("model", _DEFAULTS["model"])
    api_key = settings.get("ai_api_key", "")

    if not api_key:
        raise RuntimeError("Geen AI API-key geconfigureerd.")

    b64_data = base64.b64encode(image_bytes).decode("utf-8")

    if provider == "openai":
        import openai
        client = openai.OpenAI(api_key=api_key)
        # OpenAI vision uses gpt-4o models; fall back if model doesn't support vision
        vision_model = model if "gpt-4" in model else "gpt-4o"
        response = client.chat.completions.create(
            model=vision_model,
            max_tokens=max_tokens,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{b64_data}"}},
                    {"type": "text", "text": prompt},
                ],
            }],
        )
        return response.choices[0].message.content

    else:  # anthropic
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": mime_type, "data": b64_data}},
                    {"type": "text", "text": prompt},
                ],
            }],
        )
        return response.content[0].text
