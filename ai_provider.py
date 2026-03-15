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
    "ollama_base_url": "http://localhost:11434",
    "openrouter_api_key": "",
    "tavily_api_key": "",
    "stt_provider": "browser",       # "browser", "openai_whisper", "local_whisper"
    "openai_stt_key": "",             # Separate key for OpenAI Whisper STT
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
    """Return True if an AI API key is present (or Ollama is selected)."""
    settings = load_settings()
    if settings.get("provider") == "ollama":
        return True  # Ollama needs no API key
    if settings.get("provider") == "openrouter":
        return bool(settings.get("openrouter_api_key", "").strip())
    return bool(settings.get("ai_api_key", "").strip())


def get_active_model() -> str:
    return load_settings().get("model", _DEFAULTS["model"])


def get_active_provider() -> str:
    return load_settings().get("provider", _DEFAULTS["provider"])


def _extract_usage(response, provider: str) -> dict:
    """Extract token usage from API response."""
    usage = {"input_tokens": 0, "output_tokens": 0}
    try:
        if provider == "anthropic":
            if hasattr(response, "usage"):
                usage["input_tokens"] = getattr(response.usage, "input_tokens", 0)
                usage["output_tokens"] = getattr(response.usage, "output_tokens", 0)
        elif provider == "openai_responses":
            # OpenAI Responses API uses input_tokens / output_tokens directly
            if hasattr(response, "usage") and response.usage:
                usage["input_tokens"] = getattr(response.usage, "input_tokens", 0) or 0
                usage["output_tokens"] = getattr(response.usage, "output_tokens", 0) or 0
        else:  # openai chat completions / ollama
            if hasattr(response, "usage") and response.usage:
                usage["input_tokens"] = getattr(response.usage, "prompt_tokens", 0) or 0
                usage["output_tokens"] = getattr(response.usage, "completion_tokens", 0) or 0
    except Exception as e:
        logger.warning(f"Kon token usage niet uitlezen: {e}")
    return usage


def create_chat_completion(
    messages: list,
    system: str = "",
    max_tokens: int = None,
    return_usage: bool = False,
):
    """Unified AI call. Returns the response text as a string.

    If return_usage=True, returns a dict: {"text": str, "usage": {"input_tokens": int, "output_tokens": int}}

    Works with both Anthropic and OpenAI. Messages format:
        [{"role": "user"|"assistant", "content": "..."}]
    """
    if max_tokens is None:
        max_tokens = config.MAX_TOKENS

    settings = load_settings()
    provider = settings.get("provider", "anthropic")
    model = settings.get("model", _DEFAULTS["model"])
    api_key = settings.get("ai_api_key", "")

    if provider == "openrouter":
        api_key = settings.get("openrouter_api_key", "")
    if provider not in ("ollama", "openrouter") and not api_key:
        raise RuntimeError(
            "Geen AI API-key geconfigureerd. Ga naar de Setup tab om een key in te stellen."
        )
    if provider == "openrouter" and not api_key:
        raise RuntimeError(
            "Geen OpenRouter API-key geconfigureerd. Ga naar de Setup tab om een key in te stellen."
        )

    if provider == "ollama":
        import openai as _openai
        base_url = settings.get("ollama_base_url", "http://localhost:11434")
        client = _openai.OpenAI(api_key="ollama", base_url=f"{base_url}/v1")
        full_messages = []
        if system:
            full_messages.append({"role": "system", "content": system})
        # Strip image blocks for Ollama (most local models don't support vision)
        full_messages.extend(_strip_image_blocks(messages))
        response = client.chat.completions.create(
            model=model,
            messages=full_messages,
        )
        text = response.choices[0].message.content
        if return_usage:
            return {"text": text, "usage": _extract_usage(response, "ollama")}
        return text

    elif provider == "openrouter":
        import openai as _openai
        client = _openai.OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
        )
        full_messages = []
        if system:
            full_messages.append({"role": "system", "content": system})
        full_messages.extend(_normalize_messages_for_openai(messages))
        response = client.chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            messages=full_messages,
        )
        text = response.choices[0].message.content
        if return_usage:
            return {"text": text, "usage": _extract_usage(response, "ollama")}  # same format
        return text

    elif provider == "openai":
        import openai
        client = openai.OpenAI(api_key=api_key)

        # Build input for Responses API
        input_items = []
        for msg in _normalize_messages_for_openai(messages):
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if isinstance(content, str):
                input_items.append({"role": role, "content": content})
            elif isinstance(content, list):
                # Multimodal content blocks (text + images)
                input_items.append({"role": role, "content": content})
            else:
                input_items.append({"role": role, "content": str(content)})

        kwargs = {"model": model, "input": input_items}
        if system:
            kwargs["instructions"] = system
        if max_tokens:
            kwargs["max_output_tokens"] = max_tokens

        response = client.responses.create(**kwargs)
        text = response.output_text
        if return_usage:
            return {"text": text, "usage": _extract_usage(response, "openai_responses")}
        return text

    else:  # anthropic (default)
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        # Normalize image content blocks for Anthropic format
        normalized = _normalize_messages_for_anthropic(messages)
        kwargs = dict(model=model, max_tokens=max_tokens, messages=normalized)
        if system:
            kwargs["system"] = system
        response = client.messages.create(**kwargs)
        text = response.content[0].text
        if return_usage:
            return {"text": text, "usage": _extract_usage(response, "anthropic")}
        return text


# ── Multimodal helpers ────────────────────────────────────────────────


def _strip_image_blocks(messages: list) -> list:
    """Remove image content blocks from messages (for providers that don't support vision).

    Converts multimodal content lists to plain text strings.
    """
    result = []
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, str):
            result.append(msg)
        elif isinstance(content, list):
            text_parts = []
            for block in content:
                if block.get("type") == "text":
                    text_parts.append(block["text"])
                elif block.get("type") in ("image", "image_url"):
                    text_parts.append("[afbeelding]")
            result.append({**msg, "content": "\n".join(text_parts) if text_parts else ""})
        else:
            result.append(msg)
    return result


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
    return_usage: bool = False,
):
    """Send an image to the vision API and return the text description.

    Works with both Anthropic (Claude) and OpenAI (GPT-4o) vision.
    """
    settings = load_settings()
    provider = settings.get("provider", "anthropic")
    model = settings.get("model", _DEFAULTS["model"])
    api_key = settings.get("ai_api_key", "")

    if provider == "openrouter":
        api_key = settings.get("openrouter_api_key", "")
    if provider not in ("ollama", "openrouter") and not api_key:
        raise RuntimeError("Geen AI API-key geconfigureerd.")
    if provider == "openrouter" and not api_key:
        raise RuntimeError("Geen OpenRouter API-key geconfigureerd.")

    b64_data = base64.b64encode(image_bytes).decode("utf-8")

    if provider == "ollama":
        # Ollama vision via compatible models (llava, etc.)
        import openai as _openai
        base_url = settings.get("ollama_base_url", "http://localhost:11434")
        client = _openai.OpenAI(api_key="ollama", base_url=f"{base_url}/v1")
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{b64_data}"}},
                        {"type": "text", "text": prompt},
                    ],
                }],
            )
            text = response.choices[0].message.content
            if return_usage:
                return {"text": text, "usage": _extract_usage(response, "ollama")}
            return text
        except Exception as e:
            logger.warning(f"Ollama vision mislukt ({model}): {e}. Afbeelding overgeslagen.")
            fallback = f"[Afbeelding kon niet worden beschreven door lokaal model {model}]"
            if return_usage:
                return {"text": fallback, "usage": {"input_tokens": 0, "output_tokens": 0}}
            return fallback

    elif provider == "openrouter":
        import openai as _openai
        client = _openai.OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
        )
        try:
            response = client.chat.completions.create(
                model=model,
                max_tokens=max_tokens,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{b64_data}"}},
                        {"type": "text", "text": prompt},
                    ],
                }],
            )
            text = response.choices[0].message.content
            if return_usage:
                return {"text": text, "usage": _extract_usage(response, "ollama")}
            return text
        except Exception as e:
            logger.warning(f"OpenRouter vision mislukt ({model}): {e}. Afbeelding overgeslagen.")
            fallback = f"[Afbeelding kon niet worden beschreven via OpenRouter model {model}]"
            if return_usage:
                return {"text": fallback, "usage": {"input_tokens": 0, "output_tokens": 0}}
            return fallback

    elif provider == "openai":
        import openai
        client = openai.OpenAI(api_key=api_key)
        # OpenAI Responses API with vision
        vision_model = model if "gpt-4" in model or "gpt-5" in model else "gpt-4o"
        response = client.responses.create(
            model=vision_model,
            max_output_tokens=max_tokens,
            input=[{
                "role": "user",
                "content": [
                    {"type": "input_image", "image_url": f"data:{mime_type};base64,{b64_data}"},
                    {"type": "input_text", "text": prompt},
                ],
            }],
        )
        text = response.output_text
        if return_usage:
            return {"text": text, "usage": _extract_usage(response, "openai_responses")}
        return text

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
        text = response.content[0].text
        if return_usage:
            return {"text": text, "usage": _extract_usage(response, "anthropic")}
        return text
