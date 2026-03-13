"""AI provider abstraction layer for StudyPilot.

Supports Anthropic (Claude) and OpenAI (ChatGPT) via a unified interface.
Settings are stored in settings.json and fall back to .env values.
"""

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
        full_messages.extend(messages)
        response = client.chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            messages=full_messages,
        )
        return response.choices[0].message.content

    else:  # anthropic (default)
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        kwargs = dict(model=model, max_tokens=max_tokens, messages=messages)
        if system:
            kwargs["system"] = system
        response = client.messages.create(**kwargs)
        return response.content[0].text
