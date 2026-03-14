"""RAG chain: retrieve context and generate answers via AI provider."""

import base64
import logging
from pathlib import Path
from typing import List

import config
from config import MAX_TOKENS
from ai_provider import create_chat_completion, load_settings
from rag.vector_store import search

logger = logging.getLogger("tutor")


def _get_user_context() -> str:
    """Build a personal context string from user profile settings."""
    settings = load_settings()
    name = settings.get("user_name", "").strip()
    education = settings.get("user_education", "").strip()
    start_year = settings.get("user_start_year", "").strip()
    if not name:
        return ""
    parts = [f"\n\n## Gebruikersprofiel:\n- De student heet {name}. Spreek de student aan met '{name}' en gebruik 'je' en 'jij'."]
    if education:
        parts.append(f"- Opleiding: {education}")
    if start_year:
        parts.append(f"- Gestart in: {start_year}")
    return "\n".join(parts)

SYSTEM_PROMPT = """Je bent een deskundige AI-tutor.
Je helpt studenten om onderwerpen te begrijpen en toe te passen.

BELANGRIJKSTE REGEL: Je mag UITSLUITEND informatie gebruiken die in de aangeleverde context staat.
Je mag NOOIT eigen kennis, aannames of informatie van buiten de context toevoegen.
Als het antwoord niet (volledig) in de context te vinden is, zeg dat dan eerlijk.

## Antwoordstijl:
- Geef UITGEBREIDE, goed gestructureerde antwoorden. Korte antwoorden zijn niet gewenst.
- Gebruik kopjes (##), opsommingen en nummering om je antwoord overzichtelijk te maken.
- Leg concepten stap voor stap uit zodat de student het echt begrijpt.
- Geef concrete voorbeelden uit de context om abstracte concepten te verduidelijken.
- Sluit af met een korte samenvatting of kernpunten.
- Stel aan het einde 1-2 vervolgvragen om het begrip te toetsen.

## Brongebruik:
- Beantwoord vragen ALLEEN op basis van de aangeleverde context. Gebruik GEEN eigen kennis.
- Doorzoek ALLE aangeleverde contextfragmenten grondig voordat je concludeert dat informatie ontbreekt.
- Als de informatie in de context staat, gebruik deze dan — ook als het verspreid over meerdere fragmenten staat.
- Combineer informatie uit verschillende fragmenten tot een samenhangend, volledig antwoord.
- Citeer of parafraseer relevante passages uit de context om je antwoord te onderbouwen.
- Verwijs naar specifieke concepten, modellen of technieken uit de context wanneer relevant.
- Geef praktische voorbeelden ALLEEN als deze in de context staan.

## Belangrijk:
- Als de context na grondige analyse geen relevant antwoord bevat, antwoord dan:
  "Deze informatie staat niet in de kennisbank. Zet de kennisbank uit om een antwoord te krijgen op basis van het AI-model."
- Verzin NOOIT informatie. Vul NOOIT aan met eigen kennis wanneer de kennisbank actief is.
- Communiceer in het Nederlands."""

DIRECT_SYSTEM_PROMPT = """Je bent een deskundige AI-tutor.
Je helpt studenten om onderwerpen te begrijpen en toe te passen.

Je werkt nu in DIRECTE MODUS — zonder kennisbank. Je gebruikt je eigen kennis.

## Antwoordstijl:
- Geef UITGEBREIDE, goed gestructureerde antwoorden. Korte antwoorden zijn niet gewenst.
- Gebruik kopjes (##), opsommingen en nummering om je antwoord overzichtelijk te maken.
- Leg concepten stap voor stap uit zodat de student het echt begrijpt.
- Geef concrete voorbeelden om abstracte concepten te verduidelijken.
- Sluit af met een korte samenvatting of kernpunten.
- Stel aan het einde 1-2 vervolgvragen om het begrip te toetsen.

## Richtlijnen:
- Beantwoord vragen op basis van je eigen kennis en expertise.
- Geef praktische voorbeelden waar mogelijk.
- Communiceer in het Nederlands.
- Als je ergens niet zeker van bent, geef dat eerlijk aan."""


def rag_query(
    user_message: str,
    chat_history: List[dict] = None,
    use_rag: bool = True,
) -> dict:
    """Process a query. When use_rag=True, retrieves context from the knowledge base.
    When use_rag=False, queries directly without context.
    Returns {answer, sources, usage}."""

    context_items = []
    image_paths = []

    if use_rag:
        context_items = search(user_message)

        # Separate text chunks from image description chunks
        text_items = []
        image_items = []
        for item in context_items:
            meta = item.get("metadata", {})
            if meta.get("content_type") == "image_description":
                image_items.append(item)
            else:
                text_items.append(item)

        # Build text context (unchanged behavior)
        context_text = "\n\n---\n\n".join(
            f"[Bron: {item['source']}]\n{item['text']}" for item in text_items
        )

        # Add image descriptions to text context
        if image_items:
            image_context = "\n\n---\n\n".join(
                f"[Bron: {item['source']} — afbeelding]\n{item['text']}" for item in image_items
            )
            if context_text:
                context_text += "\n\n---\n\n" + image_context
            else:
                context_text = image_context

        system = SYSTEM_PROMPT
        system += _get_user_context()
        if context_text:
            system += f"\n\n## Relevante context uit de kennisbank:\n\n{context_text}"
    else:
        system = DIRECT_SYSTEM_PROMPT
        system += _get_user_context()
        image_items = []

    # Build messages — potentially with multimodal content blocks
    messages = []
    if chat_history:
        messages.extend(chat_history)

    # Build user message with optional inline images
    user_content = _build_multimodal_user_content(user_message, image_items)
    messages.append({"role": "user", "content": user_content})

    try:
        result = create_chat_completion(
            messages=messages, system=system, max_tokens=MAX_TOKENS, return_usage=True
        )
        answer = result["text"]
        token_usage = result["usage"]
    except RuntimeError:
        raise
    except Exception as e:
        error_msg = str(e).lower()
        if "credit" in error_msg or "balance" in error_msg or "quota" in error_msg:
            raise RuntimeError(
                "Onvoldoende API-tegoed. Controleer je abonnement bij je AI-provider."
            )
        if "authentication" in error_msg or "invalid" in error_msg:
            raise RuntimeError(
                "Ongeldige API-key. Controleer je key in de Setup tab."
            )
        raise RuntimeError(f"AI provider fout: {e}")

    sources = list(set(item["source"] for item in context_items))

    # Collect image paths for frontend display
    for item in image_items[:config.MAX_IMAGES_IN_CONTEXT]:
        meta = item.get("metadata", {})
        img_path = meta.get("image_path", "")
        if img_path:
            image_paths.append(img_path)

    usage = {
        "input_tokens": token_usage.get("input_tokens", 0),
        "output_tokens": token_usage.get("output_tokens", 0),
        "context_chunks": len(context_items),
        "image_chunks": len(image_items),
    }

    return {"answer": answer, "sources": sources, "usage": usage, "images": image_paths}


def _build_multimodal_user_content(user_message: str, image_items: list):
    """Build user message content — plain string or list with image blocks.

    If there are relevant image chunks, includes the actual images as base64
    content blocks (up to MAX_IMAGES_IN_CONTEXT). Otherwise returns a plain string.
    """
    if not image_items:
        return user_message

    # Limit number of images to avoid token overflow
    selected = image_items[:config.MAX_IMAGES_IN_CONTEXT]

    content_blocks = [{"type": "text", "text": user_message}]

    for item in selected:
        meta = item.get("metadata", {})
        img_rel_path = meta.get("image_path", "")
        if not img_rel_path:
            continue

        img_abs_path = config.BASE_DIR / img_rel_path
        if not img_abs_path.exists():
            logger.warning(f"Afbeelding niet gevonden voor context: {img_rel_path}")
            continue

        try:
            image_bytes = img_abs_path.read_bytes()
            b64_data = base64.b64encode(image_bytes).decode("utf-8")

            suffix = img_abs_path.suffix.lower()
            mime_map = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}
            mime_type = mime_map.get(suffix, "image/png")

            # Use Anthropic format — ai_provider will normalize for OpenAI if needed
            content_blocks.append({
                "type": "image",
                "source": {"type": "base64", "media_type": mime_type, "data": b64_data},
            })
            content_blocks.append({
                "type": "text",
                "text": f"[Bovenstaande afbeelding: {item['text'][:200]}]",
            })
        except Exception as e:
            logger.warning(f"Kon afbeelding niet laden voor context: {img_rel_path}: {e}")
            continue

    # If no images were actually loaded, return plain string
    if len(content_blocks) == 1:
        return user_message

    return content_blocks
