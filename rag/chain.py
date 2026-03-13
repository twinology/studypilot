"""RAG chain: retrieve context and generate answers via AI provider."""

from typing import List

from config import MAX_TOKENS
from ai_provider import create_chat_completion
from rag.vector_store import search

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

    if use_rag:
        context_items = search(user_message)
        context_text = "\n\n---\n\n".join(
            f"[Bron: {item['source']}]\n{item['text']}" for item in context_items
        )
        system = SYSTEM_PROMPT
        if context_text:
            system += f"\n\n## Relevante context uit de kennisbank:\n\n{context_text}"
    else:
        system = DIRECT_SYSTEM_PROMPT

    messages = []
    if chat_history:
        messages.extend(chat_history)
    messages.append({"role": "user", "content": user_message})

    try:
        answer = create_chat_completion(
            messages=messages, system=system, max_tokens=MAX_TOKENS
        )
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

    usage = {
        "input_tokens": "?",
        "output_tokens": "?",
        "context_chunks": len(context_items),
    }

    return {"answer": answer, "sources": sources, "usage": usage}
