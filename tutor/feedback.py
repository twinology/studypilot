"""Generate detailed feedback after practice sessions."""

from config import MAX_TOKENS
from ai_provider import create_chat_completion
from rag.vector_store import search
from tutor.session import Session


async def generate_feedback(session: Session, use_rag: bool = True) -> str:
    """Analyze a completed session and generate detailed feedback."""
    conversation_log = session.get_conversation_log()

    # Optionally get relevant theory from documents
    theory_block = ""
    if use_rag:
        context_items = search(
            f"feedback {session.scenario.conversation_style} {session.scenario.name}",
            top_k=5,
        )
        theory_context = "\n\n".join(
            f"[{item['source']}]: {item['text']}" for item in context_items
        )
        if theory_context.strip():
            theory_block = f"""
## Relevante theorie uit de kennisbank
{theory_context}

Gebruik bovenstaande theorie waar relevant om je feedback te onderbouwen.
"""

    prompt = f"""Je bent een expert-coach. Analyseer het volgende oefengesprek en geef gedetailleerde feedback.

## Scenario
**{session.scenario.name}**: {session.scenario.description}
Gespreksstijl van de oefenpartner: {session.scenario.conversation_style}
Moeilijkheidsgraad: {session.scenario.difficulty}

## Gesprekslog
{conversation_log}
{theory_block}
## Geef feedback in het volgende format:

### Samenvatting
Een korte samenvatting van het gesprek (2-3 zinnen).

### Sterke punten
- Benoem 2-4 dingen die de student goed deed
- Onderbouw met theorie of vakkennis waar mogelijk

### Verbeterpunten
- Benoem 2-4 concrete verbeterpunten
- Leg uit WAAROM dit beter kan en HOE
- Verwijs naar relevante technieken of kennis die hier hadden kunnen helpen

### Concrete tips
- Geef 2-3 direct toepasbare tips voor een volgend gesprek
- Geef voorbeeldzinnen die de student had kunnen gebruiken

### Score
Geef een score van 1-10 met een korte toelichting.

Communiceer in het Nederlands. Wees eerlijk maar aanmoedigend."""

    return create_chat_completion(
        messages=[{"role": "user", "content": prompt}],
        max_tokens=MAX_TOKENS,
    )
