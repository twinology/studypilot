"""Scenario definitions and dynamic scenario generation."""

import json
from dataclasses import dataclass, asdict
from typing import Optional

from ai_provider import create_chat_completion
from rag.vector_store import search


# Difficulty levels with descriptions
DIFFICULTY_LEVELS = {
    "basis": "Eenvoudige situatie, duidelijke signalen, voorspelbare reacties",
    "gevorderd": "Complexere situatie, subtielere signalen, meerdere lagen",
    "expert": "Zeer complexe situatie, verborgen agenda's, sterke emoties, meerdere dilemma's",
}


@dataclass
class Scenario:
    id: str
    name: str
    description: str
    role_instruction: str
    conversation_style: str
    difficulty: str  # "basis", "gevorderd", "expert"
    gender: str = "female"  # "male" of "female" — bepaalt TTS stem
    emotion: str = "neutraal"  # emotie van het personage — bepaalt TTS stemkleur
    character_name: str = ""  # naam van het personage
    is_dynamic: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


# In-memory store for generated scenarios (so they can be started by ID)
_generated_scenarios: dict[str, Scenario] = {}


def get_scenario_by_id(scenario_id: str) -> Optional[Scenario]:
    """Get a generated scenario by ID."""
    return _generated_scenarios.get(scenario_id)


def store_scenario(scenario: Scenario) -> None:
    """Store a generated scenario so it can be retrieved by ID."""
    _generated_scenarios[scenario.id] = scenario


async def generate_dynamic_scenario(
    topic: Optional[str] = None,
    difficulty: str = "basis",
    use_rag: bool = True,
) -> Scenario:
    """Generate a new scenario using the configured AI provider."""
    if difficulty not in DIFFICULTY_LEVELS:
        difficulty = "basis"

    difficulty_desc = DIFFICULTY_LEVELS[difficulty]

    context_block = ""
    if use_rag:
        query = topic or "oefensituaties praktijksituaties"
        context_items = search(query, top_k=5)
        context_text = "\n\n".join(item["text"] for item in context_items)
        if context_text.strip():
            context_block = f"""
Hieronder staat relevante context uit de kennisbank. Gebruik dit als inspiratie en basis voor het scenario.

Context:
{context_text}
"""

    prompt = f"""Genereer een realistisch, interactief oefenscenario.
{context_block}
Thema / onderwerp: {topic or "kies een passend onderwerp"}

Moeilijkheidsgraad: {difficulty}
Beschrijving niveau: {difficulty_desc}

Het scenario moet een interactieve oefensituatie zijn waarbij de student in gesprek gaat met een personage.
Het personage en de situatie moeten passen bij het opgegeven thema.

Pas de complexiteit aan op het niveau:
- basis: Duidelijke situatie, direct herkenbaar, voorspelbare reacties
- gevorderd: Meer nuance, subtielere signalen, de student moet goed doorvragen
- expert: Zeer uitdagende situatie, complexe dynamiek, meerdere lagen

Genereer een scenario in exact dit JSON-formaat (zonder markdown codeblokken):
{{
    "id": "uniek_id_in_snake_case",
    "name": "Korte naam",
    "description": "Beschrijving van de situatie (1-2 zinnen)",
    "role_instruction": "Gedetailleerde instructie voor de rollenspeler (minimaal 5 regels met specifiek gedrag, persoonlijkheid en achtergrond)",
    "conversation_style": "Korte beschrijving van de gespreksstijl",
    "difficulty": "{difficulty}",
    "gender": "male of female — het geslacht van het personage",
    "emotion": "de dominante emotie van het personage — kies uit: boos, gefrustreerd, verdrietig, angstig, nerveus, gestrest, teleurgesteld, passief_agressief, manipulatief, neutraal, vriendelijk",
    "character_name": "Voornaam van het personage (bijv. Marie, Jan, Ahmed)"
}}"""

    result = create_chat_completion(
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1024,
        return_usage=True,
    )
    text = result["text"]
    usage = result["usage"]

    data = json.loads(text)
    gender = data.get("gender", "female").lower()
    if gender not in ("male", "female"):
        gender = "female"

    emotion = data.get("emotion", "neutraal").lower()
    character_name = data.get("character_name", "").strip()

    scenario = Scenario(
        id=data["id"],
        name=data["name"],
        description=data["description"],
        role_instruction=data["role_instruction"],
        conversation_style=data["conversation_style"],
        difficulty=data.get("difficulty", difficulty),
        gender=gender,
        emotion=emotion,
        character_name=character_name,
        is_dynamic=True,
    )

    store_scenario(scenario)
    return scenario, usage
