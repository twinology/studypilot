"""Telegram bot for the AI Tutor."""

import asyncio
import logging

from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

import config
from ai_provider import create_chat_completion, is_configured
from rag.chain import rag_query
from rag.vector_store import list_documents, search as rag_search
from tutor.feedback import generate_feedback
from tutor.scenarios import get_scenario_by_id, generate_dynamic_scenario
from tutor.session import SessionManager

logger = logging.getLogger(__name__)

session_manager = SessionManager()
chat_histories: dict[str, list[dict]] = {}

# Per-user store of generated scenarios (so /toon can list them)
user_scenarios: dict[str, list[dict]] = {}


async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Welkom bij de AI StudyPilot van twinology.ai!\n\n"
        "Beschikbare commando's:\n"
        "/genereer <onderwerp> [niveau] - Genereer een oefenscenario\n"
        "/toon - Toon je gegenereerde scenario's\n"
        "/oefen <nummer> - Start een oefensessie met een scenario\n"
        "/mctoets [onderwerp] - Multiple choice toets (5 vragen)\n"
        "/opentoets [onderwerp] - Open vragen toets (3 vragen)\n"
        "/stop - Stop sessie/toets en ontvang feedback\n"
        "/documenten - Toon geUploade documenten\n\n"
        "Niveaus: basis, gevorderd, expert\n"
        "Of stel gewoon een vraag!"
    )


async def cmd_documenten(update: Update, context: ContextTypes.DEFAULT_TYPE):
    docs = list_documents()
    if not docs:
        await update.message.reply_text(
            "Er zijn nog geen documenten geupload.\n"
            "Upload documenten via de web interface: http://localhost:8000"
        )
        return
    text = "Geploade documenten:\n\n" + "\n".join(f"- {d}" for d in docs)
    await update.message.reply_text(text)


# ── /genereer ─────────────────────────────────────────────────
async def cmd_genereer(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Generate a scenario based on a topic and optional difficulty level."""
    user_id = str(update.effective_user.id)

    if not context.args:
        await update.message.reply_text(
            "Gebruik: `/genereer <onderwerp>` of `/genereer <onderwerp> <niveau>`\n\n"
            "Niveaus:\n"
            "🟢 `basis` - Fundamenten en basisvaardigheden\n"
            "🟠 `gevorderd` - Zelfstandig werken met ervaring\n"
            "🔴 `expert` - Complexe problemen oplossen\n\n"
            "Voorbeeld: `/genereer boze klant gevorderd`",
            parse_mode="Markdown",
        )
        return

    # Parse arguments: last word might be difficulty level
    valid_levels = {"basis", "gevorderd", "expert"}
    args = context.args
    if args[-1].lower() in valid_levels:
        difficulty = args[-1].lower()
        topic = " ".join(args[:-1]) if len(args) > 1 else None
    else:
        difficulty = "basis"
        topic = " ".join(args)

    if not topic:
        await update.message.reply_text("Geef een onderwerp op. Voorbeeld: `/genereer boze klant gevorderd`", parse_mode="Markdown")
        return

    level_emoji = {"basis": "🟢", "gevorderd": "🟠", "expert": "🔴"}
    await update.message.reply_text(
        f"Scenario wordt gegenereerd...\n"
        f"Onderwerp: {topic}\n"
        f"Niveau: {level_emoji.get(difficulty, '')} {difficulty}"
    )

    try:
        scenario = await generate_dynamic_scenario(topic, difficulty)
    except Exception as e:
        await update.message.reply_text(f"Fout bij genereren scenario: {e}")
        return

    # Store scenario for this user
    if user_id not in user_scenarios:
        user_scenarios[user_id] = []
    user_scenarios[user_id].append(scenario.to_dict())
    nr = len(user_scenarios[user_id])

    await update.message.reply_text(
        f"*Scenario #{nr} gegenereerd!*\n\n"
        f"*{scenario.name}*\n"
        f"_{scenario.description}_\n\n"
        f"{level_emoji.get(scenario.difficulty, '')} {scenario.difficulty} | {scenario.conversation_style}\n\n"
        f"Start dit scenario met: `/oefen {nr}`\n"
        f"Bekijk al je scenario's met: /toon",
        parse_mode="Markdown",
    )


# ── /toon ─────────────────────────────────────────────────────
async def cmd_toon(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show the user's generated scenarios."""
    user_id = str(update.effective_user.id)
    scenarios = user_scenarios.get(user_id, [])

    if not scenarios:
        await update.message.reply_text(
            "Je hebt nog geen scenario's gegenereerd.\n"
            "Gebruik `/genereer <onderwerp> [niveau]` om er een te maken.\n\n"
            "Voorbeeld: `/genereer slechtnieuwsgesprek gevorderd`",
            parse_mode="Markdown",
        )
        return

    level_emoji = {"basis": "🟢", "gevorderd": "🟠", "expert": "🔴"}
    lines = ["*Je scenario's:*\n"]
    for i, s in enumerate(scenarios, 1):
        emoji = level_emoji.get(s["difficulty"], "")
        lines.append(
            f"*{i}.* {s['name']}\n"
            f"   {emoji} {s['difficulty']} | {s['conversation_style']}\n"
            f"   _{s['description']}_\n"
        )
    lines.append(f"Start een scenario met: `/oefen <nummer>` (1-{len(scenarios)})")

    await update.message.reply_text("\n".join(lines), parse_mode="Markdown")


# ── /oefen ────────────────────────────────────────────────────
async def cmd_oefen(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Start a practice session with a generated scenario (by number)."""
    user_id = str(update.effective_user.id)

    if session_manager.has_active_session(user_id):
        await update.message.reply_text("Je hebt al een actieve sessie. Gebruik /stop om deze te beëindigen.")
        return

    scenarios = user_scenarios.get(user_id, [])

    if not context.args:
        if not scenarios:
            await update.message.reply_text(
                "Je hebt nog geen scenario's. Genereer er eerst een:\n"
                "`/genereer <onderwerp> [niveau]`\n\n"
                "Voorbeeld: `/genereer boze klant gevorderd`",
                parse_mode="Markdown",
            )
        else:
            await update.message.reply_text(
                f"Gebruik: `/oefen <nummer>` (1-{len(scenarios)})\n"
                "Bekijk je scenario's met /toon",
                parse_mode="Markdown",
            )
        return

    # Parse scenario number
    try:
        nr = int(context.args[0])
    except ValueError:
        await update.message.reply_text(
            "Geef een scenario-nummer op.\n"
            "Bekijk je scenario's met /toon",
        )
        return

    if nr < 1 or nr > len(scenarios):
        await update.message.reply_text(
            f"Ongeldig nummer. Je hebt {len(scenarios)} scenario('s).\n"
            "Bekijk ze met /toon",
        )
        return

    # Get the scenario object from the store
    scenario_data = scenarios[nr - 1]
    from tutor.scenarios import Scenario, store_scenario
    scenario = Scenario(**{k: v for k, v in scenario_data.items()})
    store_scenario(scenario)  # ensure it's in the in-memory store

    session = session_manager.start_session(user_id, scenario)

    # Generate opening message
    system = f"""Je bent een acteur die een rol speelt voor een oefengesprek.

## Je rol:
{scenario.role_instruction}

## Belangrijk:
- Blijf ALTIJD in je rol
- Communiceer in het Nederlands
- Begin het gesprek vanuit je rol"""

    try:
        opening = create_chat_completion(
            messages=[{"role": "user", "content": "Start het gesprek vanuit je rol."}],
            system=system,
            max_tokens=config.MAX_TOKENS,
        )
    except Exception as e:
        session_manager.end_session(user_id)
        await update.message.reply_text(f"Fout bij starten sessie: {e}")
        return

    session.add_message("tutor", opening)

    level_emoji = {"basis": "🟢", "gevorderd": "🟠", "expert": "🔴"}
    await update.message.reply_text(
        f"*Oefensessie gestart: {scenario.name}*\n"
        f"{level_emoji.get(scenario.difficulty, '')} {scenario.difficulty}\n\n"
        f"---\n\n{opening}\n\n---\n"
        f"_Reageer in karakter. Gebruik /stop om de sessie te beëindigen en feedback te ontvangen._",
        parse_mode="Markdown",
    )


# ── /mctoets ──────────────────────────────────────────────────
async def cmd_mctoets(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Generate a multiple choice quiz (5 questions, A/B/C answers)."""
    user_id = str(update.effective_user.id)

    if session_manager.has_active_session(user_id):
        await update.message.reply_text("Je hebt een actieve sessie. Gebruik /stop om deze eerst te beëindigen.")
        return

    topic = " ".join(context.args) if context.args else None
    query = topic or "belangrijkste concepten theorie modellen"

    await update.message.reply_text("Multiple choice toets wordt gegenereerd... ⏳")

    context_items = rag_search(query, top_k=8)
    if not context_items:
        await update.message.reply_text(
            "Er zijn geen documenten in de kennisbank gevonden.\n"
            "Upload eerst documenten via de web interface."
        )
        return

    context_text = "\n\n".join(item["text"] for item in context_items)

    prompt = f"""Op basis van de volgende context uit de kennisbank, genereer een multiple choice kennistoets.

Context:
{context_text}

{"Specifiek onderwerp: " + topic if topic else "Kies de belangrijkste concepten uit de context."}

Genereer precies 5 multiple choice vragen. Elke vraag heeft precies 3 antwoordmogelijkheden: A, B en C.

BELANGRIJK - Gebruik EXACT dit formaat (de student antwoordt via Telegram):

1. [Vraag tekst]
   A) [antwoord optie A]
   B) [antwoord optie B]
   C) [antwoord optie C]

2. [Vraag tekst]
   A) [antwoord optie A]
   B) [antwoord optie B]
   C) [antwoord optie C]

(enzovoort voor alle 5 vragen)

Regels:
- Varieer in moeilijkheid
- Focus op begrip en toepassing, niet alleen feitenkennis
- Maak de foute antwoorden plausibel (geen onzin-opties)
- Geef NIET aan welk antwoord juist is

Eindig met exact deze tekst:
"Stuur je antwoorden als: 1A 2B 3C 4A 5B (of per vraag apart, bijv. '1A'). Typ /stop als je klaar bent."
"""

    try:
        quiz = create_chat_completion(
            messages=[{"role": "user", "content": prompt}],
            system="Je bent een deskundige AI-tutor die kennistoetsen maakt. Communiceer in het Nederlands.",
            max_tokens=config.MAX_TOKENS,
        )
    except Exception as e:
        await update.message.reply_text(f"Fout bij genereren toets: {e}")
        return

    from tutor.scenarios import Scenario, store_scenario
    quiz_scenario = Scenario(
        id=f"mctoets_{user_id}_{len(user_scenarios.get(user_id, []))}",
        name="MC Toets" + (f": {topic}" if topic else ""),
        description="Multiple choice kennistoets",
        role_instruction=f"""Je bent een eerlijke examinator voor een multiple choice toets.
De student maakt deze toets:

{quiz}

## Je gedrag:
- De student stuurt antwoorden in het formaat "1A 2B 3C" of per vraag "1A" of "3C"
- Accepteer ook losse letters als het duidelijk is welke vraag bedoeld wordt (bijv. de volgende onbeantwoorde vraag)
- Bij elk antwoord: geef aan of het GOED ✅ of FOUT ❌ is
- Bij een fout antwoord: geef kort het juiste antwoord en een korte uitleg
- Bij een goed antwoord: bevestig kort waarom dit juist is
- Houd de score bij (bijv. "Score: 3/5")
- Als alle 5 vragen beantwoord zijn, geef automatisch een samenvatting met de eindscore""",
        conversation_style="Examinerend, eerlijk, beknopt",
        difficulty="gevorderd",
        is_dynamic=True,
    )
    store_scenario(quiz_scenario)

    session = session_manager.start_session(user_id, quiz_scenario)
    session.add_message("tutor", quiz)

    if len(quiz) <= 4000:
        await update.message.reply_text(f"*Multiple Choice Toets*\n\n{quiz}", parse_mode="Markdown")
    else:
        parts = [quiz[i:i + 4000] for i in range(0, len(quiz), 4000)]
        for i, part in enumerate(parts):
            prefix = "*Multiple Choice Toets*\n\n" if i == 0 else ""
            await update.message.reply_text(f"{prefix}{part}", parse_mode="Markdown")


# ── /opentoets ────────────────────────────────────────────────
async def cmd_opentoets(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Generate an open questions quiz (3 questions)."""
    user_id = str(update.effective_user.id)

    if session_manager.has_active_session(user_id):
        await update.message.reply_text("Je hebt een actieve sessie. Gebruik /stop om deze eerst te beëindigen.")
        return

    topic = " ".join(context.args) if context.args else None
    query = topic or "belangrijkste concepten theorie modellen"

    await update.message.reply_text("Open vragen toets wordt gegenereerd... ⏳")

    context_items = rag_search(query, top_k=8)
    if not context_items:
        await update.message.reply_text(
            "Er zijn geen documenten in de kennisbank gevonden.\n"
            "Upload eerst documenten via de web interface."
        )
        return

    context_text = "\n\n".join(item["text"] for item in context_items)

    prompt = f"""Op basis van de volgende context uit de kennisbank, genereer een toets met open vragen.

Context:
{context_text}

{"Specifiek onderwerp: " + topic if topic else "Kies de belangrijkste concepten uit de context."}

Genereer precies 3 open vragen. Gebruik EXACT dit formaat:

Vraag 1: [vraag tekst]

Vraag 2: [vraag tekst]

Vraag 3: [vraag tekst]

Regels:
- Vraag 1: basisniveau (begrip toetsen)
- Vraag 2: toepassingsniveau (scenario/voorbeeld)
- Vraag 3: analyseniveau (vergelijken, beargumenteren)
- Houd de vragen beknopt en helder
- De student moet in 2-4 zinnen kunnen antwoorden

Eindig met exact deze tekst:
"Beantwoord per vraag. Stuur bijv.: '1. [jouw antwoord]' of beantwoord ze allemaal tegelijk. Typ /stop als je klaar bent."
"""

    try:
        quiz = create_chat_completion(
            messages=[{"role": "user", "content": prompt}],
            system="Je bent een deskundige AI-tutor die kennistoetsen maakt. Communiceer in het Nederlands.",
            max_tokens=config.MAX_TOKENS,
        )
    except Exception as e:
        await update.message.reply_text(f"Fout bij genereren toets: {e}")
        return

    from tutor.scenarios import Scenario, store_scenario
    quiz_scenario = Scenario(
        id=f"opentoets_{user_id}_{len(user_scenarios.get(user_id, []))}",
        name="Open Toets" + (f": {topic}" if topic else ""),
        description="Open vragen kennistoets",
        role_instruction=f"""Je bent een eerlijke examinator voor een toets met open vragen.
De student maakt deze toets:

{quiz}

## Je gedrag:
- De student stuurt antwoorden per vraag (bijv. "1. [antwoord]") of meerdere tegelijk
- Beoordeel elk antwoord als: GOED ✅, DEELS GOED 🟡, of ONVOLDOENDE ❌
- Geef per antwoord:
  1. Je beoordeling (✅/🟡/❌)
  2. Wat goed was (1 zin)
  3. Wat miste of beter kon (1 zin, alleen bij 🟡 of ❌)
  4. Het ideale antwoord in het kort (alleen bij ❌)
- Houd bij welke vragen al beantwoord zijn
- Als alle 3 vragen beantwoord zijn, geef automatisch een samenvatting met eindbeoordeling""",
        conversation_style="Examinerend, eerlijk, constructief",
        difficulty="gevorderd",
        is_dynamic=True,
    )
    store_scenario(quiz_scenario)

    session = session_manager.start_session(user_id, quiz_scenario)
    session.add_message("tutor", quiz)

    if len(quiz) <= 4000:
        await update.message.reply_text(f"*Open Vragen Toets*\n\n{quiz}", parse_mode="Markdown")
    else:
        parts = [quiz[i:i + 4000] for i in range(0, len(quiz), 4000)]
        for i, part in enumerate(parts):
            prefix = "*Open Vragen Toets*\n\n" if i == 0 else ""
            await update.message.reply_text(f"{prefix}{part}", parse_mode="Markdown")


# ── /stop ─────────────────────────────────────────────────────
async def cmd_stop(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = str(update.effective_user.id)
    session = session_manager.end_session(user_id)

    if not session:
        await update.message.reply_text("Je hebt geen actieve oefensessie.")
        return

    await update.message.reply_text("Sessie beëindigd. Feedback wordt gegenereerd... ⏳")

    if len(session.messages) < 2:
        await update.message.reply_text("Te weinig berichten voor feedback. Probeer volgende keer langer te oefenen.")
        return

    feedback = await generate_feedback(session)

    # Split long messages (Telegram limit is 4096 chars)
    if len(feedback) <= 4000:
        await update.message.reply_text(f"*Feedback*\n\n{feedback}", parse_mode="Markdown")
    else:
        parts = [feedback[i:i + 4000] for i in range(0, len(feedback), 4000)]
        for i, part in enumerate(parts):
            prefix = f"*Feedback ({i + 1}/{len(parts)})*\n\n" if i == 0 else ""
            await update.message.reply_text(f"{prefix}{part}", parse_mode="Markdown")


# ── Message handler ───────────────────────────────────────────
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle regular text messages."""
    user_id = str(update.effective_user.id)
    text = update.message.text

    # Check for active practice session
    session = session_manager.get_session(user_id)
    if session:
        session.add_message("student", text)

        context_items = rag_search(text, top_k=3)
        context_hint = ""
        if context_items:
            context_hint = "\n\n[Interne hint: " + " ".join(
                item["text"][:200] for item in context_items
            ) + "]"

        system = f"""Je bent een acteur die een rol speelt voor een oefengesprek.

## Je rol:
{session.scenario.role_instruction}

## Belangrijk:
- Blijf ALTIJD in je rol
- Communiceer in het Nederlands
- Houd antwoorden beknopt (2-4 zinnen){context_hint}"""

        try:
            reply = create_chat_completion(
                messages=session.to_claude_messages(),
                system=system,
                max_tokens=512,
            )
        except Exception as e:
            session.messages.pop()
            await update.message.reply_text(f"Fout: {e}")
            return

        session.add_message("tutor", reply)
        await update.message.reply_text(reply)
        return

    # Regular RAG chat
    history = chat_histories.get(user_id, [])
    try:
        result = rag_query(text, chat_history=history[-10:])
    except Exception as e:
        await update.message.reply_text(f"Fout: {e}")
        return

    if user_id not in chat_histories:
        chat_histories[user_id] = []
    chat_histories[user_id].append({"role": "user", "content": text})
    chat_histories[user_id].append({"role": "assistant", "content": result["answer"]})

    answer = result["answer"]
    if result.get("sources"):
        answer += f"\n\n Bronnen: {', '.join(result['sources'])}"

    # Split if too long
    if len(answer) <= 4000:
        await update.message.reply_text(answer)
    else:
        parts = [answer[i:i + 4000] for i in range(0, len(answer), 4000)]
        for part in parts:
            await update.message.reply_text(part)


# ── Bot runner ────────────────────────────────────────────────
def run_bot():
    """Run the Telegram bot (blocking, call from a thread)."""
    if not config.TELEGRAM_BOT_TOKEN:
        logger.warning("TELEGRAM_BOT_TOKEN niet ingesteld. Bot niet gestart.")
        return

    app = Application.builder().token(config.TELEGRAM_BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("documenten", cmd_documenten))
    app.add_handler(CommandHandler("genereer", cmd_genereer))
    app.add_handler(CommandHandler("toon", cmd_toon))
    app.add_handler(CommandHandler("oefen", cmd_oefen))
    app.add_handler(CommandHandler("mctoets", cmd_mctoets))
    app.add_handler(CommandHandler("opentoets", cmd_opentoets))
    app.add_handler(CommandHandler("stop", cmd_stop))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    logger.info("Telegram bot gestart")
    app.run_polling(allowed_updates=Update.ALL_TYPES)
