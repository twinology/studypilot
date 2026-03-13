"""FastAPI backend for StudyPilot."""

import asyncio
import io
import logging
import logging.handlers
import os
import re
import shutil
import threading
from collections import deque
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional

import json

import uvicorn
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import config
from ai_provider import create_chat_completion, is_configured, load_settings, save_settings

# ── Logging setup ────────────────────────────────────────────────────
LOG_DIR = config.BASE_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / "tutor.log"

# In-memory ring buffer for quick access via API
log_buffer: deque[dict] = deque(maxlen=500)


class BufferHandler(logging.Handler):
    """Custom handler that stores log records in an in-memory buffer."""
    def emit(self, record):
        log_buffer.append({
            "timestamp": datetime.fromtimestamp(record.created).strftime("%Y-%m-%d %H:%M:%S"),
            "level": record.levelname,
            "module": record.module,
            "message": record.getMessage(),
        })


# Configure tutor logger (propagate=False to avoid uvicorn duplication)
logger = logging.getLogger("tutor")
logger.setLevel(logging.INFO)
logger.propagate = False

# File handler (rotating, max 5MB, keep 3 backups)
file_handler = logging.handlers.RotatingFileHandler(
    LOG_FILE, maxBytes=5_000_000, backupCount=3, encoding="utf-8"
)
file_handler.setFormatter(logging.Formatter(
    "%(asctime)s | %(levelname)-7s | %(module)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
))
logger.addHandler(file_handler)

# In-memory buffer handler
logger.addHandler(BufferHandler())

# Console handler
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter(
    "%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
))
logger.addHandler(console_handler)
from rag.chain import rag_query
from rag.chunker import chunk_text
from rag.document_loader import load_document
from rag.vector_store import (
    add_document,
    delete_document,
    list_documents,
    preload_model,
    reset_collection,
)
from tutor.feedback import generate_feedback
from tutor.scenarios import (
    generate_dynamic_scenario,
    get_scenario_by_id,
)
from tutor.session import SessionManager


# ── Conversation persistence ────────────────────────────────────────

def _save_conversation(conv_type: str, user_id: str, messages: list, metadata: dict = None):
    """Save a conversation (chat or session) to a JSON file."""
    import uuid as _uuid
    conv_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + _uuid.uuid4().hex[:6]
    filename = f"{conv_type}_{conv_id}.json"
    data = {
        "id": conv_id,
        "type": conv_type,  # "chat" or "session"
        "user_id": user_id,
        "timestamp": datetime.now().isoformat(),
        "messages": messages,
        "metadata": metadata or {},
    }
    filepath = config.CONVERSATIONS_DIR / filename
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    logger.info(f"Gesprek opgeslagen: {filename} ({len(messages)} berichten)")
    return conv_id


def _load_conversations():
    """Load all saved conversations (metadata only)."""
    conversations = []
    for filepath in sorted(config.CONVERSATIONS_DIR.glob("*.json"), reverse=True):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            conversations.append({
                "id": data.get("id", filepath.stem),
                "type": data.get("type", "unknown"),
                "user_id": data.get("user_id", ""),
                "timestamp": data.get("timestamp", ""),
                "message_count": len(data.get("messages", [])),
                "metadata": data.get("metadata", {}),
                "filename": filepath.name,
            })
        except Exception as e:
            logger.error(f"Fout bij laden gesprek {filepath.name}: {e}")
    return conversations


def _load_conversation_detail(conv_id: str):
    """Load a single conversation with full messages."""
    for filepath in config.CONVERSATIONS_DIR.glob("*.json"):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            if data.get("id") == conv_id:
                return data
        except Exception:
            continue
    return None


def _start_telegram_bot():
    """Start Telegram bot in a separate thread."""
    if not config.TELEGRAM_BOT_TOKEN:
        logger.info("Geen TELEGRAM_BOT_TOKEN - Telegram bot niet gestart")
        return
    try:
        from bot.telegram_bot import run_bot
        thread = threading.Thread(target=run_bot, daemon=True)
        thread.start()
        logger.info("Telegram bot gestart")
    except Exception as e:
        logger.error(f"Telegram bot starten mislukt: {e}")


@asynccontextmanager
async def lifespan(app):
    port = os.environ.get("PORT", "8000")
    logger.info("=" * 50)
    logger.info("  StudyPilot")
    logger.info("=" * 50)
    logger.info(f"Web UI: http://localhost:{port}")
    logger.info(f"Documenten: {config.DOCUMENTS_DIR}")
    logger.info(f"ChromaDB: {config.CHROMA_DB_DIR}")
    settings = load_settings()
    logger.info(f"AI provider: {settings.get('provider', 'anthropic')}")
    logger.info(f"AI model: {settings.get('model', config.CLAUDE_MODEL)}")
    logger.info(f"Embedding model: {config.EMBEDDING_MODEL}")
    logger.info(f"Logbestand: {LOG_FILE}")
    _start_telegram_bot()
    yield


app = FastAPI(title="StudyPilot", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

session_manager = SessionManager()

# Store chat histories per user (simple in-memory)
chat_histories: dict[str, list[dict]] = {}

# RAG toggle: when False, chat and practice use Claude directly without knowledge base
rag_enabled: bool = True


# ── Request/Response models ──────────────────────────────────────────


class ChatRequest(BaseModel):
    message: str
    user_id: str = "web_user"


class SessionStartRequest(BaseModel):
    scenario_id: str
    user_id: str = "web_user"


class SessionMessageRequest(BaseModel):
    message: str
    user_id: str = "web_user"


class SessionFeedbackRequest(BaseModel):
    user_id: str = "web_user"


class TTSRequest(BaseModel):
    text: str
    voice: str = ""  # "male" of "female", leeg = default
    emotion: str = ""  # emotie voor stemkleur, bijv. "boos", "verdrietig"


class GenerateScenarioRequest(BaseModel):
    topic: Optional[str] = None
    difficulty: str = "basis"


# ── Document endpoints ───────────────────────────────────────────────


@app.post("/api/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload and process a document."""
    allowed = {".pdf", ".docx", ".doc", ".txt", ".md", ".html", ".htm"}
    suffix = Path(file.filename).suffix.lower()
    if suffix not in allowed:
        raise HTTPException(400, f"Niet-ondersteund formaat: {suffix}. Toegestaan: {', '.join(allowed)}")

    # Save file
    dest = config.DOCUMENTS_DIR / file.filename
    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Process
    try:
        text = load_document(dest)
        if not text or not text.strip():
            os.remove(dest)
            raise HTTPException(400, "Kon geen tekst uit het document extraheren.")
        chunks = chunk_text(text)
        num_chunks = add_document(chunks, file.filename)
        logger.info(f"Document geupload: {file.filename} ({len(text)} tekens, {num_chunks} chunks)")
        return {
            "status": "ok",
            "filename": file.filename,
            "chunks": num_chunks,
            "characters": len(text),
        }
    except ValueError as e:
        os.remove(dest)
        raise HTTPException(400, str(e))


@app.get("/api/documents")
async def get_documents():
    """List all uploaded documents."""
    docs = list_documents()
    file_info = []
    for name in docs:
        path = config.DOCUMENTS_DIR / name
        size = path.stat().st_size if path.exists() else 0
        file_info.append({"name": name, "size": size})
    return {"documents": file_info}


@app.delete("/api/documents/{doc_name}")
async def remove_document(doc_name: str):
    """Remove a document from the system."""
    deleted = delete_document(doc_name)
    file_path = config.DOCUMENTS_DIR / doc_name
    if file_path.exists():
        os.remove(file_path)
    return {"status": "ok", "chunks_deleted": deleted}


# ── Re-index endpoint ─────────────────────────────────────────────────


def _reindex_sync():
    """Synchronous reindex (runs in thread to avoid blocking event loop)."""
    doc_dir = config.DOCUMENTS_DIR
    files = [f for f in doc_dir.iterdir() if f.is_file() and f.suffix.lower() in {
        ".pdf", ".docx", ".doc", ".txt", ".md", ".html", ".htm"
    }]

    if not files:
        return {"status": "ok", "message": "Geen documenten gevonden om te herindexeren.", "documents": 0}

    logger.info(f"Herindexering gestart: {len(files)} documenten")

    # Reset the vector store (delete old embeddings)
    reset_collection()

    results = []
    total_chunks = 0
    for file_path in files:
        try:
            text = load_document(file_path)
            if text and text.strip():
                chunks = chunk_text(text)
                num_chunks = add_document(chunks, file_path.name)
                total_chunks += num_chunks
                results.append({"name": file_path.name, "chunks": num_chunks, "status": "ok"})
                logger.info(f"Herindexering: {file_path.name} → {num_chunks} chunks")
            else:
                results.append({"name": file_path.name, "chunks": 0, "status": "leeg"})
        except Exception as e:
            results.append({"name": file_path.name, "chunks": 0, "status": f"fout: {e}"})
            logger.error(f"Herindexering fout: {file_path.name}: {e}")

    logger.info(f"Herindexering voltooid: {len(files)} documenten, {total_chunks} chunks totaal")
    return {
        "status": "ok",
        "documents": len(files),
        "total_chunks": total_chunks,
        "details": results,
    }


@app.post("/api/reindex")
async def reindex_documents():
    """Re-index all documents with current embedding model. Use after model change."""
    result = await asyncio.to_thread(_reindex_sync)
    return result


# ── RAG toggle endpoint ──────────────────────────────────────────────


@app.get("/api/rag-toggle")
async def get_rag_toggle():
    """Get the current RAG toggle state."""
    return {"enabled": rag_enabled}


@app.post("/api/rag-toggle")
async def set_rag_toggle():
    """Toggle RAG on/off. When off, chat and practice use Claude directly."""
    global rag_enabled
    rag_enabled = not rag_enabled
    state = "ingeschakeld" if rag_enabled else "uitgeschakeld"
    logger.info(f"Kennisbank {state}")
    return {"enabled": rag_enabled}


# ── Chat endpoint ────────────────────────────────────────────────────


@app.post("/api/chat")
async def chat(req: ChatRequest):
    """Regular RAG chat about conversation techniques."""
    if not is_configured():
        raise HTTPException(503, "Geen AI API-key geconfigureerd. Ga naar Setup.")
    # Check if user has an active session
    if session_manager.has_active_session(req.user_id):
        raise HTTPException(400, "Je hebt een actieve oefensessie. Beëindig deze eerst met /api/session/feedback.")

    history = chat_histories.get(req.user_id, [])
    mode = "RAG" if rag_enabled else "Direct"
    logger.info(f"Chat [{req.user_id}] ({mode}): {req.message[:80]}...")
    try:
        result = rag_query(req.message, chat_history=history[-10:], use_rag=rag_enabled)
        usage = result.get("usage", {})
        logger.info(
            f"Chat antwoord [{req.user_id}] ({mode}): {result['answer'][:80]}... "
            f"| tokens in={usage.get('input_tokens', '?')} out={usage.get('output_tokens', '?')} "
            f"chunks={usage.get('context_chunks', '?')}"
        )
    except RuntimeError as e:
        logger.error(f"Chat fout [{req.user_id}]: {e}")
        raise HTTPException(503, str(e))

    # Update history
    if req.user_id not in chat_histories:
        chat_histories[req.user_id] = []
    chat_histories[req.user_id].append({"role": "user", "content": req.message})
    chat_histories[req.user_id].append({"role": "assistant", "content": result["answer"]})

    return result


# ── Scenario endpoints ───────────────────────────────────────────────


@app.post("/api/scenarios/generate")
async def create_dynamic_scenario(req: GenerateScenarioRequest):
    """Generate a new scenario from document content."""
    if not is_configured():
        raise HTTPException(503, "Geen AI API-key geconfigureerd. Ga naar Setup.")
    logger.info(f"Scenario genereren: thema='{req.topic or 'geen'}', niveau={req.difficulty}, rag={rag_enabled}")
    scenario = await generate_dynamic_scenario(req.topic, req.difficulty, use_rag=rag_enabled)
    logger.info(f"Scenario gegenereerd: {scenario.name} ({scenario.difficulty})")
    return {"scenario": scenario.to_dict()}


# ── Session endpoints ────────────────────────────────────────────────


@app.post("/api/session/start")
async def start_session(req: SessionStartRequest):
    """Start a practice session with a scenario."""
    if session_manager.has_active_session(req.user_id):
        # Automatically end the previous session so the student can start fresh
        session_manager.end_session(req.user_id)
        logger.info(f"Vorige sessie automatisch beëindigd [{req.user_id}]")

    scenario = get_scenario_by_id(req.scenario_id)
    if not scenario:
        raise HTTPException(404, f"Scenario '{req.scenario_id}' niet gevonden.")

    session = session_manager.start_session(req.user_id, scenario)
    logger.info(f"Sessie gestart [{req.user_id}]: {scenario.name} ({scenario.difficulty})")

    # Generate opening message from tutor in role
    if not is_configured():
        raise HTTPException(503, "Geen AI API-key geconfigureerd. Ga naar Setup.")
    char_name = scenario.character_name or "het personage"
    system = f"""Je bent een acteur die een rol speelt voor een oefengesprek.
Je personage heet: {char_name}

## Je rol:
{scenario.role_instruction}

## Belangrijk:
- Blijf ALTIJD in je rol
- Reageer realistisch en consistent met je karakter
- Communiceer in het Nederlands
- Begin het gesprek vanuit je rol (stel je voor of begin de situatie)
- Markeer ELKE actie, gebaar of toonverandering met het [CONTEXT] prefix op een eigen regel.
  Formaat: [CONTEXT] {char_name} <beschrijving>
  Gebruik dit niet alleen aan het begin, maar ook TUSSENDOOR wanneer het personage iets doet.
  Gebruik NOOIT cursieve tekst (*...*) voor acties — altijd [CONTEXT].
  Na elke [CONTEXT] regel volgt een lege regel en dan de gesproken tekst."""

    try:
        opening = create_chat_completion(
            messages=[{"role": "user", "content": "Start het gesprek vanuit je rol."}],
            system=system,
            max_tokens=config.MAX_TOKENS,
        )
    except RuntimeError as e:
        session_manager.end_session(req.user_id)
        raise HTTPException(503, str(e))
    except Exception as e:
        session_manager.end_session(req.user_id)
        raise HTTPException(503, f"AI-fout: {e}")
    session.add_message("tutor", opening)

    return {
        "session_id": session.session_id,
        "scenario": scenario.to_dict(),
        "opening_message": opening,
    }


@app.post("/api/session/message")
async def session_message(req: SessionMessageRequest):
    """Send a message in an active practice session."""
    session = session_manager.get_session(req.user_id)
    if not session:
        raise HTTPException(400, "Geen actieve oefensessie. Start er eerst een.")

    session.add_message("student", req.message)

    # Get RAG context for better in-character responses (only when RAG is enabled)
    context_hint = ""
    if rag_enabled:
        from rag.vector_store import search as rag_search
        context_items = rag_search(req.message, top_k=3)
        if context_items:
            context_hint = "\n\n[Interne hint - niet benoemen in je antwoord: " + " ".join(
                item["text"][:200] for item in context_items
            ) + "]"

    char_name = session.scenario.character_name or "het personage"
    system = f"""Je bent een acteur die een rol speelt voor een oefengesprek.
Je personage heet: {char_name}

## Je rol:
{session.scenario.role_instruction}

## Belangrijk:
- Blijf ALTIJD in je rol
- Reageer realistisch en consistent met je karakter
- Communiceer in het Nederlands
- Reageer op wat de student zegt vanuit je rol
- Houd je antwoorden beknopt (2-4 zinnen), zoals in een echt gesprek
- Markeer ELKE actie, gebaar of toonverandering met het [CONTEXT] prefix op een eigen regel.
  Formaat: [CONTEXT] {char_name} <beschrijving>
  Gebruik dit niet alleen aan het begin, maar ook TUSSENDOOR wanneer het personage iets doet.
  Gebruik NOOIT cursieve tekst (*...*) voor acties — altijd [CONTEXT].
  Na elke [CONTEXT] regel volgt een lege regel en dan de gesproken tekst.{context_hint}"""

    try:
        tutor_reply = create_chat_completion(
            messages=session.to_claude_messages(),
            system=system,
            max_tokens=512,
        )
    except RuntimeError as e:
        session.messages.pop()
        raise HTTPException(503, str(e))
    except Exception as e:
        session.messages.pop()
        raise HTTPException(503, f"AI-fout: {e}")
    session.add_message("tutor", tutor_reply)
    logger.info(f"Sessie bericht [{req.user_id}]: student={req.message[:50]}... | tutor={tutor_reply[:50]}...")

    return {"reply": tutor_reply, "message_count": len(session.messages)}


@app.post("/api/session/feedback")
async def session_feedback(req: SessionFeedbackRequest):
    """End session and generate detailed feedback."""
    session = session_manager.end_session(req.user_id)
    if not session:
        raise HTTPException(400, "Geen actieve oefensessie om te beëindigen.")

    if len(session.messages) < 2:
        return {
            "feedback": "Er zijn te weinig berichten uitgewisseld voor een zinvolle analyse. Probeer langer te oefenen.",
            "message_count": len(session.messages),
        }

    feedback = await generate_feedback(session, use_rag=rag_enabled)
    session.feedback = feedback
    logger.info(f"Feedback gegenereerd [{req.user_id}]: {len(session.messages)} berichten")

    # Save completed session with feedback
    session_messages = [
        {"role": m.role, "content": m.content} for m in session.messages
    ]
    _save_conversation(
        "session", req.user_id,
        session_messages,
        {
            "scenario_id": session.scenario.id,
            "scenario_name": session.scenario.name,
            "scenario_difficulty": session.scenario.difficulty,
            "feedback": feedback,
        },
    )

    return {
        "feedback": feedback,
        "message_count": len(session.messages),
        "scenario": session.scenario.to_dict(),
    }


# ── TTS endpoint ─────────────────────────────────────────────────────


@app.post("/api/tts")
async def text_to_speech(req: TTSRequest):
    """Convert text to speech using ElevenLabs."""
    el_key = load_settings().get("elevenlabs_api_key", "") or config.ELEVENLABS_API_KEY
    if not el_key:
        raise HTTPException(400, "Geen ElevenLabs API key. Browser TTS wordt gebruikt.")

    from elevenlabs import ElevenLabs, VoiceSettings

    try:
        # Select voice based on gender
        voice_key = req.voice if req.voice in config.ELEVENLABS_VOICES else config.ELEVENLABS_DEFAULT_VOICE
        voice_id = config.ELEVENLABS_VOICES[voice_key]

        # Apply emotion-based voice settings
        emotion_cfg = config.ELEVENLABS_EMOTION_SETTINGS.get(
            req.emotion, config.ELEVENLABS_EMOTION_SETTINGS["neutraal"]
        )
        voice_settings = VoiceSettings(
            stability=emotion_cfg["stability"],
            similarity_boost=emotion_cfg["similarity_boost"],
            style=emotion_cfg["style"],
            use_speaker_boost=emotion_cfg["use_speaker_boost"],
        )

        client = ElevenLabs(api_key=el_key)
        audio_generator = client.text_to_speech.convert(
            voice_id=voice_id,
            model_id=config.ELEVENLABS_MODEL_ID,
            text=req.text,
            voice_settings=voice_settings,
        )

        # Collect audio bytes from generator
        audio_bytes = b"".join(audio_generator)

        return StreamingResponse(
            iter([audio_bytes]),
            media_type="audio/mpeg",
            headers={"Content-Disposition": "inline"},
        )
    except Exception as e:
        error_msg = str(e)
        logger.error(f"TTS fout: {error_msg}")
        if "text_to_speech" in error_msg.lower() and "permission" in error_msg.lower():
            raise HTTPException(
                403,
                "Je ElevenLabs API-key mist de 'text_to_speech' permissie. "
                "Ga naar https://elevenlabs.io/app/settings/api-keys en maak "
                "een nieuwe key aan met de juiste permissies.",
            )
        raise HTTPException(500, f"TTS fout: {error_msg}")


# ── Logs endpoint ────────────────────────────────────────────────────


@app.get("/api/logs")
async def get_logs(
    limit: int = Query(100, ge=1, le=500),
    level: Optional[str] = Query(None),
):
    """Return recent log entries from the in-memory buffer."""
    entries = list(log_buffer)
    if level:
        level_upper = level.upper()
        entries = [e for e in entries if e["level"] == level_upper]
    return {"logs": entries[-limit:], "total": len(entries)}


@app.get("/api/logs/file")
async def get_log_file():
    """Download the full log file."""
    if not LOG_FILE.exists():
        raise HTTPException(404, "Logbestand niet gevonden.")
    return FileResponse(LOG_FILE, filename="tutor.log", media_type="text/plain")


# ── Conversation history endpoints ──────────────────────────────────


@app.get("/api/conversations")
async def get_conversations(conv_type: Optional[str] = Query(None)):
    """List all saved conversations."""
    conversations = _load_conversations()
    if conv_type:
        conversations = [c for c in conversations if c["type"] == conv_type]
    return {"conversations": conversations}


@app.get("/api/conversations/{conv_id}")
async def get_conversation_detail(conv_id: str):
    """Get full conversation detail."""
    data = _load_conversation_detail(conv_id)
    if not data:
        raise HTTPException(404, "Gesprek niet gevonden.")
    return data


@app.get("/api/conversations/{conv_id}/export")
async def export_conversation(conv_id: str, format: str = Query("md")):
    """Export a conversation as MD, DOCX, or PDF."""
    if format not in ("md", "docx", "pdf"):
        raise HTTPException(400, "Ongeldig formaat. Kies 'md', 'docx' of 'pdf'.")

    data = _load_conversation_detail(conv_id)
    if not data:
        raise HTTPException(404, "Gesprek niet gevonden.")

    conv_type = data.get("type", "gesprek")
    meta = data.get("metadata", {})
    title = meta.get("scenario_name") or meta.get("topic") or "Gesprek"
    timestamp = data.get("timestamp", "")
    messages = data.get("messages", [])
    difficulty = meta.get("scenario_difficulty", "")
    feedback = meta.get("feedback", "")

    safe_title = re.sub(r'[^\w\-_ ]', '_', title)[:60].strip()
    base_filename = f"{safe_title}_{conv_id[:8]}"

    if format == "md":
        lines = [f"# {title}", ""]
        if timestamp:
            lines += [f"**Datum:** {timestamp}", ""]
        if conv_type == "session" and difficulty:
            lines += [f"**Niveau:** {difficulty}", ""]
        lines += ["---", ""]
        for msg in messages:
            role_label = "Student" if msg["role"] in ("user", "student") else "Tutor"
            lines += [f"**{role_label}:** {msg['content']}", ""]
        if feedback:
            lines += ["---", "## Feedback", "", feedback]
        content = "\n".join(lines)
        return StreamingResponse(
            iter([content.encode("utf-8")]),
            media_type="text/markdown",
            headers={"Content-Disposition": f'attachment; filename="{base_filename}.md"'},
        )

    elif format == "docx":
        from docx import Document as DocxDocument
        from docx.shared import Pt

        doc = DocxDocument()
        doc.add_heading(title, level=1)
        if timestamp:
            doc.add_paragraph(f"Datum: {timestamp}")
        if conv_type == "session" and difficulty:
            doc.add_paragraph(f"Niveau: {difficulty}")
        doc.add_paragraph("")
        for msg in messages:
            role_label = "Student" if msg["role"] in ("user", "student") else "Tutor"
            p = doc.add_paragraph()
            run_label = p.add_run(f"{role_label}: ")
            run_label.bold = True
            run_label.font.size = Pt(11)
            p.add_run(msg["content"])
        if feedback:
            doc.add_paragraph("")
            doc.add_heading("Feedback", level=2)
            doc.add_paragraph(feedback)

        buf = io.BytesIO()
        doc.save(buf)
        buf.seek(0)
        return StreamingResponse(
            buf,
            media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            headers={"Content-Disposition": f'attachment; filename="{base_filename}.docx"'},
        )

    elif format == "pdf":
        from fpdf import FPDF

        pdf = FPDF()
        pdf.add_page()
        pdf.set_auto_page_break(auto=True, margin=15)

        pdf.set_font("Helvetica", "B", 16)
        pdf.cell(0, 10, title.encode("latin-1", errors="replace").decode("latin-1"), ln=True)
        pdf.set_font("Helvetica", "", 10)
        if timestamp:
            pdf.cell(0, 6, f"Datum: {timestamp}", ln=True)
        if conv_type == "session" and difficulty:
            pdf.cell(0, 6, f"Niveau: {difficulty}", ln=True)
        pdf.ln(4)

        for msg in messages:
            role_label = "Student" if msg["role"] in ("user", "student") else "Tutor"
            pdf.set_font("Helvetica", "B", 10)
            pdf.cell(0, 6, f"{role_label}:", ln=True)
            pdf.set_font("Helvetica", "", 10)
            safe_content = msg["content"].encode("latin-1", errors="replace").decode("latin-1")
            pdf.multi_cell(0, 5, safe_content)
            pdf.ln(2)

        if feedback:
            pdf.ln(4)
            pdf.set_font("Helvetica", "B", 12)
            pdf.cell(0, 8, "Feedback", ln=True)
            pdf.set_font("Helvetica", "", 10)
            safe_fb = feedback.encode("latin-1", errors="replace").decode("latin-1")
            pdf.multi_cell(0, 5, safe_fb)

        pdf_bytes = pdf.output()
        return StreamingResponse(
            iter([bytes(pdf_bytes)]),
            media_type="application/pdf",
            headers={"Content-Disposition": f'attachment; filename="{base_filename}.pdf"'},
        )


@app.delete("/api/conversations/{conv_id}")
async def delete_conversation(conv_id: str):
    """Delete a saved conversation."""
    for filepath in config.CONVERSATIONS_DIR.glob("*.json"):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            if data.get("id") == conv_id:
                os.remove(filepath)
                logger.info(f"Gesprek verwijderd: {filepath.name}")
                return {"status": "ok"}
        except Exception:
            continue
    raise HTTPException(404, "Gesprek niet gevonden.")


@app.post("/api/chat/save")
async def save_chat_history(req: ChatRequest):
    """Manually save the current chat history."""
    history = chat_histories.get(req.user_id, [])
    if not history:
        raise HTTPException(400, "Geen chatgeschiedenis om op te slaan.")
    # Use the first user message as topic
    first_msg = next((m["content"][:80] for m in history if m["role"] == "user"), "Chat")
    conv_id = _save_conversation("chat", req.user_id, history, {"topic": first_msg})
    # Clear the chat history after saving
    chat_histories[req.user_id] = []
    return {"status": "ok", "id": conv_id, "message_count": len(history)}


# ── Web interface ────────────────────────────────────────────────────


@app.get("/api/health")
async def health_check():
    """Check if the AI provider is configured."""
    settings = load_settings()
    configured = bool(settings.get("ai_api_key"))
    return {
        "status": "ok" if configured else "no_key",
        "configured": configured,
        "provider": settings.get("provider", "anthropic"),
        "model": settings.get("model", config.CLAUDE_MODEL),
    }


class SettingsRequest(BaseModel):
    provider: str
    model: str
    ai_api_key: str = ""
    elevenlabs_api_key: str = ""


@app.get("/api/settings")
async def get_settings():
    """Return current settings (keys are masked for security)."""
    s = load_settings()
    ai_key = s.get("ai_api_key", "")
    el_key = s.get("elevenlabs_api_key", "")
    masked_ai = (ai_key[:4] + "..." + ai_key[-4:]) if len(ai_key) > 8 else ("***" if ai_key else "")
    masked_el = (el_key[:4] + "..." + el_key[-4:]) if len(el_key) > 8 else ("***" if el_key else "")
    return {
        "provider": s.get("provider", "anthropic"),
        "model": s.get("model", config.CLAUDE_MODEL),
        "configured": bool(ai_key),
        "ai_api_key_set": bool(ai_key),
        "ai_api_key_masked": masked_ai,
        "elevenlabs_api_key_set": bool(el_key),
        "elevenlabs_api_key_masked": masked_el,
    }


@app.post("/api/settings")
async def update_settings(req: SettingsRequest):
    """Save provider, model, and API keys."""
    valid_providers = {"anthropic", "openai"}
    if req.provider not in valid_providers:
        raise HTTPException(400, f"Ongeldige provider: {req.provider}")
    new_settings = {"provider": req.provider, "model": req.model}
    if req.ai_api_key:
        new_settings["ai_api_key"] = req.ai_api_key
    if req.elevenlabs_api_key:
        new_settings["elevenlabs_api_key"] = req.elevenlabs_api_key
    save_settings(new_settings)
    logger.info(f"Instellingen bijgewerkt: provider={req.provider}, model={req.model}")
    return {"status": "ok"}


@app.get("/")
async def serve_web():
    html_path = config.BASE_DIR / "web" / "index.html"
    content = html_path.read_text(encoding="utf-8")
    return HTMLResponse(
        content=content,
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )



# Serve static files (images, etc.) from web/ directory
app.mount("/images", StaticFiles(directory=str(config.BASE_DIR / "web" / "images")), name="images")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
