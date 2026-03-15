import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(override=True)

# ── Version ─────────────────────────────────────────────────────────
VERSION = "2.15.5"
VERSION_NAME = "Multimodal RAG Edition"

BASE_DIR = Path(__file__).parent

# API Keys
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")

# Paths
DOCUMENTS_DIR = BASE_DIR / "documents"
CHROMA_DB_DIR = BASE_DIR / "chroma_db"
CONVERSATIONS_DIR = BASE_DIR / "conversations"

# RAG settings
CHUNK_SIZE = 800        # Grotere chunks = meer samenhangende context per fragment
CHUNK_OVERLAP = 150     # Ruime overlap voor context-continuiteit
EMBEDDING_MODEL = "intfloat/multilingual-e5-large"  # E5-Large meertalig (1024-dim), state-of-the-art voor Nederlands
EMBEDDING_QUERY_PREFIX = "query: "      # E5 modellen vereisen prefix voor zoekopdrachten
EMBEDDING_PASSAGE_PREFIX = "passage: "  # E5 modellen vereisen prefix voor documenten
RETRIEVAL_TOP_K = 25    # Maximale recall: meer chunks ophalen, Claude filtert het relevante

# Multimodal / Image settings
IMAGES_DIR = DOCUMENTS_DIR / "images"
EXTRACT_IMAGES = True           # Afbeeldingen uit PDF/DOCX extraheren en beschrijven
IMAGE_MIN_SIZE = 50             # Minimum pixels (breedte/hoogte) — filtert iconen/bullets
IMAGE_MAX_DIMENSION = 1024      # Maximale dimensie in pixels — resize voor vision API
MAX_IMAGES_IN_CONTEXT = 3       # Maximaal aantal afbeeldingen meegestuurd per query

# STT settings (Speech-to-Text)
LOCAL_WHISPER_MODEL = "large-v3"        # Model size for faster-whisper
LOCAL_WHISPER_DEVICE = "cuda"            # "cpu" or "cuda"
LOCAL_WHISPER_COMPUTE_TYPE = "float16"   # "int8" for CPU, "float16" for GPU
WHISPER_LANGUAGE = "nl"                 # Default language for transcription

# Token stats persistence
TOKEN_STATS_FILE = BASE_DIR / "token_stats.json"

# LLM settings
CLAUDE_MODEL = "claude-sonnet-4-6"
MAX_TOKENS = 4096       # Meer ruimte voor uitgebreide antwoorden met veel context

# ElevenLabs settings
ELEVENLABS_MODEL_ID = "eleven_multilingual_v2"
ELEVENLABS_VOICES = {
    "female": "EXAVITQu4vr4xnSDxMaL",  # Sarah — warm, helder, vrouwelijk
    "male": "cjVigY5qzO86Huf0OWal",     # Eric — rustig, betrouwbaar, mannelijk
}
ELEVENLABS_DEFAULT_VOICE = "female"

# Emotie → ElevenLabs voice settings mapping
# stability: lager = meer emotionele variatie, hoger = stabieler/kalmer
# similarity_boost: hoe dicht bij de originele stem
# style: hoger = expressiever / emotioneler
ELEVENLABS_EMOTION_SETTINGS = {
    "boos":              {"stability": 0.30, "similarity_boost": 0.75, "style": 0.90, "use_speaker_boost": True},
    "gefrustreerd":      {"stability": 0.35, "similarity_boost": 0.75, "style": 0.80, "use_speaker_boost": True},
    "verdrietig":        {"stability": 0.45, "similarity_boost": 0.80, "style": 0.65, "use_speaker_boost": False},
    "angstig":           {"stability": 0.30, "similarity_boost": 0.70, "style": 0.75, "use_speaker_boost": False},
    "nerveus":           {"stability": 0.30, "similarity_boost": 0.70, "style": 0.70, "use_speaker_boost": False},
    "gestrest":          {"stability": 0.35, "similarity_boost": 0.70, "style": 0.75, "use_speaker_boost": True},
    "teleurgesteld":     {"stability": 0.45, "similarity_boost": 0.80, "style": 0.60, "use_speaker_boost": False},
    "passief_agressief": {"stability": 0.50, "similarity_boost": 0.80, "style": 0.70, "use_speaker_boost": True},
    "manipulatief":      {"stability": 0.55, "similarity_boost": 0.80, "style": 0.65, "use_speaker_boost": True},
    "neutraal":          {"stability": 0.60, "similarity_boost": 0.75, "style": 0.40, "use_speaker_boost": True},
    "vriendelijk":       {"stability": 0.55, "similarity_boost": 0.75, "style": 0.50, "use_speaker_boost": True},
}

# Ensure directories exist
DOCUMENTS_DIR.mkdir(exist_ok=True)
CHROMA_DB_DIR.mkdir(exist_ok=True)
CONVERSATIONS_DIR.mkdir(exist_ok=True)
IMAGES_DIR.mkdir(exist_ok=True)
