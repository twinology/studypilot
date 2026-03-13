<p align="center">
  <img src="web/images/logo.png" alt="StudyPilot logo" width="100" height="100" style="border-radius:50%">
</p>

<h1 align="center">StudyPilot</h1>
<p align="center"><em>Minder zoeken, meer oefenen en sneller leren!</em></p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.10%2B-blue" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/license-MIT-green" alt="MIT License">
  <img src="https://img.shields.io/badge/AI-Anthropic%20%7C%20OpenAI-purple" alt="AI Providers">
</p>

---

StudyPilot is een AI-gestuurde studietutor die je helpt bij het leren door middel van:

- **RAG-chat** — stel vragen over je eigen documenten (PDF, Word, TXT, Markdown, HTML)
- **Oefensessies** — oefen gesprekken met AI-gegenereerde scenario's op drie niveaus
- **Toetsen** — multiple choice en open vragen op basis van je kennisbank
- **Spraak** — spreek met je tutor via microfoon en luister naar antwoorden (ElevenLabs / browser TTS)
- **Telegram bot** — oefen ook via Telegram
- **Multi-provider** — kies tussen Anthropic Claude of OpenAI GPT

---

## Inhoudsopgave

1. [Systeemvereisten](#systeemvereisten)
2. [Installatie](#installatie)
   - [Windows](#windows)
   - [Linux (Ubuntu/Debian)](#linux-ubuntudebian)
   - [macOS](#macos)
3. [Configuratie](#configuratie)
4. [Starten](#starten)
5. [Gebruik](#gebruik)
6. [Projectstructuur](#projectstructuur)
7. [Veelgestelde vragen](#veelgestelde-vragen)

---

## Systeemvereisten

| Vereiste | Minimum |
|----------|---------|
| **Python** | 3.10 of hoger |
| **RAM** | 4 GB (8 GB aanbevolen vanwege embedding model) |
| **Schijfruimte** | ~2 GB (voor Python packages + embedding model) |
| **Internet** | Vereist voor AI API-calls en eerste installatie |
| **Browser** | Chrome, Edge, Firefox of Safari (voor de web interface) |

### API Keys (minimaal 1 vereist)

| Service | Verplicht | Doel |
|---------|-----------|------|
| **Anthropic API key** | Ja* | Claude AI voor chat, scenarios, feedback |
| **OpenAI API key** | Ja* | GPT modellen als alternatief voor Claude |
| **ElevenLabs API key** | Nee | Natuurlijke spraaksynthese (optioneel, browser TTS als fallback) |
| **Telegram Bot Token** | Nee | Telegram bot integratie (optioneel) |

\* Je hebt minimaal een Anthropic OF OpenAI key nodig.

---

## Installatie

### Windows

#### 1. Python installeren

Download Python 3.10+ van [python.org](https://www.python.org/downloads/).

> **Belangrijk:** Vink tijdens de installatie **"Add Python to PATH"** aan!

Controleer na installatie:

```powershell
python --version
pip --version
```

#### 2. Repository klonen

```powershell
git clone https://github.com/twinology/studypilot.git
cd studypilot
```

Of download als ZIP via de groene **Code** knop op GitHub en pak het uit.

#### 3. Virtuele omgeving aanmaken (aanbevolen)

```powershell
python -m venv .venv
.venv\Scripts\activate
```

> Je ziet nu `(.venv)` voor je prompt. Dit houdt de packages gescheiden van je systeem-Python.

#### 4. Dependencies installeren

```powershell
pip install -r requirements.txt
```

> **Let op:** Het `sentence-transformers` package downloadt bij eerste gebruik automatisch het embedding model (~1.5 GB). Dit gebeurt eenmalig.

#### 5. Configuratie

```powershell
copy .env.example .env
notepad .env
```

Vul minimaal je AI API key in (zie [Configuratie](#configuratie)).

#### 6. Starten

```powershell
python main.py
```

Open je browser en ga naar: **http://localhost:8000**

---

### Linux (Ubuntu/Debian)

#### 1. Python en pip installeren

```bash
sudo apt update
sudo apt install python3 python3-pip python3-venv git
```

Controleer:

```bash
python3 --version
pip3 --version
```

#### 2. Repository klonen

```bash
git clone https://github.com/twinology/studypilot.git
cd studypilot
```

#### 3. Virtuele omgeving aanmaken (aanbevolen)

```bash
python3 -m venv .venv
source .venv/bin/activate
```

#### 4. Dependencies installeren

```bash
pip install -r requirements.txt
```

> **Opmerking:** Op sommige systemen moet je eerst build-tools installeren:
> ```bash
> sudo apt install build-essential python3-dev
> ```

#### 5. Configuratie

```bash
cp .env.example .env
nano .env
```

Vul minimaal je AI API key in (zie [Configuratie](#configuratie)).

#### 6. Starten

```bash
python main.py
```

Open je browser en ga naar: **http://localhost:8000**

#### Optioneel: als achtergrondservice draaien

```bash
nohup python main.py > studypilot.log 2>&1 &
```

Of met `systemd` (zie [Veelgestelde vragen](#veelgestelde-vragen)).

---

### macOS

#### 1. Python installeren

macOS heeft vaak een oudere Python-versie. Installeer Python 3.10+ via:

**Optie A — Homebrew (aanbevolen):**

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
brew install python@3.12 git
```

**Optie B — python.org installer:**

Download van [python.org](https://www.python.org/downloads/macos/).

Controleer:

```bash
python3 --version
pip3 --version
```

#### 2. Repository klonen

```bash
git clone https://github.com/twinology/studypilot.git
cd studypilot
```

#### 3. Virtuele omgeving aanmaken (aanbevolen)

```bash
python3 -m venv .venv
source .venv/bin/activate
```

#### 4. Dependencies installeren

```bash
pip install -r requirements.txt
```

> **Apple Silicon (M1/M2/M3):** De packages zijn compatibel met ARM64. Als je problemen ondervindt met `chromadb`, installeer dan eerst:
> ```bash
> brew install cmake
> ```

#### 5. Configuratie

```bash
cp .env.example .env
nano .env
```

Vul minimaal je AI API key in (zie [Configuratie](#configuratie)).

#### 6. Starten

```bash
python main.py
```

Open je browser en ga naar: **http://localhost:8000**

---

## Configuratie

### Methode 1: Via de web interface (aanbevolen)

1. Start de applicatie: `python main.py`
2. Open http://localhost:8000
3. Je wordt automatisch naar de **Setup** tab geleid
4. Kies je AI provider (Anthropic of OpenAI)
5. Selecteer een model
6. Vul je API key in en klik op **Opslaan**

Instellingen worden opgeslagen in `settings.json` (lokaal, niet in git).

### Methode 2: Via .env bestand

Bewerk het `.env` bestand in de projectmap:

```env
# Kies een van de twee:
ANTHROPIC_API_KEY=sk-ant-api03-jouw-key-hier
# OF
# OPENAI_API_KEY=sk-jouw-openai-key-hier

# Optioneel:
ELEVENLABS_API_KEY=sk_jouw-elevenlabs-key
TELEGRAM_BOT_TOKEN=123456789:ABCdefGhIjKlMnOpQrStUvWxYz
```

### API keys verkrijgen

| Service | Registreren | Kosten |
|---------|-------------|--------|
| **Anthropic** | [console.anthropic.com](https://console.anthropic.com/) | Betaald per gebruik (~$3/$15 per 1M tokens in/uit) |
| **OpenAI** | [platform.openai.com](https://platform.openai.com/) | Betaald per gebruik (varieert per model) |
| **ElevenLabs** | [elevenlabs.io](https://elevenlabs.io/) | Gratis tier beschikbaar (10.000 tekens/maand) |
| **Telegram** | Chat met [@BotFather](https://t.me/BotFather) op Telegram | Gratis |

### Poort wijzigen

Standaard draait StudyPilot op poort **8000**. Wijzig dit met:

```bash
PORT=3000 python main.py
```

---

## Starten

```bash
# Activeer eerst je virtuele omgeving (indien gebruikt):
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

# Start de applicatie:
python main.py
```

Je ziet in de terminal:

```
╔══════════════════════════════════════════════╗
║         StudyPilot - AI Tutor               ║
╠══════════════════════════════════════════════╣
║  Server:     http://0.0.0.0:8000            ║
║  Provider:   anthropic                      ║
║  Model:      claude-sonnet-4-6              ║
╚══════════════════════════════════════════════╝
```

Open **http://localhost:8000** in je browser.

---

## Gebruik

### 1. Documenten uploaden (Kennisbank)

- Ga naar het tabblad **Kennisbank**
- Sleep bestanden naar het uploadgebied of klik om te selecteren
- Ondersteunde formaten: **PDF, Word (.docx), TXT, Markdown (.md), HTML**
- Documenten worden automatisch gesplitst en geïndexeerd voor RAG

### 2. Chat

- Ga naar het tabblad **Chat**
- Stel vragen over je geüploade documenten
- Zet de **Kennisbank** toggle UIT om direct met het AI-model te chatten

### 3. Oefenen

- Ga naar het tabblad **Oefenen**
- Voer een onderwerp in en kies een moeilijkheidsgraad (basis / gevorderd / expert)
- Klik op **Genereer scenario** — het AI-model maakt een rollenspel-scenario
- Start de oefensessie en oefen het gesprek
- Gebruik de **microfoon** om te spreken of typ je antwoorden
- Klik op **Stop & Feedback** voor een uitgebreide beoordeling

### 4. Toetsen (via Telegram)

- Configureer je Telegram Bot Token
- Open je bot in Telegram
- Gebruik `/mctoets [onderwerp]` voor multiple choice
- Gebruik `/opentoets [onderwerp]` voor open vragen

### 5. Gesprekken terugkijken

- Ga naar het tabblad **Gesprekken**
- Bekijk eerdere chats en oefensessies
- Exporteer als **PDF**, **Word (.docx)** of **Markdown (.md)**

---

## Projectstructuur

```
studypilot/
├── main.py                 # FastAPI server (entry point)
├── config.py               # Configuratie en constanten
├── ai_provider.py          # AI provider abstractie (Anthropic/OpenAI)
├── reindex.py              # CLI tool voor herindexering
├── requirements.txt        # Python dependencies
├── .env.example            # Voorbeeld configuratie
├── .gitignore
│
├── web/
│   ├── index.html          # Volledige web interface (single-page app)
│   └── images/
│       └── logo.png        # StudyPilot logo
│
├── rag/                    # RAG (Retrieval-Augmented Generation)
│   ├── chain.py            # RAG query pipeline
│   ├── chunker.py          # Document chunking
│   ├── document_loader.py  # PDF/DOCX/TXT/MD/HTML parsers
│   └── vector_store.py     # ChromaDB vector store
│
├── tutor/                  # Tutor logica
│   ├── scenarios.py        # Scenario generatie en opslag
│   ├── session.py          # Sessie management
│   └── feedback.py         # Feedback generatie
│
└── bot/                    # Telegram bot
    └── telegram_bot.py     # Bot commando's en handlers
```

### Automatisch aangemaakte mappen (niet in git)

```
├── documents/              # Geüploade documenten
├── chroma_db/              # ChromaDB embeddings database
├── conversations/          # Opgeslagen gesprekken (JSON)
├── logs/                   # Logbestanden
└── settings.json           # Gebruikersinstellingen (API keys)
```

---

## Veelgestelde vragen

### Het embedding model downloaden duurt lang

Bij de eerste start downloadt `sentence-transformers` het model `intfloat/multilingual-e5-large` (~1.5 GB). Dit is eenmalig. Zorg voor een stabiele internetverbinding.

### Ik krijg "Geen API-key geconfigureerd"

Ga naar de **Setup** tab in de web interface en vul je Anthropic of OpenAI API key in. Of stel deze in via het `.env` bestand.

### ChromaDB geeft een foutmelding op Linux

Installeer de benodigde build-tools:

```bash
sudo apt install build-essential python3-dev
```

### De spraakfunctie werkt niet

- **ElevenLabs**: controleer je API key en of je credits hebt
- **Browser TTS (fallback)**: werkt automatisch als ElevenLabs niet beschikbaar is
- Spraakherkenning (microfoon) vereist **Chrome** of **Edge** en een HTTPS-verbinding (of localhost)

### Kan ik een ander AI-model gebruiken?

Ja! Ga naar **Setup** en kies je provider en model:

| Provider | Beschikbare modellen |
|----------|---------------------|
| Anthropic | Claude Sonnet 4.6, Claude Sonnet 4, Claude Haiku 4, Claude Opus 4 |
| OpenAI | GPT-4o, GPT-4o Mini, GPT-4 Turbo, GPT-3.5 Turbo |

### Hoe draai ik StudyPilot als achtergrondservice (Linux)?

Maak een systemd service aan:

```bash
sudo nano /etc/systemd/system/studypilot.service
```

```ini
[Unit]
Description=StudyPilot AI Tutor
After=network.target

[Service]
Type=simple
User=jouw-gebruiker
WorkingDirectory=/pad/naar/studypilot
Environment=PATH=/pad/naar/studypilot/.venv/bin
ExecStart=/pad/naar/studypilot/.venv/bin/python main.py
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable studypilot
sudo systemctl start studypilot
```

### Hoe update ik naar een nieuwere versie?

```bash
cd studypilot
git pull
pip install -r requirements.txt
# Herstart de applicatie
```

---

## Licentie

MIT License — zie [LICENSE](LICENSE) voor details.

---

<p align="center">
  <strong>StudyPilot</strong> — Gebouwd met ❤️ door <a href="https://twinology.ai">twinology.ai</a>
</p>
