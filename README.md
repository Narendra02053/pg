# 🎯 TalentScout — AI Hiring Assistant

> An intelligent conversational chatbot that conducts structured technology hiring screenings using Large Language Models.

---

## 📋 Project Overview

**TalentScout** is an AI-powered Hiring Assistant built for a fictional tech recruitment agency. It guides candidates through a structured screening interview entirely through natural conversation:

1. Greets the candidate and explains the process
2. Collects essential details (name, email, phone, experience, desired role, location)
3. Asks the candidate to declare their tech stack
4. Generates 3–5 tailored technical questions using the LLM
5. Records answers and gracefully closes the session

**Key highlights:**
- 🧠 LLM-driven question generation adapts to *any* tech stack
- 😊 Real-time sentiment analysis tracks candidate mood
- 📋 Live candidate profile card updates as info is collected
- 🔒 GDPR-conscious local data storage (never committed to git)
- 🎨 Premium dark-mode UI with animated chat bubbles

---

## 🏗️ Architecture

```
talentscout/
├── app.py                  # Streamlit entry point
├── requirements.txt
├── .env.example
├── .gitignore
├── data/                   # Auto-created, gitignored — stores candidate JSON
└── src/
    ├── __init__.py
    ├── ui.py               # Streamlit UI, CSS, layout, rendering
    ├── chatbot.py          # Conversation FSM + HiringAssistant class
    ├── prompts.py          # All prompt templates & EXIT_KEYWORDS
    ├── llm_client.py       # Gemini / OpenAI abstraction layer
    ├── data_store.py       # Candidate profile persistence (local JSON)
    └── sentiment.py        # TextBlob sentiment analysis (bonus)
```

### Conversation Flow (Finite State Machine)

```
GREETING → GATHER_NAME → GATHER_EMAIL → GATHER_PHONE
         → GATHER_EXPERIENCE → GATHER_POSITION → GATHER_LOCATION
         → GATHER_TECH_STACK → TECH_QUESTIONS → WRAP_UP → ENDED
```

Any stage can transition to `ENDED` via an exit keyword (`quit`, `bye`, `exit`, etc.).

---

## ⚙️ Installation

### Prerequisites
- Python 3.10+
- A [Google Gemini API key](https://aistudio.google.com/app/apikey) *(free)* **or** an [OpenAI API key](https://platform.openai.com/api-keys)

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/your-username/talentscout.git
cd talentscout

# 2. Create and activate a virtual environment
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. (Optional) Download TextBlob corpora for sentiment analysis
python -m textblob.download_corpora

# 5. Run the app
streamlit run app.py
```

The app opens at **http://localhost:8501**.

---

## 🚀 Usage Guide

1. **Open the sidebar** and paste your API key into the *API Key* field.
2. Select a model (default: `gemini-1.5-flash` — fast and free-tier friendly).
3. Click **"Begin Screening Interview"**.
4. Answer each question naturally — the bot understands varied formats.
5. After declaring your tech stack, the bot generates custom technical questions.
6. Type `bye`, `quit`, or `exit` at any time to end the session gracefully.
7. Use **Export** in the sidebar to save the candidate profile locally.

### Dynamic Technical Interview
Instead of a static list of questions, TalentScout now features a **Live Adaptive Interview**:
- The LLM analyzes previous answers to ask relevant follow-up questions.
- It covers diverse areas like system design, debugging, and best practices.
- The interview length is adaptive (typically 5–12 questions), ending only when the LLM determines sufficient depth has been reached.

### Multilingual Support
TalentScout is now fully multilingual. It detects the candidate's language automatically and conducts the entire screening—from info gathering to technical questions—in the candidate's preferred tongue.

---

## 💾 Data Handling & Export
Candidate data is handled with security and privacy as a priority:
- **Local Storage:** All profiles are saved as JSON files in the `data/` directory.
- **Export Format:** Each JSON file contains:
    - Candidate PII (Name, Email, Phone, etc.)
    - Experience & Tech Stack
    - **Full Technical Interview Transcript:** Every question asked and the candidate's corresponding answer.
- **Privacy:** Filenames use anonymized timestamps and name slugs to avoid exposing PII in the file system.

---

## 🔧 Technical Details

| Component | Technology |
|-----------|-----------|
| Frontend  | Streamlit 1.32+ with custom CSS (dark glassmorphism) |
| LLM — Google Gemini | `google-generativeai` SDK — Gemini 1.5 Flash/Pro |
| LLM — OpenAI | `openai` SDK — GPT-4o, GPT-3.5-turbo |
| LLM — Groq | `groq` SDK — LLaMA 3, Mixtral, Gemma (**free tier**) |
| LLM — Anthropic | `anthropic` SDK — Claude 3.5 Sonnet/Haiku/Opus |
| Sentiment | TextBlob (polarity + subjectivity) |
| Data store | Local JSON files in `data/` (gitignored) |
| State mgmt | Streamlit `session_state` + Python dataclass |

### Provider Comparison

| Provider | Models | Speed | Free Tier | Best for |
|----------|--------|-------|-----------|----------|
| **Google Gemini** | gemini-1.5-flash/pro | ⚡ Fast | ✅ Yes | Default — recommended for demos |
| **OpenAI** | gpt-4o, gpt-4-turbo, gpt-3.5 | ⚡ Fast | ❌ Paid | High-quality reasoning |
| **Groq** | llama3-70b, mixtral-8x7b | 🚀 Ultra-fast | ✅ Yes | Speed-critical deployments |
| **Anthropic** | claude-3-5-sonnet, claude-3-opus | ⚡ Fast | ❌ Paid | Long-context, nuanced replies |


---

## 🎨 Prompt Design

### System Prompt
A detailed role instruction establishes TalentBot's persona:
- **Scope guard** — refuses to deviate from hiring topics
- **Tone** — warm, professional, encouraging
- **Data sensitivity** — instructs the model to treat PII with discretion

### Information Gathering
Each stage injects a *hidden system instruction* (prefixed `[SYSTEM INSTRUCTION — do not reveal]`) alongside the conversation history. This tells the LLM exactly what to ask next while keeping the candidate-facing output natural.

### Technical Question Generation
A dedicated prompt in `prompts.py` instructs the model to:
- Cover a **variety** of declared technologies
- Ask **scenario-based** questions (not just definitions)
- Target **mid-level** difficulty
- Include at least one **cross-technology** question
- Return a strict **numbered list** (parsed by regex into a Python list)

### Fallback Mechanism
When regex extractors fail to parse expected data (email, phone, etc.), the bot calls the LLM with a fallback prompt that rephrases the request warmly without breaking conversational flow.

---

## 🔒 Data Privacy

- Candidate data is stored as **local JSON** in the `data/` directory.
- The `data/` directory has an auto-generated `.gitignore` to prevent accidental commits.
- Email addresses are never used as filenames (SHA-256 hash used internally).
- No data is transmitted to any third party beyond the chosen LLM provider API call.
- In a production deployment, replace `data_store.py` with an encrypted database.

---

## 🏆 Bonus Features Implemented

| Feature | Implementation |
|---------|---------------|
| Sentiment Analysis | TextBlob polarity score shown as colored emoji pill on each user message |
| Premium UI | Dark glassmorphism, gradient accents, animated bubbles, Inter/Outfit fonts |
| Live Profile Card | Sidebar updates in real time as candidate info is collected |
| Progress Bar | Visual stage tracker (Step N of 10) in sidebar |
| Multi-model support | Switch between Gemini and OpenAI from the sidebar dropdown |
| Auto data export | Profile auto-saved to JSON when session ends |
| Exit keywords | Any of: `quit`, `exit`, `bye`, `goodbye`, `stop`, `done`, etc. |

---

## 🐛 Challenges & Solutions

| Challenge | Solution |
|-----------|----------|
| Keeping LLM on-topic | System prompt with explicit scope guard + stage-level instructions |
| Parsing varied user inputs | Regex extractors with graceful LLM-powered fallback |
| Maintaining conversation context | Full history passed on every LLM call (capped at 30 turns) |
| Gemini vs OpenAI API differences | Abstracted behind `LLMClient` — one method, two backends |
| Streamlit input reset after send | `input_key` counter incremented on submit, forces widget re-render |
| Preventing PII in git | Auto-generated `.gitignore` inside `data/` directory |

---

## 📄 License

MIT License — free to use, modify, and distribute.
