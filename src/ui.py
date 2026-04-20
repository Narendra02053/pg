"""
Streamlit UI for TalentScout Hiring Assistant.
Renders the chat interface, sidebar, and handles session state.
"""

import streamlit as st
import logging
from src.chatbot import HiringAssistant, Stage
from src.data_store import save_candidate
from src import sentiment as sent_module
from src.llm_client import PROVIDER_MODELS

logger = logging.getLogger(__name__)

# ── CSS ──────────────────────────────────────────────────────────────────────
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Outfit:wght@400;600;700&display=swap');

:root {
  --bg-primary:   #0f1117;
  --bg-card:      #1a1d2e;
  --bg-card2:     #16192a;
  --accent:       #6c63ff;
  --accent-light: #8b85ff;
  --accent-glow:  rgba(108,99,255,0.25);
  --gold:         #f5c518;
  --text-primary: #e8eaf0;
  --text-muted:   #8b8fa8;
  --user-bubble:  #6c63ff;
  --bot-bubble:   #1e2235;
  --success:      #4ade80;
  --border:       rgba(108,99,255,0.2);
}

html, body, [class*="css"] {
  font-family: 'Inter', sans-serif;
  background-color: var(--bg-primary) !important;
  color: var(--text-primary) !important;
}

/* Hide default Streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1rem !important; max-width: 900px; }

/* ── Header banner ── */
.ts-header {
  background: linear-gradient(135deg, #1a1d2e 0%, #0f1117 100%);
  border: 1px solid var(--border);
  border-radius: 16px;
  padding: 1.5rem 2rem;
  margin-bottom: 1.5rem;
  display: flex;
  align-items: center;
  gap: 1rem;
  box-shadow: 0 0 40px var(--accent-glow);
}
.ts-header .logo { font-size: 2.2rem; }
.ts-header h1 {
  font-family: 'Outfit', sans-serif;
  font-size: 1.8rem;
  font-weight: 700;
  margin: 0;
  background: linear-gradient(90deg, #6c63ff, #a78bfa);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
}
.ts-header p { color: var(--text-muted); margin: 0; font-size: 0.88rem; }

/* ── Chat container ── */
.chat-wrapper {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: 16px;
  padding: 1.25rem;
  min-height: 440px;
  max-height: 520px;
  overflow-y: auto;
  margin-bottom: 1rem;
  scrollbar-width: thin;
  scrollbar-color: var(--accent) transparent;
}
.chat-wrapper::-webkit-scrollbar { width: 4px; }
.chat-wrapper::-webkit-scrollbar-thumb { background: var(--accent); border-radius: 4px; }

/* ── Message bubbles ── */
.msg-row { display: flex; margin-bottom: 1rem; align-items: flex-end; gap: 0.6rem; }
.msg-row.user  { flex-direction: row-reverse; }
.msg-row.bot   { flex-direction: row; }

.avatar {
  width: 36px; height: 36px; border-radius: 50%;
  display: flex; align-items: center; justify-content: center;
  font-size: 1.1rem; flex-shrink: 0;
}
.avatar.bot-av  { background: linear-gradient(135deg, #6c63ff, #a78bfa); }
.avatar.user-av { background: linear-gradient(135deg, #f5c518, #f97316); }

.bubble {
  max-width: 75%;
  padding: 0.75rem 1rem;
  border-radius: 14px;
  font-size: 0.92rem;
  line-height: 1.6;
  animation: fadeUp 0.25s ease;
}
.bubble.bot  {
  background: var(--bot-bubble);
  border: 1px solid var(--border);
  border-bottom-left-radius: 4px;
  color: var(--text-primary);
}
.bubble.user {
  background: var(--user-bubble);
  border-bottom-right-radius: 4px;
  color: #fff;
}

@keyframes fadeUp {
  from { opacity: 0; transform: translateY(8px); }
  to   { opacity: 1; transform: translateY(0); }
}

/* ── Typing indicator ── */
.typing-dot {
  width: 7px; height: 7px; border-radius: 50%;
  background: var(--accent-light);
  display: inline-block; margin: 0 2px;
  animation: blink 1.2s infinite ease-in-out;
}
.typing-dot:nth-child(2) { animation-delay: 0.2s; }
.typing-dot:nth-child(3) { animation-delay: 0.4s; }
@keyframes blink {
  0%,80%,100% { transform: scale(0.7); opacity: 0.5; }
  40%          { transform: scale(1);   opacity: 1;   }
}

/* ── Input area ── */
.stTextInput input {
  background: var(--bg-card2) !important;
  border: 1px solid var(--border) !important;
  border-radius: 12px !important;
  color: var(--text-primary) !important;
  font-size: 0.92rem !important;
  padding: 0.7rem 1rem !important;
  transition: border-color 0.2s;
}
.stTextInput input:focus {
  border-color: var(--accent) !important;
  box-shadow: 0 0 0 3px var(--accent-glow) !important;
}

/* ── Buttons ── */
.stButton > button {
  background: linear-gradient(135deg, #6c63ff, #a78bfa) !important;
  color: #fff !important;
  border: none !important;
  border-radius: 10px !important;
  font-weight: 600 !important;
  padding: 0.55rem 1.4rem !important;
  transition: transform 0.15s, box-shadow 0.15s !important;
}
.stButton > button:hover {
  transform: translateY(-2px) !important;
  box-shadow: 0 6px 20px var(--accent-glow) !important;
}

/* ── Progress bar ── */
.stProgress > div > div { background: linear-gradient(90deg, #6c63ff, #a78bfa) !important; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
  background: var(--bg-card) !important;
  border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { color: var(--text-primary) !important; }

/* ── Profile card ── */
.profile-card {
  background: var(--bg-card2);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 0.9rem 1rem;
  margin-top: 0.5rem;
  font-size: 0.82rem;
  line-height: 1.7;
}
.profile-card .label { color: var(--text-muted); font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.05em; }
.profile-card .value { color: var(--text-primary); font-weight: 500; }

/* ── Stage badge ── */
.stage-badge {
  display: inline-block;
  background: var(--accent-glow);
  border: 1px solid var(--accent);
  color: var(--accent-light);
  border-radius: 20px;
  padding: 0.2rem 0.75rem;
  font-size: 0.75rem;
  font-weight: 600;
  letter-spacing: 0.04em;
  margin-bottom: 0.75rem;
}

/* ── Sentiment pill ── */
.sent-pill {
  display: inline-flex; align-items: center; gap: 0.35rem;
  background: var(--bg-card2);
  border: 1px solid var(--border);
  border-radius: 20px;
  padding: 0.25rem 0.7rem;
  font-size: 0.78rem;
  margin-top: 0.4rem;
}

/* ── Divider ── */
.ts-divider { border: none; border-top: 1px solid var(--border); margin: 1rem 0; }

/* ── End screen ── */
.end-card {
  background: linear-gradient(135deg, #1a1d2e, #0f1117);
  border: 1px solid var(--accent);
  border-radius: 16px;
  padding: 2rem;
  text-align: center;
  box-shadow: 0 0 60px var(--accent-glow);
}
.end-card h2 { font-family: 'Outfit', sans-serif; font-size: 1.6rem; margin-bottom: 0.5rem; }
.end-card p  { color: var(--text-muted); }
</style>
"""

# ── Stage metadata ─────────────────────────────────────────────────────────
STAGE_LABELS = {
    Stage.GREETING:         ("👋", "Greeting",             1),
    Stage.GATHER_NAME:      ("📝", "Collecting Name",       2),
    Stage.GATHER_EMAIL:     ("📧", "Collecting Email",      3),
    Stage.GATHER_PHONE:     ("📱", "Collecting Phone",      4),
    Stage.GATHER_EXPERIENCE:("💼", "Years of Experience",   5),
    Stage.GATHER_POSITION:  ("🎯", "Desired Role",          6),
    Stage.GATHER_LOCATION:  ("📍", "Location",              7),
    Stage.GATHER_TECH_STACK:("🛠️", "Tech Stack",           8),
    Stage.TECH_QUESTIONS:   ("🧠", "Technical Questions",   9),
    Stage.WRAP_UP:          ("✅", "Wrapping Up",           10),
    Stage.ENDED:            ("🎉", "Complete",              10),
}
TOTAL_STAGES = 10


# ── Session helpers ────────────────────────────────────────────────────────

def _get_api_config() -> tuple[str, str]:
    """Return (api_key, model) from session state."""
    return st.session_state.get("api_key", ""), st.session_state.get("model", "gemini-1.5-flash")


def _init_session():
    """Initialise all session-state keys on first run."""
    defaults = {
        "messages":       [],   # [{role, content, sentiment}]
        "bot":            None,
        "started":        False,
        "api_key":        "",
        "model":          "gemini-1.5-flash",
        "input_key":      0,    # used to reset the text input
        "data_saved":     False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def _start_conversation():
    """Instantiate the bot and fire the opening greeting."""
    api_key, model = _get_api_config()
    bot = HiringAssistant(api_key=api_key, model=model)
    greeting = bot.start()
    st.session_state.bot = bot
    st.session_state.started = True
    st.session_state.messages = [{"role": "assistant", "content": greeting, "sentiment": None}]


def _reset_session():
    for key in ["messages", "bot", "started", "data_saved"]:
        if key in st.session_state:
            del st.session_state[key]
    st.session_state["input_key"] = st.session_state.get("input_key", 0) + 1


# ── Render helpers ─────────────────────────────────────────────────────────

def _render_header():
    st.markdown("""
    <div class="ts-header">
      <span class="logo">🎯</span>
      <div>
        <h1>TalentScout</h1>
        <p>AI-Powered Hiring Assistant · Technology Recruitment Specialists</p>
      </div>
    </div>
    """, unsafe_allow_html=True)


def _render_bubble(role: str, content: str, sentiment=None):
    if role == "assistant":
        avatar_cls, bubble_cls, av_icon = "bot-av", "bot", "🤖"
        row_cls = "bot"
    else:
        avatar_cls, bubble_cls, av_icon = "user-av", "user", "👤"
        row_cls = "user"

    # Escape content for HTML safety, then restore markdown bold/newlines
    import re
    safe = (
        content
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace("\n", "<br>")
    )
    # Correctly replace **bold** with <b>bold</b>
    safe = re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", safe)

    sent_html = ""
    if sentiment and role == "user":
        sent_html = (
            f'<div class="sent-pill" style="border-color:{sentiment.color}">'
            f'{sentiment.emoji} {sentiment.label.capitalize()}'
            f'</div>'
        )

    st.markdown(f"""
    <div class="msg-row {row_cls}">
      <div class="avatar {avatar_cls}">{av_icon}</div>
      <div>
        <div class="bubble {bubble_cls}">{safe}</div>
        {sent_html}
      </div>
    </div>
    """, unsafe_allow_html=True)


def _render_chat_history():
    st.markdown('<div class="chat-wrapper" id="chat-box">', unsafe_allow_html=True)
    for msg in st.session_state.messages:
        _render_bubble(msg["role"], msg["content"], msg.get("sentiment"))
    st.markdown('</div>', unsafe_allow_html=True)


def _render_sidebar():
    with st.sidebar:
        st.markdown("### ⚙️ Configuration")

        # ── Provider selection ──────────────────────────────────────
        provider_names = list(PROVIDER_MODELS.keys())
        prev_provider = st.session_state.get("_prev_provider", "")
        saved_provider = st.session_state.get("provider", "Google Gemini")
        provider_idx = provider_names.index(saved_provider) if saved_provider in provider_names else 0

        provider = st.selectbox(
            "Provider",
            options=provider_names,
            index=provider_idx,
            key="provider",
        )

        # Clear API key when provider changes so stale keys aren't reused
        if provider != prev_provider and prev_provider:
            st.session_state["api_key"] = ""
        st.session_state["_prev_provider"] = provider

        # Dynamic model list for selected provider
        model_options = PROVIDER_MODELS[provider]
        saved_model = st.session_state.get("model", model_options[0])
        model_idx = model_options.index(saved_model) if saved_model in model_options else 0

        st.selectbox(
            "Model",
            options=model_options,
            index=model_idx,
            key="model",
        )
        # Note: st.session_state.model is set automatically by the widget above

        # ── API Key (hint changes per provider) ────────────────────
        _KEY_HINTS = {
            "Google Gemini": ("AIza…",       "Get free key → aistudio.google.com"),
            "OpenAI":        ("sk-…",         "Get key → platform.openai.com"),
            "Groq":          ("gsk_…",        "Free key → console.groq.com"),
            "Anthropic":     ("sk-ant-…",     "Get key → console.anthropic.com"),
        }
        placeholder, hint = _KEY_HINTS.get(provider, ("Paste API key…", ""))

        api_key = st.text_input(
            f"{provider} API Key",
            type="password",
            value=st.session_state.get("api_key", ""),
            placeholder=placeholder,
            help=hint,
            key="api_key_input",
        )
        st.session_state.api_key = api_key

        st.markdown('<hr class="ts-divider">', unsafe_allow_html=True)

        # Progress
        bot: HiringAssistant | None = st.session_state.get("bot")
        if bot:
            stage = bot.stage
            icon, label, step = STAGE_LABELS.get(stage, ("💬", stage, 1))

            # During interview show live question counter
            if stage == Stage.TECH_QUESTIONS:
                q_count = bot.interview_q_count
                min_q   = bot.MIN_QUESTIONS
                max_q   = bot.MAX_QUESTIONS
                st.markdown(
                    f'<div class="stage-badge">🧠 Technical Interview</div>',
                    unsafe_allow_html=True,
                )
                pct = min(q_count / max_q, 1.0)
                st.progress(pct)
                st.caption(
                    f"Question {q_count} asked · min {min_q}, max {max_q}"
                )
            else:
                pct = step / TOTAL_STAGES
                st.markdown(
                    f'<div class="stage-badge">{icon} {label}</div>',
                    unsafe_allow_html=True,
                )
                st.progress(pct)
                st.caption(f"Step {step} of {TOTAL_STAGES}")

            st.markdown('<hr class="ts-divider">', unsafe_allow_html=True)

            # Candidate profile card
            c = bot.candidate
            fields = [
                ("Name",       c.full_name),
                ("Email",      c.email),
                ("Phone",      c.phone),
                ("Experience", c.years_experience),
                ("Position",   c.desired_positions),
                ("Location",   c.location),
                ("Tech Stack", ", ".join(c.tech_stack) if c.tech_stack else None),
            ]
            card_html = '<div class="profile-card">'
            for label, val in fields:
                if val:
                    card_html += (
                        f'<div class="label">{label}</div>'
                        f'<div class="value">{val}</div>'
                    )
            card_html += "</div>"
            if any(v for _, v in fields):
                st.markdown("**📋 Candidate Profile**")
                st.markdown(card_html, unsafe_allow_html=True)

        st.markdown('<hr class="ts-divider">', unsafe_allow_html=True)

        # Control buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Reset", use_container_width=True):
                _reset_session()
                st.rerun()
        with col2:
            if st.button("💾 Export", use_container_width=True):
                bot = st.session_state.get("bot")
                if bot and bot.candidate.full_name:
                    path = save_candidate(bot.candidate.to_dict())
                    if path:
                        st.success("Saved!")
                    else:
                        st.error("Save failed")
                else:
                    st.warning("No data yet")

        st.markdown('<hr class="ts-divider">', unsafe_allow_html=True)
        st.caption("🔒 All data handled per GDPR guidelines.\nNo data sent to third parties.")


def _render_setup_screen():
    """Shown before the user provides an API key and starts."""
    st.markdown("""
    <div style="text-align:center;padding:2rem 0 1rem;">
      <div style="font-size:4rem;margin-bottom:1rem;">🎯</div>
      <h2 style="font-family:'Outfit',sans-serif;font-size:1.8rem;
                 background:linear-gradient(90deg,#6c63ff,#a78bfa);
                 -webkit-background-clip:text;-webkit-text-fill-color:transparent;">
        Welcome to TalentScout
      </h2>
      <p style="color:#8b8fa8;max-width:520px;margin:0 auto 1.5rem;">
        Your AI-powered hiring assistant for technology recruitment.
        Pick a provider in the sidebar, paste your API key, and begin.
      </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Feature cards ──────────────────────────────────────────────
    feat_cols = st.columns(3)
    features = [
        ("📝", "Smart Info Gathering", "Collects all candidate details conversationally"),
        ("🧠", "Tech Question Gen",    "Generates role-specific technical questions"),
        ("😊", "Sentiment Analysis",   "Tracks candidate mood in real time"),
    ]
    for col, (icon, title, desc) in zip(feat_cols, features):
        with col:
            st.markdown(f"""
            <div class="profile-card" style="text-align:center;padding:1.2rem;">
              <div style="font-size:1.8rem;margin-bottom:0.5rem;">{icon}</div>
              <div style="font-weight:600;margin-bottom:0.3rem;">{title}</div>
              <div style="color:#8b8fa8;font-size:0.82rem;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    # ── Supported providers ─────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        "<div style='text-align:center;color:#8b8fa8;font-size:0.8rem;'"
        ">Supported LLM Providers</div>",
        unsafe_allow_html=True,
    )
    prov_cols = st.columns(4)
    providers_info = [
        ("🔵", "Google Gemini",  "Free tier · Fast",         "#4285F4"),
        ("⚫", "OpenAI",         "GPT-4o · GPT-3.5",         "#10a37f"),
        ("🟠", "Groq",           "Free tier · Ultra-fast",   "#f55036"),
        ("🟣", "Anthropic",      "Claude 3.5 Sonnet",        "#c25df7"),
    ]
    for col, (dot, name, tagline, color) in zip(prov_cols, providers_info):
        with col:
            st.markdown(f"""
            <div class="profile-card" style="text-align:center;padding:0.9rem 0.6rem;">
              <div style="font-size:1.4rem;">{dot}</div>
              <div style="font-weight:600;font-size:0.85rem;color:{color};">{name}</div>
              <div style="color:#8b8fa8;font-size:0.75rem;">{tagline}</div>
            </div>
            """, unsafe_allow_html=True)


def _render_end_screen():
    bot: HiringAssistant = st.session_state.bot
    # Auto-save on first end
    if not st.session_state.data_saved and bot.candidate.full_name:
        save_candidate(bot.candidate.to_dict())
        st.session_state.data_saved = True

    name = bot.candidate.full_name or "Candidate"
    first = name.split()[0]
    st.markdown(f"""
    <div class="end-card">
      <div style="font-size:3rem;margin-bottom:0.75rem;">🎉</div>
      <h2>Thank you, {first}!</h2>
      <p>Your screening has been completed and your profile has been saved.</p>
      <p style="margin-top:1rem;font-size:0.85rem;">
        A TalentScout recruiter will review your responses and reach out within
        <strong style="color:#6c63ff;">3–5 business days</strong>.
      </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🔄 Start New Session", use_container_width=True):
        _reset_session()
        st.rerun()


# ── Main render entry point ────────────────────────────────────────────────

def render_ui():
    _init_session()
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    _render_header()
    _render_sidebar()

    api_key = st.session_state.get("api_key", "").strip()

    # ── Setup screen (no API key yet) ──
    if not api_key:
        _render_setup_screen()
        return

    # ── Start button ──
    if not st.session_state.started:
        st.markdown('<div style="text-align:center;padding:1rem 0;">', unsafe_allow_html=True)
        if st.button("🚀 Begin Screening Interview", use_container_width=False):
            with st.spinner("Connecting to TalentBot..."):
                try:
                    _start_conversation()
                except Exception as e:
                    st.error(f"Failed to connect: {e}")
                    return
        st.markdown("</div>", unsafe_allow_html=True)
        _render_setup_screen()
        return

    # ── Ended state ──
    bot: HiringAssistant = st.session_state.bot
    if bot.stage == Stage.ENDED:
        _render_chat_history()
        _render_end_screen()
        return

    # ── Active chat ──
    _render_chat_history()

    # Input row
    col_input, col_send = st.columns([5, 1])
    with col_input:
        user_text = st.text_input(
            label="Your message",
            label_visibility="collapsed",
            placeholder="Type your response here…",
            key=f"user_input_{st.session_state.input_key}",
        )
    with col_send:
        send_clicked = st.button("Send ➤", use_container_width=True)

    if (send_clicked or user_text) and user_text and user_text.strip():
        # Sentiment analysis on user message
        sentiment = sent_module.analyze(user_text)

        # Append user message
        st.session_state.messages.append({
            "role": "user",
            "content": user_text,
            "sentiment": sentiment,
        })

        # Get bot reply
        with st.spinner(""):
            reply = bot.chat(user_text)

        st.session_state.messages.append({
            "role": "assistant",
            "content": reply,
            "sentiment": None,
        })

        # Reset input
        st.session_state.input_key += 1
        st.rerun()
