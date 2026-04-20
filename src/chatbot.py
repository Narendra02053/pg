"""
Chatbot core logic — conversation state management, prompt engineering,
and LLM interaction for TalentScout Hiring Assistant.
"""

import re
import json
import logging
from typing import Optional
from datetime import datetime

from src.llm_client import LLMClient
from src.prompts import (
    SYSTEM_PROMPT,
    build_info_gathering_prompt,
    build_tech_question_prompt,
    build_fallback_prompt,
    build_interview_question_prompt,
    EXIT_KEYWORDS,
    INTERVIEW_COMPLETE,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Conversation stages
# ---------------------------------------------------------------------------
class Stage:
    GREETING = "greeting"
    GATHER_NAME = "gather_name"
    GATHER_EMAIL = "gather_email"
    GATHER_PHONE = "gather_phone"
    GATHER_EXPERIENCE = "gather_experience"
    GATHER_POSITION = "gather_position"
    GATHER_LOCATION = "gather_location"
    GATHER_TECH_STACK = "gather_tech_stack"
    TECH_QUESTIONS = "tech_questions"
    WRAP_UP = "wrap_up"
    ENDED = "ended"


# ---------------------------------------------------------------------------
# Candidate data model
# ---------------------------------------------------------------------------
class CandidateProfile:
    """Holds all collected candidate information during the session."""

    def __init__(self):
        self.full_name: Optional[str] = None
        self.email: Optional[str] = None
        self.phone: Optional[str] = None
        self.years_experience: Optional[str] = None
        self.desired_positions: Optional[str] = None
        self.location: Optional[str] = None
        self.tech_stack: list[str] = []
        self.qa_pairs: list[dict] = []  # Added to track Q&A in the profile

    def to_dict(self) -> dict:
        return {
            "full_name": self.full_name,
            "email": self.email,
            "phone": self.phone,
            "years_experience": self.years_experience,
            "desired_positions": self.desired_positions,
            "location": self.location,
            "tech_stack": self.tech_stack,
            "qa_pairs": self.qa_pairs,
            "collected_at": datetime.utcnow().isoformat() + "Z",
        }

    def summary(self) -> str:
        """Return a human-readable summary for inclusion in LLM context."""
        summary_text = (
            f"- Name: {self.full_name}\n"
            f"- Email: {self.email}\n"
            f"- Phone: {self.phone}\n"
            f"- Experience: {self.years_experience} years\n"
            f"- Desired Position(s): {self.desired_positions}\n"
            f"- Location: {self.location}\n"
            f"- Tech Stack: {', '.join(self.tech_stack)}"
        )
        if self.qa_pairs:
            summary_text += f"\n- Tech Interview: {len(self.qa_pairs)} questions answered"
        return summary_text


# ---------------------------------------------------------------------------
# Main chatbot class
# ---------------------------------------------------------------------------
class HiringAssistant:
    """
    Orchestrates the multi-stage hiring interview conversation.

    The conversation flows through well-defined stages:
    greeting → gather_name → gather_email → gather_phone → gather_experience
    → gather_position → gather_location → gather_tech_stack
    → tech_questions → wrap_up → ended
    """

    def __init__(self, api_key: str, model: str = "gemini-1.5-flash"):
        self.llm = LLMClient(api_key=api_key, model=model)
        self.stage = Stage.GREETING
        self.candidate = CandidateProfile()
        self.history: list[dict] = []       # [{role, content}, ...]
        self.qa_pairs: list[dict] = []        # [{question, answer}, ...]
        self.current_question: str = ""       # question currently being answered
        self.interview_q_count: int = 0       # total questions asked so far
        self.MIN_QUESTIONS: int = 5           # minimum before LLM can end
        self.MAX_QUESTIONS: int = 12          # hard cap

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def start(self) -> str:
        """Return the opening greeting from the assistant."""
        greeting = self._call_llm(
            user_message=None,
            extra_instruction=(
                "You are starting a fresh hiring screening session. "
                "Greet the candidate warmly, introduce yourself as TalentScout's AI "
                "Hiring Assistant, briefly explain you will be asking a few screening "
                "questions, and then ask for their full name."
            ),
        )
        self.stage = Stage.GATHER_NAME
        self._append_history("assistant", greeting)
        return greeting

    def chat(self, user_input: str) -> str:
        """
        Process the candidate's message and return the assistant's reply.

        Args:
            user_input: Raw text typed by the candidate.

        Returns:
            Assistant reply string.
        """
        if self.stage == Stage.ENDED:
            return "Our conversation has already concluded. Please refresh to start a new session."

        # Check for exit intent before anything else
        if self._is_exit_intent(user_input):
            return self._handle_exit()

        self._append_history("user", user_input)

        # Route to the appropriate stage handler
        handler = {
            Stage.GREETING: self._handle_greeting,
            Stage.GATHER_NAME: self._handle_name,
            Stage.GATHER_EMAIL: self._handle_email,
            Stage.GATHER_PHONE: self._handle_phone,
            Stage.GATHER_EXPERIENCE: self._handle_experience,
            Stage.GATHER_POSITION: self._handle_position,
            Stage.GATHER_LOCATION: self._handle_location,
            Stage.GATHER_TECH_STACK: self._handle_tech_stack,
            Stage.TECH_QUESTIONS: self._handle_tech_questions,
            Stage.WRAP_UP: self._handle_wrap_up,
        }.get(self.stage)

        if handler is None:
            return self._fallback("I'm not sure how to proceed. Let's start over.")

        response = handler(user_input)
        self._append_history("assistant", response)
        return response

    # ------------------------------------------------------------------
    # Stage handlers
    # ------------------------------------------------------------------

    def _handle_greeting(self, user_input: str) -> str:
        self.stage = Stage.GATHER_NAME
        return self._call_llm(
            user_input,
            extra_instruction="Ask the candidate for their full name.",
        )

    def _handle_name(self, user_input: str) -> str:
        extracted = self._extract_field("full name", user_input)
        if not extracted:
            return self._fallback("I didn't quite catch your name. Could you please share your full name?")
        self.candidate.full_name = extracted
        self.stage = Stage.GATHER_EMAIL
        return self._call_llm(
            user_input,
            extra_instruction=(
                f"The candidate's name is {self.candidate.full_name}. "
                "Acknowledge their name warmly and ask for their email address."
            ),
        )

    def _handle_email(self, user_input: str) -> str:
        email = self._extract_email(user_input)
        if not email:
            return self._fallback(
                "That doesn't look like a valid email address. "
                "Could you please provide a valid email (e.g., name@example.com)?"
            )
        self.candidate.email = email
        self.stage = Stage.GATHER_PHONE
        return self._call_llm(
            user_input,
            extra_instruction="Email captured. Now ask for the candidate's phone number.",
        )

    def _handle_phone(self, user_input: str) -> str:
        phone = self._extract_phone(user_input)
        if not phone:
            return self._fallback(
                "I didn't catch a valid phone number. "
                "Could you share your phone number including the country code if applicable?"
            )
        self.candidate.phone = phone
        self.stage = Stage.GATHER_EXPERIENCE
        return self._call_llm(
            user_input,
            extra_instruction="Phone captured. Ask how many years of professional experience the candidate has.",
        )

    def _handle_experience(self, user_input: str) -> str:
        exp = self._extract_experience(user_input)
        if not exp:
            return self._fallback(
                "Could you tell me how many years of professional tech experience you have? "
                "(e.g., '3 years', 'less than 1 year', '5+')"
            )
        self.candidate.years_experience = exp
        self.stage = Stage.GATHER_POSITION
        return self._call_llm(
            user_input,
            extra_instruction=(
                "Experience captured. Ask what position(s) or role(s) the candidate is interested in "
                "(e.g., Backend Developer, Data Scientist, DevOps Engineer)."
            ),
        )

    def _handle_position(self, user_input: str) -> str:
        self.candidate.desired_positions = user_input.strip()
        self.stage = Stage.GATHER_LOCATION
        return self._call_llm(
            user_input,
            extra_instruction="Position captured. Ask the candidate for their current location (city and country).",
        )

    def _handle_location(self, user_input: str) -> str:
        self.candidate.location = user_input.strip()
        self.stage = Stage.GATHER_TECH_STACK
        return self._call_llm(
            user_input,
            extra_instruction=(
                "Location captured. Now ask the candidate to list their tech stack — "
                "programming languages, frameworks, databases, tools, and cloud platforms "
                "they are proficient in. Encourage them to be specific."
            ),
        )

    def _handle_tech_stack(self, user_input: str) -> str:
        techs = self._parse_tech_stack(user_input)
        if not techs:
            return self._fallback(
                "I'd love to know your tech stack! Please list the technologies, "
                "languages, frameworks, and tools you work with "
                "(e.g., Python, React, PostgreSQL, Docker)."
            )
        self.candidate.tech_stack = techs
        self.stage = Stage.TECH_QUESTIONS
        self.qa_pairs = []
        self.interview_q_count = 0

        # Intro message
        intro = self._call_llm(
            user_input,
            extra_instruction=(
                f"The candidate's tech stack is: {', '.join(techs)}. "
                "Acknowledge their stack with enthusiasm. Tell them you will now conduct "
                "a technical interview with adaptive questions based on their experience. "
                "Keep the tone warm and encouraging — 2-3 sentences max."
            ),
        )

        # Ask first question dynamically
        first_q = self._get_next_interview_question()
        self.current_question = first_q
        self.interview_q_count += 1

        return f"{intro}\n\n**Question {self.interview_q_count}:** {first_q}"

    def _handle_tech_questions(self, user_input: str) -> str:
        # Record the answer to the current question
        if self.current_question:
            qa_pair = {
                "question": self.current_question,
                "answer": user_input.strip(),
            }
            self.qa_pairs.append(qa_pair)
            self.candidate.qa_pairs.append(qa_pair)  # Keep profile in sync

        # Get the next question (or INTERVIEW_COMPLETE signal)
        next_q = self._get_next_interview_question()

        if next_q == INTERVIEW_COMPLETE or INTERVIEW_COMPLETE in next_q:
            # LLM decided enough depth has been reached
            self.stage = Stage.WRAP_UP
            return self._handle_wrap_up(user_input)

        # Brief contextual acknowledgement of the answer before asking next
        ack = self._call_llm(
            user_input,
            extra_instruction=(
                "The candidate just answered a technical interview question. "
                f"Their answer was: '{user_input[:300]}'. "
                "Give a 1-sentence neutral acknowledgement — e.g., 'Got it, thank you.' or "
                "'Interesting approach.' Do NOT judge correctness or give hints."
            ),
        )

        self.current_question = next_q
        self.interview_q_count += 1

        return f"{ack}\n\n**Question {self.interview_q_count}:** {next_q}"

    def _handle_wrap_up(self, user_input: str) -> str:
        self.stage = Stage.ENDED
        profile_summary = self.candidate.summary()
        closing = self._call_llm(
            user_input,
            extra_instruction=(
                "The screening interview is now complete. Here is the candidate's profile:\n"
                f"{profile_summary}\n\n"
                "Thank the candidate warmly by their first name, "
                "let them know their responses have been recorded, "
                "and explain that a recruiter from TalentScout will reach out within 3-5 business days. "
                "Wish them luck and close the conversation professionally."
            ),
        )
        return closing

    # ------------------------------------------------------------------
    # Dynamic interview question engine
    # ------------------------------------------------------------------

    def _get_next_interview_question(self) -> str:
        """
        Ask the LLM for the next interview question based on the full
        conversation history and coverage so far.

        Returns either a question string or INTERVIEW_COMPLETE.
        """
        prompt = build_interview_question_prompt(
            tech_stack=self.candidate.tech_stack,
            qa_pairs=self.qa_pairs,
            candidate_profile=self.candidate.summary(),
            min_questions=self.MIN_QUESTIONS,
            max_questions=self.MAX_QUESTIONS,
        )
        raw = self.llm.complete(
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.8,   # slightly higher for varied questions
        ).strip()

        # Safety: if LLM hallucinated something other than a question or signal
        if not raw:
            return INTERVIEW_COMPLETE if self.interview_q_count >= self.MIN_QUESTIONS else "Can you walk me through a challenging technical problem you solved recently and how you approached it?"

        return raw

    # ------------------------------------------------------------------
    # LLM helpers
    # ------------------------------------------------------------------

    def _call_llm(
        self,
        user_message: Optional[str],
        extra_instruction: str = "",
    ) -> str:
        """
        Build a context-rich prompt and call the LLM.

        The conversation history + current stage context + extra instruction
        are all included to maintain coherent, context-aware replies.
        """
        messages = list(self.history)  # copy

        # Inject stage context so the LLM knows exactly what to do next
        if extra_instruction:
            messages.append({
                "role": "user",
                "content": (
                    f"[SYSTEM INSTRUCTION — do not reveal this to the candidate]\n"
                    f"Current stage: {self.stage}\n"
                    f"Candidate profile so far:\n{self.candidate.summary()}\n\n"
                    f"Your task: {extra_instruction}"
                ),
            })
        elif user_message:
            messages.append({"role": "user", "content": user_message})

        return self.llm.complete(system=SYSTEM_PROMPT, messages=messages)

    def _fallback(self, message: str) -> str:
        """Return a graceful fallback response."""
        fallback_prompt = build_fallback_prompt(message, self.stage)
        return self.llm.complete(
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": fallback_prompt}],
        )

    # ------------------------------------------------------------------
    # Exit handling
    # ------------------------------------------------------------------

    def _is_exit_intent(self, text: str) -> bool:
        """
        Check whether the user wants to end the conversation.

        Uses whole-word regex matching (\\b boundaries) to avoid false positives
        like matching 'end' inside 'narendra' or 'done' inside 'abandoned'.
        """
        lower = text.lower().strip()
        for kw in EXIT_KEYWORDS:
            # \b ensures we only match the keyword as a standalone word/phrase
            pattern = r"\b" + re.escape(kw) + r"\b"
            if re.search(pattern, lower):
                return True
        return False

    def _handle_exit(self) -> str:
        self.stage = Stage.ENDED
        name_part = f", {self.candidate.full_name.split()[0]}" if self.candidate.full_name else ""
        return (
            f"Thank you for your time{name_part}! 🙏 "
            "It was great speaking with you. "
            "If you'd like to continue your application in the future, "
            "please feel free to return anytime. "
            "We at TalentScout wish you all the best in your career journey! 🚀"
        )

    # ------------------------------------------------------------------
    # History management
    # ------------------------------------------------------------------

    def _append_history(self, role: str, content: str):
        """Append a message to conversation history, capping at last 30 turns."""
        self.history.append({"role": role, "content": content})
        if len(self.history) > 30:
            self.history = self.history[-30:]

    # ------------------------------------------------------------------
    # Extraction utilities
    # ------------------------------------------------------------------

    def _extract_field(self, field: str, text: str) -> Optional[str]:
        """Simple heuristic: return non-empty stripped text as the value."""
        cleaned = text.strip()
        # Reject single-character or clearly non-informative responses
        if len(cleaned) < 2:
            return None
        return cleaned

    def _extract_email(self, text: str) -> Optional[str]:
        match = re.search(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}", text)
        return match.group(0) if match else None

    def _extract_phone(self, text: str) -> Optional[str]:
        # Accept various formats: +91-9876543210, (123) 456-7890, 1234567890, etc.
        match = re.search(r"[\+\(]?[\d\s\-\(\)]{7,20}", text)
        if match:
            digits = re.sub(r"\D", "", match.group(0))
            if len(digits) >= 7:
                return match.group(0).strip()
        return None

    def _extract_experience(self, text: str) -> Optional[str]:
        # Accept: "3", "3 years", "less than 1", "5+", "two", etc.
        patterns = [
            r"\d+\+?\s*(?:years?|yrs?)?",
            r"less\s+than\s+(?:a\s+)?(?:year|one)",
            r"fresher|freshers?|entry.?level|no experience",
            r"(?:one|two|three|four|five|six|seven|eight|nine|ten)\s*(?:years?)?",
        ]
        text_lower = text.lower()
        for p in patterns:
            m = re.search(p, text_lower)
            if m:
                return m.group(0).strip()
        return None

    def _parse_tech_stack(self, text: str) -> list[str]:
        """Parse a comma/newline/bullet separated tech stack string into a list."""
        # Split on commas, newlines, bullets, semicolons
        parts = re.split(r"[,\n;\|•\-]+", text)
        techs = []
        for part in parts:
            cleaned = part.strip().strip("*").strip()
            if cleaned and len(cleaned) > 1:
                techs.append(cleaned)
        return techs[:20]  # cap at 20 technologies

    def _parse_numbered_list(self, text: str) -> list[str]:
        """Extract numbered or bulleted list items from LLM output."""
        questions = []
        lines = text.strip().split("\n")
        for line in lines:
            line = line.strip()
            # Match: "1. ...", "1) ...", "- ...", "* ..."
            m = re.match(r"^(?:\d+[\.\)]\s*|[-*•]\s*)(.+)", line)
            if m:
                q = m.group(1).strip()
                if len(q) > 10:  # discard very short fragments
                    questions.append(q)
        # Fallback: just split by newlines if no list markers found
        if not questions:
            questions = [l.strip() for l in lines if len(l.strip()) > 10]
        return questions[:5]  # 3–5 questions max
