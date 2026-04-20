"""
LLM client abstraction layer for TalentScout Hiring Assistant.

Supports:
  - Google Gemini   (model prefix: "gemini-")
  - OpenAI          (model prefix: "gpt-")
  - Groq            (model prefix: "groq-" or provider kwarg)
  - Anthropic       (model prefix: "claude-")

All API interactions are routed through this single module so that
swapping providers requires zero changes in business logic.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# ── Provider catalogue ─────────────────────────────────────────────────────
# Maps a friendly provider name → list of supported model IDs
PROVIDER_MODELS: dict[str, list[str]] = {
    "Google Gemini": [
        "gemini-1.5-flash",
        "gemini-1.5-pro",
        "gemini-2.0-flash",
        "gemini-pro",
    ],
    "OpenAI": [
        "gpt-4o",
        "gpt-4o-mini",
        "gpt-4-turbo",
        "gpt-4",
        "gpt-3.5-turbo",
    ],
    "Groq": [
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
        "llama3-70b-8192",
        "llama3-8b-8192",
        "mixtral-8x7b-32768",
        "gemma2-9b-it",
    ],
    "Anthropic": [
        "claude-3-5-sonnet-20241022",
        "claude-3-5-haiku-20241022",
        "claude-3-opus-20240229",
        "claude-3-sonnet-20240229",
        "claude-3-haiku-20240307",
    ],
}

# Reverse map: model_id → provider name
_MODEL_TO_PROVIDER: dict[str, str] = {
    model: provider
    for provider, models in PROVIDER_MODELS.items()
    for model in models
}


def detect_provider(model: str) -> str:
    """
    Infer the provider from the model name.

    Priority:
      1. Exact match in catalogue
      2. Prefix heuristics
    """
    if model in _MODEL_TO_PROVIDER:
        return _MODEL_TO_PROVIDER[model]
    if model.startswith("gemini"):
        return "Google Gemini"
    if model.startswith("gpt"):
        return "OpenAI"
    if model.startswith("claude"):
        return "Anthropic"
    # Treat everything else as Groq (LLaMA / Mixtral / Gemma)
    return "Groq"


# ── Main client class ──────────────────────────────────────────────────────

class LLMClient:
    """
    Unified wrapper around multiple LLM provider APIs.

    Args:
        api_key: The provider's API key.
        model:   Model identifier (see PROVIDER_MODELS above).
    """

    def __init__(self, api_key: str, model: str = "gemini-1.5-flash"):
        self.api_key = api_key
        self.model = model
        self.provider = detect_provider(model)
        self._client = self._init_client()

    # ── Public ──────────────────────────────────────────────────────────────

    def complete(
        self,
        system: str,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> str:
        """
        Send a conversation to the LLM and return the assistant's reply.

        Args:
            system:      System prompt string.
            messages:    List of {'role': ..., 'content': ...} dicts.
            temperature: Sampling temperature.
            max_tokens:  Maximum tokens in the response.

        Returns:
            The assistant's reply as a plain string.
        """
        try:
            dispatch = {
                "Google Gemini": self._gemini_complete,
                "OpenAI":        self._openai_complete,
                "Groq":          self._groq_complete,
                "Anthropic":     self._anthropic_complete,
            }
            fn = dispatch.get(self.provider, self._openai_complete)
            return fn(system, messages, temperature, max_tokens)
        except Exception as e:
            logger.error("LLM call failed [%s / %s]: %s", self.provider, self.model, e)
            return (
                "I'm experiencing a brief technical difficulty. "
                "Could you please repeat that?"
            )

    # ── Provider initialisation ──────────────────────────────────────────────

    def _init_client(self):
        if self.provider == "Google Gemini":
            return self._init_gemini()
        if self.provider == "OpenAI":
            return self._init_openai()
        if self.provider == "Groq":
            return self._init_groq()
        if self.provider == "Anthropic":
            return self._init_anthropic()
        return self._init_openai()  # safe fallback

    def _init_gemini(self):
        try:
            import google.generativeai as genai  # type: ignore
            genai.configure(api_key=self.api_key)
            return genai
        except ImportError as e:
            raise ImportError(
                "google-generativeai is not installed. Run: pip install google-generativeai"
            ) from e

    def _init_openai(self):
        try:
            from openai import OpenAI  # type: ignore
            return OpenAI(api_key=self.api_key)
        except ImportError as e:
            raise ImportError(
                "openai is not installed. Run: pip install openai"
            ) from e

    def _init_groq(self):
        try:
            from groq import Groq  # type: ignore
            return Groq(api_key=self.api_key)
        except ImportError as e:
            raise ImportError(
                "groq is not installed. Run: pip install groq"
            ) from e

    def _init_anthropic(self):
        try:
            import anthropic  # type: ignore
            return anthropic.Anthropic(api_key=self.api_key)
        except ImportError as e:
            raise ImportError(
                "anthropic is not installed. Run: pip install anthropic"
            ) from e

    # ── Provider-specific completions ────────────────────────────────────────

    def _gemini_complete(
        self,
        system: str,
        messages: list[dict],
        temperature: float,
        max_tokens: int,
    ) -> str:
        """Google Gemini via google-generativeai SDK."""
        import google.generativeai as genai  # type: ignore

        model_obj = genai.GenerativeModel(
            model_name=self.model,
            system_instruction=system,
            generation_config=genai.GenerationConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
            ),
        )
        # Convert to Gemini history format (all but last message)
        history = []
        for msg in messages[:-1]:
            role = "model" if msg["role"] == "assistant" else "user"
            history.append({"role": role, "parts": [msg["content"]]})

        chat = model_obj.start_chat(history=history)
        last = messages[-1]["content"] if messages else "Hello"
        response = chat.send_message(last)
        return response.text.strip()

    def _openai_complete(
        self,
        system: str,
        messages: list[dict],
        temperature: float,
        max_tokens: int,
    ) -> str:
        """OpenAI-compatible API (works for OpenAI SDK)."""
        full_messages = [{"role": "system", "content": system}] + messages
        response = self._client.chat.completions.create(
            model=self.model,
            messages=full_messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content.strip()

    def _groq_complete(
        self,
        system: str,
        messages: list[dict],
        temperature: float,
        max_tokens: int,
    ) -> str:
        """
        Groq API — uses the same OpenAI-compatible chat completions interface
        but through the Groq SDK for correct authentication.
        """
        full_messages = [{"role": "system", "content": system}] + messages
        response = self._client.chat.completions.create(
            model=self.model,
            messages=full_messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content.strip()

    def _anthropic_complete(
        self,
        system: str,
        messages: list[dict],
        temperature: float,
        max_tokens: int,
    ) -> str:
        """
        Anthropic Claude API — uses the Messages API which separates
        the system prompt from the user/assistant turns.
        """
        # Anthropic requires alternating user/assistant turns starting with user
        anthropic_msgs = []
        for msg in messages:
            role = "assistant" if msg["role"] == "assistant" else "user"
            anthropic_msgs.append({"role": role, "content": msg["content"]})

        # Ensure conversation starts with a user message
        if not anthropic_msgs or anthropic_msgs[0]["role"] != "user":
            anthropic_msgs.insert(0, {"role": "user", "content": "Hello"})

        response = self._client.messages.create(
            model=self.model,
            system=system,
            messages=anthropic_msgs,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.content[0].text.strip()
