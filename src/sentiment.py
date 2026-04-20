"""
Lightweight sentiment analysis module for TalentScout.

Uses TextBlob (no heavy model download required) to infer candidate
sentiment from their messages and surface emotional cues to the UI.
"""

import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class SentimentResult:
    """Holds the result of a sentiment analysis pass."""
    label: str          # "positive", "neutral", "negative"
    polarity: float     # -1.0 (very negative) to +1.0 (very positive)
    subjectivity: float # 0.0 (objective) to 1.0 (subjective)
    emoji: str          # A representative emoji for the UI
    color: str          # CSS hex color for visual indicator


_AVAILABLE = False

try:
    from textblob import TextBlob  # type: ignore
    _AVAILABLE = True
except ImportError:
    logger.warning(
        "TextBlob not installed — sentiment analysis disabled. "
        "Install with: pip install textblob"
    )


def analyze(text: str) -> Optional[SentimentResult]:
    """
    Analyze the sentiment of a candidate's message.

    Args:
        text: The raw candidate message.

    Returns:
        SentimentResult if TextBlob is available, else None.
    """
    if not _AVAILABLE or not text.strip():
        return None

    try:
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity

        if polarity >= 0.15:
            label, emoji, color = "positive", "😊", "#4ade80"
        elif polarity <= -0.15:
            label, emoji, color = "negative", "😟", "#f87171"
        else:
            label, emoji, color = "neutral", "😐", "#94a3b8"

        return SentimentResult(
            label=label,
            polarity=round(polarity, 3),
            subjectivity=round(subjectivity, 3),
            emoji=emoji,
            color=color,
        )
    except Exception as e:
        logger.warning("Sentiment analysis failed: %s", e)
        return None


def describe(result: Optional[SentimentResult]) -> str:
    """
    Return a human-readable description of the sentiment result for
    display in the Streamlit sidebar.
    """
    if result is None:
        return "—"
    return (
        f"{result.emoji} {result.label.capitalize()} "
        f"(polarity: {result.polarity:+.2f})"
    )
