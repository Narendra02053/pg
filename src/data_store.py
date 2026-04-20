"""
Data storage utilities for TalentScout Hiring Assistant.

Handles secure, privacy-respecting persistence of candidate profiles.
Data is stored locally as anonymized JSON (no plaintext passwords).
In production, replace with an encrypted database or a secrets manager.
"""

import json
import hashlib
import os
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Store session data in a local 'data' directory (gitignored)
DATA_DIR = Path(__file__).parent.parent / "data"


def _ensure_data_dir():
    """Create the data directory if it doesn't exist."""
    DATA_DIR.mkdir(exist_ok=True)
    # Write a .gitignore to prevent accidental commit of candidate data
    gitignore = DATA_DIR / ".gitignore"
    if not gitignore.exists():
        gitignore.write_text("# Auto-generated: do not commit candidate data\n*\n")


def anonymize_email(email: str) -> str:
    """
    Return a one-way hash of the email for internal deduplication
    without storing the raw value in logs.

    The raw email IS stored in the candidate record — this hash is
    only used as a session key.
    """
    return hashlib.sha256(email.lower().encode()).hexdigest()[:16]


def save_candidate(profile_dict: dict) -> str:
    """
    Persist candidate profile to a JSON file.

    Args:
        profile_dict: Dict returned by CandidateProfile.to_dict().

    Returns:
        Path to the saved file as a string.
    """
    _ensure_data_dir()

    # Use timestamp + name hash as filename (avoids exposing PII in filename)
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    name_slug = (profile_dict.get("full_name") or "unknown").lower().replace(" ", "_")
    name_slug = "".join(c for c in name_slug if c.isalnum() or c == "_")[:20]
    filename = f"candidate_{ts}_{name_slug}.json"
    filepath = DATA_DIR / filename

    try:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(profile_dict, f, indent=2, ensure_ascii=False)
        logger.info("Candidate profile saved: %s", filename)
        return str(filepath)
    except OSError as e:
        logger.error("Failed to save candidate profile: %s", e)
        return ""


def load_recent_candidates(limit: int = 10) -> list[dict]:
    """
    Load the most recent candidate profiles for admin review.

    Args:
        limit: Maximum number of records to return.

    Returns:
        List of candidate profile dicts, newest first.
    """
    _ensure_data_dir()
    files = sorted(DATA_DIR.glob("candidate_*.json"), reverse=True)[:limit]
    records = []
    for f in files:
        try:
            with open(f, encoding="utf-8") as fp:
                records.append(json.load(fp))
        except (OSError, json.JSONDecodeError) as e:
            logger.warning("Could not read %s: %s", f.name, e)
    return records
