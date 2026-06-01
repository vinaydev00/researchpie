"""
app/history.py — Persistent chat history stored as JSON on disk.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List

from app.config import CHAT_HISTORY_PATH

logger = logging.getLogger(__name__)


def _ensure_file() -> None:
    CHAT_HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not CHAT_HISTORY_PATH.exists():
        CHAT_HISTORY_PATH.write_text("[]", encoding="utf-8")


def load_history() -> List[dict]:
    try:
        _ensure_file()
        return json.loads(CHAT_HISTORY_PATH.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Could not load chat history: %s", exc)
        return []


def append_message(role: str, content: str) -> None:
    try:
        history = load_history()
        history.append({
            "role": role,
            "content": content,
            "timestamp": datetime.utcnow().isoformat(),
        })
        CHAT_HISTORY_PATH.write_text(
            json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    except Exception as exc:
        logger.warning("Could not save message: %s", exc)


def clear_history() -> None:
    try:
        CHAT_HISTORY_PATH.write_text("[]", encoding="utf-8")
    except Exception as exc:
        logger.warning("Could not clear history: %s", exc)