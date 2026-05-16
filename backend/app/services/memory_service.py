"""
Memory Service
--------------
Session-isolated conversation memory with a sliding window.

Changes vs original:
  • Memory is keyed by session_id (dict of lists) — no more shared global state
    across concurrent users.
  • get_conversation_history() returns only the last WINDOW_TURNS turns so the
    LLM context window doesn't fill with early conversation.
  • A "default" session key is used when callers don't pass a session_id,
    preserving backward-compatibility with existing call-sites.
"""

import logging

logger = logging.getLogger(__name__)

WINDOW_TURNS = 6   # number of (user + AI) pairs kept in the live window

# session_id → list of {"role": str, "content": str}
_sessions: dict[str, list[dict]] = {}


def _get_session(session_id: str) -> list[dict]:
    return _sessions.setdefault(session_id, [])


# ── Public API ─────────────────────────────────────────────────────────────────

def add_message(role: str, content: str, session_id: str = "default") -> None:
    _get_session(session_id).append({"role": role, "content": content})


def get_conversation_history(session_id: str = "default") -> str:
    messages = _get_session(session_id)

    # Keep last WINDOW_TURNS * 2 messages (each turn = 1 user + 1 AI)
    window = messages[-(WINDOW_TURNS * 2):]

    return "".join(
        f"{m['role']}: {m['content']}\n"
        for m in window
    )


def clear_memory(session_id: str = "default") -> None:
    """Clear a specific session, or all sessions if session_id is None."""
    if session_id is None:
        _sessions.clear()
        logger.info("All conversation sessions cleared")
    else:
        _sessions.pop(session_id, None)
        logger.info("Session '%s' cleared", session_id)