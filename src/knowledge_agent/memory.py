"""
Persistent session memory for the knowledge agent.

Stores the last N questions asked and a fact ledger (key facts the agent has confirmed across sessions).
Loaded at agent startup, saved after every turn.

Why JSON and not a DB: the state is tiny (< 1KB), human-readable,
and needs no query interface.

Run: imported by main.py - no standalone execution.
"""

import json
from pathlib import Path
from src.knowledge_agent.config import STATE_FILE

MAX_HISTORY = 10 # keep last N questions

def load_state() -> dict:
    """
    Load agent state from disk. Returns empty state if file doesn't exist.

    State schema:
        {
            "history": ["question1", "question2", ...],  # last MAX_HISTORY questions
            "fact_ledger": {"claim": "source chunk ref", ...}  # verified facts
        }

    Vulnerability: if STATE_FILE is corrupted (partial write), json.loads raises.
    Guard: catch JSONDecodeError and return empty state rather than crashing.
    """
    path = Path(STATE_FILE)
    if not path.exists():
        return {"history": [], "fact_ledger": {}}
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return {"history": [], "fact_ledger": {}}


def save_state(state: dict) -> None:
    """
    Write state to disk automatically.

    Vulnerability: direct write can corrupt if process is killed mid-write.
    For this scale (< 1KB) the risk is acceptable. Production would use
    write-to-temp + atomic rename.
    """
    Path(STATE_FILE).write_text(json.dumps(state, indent=2))


def record_question(state: dict, question: str) -> dict:
    """
    Append question to history, trim to MAX_HISTORY.
    Returns updated state (does not save - caller must call save_state).
    """
    history = state.get("history", [])
    history.append(question)
    state["history"] = history[-MAX_HISTORY:]
    return state


def record_fact(state: dict, claim: str, source_ref: str) -> dict:
    """
    Add a verified fact to the ledger.
    source_ref format: "filename: | chunk N" - matches verify_claim output.
    Returns updated state (does not save).
    """
    state.setdefault("fact_ledger", {})[claim] = source_ref
    return state


def format_history_for_prompt(state: dict) -> str:
    """
    Format recent history as a string to inject into the system prompt.
    Returns empty string if no history - caller checks before injecting.
    """
    history = state.get("history", [])
    if not history:
        return ""
    lines = ["Recent questions in this agent's history:"]
    for i, q in enumerate(history[-5:], 1):   # show last 5 only
        lines.append(f"   {i}.  {q}")
    return "\n".join(lines)







