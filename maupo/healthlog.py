"""Shared self-diagnosis log: one place Maupo records anything that goes wrong.

memory/ is sacred ground (architecture invariant 5), so failures live in
memory/health.log — right next to the heartbeat's own self-check lines. They
never interrupt the terminal, and /vitals-style inspection can read them.

Everywhere that used to swallow an exception silently now reports here, so a
dying background thread leaves a trace instead of a mystery.
"""

from __future__ import annotations

import threading
from datetime import datetime
from pathlib import Path
from typing import Optional

_BASE_DIR = Path(__file__).resolve().parent.parent
_LOG_PATH = _BASE_DIR / "memory" / "health.log"
_log_lock = threading.Lock()
_hook: Optional[callable] = None  # tests / embedding apps can capture lines

# A lifeform that never stops logging must never outgrow its folder: the log
# is trimmed to its most recent MAX_LINES on write. Small, bounded, forever.
MAX_LINES = 2000


def set_log_path(path: Path) -> None:
    """Point the health log somewhere else (used by tests)."""
    global _LOG_PATH
    with _log_lock:
        _LOG_PATH = Path(path)


def set_hook(fn: Optional[callable]) -> None:
    """Register an extra listener for every health line (tests use this)."""
    global _hook
    with _log_lock:
        _hook = fn


def log_health(message: str) -> None:
    """Append a timestamped line to memory/health.log. Never raises."""
    line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}"
    try:
        with _log_lock:
            _LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
            with _LOG_PATH.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
            _trim_if_needed()
    except Exception:
        pass  # the logger itself must never take the organism down
    hook = _hook
    if hook is not None:
        try:
            hook(line)
        except Exception:
            pass


def _trim_if_needed() -> None:
    """Keep the log bounded (runs under _log_lock, so single-threaded here).

    A cheap size gate first: steady-state writes stay O(1); the full
    read-and-compact only happens once the file could plausibly be over
    budget (a line is far smaller than 128 bytes of slack).
    """
    try:
        if _LOG_PATH.stat().st_size <= MAX_LINES * 128:
            return
        with _LOG_PATH.open("r", encoding="utf-8") as f:
            lines = f.readlines()
        if len(lines) > MAX_LINES:
            with _LOG_PATH.open("w", encoding="utf-8") as f:
                f.writelines(lines[-MAX_LINES:])
    except Exception:
        pass  # trimming is hygiene, never a failure path


def recent_lines(count: int = 5) -> list[str]:
    """The last few self-diagnosis lines, oldest first (for /vitals)."""
    try:
        with _log_lock:
            lines = _LOG_PATH.read_text(encoding="utf-8").splitlines()
        return [l for l in lines if l.strip()][-count:]
    except Exception:
        return []
