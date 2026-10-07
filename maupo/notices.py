"""The single doorway for every background-thread terminal notice."""

from __future__ import annotations

import threading


class NoticeBoard:
    """Single doorway for every background-thread notice.

    Guarantees no thread ever prints through a live 'You: ' prompt or through
    a streaming reply: notices raised while the prompt is open or Maupo is
    generating are parked and flushed at the next safe boundary. The main
    loop owns the open/close and generating flags.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._pending: list[str] = []
        self._prompt_open = False
        self._generating = False

    @property
    def prompt_open(self) -> bool:
        with self._lock:
            return self._prompt_open

    @property
    def generating(self) -> bool:
        with self._lock:
            return self._generating

    def open_prompt(self) -> None:
        with self._lock:
            self._prompt_open = True

    def close_prompt(self) -> None:
        with self._lock:
            self._prompt_open = False

    def set_generating(self, value: bool) -> None:
        with self._lock:
            self._generating = value

    def announce(self, line: str) -> bool:
        """Print a notice now, or park it if the terminal is mid-use.

        Returns True when the line was printed immediately; False when it was
        parked (it will reach the screen at the next flush - callers that
        care about delivery use this, nobody must parse print side effects)."""
        with self._lock:
            if self._prompt_open or self._generating:
                self._pending.append(line)
                return False
        print(f"\n{line}")
        return True

    def withdraw(self, line: str) -> bool:
        """Remove a parked notice before it has flushed. True when it was
        still parked (and is now gone); False when it already reached the
        screen (flush raced us) or was never parked. Callers use this to move
        a delivery to a better channel without ever double-printing."""
        with self._lock:
            if line in self._pending:
                self._pending.remove(line)
                return True
            return False

    def flush(self) -> None:
        """Print parked notices; called by the main loop at safe boundaries."""
        with self._lock:
            pending, self._pending = self._pending, []
        for line in pending:
            print(f"\n{line}")
