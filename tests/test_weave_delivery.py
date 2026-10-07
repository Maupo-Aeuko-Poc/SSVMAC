"""Exactly-once delivery of a realisation thought, end to end.

A thought has three possible lives: it printed out loud the moment it formed
(announce returned True, pending cleared), it parked while the user was
typing or a reply was streaming (pending kept, line waiting on the board), or
it is being carried into the next reply as a woven suggestion. Every path
must deliver it EXACTLY once - the screenshot bugs were always delivery bugs:
showed twice, or stuck as a bracketed note nobody ever answered.
"""

from __future__ import annotations

import threading

import pytest

from maupo.engine import RealisationEngine
from maupo.memory import HardMemory, SessionManager, SoftMemory
from maupo.notices import NoticeBoard

THOUGHT = "what happened with the porto plan"


@pytest.fixture
def engine(tmp_path):
    sm = SessionManager(tmp_path / "sessions", "test-model")
    hard = HardMemory(tmp_path / "hard.md")
    hard.append("the creator moved to porto once")   # _fire needs real material
    engine = RealisationEngine("test-model", "http://localhost:11434",
                               hard, SoftMemory(tmp_path / "soft.md"), sm)
    return engine


class TestFireDeliverySemantics:
    def test_headless_keeps_pending_for_the_weave(self, engine, capsys, monkeypatch):
        """No board (tests/headless): print IS the channel, and the queue stays
        the taker's channel (pinned by test_fire_queues_question_and_ledgers_it)."""
        engine.notices = None
        monkeypatch.setattr(engine, "_generate_question", lambda m: THOUGHT)
        engine._fire()
        captured = capsys.readouterr()
        assert "[Maupo, thinking: " in captured.out and THOUGHT in captured.out
        assert engine.take_pending() == THOUGHT   # queue still serves the taker
        assert engine.take_pending() == ""

    def test_announce_delivered_clears_pending(self, engine, capsys, monkeypatch):
        board = NoticeBoard()          # nothing mid-use: prints immediately
        engine.notices = board
        monkeypatch.setattr(engine, "_generate_question", lambda m: THOUGHT)
        engine._fire()
        assert THOUGHT in capsys.readouterr().out
        assert engine.take_pending() == ""   # user has SEEN it: no weave repeat

    def test_generate_yield_spends_nothing_and_reschedules(self, engine, monkeypatch):
        """The one GPU belongs to the reply: a fire that lands mid-reply must
        not print, not park, and not cost any generation - it is RESCHEDULED
        (a postponed thought, not an instant burst after the reply)."""
        board = NoticeBoard()
        board.set_generating(True)
        engine.notices = board
        monkeypatch.setattr(engine, "_generate_question",
                            lambda m: pytest.fail("GPU spent during a reply"))
        with engine._lock:
            engine._last_realisation -= engine._next_interval   # make it due
        engine._fire()
        assert engine.take_pending() == ""
        assert board._pending == []
        assert engine.is_due() is False   # rescheduled for a later, quiet beat

    def test_prompt_open_parks_line_and_pending_is_the_weave_channel(self, engine, monkeypatch):
        """While the user types, the line parks on the board AND pending keeps
        the thought. Without the withdraw loop below, that parked line prints
        at the next flush AND the weave repeats it - the double-delivery leak
        the withdraw loop unclogs."""
        board = NoticeBoard()
        board.open_prompt()
        engine.notices = board
        monkeypatch.setattr(engine, "_generate_question", lambda m: THOUGHT)
        engine._fire()
        assert f"[Maupo, thinking: {THOUGHT}]" in board._pending  # parked
        assert engine.take_pending() == THOUGHT                   # weave channel


class TestWithdrawParkedRealisations:
    def test_parked_line_moves_back_to_pending_before_flush(self, engine, monkeypatch):
        """The race the fix exists for: _fire parks while you were typing."""
        board = NoticeBoard()
        board.open_prompt()            # mid-use BY TYPING: parks, not yields
        engine.notices = board
        monkeypatch.setattr(engine, "_generate_question", lambda m: THOUGHT)
        engine._fire()
        board.close_prompt()
        assert f"[Maupo, thinking: {THOUGHT}]" in board._pending
        # Deliverer cleared pending; withdraw must restore it.

        import qwen_chat
        qwen_chat.withdraw_parked_realisations(board, engine)

        assert board._pending == []          # flush will print nothing...
        assert engine.take_pending() == THOUGHT   # ...the weave owns it now

    def test_delivered_while_reading_is_never_requeued(self, engine, monkeypatch, capsys):
        """announce printed-and-cleared racing our read: no resurrection."""
        board = NoticeBoard()
        engine.notices = board
        monkeypatch.setattr(engine, "_generate_question", lambda m: THOUGHT)
        engine._fire()                       # board idle -> printed, pending ""
        assert THOUGHT in capsys.readouterr().out
        assert board._pending == []
        assert engine.take_pending() == ""

        import qwen_chat
        qwen_chat.withdraw_parked_realisations(board, engine)
        assert board._pending == []
        assert engine.take_pending() == ""   # nothing came back

    def test_no_pending_is_a_noop(self, engine):
        import qwen_chat

        board = NoticeBoard()
        board._pending.append("[heartbeat: stuff]")   # other queued notices stay
        with engine._lock:
            engine._pending = THOUGHT
        qwen_chat.withdraw_parked_realisations(board, engine)
        assert board._pending == ["[heartbeat: stuff]"]

    def test_withdraw_of_other_line_never_steals_ours(self, engine):
        import qwen_chat

        board = NoticeBoard()
        board._pending.append("[Maupo, thinking: another thought]")
        with engine._lock:
            engine._pending = THOUGHT
        qwen_chat.withdraw_parked_realisations(board, engine)
        # The other parked line is untouched; our target was not on the board.
        assert board._pending == ["[Maupo, thinking: another thought]"]
        assert engine.take_pending() == THOUGHT


class TestWeavePicksUpPending:
    """The full sequence after a withdraw: park -> withdraw -> take_pending
    -> the runner weaves it instead of printing a bracketed note."""

    def test_weave_picks_up_pending_after_withdraw(self):
        import qwen_chat

        board = NoticeBoard()

        class _Engine:
            _lock = threading.Lock()
            _pending = THOUGHT

            def take_pending(self) -> str:
                with self._lock:
                    pending, self._pending = self._pending, ""
                    return pending

        eng = _Engine()
        board._pending.append(f"[Maupo, thinking: {THOUGHT}]")
        qwen_chat.withdraw_parked_realisations(board, eng)
        pending_realisation = eng.take_pending()
        assert pending_realisation == THOUGHT          # something to weave
        assert f"[Maupo, thinking: {THOUGHT}]" not in board._pending  # nothing to flush


class TestGrowthNoteFiltering:
    def test_compressed_files_are_not_sessions(self, tmp_path):
        """Artifact files (.compressed.md) are not logged sessions."""
        from maupo.engine import growth_note

        sm = SessionManager(tmp_path / "sessions", "test-model")
        sm.session_file.write_text("", encoding="utf-8")
        (tmp_path / "sessions" / "session_20260101_000000.md").write_text("", encoding="utf-8")
        (tmp_path / "sessions" / "session_20260102_000000.md.compressed.md").write_text("", encoding="utf-8")

        note = growth_note(sm, HardMemory(tmp_path / "hard.md"))
        assert "across 2 logged sessions" in note, note

    def test_empty_hard_memory_is_zero_lessons(self, tmp_path):
        """An empty hard file parses to header-boilerplate, not one lesson."""
        from maupo.engine import growth_note

        sm = SessionManager(tmp_path / "sessions", "test-model")
        sm.session_file.write_text("", encoding="utf-8")
        note = growth_note(sm, HardMemory(tmp_path / "hard.md"))
        assert "carrying 0 permanent memories" in note, note
