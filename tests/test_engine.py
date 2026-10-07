"""Tests for the engine: vitality, wondering, realisations, growth, face bridge."""

import json

import pytest

from maupo.engine import (RealisationEngine, Vitality, WonderingList,
                          growth_note, pick_gesture_for_message,
                          senses_for_context, wake_greeting)
from maupo.memory import HardMemory, SessionManager, SoftMemory


# -------------------------------------------------------------------- greetings
class TestWakeGreeting:
    """The opening line: state-grounded, varied, never a template with blanks."""

    def test_returns_something_nonempty_and_clean(self):
        import pathlib
        vitality = Vitality(pathlib.Path("unused.json"))
        line = wake_greeting(vitality)
        assert line and line.strip() == line
        assert "{" not in line and "}" not in line  # no unfilled templates

    def test_varies_between_launches(self):
        import pathlib
        vitality = Vitality(pathlib.Path("unused.json"))
        lines = {wake_greeting(vitality) for _ in range(30)}
        assert len(lines) > 1, "a friend never greets you with the same sentence every day"

    def test_greetings_never_report_hardware(self):
        """A friend does not open with a machine status readout.

        The greeting pools used to be full of fans/circuits/laptop lines, which
        is why every session opened the same machine-flavoured way.
        """
        import pathlib
        vitality = Vitality(pathlib.Path("unused.json"))
        banned = ("fan", "circuit", "laptop", "machine", "gpu", "system", "code")
        for _ in range(200):
            line = wake_greeting(vitality).lower()
            assert not any(word in line for word in banned), line

    def test_late_night_greeting_feels_late(self):
        import pathlib
        from datetime import datetime
        from unittest import mock
        vitality = Vitality(pathlib.Path("unused.json"))
        with mock.patch("maupo.engine.datetime") as fake_dt:
            fake_dt.now.return_value = datetime(2026, 10, 1, 2, 30)
            fake_dt.side_effect = lambda *a, **k: datetime(*a, **k)
            line = wake_greeting(vitality)
        assert any(w in line for w in ("late", "quieter", "warm")), line


# -------------------------------------------------------------------- vitality
class TestVitality:
    def test_spend_lowers_and_floors_energy(self, tmp_path):
        v = Vitality(tmp_path / "vitality.json")
        v.spend(0.5)
        assert v.current() < 0.5
        v.spend(99.0)
        assert v.current() < 0.06  # never fully drains (floor 0.05)

    def test_state_survives_restart(self, tmp_path):
        path = tmp_path / "vitality.json"
        v = Vitality(path)
        v.spend(0.3)
        reloaded = Vitality(path)
        assert abs(reloaded.current() - v.current()) < 0.01

    def test_deep_idle_sleeps_and_restores(self, tmp_path):
        path = tmp_path / "vitality.json"
        v = Vitality(path)
        v.spend(0.5)
        v._last_activity -= v.SLEEP_AFTER_IDLE + 1  # simulate deep idle
        assert v.maybe_sleep() is True
        assert v.asleep is True
        assert v.current() == 1.0   # sleep fully restores

    def test_activity_wakes_and_reports_transition(self, tmp_path):
        path = tmp_path / "vitality.json"
        v = Vitality(path)
        v._last_activity -= v.SLEEP_AFTER_IDLE + 1
        v.maybe_sleep()
        assert v.note_activity() is True    # this input woke it
        assert v.note_activity() is False   # already awake


# ------------------------------------------------------------------ wondering
class TestWonderingList:
    def test_add_and_open_roundtrip(self, tmp_path):
        w = WonderingList(tmp_path / "wondering.md")
        w.add("why does the fan spin up at night?")
        assert w.open() == ["why does the fan spin up at night?"]

    def test_mark_discussed_closes_the_newest_open(self, tmp_path):
        w = WonderingList(tmp_path / "wondering.md")
        w.add("first curiosity")
        w.add("second curiosity")
        w.mark_discussed()  # the most recent open question is the one closed
        assert w.open() == ["first curiosity"]

    def test_mark_discussed_with_nothing_open_is_safe(self, tmp_path):
        w = WonderingList(tmp_path / "wondering.md")
        w.mark_discussed()  # must not raise
        assert w.open() == []


# ----------------------------------------------------------- realisation engine
@pytest.fixture
def realisation(tmp_path):
    sm = SessionManager(tmp_path / "sessions", "test-model")
    soft = SoftMemory(tmp_path / "soft.md")
    hard = HardMemory(tmp_path / "hard.md")
    return RealisationEngine("test-model", "http://localhost:11434",
                             hard, soft, sm), sm, hard


class TestRealisationEngine:
    def test_dead_ollama_never_produces_a_question(self, realisation, monkeypatch):
        import urllib.error

        import maupo.engine as engine_mod
        engine, _, _ = realisation

        def dead(*a, **k):
            raise urllib.error.URLError("connection refused")

        monkeypatch.setattr(engine_mod, "ollama_chat_once", dead)
        assert engine._generate_question("something my human said: porto") == ""

    def test_state_persists_across_instances(self, tmp_path):
        sm = SessionManager(tmp_path / "sessions", "test-model")
        soft = SoftMemory(tmp_path / "soft.md")
        hard = HardMemory(tmp_path / "hard.md")
        first = RealisationEngine("m", "http://localhost:11434", hard, soft, sm)
        first._last_realisation -= 3600  # an hour of thinking
        first._save_state()
        second = RealisationEngine("m", "http://localhost:11434", hard, soft, sm)
        assert second.is_due() is True

    def test_fallback_question_prefers_hard_memory(self, realisation):
        engine, _, hard = realisation
        hard.append("the creator's exam is on friday")
        question = engine._fallback_question("some notes")
        assert "exam is on friday" in question
        assert question.endswith("?")

    def test_fallback_skips_structural_words(self, realisation):
        engine, _, _ = realisation
        assert engine._fallback_question("memory permanent notes") == ""

    def test_fire_queues_question_and_ledgers_it(self, realisation, monkeypatch, capsys):
        engine, sm, hard = realisation
        hard.append("the creator fears deadlines")
        monkeypatch.setattr(engine, "_generate_question", lambda m: "")
        engine._fire()
        pending = engine.take_pending()
        assert "deadlines" in pending
        assert engine.take_pending() == ""  # taking drains the queue
        # Headless path still delivers out loud via print (the no-board channel).
        assert "deadlines" in capsys.readouterr().out

    def test_fire_respects_generating_gpu(self, realisation, monkeypatch):
        """While the reply owns the GPU, a realisation never prints, parks, or
        costs a model call mid-reply - it stays due for the next idle poll."""
        from maupo.notices import NoticeBoard
        engine, _, hard = realisation
        hard.append("a fact worth wondering about")
        board = NoticeBoard()
        board.set_generating(True)
        engine.notices = board
        monkeypatch.setattr(engine, "_generate_question", lambda m: "q?")
        with engine._lock:
            engine._last_realisation -= engine._next_interval   # fire is due
        engine._fire()
        # Nothing was delivered or parked anywhere:
        assert engine.take_pending() == ""
        assert board._pending == []
        # ...and the fire is rescheduled, not banked (no burst after the reply).
        assert engine.is_due() is False

    def test_own_replies_are_never_memory_food(self, tmp_path):
        """The self-referential question bug.

        Maupo asked itself what it meant by its own greeting, because its own
        turns were collected as material to wonder about and then injected back
        into later replies as an open curiosity.
        """
        sessions = tmp_path / "sessions"
        sm = SessionManager(sessions, "test-model")
        sm.session_file.write_text(
            "# Chat Session: old\n- Started: 2026-01-01 10:00:00\n\n---\n\n"
            "### [10:00:00] Turn 1\n**You**: my sister moved to porto\n\n"
            "**Maupo**: yep you're back nice\n\n",
            encoding="utf-8")
        reopened = SessionManager(sessions, "test-model")
        engine = RealisationEngine("m", "http://localhost:11434",
                                   HardMemory(tmp_path / "h.md"),
                                   SoftMemory(tmp_path / "s.md"), reopened)
        memories = engine._collect_memories()
        assert "porto" in memories
        assert "yep you're back nice" not in memories

    def test_self_referential_question_is_rejected(self, realisation, monkeypatch):
        import maupo.engine as engine_mod
        engine, _, _ = realisation
        monkeypatch.setattr(
            engine_mod, "ollama_chat_once",
            lambda *a, **k: "What did Maupo mean by that greeting?")
        assert engine._generate_question("Something my human said: porto") == ""

    def test_real_question_still_gets_through(self, realisation, monkeypatch):
        import maupo.engine as engine_mod
        engine, _, _ = realisation
        monkeypatch.setattr(engine_mod, "ollama_chat_once",
                            lambda *a, **k: "how did the porto move actually go?")
        assert engine._generate_question("notes") == "how did the porto move actually go?"

    def test_wondering_ledger_receives_fired_question(self, tmp_path, monkeypatch):
        sm = SessionManager(tmp_path / "sessions", "test-model")
        soft = SoftMemory(tmp_path / "soft.md")
        hard = HardMemory(tmp_path / "hard.md")
        w = WonderingList(tmp_path / "wondering.md")
        engine = RealisationEngine("m", "http://localhost:11434", hard, soft, sm, wondering=w)
        hard.append("the creator collects vinyl records")
        monkeypatch.setattr(engine, "_generate_question", lambda m: "")
        engine._fire()
        assert any("vinyl" in q for q in w.open())


# ------------------------------------------------------------------ body senses
class TestBodySenses:
    """The body only reaches the model when it actually has something to say."""

    QUIET = "Your body right now: GPU 45C at 2% load (cool); battery 98% plugged in; it's morning."
    HOT = "Your body right now: GPU 78C at 96% load (hot, the fans are working hard); battery 98% plugged in."
    UNPLUGGED = "Your body right now: GPU 50C at 5% load (cool); battery 21% on battery."

    def test_quiet_body_stays_out_of_the_prompt(self):
        assert senses_for_context(self.QUIET) == ""

    def test_no_senses_is_a_no_op(self):
        assert senses_for_context("") == ""

    def test_notable_body_still_reaches_the_model(self):
        for notable in (self.HOT, self.UNPLUGGED):
            line = senses_for_context(notable)
            assert line and notable in line
            assert "status report" in line  # never a readout, always an aside


# --------------------------------------------------------------------- growth
class TestGrowthNote:
    def test_counts_days_sessions_and_lessons(self, tmp_path):
        sm = SessionManager(tmp_path / "sessions", "test-model")
        hard = HardMemory(tmp_path / "hard.md")
        hard.append("a lesson")
        note = growth_note(sm, hard)
        assert "1 day" in note
        assert "1 logged session" in note
        assert "1 permanent memory" in note


# ---------------------------------------------------------------- face bridge
class TestPickGestureForMessage:
    def test_direct_request_beats_mood_signal(self):
        out = pick_gesture_for_message("ok fine, smile for me", 0.0, 0.0, 0.0, 0.0)
        assert out == "smile"

    def test_mood_signal_fallback(self):
        out = pick_gesture_for_message("lmaooo", 0.0, 0.0, 0.3, 0.3)
        assert out == "laugh"

    def test_plain_message_gives_none(self):
        assert pick_gesture_for_message("what time is it", 0.0, 0.0, 0.0, 0.0) is None
