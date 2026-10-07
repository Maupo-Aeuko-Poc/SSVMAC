"""Tests for memory: sessions, parsing, soft/hard memory, recall, compressor."""

import json

import pytest

from maupo.memory import (HardMemory, MemoryRecall, SessionManager, SoftMemory,
                          UniversalCompressor, parse_session_turns)


@pytest.fixture
def session_manager(tmp_path):
    return SessionManager(tmp_path / "sessions", "test-model")


# --------------------------------------------------------------------- sessions
class TestSessionManager:
    def test_creates_session_file_with_header(self, session_manager):
        content = session_manager.session_file.read_text(encoding="utf-8")
        assert content.startswith("# Chat Session:")
        assert "test-model" in content
        assert "---" in content

    def test_log_turn_appends_and_counts(self, session_manager):
        session_manager.log_turn("hello", "yo")
        session_manager.log_turn("how are you", "fine")
        assert session_manager.turn_count == 2
        content = session_manager.session_file.read_text(encoding="utf-8")
        assert "**You**: hello" in content
        assert "**Maupo**: yo" in content
        assert "Turn 2" in content

    def test_past_sessions_excludes_current(self, tmp_path):
        first = SessionManager(tmp_path / "s", "m")
        first.log_turn("a", "b")
        second = SessionManager(tmp_path / "s", "m")
        past = second.get_past_session_files()
        assert [p.name for p in past] == [first.session_file.name]

    def test_list_all_sessions_reports_turns(self, tmp_path):
        first = SessionManager(tmp_path / "s", "m")
        first.log_turn("one", "reply")
        first.log_turn("two", "reply")
        second = SessionManager(tmp_path / "s", "m")
        sessions = second.list_all_sessions()
        by_name = {s["name"]: s for s in sessions}
        assert by_name[first.session_file.name]["turns"] == 2
        assert by_name[second.session_file.name]["is_current"]


# ---------------------------------------------------------------------- parsing
class TestParseSessionTurns:
    def test_parses_new_maupo_format(self, tmp_path):
        log = tmp_path / "session_x.md"
        log.write_text(
            "# Chat Session\n\n---\n\n"
            "### [10:00:00] Turn 1\n**You**: hey there\n\n**Maupo**: yo\n\n",
            encoding="utf-8")
        turns = parse_session_turns(log)
        assert turns == [("10:00:00", "You", "hey there"), ("10:00:00", "Maupo", "yo")]

    def test_parses_old_qwen_format(self, tmp_path):
        log = tmp_path / "session_old.md"
        log.write_text(
            "### [09:15:30] Turn 3\n**You**: remember the plan\n\n**Qwen**: got it\n\n",
            encoding="utf-8")
        turns = parse_session_turns(log)
        assert ("09:15:30", "You", "remember the plan") in turns
        assert ("09:15:30", "Qwen", "got it") in turns

    def test_multiline_turns_join(self, tmp_path):
        log = tmp_path / "session_x.md"
        log.write_text(
            "### [10:00:00] Turn 1\n**Maupo**: line one\nline two\n\n",
            encoding="utf-8")
        turns = parse_session_turns(log)
        assert turns == [("10:00:00", "Maupo", "line one line two")]

    def test_missing_file_returns_empty(self, tmp_path):
        assert parse_session_turns(tmp_path / "nope.md") == []


# ----------------------------------------------------------------- soft memory
class TestSoftMemory:
    def test_creates_file_with_header(self, tmp_path):
        mem = SoftMemory(tmp_path / "soft.md")
        assert "# Soft Memory" in mem.read()

    def test_append_and_get_entries_roundtrip(self, tmp_path):
        mem = SoftMemory(tmp_path / "soft.md")
        mem.append("Query: bitcoin\nResult: fake internet money")
        mem.append("Query: pygame\nResult: a game library")
        entries = mem.get_entries()
        assert len(entries) == 2
        assert "bitcoin" in entries[0]
        assert "pygame" in entries[1]

    def test_search_scores_keyword_overlap(self, tmp_path):
        mem = SoftMemory(tmp_path / "soft.md")
        mem.append("Query: python threading\nResult: threads share memory")
        mem.append("Query: pizza toppings\nResult: pineapple is divisive")
        hits = mem.search("how does python threading work")
        assert len(hits) == 1
        assert "threading" in hits[0]

    def test_search_empty_query_returns_nothing(self, tmp_path):
        mem = SoftMemory(tmp_path / "soft.md")
        mem.append("Query: x\nResult: y")
        assert mem.search("the a of") == []


# ----------------------------------------------------------------- hard memory
class TestHardMemory:
    def test_creates_file_with_header(self, tmp_path):
        mem = HardMemory(tmp_path / "hard.md")
        assert "# Hard Memory" in mem.read()

    def test_append_and_count_entries(self, tmp_path):
        mem = HardMemory(tmp_path / "hard.md")
        mem.append("the user hates mornings")
        mem.append("never mention pineapple")
        assert len(mem.get_entries()) == 2

    def test_entries_do_not_leak_timestamp_headers(self, tmp_path):
        mem = HardMemory(tmp_path / "hard.md")
        mem.append("never mention pineapple")
        entries = mem.get_entries()
        assert entries == ["never mention pineapple"]


# ----------------------------------------------------------------------- recall
@pytest.fixture
def recall_setup(tmp_path):
    sm = SessionManager(tmp_path / "sessions", "test-model")
    # Manually age a past session file so it counts as archived history.
    past = tmp_path / "sessions" / "session_20260101_000000.md"
    past.write_text(
        "# Chat Session: session_20260101_000000\n\n---\n\n"
        "### [10:00:00] Turn 1\n**You**: my sister moved to porto for the new job\n\n"
        "**Maupo**: that's huge\n\n",
        encoding="utf-8")
    soft = SoftMemory(tmp_path / "soft.md")
    hard = HardMemory(tmp_path / "hard.md")
    compressor = UniversalCompressor("test-model", "http://localhost:11434", sm.sessions_dir)
    recall = MemoryRecall(sm, soft, hard, compressor)
    return recall, past


class TestMemoryRecall:
    def test_keyword_recall_finds_past_moment(self, recall_setup, monkeypatch):
        recall, _ = recall_setup
        monkeypatch.setattr(recall, "_model_focus_terms", lambda q: [])
        out = recall.recall("how is my sister doing")
        assert "sister" in out
        assert "porto" in out
        assert recall.last_hits > 0

    def test_recall_miss_returns_empty(self, recall_setup, monkeypatch):
        recall, _ = recall_setup
        monkeypatch.setattr(recall, "_model_focus_terms", lambda q: [])
        assert recall.recall("zebra quantum xylophone") == ""

    def test_model_focus_terms_are_cached(self, recall_setup, monkeypatch):
        recall, _ = recall_setup
        calls = []

        def fake_terms(query):
            calls.append(query)
            return ["porto", "sister"]

        monkeypatch.setattr(recall, "_model_focus_terms", fake_terms)
        recall.recall("the porto thing")
        recall.recall("the porto thing")
        assert calls == ["the porto thing"]  # second pass served from cache

    def test_turns_cache_invalidated_by_mtime(self, recall_setup, monkeypatch):
        import os
        recall, past = recall_setup
        monkeypatch.setattr(recall, "_model_focus_terms", lambda q: [])
        assert recall.recall("sister") != ""
        cached_turns = recall._turns_cache[past.name]["turns"]

        past.write_text(past.read_text(encoding="utf-8") +
                        "### [11:00:00] Turn 2\n**You**: update: she loves it there\n\n",
                        encoding="utf-8")
        os.utime(past, (1e9 + 5, 1e9 + 5))  # force a new mtime
        assert recall.recall("loves it there") != ""
        assert recall._turns_cache[past.name]["turns"] != cached_turns


# ------------------------------------------------------------------- compressor
class TestUniversalCompressor:
    def test_compress_session_text_strips_header(self, tmp_path):
        comp = UniversalCompressor("m", "http://localhost:11434", tmp_path)
        monkey_target = comp

        def fake_call(prompt):
            monkey_target.last_prompt = prompt
            return "- bullets here"

        comp._call_model = fake_call
        out = comp.compress_session_text("s.md", "# header\n\n---\n\n**You**: hi")
        assert out == "- bullets here"
        assert "**You**: hi" in comp.last_prompt
        assert "# header" not in comp.last_prompt

    def test_compress_session_text_empty_body(self, tmp_path):
        comp = UniversalCompressor("m", "http://localhost:11434", tmp_path)

        def must_not_call(prompt):
            raise AssertionError("model called for an empty session body")

        comp._call_model = must_not_call
        # Header with an empty dialogue section: nothing to compress.
        assert comp.compress_session_text("s.md", "# header\n\n---\n\n") == ""

    def test_soft_memory_compression_backs_up_original(self, tmp_path):
        mem_path = tmp_path / "soft.md"
        mem_path.write_text(
            "# Soft Memory\n\n---\n\n## 2026-01-01\nQuery: a\nResult: b\n",
            encoding="utf-8")
        comp = UniversalCompressor("m", "http://localhost:11434", tmp_path)
        comp._call_model = lambda prompt: "## merged\nQuery: a\nKnowledge: b"
        orig, new, msg = comp.compress_soft_memory(mem_path)
        assert "saved" in msg
        assert mem_path.with_suffix(".md.bak").exists()  # original is recoverable
        assert "## merged" in mem_path.read_text(encoding="utf-8")

    def test_soft_memory_compression_survives_dead_model(self, tmp_path):
        mem_path = tmp_path / "soft.md"
        original = "# Soft Memory\n\n---\n\n## 2026-01-01\nQuery: a\nResult: b\n"
        mem_path.write_text(original, encoding="utf-8")
        comp = UniversalCompressor("m", "http://localhost:11434", tmp_path)

        def dead_model(prompt):
            raise ConnectionError("refused")

        comp._call_model = dead_model
        orig, new, msg = comp.compress_soft_memory(mem_path)
        assert "skipped" in msg
        assert mem_path.read_text(encoding="utf-8") == original  # untouched

    def test_rolling_cache_hit_costs_zero_model_calls(self, tmp_path):
        comp = UniversalCompressor("m", "http://localhost:11434", tmp_path)
        files = []
        for i in range(5):
            p = tmp_path / f"session_2026010{i}_000000.md"
            p.write_text(f"# s{i}\n\n---\n\n**You**: hi {i}", encoding="utf-8")
            files.append(p)
        summaries = [f"### s{i}\n- summary {i}" for i in range(5)]
        comp._call_model = lambda prompt: "MERGED HEAD"
        # Seed: cache the individual summaries and the combined state.
        cache = {f.name: {"mtime": f.stat().st_mtime, "summary": s}
                 for f, s in zip(files, summaries)}
        comp._save_cache(cache)
        comp.combined_cache_file.write_text(json.dumps({
            "window": [f.name for f in files], "head": "OLD HEAD",
            "tail": files[-1].name, "full": "OLD HEAD\n\n" + summaries[-1],
        }), encoding="utf-8")

        calls = []
        comp._call_model = lambda prompt: calls.append(1) or "X"

        out = comp.get_recent_sessions_summary(files)
        assert calls == []                      # pure cache hit
        assert out == "OLD HEAD\n\n" + summaries[-1]
