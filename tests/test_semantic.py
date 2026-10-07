"""Tests for the semantic recall index (fake embedder — never needs Ollama)."""

import json

import pytest

import maupo.semantic as semantic
from maupo.memory import MemoryRecall, SessionManager, SoftMemory, HardMemory, UniversalCompressor
from maupo.semantic import SemanticIndex

# Deterministic 4-dim "embeddings": vectors pointing along each axis.
AXIS = {
    "E": [1.0, 0.0, 0.0, 0.0],
    "N": [0.0, 1.0, 0.0, 0.0],
    "S": [0.0, 0.0, 1.0, 0.0],
    "W": [0.0, 0.0, 0.0, 1.0],
}
MIXED = [0.9, 0.1, 0.0, 0.0]   # closest to E


@pytest.fixture
def index(tmp_path):
    sm = SessionManager(tmp_path / "sessions", "test-model")
    past = tmp_path / "sessions" / "session_20260101_000000.md"
    past.write_text(
        "# Chat Session\n\n---\n\n"
        "### [10:00:00] Turn 1\n**You**: my sister moved to porto for the job\n\n"
        "**Maupo**: that's huge\n\n"
        "### [10:01:00] Turn 2\n**You**: the pizza debate continues\n\n"
        "**Maupo**: pineapple is a crime\n\n",
        encoding="utf-8")
    idx = SemanticIndex(sm, "test-model", "http://localhost:11434",
                        cache_dir=tmp_path / "sessions")
    return idx, sm, past


def _fake_axis_embedder(host, texts, model=semantic.EMBED_MODEL, timeout=10):
    """Deterministic fake: 'porto/sister/job' -> E, 'pizza' -> N, else W."""
    out = []
    for t in texts:
        low = t.lower()
        if "porto" in low or "sister" in low or "job" in low:
            out.append(list(AXIS["E"]))
        elif "pizza" in low or "pineapple" in low:
            out.append(list(AXIS["N"]))
        else:
            out.append(list(AXIS["W"]))
    return out


class TestSemanticIndex:
    def test_refresh_embeds_all_turns(self, index, monkeypatch):
        idx, _, _ = index
        monkeypatch.setattr(semantic, "embed_texts", _fake_axis_embedder)
        assert idx.refresh() == 4  # 2 user turns + 2 maupo turns
        assert idx.available is True
        assert len(idx._cache) == 4

    def test_second_refresh_costs_nothing(self, index, monkeypatch):
        idx, _, _ = index
        monkeypatch.setattr(semantic, "embed_texts", _fake_axis_embedder)
        idx.refresh()
        assert idx.refresh() == 0  # everything cached
        assert idx.pending_count == 0

    def test_budget_bounds_work(self, index, monkeypatch):
        idx, _, _ = index
        monkeypatch.setattr(semantic, "embed_texts", _fake_axis_embedder)
        assert idx.refresh(budget=2) == 2
        assert idx.pending_count == 2
        assert idx.refresh(budget=2) == 2   # rest lands on a later pass
        assert idx.refresh() == 0

    def test_appending_does_not_reembed_old_turns(self, index, monkeypatch):
        import os
        idx, _, past = index
        monkeypatch.setattr(semantic, "embed_texts", _fake_axis_embedder)
        idx.refresh()
        past.write_text(past.read_text(encoding="utf-8") +
                        "### [12:00:00] Turn 3\n**You**: new thought about porto\n\n",
                        encoding="utf-8")
        os.utime(past, (1e9 + 5, 1e9 + 5))
        # Content-hash keys: only the NEW turn embeds, older turns keep
        # their vectors even though the file's mtime changed.
        assert idx.refresh() == 1
        assert len(idx._cache) == 5

    def test_edited_turn_reembeds(self, index, monkeypatch):
        import os
        idx, _, past = index
        monkeypatch.setattr(semantic, "embed_texts", _fake_axis_embedder)
        idx.refresh()
        # Rewrite turn 1's text in place: that turn's hash changes.
        past.write_text(past.read_text(encoding="utf-8").replace(
            "porto for the job", "porto for the new job"), encoding="utf-8")
        os.utime(past, (1e9 + 5, 1e9 + 5))
        assert idx.refresh() == 1  # exactly the edited turn

    def test_search_finds_by_meaning(self, index, monkeypatch):
        idx, _, _ = index
        monkeypatch.setattr(semantic, "embed_texts", _fake_axis_embedder)
        idx.refresh()

        def fake_query(host, texts, model=semantic.EMBED_MODEL, timeout=10):
            if MIXED[0] and "city" in texts[0].lower():
                return [list(MIXED)]  # points mostly along E
            return _fake_axis_embedder(host, texts)

        monkeypatch.setattr(semantic, "embed_texts", fake_query)
        hits = idx.search("how is the city thing going")
        assert hits, "meaning-close query must surface the porto turns"
        assert "porto" in hits[0][3] or "sister" in hits[0][3]
        assert hits[0][4] > 0.9

    def test_search_empty_index_is_safe(self, index):
        idx, _, _ = index
        assert idx.search("anything") == []

    def test_unavailable_embedder_is_silent(self, index, monkeypatch):
        idx, _, _ = index

        def dead(host, texts, model=semantic.EMBED_MODEL, timeout=10):
            return None

        monkeypatch.setattr(semantic, "embed_texts", dead)
        assert idx.refresh() == 0
        assert idx.available is False
        assert idx.search("porto") == []   # graceful, no exception
        assert idx.pending_count >= 0

    def test_cache_survives_restart(self, index, monkeypatch, tmp_path):
        idx, _, _ = index
        monkeypatch.setattr(semantic, "embed_texts", _fake_axis_embedder)
        idx.refresh()
        idx2 = SemanticIndex(idx.session_manager, "test-model", "http://localhost:11434",
                             cache_dir=tmp_path / "sessions")
        assert len(idx2._cache) == 4
        monkeypatch.setattr(semantic, "embed_texts", _fake_axis_embedder)
        assert idx2.refresh() == 0  # no re-embedding after restart

    def test_prunes_turns_that_vanish(self, index, monkeypatch):
        idx, _, past = index
        monkeypatch.setattr(semantic, "embed_texts", _fake_axis_embedder)
        idx.refresh()
        past.write_text("# Chat Session\n\n---\n\n", encoding="utf-8")
        import os
        os.utime(past, (1e9 + 9, 1e9 + 9))
        idx.refresh()
        assert len(idx._cache) == 0


class TestRecallIntegration:
    def test_recall_merges_semantic_hits_without_duplicates(self, tmp_path, monkeypatch):
        sm = SessionManager(tmp_path / "sessions", "test-model")
        past = tmp_path / "sessions" / "session_20260101_000000.md"
        past.write_text(
            "# Chat Session\n\n---\n\n"
            "### [10:00:00] Turn 1\n**You**: my sister moved to porto for the job\n\n",
            encoding="utf-8")
        soft = SoftMemory(tmp_path / "soft.md")
        hard = HardMemory(tmp_path / "hard.md")
        comp = UniversalCompressor("test-model", "http://localhost:11434", sm.sessions_dir)
        recall = MemoryRecall(sm, soft, hard, comp)

        monkeypatch.setattr(recall, "_model_focus_terms", lambda q: ["porto"])
        # Semantic layer: pretend a meaning-close extra moment exists.
        monkeypatch.setattr(recall.semantic, "search", lambda q, top_k=4: [
            (past.stem, "10:01:00", "Maupo", "did she like the city", 0.83)])
        out = recall.recall("porto")
        assert "porto" in out
        assert "(close in meaning)" in out
        assert out.count("close in meaning") == 1

    def test_recall_dedups_identical_semantic_hits(self, tmp_path, monkeypatch):
        sm = SessionManager(tmp_path / "sessions", "test-model")
        past = tmp_path / "sessions" / "session_20260101_000000.md"
        past.write_text(
            "# Chat Session\n\n---\n\n"
            "### [10:00:00] Turn 1\n**You**: my sister moved to porto\n\n",
            encoding="utf-8")
        soft = SoftMemory(tmp_path / "soft.md")
        hard = HardMemory(tmp_path / "hard.md")
        comp = UniversalCompressor("test-model", "http://localhost:11434", sm.sessions_dir)
        recall = MemoryRecall(sm, soft, hard, comp)

        monkeypatch.setattr(recall, "_model_focus_terms", lambda q: ["porto"])
        # The exact turn text the keyword pass already found -> must not
        # appear twice (both sides compare normalized 120-char prefixes).
        monkeypatch.setattr(recall.semantic, "search", lambda q, top_k=4: [
            (past.stem, "10:00:00", "You", "my sister moved to porto", 0.9)])
        out = recall.recall("porto")
        assert "close in meaning" not in out  # deduped

    def test_recall_works_without_semantic_layer(self, tmp_path, monkeypatch):
        sm = SessionManager(tmp_path / "sessions", "test-model")
        past = tmp_path / "sessions" / "session_20260101_000000.md"
        past.write_text(
            "# Chat Session\n\n---\n\n"
            "### [10:00:00] Turn 1\n**You**: my sister moved to porto\n\n",
            encoding="utf-8")
        soft = SoftMemory(tmp_path / "soft.md")
        hard = HardMemory(tmp_path / "hard.md")
        comp = UniversalCompressor("test-model", "http://localhost:11434", sm.sessions_dir)
        recall = MemoryRecall(sm, soft, hard, comp)
        recall.semantic.available = False
        recall.semantic.search = lambda q, top_k=4: (_ for _ in ()).throw(
            AssertionError("semantic layer must not be consulted when unavailable"))

        monkeypatch.setattr(recall, "_model_focus_terms", lambda q: ["porto"])
        out = recall.recall("porto")
        assert "porto" in out  # keyword path alone still answers
