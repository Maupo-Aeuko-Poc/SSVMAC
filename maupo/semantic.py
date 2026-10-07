"""Semantic recall: meaning-based search over every archived turn.

Maupo's keyword recall matches words; this index matches *meaning*, so "that
thing about the thing" can find a moment even when no word overlaps. It embeds
every archived turn with a tiny local model (nomic-embed-text via Ollama —
runs on the same one GPU) and keeps an mtime-cache on disk.

Invariants kept:
- Bounded work: at most EMBED_BUDGET_TURNS new turns are embedded per refresh;
  the rest stay pending until a later idle heartbeat. Recall never blocks on a
  cold index — the keyword path answers first.
- One GPU: refreshes are only requested by the heartbeat (idle) or explicit
  /recall, never while a reply is generating.
- Raw logs are never touched; the index is a derived cache beside them.

Cosine similarity on L2-normalized vectors is a plain dot product, so the
whole index is just an in-memory matrix and one pass — instant at this scale.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
import urllib.request
from pathlib import Path
from typing import Any, Optional

EMBED_MODEL = "nomic-embed-text"
EMBED_DIM = 768
EMBED_TIMEOUT = 10
EMBED_BUDGET_TURNS = 64          # max new turns embedded per refresh
INDEX_FILE = ".semantic_index.json"
SIMILARITY_FLOOR = 0.42          # below this, a hit is noise


def embed_texts(host: str, texts: list[str], model: str = EMBED_MODEL,
                timeout: int = EMBED_TIMEOUT) -> Optional[list[list[float]]]:
    """Embed texts via Ollama. Returns None if the model is unavailable —
    callers treat that as 'no semantic layer', never as an error."""
    clean = [t if t.strip() else " " for t in texts]
    payload = {"model": model, "input": clean}
    request = urllib.request.Request(
        f"{host.rstrip('/')}/api/embed",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Connection": "keep-alive"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            data = json.loads(response.read().decode("utf-8"))
    except Exception:
        return None
    embeddings = data.get("embeddings")
    if not isinstance(embeddings, list) or len(embeddings) != len(clean):
        return None
    return embeddings


class SemanticIndex:
    """Turn embeddings + metadata, keyed per turn and content-hashed.

    A turn is re-embedded only when its text actually changes (appending a new
    turn to a session never re-embeds the older turns in that file).
    """

    def __init__(self, session_manager, model: str, host: str,
                 cache_dir: Optional[Path] = None) -> None:
        self.session_manager = session_manager
        self.model = model          # the chat model (embedding uses EMBED_MODEL)
        self.host = host
        self.cache_path = (cache_dir or session_manager.sessions_dir) / INDEX_FILE
        self.available: Optional[bool] = None   # None = never probed
        self.last_refresh: float = 0.0
        self.pending_count: int = 0
        self._load_cache()

    # ------------------------------------------------------------- persistence
    def _load_cache(self) -> None:
        self._cache: dict[str, dict[str, Any]] = {}
        try:
            data = json.loads(self.cache_path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                self._cache = data
        except Exception:
            self._cache = {}

    def _save_cache(self) -> None:
        try:
            self.cache_path.write_text(json.dumps(self._cache), encoding="utf-8")
        except Exception:
            pass

    # ----------------------------------------------------------------- gathering
    def _session_files(self) -> list[Path]:
        files = [
            p for p in self.session_manager.sessions_dir.glob("session_*.md")
            if p.is_file() and not p.name.endswith(".compressed.md")
            and p.name != self.session_manager.session_file.name
        ]
        files.sort(key=lambda p: p.name)
        return files

    def _all_turns(self) -> list[tuple[str, int, str, str, str]]:
        """(session_stem, turn_index, timestamp, speaker, text) for every archived turn."""
        from maupo.memory import parse_session_turns

        turns: list[tuple[str, int, str, str, str]] = []
        for path in self._session_files():
            for i, (ts, speaker, text) in enumerate(parse_session_turns(path)):
                turns.append((path.stem, i, ts, speaker, text))
        return turns

    @staticmethod
    def _text_key(text: str) -> str:
        return hashlib.sha1(text[:1200].encode("utf-8")).hexdigest()[:16]

    # ------------------------------------------------------------------ updating
    def refresh(self, budget: int = EMBED_BUDGET_TURNS) -> int:
        """Embed up to `budget` new-or-changed turns. Returns how many new
        vectors were stored (0 when Ollama or the model is unavailable)."""
        self.last_refresh = time.time()
        all_turns = self._all_turns()
        cache = self._cache

        todo: list[tuple[str, int, str, str, str, str]] = []  # + text_key
        live_keys: set[str] = set()
        for stem, idx, ts, speaker, text in all_turns:
            key = f"{stem}::{idx}"
            live_keys.add(key)
            txk = self._text_key(text)
            entry = cache.get(key)
            if entry is None or entry.get("txk") != txk:
                todo.append((stem, idx, ts, speaker, text, txk))

        # Prune entries whose turns no longer exist (edited/pruned files).
        for key in [k for k in cache if k not in live_keys]:
            cache.pop(key, None)

        self.pending_count = max(0, len(todo) - budget)
        if not todo:
            return 0
        todo = todo[:budget]
        texts = [t[4][:1200] for t in todo]
        vectors = embed_texts(self.host, texts)
        if vectors is None:
            self.available = False
            return 0
        self.available = True
        for (stem, idx, ts, speaker, text, txk), vec in zip(todo, vectors):
            cache[f"{stem}::{idx}"] = {
                "txk": txk,
                "ts": ts,
                "who": speaker,
                "text": text[:600],
                "vec": vec,
            }
        self._save_cache()
        return len(todo)

    # ----------------------------------------------------------------- searching
    def search(self, query: str, top_k: int = 4) -> list[tuple[str, str, str, str, float]]:
        """Return [(session_stem, timestamp, speaker, text, score)] most similar
        turns. Empty list when the index is empty or embeddings are unavailable."""
        if not self._cache:
            return []
        qvecs = embed_texts(self.host, [query])
        if not qvecs:
            return []
        q = qvecs[0]
        qnorm = math.sqrt(sum(x * x for x in q)) or 1.0
        scored: list[tuple[str, str, str, str, float]] = []
        for key, entry in self._cache.items():
            vec = entry.get("vec")
            if not vec:
                continue
            vnorm = math.sqrt(sum(x * x for x in vec)) or 1.0
            dot = sum(a * b for a, b in zip(q, vec))
            score = dot / (qnorm * vnorm)
            if score >= SIMILARITY_FLOOR:
                stem, _idx = key.split("::", 1)
                scored.append((stem, entry.get("ts", ""), entry.get("who", ""),
                               entry.get("text", ""), score))
        scored.sort(key=lambda item: -item[4])
        return scored[:top_k]
