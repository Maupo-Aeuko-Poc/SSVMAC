"""Memory & sessions: session logs, the UniversalCompressor, soft/hard memory,
and deep recall across every session ever logged.

Architecture invariants kept here:
1. Raw session logs are NEVER deleted - compression summarizes, recall
   resurrects. Old `**Qwen**:` and new `**Maupo**:` log formats both parse.
"""

from __future__ import annotations

import json
import re
import threading
import time
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any

from maupo.net import STOP_WORDS, ollama_chat_once
from maupo.semantic import SemanticIndex

# Memory & Sessions Paths
BASE_DIR = Path(__file__).parent.parent
MEMORY_DIR = BASE_DIR / "memory"
SOFT_MEM_PATH = MEMORY_DIR / "soft_mem.md"
HARD_MEM_PATH = MEMORY_DIR / "hard_mem.md"
SESSIONS_DIR = MEMORY_DIR / "sessions"


class SessionManager:
    """Manages chat session logs under memory/sessions/."""

    def __init__(self, sessions_dir: Path, model_name: str) -> None:
        self.sessions_dir = sessions_dir
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name
        # Bump seconds until the id is unique: launching twice within one
        # second must never overwrite the previous session log.
        self.session_id = datetime.now().strftime("session_%Y%m%d_%H%M%S")
        while (self.sessions_dir / f"{self.session_id}.md").exists():
            time.sleep(1.0)
            self.session_id = datetime.now().strftime("session_%Y%m%d_%H%M%S")
        self.session_file = self.sessions_dir / f"{self.session_id}.md"
        self.turn_count = 0
        self._init_session_file()

    def _init_session_file(self) -> None:
        start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        header = (
            f"# Chat Session: {self.session_id}\n"
            f"- Started: {start_time}\n"
            f"- Model: {self.model_name}\n\n"
            f"---\n\n"
        )
        self.session_file.write_text(header, encoding="utf-8")

    def log_turn(self, user_text: str, assistant_text: str) -> None:
        """Append a conversational turn immediately to the session file."""
        self.turn_count += 1
        timestamp = datetime.now().strftime("%H:%M:%S")
        turn_md = (
            f"### [{timestamp}] Turn {self.turn_count}\n"
            f"**You**: {user_text.strip()}\n\n"
            f"**Maupo**: {assistant_text.strip()}\n\n"
        )
        with self.session_file.open("a", encoding="utf-8") as f:
            f.write(turn_md)

    def get_past_session_files(self, limit: int = 5) -> list[Path]:
        """Return the most recent past session files, excluding the current session."""
        all_sessions = [
            p for p in self.sessions_dir.glob("session_*.md")
            if p.is_file() and p.name != self.session_file.name and not p.name.endswith(".compressed.md")
        ]
        all_sessions.sort(key=lambda p: p.name)
        return all_sessions[-limit:]

    def list_all_sessions(self) -> list[dict[str, Any]]:
        """Return metadata for all sessions."""
        all_sessions = [
            p for p in self.sessions_dir.glob("session_*.md")
            if p.is_file() and not p.name.endswith(".compressed.md")
        ]
        all_sessions.sort(key=lambda p: p.name, reverse=True)
        results = []
        for p in all_sessions:
            content = p.read_text(encoding="utf-8")
            turns = len(re.findall(r"### \[\d{2}:\d{2}:\d{2}\]", content))
            results.append({
                "path": p,
                "name": p.name,
                "is_current": p.name == self.session_file.name,
                "turns": turns,
                "size_bytes": p.stat().st_size,
            })
        return results


class UniversalCompressor:
    """Uses Qwen model to perform high-density, semantic lossless compression on sessions & memory."""

    def __init__(self, model: str, host: str, sessions_dir: Path) -> None:
        self.model = model
        self.host = host.rstrip("/")
        self.sessions_dir = sessions_dir
        self.cache_file = self.sessions_dir / ".sessions_cache.json"
        self.combined_cache_file = self.sessions_dir / ".sessions_combined_cache.json"

    def _call_model(self, prompt: str) -> str:
        """Call Ollama with fast options and think=False for semantic execution."""
        url = f"{self.host}/api/chat"
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": True,
            "think": False,
            "keep_alive": "60m",
            "options": {
                "num_ctx": 4096,
                "num_predict": 384,
                "temperature": 0.5,
            },
        }
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", "Connection": "keep-alive"},
            method="POST",
        )
        content_parts: list[str] = []
        with urllib.request.urlopen(req, timeout=120) as resp:
            for raw_line in resp:
                if not raw_line.strip():
                    continue
                chunk = json.loads(raw_line.decode("utf-8"))
                message = chunk.get("message") or {}
                content = message.get("content")
                if isinstance(content, str) and content:
                    content_parts.append(content)
                if chunk.get("done"):
                    break
        return "".join(content_parts).strip()

    def _load_cache(self) -> dict[str, Any]:
        if self.cache_file.exists():
            try:
                return json.loads(self.cache_file.read_text(encoding="utf-8"))
            except Exception:
                return {}
        return {}

    def _save_cache(self, cache: dict[str, Any]) -> None:
        try:
            self.cache_file.write_text(json.dumps(cache, indent=2), encoding="utf-8")
        except Exception:
            pass

    def compress_session_text(self, session_name: str, session_text: str) -> str:
        """Semantically compress a single session's text."""
        dialogue_idx = session_text.find("---")
        dialogue_content = session_text[dialogue_idx + 3:].strip() if dialogue_idx != -1 else session_text.strip()
        if not dialogue_content:
            return ""

        prompt = f"""You are a high-density universal data compressor.
Compress the following session log semantically without losing ANY factual information:
- Remove duplicate phrases, polite filler, greetings, and repetitive conversational fluff.
- Merge similar thoughts and consecutive queries into unified concise points.
- CRITICAL: Retain ALL unique facts, names, dates, numbers, technical specifics, topics discussed, code/commands, and user preferences.
- Output ONLY compact bullet points.

Session [{session_name}]:
{dialogue_content}

Compressed Summary:"""

        return self._call_model(prompt)

    def get_recent_sessions_summary(self, past_files: list[Path], on_updated=None) -> str:
        """Summarize and compress the last up to 5 session logs with caching.

        Two tiers keep launches fast no matter how long history grows:
        per-session summaries are cached forever (past logs never change), and
        the combined summary is maintained incrementally - a sliding window
        only merges the previous head with the one incoming tail instead of
        re-reading everything. The stitched result (head + newest session)
        is what rides in the system prompt. In async contexts (on_updated
        given), a slid window serves its cached stitch instantly and the
        delta-merge refreshes via on_updated(fresh_summary) when it lands.
        """
        if not past_files:
            return ""

        cache = self._load_cache()
        individual_summaries: list[str] = []
        mtimes: dict[str, float] = {}
        cache_updated = False

        for f in past_files:
            mtime = f.stat().st_mtime
            mtimes[f.name] = mtime
            cached_item = cache.get(f.name)
            if cached_item and cached_item.get("mtime") == mtime and cached_item.get("summary"):
                summary = cached_item["summary"]
            else:
                text = f.read_text(encoding="utf-8")
                try:
                    summary = self.compress_session_text(f.name, text)
                    if summary:
                        cache[f.name] = {"mtime": mtime, "summary": summary}
                        cache_updated = True
                except Exception as e:
                    summary = f"- Session {f.name}: (Compression skipped: {e})"
            if summary:
                individual_summaries.append(f"### {f.stem}\n{summary}")

        if cache_updated:
            self._save_cache(cache)

        if not individual_summaries:
            return ""

        if len(individual_summaries) == 1:
            return individual_summaries[0]

        def _load_combined() -> dict:
            try:
                return json.loads(self.combined_cache_file.read_text(encoding="utf-8")) if self.combined_cache_file.exists() else {}
            except Exception:
                return {}

        def _save_combined(st: dict) -> None:
            try:
                self.combined_cache_file.write_text(json.dumps(st, indent=2), encoding="utf-8")
            except Exception:
                pass

        if len(individual_summaries) == 2:
            # Nothing to merge yet: the older session IS the head.
            stitched = f"{individual_summaries[0]}\n\n{individual_summaries[1]}"
            _save_combined({"window": [f.name for f in past_files], "head": individual_summaries[0],
                            "tail": past_files[-1].name, "full": stitched})
            return stitched

        # ---- rolling combined summary (only for the standard window) ------
        if len(past_files) < 4:
            # Non-standard windows (e.g. the dream's 3-file consolidation)
            # merge fully and never touch the rolling cache.
            combined_text = "\n\n".join(individual_summaries[:-1])
            try:
                merged = self._call_model(f"""You are a universal memory compressor.
The following are summaries of the user's past chat sessions.
Compress them into a single, cohesive, ultra-dense summary of what was discussed, what the user cares about, and what was accomplished:
- Eliminate redundant cross-session topics.
- Retain all facts, user preferences, names, and topics.
- Keep it concise, high-density, and structured in bullet points.

Past Sessions:
{combined_text}

Mini Compressed Past Sessions Summary:""")
                if merged:
                    return f"{merged}\n\n{individual_summaries[-1]}"
            except Exception:
                pass
            return "\n\n".join(individual_summaries)

        window = [f.name for f in past_files]
        tail_name = window[-1]
        tail_item = cache.get(tail_name) or {}
        state = _load_combined()

        # Same window as before: pure cache hit, zero model calls.
        if (state.get("window") == window and state.get("head")
                and state.get("tail") == tail_name
                and tail_item.get("mtime") == mtimes.get(tail_name)):
            return state.get("full") or f"{state['head']}\n\n{individual_summaries[-1]}"

        slide_hit = (state.get("full") and state.get("window") and len(state["window"]) > 1
                     and state["window"][1:] == window[:-1])

        # Window slid by one (the normal every-launch case).
        if slide_hit and on_updated is not None:
            # Async context (startup): serve the previous stitched summary
            # instantly, then delta-merge in the background and re-apply when
            # the fresh result lands. Bounded work at any history size.
            old_window, old_head = state["window"], state["head"]
            old_tail_summary = (cache.get(old_window[-1]) or {}).get("summary", "")
            tail_part = individual_summaries[-1]

            def _bg_refresh() -> None:
                try:
                    head = self._call_model(f"""You are a universal memory compressor.
The following are summaries of the user's past chat sessions.

[Rolling summary of older sessions:]
{old_head}

[Summary of the most recent session:]
{old_tail_summary}

Compress them into a single, cohesive, ultra-dense summary of what was discussed, what the user cares about, and what was accomplished:
- Eliminate redundant cross-session topics.
- Retain all facts, user preferences, names, and topics.
- Keep it concise, high-density, and structured in bullet points.

Mini Compressed Past Sessions Summary:""") or ""
                    if head:
                        _save_combined({"window": window, "head": head, "tail": tail_name,
                                        "full": f"{head}\n\n{tail_part}"})
                        try:
                            on_updated(f"{head}\n\n{tail_part}")
                        except Exception:
                            pass
                except Exception:
                    pass

            threading.Thread(target=_bg_refresh, name="summary-refresh", daemon=True).start()
            return state["full"]

        head = ""
        # Sync context (explicit /compress, dreams) or no cached stitch yet:
        # merge the previous head with the incoming session - bounded work.
        if slide_hit and state.get("head"):
            old_tail_item = cache.get(state["window"][-1]) or {}
            if old_tail_item.get("summary"):
                try:
                    head = self._call_model(f"""You are a universal memory compressor.
The following are summaries of the user's past chat sessions.

[Rolling summary of older sessions:]
{state['head']}

[Summary of the most recent session:]
{old_tail_item['summary']}

Compress them into a single, cohesive, ultra-dense summary of what was discussed, what the user cares about, and what was accomplished:
- Eliminate redundant cross-session topics.
- Retain all facts, user preferences, names, and topics.
- Keep it concise, high-density, and structured in bullet points.

Mini Compressed Past Sessions Summary:""") or ""
                except Exception:
                    head = ""

        if not head:
            # Cold cache or a jumped window: merge everything but the tail.
            base = individual_summaries[:-1]
            if len(base) == 1:
                head = base[0]
            else:
                combined_text = "\n\n".join(base)
                try:
                    head = self._call_model(f"""You are a universal memory compressor.
The following are summaries of the user's past 5 chat sessions.
Compress them into a single, cohesive, ultra-dense summary of what was discussed, what the user cares about, and what was accomplished:
- Eliminate redundant cross-session topics.
- Retain all facts, user preferences, names, and topics.
- Keep it concise, high-density, and structured in bullet points.

Past Sessions:
{combined_text}

Mini Compressed Past Sessions Summary:""") or ""
                except Exception:
                    head = combined_text

        if head:
            stitched_fresh = f"{head}\n\n{individual_summaries[-1]}"
            _save_combined({"window": window, "head": head, "tail": tail_name, "full": stitched_fresh})
        return f"{head}\n\n{individual_summaries[-1]}"

    def compress_soft_memory(self, soft_memory_path: Path) -> tuple[int, int, str]:
        """Compress soft_mem.md: merges duplicate queries and consolidates knowledge."""
        if not soft_memory_path.exists():
            return 0, 0, "Soft memory file does not exist."

        content = soft_memory_path.read_text(encoding="utf-8")
        orig_size = len(content)
        lines = content.splitlines()

        entry_lines = [l for l in lines if l.startswith("## ") or l.startswith("Query:") or l.startswith("Result:") or l.startswith("User provided:")]
        if not entry_lines:
            return orig_size, orig_size, "No soft memory entries to compress."

        prompt = f"""You are a universal knowledge compressor.
The following is the soft memory bank of an AI companion containing learned queries, facts, and internet lookups.
Compress and consolidate these entries:
1. Merge duplicate queries or topics into single, comprehensive knowledge entries.
2. Eliminate redundant or outdated phrases.
3. PRESERVE ALL unique facts, names, dates, numbers, definitions, and technical details without losing any real information.
4. Format the output strictly as markdown entries with:
## [Topic or Date]
Query: [Consolidated query/topic]
Knowledge: [Accurate, dense synthesized information]

Original Soft Memory:
{content}

Consolidated Soft Memory:"""

        try:
            compressed_content = self._call_model(prompt)
        except Exception as e:
            # Never let a dead/unreachable Ollama crash the caller (/compress,
            # dream consolidation). The original memory stays untouched.
            return orig_size, orig_size, f"Compression skipped (model unreachable: {e})."
        if not compressed_content:
            return orig_size, orig_size, "Compression returned empty output."

        new_file_text = (
            "# Soft Memory\n\n"
            "*Entries semantically consolidated by Universal Compressor without loss of information.*\n\n"
            "---\n\n"
            f"{compressed_content}\n"
        )
        backup_path = soft_memory_path.with_suffix(".md.bak")
        backup_path.write_text(content, encoding="utf-8")
        soft_memory_path.write_text(new_file_text, encoding="utf-8")
        new_size = len(new_file_text)

        return orig_size, new_size, f"Compressed {orig_size} bytes -> {new_size} bytes (saved {orig_size - new_size} bytes)."


class SoftMemory:
    """Manages soft memory - information the model learns and can expand."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.write_text(
                "# Soft Memory\n\n*Entries added when the model retrieves information via internet lookup or user query.*\n\n---\n",
                encoding="utf-8",
            )

    def read(self) -> str:
        return self.path.read_text(encoding="utf-8")

    def append(self, entry: str) -> None:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with self.path.open("a", encoding="utf-8") as f:
            f.write(f"\n## {timestamp}\n{entry.strip()}\n")

    def get_entries(self) -> list[str]:
        content = self.read()
        lines = content.splitlines()
        entries = []
        current_entry: list[str] = []
        in_entry = False

        for line in lines:
            if line.startswith("## "):
                if in_entry and current_entry:
                    entry_text = "\n".join(current_entry).strip()
                    if entry_text:
                        entries.append(entry_text)
                current_entry = [line]
                in_entry = True
            elif in_entry:
                current_entry.append(line)

        if in_entry and current_entry:
            entry_text = "\n".join(current_entry).strip()
            if entry_text:
                entries.append(entry_text)

        return entries

    def search(self, query: str) -> list[str]:
        """Search for relevant entries in soft memory using keyword overlap."""
        entries = self.get_entries()
        if not entries:
            return []

        words = set(re.findall(r"\b[a-z0-9_]{3,}\b", query.lower()))
        keywords = words - STOP_WORDS
        if not keywords:
            return []

        scored_entries: list[tuple[int, str]] = []
        for entry in entries:
            entry_words = set(re.findall(r"\b[a-z0-9_]{3,}\b", entry.lower()))
            overlap = keywords & entry_words
            if overlap:
                scored_entries.append((len(overlap), entry))

        scored_entries.sort(key=lambda item: item[0], reverse=True)
        return [entry for _, entry in scored_entries[:5]]


class HardMemory:
    """Manages hard memory - permanent knowledge only user can write via special phrases."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.write_text(
                "# Hard Memory\n\n*Permanent knowledge entries added when the user uses phrases like \"listen to me\" or \"never forget\".*\n\n---\n",
                encoding="utf-8",
            )

    def read(self) -> str:
        return self.path.read_text(encoding="utf-8")

    def append(self, entry: str) -> None:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with self.path.open("a", encoding="utf-8") as f:
            f.write(f"\n## {timestamp}\n{entry.strip()}\n")

    def get_entries(self) -> list[str]:
        content = self.read()
        lines = content.splitlines()
        entries = []
        current_entry: list[str] = []
        in_entry = False

        for line in lines:
            if line.startswith("## "):
                if in_entry and current_entry:
                    entry_text = "\n".join(current_entry).strip()
                    if entry_text:
                        entries.append(entry_text)
                current_entry = []
                in_entry = True
            elif in_entry:
                current_entry.append(line)

        if in_entry and current_entry:
            entry_text = "\n".join(current_entry).strip()
            if entry_text:
                entries.append(entry_text)

        return entries


def parse_session_turns(session_path: Path) -> list[tuple[str, str, str]]:
    """Parse any session log (old or new format) into (timestamp, speaker, text) turns."""
    turns: list[tuple[str, str, str]] = []
    ts, speaker, buf = "", "", []
    try:
        lines = session_path.read_text(encoding="utf-8").splitlines()
    except Exception:
        return turns
    for line in lines:
        m = re.match(r"^### \[(\d{2}:\d{2}:\d{2})\]", line)
        if m:
            ts = m.group(1)
            continue
        m = re.match(r"^\*\*(You|Qwen|Maupo)\*\*: ?(.*)$", line)
        if m:
            if speaker and buf:
                turns.append((ts, speaker, " ".join(buf).strip()))
            speaker, buf = m.group(1), [m.group(2)]
            continue
        if speaker and line.strip():
            buf.append(line.strip())
        elif speaker and not line.strip() and buf:
            turns.append((ts, speaker, " ".join(buf).strip()))
            speaker, buf = "", []
    if speaker and buf:
        turns.append((ts, speaker, " ".join(buf).strip()))
    return turns


class MemoryRecall:
    """Deep recall across EVERY session ever logged, plus soft and hard memory.

    Guarantees the never-forget promise: raw session logs stay on disk forever,
    and every recall scans them whole. A keyword-scored pass finds exact
    moments from any past day; one cached model pass distills fuzzy phrasings
    into extra search terms so 'that thing about the thing' still matches.
    """

    MAX_QUERY_CHARS = 400
    MAX_TURN_SNIPPET = 220
    CACHE_FILE = ".recall_cache.json"

    def __init__(self, session_manager: SessionManager, soft_memory: SoftMemory,
                 hard_memory: HardMemory, compressor: UniversalCompressor) -> None:
        self.session_manager = session_manager
        self.soft_memory = soft_memory
        self.hard_memory = hard_memory
        self.compressor = compressor
        self.cache_path = session_manager.sessions_dir / self.CACHE_FILE
        self.turns_cache_path = session_manager.sessions_dir / ".turns_cache.json"
        self._turns_cache: dict[str, Any] = {}
        try:
            self._turns_cache = json.loads(self.turns_cache_path.read_text(encoding="utf-8"))
        except Exception:
            self._turns_cache = {}
        self.last_hits = 0
        # Meaning-based layer over the same logs: embeds with nomic-embed-text
        # when available, silently absent when it is not. Keyword recall
        # always works; semantic recall just makes it better.
        self.semantic = SemanticIndex(session_manager, compressor.model, compressor.host)

    def _all_session_files(self) -> list[Path]:
        """Every archived session (oldest first), excluding the active one."""
        files = [
            p for p in self.session_manager.sessions_dir.glob("session_*.md")
            if p.is_file() and not p.name.endswith(".compressed.md")
            and p.name != self.session_manager.session_file.name
        ]
        files.sort(key=lambda p: p.name)
        return files

    def _load_cache(self) -> dict[str, Any]:
        try:
            return json.loads(self.cache_path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _save_cache(self, cache: dict[str, Any]) -> None:
        try:
            self.cache_path.write_text(json.dumps(cache), encoding="utf-8")
        except Exception:
            pass

    def _save_turns_cache(self) -> None:
        try:
            self.turns_cache_path.write_text(json.dumps(self._turns_cache), encoding="utf-8")
        except Exception:
            pass

    def _keywords(self, query: str) -> list[str]:
        words = [w for w in re.findall(r"\b[a-zA-Z0-9_]{3,}\b", query.lower()) if w not in STOP_WORDS]
        return words or [w.lower() for w in re.findall(r"\b[a-zA-Z0-9_]{2,}\b", query)]

    def _gather_turns(self) -> list[tuple[str, str, str, str]]:
        """(session, timestamp, speaker, text) for every archived turn ever.

        Parsed turns are cached per file and invalidated by file mtime, so
        recall stays fast no matter how much history accumulates - each file
        is parsed once per change, not once per recall.
        """
        pool: list[tuple[str, str, str, str]] = []
        cache = self._turns_cache
        dirty = False
        for path in self._all_session_files():
            try:
                mtime = path.stat().st_mtime
            except OSError:
                continue
            entry = cache.get(path.name)
            if entry is None or entry["mtime"] != mtime:
                turns = parse_session_turns(path)
                cache[path.name] = {"mtime": mtime, "turns": turns}
                dirty = True
            for ts, speaker, text in cache[path.name]["turns"]:
                pool.append((path.stem, ts, speaker, text))
        if dirty:
            self._save_turns_cache()
        return pool

    def _model_focus_terms(self, query: str) -> list[str]:
        """One cheap model pass: turn fuzzy phrasing into searchable terms."""
        prompt = (
            "Turn this recall request into 3-6 short search terms that would "
            "literally appear in the original chat messages. Reply with the "
            "terms only, comma separated, nothing else.\n"
            f"Request: {query}\nTerms:"
        )
        try:
            raw = ollama_chat_once(self.compressor.model, self.compressor.host, prompt,
                                   temperature=0.2, num_predict=48, timeout=20)
            return [t.strip().lower() for t in re.split(r"[,\n]", raw) if 2 < len(t.strip()) < 30][:8]
        except Exception:
            return []

    def recall(self, query: str) -> str:
        """Search everything ever said and return the closest moments, formatted."""
        query = (query or "").strip()[: self.MAX_QUERY_CHARS]
        if not query:
            return ""
        pool = self._gather_turns()
        if not pool:
            return ""

        cache = self._load_cache()
        cached = cache.get(query.lower())
        if cached and cached.get("terms") is not None:
            terms = cached["terms"]
        else:
            terms = self._model_focus_terms(query)
            cache[query.lower()] = {"terms": terms}
            self._save_cache(cache)

        keywords = set(self._keywords(query)) | {t for t in terms if t}
        scored: list[tuple[int, str, str, str, str]] = []
        for name, ts, speaker, text in pool:
            low = text.lower()
            score = sum(1 for k in keywords if k in low)
            if score:
                scored.append((score, name, ts, speaker, text))
        scored.sort(key=lambda item: (-item[0], item[1], item[2]))
        self.last_hits = len(scored)

        lines: list[str] = []
        seen_norms: set[str] = set()
        for score, name, ts, speaker, text in scored[:6]:
            who = "You" if speaker == "You" else "Maupo"
            lines.append(f"[{name} {ts}] {who}: {text[: self.MAX_TURN_SNIPPET]}")
            seen_norms.add(re.sub(r"[^a-z0-9 ]", "", text.lower()).strip()[:120])

        # Semantic pass: moments close in MEANING even with zero word overlap
        # ("that thing about the thing"). Silent no-op when the embedding
        # model is absent - it is a bonus layer, never an error path.
        if self.semantic.available is not False:
            try:
                semantic_hits = self.semantic.search(query, top_k=4)
            except Exception:
                semantic_hits = []
            for name, ts, speaker, text, _score in semantic_hits:
                norm = re.sub(r"[^a-z0-9 ]", "", text.lower()).strip()[:120]
                if norm and norm in seen_norms:
                    continue  # already surfaced by the keyword pass
                who = "You" if speaker == "You" else "Maupo"
                lines.append(f"[{name} {ts}] {who}: {text[: self.MAX_TURN_SNIPPET]} (close in meaning)")
                self.last_hits += 1

        if not lines:
            return ""
        return "\n".join(lines)
