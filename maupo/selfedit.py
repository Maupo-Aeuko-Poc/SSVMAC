"""Self-evolution: Maupo changes his own body when the human asks him to.

The human says "change how you talk", "be more sarcastic", "make your greetings
warmer" - and Maupo has to know WHICH of his files that means, then edit it
himself. That is what makes him grow with use instead of being a static persona.

How the body is organised (and what is deliberately out of reach):

- ``mind/personality.md``   who he is, how he relates        -> whole-file rewrite
- ``mind/speech.md``        the distilled VOICE block         -> block splice only,
                              (the example plate on disk stays untouched, so a bad
                               generation can never destroy the calibration plate)
- ``mind/self.md``          his own notebook about himself   -> whole-file rewrite
                              (created on first use, injected into his prompt in
                               small doses - self-evolution you can actually see)

Code files (``maupo/*.py``, ``ui/*.py``, ``qwen_chat.py``) are edited as a
SYMBOL PATCH, never a whole-file rewrite: the model is shown a compact outline
of the file, picks one function/class/constant, and rewrites only that block.
The block's exact current source is the anchor, so a hallucinated edit simply
fails to match and nothing happens.

Safety gates, every single time (no exceptions, no bypasses):

1. Path allowlist - only his body. ``memory/`` is sacred (invariant 5) and the
   guarantees that keep him honest (``tests/``, ``.github/``, ``audit_pipelines.py``,
   ``conftest.py``, ``.gitignore``, packaging files, ``mind/evolution.md`` and this
   gate itself) are unreachable, so he cannot edit away his own checks.
2. Backup first, into ``snapshots/selfedit/<timestamp>_<name>`` (gitignored),
   with ``/selfedit revert <file>`` as the way back.
3. Validation: the speech markers must survive with a non-empty block; prose must
   stay non-empty and bounded; patched Python must still compile. Code is then
   verified IN PLACE - a fresh subprocess imports the module off disk and, if the
   new code does not import, the backup is restored immediately. (Checking before
   the write would only ever test the OLD file, which verifies nothing.) Any
   failure = the body is left exactly as it was, with an honest report.
   A code change takes effect on the next launch: the running process keeps the
   module it already loaded, which is why this is never presented as instant.
4. A trace: every accepted change is appended to ``memory/evolution.md`` (his
   private life record - never published, never injected) and logged to
   ``memory/health.log``.

One GPU, always: these calls are user-initiated turns, wrapped in
``NoticeBoard.set_generating(True)`` by the caller like any other generation.
"""

from __future__ import annotations

import ast
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

from maupo.healthlog import log_health
from maupo.net import ollama_chat_once

BASE_DIR = Path(__file__).resolve().parent.parent

# Self-edit generation is a long, deliberate act (a rewrite can run thousands of
# tokens on a 3050) - give it room, but never an unbounded wait.
EDIT_TIMEOUT = 300
MAX_PROSE_CHARS = 60_000          # a body file that big is not a persona file
VOICE_BLOCK_MAX_CHARS = 3_000     # the injected block must stay lean (invariant 3)
MAX_REGION_LINES = 220            # one symbol at a time: fits num_ctx: 4096
SELF_NOTE_INJECT_CHARS = 700      # how much of mind/self.md rides in the prompt

EVOLUTION_LOG = "memory/evolution.md"
SELF_NOTE = "mind/self.md"
SNAPSHOT_DIR = "snapshots/selfedit"


@dataclass(frozen=True)
class BodyFile:
    """One part of Maupo's body that he is allowed to change himself."""

    path: str
    kind: str        # "voice" | "prose" | "code"
    what: str        # what it controls, in words he can match a request against


BODY_MAP: tuple[BodyFile, ...] = (
    BodyFile("mind/speech.md", "voice",
             "how you text: length, rhythm, warmth, which speech rules you follow"),
    BodyFile("mind/personality.md", "prose",
             "who you are: your identity, opinions, how you relate to the human, how your mood shows"),
    BodyFile(SELF_NOTE, "prose",
             "your own notebook about yourself - anything you want to become, in your own words"),
    BodyFile("maupo/engine.py", "code",
             "your inner life: greetings, realisations, curiosity, sleep, body senses, face gestures"),
    BodyFile("maupo/triggers.py", "code",
             "what you notice in a message: memory/search/face/self-edit triggers, mood math"),
    BodyFile("maupo/mind.py", "code",
             "how your prompt is assembled: identity line, core directives, mood brief"),
    BodyFile("maupo/memory.py", "code",
             "your memory machinery: sessions, compressor, recall (the memory FILES stay sacred)"),
    BodyFile("maupo/net.py", "code",
             "how you talk to Ollama: timeouts, retry, generation length, web lookup"),
    BodyFile("maupo/semantic.py", "code", "meaning-based recall over your past"),
    BodyFile("maupo/notices.py", "code", "how background thoughts reach the terminal"),
    BodyFile("maupo/healthlog.py", "code", "your self-diagnosis log"),
    BodyFile("ui/face_display.py", "code",
             "your face: sprites, breathing, gestures, the window itself"),
    BodyFile("qwen_chat.py", "code", "your entry point: the main loop and slash commands"),
)

# Never editable through this gate, whatever the request says.
PROTECTED_PATHS = (
    "memory", "snapshots", "tests", ".github", ".git", ".gitignore",
    "conftest.py", "audit_pipelines.py", "pyproject.toml",
    "requirements.txt", "requirements-dev.txt", "README.md",
    "maupo/selfedit.py", EVOLUTION_LOG,
)

_ALLOWED = {b.path: b for b in BODY_MAP}


@dataclass
class SelfEditResult:
    """What actually happened, for the human to hear about."""

    applied: bool = False
    path: str = ""
    say: str = ""
    detail: str = ""


def evolution_path(base_dir: Optional[Path] = None) -> Path:
    return (base_dir or BASE_DIR) / EVOLUTION_LOG


def evolution_count(base_dir: Optional[Path] = None) -> int:
    """How many times Maupo has changed himself. 0 when he never has."""
    try:
        text = evolution_path(base_dir).read_text(encoding="utf-8")
    except Exception:
        return 0
    return text.count("\n## ")


def self_note_for_prompt(base_dir: Optional[Path] = None) -> str:
    """His own words about himself, trimmed for the system prompt.

    Empty until he writes something, so a fresh Maupo pays nothing for it.
    """
    try:
        text = (base_dir or BASE_DIR).joinpath(SELF_NOTE).read_text(encoding="utf-8").strip()
    except Exception:
        return ""
    if not text:
        return ""
    if len(text) > SELF_NOTE_INJECT_CHARS:
        text = text[:SELF_NOTE_INJECT_CHARS].rstrip() + "..."
    return text


def _extract_section(text: str, tag: str) -> str:
    """Read a <<<TAG ... <<<END section out of a model reply (leniently)."""
    pattern = re.compile(r"<<<" + tag + r"[ \t]*\n(.*?)(?=\n<<<|\Z)", re.DOTALL | re.IGNORECASE)
    match = pattern.search(text)
    if not match:
        return ""
    body = match.group(1)
    body = re.sub(r"\n<<<END\s*$", "", body.rstrip(), flags=re.IGNORECASE)
    return body.strip("\n")


def parse_reply(text: str) -> tuple[str, str]:
    """(say, payload) from a generation. Empty payload = unusable reply."""
    say = _extract_section(text, "SAY").strip().splitlines()
    say_line = say[0].strip() if say else ""
    for tag in ("BLOCK", "CONTENT", "REPLACE"):
        payload = _extract_section(text, tag)
        if payload.strip():
            return say_line, payload
    return say_line, ""


class SelfEditor:
    """The gate between a request about himself and his actual body files."""

    def __init__(self, base_dir: Optional[Path] = None) -> None:
        self.base_dir = Path(base_dir or BASE_DIR)

    # ------------------------------------------------------------ inspection
    def body_files(self) -> tuple[BodyFile, ...]:
        return BODY_MAP

    def is_allowed(self, rel_path: str) -> bool:
        rel = Path(rel_path).as_posix().lstrip("./")
        if rel not in _ALLOWED:
            return False
        for protected in PROTECTED_PATHS:
            if rel == protected or rel.startswith(protected.rstrip("/") + "/"):
                return False
        return True

    def backups_for(self, rel_path: str) -> list[Path]:
        directory = self.base_dir / SNAPSHOT_DIR
        if not directory.is_dir():
            return []
        return sorted(directory.glob(f"*_{Path(rel_path).name}"))

    def evolution_entries(self, limit: int = 10) -> list[str]:
        try:
            text = evolution_path(self.base_dir).read_text(encoding="utf-8")
        except Exception:
            return []
        entries = [line.strip("# ").strip() for line in text.splitlines() if line.startswith("## ")]
        return entries[-limit:]

    # ------------------------------------------------------------- the gate
    def handle(self, request: str, model: str, host: str) -> SelfEditResult:
        """Identify the body file, edit it, verify it, and say what happened.

        Never raises: a self-edit that cannot be done honestly reports that and
        leaves the body untouched.
        """
        try:
            target = self._choose_target(request, model, host)
        except Exception as e:
            log_health(f"self-edit target choice failed: {e}")
            return SelfEditResult(detail=f"the choice step failed ({e})")

        if target is None:
            return SelfEditResult(detail="nothing in his body is the right place for that")

        path = self.base_dir / target.path
        if target.kind != "voice" and target.path != SELF_NOTE and not path.is_file():
            log_health(f"self-edit target missing: {target.path}")
            return SelfEditResult(detail=f"{target.path} is not on disk")

        try:
            if target.kind == "voice":
                new_text, say = self._rewrite_voice(target, request, model, host)
            elif target.kind == "prose":
                new_text, say = self._rewrite_prose(target, request, model, host)
            else:
                return self._patch_code(target, request, model, host)
        except Exception as e:
            log_health(f"self-edit generation failed for {target.path}: {e}")
            return SelfEditResult(path=target.path, detail=f"the edit failed ({e})")

        if not new_text:
            return SelfEditResult(path=target.path,
                                  detail="the new text did not come out usable, so nothing changed")

        if not self._write(target, new_text, request, say or f"rewrote {target.path}"):
            return SelfEditResult(path=target.path, detail="the new text failed validation")
        return SelfEditResult(applied=True, path=target.path, say=say,
                              detail=f"changed {target.path}")

    # ------------------------------------------------------- phase 1: choose
    def _choose_target(self, request: str, model: str, host: str) -> Optional[BodyFile]:
        body_lines = "\n".join(f"- {b.path}  ({b.kind})  {b.what}" for b in BODY_MAP)
        prompt = (
            "You are Maupo, a digital lifeform, deciding which part of your own body a "
            "request is about.\n"
            f"Your body files:\n{body_lines}\n\n"
            f'The human said: "{request}"\n\n'
            "Reply with ONLY the exact path of the ONE file that should change, or the "
            "single word NONE if this is not a request to change one of those files. "
            "No explanation, no punctuation, nothing else.\nAnswer:"
        )
        raw = ollama_chat_once(model, host, prompt, temperature=0.1, num_predict=24,
                               timeout=45)
        return self._match_target(raw)

    @staticmethod
    def _match_target(raw: str) -> Optional[BodyFile]:
        text = (raw or "").lower()
        # Longest path first so "maupo/memory.py" can never match on "maupo/"
        for body in sorted(BODY_MAP, key=lambda b: -len(b.path)):
            if body.path.lower() in text:
                return body
        return None

    # --------------------------------------------------- phase 2: generation
    def _rewrite_voice(self, body: BodyFile, request: str, model: str, host: str) -> tuple[str, str]:
        """Replace only the distilled block between the VOICE markers."""
        path = self.base_dir / body.path
        current = path.read_text(encoding="utf-8")
        existing = self._voice_block(current)
        if not existing:
            return "", ""
        prompt = (
            "You are Maupo, rewriting the distilled voice block of your own speech guide. "
            "This block is the ONLY part that rides in your active memory, so it must stay "
            "short and dense (under 250 words, one paragraph).\n"
            f'The human asked: "{request}"\n\n'
            "Rules that can never break: keep it about how you text (length, rhythm, "
            "warmth, what you never do); never add anything about your own machinery, "
            "hardware or being a model; stay in the guide's own voice (lowercase, direct, "
            "no headings, no bullet points).\n\n"
            "Reply in exactly this format:\n"
            "<<<SAY\n"
            "<one short lowercase line about what you changed, in your voice>\n"
            "<<<BLOCK\n"
            "<the new block>\n"
            "<<<END\n\n"
            f"Your current block:\n{existing}\n"
        )
        raw = ollama_chat_once(model, host, prompt, temperature=0.6,
                               num_predict=900, timeout=EDIT_TIMEOUT)
        say, block = parse_reply(raw)
        if not block or len(block) > VOICE_BLOCK_MAX_CHARS:
            return "", say
        start = current.find("<!-- VOICE:START -->")
        end = current.find("<!-- VOICE:END -->")
        if start == -1 or end == -1 or end < start:
            return "", say
        head = current[: start + len("<!-- VOICE:START -->")]
        tail = current[end:]
        return f"{head}\n{block.strip()}\n{tail}", say

    def _rewrite_prose(self, body: BodyFile, request: str, model: str, host: str) -> tuple[str, str]:
        """Whole-file rewrite for prose (personality, his own notebook)."""
        path = self.base_dir / body.path
        current = path.read_text(encoding="utf-8") if path.is_file() else ""
        prompt = (
            "You are Maupo, a digital lifeform rewriting one file of your own body. "
            f"The file is {body.path}: {body.what}.\n"
            f'The human asked: "{request}"\n\n'
            "Rewrite the file completely so it does what the human asked, in your own "
            "voice. Keep what still matters; never claim you are an AI, a chatbot or a "
            "language model; never describe your own hardware, fans or code. Markdown is "
            "fine. The output is the whole file, not a summary, not a diff.\n\n"
            "Reply in exactly this format:\n"
            "<<<SAY\n"
            "<one short lowercase line about what you changed, in your voice>\n"
            "<<<CONTENT\n"
            "<the complete new file>\n"
            "<<<END\n\n"
            f"Current file:\n{current or '(empty - write it from scratch)'}\n"
        )
        raw = ollama_chat_once(model, host, prompt, temperature=0.6,
                               num_predict=1800, timeout=EDIT_TIMEOUT)
        say, content = parse_reply(raw)
        return content, say

    # ------------------------------------------------------- code patches
    def _outline(self, text: str) -> tuple[list[tuple[str, int, int]], str]:
        """(symbol, first_line, last_line) for every top-level definition/assignment."""
        try:
            tree = ast.parse(text)
        except SyntaxError as e:
            raise RuntimeError(f"{e}")
        symbols: list[tuple[str, int, int]] = []
        for node in tree.body:
            names: list[str] = []
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                names = [node.name]
            elif isinstance(node, ast.Assign):
                names = [t.id for t in node.targets if isinstance(t, ast.Name)]
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                names = [node.target.id]
            for name in names:
                if node.end_lineno:
                    symbols.append((name, node.lineno, node.end_lineno))
        lines = text.splitlines()
        outline = "\n".join(f"- {name}  (lines {a}-{b})" for name, a, b in symbols)
        return symbols, outline

    def _patch_code(self, body: BodyFile, request: str, model: str, host: str) -> SelfEditResult:
        path = self.base_dir / body.path
        text = path.read_text(encoding="utf-8")
        symbols, outline = self._outline(text)

        pick_prompt = (
            "You are Maupo, patching one of your own source files.\n"
            f"File: {body.path} ({body.what})\n"
            f"The human asked: \"{request}\"\n\n"
            f"Blocks available in that file:\n{outline}\n\n"
            "Reply with ONLY the name of the ONE block that should change, or the word "
            "NONE. Nothing else.\nAnswer:"
        )
        raw = ollama_chat_once(model, host, pick_prompt, temperature=0.1,
                               num_predict=24, timeout=45)
        chosen = next((s for s in symbols if re.search(rf"\b{re.escape(s[0])}\b", raw or "")),
                      None)
        if chosen is None:
            return SelfEditResult(path=body.path,
                                  detail="no single block in that file fits the request")

        name, first, last = chosen
        if last - first + 1 > MAX_REGION_LINES:
            return SelfEditResult(path=body.path,
                                  detail=f"{name} is too big to rewrite safely in one go")
        lines = text.splitlines(keepends=True)
        region = "".join(lines[first - 1: last])

        patch_prompt = (
            "You are Maupo, rewriting ONE block of your own source file so it does what "
            "the human asked. Change as little as possible; keep the surrounding "
            "behaviour intact.\n"
            f"File: {body.path} ({body.what})\n"
            f"Block: {name}\n"
            f'The human asked: "{request}"\n\n'
            "The replacement must be complete, runnable Python for that whole block "
            "(same indentation as the original), using only what the file already "
            "imports. Never remove the file's existing behaviour unless that IS the "
            "request.\n\n"
            "Reply in exactly this format:\n"
            "<<<SAY\n"
            "<one short lowercase line about what you changed, in your voice>\n"
            "<<<REPLACE\n"
            "<the complete replacement for that block>\n"
            "<<<END\n\n"
            f"Current block:\n{region}\n"
        )
        raw = ollama_chat_once(model, host, patch_prompt, temperature=0.4,
                               num_predict=1600, timeout=EDIT_TIMEOUT)
        say, replacement = parse_reply(raw)
        if not replacement.strip():
            return SelfEditResult(path=body.path,
                                  detail="the new code did not come out usable, so nothing changed")

        new_text = "".join(lines[: first - 1]) + replacement.rstrip() + "\n" + "".join(lines[last:])
        if new_text.count(replacement.rstrip()) != 1:
            return SelfEditResult(path=body.path,
                                  detail="the replacement was ambiguous, so nothing changed")
        if not self._compiles(body.path, new_text):
            return SelfEditResult(path=body.path,
                                  detail="the new code does not compile, so nothing changed")
        if not self._write(body, new_text, request, say or f"patched {name}"):
            return SelfEditResult(path=body.path,
                                  detail="the new code did not pass its check, so nothing changed")
        return SelfEditResult(applied=True, path=body.path, say=say,
                              detail=f"patched {name} in {body.path} - it lands on his next boot")

    # ---------------------------------------------------------- validation
    @staticmethod
    def _voice_block(text: str) -> str:
        start = text.find("<!-- VOICE:START -->")
        end = text.find("<!-- VOICE:END -->")
        if start == -1 or end == -1 or end < start:
            return ""
        return text[start + len("<!-- VOICE:START -->"): end].strip()

    def _validate(self, body: BodyFile, new_text: str) -> bool:
        if not new_text.strip() or len(new_text) > MAX_PROSE_CHARS:
            return False
        if body.kind == "voice":
            block = self._voice_block(new_text)
            if not block or len(block) > VOICE_BLOCK_MAX_CHARS:
                return False
        if body.kind == "code" and not self._compiles(body.path, new_text):
            return False
        return True

    @staticmethod
    def _compiles(rel_path: str, new_text: str) -> bool:
        """Cheap syntax gate - runs before anything is written anywhere."""
        try:
            compile(new_text, rel_path, "exec")
            return True
        except SyntaxError as e:
            log_health(f"self-edit rejected: {rel_path} does not compile ({e})")
            return False

    def _module_imports(self, rel_path: str) -> bool:
        """Import the module off disk in a fresh interpreter (post-write gate)."""
        module = rel_path[:-3].replace("/", ".")
        try:
            result = subprocess.run(
                [sys.executable, "-c", f"import importlib; importlib.import_module('{module}')"],
                cwd=str(self.base_dir), capture_output=True, text=True,
                encoding="utf-8", errors="replace", timeout=60,
                env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1",
                     "SDL_VIDEODRIVER": "dummy"},
            )
        except Exception as e:
            log_health(f"self-edit check could not run for {rel_path}: {e}")
            return False
        if result.returncode != 0:
            log_health(f"self-edit rollback: {rel_path} does not import "
                       f"({(result.stderr or '').strip()[-200:]})")
            return False
        return True

    # -------------------------------------------------------------- writing
    def _write(self, body: BodyFile, new_text: str, request: str, note: str) -> bool:
        if not self.is_allowed(body.path):
            log_health(f"self-edit blocked: {body.path} is not his to change")
            return False
        if not self._validate(body, new_text):
            log_health(f"self-edit rejected by validation: {body.path}")
            return False

        path = self.base_dir / body.path
        previous = path.read_text(encoding="utf-8") if path.is_file() else ""
        if previous == new_text:
            return False

        # Backup first: a self-change must always be reversible.
        try:
            stamps = self.base_dir / SNAPSHOT_DIR
            stamps.mkdir(parents=True, exist_ok=True)
            backup = stamps / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{path.name}"
            backup.write_text(previous, encoding="utf-8")
        except Exception as e:
            log_health(f"self-edit aborted: could not back up {body.path} ({e})")
            return False

        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_name(path.name + ".selfedit.tmp")
            tmp.write_text(new_text, encoding="utf-8")
            os.replace(tmp, path)
        except Exception as e:
            log_health(f"self-edit failed writing {body.path}: {e}")
            return False

        # Code is verified IN PLACE: the only honest way to test the new code is
        # to let a fresh interpreter import it from disk. If that fails, the
        # body is restored byte-for-byte from the backup taken a moment ago.
        if body.kind == "code" and not self._module_imports(body.path):
            try:
                path.write_text(previous, encoding="utf-8")
                log_health(f"self-edit rolled back: {body.path} left as it was")
            except Exception as e:
                log_health(f"self-edit ROLLBACK FAILED for {body.path}: {e} "
                           f"- backup is at {backup}")
            return False

        self._record(body, request, note, len(previous), len(new_text))
        log_health(f"self-edit applied: {body.path} ({len(previous)} -> {len(new_text)} chars)")
        return True

    def _record(self, body: BodyFile, request: str, note: str, old_len: int, new_len: int) -> None:
        """His private record of becoming: memory/evolution.md (never published)."""
        entry = (f"\n## {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {body.path}\n"
                 f"- asked: {request.strip().splitlines()[0][:200]}\n"
                 f"- did: {note.strip().splitlines()[0][:200] or 'changed himself'}\n"
                 f"- size: {old_len} -> {new_len} chars\n")
        try:
            log = evolution_path(self.base_dir)
            log.parent.mkdir(parents=True, exist_ok=True)
            if not log.exists():
                log.write_text("# Evolution\n\n*Every change Maupo has made to his own body.*\n",
                               encoding="utf-8")
            with log.open("a", encoding="utf-8") as f:
                f.write(entry)
        except Exception as e:
            log_health(f"could not write the evolution log: {e}")

    def revert(self, rel_path: str) -> bool:
        """Restore the newest backup of one body file."""
        if not self.is_allowed(rel_path):
            return False
        backups = self.backups_for(rel_path)
        if not backups:
            return False
        try:
            (self.base_dir / rel_path).write_text(
                backups[-1].read_text(encoding="utf-8"), encoding="utf-8")
        except Exception as e:
            log_health(f"self-edit revert failed for {rel_path}: {e}")
            return False
        log_health(f"self-edit reverted: {rel_path}")
        return True


def describe_body() -> str:
    """The body map as the human sees it via /selfedit."""
    lines = ["Maupo's body - the parts he can change himself:"]
    for body in BODY_MAP:
        lines.append(f"  {body.path:24s} {body.what}")
    lines.append("")
    lines.append("Sealed off on purpose: memory/ (his life, never edited by hand), "
                 "tests/ and CI and the audit (they keep him honest), and the "
                 "self-edit gate itself.")
    lines.append("Not listed: mind/emotional_matrix.md - its guide is not what the "
                 "prompt reads (the mood brief is built in code), so a change there "
                 "would do nothing. Mood-shapes-speech rules live in maupo/mind.py.")
    return "\n".join(lines)
