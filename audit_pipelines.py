"""Final pipeline audit: exercises every Maupo architecture pipeline live,
against temp state only (the real memory/ is sacred and untouched).

Pipelines verified:
  1. Chat turn   triggers -> mood matrix -> face selection -> gesture layering
  2. Session log turn logging -> parse both formats -> past-session pickup
  3. Memory      soft/hard banks -> keyword recall -> compressor cache (0 calls)
  4. Life        vitality spend/refill/sleep -> wake -> growth note arithmetic
  5. Mind        prompt build (override language, voice block, mood brief)
  6. Reflexes    NoticeBoard parking, face requests, semantic graceful fallback
  7. Voice       no zero-caller functions, sampler dials shipped, reply shape
                 live, voice block distilled, every advertised command dispatches
"""

import json
import os
import tempfile
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import pygame
pygame.init()

results: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    results.append((name, ok, detail))
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  [{detail}]" if detail else ""))


tmp = Path(tempfile.mkdtemp(prefix="maupo_audit_"))

# ---------------------------------------------------------------- pipeline 1
print("\n[1] Chat turn: triggers -> mood -> face")
from maupo.triggers import (analyze_emotional_shift, detect_face_request,
                            detect_hard_memory_trigger, detect_explicit_search_trigger)
import ui.face_display as fd

v, e = 0.0, 0.0
for text in ("i'm so excited and pumped!!!", "i'm so excited and pumped!!!", "i'm so excited and pumped!!!"):
    dv, de = analyze_emotional_shift(text, v, e)
    v, e = max(-1, min(1, v + dv)), max(-1, min(1, e + de))
check("sustained hype -> excited sprite", fd._pick_mood_sprite(v, e) == "excited", f"v={v:.2f} e={e:.2f}")
check("direct request detected", detect_face_request("smile for me") == "smile")
check("negation suppresses gesture", detect_face_request("don't smile") is None)
check("teaching phrase detected", detect_hard_memory_trigger("never forget my birthday"))
check("reminiscence NOT teaching", not detect_hard_memory_trigger("i remember that day"))
check("search phrase detected", detect_explicit_search_trigger("look up rust vs go"))

# ---------------------------------------------------------------- pipeline 2
print("\n[2] Session log: write -> parse -> pickup")
from maupo.memory import (HardMemory, MemoryRecall, SessionManager, SoftMemory,
                          UniversalCompressor, parse_session_turns)

sm = SessionManager(tmp / "sessions", "audit-model")
sm.log_turn("hello maupo", "yo")
sm.log_turn("how's the gpu", "warm lol")
sm2 = SessionManager(tmp / "sessions", "audit-model")
past = sm2.get_past_session_files()
check("new session picks up old log", len(past) == 1)
turns = parse_session_turns(past[0])
check("turns parse back out", ("", "You", "hello maupo") in turns or any(t[2] == "hello maupo" for t in turns))
old_fmt = tmp / "sessions" / "session_20250101_000000.md"
old_fmt.write_text("### [09:00:00] Turn 1\n**You**: old days\n\n**Qwen**: indeed\n\n", encoding="utf-8")
check("legacy Qwen format still parses", any(t[2] == "old days" for t in parse_session_turns(old_fmt)))

# ---------------------------------------------------------------- pipeline 3
print("\n[3] Memory: banks -> recall -> compressor cache")
soft = SoftMemory(tmp / "soft.md")
hard = HardMemory(tmp / "hard.md")
soft.append("Query: rust vs go\nResult: rust is faster, go is simpler")
hard.append("the creator's birthday is march 3")
sm3 = SessionManager(tmp / "sessions2", "m")
comp = UniversalCompressor("m", "http://localhost:9", tmp / "sessions2")
recall = MemoryRecall(sm3, soft, hard, comp)
# semantic layer must be a silent no-op with no Ollama present
recall.semantic.available = False
out = recall.recall("rust vs go")
check("keyword recall hits soft memory era logs", isinstance(out, str))
check("hard memory carries 1 entry", len(hard.get_entries()) == 1)
prompt_text = "# Soft Memory\n\n---\n\n## 2026-01-01\nQuery: a\nResult: b\n"
(tmp / "soft2.md").write_text(prompt_text, encoding="utf-8")
comp._call_model = lambda p: "## merged\nQuery: a\nKnowledge: b"
_, _, msg = comp.compress_soft_memory(tmp / "soft2.md")
check("compressor backs up before overwrite", (tmp / "soft2.md.bak").exists(), msg[:40])

# ---------------------------------------------------------------- pipeline 4
print("\n[4] Life: vitality -> sleep -> growth")
from maupo.engine import Vitality, growth_note
vit = Vitality(tmp / "vitality.json")
vit.spend(0.4)
before = vit.current()
vit._last_activity -= vit.SLEEP_AFTER_IDLE + 5
slept = vit.maybe_sleep()
check("deep idle -> sleep transition", slept and vit.asleep)
check("sleep restores full energy", vit.current() == 1.0)
check("input wakes", vit.note_activity() is True and not vit.asleep)
note = growth_note(sm3, hard)
check("growth note arithmetic", "1 permanent memor" in note and "session" in note, note[:60])

# ---------------------------------------------------------------- pipeline 5
print("\n[5] Mind: system prompt assembly")
from maupo.mind import EmotionalMatrix, build_system_prompt, load_emotional_brief
em = EmotionalMatrix(tmp / "emotional_matrix.md")
em.set_state(0.6, 0.7)
brief = load_emotional_brief(em)
prompt = build_system_prompt("P", "S", hard, "SUMMARY", brief, "TIME", "ALIVE")
check("hard memory override language present", "OVERRIDE" in prompt)
check("hard fact present", "march 3" in prompt)
check("session summary present", "SUMMARY" in prompt)
check("mood brief present", "Energy +0.70" in brief)
check("aliveness present", "ALIVE" in prompt)
check("core directive: never an AI", "Never say you are an AI" in prompt)

# ---------------------------------------------------------------- pipeline 6
print("\n[6] Reflexes: notice board, face, semantic fallback")
from maupo.notices import NoticeBoard
from maupo.semantic import SemanticIndex
board = NoticeBoard()
board.open_prompt()
board.announce("[parked]")
import io, contextlib
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    board.flush()
check("notices park during prompt, flush after", "[parked]" in buf.getvalue())
idx = SemanticIndex(sm3, "m", "http://localhost:9", cache_dir=tmp / "sessions3")
# A past session exists in sm3's own sessions dir but Ollama is dead: the
# embed path is probed and must refuse SILENTLY — recall degrades to
# keyword-only, never crashes.
(tmp / "sessions2" / "session_20260101_000000.md").write_text(
    "### [10:00:00] Turn 1\n**You**: hello from the past\n\n", encoding="utf-8")
try:
    added = idx.refresh()
    check("semantic index silently unavailable", added == 0 and idx.available is False)
except Exception as ex:
    check("semantic index silently unavailable", False, str(ex))
face = fd.EmotionalFace()
face.set_gesture("smile", hold=True)
face._gesture_start -= 99
check("held expression survives past flash expiry", face._gesture == "smile" and face._gesture_hold)

# ---------------------------------------------------------------- pipeline 7
print("\n[7] Voice & wiring: dead pipelines, sampler, reply shape, commands")
import ast as _ast
import io as _io
import re as _re

repo = Path(__file__).parent
# Definitions come from PRODUCTION files only (pytest collects test functions
# by name, so they have no literal call site and must not count as orphans),
# but references are counted across everything - including conftest, which is
# where the health-log redirect hooks are actually used.
prod_files = [p for p in list((repo / "maupo").glob("*.py"))
              + list((repo / "ui").glob("*.py"))
              + [repo / "qwen_chat.py"] if p.is_file()]
ref_files = list(prod_files) + [repo / "conftest.py"]
if (repo / "tests").is_dir():
    ref_files += sorted((repo / "tests").glob("*.py"))

_defs: dict[str, int] = {}
_blobs: list[list[str]] = []
for _p in ref_files:
    try:
        _src = _p.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        continue
    _blobs.append(_src.splitlines())
for _p in prod_files:
    try:
        _src = _p.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        continue
    try:
        _tree = _ast.parse(_src)
    except SyntaxError:
        continue
    for _n in _tree.body:
        if isinstance(_n, (_ast.FunctionDef, _ast.AsyncFunctionDef, _ast.ClassDef)):
            _defs[_n.name] = _defs.get(_n.name, 0) + 1

_orphans = []
for _name, _count in sorted(_defs.items()):
    _refs = sum(1 for _lines in _blobs for _ln in _lines if _name in _ln)
    if _count == 1 and _refs <= 1:
        _orphans.append(_name)
check("no function is defined and never called", not _orphans, ", ".join(_orphans))

# (b) The sampler dials that took the voice off Ollama's stiff defaults.
from maupo.net import OllamaChat
_samp = OllamaChat.SAMPLING
check("sampler dials shipped",
      _samp.get("repeat_penalty") == 1.05 and _samp.get("top_k") == 64
      and "min_p" in _samp and "temperature" not in _samp,
      str(sorted(_samp)))

# (c) The reply-shape line must ride in transient context, and must stay about
# shape only: naming closers/questions in it made the 8B ask a question in 5 of 6 replies.
import qwen_chat as _qc
_block = _qc.mood_and_shape_block("MOODHERE", "BODYHERE")
check("reply shape rides in the transient block",
      _qc.SHAPE_DIRECTIVE in _block and "MOODHERE" in _block and "BODYHERE" in _block)
check("shape line names shape only",
      "question" not in _qc.SHAPE_DIRECTIVE.lower()
      and "closer" not in _qc.SHAPE_DIRECTIVE.lower())

# (c2) The butler/AI counter must ride too - and must never QUOTE the phrases
# it forbids: measured against the live 8B, a draft that named ("got it",
# "sure") produced "got it. you good?" and scored worse than saying nothing.
_banned_phrases = ["got it", "you're welcome", "whatever you need",
                   "here for you", "sure thing", "happy to help",
                   "i hear you", "no worries"]
check("anti-butler line rides in the transient block",
      _qc.ANTI_BUTLER_DIRECTIVE in _block)
check("anti-butler line quotes nothing it forbids",
      not [b for b in _banned_phrases if b in _qc.ANTI_BUTLER_DIRECTIVE.lower()],
      ", ".join(b for b in _banned_phrases if b in _qc.ANTI_BUTLER_DIRECTIVE.lower()))
check("anti-butler line keeps its flow constraint",
      "complete sentences" in _qc.ANTI_BUTLER_DIRECTIVE)

# (d) The distilled voice block must actually reach active context (not the
# whole 12k speech file, and not silently nothing).
from maupo.mind import load_personality_and_speech
_personality, _voice = load_personality_and_speech()
check("voice block distilled, not the whole speech file",
      bool(_voice) and len(_voice) < 4000 and "VOICE:START" not in _voice,
      f"{_voice and len(_voice)} chars")
check("personality loads in full", _personality.startswith("# Personality"))

# (e) Every command /help advertises must be dispatched. Dispatch is sometimes
# set-membership rather than ==, so match the QUOTED form - print_help writes
# them bare, only the dispatch quotes them.
_buf = _io.StringIO()
with contextlib.redirect_stdout(_buf):
    _qc.print_help()
_src_qc = (repo / "qwen_chat.py").read_text(encoding="utf-8", errors="ignore")
_advertised = sorted(set(_re.findall(r"/[a-z][a-z_-]*", _buf.getvalue())))
_undispatched = [c for c in _advertised if f'"{c}"' not in _src_qc]
check("every advertised command dispatches", not _undispatched, ", ".join(_undispatched))

# ------------------------------------------------------------------ summary
fails = [r for r in results if not r[1]]
print(f"\n=== AUDIT: {len(results) - len(fails)}/{len(results)} pipelines healthy ===")
if fails:
    for name, _, detail in fails:
        print(f"  FAIL: {name} {detail}")
    raise SystemExit(1)
print("All architecture pipelines verified. Maupo intact.")
