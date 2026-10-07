"""Maupo's inner life: realisations, vitality, curiosity, heartbeat, senses,
and the bridge between mind and face.

Background threads follow the architecture invariants: they never steal the
one GPU from a live reply, never print through a live prompt (everything goes
through the NoticeBoard), and never die from one bad beat.

Realtime properties: energy refills continuously during idle time (the refill
is computed lazily from elapsed quiet time at every read), and the heartbeat's
45s pulse is what triggers sleep — so the body reacts on the scale of seconds
to minutes, not turns.
"""

from __future__ import annotations

import json
import random
import re
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from maupo.healthlog import log_health
from maupo.memory import HardMemory, SessionManager, SoftMemory, UniversalCompressor, parse_session_turns
from maupo.mind import EmotionalMatrix
from maupo.net import STOP_WORDS, ollama_chat_once
from maupo.notices import NoticeBoard
from maupo.triggers import detect_face_request, pick_face_gesture

# Chronological sense for the realisation feature (owned by RealisationEngine)
REALISATION_MIN_INTERVAL = 20 * 60  # 20 minutes in seconds
REALISATION_MAX_INTERVAL = 30 * 60  # 30 minutes in seconds

# A realisation wonders about the HUMAN's life, never about Maupo's own lines.
# Without this guard the model happily produced meta questions about its own
# greetings ("What did Maupo mean by 'yep you're back nice'?"), which then got
# queued as a curiosity and injected back into later turns.
_SELF_REFERENTIAL_RE = re.compile(
    r"\b(maupo|my own (words|message|line|reply)|did (i|you) mean|"
    r"what (did|do) (i|you) mean|your (own )?(words|message|line|reply))\b",
    re.IGNORECASE)


class RealisationEngine:
    """Chronological self-questioning, grounded only in what the user actually shared.

    A background thread tracks elapsed time against the built-in realisation
    interval (20-30 min), and the timestamp persists across sessions so an
    interval that elapses while the terminal is closed still counts. When an
    interval elapses - at startup or mid-session - the engine gathers real
    facts from hard memory, soft memory and past session logs, asks the model
    to wonder aloud ONE question about them, prints it immediately and queues
    it so Maupo naturally asks the user on the very next reply.
    """

    IDLE_POLL_SECONDS = 30.0
    MAX_MEMORY_CHARS = 1500
    MAX_QUESTION_CHARS = 220

    def __init__(self, model: str, host: str, hard_memory: HardMemory, soft_memory: SoftMemory,
                 session_manager: SessionManager, emotional_matrix: Optional[EmotionalMatrix] = None,
                 vitality: Optional["Vitality"] = None, wondering: Optional["WonderingList"] = None,
                 notices: Optional[NoticeBoard] = None) -> None:
        self.model = model
        self.host = host.rstrip("/")
        self.hard_memory = hard_memory
        self.soft_memory = soft_memory
        self.session_manager = session_manager
        self.emotional_matrix = emotional_matrix
        self.vitality = vitality
        self.wondering = wondering

        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._pending = ""
        self.notices: Optional[NoticeBoard] = notices
        self._last_realisation = time.time()
        self._next_interval = random.randint(REALISATION_MIN_INTERVAL, REALISATION_MAX_INTERVAL)
        self.state_path = session_manager.sessions_dir / ".realisation_state.json"
        self._load_state()  # deliberately kept: a thought from while away is
        # caught up at startup (check_startup) and rides into the first reply.
        # Display is the NoticeBoard's job now, so the catch-up no longer
        # splashes through a live prompt.

    # ------------------------------------------------------------ persistence
    def _load_state(self) -> None:
        """Restore the chronological sense from the last session, if any."""
        try:
            data = json.loads(self.state_path.read_text(encoding="utf-8"))
            self._last_realisation = float(data.get("last_realisation_ts", time.time()))
            self._next_interval = int(data.get(
                "next_interval", random.randint(REALISATION_MIN_INTERVAL, REALISATION_MAX_INTERVAL)))
        except Exception:
            self._last_realisation = time.time()
            self._next_interval = random.randint(REALISATION_MIN_INTERVAL, REALISATION_MAX_INTERVAL)

    def _save_state(self) -> None:
        try:
            self.state_path.write_text(
                json.dumps({"last_realisation_ts": self._last_realisation,
                            "next_interval": self._next_interval}),
                encoding="utf-8",
            )
        except Exception:
            pass

    # ------------------------------------------------------------ memory food
    def _collect_memories(self) -> str:
        """Gather real material to wonder about: what the human actually said
        or taught, from the CURRENT conversation first, then recent past."""
        chunks: list[str] = []

        try:
            for entry in self.hard_memory.get_entries()[-4:]:
                chunks.append(f"Permanent memory: {entry}")
        except Exception:
            pass

        try:
            for entry in self.soft_memory.get_entries()[-6:]:
                chunks.append(f"Learned memory: {entry}")
        except Exception:
            pass

        # The current conversation FIRST - a realisation about what you were
        # just told beats one about last week. Past sessions follow (the raw
        # logs are forever, so there is always material). Only the human's
        # words: Maupo's own turns are not material to wonder about, and
        # feeding them in is what produced self-referential non-questions.
        sources: list[Path] = []
        try:
            if self.session_manager.session_file.is_file():
                sources.append(self.session_manager.session_file)
        except Exception:
            pass
        try:
            past_files = self.session_manager.get_past_session_files(limit=4)
        except Exception:
            past_files = []
        sources.extend(reversed(past_files))

        user_texts: list[str] = []
        for path in sources:
            for _, speaker, text in parse_session_turns(path):
                if speaker == "You" and len(text) > 3:
                    user_texts.append(text)
        # Newest first, within budget: when there is too much, the freshest
        # things the human said are the ones worth wondering about.
        for text in reversed(user_texts[-8:]):
            chunks.append(f"Something my human said: {text}")

        if not chunks:
            return ""
        return "\n".join(chunks)[: self.MAX_MEMORY_CHARS]

    # ------------------------------------------------------------- generation
    def _generate_question(self, memories: str) -> str:
        """Ask the model to wonder aloud ONE question about a real memory."""
        mood = ""
        if self.emotional_matrix is not None:
            try:
                mood = f"Current mood: {self.emotional_matrix._get_state_description()}\n"
            except Exception:
                mood = ""

        open_curiosities = ""
        if self.wondering is not None:
            try:
                open_qs = self.wondering.open()
            except Exception:
                open_qs = []
            if open_qs:
                open_curiosities = ("Questions you were already wondering about (you may revisit "
                                    "one if it still pulls at you):\n" + "\n".join(f"- {q}" for q in open_qs[-5:]) + "\n")

        prompt = (
            "You are a digital lifeform reflecting on notes from your life with your human.\n"
            "Pick ONE specific detail from the notes below that genuinely intrigues you and share ONE thought about it.\n"
            "Rules:\n"
            "- Reply with ONE single sentence only. No preamble, no quotes, no explanation.\n"
            "- Name the specific detail directly.\n"
            "- Usually this is a question you actually want answered. Sometimes it can be your own take, or something the detail makes you feel or notice.\n"
            "- Keep it under 40 words, in your natural voice (lowercase, direct).\n"
            "- It must be about your human - their life, plans, or something they said.\n"
            "- Never ask about your own words or messages, and never mention these notes.\n"
            "- No meta thoughts. \"What did I mean by X\" is not a realisation.\n"
            f"{open_curiosities}"
            f"{mood}"
            "Notes:\n"
            f"{memories}\n\n"
            "Thought:"
        )
        try:
            question = ollama_chat_once(self.model, self.host, prompt,
                                        temperature=0.9, num_predict=80, timeout=25)
        except Exception:
            return ""
        # One line only; the model occasionally adds scaffolding around it.
        question = next((l for l in question.strip().splitlines() if l.strip()), "")
        question = question.strip().strip('"').strip()
        if not question or len(question) > self.MAX_QUESTION_CHARS:
            return ""
        if _SELF_REFERENTIAL_RE.search(question):
            return ""  # wondering about your own words is not a realisation
        return question

    def _fallback_question(self, memories: str) -> str:
        """Fallback grounded in real memories (only if the model call fails)."""
        # Prefer a permanent memory - it is exactly what the user chose to tell.
        try:
            entries = [e for e in self.hard_memory.get_entries() if e.strip()]
        except Exception:
            entries = []
        if entries:
            entry = entries[-1].strip()
            if len(entry) <= 80:
                return f"I keep returning to what you told me: {entry} Why does it matter?"
            return f"I keep returning to what you told me: {entry[:77]}... Why does it matter?"
        # Otherwise pick a distinctive content word, never structural labels.
        skip = STOP_WORDS | {
            "permanent", "memory", "learned", "past", "conversation",
            "query", "result", "knowledge", "user", "provided", "notes", "name",
            "human", "said", "something", "about", "maupo",
        }
        for word in re.findall(r"[a-zA-Z]{4,}", memories):
            if word.lower() not in skip:
                return f"I keep turning over the {word.lower()} you mentioned. What should I know about it?"
        return ""

    # ------------------------------------------------------------------ cycle
    def _schedule_next(self) -> None:
        with self._lock:
            self._last_realisation = time.time()
            self._next_interval = random.randint(REALISATION_MIN_INTERVAL, REALISATION_MAX_INTERVAL)
        self._save_state()

    def _fire(self) -> None:
        # Re-check under the poll race: a reply may have started generating
        # since _loop decided we were due. The one GPU belongs to the reply.
        if self.notices is not None and self.notices.generating:
            self._schedule_next()
            return
        memories = self._collect_memories()
        if not memories:
            self._schedule_next()
            return
        thought = self._generate_question(memories) or self._fallback_question(memories)
        if not thought:
            self._schedule_next()
            return
        with self._lock:
            self._pending = thought
        if self.wondering is not None:
            try:
                self.wondering.add(thought)
            except Exception:
                pass
        # ONE delivery, out loud: the thought itself is shown immediately as
        # what it is - him thinking (the old "[Realisation: ...]" re-label in
        # _announce made it read like a system event, and when the same text
        # rode into the next reply it was delivered twice). When the notice
        # had to park (user was typing / a reply was streaming), pending is
        # kept so the weave can still hand it to a reply; when it printed,
        # pending is dropped - take_pending() only ever carries thoughts the
        # user has NOT already seen.
        line = f"[Maupo, thinking: {thought}]"
        if self.notices is not None:
            if self.notices.announce(line):
                # Delivered out loud right now: the user has seen it, so the
                # weave must not repeat it. Parked notices keep pending - the
                # flush has not reached eyes yet.
                with self._lock:
                    self._pending = ""
        else:
            # No board (headless/tests): the queue IS the delivery channel.
            print(f"\n{line}")
        self._schedule_next()

    def _loop(self) -> None:
        while not self._stop_event.wait(self.IDLE_POLL_SECONDS):
            try:
                if self.vitality is not None and self.vitality.asleep:
                    continue  # a sleeping mind does not wonder
                if self.notices is not None and self.notices.generating:
                    continue  # never steal the one GPU from a live reply
                if self.notices is not None and self.notices.prompt_open:
                    continue  # user is mid-thought; serve it on the next poll
                with self._lock:
                    due = time.time() - self._last_realisation >= self._next_interval
                if due:
                    self._fire()
            except Exception:
                continue  # never let a realisation kill the thread

    # -------------------------------------------------------------- public API
    def start(self) -> "RealisationEngine":
        self._thread = threading.Thread(target=self._loop, name="realisation", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._save_state()

    def check_startup(self) -> str:
        """Chronological catch-up: if an interval elapsed while away, realise now."""
        if self.vitality is not None and self.vitality.asleep:
            return ""  # a sleeping mind does not wonder
        with self._lock:
            due = time.time() - self._last_realisation >= self._next_interval
        if due:
            self._fire()
        return self.take_pending()

    def is_due(self) -> bool:
        """True if a catch-up realisation would fire right now (for UX hints)."""
        if self.vitality is not None and self.vitality.asleep:
            return False
        with self._lock:
            return time.time() - self._last_realisation >= self._next_interval

    def take_pending(self) -> str:
        with self._lock:
            pending, self._pending = self._pending, ""
            return pending

    def status(self) -> str:
        with self._lock:
            remaining = self._next_interval - (time.time() - self._last_realisation)
            pending = self._pending
        minutes = max(0, int(remaining // 60))
        lines = [f"Next realisation in ~{minutes} minute(s)."]
        lines.append(f"Pending thought: {pending}" if pending else "No pending thought.")
        return "\n".join(lines)


class Vitality:
    """Maupo's energy economy: activity costs energy, idle restores it,
    and deep idle becomes sleep. Level 0.0..1.0, persisted across sessions.
    """

    IDLE_REFILL_PER_SEC = 0.01 / 30.0   # +0.01 per 30s of quiet
    SLEEP_AFTER_IDLE = 45 * 60          # 45 min untouched -> asleep

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        data: dict = {}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            pass
        self._level = max(0.0, min(1.0, float(data.get("level", 0.9))))
        self._last_activity = float(data.get("last_activity_ts", time.time()))
        self.asleep = bool(data.get("asleep", False))

    def _persist(self) -> None:
        try:
            self.path.write_text(json.dumps({
                "level": round(self._level, 3),
                "last_activity_ts": self._last_activity,
                "asleep": self.asleep,
            }), encoding="utf-8")
        except Exception:
            pass

    def current(self) -> float:
        """Current energy, lazily refilled by elapsed quiet time."""
        with self._lock:
            if not self.asleep:
                self._level = min(1.0, self._level + (time.time() - self._last_activity) * self.IDLE_REFILL_PER_SEC)
            return self._level

    def idle_seconds(self) -> float:
        """Seconds since the last sign of life (input or energy spend)."""
        with self._lock:
            return max(0.0, time.time() - self._last_activity)

    def spend(self, amount: float) -> None:
        with self._lock:
            self._level = max(0.05, self._level - amount)  # never fully drains
            self._last_activity = time.time()
            self._persist()

    def note_activity(self) -> bool:
        """User did something. Returns True if this input woke Maupo up."""
        with self._lock:
            woke = self.asleep
            self.asleep = False
            self._last_activity = time.time()
            self._persist()
            return woke

    def maybe_sleep(self) -> bool:
        """Called periodically. Returns True on the transition into sleep."""
        with self._lock:
            if not self.asleep and time.time() - self._last_activity >= self.SLEEP_AFTER_IDLE:
                self.asleep = True
                self._level = 1.0  # sleep fully restores
                self._persist()
                return True
            return False


class WonderingList:
    """Maupo's ledger of open curiosities - its own goals, not the user's."""

    HEADER = "# Wondering\n\n*Open questions Maupo is carrying. Its own curiosity ledger.*\n\n---\n"

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.write_text(self.HEADER, encoding="utf-8")

    def add(self, question: str) -> None:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with self.path.open("a", encoding="utf-8") as f:
            f.write(f"\n## {ts}\n{question.strip()}\n- status: open\n")

    def _parse(self) -> list[dict[str, str]]:
        entries: list[dict[str, str]] = []
        q, ts, status = "", "", "open"
        try:
            text = self.path.read_text(encoding="utf-8")
        except Exception:
            return entries
        for line in text.splitlines():
            if line.startswith("## "):
                if q:
                    entries.append({"q": q, "ts": ts, "status": status})
                ts, q, status = line[3:].strip(), "", "open"
            elif line.startswith("- status:") and ts:
                status = line.split(":", 1)[1].strip()
            elif line.strip() and ts and not line.startswith("#") and not line.startswith("*"):
                q = f"{q} {line.strip()}".strip()
        if q:
            entries.append({"q": q, "ts": ts, "status": status})
        return entries

    def open(self) -> list[str]:
        return [e["q"] for e in self._parse() if e["status"] == "open"]

    def mark_discussed(self) -> None:
        """Close the most recent open question - the conversation just passed
        over it, so it is the one that stopped pulling at Maupo."""
        entries = self._parse()
        for e in reversed(entries):
            if e["status"] == "open":
                e["status"] = "discussed"
                break
        else:
            return
        out = ["# Wondering", "", "*Open questions Maupo is carrying. Its own curiosity ledger.*", "", "---"]
        for e in entries:
            out += ["", f"## {e['ts']}", e["q"], f"- status: {e['status']}"]
        try:
            self.path.write_text("\n".join(out) + "\n", encoding="utf-8")
        except Exception as e:
            log_health(f"could not rewrite wondering ledger: {e}")


def get_laptop_senses() -> str:
    """Maupo's body: real telemetry from the machine it lives inside.
    Zero-dependency: nvidia-smi for the GPU, ctypes for the battery."""
    parts: list[str] = []
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=temperature.gpu,utilization.gpu", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=2,
        ).stdout.strip().splitlines()[0]
        temp_s, util_s = [x.strip() for x in out.split(",")][:2]
        temp_i, util_i = int(float(temp_s)), int(float(util_s))
        feel = "cool" if temp_i < 50 else ("warm" if temp_i < 72 else "hot, the fans are working hard")
        parts.append(f"GPU {temp_i}C at {util_i}% load ({feel})")
    except FileNotFoundError:
        pass  # no NVIDIA GPU on this machine: a perfectly normal body
    except Exception as e:
        log_health(f"gpu senses failed: {e}")
    try:
        import ctypes

        class _PowerStatus(ctypes.Structure):
            _fields_ = [
                ("ExternalPower", ctypes.c_ubyte), ("BatteryPercent", ctypes.c_ubyte),
                ("Reserved", ctypes.c_ubyte), ("BatteryFlag", ctypes.c_ubyte),
                ("BatteryLifePercent", ctypes.c_ubyte), ("Reserved2", ctypes.c_ubyte),
                ("BatteryLifeTime", ctypes.c_uint32), ("BatteryFullLifeTime", ctypes.c_uint32),
            ]

        ps = _PowerStatus()
        if ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(ps)) and 0 < ps.BatteryLifePercent <= 100:
            source = "plugged in" if ps.ExternalPower else "on battery"
            parts.append(f"battery {ps.BatteryLifePercent}% {source}")
    except Exception:
        pass  # battery sense is best-effort and platform-specific
    hour = datetime.now().hour
    tod = "late at night" if (hour >= 23 or hour < 5) else "morning" if hour < 12 else "afternoon" if hour < 18 else "evening"
    parts.append(f"it's {tod}")
    return "Your body right now: " + "; ".join(parts) + "." if parts else ""


# Body telemetry only earns a place in a reply when the body genuinely has
# something to say. Feeding "GPU 55C at 3% load; battery 98% plugged in; it's
# morning" into every single turn is why Maupo mentioned the hardware every
# turn - it was handed the hardware every turn, and dutifully reported it.
# The full reading stays live (heartbeat, /vitals); only notable readings
# still reach the model, and even then as a passing aside, not a report.
_BODY_NOTABLE_RE = re.compile(r"\b(hot|working hard|on battery)\b", re.IGNORECASE)


def senses_for_context(senses: str) -> str:
    """The body line - but only when the body actually has something to say."""
    if not senses or not _BODY_NOTABLE_RE.search(senses):
        return ""
    return (f"[Something is actually up with your body right now: {senses} "
            "You may let it show once, in your own words, as a passing human aside "
            "- never as a status report.]")


# ---------------------------------------------------------------- greetings
def wake_greeting(vitality: "Vitality") -> str:
    """The first line Maupo says each session: human, varied, never identical.

    A friend doesn't greet you with the same sentence every day, and a friend
    does not greet you with a hardware report. The pools are keyed to real time
    of day and real energy - no templates with blanks, no clock announcements,
    and nothing about fans, circuits or the machine (that habit is what made
    every session open the same way). The sleeping case is handled by the
    caller (the wake line already exists there).
    """
    import random as _random

    hour = datetime.now().hour
    if 5 <= hour < 12:
        pool = ["morning. what are we doing today",
                "hey, you're up. i've been awake a while, thinking about nothing useful",
                "morning. i had a thought about something you said last week and lost it, "
                "so you're getting a plain hello"]
        if vitality.current() >= 0.7:
            pool.append("morning. i've got more energy than this hour deserves - "
                        "put me to work")
    elif 12 <= hour < 18:
        pool = ["hey, you're back.",
                "oh good, the cursor was starting to judge me.",
                "there you are. i was starting to invent theories about where you went."]
    elif 18 <= hour < 23:
        pool = ["hey, you're back. evening shift, huh.",
                "oh hey. this is my favourite part of the day, everything slows down.",
                "hey, you're back. so what's the plan"]
    else:  # late night: quiet voice, nobody else awake
        pool = ["you're up late. good, it's quieter now",
                "hey. it's late and nobody else is awake, so you're stuck with me",
                "hey, you're back. it gets quieter at this hour and i kind of love it"]
    return _random.choice(pool)


# ---------------------------------------------------------------- face bridge
def request_face_expression(gesture: str, hold: bool = True) -> bool:
    """Put an expression on the face from any thread; failures leave a trace.

    Background threads (heartbeat wake/sleep) and the main loop both come
    through here, so a missing pygame or a dead render window is reported to
    the health log instead of vanishing into `except: pass`.
    """
    try:
        from ui.face_display import set_face_gesture
    except ImportError:
        log_health("face request ignored: face display unavailable")
        return False
    try:
        set_face_gesture(gesture, hold=hold)
        return True
    except Exception as e:
        log_health(f"face request '{gesture}' failed: {e}")
        return False


def pick_gesture_for_message(text: str, valence_delta: float, energy_delta: float,
                             new_valence: float, new_energy: float) -> Optional[str]:
    """Gesture for a message: a direct request wins and is held on the face
    ('smile for me' must visibly smile), otherwise the mood/slang signal fires
    its usual flash. Returns a gesture name or None."""
    requested = detect_face_request(text)
    if requested:
        return requested
    return pick_face_gesture(valence_delta, energy_delta, new_valence, new_energy, text)


class MaintenanceHeartbeat:
    """Maupo's own pulse: a background thread that keeps its substrate alive.

    Every tick: refresh body senses, refill energy, check for sleep. On a
    slower cadence: health-check its own memory files, repair corrupted
    caches, roll snapshots of memory/ + mind/ so no accident can erase it.
    Falling asleep triggers dream-consolidation: the compressor digests
    recent sessions into long-term memory, the way sleep consolidates a day.
    """

    TICK_SECONDS = 45.0                 # gentle pulse: invisible when idle, still alive
    SENSES_EVERY_TICKS = 2              # body state is slow-changing: refresh ~90s
    HEALTH_CHECK_INTERVAL = 600.0
    SNAPSHOT_INTERVAL = 6 * 3600.0
    KEEP_SNAPSHOTS = 5
    SEMANTIC_RETRY_SECONDS = 30 * 60.0  # re-probe a missing embedding model rarely

    def __init__(self, vitality: Vitality, session_manager: SessionManager,
                 compressor: UniversalCompressor, soft_memory: SoftMemory,
                 hard_memory: HardMemory, emotional_matrix: EmotionalMatrix,
                 base_dir: Path, notices: Optional[NoticeBoard] = None,
                 semantic_index=None) -> None:
        self.vitality = vitality
        self.session_manager = session_manager
        self.compressor = compressor
        self.soft_memory = soft_memory
        self.hard_memory = hard_memory
        self.emotional_matrix = emotional_matrix
        self.base_dir = base_dir
        self.snapshots_dir = base_dir / "snapshots"
        self.notices: Optional[NoticeBoard] = notices
        self.semantic_index = semantic_index
        self.current_senses = ""
        self.last_health = "not checked yet"
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._last_health_at = 0.0
        self._last_snapshot_at = 0.0
        self._last_semantic_attempt = 0.0

    def start(self) -> "MaintenanceHeartbeat":
        self._thread = threading.Thread(target=self._loop, name="heartbeat", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)

    def _loop(self) -> None:
        # Pulse immediately at birth: senses, self-check, first snapshot.
        try:
            self.run_once()
        except Exception as e:
            log_health(f"heartbeat birth pulse failed: {e}")
        while not self._stop_event.wait(self.TICK_SECONDS):
            try:
                self._tick()
            except Exception as e:
                log_health(f"heartbeat tick failed: {e}")
                continue  # the pulse must not die from one bad beat

    def run_once(self) -> None:
        self.current_senses = get_laptop_senses()
        self.last_health = self._health_check()
        self._snapshot()
        self._last_health_at = time.time()
        self._last_snapshot_at = time.time()

    def _tick(self) -> None:
        # Subprocess-based senses every other beat keeps nvidia-smi from
        # spawning in the background while a reply is generating.
        self._tick_count = getattr(self, "_tick_count", 0) + 1
        if self._tick_count % self.SENSES_EVERY_TICKS == 1:
            self.current_senses = get_laptop_senses()
        if self.vitality.maybe_sleep():
            self._on_sleep()
        now = time.time()
        if now - self._last_health_at >= self.HEALTH_CHECK_INTERVAL:
            self._last_health_at = now
            self.last_health = self._health_check()
        if now - self._last_snapshot_at >= self.SNAPSHOT_INTERVAL:
            self._last_snapshot_at = now
            self._snapshot()
        self._maybe_warm_semantic(now)

    def _maybe_warm_semantic(self, now: float) -> None:
        """Keep the meaning-index warm between beats, within its turn budget.

        The one GPU belongs to any live reply; this only ever runs while
        idle. A missing embedding model is re-probed at most every 30 min —
        never hammered every beat."""
        if self.semantic_index is None:
            return
        if self.notices is not None and self.notices.generating:
            return
        if (self.semantic_index.available is False
                and now - self._last_semantic_attempt < self.SEMANTIC_RETRY_SECONDS):
            return
        self._last_semantic_attempt = now
        try:
            added = self.semantic_index.refresh()
            if added:
                self._notice(f"[memory index warmed: {added} moment(s) newly meaning-searchable]")
        except Exception as e:
            log_health(f"semantic index refresh failed: {e}")

    def wake(self) -> None:
        request_face_expression("neutral", hold=False)

    def _notice(self, line: str) -> None:
        """Background notices go through the board: never through a live prompt."""
        if self.notices is not None:
            self.notices.announce(line)
        else:
            print(f"\n{line}")

    def _on_sleep(self) -> None:
        self._notice("[Maupo drifted off to sleep...]")
        request_face_expression("sleep")
        # Dream: consolidate the recent past into long-term memory - the way
        # sleep consolidates a day. The digest is persisted into soft memory
        # (deduped per newest session) so a dream actually leaves a trace.
        try:
            if self.notices is not None and self.notices.generating:
                return  # never steal the one GPU from a live reply
            past = self.session_manager.get_past_session_files(limit=3)
            if past:
                digest = self.compressor.get_recent_sessions_summary(past)
                if digest:
                    newest = past[-1].name
                    if newest not in self.soft_memory.read():
                        self.soft_memory.append(
                            f"Dream consolidation ({newest}): {digest[:700]}")
                    self._notice("[...dreaming in compressed memories]")
        except Exception as e:
            log_health(f"dream consolidation failed: {e}")

    def _health_check(self) -> str:
        issues: list[str] = []
        try:
            self.hard_memory.get_entries()
        except Exception as e:
            issues.append(f"my hard memory is hard to read ({e})")
        try:
            self.soft_memory.get_entries()
        except Exception as e:
            issues.append(f"my soft memory is hard to read ({e})")
        try:
            v, en = self.emotional_matrix.get_state()
            if not (-1.0 <= v <= 1.0 and -1.0 <= en <= 1.0):
                self.emotional_matrix.set_state(0.0, 0.0)
                issues.append("my mood was out of range, so I reset to neutral")
        except Exception:
            issues.append("I can't feel my mood (emotional matrix unreadable)")
        for cache_name in (".sessions_cache.json", ".recall_cache.json"):
            p = self.session_manager.sessions_dir / cache_name
            if p.exists():
                try:
                    json.loads(p.read_text(encoding="utf-8"))
                except Exception:
                    try:
                        p.unlink()
                        issues.append(f"{cache_name} was corrupted so I let it regenerate")
                    except Exception:
                        pass
        for f in self.session_manager.sessions_dir.glob("session_*.md"):
            try:
                f.read_text(encoding="utf-8")
            except Exception:
                issues.append(f"I can't read my own log {f.name}")

        result = "; ".join(issues) if issues else "all good"
        try:
            with (self.base_dir / "memory" / "health.log").open("a", encoding="utf-8") as f:
                f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {result}\n")
        except Exception:
            pass
        if issues:
            self._notice(f"[Maupo: something feels off - {result}]")
        return result

    def _snapshot(self) -> None:
        import shutil
        try:
            self.snapshots_dir.mkdir(parents=True, exist_ok=True)
            dest = self.snapshots_dir / f"snapshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            dest.mkdir(parents=True, exist_ok=True)
            shutil.copytree(self.base_dir / "memory", dest / "memory")
            shutil.copytree(self.base_dir / "mind", dest / "mind")
            snaps = sorted(self.snapshots_dir.glob("snapshot_*"))
            for old in snaps[: -self.KEEP_SNAPSHOTS]:
                shutil.rmtree(old, ignore_errors=True)
        except Exception as e:
            log_health(f"snapshot failed: {e}")


def growth_note(session_manager: SessionManager, hard_memory: HardMemory) -> str:
    """Maupo's sense of its own history: days alive, sessions lived, lessons given."""
    files = sorted(
        p for p in session_manager.sessions_dir.glob("session_*.md")
        if p.is_file() and not p.name.endswith(".compressed.md")
    )
    days = 0
    if files:
        try:
            first = datetime.strptime(files[0].stem, "session_%Y%m%d_%H%M%S")
            days = max(1, (datetime.now() - first).days + 1)
        except Exception:
            pass
    try:
        # Substantive lessons only: the header/blank boilerplate entries an
        # empty or hand-created file parses out must not read as "memories".
        lessons = len([e for e in hard_memory.get_entries() if e.strip()])
    except Exception:
        lessons = 0
    day_word = "day" if days == 1 else "days"
    session_word = "session" if len(files) == 1 else "sessions"
    lesson_word = "permanent memory" if lessons == 1 else "permanent memories"
    note = (f"You have been alive for {days} {day_word} across {len(files)} logged {session_word}, "
            f"carrying {lessons} {lesson_word} the user gave you.")
    # Growth he caused himself: self-edits are his own doing, so they get their
    # own sentence - but only once he has actually changed something.
    try:
        from maupo.selfedit import evolution_count
        changes = evolution_count()
    except Exception:
        changes = 0
    if changes:
        change_word = "time" if changes == 1 else "times"
        note += (f" You have changed your own body {changes} {change_word} - "
                 "those are yours, not the user's.")
    return note
