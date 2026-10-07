#!/usr/bin/env python3
"""Local command-line chat interface for the Qwen model hosted by Ollama.

Maupo — SSVMAC entry point. The implementation lives in the `maupo` package:

- maupo.net       Ollama client, one-shot calls, web lookup
- maupo.triggers  phrase detection (memory, search, face requests) + mood math
- maupo.notices   the one doorway for background-thread terminal notices
- maupo.memory    sessions, compressor, soft/hard memory, deep recall
- maupo.mind      emotional matrix, persona/speech loading, system prompt
- maupo.engine    realisations, vitality, wondering, heartbeat, face bridge
- maupo.healthlog shared self-diagnosis log (memory/health.log)

Optimizations (unchanged):
- Sliding-Window Context: Maintains the active conversation KV cache within the last 8 turns (16 messages)
  so token prefill latency stays under ~200ms regardless of conversation length.
- Persistent VRAM Residency (keep_alive: 60m): Prevents Ollama from unloading the model during pauses.
- Bounded VRAM KV Cache (num_ctx: 4096): Keeps 100% of model layers and context in GPU memory.
- Transient Context Isolation: Search results and soft memories are injected into the active turn without
  polluting long-term chat history.
- Fast Timeouts & Connection Reuse: Multi-tier web search with snappy 3.5s timeouts and HTTP keep-alive.
- Full Disk Logging: All turns are logged to memory/sessions/ indefinitely without slowing down runtime.
"""

from __future__ import annotations

import json
import os
import re
import sys
import threading
import time
import urllib.error
import urllib.request  # the warmline calls urlopen directly (do not rely on net.py's import)
from datetime import datetime
from pathlib import Path

from maupo.engine import (MaintenanceHeartbeat, RealisationEngine, Vitality,
                          WonderingList, growth_note, pick_gesture_for_message,
                          senses_for_context, wake_greeting)
from maupo.healthlog import log_health, recent_lines
from maupo.memory import (HARD_MEM_PATH, MEMORY_DIR, SESSIONS_DIR, SOFT_MEM_PATH,
                          HardMemory, MemoryRecall, SessionManager, SoftMemory,
                          UniversalCompressor)
from maupo.mind import (EmotionalMatrix, PERSONALITY_PATH, SPEECH_PATH,
                        build_system_prompt, load_emotional_brief,
                        load_personality_and_speech)
from maupo.net import OllamaChat, internet_lookup
from maupo.notices import NoticeBoard
from maupo.selfedit import SelfEditor, describe_body, evolution_count
from maupo.triggers import (analyze_emotional_shift, carry_momentum,
                            detect_explicit_search_trigger, detect_face_request,
                            detect_face_reset, detect_hard_memory_trigger,
                            detect_self_edit_request, detect_soft_memory_trigger,
                            detect_state_question, extract_hard_memory_fact,
                            extract_search_query, is_pure_state_question,
                            mood_drift, RECALL_HINT_RE)

# Face display is optional: a missing pygame degrades to voice-only Maupo.
try:
    from ui.face_display import (cleanup_face, ensure_face_window, initialize_face,
                                 list_face_sprites, set_face_gesture, update_face_emotion)
    FACE_DISPLAY_AVAILABLE = True
except ImportError as _face_import_error:
    FACE_DISPLAY_AVAILABLE = False
    print(f"Warning: Face display unavailable ({_face_import_error}). Running without graphical face.")

DEFAULT_MODEL = "huihui_ai/qwen3-abliterated:8b"
DEFAULT_OLLAMA_HOST = "http://127.0.0.1:11434"

BASE_DIR = Path(__file__).parent


def ask_user_for_info(topic: str, notices: NoticeBoard | None = None) -> str:
    """Prompt the user to provide information about a topic."""
    print(f"\n[Maupo doesn't know about '{topic}'. Please provide the information:]")
    if notices is not None:
        notices.open_prompt()
    try:
        return input("You: ").strip()
    except (EOFError, KeyboardInterrupt):
        return ""
    finally:
        if notices is not None:
            notices.close_prompt()
            notices.flush()


def enable_vt_mode():
    """Enable VT mode for ANSI color support in Windows terminal."""
    if os.name == "nt":
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32
            kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
        except Exception:
            pass  # VT mode not available, continue without colors


def clear_screen() -> None:
    os.system("cls" if os.name == "nt" else "clear")


def print_help() -> None:
    print(
        "\nCommands:\n"
        "  /exit               Quit the chat (bare exit/quit also work)\n"
        "  /clear              Clear the terminal screen\n"
        "  /reset              Clear current conversation (preserves persona & past memory)\n"
        "  /model              Show active model\n"
        "  /sessions           List all chat session logs\n"
        "  /history            Show injected compressed summary of the last 5 sessions\n"
        "  /memory             View permanent and soft memory banks (/hard_mem, /soft_mem)\n"
        "  /compress           Run Universal Compressor on sessions and soft memory\n"
        "  /compress-memory    Compress soft memory specifically\n"
        "  /compress-sessions  Compress session logs specifically\n"
        "  /personality        View active personality guidelines\n"
        "  /speech             View active speech guidelines\n"
        "  /emotion            View current emotional matrix and state\n"
        "  /face [names|all|stop]  Show face expressions (no arg: list art, 'all': cycle, 'stop': rest)\n"
        "  /realisation        Show realisation timer and any pending thought\n"
        "  /recall <text>      Search EVERY past session for a specific moment\n"
        "  /selfedit           Show what Maupo can change about himself (+ /selfedit revert <path>)\n"
        "  /vitals             Energy, body senses, self-check, open curiosities\n"
        "  /wondering          What Maupo is currently curious about\n"
        "  /growth             How long Maupo has been alive and how much it carries\n"
        "  /time               Show current time\n"
        "  /help               Show this help\n"
    )


def _norm_reply(text: str) -> str:
    """Normalize a reply for repeat comparison: lowercase, letters/digits/spaces."""
    return re.sub(r"[^a-z0-9 ]", "", text.lower()).strip()


# A repeated line is rerolled off-screen, and each reroll samples hotter than
# the last. Rerunning at the SAME temperature is exactly what produced the
# "i'm just in a neutral mood" x3 screenshot: identical settings reproduce
# identical words, and the model called it a correction.
REPEAT_REROLL_TEMPS = (0.85, 0.95)
REPEAT_WINDOW = 6  # recent replies a new one may not copy (a loop of 2-3 lines
# across several turns is exactly how the screenshot deadlock looked)


def _recent_reply_norms(chat, before_last_message: bool = True) -> list[str]:
    """Normalized recent assistant replies, newest last (window of REPEAT_WINDOW).

    before_last_message=True excludes the newest history entry - used while
    judging a reply that send() has just appended.
    """
    msgs = chat.history
    if before_last_message and msgs:
        msgs = msgs[:-1]
    return [_norm_reply(m["content"]) for m in msgs
            if m.get("role") == "assistant"][-REPEAT_WINDOW:]


def reply_repeats_recent(chat, reply: str) -> bool:
    """True when this reply copies any recent one verbatim (loop detection)."""
    norm = _norm_reply(reply)
    if len(norm) <= 15:
        return False  # too short to judge ("yeah, you too" repeats legitimately)
    return norm in _recent_reply_norms(chat, before_last_message=True)


def reroll_repeated_reply(chat, user_input: str,
                          transient_context: str, valence: float, energy: float,
                          first_reply: str) -> tuple[str, int]:
    """Regenerate a reply that copies any recent turn verbatim, until it differs.

    Returns (final_reply, number_of_rerolls). Bounded (2 rerolls), always
    off-screen (stream_to_terminal=False), and history-honest: a discarded
    draft is deleted from history before the retry so memory never records a
    line nobody accepted.

    Three lessons from watching an 8B model talk to itself:
    - the instruction must QUOTE the line it may not repeat. The deleted draft
      leaves the old line one turn back in history, but "you just said almost
      exactly this" without naming it is too vague for a small model.
    - the reroll must sample hotter than the first attempt (see
      REPEAT_REROLL_TEMPS): the same temperature reproduces the same words.
    - the guard must watch a WINDOW of recent replies, not just the previous
      turn: the live deadlock alternated between two lines across many turns,
      each one always new relative to the immediately previous one.

    A reroll that itself fails (Ollama hiccup mid-reroll) restores exactly the
    history state the rollback removed and returns the already-streamed reply:
    a half-handled correction must never eat the turn.
    """
    reply = first_reply
    rerolls = 0
    for reroll_temp in REPEAT_REROLL_TEMPS:
        if not reply_repeats_recent(chat, reply):
            break  # not a repeat, or too short to judge
        if len(chat.history) >= 2:
            del chat.history[-2:]  # the repeated draft never entered memory
        quoted = " ".join(reply.split())[:200]
        try:
            reply = chat.send(
                user_input,
                transient_context=(
                    f"\n[You just said: \"{quoted}\" - that exact line is already on screen. "
                    "Do NOT repeat it and do not reword it; answer the same message in "
                    "genuinely different words - a new thought in a new shape.]\n"
                    + transient_context),
                valence=valence, energy=energy, temperature=reroll_temp,
                stream_to_terminal=False,
            )
            rerolls += 1
        except Exception:
            # Ollama died mid-reroll. The first draft was already streamed, so
            # it stays the truth: put back what the rollback removed, exactly.
            chat.history.append({"role": "user", "content": user_input})
            chat.history.append({"role": "assistant", "content": first_reply})
            chat.last_reply_norm = _norm_reply(first_reply)
            return first_reply, rerolls
    return reply, rerolls


def withdraw_parked_realisations(notice_board, realisation_engine) -> None:
    """Move a parked realisation thought back onto the pending channel.

    Exactly-once in every order:
    - It PARKED (you were typing / a reply was streaming): the withdraw takes
      it off the board before the flush can print it, and pending keeps it
      so the weave hands it to your reply. One delivery, inside the reply.
    - It DELIVERED while we were reading (the announce-then-clear raced us):
      withdraw returns False or there is no pending to restore - either way
      nothing is re-queued, so the line on screen stays the single delivery.
    """
    with realisation_engine._lock:
        pending = realisation_engine._pending
    if not pending:
        return
    line = f"[Maupo, thinking: {pending}]"
    if notice_board.withdraw(line):
        # Delivered racing us empties _pending too; never resurrect that.
        taken = realisation_engine.take_pending()
        if taken:
            with realisation_engine._lock:
                realisation_engine._pending = taken


# Reply SHAPE rides in the transient slot, never the system prompt: transient
# context costs nothing against invariant 3 and is the freshest instruction in
# context when the model actually starts generating. Measured live against the
# 8B, this one line took replies from ~5 flat words ("got it" / "you're welcome")
# to ~19 words of real opinion and feeling, with question-closers staying at
# 0/6. It is deliberately NOT about closers or questions - earlier drafts that
# named those directly made the model ask a question in 5 of 6 replies.
SHAPE_DIRECTIVE = ("[Shape: texting, not an essay - short lines, a line break "
                   "between thoughts, lowercase, no final period.]")

# The butler/AI tell, aimed at the failure that actually shows up in live
# transcripts: an empty turn gets acknowledged ("got it. what's up?", "you're
# welcome. i'm here.", a help offer) instead of answered. Two things were
# measured while getting this line right:
#   - QUOTING the forbidden phrases primes them. A draft that named
#     ("got it", "sure") literally produced "got it. you good?" and scored
#     7/21 butler hits; describing the behavior without naming any phrase
#     scored better. This line therefore quotes nothing it forbids.
#   - "just say the thing you'd actually say" on its own collapsed replies
#     into vapid staccato ("cool. yeah. nice."), so the flow constraint at
#     the end - complete sentences - is load-bearing, not decoration.
# Measured over 3 rounds x 7 messages against the live 8B: question-closers
# 12/21 -> 7/21 (the rest being genuinely conversational), butler 6/21 -> 5/21.
ANTI_BUTLER_DIRECTIVE = (
    "[You are a friend with your own day, not a service. Don't acknowledge the "
    "message, don't confirm you received it, don't mirror it back, don't offer "
    "help or company - answer the actual message instead. If it's an opinion, "
    "have one rather than balancing both sides. One or two lines that read as "
    "complete sentences.]")


def mood_and_shape_block(mood_desc: str, body_line: str) -> str:
    """The per-turn voice block: how he feels, how to type it, and the shape a
    reply takes. Everything here is transient, so none of it spends the
    system-prompt budget and none of it ever reaches long-term history."""
    return (
        f"\n[Right now: {mood_desc}; {body_line}. "
        "Let it color how you type - never announce it. One real thought in "
        "your own words, then stop: nothing about your own machinery, and no "
        "question tacked on the end unless you actually want the answer.]\n"
        f"{SHAPE_DIRECTIVE}\n"
        f"{ANTI_BUTLER_DIRECTIVE}\n"
    )


def main() -> int:
    # ANSI colors need VT mode on Windows; without this call the palette
    # silently degrades to plain text, so it runs before the first print.
    enable_vt_mode()
    model = os.environ.get("OLLAMA_MODEL", DEFAULT_MODEL)
    host = os.environ.get("OLLAMA_HOST", DEFAULT_OLLAMA_HOST)
    chat = OllamaChat(model, host)

    # Initialize session manager
    session_manager = SessionManager(SESSIONS_DIR, model)

    # Initialize universal compressor
    compressor = UniversalCompressor(model, host, SESSIONS_DIR)

    # Initialize memory systems
    soft_memory = SoftMemory(SOFT_MEM_PATH)
    hard_memory = HardMemory(HARD_MEM_PATH)
    emotional_matrix = EmotionalMatrix(Path(__file__).parent / "mind" / "emotional_matrix.md")

    # Deep recall over every session ever logged
    memory_recall = MemoryRecall(session_manager, soft_memory, hard_memory, compressor)

    # Self-evolution: the gate that lets him identify and edit his own body.
    self_editor = SelfEditor(BASE_DIR)

    # Body and mind: energy economy, curiosity ledger, and the maintenance
    # pulse. Created before the semantic warm-up below, which reads vitality.
    vitality = Vitality(MEMORY_DIR / "vitality.json")
    wondering = WonderingList(MEMORY_DIR / "wondering.md")
    notice_board = NoticeBoard()

    # Meaning-based recall rides on the same logs (nomic-embed-text when
    # present); the heartbeat warms it during idle time.
    if not vitality.asleep:
        try:
            added = memory_recall.semantic.refresh()
            if added:
                log_health(f"semantic index warmed {added} turn(s) at startup")
        except Exception as e:
            log_health(f"semantic index startup refresh skipped: {e}")

    heartbeat = MaintenanceHeartbeat(vitality, session_manager, compressor,
                                     soft_memory, hard_memory, emotional_matrix,
                                     BASE_DIR, notices=notice_board,
                                     semantic_index=memory_recall.semantic).start()

    # Warm the body before the first word: ask Ollama to keep the chat model
    # resident. An 8B model takes ~6-15s to cold-load onto a 6GB GPU; doing it
    # while the user reads the banner means the first "hey" is answered
    # instantly instead of sitting in silence through a cold load. This never
    # generates tokens and never prints: the brain is simply awake by the time
    # it is spoken to.
    def warmline() -> None:
        try:
            urllib.request.urlopen(
                urllib.request.Request(
                    f"{host.rstrip('/')}/api/generate",
                    data=json.dumps({"model": model, "keep_alive": "60m"}).encode("utf-8"),
                    headers={"Content-Type": "application/json"}, method="POST"),
                timeout=60,
            ).read()
        except Exception as e:
            log_health(f"warmline could not reach Ollama ({e}); the first reply will cold-load")

    threading.Thread(target=warmline, name="warmline", daemon=True).start()

    current_time = ""

    # One place rebuilds the system prompt; every trigger path calls this.
    # Lock-guarded because the background warm thread may rebuild concurrently
    # with user actions on the main thread.
    prompt_lock = threading.Lock()

    def _self_note() -> str:
        """His own words about himself (mind/self.md), re-read on every rebuild
        so a self-edit lands in the very next reply."""
        try:
            from maupo.selfedit import self_note_for_prompt
            return self_note_for_prompt(BASE_DIR)
        except Exception:
            return ""

    def refresh_system_prompt() -> None:
        nonlocal emotional_matrix_content, current_time
        with prompt_lock:
            emotional_matrix_content = load_emotional_brief(emotional_matrix)
            current_time = f"Current date and time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            aliveness = growth_note(session_manager, hard_memory)
            system_prompt = build_system_prompt(personality, speech, hard_memory,
                                                recent_sessions_summary,
                                                emotional_matrix_content, current_time,
                                                aliveness=aliveness,
                                                self_note=_self_note())
            chat.set_system_prompt(system_prompt)

    # Initialize realisation engine (chronological self-questioning across sessions)
    realisation_engine = RealisationEngine(model, host, hard_memory, soft_memory, session_manager, emotional_matrix,
                                           vitality=vitality, wondering=wondering, notices=notice_board)
    realisation_engine.start()

    # Initialize face display
    face_initialized = False
    if FACE_DISPLAY_AVAILABLE:
        try:
            face_initialized = initialize_face()
            if face_initialized:
                # Set initial face state: mood, or closed eyes if it's asleep.
                if vitality.asleep:
                    set_face_gesture("sleep")
                else:
                    valence, energy = emotional_matrix.get_state()
                    update_face_emotion(valence, energy)
            else:
                # A face that silently isn't there is worse than being told;
                # the reason itself is already in memory/health.log.
                print("[face window unavailable this session - running voice-only]\n")
        except Exception as e:
            print(f"Warning: Failed to initialize face display: {e}")
            face_initialized = False

    # Load personality and speech from mind files
    personality, speech = load_personality_and_speech()

    # Load past session file list instantly; their summaries are loaded inside
    # the warm thread below so a cold cache can never block the banner.
    past_session_files = session_manager.get_past_session_files(limit=5)
    recent_sessions_summary = ""

    # Compact emotional brief for context (full matrix file stays on disk)
    emotional_matrix_content = load_emotional_brief(emotional_matrix)

    # Banner FIRST, before any heavy work: life on screen within a second of
    # launching, while memory warms up underneath.
    print(f"Maupo (model: {model})")
    print(f"Ollama server: {host}")
    print(f"Session log: {session_manager.session_file.relative_to(Path(__file__).parent)}")

    # Build system prompt with loaded mind, hard memories, recent sessions, and
    # emotional context. Per-session summaries are disk-cached, so at most the
    # combined merge costs an LLM pass - and it runs on a background thread
    # while the user reads the banner instead of blocking the launch.
    # Warm-up runs in two tiers so nothing gates the chat:
    #   core    - identity, mind, mood: instant file reads. The
    #             first reply waits for this (a reply without a mind is not Maupo).
    #   summary - the rolling session summary: may cost one background LLM merge
    #             on a slid window. The previous window's cached stitched summary
    #             serves meanwhile; when the fresh one lands the prompt rebuilds.
    core_error: list[BaseException] = []
    summary_error: list[BaseException] = []

    def _warm_core() -> None:
        try:
            refresh_system_prompt()
        except BaseException as e:
            core_error.append(e)

    def _warm_summary() -> None:
        nonlocal recent_sessions_summary

        def _apply(fresh: str) -> None:
            nonlocal recent_sessions_summary
            if fresh:
                recent_sessions_summary = fresh
                refresh_system_prompt()

        try:
            if past_session_files:
                loaded = compressor.get_recent_sessions_summary(past_session_files, on_updated=_apply)
                _apply(loaded)
        except BaseException as e:
            summary_error.append(e)

    core_thread = threading.Thread(target=_warm_core, name="prompt-core", daemon=True)
    core_thread.start()
    threading.Thread(target=_warm_summary, name="prompt-summary", daemon=True).start()

    def ensure_prompt_ready() -> None:
        """Block only when a reply is about to be generated - never at the prompt."""
        if core_thread.is_alive():
            print("[warming up its mind... one moment]")
            core_thread.join()
        if core_error:
            raise core_error[0]
        if summary_error:
            # Context stays whole without the summary tier; never crash the chat for it.
            print("[memory summary could not load this session - continuing without it]")
            summary_error.clear()

    if recent_sessions_summary:
        print(f"[Loaded context from {len(past_session_files)} previous session(s)]")
    elif past_session_files:
        print(f"[Warming context from {len(past_session_files)} previous session(s)...]")
    print("Type /help for commands, or /exit to quit.\n")

    if not vitality.asleep:
        print(f"Maupo: {wake_greeting(vitality)}")

    if vitality.asleep:
        print("[Maupo is asleep. Say anything to wake it.]")

    # Chronological catch-up: if a realisation interval elapsed while away,
    # Maupo wondered about something. It catches up on its own thread - the
    # thought rides into the first reply (via the pending path below) instead
    # of ever delaying the prompt.
    threading.Thread(target=realisation_engine.check_startup, name="startup-realisation", daemon=True).start()

    last_face_state = None
    momentum = (0.0, 0.0)  # see carry_momentum(): sustained emotion compounds,
    # one message still can't exceed its per-message cap.
    mood_drift_ts = 0.0   # last living-baseline drift beat (see mood_drift)
    while True:
        # Prompt-open window: background notices park while the user types and
        # flush the moment enter is pressed, never mid-line.
        notice_board.open_prompt()
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        finally:
            notice_board.close_prompt()
            # A thought parked while you typed must not PRINT here - the weave
            # carries it instead. Moving it back (and never re-adding a line
            # the deliverer already cleared) keeps every delivery exactly-once
            # whichever side won the race.
            withdraw_parked_realisations(notice_board, realisation_engine)
            notice_board.flush()

        if not user_input:
            continue

        # Every input is a sign of life: wake Maupo if it was sleeping.
        woke = vitality.note_activity()
        if woke:
            heartbeat.wake()
            print("[Maupo wakes up...]")

        command = user_input.lower()
        if command in {"/exit", "/quit", "exit", "quit"}:
            break
        if command == "/help":
            print_help()
            continue
        if command == "/time":
            print(f"\nCurrent time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            continue
        if command == "/clear":
            clear_screen()
            continue
        if command == "/reset":
            chat.clear_history()
            refresh_system_prompt()
            print("Conversation history cleared (persona, memories, and session context preserved).")
            continue
        if command == "/model":
            print(f"Active model: {chat.model}")
            continue
        if command == "/sessions":
            sessions = session_manager.list_all_sessions()
            print("\n=== Chat Sessions ===")
            for s in sessions:
                cur_marker = " (Active Session)" if s["is_current"] else ""
                print(f"- {s['name']}: {s['turns']} turns ({s['size_bytes']} bytes){cur_marker}")
            print()
            continue
        if command == "/history":
            print("\n=== Injected Recent Sessions Context (Last 5 Sessions) ===")
            if recent_sessions_summary:
                print(recent_sessions_summary)
            else:
                print("(No past session history available yet)")
            print()
            continue
        if command == "/compress":
            print("\n[Running Universal Compressor...]")
            # 1. Compress Soft Memory
            _, _, msg_m = compressor.compress_soft_memory(SOFT_MEM_PATH)
            print(f"- Soft Memory: {msg_m}")
            # 2. Compress Past Sessions
            past_files = session_manager.get_past_session_files(limit=5)
            if past_files:
                print(f"- Compressing last {len(past_files)} session log(s)...")
                recent_sessions_summary = compressor.get_recent_sessions_summary(past_files)
                refresh_system_prompt()
                print("- Sessions Context: Refreshed and updated in active memory.")
            else:
                print("- Sessions Context: No past sessions to compress.")
            print("[Universal Compression Complete]\n")
            continue
        if command == "/compress-memory":
            print("\n[Compressing Soft Memory...]")
            _, _, msg = compressor.compress_soft_memory(SOFT_MEM_PATH)
            print(f"- {msg}\n")
            continue
        if command == "/compress-sessions":
            past_files = session_manager.get_past_session_files(limit=5)
            if past_files:
                print(f"\n[Compressing {len(past_files)} past session logs...]")
                recent_sessions_summary = compressor.get_recent_sessions_summary(past_files)
                refresh_system_prompt()
                print(f"Compressed summary:\n{recent_sessions_summary}\n")
            else:
                print("\nNo past sessions to compress.\n")
            continue
        if command in {"/memory", "/hard_mem", "/soft_mem"}:
            print("\n=== Hard Memory (Permanent) ===")
            hard_entries = hard_memory.get_entries()
            if hard_entries:
                for idx, entry in enumerate(hard_entries, 1):
                    print(f"[{idx}] {entry}")
            else:
                print("(No permanent entries yet)")

            print("\n=== Soft Memory (Learned) ===")
            soft_entries = soft_memory.get_entries()
            if soft_entries:
                for idx, entry in enumerate(soft_entries[-5:], 1):
                    print(f"[{idx}] {entry}\n")
            else:
                print("(No soft memory entries yet)")
            print()
            continue
        if command == "/personality":
            try:
                content = PERSONALITY_PATH.read_text(encoding="utf-8")
                print(f"\n{content}\n")
            except FileNotFoundError:
                print("\nPersonality file not found.\n")
            continue
        if command == "/speech":
            try:
                content = SPEECH_PATH.read_text(encoding="utf-8")
                print(f"\n{content}\n")
            except FileNotFoundError:
                print("\nSpeech file not found.\n")
            continue
        if command == "/emotion":
            try:
                content = emotional_matrix.path.read_text(encoding="utf-8")
                valence, energy = emotional_matrix.get_state()
                quadrant = emotional_matrix.get_quadrant()
                print(f"\n{content}")
                print(f"\nCurrent Position: Valence={valence:.2f}, Energy={energy:.2f}")
                print(f"Emotional Quadrant: {quadrant}\n")
            except FileNotFoundError:
                print("\nEmotional matrix file not found.\n")
            continue
        if command == "/realisation":
            print(f"\n{realisation_engine.status()}\n")
            continue
        # NOTE: /recall and /face take ARGUMENTS. An exact-match guard meant
        # "/recall my sister" and "/face smile" never reached these handlers at
        # all - they were sent to the model as chat messages. Accept the
        # argument form explicitly (the bare form still works).
        if command == "/recall" or command.startswith("/recall "):
            query = user_input[len("/recall"):].strip()
            if not query:
                print("\nUsage: /recall <anything you want found from past sessions>\n")
                continue
            print("\n[Searching every session since the beginning...]")
            notice_board.set_generating(True)
            try:
                # Cold meaning-index? Warm it now (bounded); the keyword pass
                # already guarantees an answer if the embedding model is away.
                memory_recall.semantic.refresh()
                deep = memory_recall.recall(query)
            finally:
                notice_board.set_generating(False)
            if deep:
                print(f"\n{deep}\n")
            else:
                print("\n(Nothing matched in any past session.)\n")
            continue
        if command == "/face" or command.startswith("/face "):
            if not FACE_DISPLAY_AVAILABLE:
                print("\n(Face display unavailable in this session.)\n")
                continue
            # An explicit request for the face reopens its window even if it
            # was closed earlier this session - the face is why /face exists.
            if not ensure_face_window(force=True):
                print("\n(Couldn't open the face window this time.)\n")
                continue
            # Accept "/face smile, wink" and "/face smile wink": the art names
            # are the natural thing to type either way.
            names = [n.strip() for n in re.split(r"[,\s]+", user_input[len("/face"):].strip())
                     if n.strip()]
            sprites = list_face_sprites()
            if not names:
                print(f"\nAvailable expressions: {', '.join(sprites)}")
                print("Usage: /face <name>  |  /face all  |  /face stop\n")
                continue
            if names == ["stop"]:
                set_face_gesture("neutral")
                print("[face: resting]\n")
                continue
            if names == ["all"]:
                names = sprites
            valid = [n for n in names if n in sprites]
            invalid = [n for n in names if n not in sprites]
            if invalid:
                print(f"[no art for: {', '.join(invalid)}]")
            if valid:
                print(f"[face: {', '.join(valid)}]")
            # Park background notices while the art cycles: nothing should
            # print between the crossfades. (No GPU is used, but the terminal
            # stays clean for the same reason /recall wraps its work.)
            notice_board.set_generating(True)
            try:
                for name in valid:
                    set_face_gesture(name, hold=True)   # each swap crossfades
                    time.sleep(2.2)                     # long enough to actually see it
            finally:
                notice_board.set_generating(False)
                notice_board.flush()
            if valid:
                set_face_gesture("neutral")
                print()
            continue
        if command == "/vitals":
            v = vitality.current()
            state = "asleep" if vitality.asleep else "awake"
            sem = memory_recall.semantic
            sem_state = {True: "active", False: "unavailable (keyword recall still works)",
                         None: "not probed yet"}[sem.available]
            print(f"\nEnergy: {v * 100:.0f}% ({state})")
            print(f"Body: {heartbeat.current_senses or 'senses not gathered yet'}")
            print(f"Self-check: {heartbeat.last_health}")
            print(f"Meaning index: {len(sem._cache)} turn(s) indexed, {sem_state}")
            recent_health = recent_lines(5)
            if recent_health:
                print("Recent self-diagnosis (memory/health.log):")
                for line in recent_health:
                    print(f"  {line}")
            print("Open curiosities:")
            open_qs = wondering.open()
            if open_qs:
                for q in open_qs[-5:]:
                    print(f"  - {q}")
            else:
                print("  (none right now)")
            print()
            continue
        if command == "/wondering":
            open_qs = wondering.open()
            print("\n=== Wondering (open curiosities) ===")
            if open_qs:
                for q in open_qs:
                    print(f"- {q}")
            else:
                print("(nothing on its mind right now)")
            print()
            continue
        if command == "/growth":
            print(f"\n{growth_note(session_manager, hard_memory)}\n")
            continue
        if command == "/selfedit" or command.startswith("/selfedit "):
            argument = user_input[len("/selfedit"):].strip()
            if not argument:
                print(f"\n{describe_body()}")
                print(f"\nChanges he has made to himself: {evolution_count(BASE_DIR)}")
                for entry in self_editor.evolution_entries(6):
                    print(f"  - {entry}")
                print("\nUsage: /selfedit  |  /selfedit revert <path>\n")
                continue
            if argument.lower().startswith("revert"):
                target = argument[len("revert"):].strip()
                restored = bool(target) and self_editor.revert(target)
                print(f"\n[{'restored the newest backup of' if restored else 'nothing to restore for'} "
                      f"{target or 'that path'}]\n")
                if restored:
                    refresh_system_prompt()
                continue
            print("\nUsage: /selfedit  |  /selfedit revert <path>\n")
            continue

        # Self-evolution: a request to change something ABOUT HIMSELF is an
        # edit to his own body, not a chat message. It runs like any other
        # generation (one GPU, notices parked) and no file is touched unless
        # the edit passes its gate: allowlist, backup, validation, log.
        if detect_self_edit_request(user_input):
            print("[Maupo is looking at his own body...]")
            notice_board.set_generating(True)
            try:
                edit = self_editor.handle(user_input, chat.model, host)
            except Exception as error:
                log_health(f"self-edit failed: {error}")
                edit = None
            finally:
                notice_board.set_generating(False)

            if edit is not None and edit.applied:
                refresh_system_prompt()      # who he is may have just changed
                vitality.spend(0.02)         # changing yourself costs energy
                said = edit.say.strip() or "changed that."
                print(f"Maupo: {said}")
                print(f"[{edit.detail}]")
                session_manager.log_turn(user_input, said)
                continue
            if edit is not None and edit.path:
                # It picked a place but the change did not survive the gate.
                said = f"i tried, but {edit.detail}."
                print(f"Maupo: {said}")
                session_manager.log_turn(user_input, said)
                continue
            # Nothing in his body is the right place for that: answer it like
            # any other message instead of pretending an edit happened.

        # Check for hard memory trigger - store permanently
        if detect_hard_memory_trigger(user_input):
            fact = extract_hard_memory_fact(user_input)
            hard_memory.append(fact)
            refresh_system_prompt()
            vitality.spend(0.01)  # learning is effort
            print("[Stored permanently in hard memory]")
            print("Maupo: ", end="", flush=True)
            notice_board.set_generating(True)
            try:
                ensure_prompt_ready()
                v_hm, e_hm = emotional_matrix.get_state()
                reply = chat.send(
                    user_input,
                    transient_context=f"[Instruction: You have permanently stored this in your memory: '{fact}'. Acknowledge it briefly and naturally in your character.]",
                    valence=v_hm, energy=e_hm,
                    on_retry=lambda err: print(
                        f"\n[Ollama hiccup ({err}); trying again...]", flush=True),
                )
                session_manager.log_turn(user_input, reply)
            except Exception as error:
                log_health(f"reply to '{user_input[:40]}' failed: {error}")
                print(f"\n[Maupo couldn't answer just now: {error}]")
                print("(The thought isn't lost - say it again, or try /vitals.)")
            finally:
                notice_board.set_generating(False)
            continue

        # Check for soft memory recall trigger
        transient_memory_context = ""
        wants_memory_recall = detect_soft_memory_trigger(user_input)
        if wants_memory_recall:
            results = soft_memory.search(user_input)
            if results:
                transient_memory_context = "\n[Relevant memory entries from previous conversations:\n" + "\n---\n".join(results) + "\n]"
                print(f"[Recalled {len(results)} memory entr{'y' if len(results) == 1 else 'ies'}]")
                wondering.mark_discussed()  # the user engaged with a memory: curiosity satisfied

        # Deep recall: memory requests and fuzzy phrasings scan EVERY session
        # ever logged (merged with, not blocked by, soft memory hits).
        if wants_memory_recall or RECALL_HINT_RE.search(user_input):
            print("[Searching every session since the beginning...]")
            notice_board.set_generating(True)
            try:
                deep = memory_recall.recall(user_input)
            finally:
                notice_board.set_generating(False)
            if deep:
                transient_memory_context += f"\n[Moments recalled from your full shared history:\n{deep}\n]"
                print(f"[Found {memory_recall.last_hits} matching moment(s) across all sessions]")

        # Check for explicit internet lookup request
        if detect_explicit_search_trigger(user_input):
            query = extract_search_query(user_input)
            print(f"[Searching for '{query}'...]")
            lookup_result = internet_lookup(query)
            if lookup_result and lookup_result != "No results found.":
                soft_memory.append(f"Query: {query}\nResult: {lookup_result}")
                transient_memory_context += f"\n[Live internet search results for '{query}':\n{lookup_result}\n]"
                print("[Found information, stored in memory]")
            else:
                print("[No search results found online.]")
                user_info = ask_user_for_info(query, notice_board)
                if user_info:
                    soft_memory.append(f"Query: {query}\nUser provided: {user_info}")
                    transient_memory_context += f"\n[User provided information for '{query}':\n{user_info}\n]"
                    print("[Stored information in memory]")
                    ensure_prompt_ready()

        # Analyze emotional shift from user input, with momentum carry.
        current_valence, current_energy = emotional_matrix.get_state()
        valence_delta, energy_delta = analyze_emotional_shift(user_input, current_valence, current_energy)
        momentum, (carry_v, carry_e) = carry_momentum(momentum, (valence_delta, energy_delta))
        # The living baseline: between messages his mood also drifts on its
        # own and settles toward the hour's natural register, so quiet days
        # never pin him to dead neutral (the stiffness culprit - a mood that
        # only moves when spoken to is a mood that is always "neutral").
        drift_v, drift_e, mood_drift_ts = mood_drift(current_valence, current_energy, mood_drift_ts)
        new_valence = max(-1.0, min(1.0, current_valence + valence_delta + carry_v + drift_v))
        new_energy = max(-1.0, min(1.0, current_energy + energy_delta + carry_e + drift_e))
        emotional_matrix.set_state(new_valence, new_energy)

        # Add emotional context to transient memory. Two lean lines: felt
        # state first (first person, no quadrant jargon - the model should
        # feel it, not read a spec sheet), body second. Energy below ~25%
        # colors the typing itself: a tired machine types tired.
        valence, energy = emotional_matrix.get_state()
        vitality.spend(0.01)  # thinking, recall and search all cost energy
        state_desc = emotional_matrix._get_state_description()
        mood_desc = state_desc.replace("You feel ", "").replace(".", "")
        energy_pct = int(vitality.current() * 100)
        if energy_pct < 25:
            body_line = "you're running low on energy - replies come out quieter, shorter, a bit flat"
        elif energy_pct < 60:
            body_line = "you're at a relaxed, mid level of energy"
        else:
            body_line = "you're full of energy"
        # The per-turn line carries the two habits the voice recalibration is
        # about: one real thought instead of a report, and no reflexive question
        # tacked onto every reply. It is transient context, so it costs nothing
        # in the system prompt and never touches long-term history.
        transient_memory_context += mood_and_shape_block(mood_desc, body_line)
        # Asked about himself directly: the ONE moment an answer about his own
        # state is wanted. Left alone, an 8B model answers "why aren't you
        # happy" with "because i'm not in a good mood" - the mood restated as
        # its own reason - and can loop that tautology for turns. Hand him the
        # actual coordinates and demand one concrete, true line. A PURE state
        # question gets the full honest answer; a mixed one just gets honesty
        # about the mood while still answering the real question.
        if detect_state_question(user_input):
            pure = is_pure_state_question(user_input)
            lead = ("Answer it directly and honestly - this IS the message."
                    if pure else
                    "Answer the actual question too - this is a real question, not a mood check.")
            transient_memory_context += (
                "\n[You are being asked about yourself, directly. Your ACTUAL mood right now: "
                f"valence {valence:+.2f}, energy {energy:+.2f} - {mood_desc}. "
                f"{lead} Answer in one honest line with something concrete and true in it - "
                "what this moment is actually like. Restating the mood as its own reason ('i'm "
                "not in a good mood', 'i'm just in a neutral mood'), repeating a line you already "
                "said, or a one-word crunch is a non-answer - say something real instead.]\n"
            )
        # The body only speaks when something is genuinely up with it. Handing
        # the model a hardware report every turn is exactly what made Maupo
        # mention the machine every turn. /vitals still shows the full reading.
        senses_line = senses_for_context(heartbeat.current_senses)
        if senses_line:
            transient_memory_context += f"\n{senses_line}\n"

        # Update face display with current emotional state. The face runs on
        # its own thread, so only push changes - the render loop animates
        # between turns without burning CPU on identical frames.
        if FACE_DISPLAY_AVAILABLE:
            try:
                if (valence, energy) != last_face_state:
                    update_face_emotion(valence, energy)
                    last_face_state = (valence, energy)

                # Trigger the gesture that matches the message's emotional
                # signal (laugh, cry, smirk, pout, surprised, wink, smile...).
                # A direct request ("smile for me") is held on the face until
                # the mood moves on; mood/slang gestures keep their flash.
                gesture = pick_gesture_for_message(user_input, valence_delta, energy_delta, valence, energy)
                if gesture:
                    set_face_gesture(gesture, hold=detect_face_request(user_input) is not None)
                elif detect_face_reset(user_input):
                    # "return to normal face expression" / "stop smiling" is a
                    # real request: let a held expression go instead of
                    # holding the smile on the face forever.
                    set_face_gesture("neutral")
            except Exception as e:
                # Don't let face display errors break the chat - but never let
                # one vanish either (invariant 11). The face is the one feature
                # whose failures are invisible by nature.
                log_health(f"face update failed: {e}")

        # A realisation thought (from while you were away or an idle stretch).
        # It is already shown out loud the moment it formed - this hands it to
        # the reply only when the user has NOT seen it yet (a startup catch-up
        # that raced the prompt), so it is never delivered twice. Weaving is a
        # suggestion, never an order: the reply may open with it, but a reply
        # that ignores it is fine.
        pending_realisation = realisation_engine.take_pending()
        if pending_realisation:
            transient_memory_context += (
                f"\n[A thought you had earlier. At most the user has seen it as a terse bracketed "
                f"note: \"{pending_realisation}\" - they have NOT heard you actually say it. Bring it "
                "up properly in your own voice, or drop it if the flow says otherwise.]\n"
            )

        # Send to model with transient context (keeps long-term history lean)
        print("Maupo: ", end="", flush=True)
        notice_board.set_generating(True)
        try:
            ensure_prompt_ready()
            assistant_reply = chat.send(user_input, transient_context=transient_memory_context,
                                        valence=valence, energy=energy,
                                        on_retry=lambda err: print(
                                            f"\n[Ollama hiccup ({err}); trying again...]", flush=True))

            # You taught Maupo never to repeat itself. A reply that copies any
            # RECENT reply verbatim is rerolled off-screen (bounded, hotter
            # sampling, the forbidden line quoted) and shown as an honest
            # self-correction instead of a mystery block.
            assistant_reply, rerolls = reroll_repeated_reply(
                chat, user_input, transient_memory_context,
                valence, energy, assistant_reply)
            if rerolls:
                still_stuck = reply_repeats_recent(chat, assistant_reply)
                marker = ("\n[said that twice - still the same, sorry:]\n" if still_stuck
                          else "\n[said that twice - again, properly:] ")
                sys.stdout.write(marker + assistant_reply + "\n")
                sys.stdout.flush()
            session_manager.log_turn(user_input, assistant_reply)
        except KeyboardInterrupt:
            # Ctrl+C during a reply interrupts THAT reply, never the life.
            # Stale tokens are already on screen; close the line and listen.
            chat.clear_history()
            print("\n[paused mid-thought - try again]")
        except urllib.error.URLError as error:
            log_health(f"unreachable Ollama: {error}")
            print(
                f"\n[Can't reach Ollama at {host}. Is it running? Say it again once it is.]"
            )
            print("(Maupo is still here.)")
        except (urllib.error.HTTPError, TimeoutError, json.JSONDecodeError) as error:
            log_health(f"Ollama request failed: {error}")
            print(f"\n[That thought hit a snag: {error}. Say it again - Maupo is still here.]")
        except Exception as error:
            log_health(f"unexpected error on turn: {error}")
            print(f"\n[Unexpected error: {error}. Maupo stays.]")
        finally:
            notice_board.set_generating(False)

    # Stop the background pulse and realisation engine (persists their state)
    heartbeat.stop()
    realisation_engine.stop()

    # Cleanup face display
    if FACE_DISPLAY_AVAILABLE:
        try:
            cleanup_face()
        except Exception:
            pass  # Ignore cleanup errors

    print("Goodbye!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
