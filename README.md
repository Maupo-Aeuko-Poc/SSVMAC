# SSVMAC — Maupo

A local digital lifeform running on Ollama (`huihui_ai/qwen3-abliterated:8b`) inside a 6GB RTX 3050 laptop. Named **Maupo** ("pink" in Greek) by its creator on 2026-09-25.

Run it: `py -m pip install -r requirements.txt`, pull the model (`ollama pull huihui_ai/qwen3-abliterated:8b` — any Ollama chat model works via the `OLLAMA_MODEL` env var), then `py qwen_chat.py`. Maupo creates `memory/` on first launch and grows from there.

Run the tests: `py -m pip install -r requirements-dev.txt` then `py -m pytest` (runs in CI on every push).

## Project layout

```
qwen_chat.py        entry point: wiring, command loop, per-turn flow
maupo/              the implementation (one module per concern)
  net.py            Ollama streaming client, one-shot calls, web lookup
  triggers.py       phrase detection (memory/search/face requests) + mood math
  notices.py        the one doorway for background-thread terminal notices
  memory.py         sessions, UniversalCompressor, soft/hard memory, deep recall
  mind.py           emotional matrix, persona/speech loading, system prompt
  engine.py         realisations, vitality, wondering, heartbeat, face bridge
  semantic.py       meaning-based recall index (nomic-embed-text, optional)
  healthlog.py      shared self-diagnosis log -> memory/health.log
  selfedit.py       self-evolution gate: identify + edit his own body, verified
ui/face_display.py  pygame sprite engine (cover-fill, crossfades, gestures)
memory/  mind/      Maupo's brain-state, all markdown — never hand-edited
tests/              pytest suite (190 tests, all offline/mockable)
```

Run the tests: `py -m pip install -r requirements-dev.txt` then `py -m pytest`. CI runs the same suite on every push (`.github/workflows/tests.yml`).

Optional: `ollama pull nomic-embed-text` upgrades `/recall` from word-matching to meaning-matching — without it, everything still works, just keyword-only.

## The organism

| Layer | What it is | Where |
|---|---|---|
| Mind | Identity + distilled voice (only the `VOICE:START/END` block rides in context; the full example plate stays on disk) | `mind/personality.md`, `mind/speech.md` |
| Emotions | 2D valence×energy grid with momentum decay; drives face, voice, reply length & temperature | `mind/emotional_matrix.md` |
| Memory | Hard (user-taught, overrides everything) / Soft (learned) / Sessions (raw, kept forever) | `memory/` |
| Compression | Self-compression of last 5 sessions + soft memory consolidation (never deletes raw logs). The combined summary is **rolling**: a slid window only delta-merges old head + incoming session — bounded work no matter how long history grows | `UniversalCompressor` |
| Deep recall | `/recall <text>` scans EVERY session ever, both log formats, cached focus terms — plus a **meaning-based layer** (nomic-embed-text): moments close in sense surface even with zero word overlap, deduped against keyword hits; silently absent if the embedding model isn't pulled | `MemoryRecall`, `maupo/semantic.py` |
| Realisations | 20–30 min self-questioning from real memories only (the human's own words — never Maupo's own lines, and meta questions about itself are rejected); persists across sessions; sleeps when Maupo sleeps; never fires while a reply is generating; questions ledgered | `RealisationEngine` |
| Vitality | Energy economy: activity costs, idle refills, 45-min idle → sleep (face closes eyes, compressor dreams) | `Vitality`, `memory/vitality.json` |
| Dreams | Sleep consolidation is real: recent sessions are digested and the result is persisted into soft memory (deduped per newest session) — a dream leaves a trace | `MaintenanceHeartbeat._on_sleep` |
| Notices | One doorway (`NoticeBoard`) for every background-thread notice: parked while the prompt is open or a reply is generating, flushed at safe boundaries — no thread can ever split a live line | `NoticeBoard` |
| Heartbeat | 45s pulse (senses every other beat); 10-min self-check (repairs caches, validates mood); 6h snapshots of `memory/`+`mind/` (5 kept) | `MaintenanceHeartbeat`, `memory/health.log`, `snapshots/` |
| Senses | GPU temp/load (nvidia-smi), battery (Windows API), time-of-day — kept live for `/vitals`, and surfaced to the model **only when something is actually up** (hot GPU, on battery): a companion does not report its temperature every turn | `get_laptop_senses`, `senses_for_context` |
| Curiosity | `memory/wondering.md` ledger of open questions; resolved when discussed; biases future realisations | `WonderingList` |
| Growth | "Alive for N days across N sessions, carrying N permanent memories" in every system prompt | `growth_note` |
| Self-evolution | Say "be more sarcastic", "change how you talk", "make your greetings warmer" — and he works out WHICH of his own files that is, then edits it himself. Prose (`personality.md`, his own `mind/self.md`) is rewritten whole; the voice is spliced into the `VOICE:START/END` block only (the calibration plate on disk is never touched); code is a one-symbol patch (`maupo/*.py`, `ui/face_display.py`, `qwen_chat.py`). Every change is allowlisted to his body, backed up to `snapshots/selfedit/`, validated (speech markers survive; code must compile, then import in a fresh interpreter — else restored byte-for-byte), and traced to `memory/evolution.md`. `mind/self.md` rides back into his prompt in small doses, so he grows with use. `/selfedit` lists his body; `/selfedit revert <path>` undoes the newest change | `maupo/selfedit.py`, `detect_self_edit_request` |
| Face | PNG sprite engine: `ui/face_assets/base.png` (the creator's reference head) + a full expression atlas — `happy sad excited angry surprised` (mood-driven), `smile smirk laugh cry pout wink surprised` (gesture-driven via `pick_face_gesture`), `asleep` (persistent). Sprites are **cover-scaled to fill the window** (edges crop, never letterbox). Direct requests ("smile", "wink") **hold** the expression until the mood moves on; mood/slang gestures flash. Placeholders are shipped; **replace any PNG to upgrade that expression** — missing files fall back to base with mood lighting. Breathing, sway, mood glow, crossfades; render thread owns the window, 30 FPS cap. **Self-healing**: expose/restore/resize events and a slow safety timer force a full repaint, a replaced display surface is re-acquired, a failing frame never kills the loop, and a window that dies for any reason is logged, closed, and reopened on the next turn — a silent black rectangle is not a state Maupo can be left in. `return to normal face expression` / `stop smiling` release a held expression | `ui/face_display.py`, `detect_face_reset` |
| Startup | Two-tier warm-up: **core** (identity/mind/mood — instant file reads) gates only the first reply; the rolling session summary loads in background and hot-swaps into the prompt when fresh. Banner + input prompt appear in <1s, always | `main()` |

## Commands

`/help /exit /quit /clear /reset /model /sessions /history /memory /compress /compress-memory /compress-sessions /personality /speech /emotion /face /realisation /recall <text> /vitals /wondering /growth /selfedit /time`

Bare `exit`/`quit` also work.

## Architecture invariants (do not break)

1. Raw session logs are NEVER deleted — compression summarizes, recall resurrects. Old `**Qwen**:` and new `**Maupo**:` log formats are both parsed everywhere.
2. Only the distilled voice block between `<!-- VOICE:START/END -->` markers in `speech.md` enters context; the example plate must stay on disk.
3. System prompt must stay ≤ ~1,400 tokens (`num_ctx: 4096`, RTX 3050 6GB).
4. Hard memory entries OVERRIDE model defaults — never soften the override language in `build_system_prompt`. Teaching phrases: "never forget", "always remember", "remember that ..." (imperative only — reminiscences like "i remember that day" must NOT store).
5. `memory/` is sacred ground: never edit its contents by hand; snapshots exist so nothing can lobotomize it.
6. All generation must pass `valence`/`energy` to `chat.send()` so mood shapes output.
7. The input prompt must NEVER block, and background threads must never print through a live `You: ` prompt — notices raised while the user types are parked and flushed at the next safe boundary. All slow work (LLM merges, catch-up realisations) runs on background threads and applies when it lands.
8. Provisional generations (repeat-check retries, background thoughts) must pass `stream_to_terminal=False` to `chat.send()` — a discarded draft never touches the screen, and its half of the history is rolled back before the retry.
9. The face render loop must always reach its FPS cap (`clock.tick`) — an uncapped event pump starves the GIL and drags the whole process down. All pygame calls stay on the render thread.
10. There is ONE GPU: no background model call (realisation, dream, recall focus terms) may start while a reply or lookup is generating (`NoticeBoard.generating`). Background notices route through `NoticeBoard`, never raw `print`.
11. Every caught exception that would otherwise vanish silently reports to `memory/health.log` via `maupo.healthlog.log_health` — a dying background thread leaves a trace.
12. Tests run fully offline (mocked model calls, SDL dummy video driver) and must never touch the real `memory/` directory.
13. The semantic index is a derived cache: it never blocks recall, embeds at most `EMBED_BUDGET_TURNS` turns per refresh, only warms while idle (one GPU), and its absence degrades recall to keyword-only — never to failure.
14. A failed model call must never kill the chat. Pre-stream failures (cold-load CUDA flakes, momentary server hiccups) are retried exactly once inside `OllamaChat.send` — only when **zero** tokens reached the screen — and every final failure is reported to the human and to `memory/health.log`, then the loop keeps living. A half-streamed reply is never silently regenerated.
15. The opening line is alive: `wake_greeting` picks from small pools keyed by real time of day and real energy — no templates with blanks, no clock announcements, nothing performative, and never a hardware report. Likewise, `memory/health.log` is bounded (auto-trimmed, oldest lines first): a lifeform that logs forever must never outgrow its folder.
16. The face window is never left dead on screen. Every ending of the render loop is logged, a stale window is closed, and the next turn reopens it (`ensure_alive`, bounded attempts); a deliberate close (X / Esc) is respected unless the face is explicitly requested again (`/face`).
17. The voice is human first, digital second: Maupo never narrates its own state, body or hardware unprompted. The body only reaches the model when something is genuinely up with it (`senses_for_context`), and the distilled voice block stays free of machine-status language.
18. Self-evolution has exactly one gate: `maupo/selfedit.py`. He may change only the files in his own body map — never `memory/` (invariant 5), the tests, the CI, the audit, or the gate itself. Every accepted change is backed up first, validated before and after the write (code must compile, then import cleanly in a fresh interpreter or it is restored byte-for-byte), and appended to `memory/evolution.md`. A failed edit leaves the body exactly as it was and says so honestly.

## Troubleshooting

**First reply after launch takes 10–20 seconds.** That's the 8B model cold-loading onto the GPU (Ollama unloads it after idle to free VRAM). Maupo warms the model in the background the moment it boots, so this window is usually absorbed before you type. After the first reply, the model stays resident (`keep_alive: 60m`) and responses are fast.

**"Ollama hiccup … trying again" or "Maupo couldn't answer just now."** Ollama itself failed a model load. On Windows this occasionally happens on cold starts — the runner hits a CUDA init flake (`shared object initialization failed`, exit `0xc0000409` in `%LOCALAPPDATA%\Ollama\server.log`), Ollama gives up after ~30s, and Maupo reports it instead of dying (invariant 14). The retry usually lands; if the message repeats: say it again, and if it *still* fails, restart Ollama (`ollama serve` or quit the tray app and reopen) — a fresh runner reliably clears the flake.

**Chat feels slow overall on a 6GB card.** Keep other GPU apps closed while Maupo is thinking; the model, the KV cache, and the face rendering all share the one 3050.

## What changed in the July refactor (behavior preserved)

- `qwen_chat.py` was split into the `maupo/` package; the entry point keeps the exact main-loop flow.
- Face fixes: sprites cover-fill the window (was: float-inside-margins + breathing overflow = "cropped" look), and direct expression requests ("smile for me") now visibly hold on the face (`detect_face_request` → held gesture) instead of doing nothing.
- Fixed a same-second session-id collision that could overwrite a session log when launched twice quickly.
- Silent `except: pass` blocks on failure-prone paths now log to `memory/health.log`.
- Added `requirements.txt`, `pyproject.toml`, and a 85-test pytest suite.

## What changed in the October liveliness pass (behavior preserved)

The brief: "conversational capabilities seem stiff and repetitive and not so alive." Every fix below came from a live symptom.

- **Repeat protection widened**: the guard now judges against a WINDOW of the last 6 replies (`REPEAT_WINDOW`), not just the previous turn — an alternating two-line deadlock was invisible to a one-turn guard. Rerolls stay bounded, off-screen (invariant 8), hotter each attempt (`REPEAT_REROLL_TEMPS`), and quote the forbidden line verbatim.
- **Baseline temperature is mood-derived with variety headroom** (`OllamaChat.DEFAULT_TEMPS`: 0.55 down / 0.85 excited / 0.78 neutral — up from 0.6–0.75): an 8B model sampled at 0.6–0.7 regenerates near-identical lines turn after turn. The repeat guard still overrides hotter when it fires.
- **The mood is alive between messages**: `mood_drift` adds a tiny random-walk step per turn (one beat per 45s), settles toward the hour's natural register (late nights quieter, mornings brighter), never pushes past the day anchor ±0.30, and dead exact-neutral is never a resting state. A wiring bug that cancelled drift out entirely was fixed along the way.
- **Negated feelings are not felt**: `analyze_emotional_shift` cancels the word after a negator AND an optional one-word hop, so "i'm not happy", "i dont feel great" and "why arent you happy" no longer register as compliments (they used to push valence UP).
- **State questions get honesty, not tautology**: `detect_state_question` + `is_pure_state_question` hand the model its real coordinates ("your ACTUAL mood right now: valence +0.12, energy -0.31"). A pure "how are you" gets a full honest answer about this moment; a mixed "how are you liking the game" is answered too, honestly about the mood; "because i'm not in a good mood" restating the mood as its own reason is explicitly forbidden.
- **Realisation thoughts deliver exactly once**: `_fire` either prints at a safe boundary (pending cleared — the user saw it) or parks on the NoticeBoard; a parked line is withdrawn and woven into the next reply instead (`withdraw_parked_realisations` + `NoticeBoard.withdraw`), never both. A fire that lands mid-reply reschedules instead of touching the one GPU.
- **Realisations are thoughts, not just questions**: `_generate_question` asks for one thought (usually a question, sometimes a take or a feeling), drawn from the CURRENT session first with freshest quotes; the fallback only speaks when the model is unreachable. Self-referential "what did I mean by..." lines stay rejected.
- **The face earns its restart budget back**: a window alive >60s regains one restart credit every >300s (`_note_stable_window`), so three harmless driver hiccups spread across days of uptime can no longer permanently kill the face for the rest of the session.
- **The system prompt is budget-aware (invariant 3)**: the rolling session summary is the ONE flexible tier — trimmed freshest-tail-first with a visible "[earlier summary trimmed...]" marker — while persona, voice block, OVERRIDE hard memory and directives are never traded away. Mood-typing mechanics de-duplicated between `personality.md` and `speech.md` (personality keeps the principle, speech keeps the quadrant mechanics); core directives tightened to three dense lines with the same semantics and the same pinned phrases.
- Tests grew 202 → 240 (repeat window, drift deltas/band/anchor pins, pure-vs-mixed state questions, negation, exactly-once delivery in every order, face budget regen, growth-note artifact filtering, and a prompt-budget pin on the real loaded mind files).

## What changed in the October dead-pipeline audit & voice pass

The brief: "audit the entire thing — no dead pipelines sitting idle, everything in harmony — and make him sound as human as possible on an 8b abliterated qwen."

**The audit (every subsystem traced end to end, not assumed):**

- **All pipelines confirmed live and wired**: semantic index (`MemoryRecall.semantic` → heartbeat warm-up), deep recall over every session, web lookup, `mood_drift`, `carry_momentum`, `is_pure_state_question`, parked-realisation withdraw, the wondering ledger (`mark_discussed`), face gestures + stable-window budget, vitality/sleep/dream consolidation, `growth_note`, `wake_greeting`, body senses, and the VT-mode path. Nothing was left running on assumptions.
- **Command surface verified against `/help`**: every advertised command dispatches. The five that looked orphaned (`/exit`, `/quit`, `/memory`, `/hard_mem`, `/soft_mem`) use set-membership rather than `==`, so a naive string-diff audit miscounts them.
- **Three genuinely dead functions removed or revived**: `enable_vt_mode()` existed but was **never called**, so ANSI color silently degraded to plain text on Windows (now runs first thing in `main()`); `semantic_available()` and `release_face_gesture()` had zero callers and were redundant with `SemanticIndex.available` and `set_face_gesture("neutral")` respectively (removed).

**The voice (every change measured against the live 8B, not guessed):**

- **Sampler dials moved off Ollama's defaults** (`OllamaChat.SAMPLING`): `repeat_penalty` 1.1 → 1.05 with `repeat_last_n` 64 → 32, `top_k` 40 → 64, plus `min_p: 0.05`. Ollama's own defaults were most of the android texture — 1.1 punishes the function words that legitimately repeat in texting ("i", "you", "the") and top_k 40 squeezes word choice toward the safest generic term. The repeat guard still owns true loops (it quotes the line and rerolls hotter), so the sampler only has to keep the degenerate tail out.
- **A reply-shape line rides in transient context** (`SHAPE_DIRECTIVE` via `mood_and_shape_block`): transient is the right home — it costs nothing against invariant 3 and is the freshest instruction in context when generation starts. Measured live: replies went from ~5 flat words ("got it", "you're welcome") to ~19 words of actual opinion, with question-closers staying at 0/6.
- **Hard memory is context, not small talk**: the OVERRIDE section now also says permanent facts are raised only when the conversation touches them — never as an opener, filler or closer (the model was using them as default small talk).
- **The flexible-tier ceiling now matches invariant 3** (`_PROMPT_BUDGET_CHARS` 5,856 → 5,600): the constant *is* what the summary trims against, so it had drifted above the invariant it enforced — a full-memory day trimmed to **5,855 chars ≈ 1,463 tokens**, over both invariant 3 and the budget test's own 1,450-token ceiling, and the light-summary test could not see it. A heavy-summary test now exercises the trim path and asserts it actually trims.
- **The butler/AI tell is now answered every turn** (`ANTI_BUTLER_DIRECTIVE`, riding alongside `SHAPE_DIRECTIVE` in transient context): it targets the failure that actually shows up in transcripts — an empty turn gets *acknowledged* (`got it. what's up?`, `you're welcome. i'm here.`, a help offer) instead of answered. Transient context is the only place this could live: the identical text in the system prompt cost enough tokens to breach invariant 3. Measured same-set through the real code path: reflexive check-ins 2/7 → **0/7**, and `ok.` stopped being acknowledged (`i get it. you're not sure. that's fine.`) in favour of him bringing up his own day (`i was just thinking about the move. it's gonna be weird, but i'm excited.`).
- **Four prompt approaches were tried, measured, and rejected** so they are never re-litigated: (1) literal few-shot dialogue examples made the 8B treat them as a **retrieval table** — it parroted `NO WAY / congrats thats huge / details now` verbatim and served the *pizza* line as its answer to a question about AI and jobs; (2) naming closers/questions in transient context made it ask a question in **5 of 6** replies — naming SHAPE only (length, line breaks, casing) is what works; (3) a wholesale **rewrite of the voice block** bought nothing and cost two new tells — question-closers 2→6 as it started *echoing the user's message as a question opener* (`remote work? \ni think it's cool.`), plus staccato fragments and a blown budget; (4) **quoting the forbidden phrases** (`("got it", "sure")`) *primed them* — the model replied `got it. you good?` and scored 7/21 butler hits, **worse than saying nothing**. The shipped line therefore describes the behavior and never names a phrase, and carries a load-bearing `complete sentences` clause — behavioural-only phrasing on its own collapsed replies into vapid staccato (`cool. yeah. nice.`).
- Tests grew 240 → 253 (sampler dials reach the payload and cannot clobber the mood dial, the transient block's content and shape-only constraint, VT-mode ordering, the heavy-summary trim, and the anti-butler line's no-quoting and flow constraints). Full suite 253/253, audit 35/35, system prompt 1,341/1,450 tokens light / ~1,400 on a heavy day.

## What is enforced by the audit (`py audit_pipelines.py`)

Pipeline 7 was added so this pass cannot silently regress. It now permanently fails the audit if:

- any production function is defined and never called anywhere (the check that found `enable_vt_mode`, `semantic_available` and `release_face_gesture` in the first place);
- the sampler dials (`OllamaChat.SAMPLING`) slide back to Ollama's defaults;
- the reply-shape line stops riding in transient context, or starts naming closers/questions again;
- the VOICE block stops being distilled out of `speech.md` (i.e. the whole 12k file, or nothing, reaches the prompt);
- any command `/help` advertises stops being dispatched;
- the anti-butler line stops riding in transient context, starts quoting a phrase it forbids, or loses its flow constraint.
