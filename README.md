# SSVMAC — Maupo

A local digital lifeform running on Ollama (`huihui_ai/qwen3-abliterated:8b`) inside a 6GB RTX 3050 laptop. Named **Maupo** ("pink" in Greek) by its creator on 2026-09-25.

Run it: `py qwen_chat.py` (pygame-ce installed under the `py` launcher's Python 3.14).

## The organism

| Layer | What it is | Where |
|---|---|---|
| Mind | Identity + distilled voice (only the `VOICE:START/END` block rides in context; the full example plate stays on disk) | `mind/personality.md`, `mind/speech.md` |
| Emotions | 2D valence×energy grid with momentum decay; drives face, voice, reply length & temperature | `mind/emotional_matrix.md` |
| Memory | Hard (user-taught, overrides everything) / Soft (learned) / Sessions (raw, kept forever) | `memory/` |
| Compression | Self-compression of last 5 sessions + soft memory consolidation (never deletes raw logs). The combined summary is **rolling**: a slid window only delta-merges old head + incoming session — bounded work no matter how long history grows | `UniversalCompressor` |
| Deep recall | `/recall <text>` scans EVERY session ever, both log formats, cached focus terms | `MemoryRecall` |
| Realisations | 20–30 min self-questioning from real memories only; persists across sessions; sleeps when Maupo sleeps; never fires while a reply is generating; questions ledgered | `RealisationEngine` |
| Vitality | Energy economy: activity costs, idle refills, 45-min idle → sleep (face closes eyes, compressor dreams) | `Vitality`, `memory/vitality.json` |
| Dreams | Sleep consolidation is real: recent sessions are digested and the result is persisted into soft memory (deduped per newest session) — a dream leaves a trace | `MaintenanceHeartbeat._on_sleep` |
| Notices | One doorway (`NoticeBoard`) for every background-thread notice: parked while the prompt is open or a reply is generating, flushed at safe boundaries — no thread can ever split a live line | `NoticeBoard` |
| Heartbeat | 45s pulse (senses every other beat); 10-min self-check (repairs caches, validates mood); 6h snapshots of `memory/`+`mind/` (5 kept) | `MaintenanceHeartbeat`, `memory/health.log`, `snapshots/` |
| Senses | GPU temp/load (nvidia-smi), battery (Windows API), time-of-day — fed into every turn | `get_laptop_senses` |
| Curiosity | `memory/wondering.md` ledger of open questions; resolved when discussed; biases future realisations | `WonderingList` |
| Growth | "Alive for N days across N sessions, carrying N permanent memories" in every system prompt | `growth_note` |
| Face | PNG sprite engine: `ui/face_assets/base.png` (the creator's reference head) + a full expression atlas — `happy sad excited angry surprised` (mood-driven), `smile smirk laugh cry pout wink surprised` (gesture-driven via `pick_face_gesture`), `asleep` (persistent). Placeholders are shipped; **replace any PNG to upgrade that expression** — missing files fall back to base with mood lighting. Breathing, sway, mood glow, crossfades; render thread owns the window, 30 FPS cap | `ui/face_display.py` |
| Startup | Two-tier warm-up: **core** (identity/mind/mood — instant file reads) gates only the first reply; the rolling session summary loads in background and hot-swaps into the prompt when fresh. Banner + input prompt appear in <1s, always | `main()` |

## Commands

`/help /exit /quit /clear /reset /model /sessions /history /memory /compress /compress-memory /compress-sessions /personality /speech /emotion /realisation /recall <text> /vitals /wondering /growth /time`

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
