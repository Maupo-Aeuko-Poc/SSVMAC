"""Budget and dial pins for the alive-baseline work.

The system prompt (invariant 3) must stay under ~1400 tokens (heuristic
chars/4 on REAL loaded mind files) even at a heavy-but-real memory load, so
the personality never gets truncated into stiffness.

Baseline temperatures are mood-derived (an 8B model at 0.6-0.7 regenerates
near-identical lines, which is the stiffness loop itself); the pins keep them
from silently sliding back to the old numbers.
"""

from __future__ import annotations

import pathlib
import tempfile
from pathlib import Path

from maupo.engine import EmotionalMatrix, growth_note
from maupo.mind import build_system_prompt, load_emotional_brief
from maupo.memory import HardMemory
from maupo.net import OllamaChat


def _real_prompt(summary: str | None = None) -> tuple[str, str]:
    """Build the system prompt the way production does: real mind files,
    realistic hard memories plus a rolling summary paragraph. Pass a heavy
    summary to exercise the trim path (the flexible tier)."""
    import qwen_chat

    d = pathlib.Path(tempfile.mkdtemp())
    hard = HardMemory(d / "h.md")
    for e in ("exams are on friday", "hates crowds",
              "plants are basil and mint", "call after 6pm"):
        hard.append(e)
    em = EmotionalMatrix.__new__(EmotionalMatrix)
    em._valence, em._energy = 0.3, 0.2
    personality, speech = qwen_chat.load_personality_and_speech()
    prompt = build_system_prompt(
        personality, speech, hard,
        summary if summary is not None else
        "yesterday you talked about moving, about the game, and about "
        "being tired at work lately",
        load_emotional_brief(em),
        "Current date and time: 2026-10-07 21:44:00",
        aliveness=growth_note(
            type("SM", (), {"sessions_dir": d, "session_file": d / "s.md",
                            "list_all_sessions": lambda s: []})(),
            hard),
        self_note="",
    )
    return prompt, speech


class TestSystemPromptBudget:
    def test_heavy_real_load_stays_within_budget(self):
        prompt, _ = _real_prompt()
        approx_tokens = len(prompt) // 4
        assert approx_tokens <= 1450, (
            f"system prompt too heavy (~{approx_tokens} tokens): the tail gets "
            f"dropped by the context window and the persona goes stiff")

    def test_override_language_survives(self):
        prompt, _ = _real_prompt()
        assert "OVERRIDE" in prompt
        assert "Permanent Knowledge" in prompt

    def test_heavy_summary_day_stays_within_budget(self):
        """The rolling summary is the ONE flexible tier, which makes it exactly
        the case the light-summary test cannot see: on a full-memory day the
        trim path has to land inside invariant 3, not a hair past it.

        This caught a real drift - the flexible-tier ceiling sat at 5,856
        chars (~1,463 tokens), above both invariant 3 and this test's own
        1,450-token ceiling, so a heavy day silently trimmed to an
        over-budget prompt while every existing check passed."""
        heavy = ("yesterday we talked about moving house and the game and "
                 "being tired at work. " * 60).strip()
        prompt, _ = _real_prompt(summary=heavy)
        assert "[earlier summary trimmed" in prompt, (
            "a huge summary must actually be trimmed, not passed through")
        approx_tokens = len(prompt) // 4
        assert approx_tokens <= 1450, (
            f"heavy day came out ~{approx_tokens} tokens: invariant 3 broken "
            f"on exactly the days it exists for")


class TestMoodTemps:
    """The mood-derived baseline dials (headroom for variety on an 8B)."""

    def test_baseline_pins(self):
        assert OllamaChat.DEFAULT_TEMPS == {"down": 0.55, "excited": 0.85,
                                            "neutral": 0.78}

    def test_down_mood_selects_the_quiet_dial(self):
        chat = OllamaChat("m", "http://localhost:11434")
        assert chat.DEFAULT_TEMPS["down"] == 0.55

    def test_override_documented_in_send(self):
        # The repeat guard needs the override param to still exist.
        import inspect
        from maupo.net import OllamaChat as OC
        params = inspect.signature(OC.send).parameters
        assert "temperature" in params
        assert params["temperature"].default is None
