"""Per-turn transient voice block: mood + reply shape.

The system prompt is budgeted (invariant 3), so reply-shape guidance rides in
transient context instead - zero prompt cost, and it is the freshest
instruction in context when generation actually starts.

Pinned because it was measured live against the 8B model rather than assumed:
without the shape line replies collapsed to ~5 flat words ("got it",
"you're welcome"); with it they averaged ~19 words of actual opinion while
question-closers stayed at 0/6. Two earlier drafts that named closers or
questions directly made the model ask a question in 5 of 6 replies, so the
shape line must stay about SHAPE only.
"""

from __future__ import annotations

import inspect

import qwen_chat


class TestMoodAndShapeBlock:
    def test_carries_the_mood_it_was_handed(self):
        block = qwen_chat.mood_and_shape_block("somewhat positive with moderate energy",
                                               "you're full of energy")
        assert "somewhat positive with moderate energy" in block
        assert "you're full of energy" in block

    def test_carries_the_shape_line(self):
        assert qwen_chat.SHAPE_DIRECTIVE in qwen_chat.mood_and_shape_block("neutral", "relaxed")

    def test_mood_is_typed_from_never_announced(self):
        """The whole point of the emotional system: it colours the typing and
        is never reported."""
        block = qwen_chat.mood_and_shape_block("flat", "low")
        assert "never announce" in block

    def test_shape_line_names_shape_only(self):
        """Naming closers/questions here backfired on the 8B: it then asked a
        question in 5 of 6 replies. Shape only, no mention of endings."""
        shape = qwen_chat.SHAPE_DIRECTIVE.lower()
        assert "question" not in shape
        assert "closer" not in shape

    def test_block_is_built_as_one_transient_string(self):
        """It is appended to transient context, never stored in history."""
        block = qwen_chat.mood_and_shape_block("neutral", "relaxed")
        assert isinstance(block, str) and block.startswith("\n[")
        assert block.rstrip().endswith("]")


class TestAntiButlerDirective:
    # The phrases the model actually reached for in live transcripts.
    FORBIDDEN = ["got it", "you're welcome", "whatever you need", "here for you",
                 "i'm here if", "sure thing", "happy to help", "no problem",
                 "i hear you", "i got you", "no worries"]

    def test_rides_in_the_block(self):
        assert qwen_chat.ANTI_BUTLER_DIRECTIVE in qwen_chat.mood_and_shape_block("flat", "low")

    def test_quotes_nothing_it_forbids(self):
        """Measured against the live 8B: a draft that named ("got it", "sure")
        literally produced "got it. you good?" and scored 7/21 butler hits,
        worse than saying nothing at all. Quoting the phrase primes it, so the
        instruction must describe the behavior without ever naming a phrase."""
        text = qwen_chat.ANTI_BUTLER_DIRECTIVE.lower()
        named = [b for b in self.FORBIDDEN if b in text]
        assert not named, f"directive primes what it forbids: {named}"

    def test_keeps_the_flow_constraint(self):
        """The behavioural half alone collapsed replies into vapid staccato
        ("cool. yeah. nice."); the complete-sentences clause is load-bearing."""
        assert "complete sentences" in qwen_chat.ANTI_BUTLER_DIRECTIVE

    def test_avoids_the_word_that_caused_question_spam(self):
        """Naming closers/questions in transient context made the model ask a
        question in 5 of 6 replies. Shape and presence only."""
        text = qwen_chat.ANTI_BUTLER_DIRECTIVE.lower()
        assert "question" not in text and "closer" not in text


class TestVtModeWiring:
    def test_vt_mode_runs_before_the_first_output(self):
        """ANSI colors silently degrade to plain text on Windows unless VT
        mode is on, and it has to be on before anything prints."""
        src = inspect.getsource(qwen_chat.main)
        assert "enable_vt_mode()" in src
        assert src.index("enable_vt_mode()") < src.index("os.environ.get")
