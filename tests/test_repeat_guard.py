"""The repeat guard: a reply that copies a RECENT reply verbatim is rerolled
until it differs.

Seen live: an 8B model answered "dont be in a neutral mood" with
"i'm just in a neutral mood." three times in a row - twice under the
"[said that twice - again, properly:]" banner - because the old reroll reused
the identical sampling settings and an instruction that never named the line
to avoid. Worse, the deadlock itself alternated between two lines across many
turns, so a guard watching only the immediately previous turn could never see
it. These tests pin the fix: quote the line, raise the temperature, bound the
retries, keep history honest, and judge against a WINDOW of recent replies
(REPEAT_WINDOW), not one turn.
"""

from __future__ import annotations

import qwen_chat

REPEATED = "because i'm not in a good mood."
FRESH = "honestly, today's been flat and i can't point at why"

# A short protocol line that keeps returning after being dismissed - exactly
# the alt-line deadlock the one-turn guard could never catch.
ALT = "im here, calm as ever, just keeping quiet."


def _seed_history(chat: _FakeChat, *replies: str) -> None:
    """Seed the window with past turns, in production shape: the guard runs
    AFTER chat.send() has appended the current input, so the seed's user line
    comes last (see _reroll below)."""
    for i, r in enumerate(replies):
        chat.history.append({"role": "user", "content": f"msg {i}"})
        chat.history.append({"role": "assistant", "content": r})
    # The current input has just been appended by send() in production:
    chat.history.append({"role": "user", "content": "why?"})


class _FakeChat:
    """Records sends, plays scripted replies, keeps history like OllamaChat."""

    def __init__(self, replies: list[str]) -> None:
        self.replies = list(replies)
        self.history: list[dict[str, str]] = []
        self.last_reply_norm = ""
        self.calls: list[dict] = []

    def send(self, user_input, transient_context="", *, valence=0.0, energy=0.0,
             stream_to_terminal=True, on_retry=None, temperature=None) -> str:
        reply = self.replies.pop(0)
        self.calls.append({"temperature": temperature, "stream": stream_to_terminal,
                           "context": transient_context})
        self.history.append({"role": "user", "content": user_input})
        self.history.append({"role": "assistant", "content": reply})
        self.last_reply_norm = qwen_chat._norm_reply(reply)
        return reply


def _reroll(chat: _FakeChat, first_reply: str, user_input: str = "why?"):
    """Mirror the production call shape: send() already appended the input into
    history (and streamed the first draft on screen) before the guard runs."""
    chat.history.append({"role": "user", "content": user_input})
    return qwen_chat.reroll_repeated_reply(
        chat, user_input, "[mood context]",
        valence=0.0, energy=0.0, first_reply=first_reply)


class TestRepeatGuard:
    def test_no_reroll_when_reply_differs(self):
        chat = _FakeChat([FRESH])
        reply, rerolls = _reroll(chat, FRESH)
        assert reply == FRESH
        assert rerolls == 0
        assert chat.calls == []          # the guard never even woke up

    def test_short_repeats_are_exempt(self):
        # Two-word answers legitimately repeat in real conversation.
        chat = _FakeChat([FRESH])
        reply, rerolls = _reroll(chat, "yeah.")
        assert reply == "yeah."
        assert rerolls == 0
        assert chat.calls == []

    def test_repeat_of_previous_turn_is_rerolled_once_and_succeeds(self):
        chat = _FakeChat([FRESH])
        _seed_history(chat, REPEATED)
        reply, rerolls = _reroll(chat, REPEATED)
        assert reply == FRESH
        assert rerolls == 1
        # The reroll was off-screen (invariant 8) and sampled hotter.
        assert chat.calls[0]["stream"] is False
        assert chat.calls[0]["temperature"] == qwen_chat.REPEAT_REROLL_TEMPS[0]
        # The instruction QUOTES the line it may not repeat - that is the fix.
        assert REPEATED in chat.calls[0]["context"]
        # History stays honest: exactly one turn for this input, the accepted one.
        assert chat.history[-1]["content"] == FRESH
        assert chat.history[-2]["content"] == "why?"
        assert sum(1 for m in chat.history if m["role"] == "user") == 2

    def test_repeat_of_older_window_reply_is_caught(self):
        """The alt-line deadlock: the repeated line is NOT the previous turn.

        The old guard compared against history[-2] only, so a conversation
        alternating A-B-A-B was forever invisible to it. The window guard
        sees all REPEAT_WINDOW recent replies.
        """
        chat = _FakeChat([FRESH])
        _seed_history(chat, ALT, REPEATED, ALT)
        reply, rerolls = _reroll(chat, ALT)
        assert reply == FRESH
        assert rerolls == 1

    def test_window_ignores_replies_that_fell_out_of_it(self):
        """After REPEAT_WINDOW newer turns, an old line is fair game again."""
        chat = _FakeChat([FRESH])
        # REPEAT_WINDOW+1 seeded turns: reply #0 is now the OLDEST and has
        # slid out of the -REPEAT_WINDOW: slice.
        _seed_history(chat, *(f"distinct reply number {i} for the window"
                              for i in range(qwen_chat.REPEAT_WINDOW + 1)))
        reply, rerolls = _reroll(chat, "distinct reply number 0 for the window")
        assert reply == "distinct reply number 0 for the window"
        assert rerolls == 0

    def test_window_judgment_counts_only_assistant_lines(self):
        """User lines in the window must never count as prior replies."""
        chat = _FakeChat([FRESH])
        _seed_history(chat, "something wildly repeated, definitely repeated")
        # A reply that equals the taped USER line is still fresh.
        reply, rerolls = _reroll(chat, "msg 0")
        assert rerolls == 0

    def test_stubborn_repeat_rerolls_twice_with_escalating_temperature(self):
        chat = _FakeChat([REPEATED, FRESH])
        _seed_history(chat, REPEATED)
        reply, rerolls = _reroll(chat, REPEATED)
        assert reply == FRESH
        assert rerolls == 2
        assert [c["temperature"] for c in chat.calls] == list(qwen_chat.REPEAT_REROLL_TEMPS)
        # Each discarded draft was rolled back before the next attempt: one
        # user line from the guard turns plus the original seed turn.
        assert sum(1 for m in chat.history if m["role"] == "user") == 2

    def test_both_rerolls_still_repeat_is_returned_honestly(self):
        chat = _FakeChat([REPEATED, REPEATED])
        _seed_history(chat, REPEATED)
        reply, rerolls = _reroll(chat, REPEATED)
        assert reply == REPEATED         # bounded: no infinite reroll
        assert rerolls == 2
        assert sum(1 for m in chat.history if m["role"] == "user") == 2

    def test_failed_reroll_restores_the_streamed_reply(self):
        class _RerollDies:
            def __init__(self) -> None:
                self.history: list[dict[str, str]] = []
                self.last_reply_norm = ""

            def send(self, user_input, transient_context="", **kwargs):
                if kwargs.get("temperature") is not None:
                    raise TimeoutError("ollama died mid-reroll")
                self.history.append({"role": "user", "content": user_input})
                self.history.append({"role": "assistant", "content": REPEATED})
                self.last_reply_norm = qwen_chat._norm_reply(REPEATED)
                return REPEATED

        chat = _RerollDies()
        _seed_history(chat, REPEATED)
        reply, rerolls = qwen_chat.reroll_repeated_reply(
            chat, "why?", "[mood context]",
            valence=0.0, energy=0.0, first_reply=REPEATED)
        # The already-streamed draft stays the truth; the turn is not eaten.
        assert reply == REPEATED
        assert rerolls == 0
        # The rollback that preceded the failed reroll was restored exactly.
        assert chat.history[-2]["role"] == "user"
        assert chat.history[-1]["content"] == REPEATED
        assert chat.last_reply_norm == qwen_chat._norm_reply(REPEATED)

    def test_no_window_yet_first_reply_ever(self):
        """A brand-new session has no history: nothing can be a repeat."""
        chat = _FakeChat([FRESH])
        reply, rerolls = _reroll(chat, FRESH)
        assert reply == FRESH
        assert rerolls == 0

    def test_empty_first_reply_is_returned_as_is(self):
        chat = _FakeChat([FRESH])
        reply, rerolls = _reroll(chat, "")
        assert reply == ""
        assert rerolls == 0
