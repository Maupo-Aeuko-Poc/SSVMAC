"""Triggers addendum: the negation fix, the living-baseline drift, and the
pure-vs-mixed state question split.

Each one came from a live failure:
- "why arent you happy" registered 'happy' as a compliment and pushed valence
  UP while he was being asked why he was down,
- a mood that only moves when spoken to sits at dead neutral forever (the
  stiffness loop), and
- "how are you liking the game" deserved the same honesty nudge as "how are
  you" but is NOT the same question.
"""

from __future__ import annotations

import random
from datetime import datetime

from maupo.triggers import (_TIME_OF_DAY_MOOD, _MOOD_DRIFT_BAND,
                            _MOOD_DRIFT_MIN_GAP_SECONDS, _MOOD_DRIFT_STEP,
                            _time_of_day_bucket, analyze_emotional_shift,
                            is_pure_state_question, mood_drift)


class TestNegationFix:
    def test_negated_happy_never_lifts_valence(self):
        # THE bug: the 'happy' in "why arent you happy" fed the positive set.
        # The sentence is a question about him being DOWN.
        dv, de = analyze_emotional_shift("why arent you happy", 0.2, 0.3)
        assert dv <= 0, f"negated 'happy' lifted valence: {dv}"

    def test_explicit_negated_words_are_removed(self):
        dv, _ = analyze_emotional_shift("i'm not happy about this", 0.0, 0.0)
        assert dv <= 0, f"'not happy' must not be positive: {dv}"

    def test_contractions_negate_too(self):
        for text in ("i dont feel great today", "i can't enjoy this",
                     "the party wasnt fun"):
            dv, _ = analyze_emotional_shift(text, 0.0, 0.0)
            assert dv <= 0.01, f"{text!r} lifted valence: {dv}"

    def test_plain_positives_still_work(self):
        dv, _ = analyze_emotional_shift("i'm so happy today, everything is great", 0.0, 0.0)
        assert dv > 0

    def test_per_message_cap_still_holds(self):
        text = "i'm so excited and pumped, this is amazing incredible fantastic"
        dv, de = analyze_emotional_shift(text, 0.0, 0.0)
        assert abs(dv) <= 0.3 + 1e-9 and abs(de) <= 0.3 + 1e-9


class TestMoodDrift:
    def _fixed_now(self):
        return datetime(2026, 10, 1, 14, 0, 0)   # stable afternoon anchor

    def test_min_gap_is_a_noop(self):
        now = self._fixed_now()
        dv, de, ts = mood_drift(0.4, -0.2, last_drift_ts=now.timestamp(),
                                now=now, rng=random.Random(7))
        assert (dv, de) == (0.0, 0.0)
        assert ts == now.timestamp()             # next-eligible beat unchanged

    def test_after_gap_returns_a_delta_and_new_timestamp(self):
        now = self._fixed_now()
        dv, de, ts = mood_drift(0.0, 0.0, last_drift_ts=0.0,
                                now=now, rng=random.Random(7))
        assert ts == now.timestamp()
        # A beat moves the mood on the scale of one step (step + settle pull).
        assert abs(dv) <= _MOOD_DRIFT_STEP * 2 and abs(de) <= _MOOD_DRIFT_STEP * 2

    def test_never_pushes_mood_past_anchor_band(self):
        """Ambience never drags him beyond the band AROUND the hour anchor."""
        for seed in range(24):
            rng = random.Random(seed)
            anchor_v, anchor_e = _TIME_OF_DAY_MOOD[_time_of_day_bucket(self._fixed_now())]
            dv, de, _ = mood_drift(anchor_v + _MOOD_DRIFT_BAND + 0.1,
                                   anchor_e + _MOOD_DRIFT_BAND + 0.1,
                                   last_drift_ts=0.0, now=self._fixed_now(), rng=rng)
            new_v = anchor_v + _MOOD_DRIFT_BAND + 0.1 + dv
            new_e = anchor_e + _MOOD_DRIFT_BAND + 0.1 + de
            assert new_v <= anchor_v + _MOOD_DRIFT_BAND + 1e-9, (seed, dv)
            assert new_e <= anchor_e + _MOOD_DRIFT_BAND + 1e-9, (seed, de)

    def test_far_mood_is_brought_home_to_the_band_edge(self):
        """Past the band, the stale-right-before shows itself: drift pins the
        mood exactly to the band edge (never drifts deeper out)."""
        now = self._fixed_now()
        anchor_v, _ = _TIME_OF_DAY_MOOD[_time_of_day_bucket(now)]
        start = anchor_v + _MOOD_DRIFT_BAND + 0.15
        dv, _, _ = mood_drift(start, 0.0, last_drift_ts=0.0, now=now,
                              rng=random.Random(1))
        assert dv < 0                      # heading home
        assert abs(start + dv - (anchor_v + _MOOD_DRIFT_BAND)) < 1e-9

    def test_neutral_is_never_a_resting_state(self):
        """Dead-center moods always sidle off, in random directions."""
        ups = downs = 0
        for seed in range(200):
            dv, de, _ = mood_drift(0.0, 0.0, last_drift_ts=0.0,
                                   now=self._fixed_now(), rng=random.Random(seed))
            assert dv != 0.0 and de != 0.0, f"seed {seed} stayed pinned at 0"
            ups += dv > 0
            downs += dv < 0
        assert ups > 40 and downs > 40     # no preferred corner

    def test_anchor_pull_into_the_band(self):
        """Inside the band it settles toward the hour's natural register."""
        now = self._fixed_now()
        anchor_v, _ = _TIME_OF_DAY_MOOD[_time_of_day_bucket(now)]
        bv, be, _ = mood_drift(-0.5, 0.0, last_drift_ts=0.0, now=now,
                               rng=random.Random(3))
        assert bv > -0.5                   # pulled upward toward the anchor


class TestTimeOfDayBucket:
    def test_buckets(self):
        assert _time_of_day_bucket(datetime(2026, 10, 1, 3, 0)) == "late night"
        assert _time_of_day_bucket(datetime(2026, 10, 1, 9, 0)) == "morning"
        assert _time_of_day_bucket(datetime(2026, 10, 1, 14, 0)) == "afternoon"
        assert _time_of_day_bucket(datetime(2026, 10, 1, 20, 0)) == "evening"
        assert _time_of_day_bucket(datetime(2026, 10, 1, 23, 0)) == "late night"


class TestPureStateQuestion:
    def test_pure_state_questions(self):
        for text in ("how are you", "how are you feeling right now",
                     "are you ok?", "why arent you happy", "you seem quiet",
                     "so how are you doing today", "hey, are you tired?"):
            assert is_pure_state_question(text), text

    def test_mixed_is_not_pure(self):
        for text in ("how are you liking the game",
                     "how are you so good at this",
                     "are you okay with pizza tonight",
                     "why are you not helping with the dishes"):
            assert not is_pure_state_question(text), text
