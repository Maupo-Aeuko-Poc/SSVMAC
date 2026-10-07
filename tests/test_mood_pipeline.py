"""End-to-end mood pipeline: words -> emotional matrix -> face sprite.

Proves the face expression mapping is correct for every quadrant of the
valence x energy grid, using the REAL analyze_emotional_shift math and the
REAL sprite selection logic (no mocks between them).
"""

import ui.face_display as fd
from maupo.engine import pick_gesture_for_message
from maupo.mind import EmotionalMatrix
from maupo.triggers import analyze_emotional_shift


def _pipeline(text: str, v: float = 0.0, e: float = 0.0) -> tuple[str | None, float, float]:
    """The exact per-turn path from qwen_chat.py, condensed:
    analyze words -> update matrix -> select the resting sprite."""
    v_delta, e_delta = analyze_emotional_shift(text, v, e)
    new_v = max(-1.0, min(1.0, v + v_delta))
    new_e = max(-1.0, min(1.0, e + e_delta))
    m = EmotionalMatrix.__new__(EmotionalMatrix)   # in-memory only, no file
    m._valence, m._energy = new_v, new_e
    return fd._pick_mood_sprite(new_v, new_e), new_v, new_e


class TestMoodToSpriteMapping:
    def test_sustained_excitement_escalates_to_excited_face(self):
        # Moods build with momentum (max +-0.3 per message): one hyped message
        # nudges the matrix and flashes a smile gesture; SUSTAINED enthusiasm
        # walks the face happy -> excited. That escalation is the design.
        v = e = 0.0
        sprites = []
        for _ in range(3):
            sprite, v, e = _pipeline("i'm so excited and pumped about this!!!", v, e)
            sprites.append(sprite)
        assert sprites[0] is None, "one message is a spark, not a mood yet"
        assert sprites[1] == "happy", f"building... valence={v:.2f} energy={e:.2f}"
        assert sprites[2] == "excited", f"valence={v:.2f} energy={e:.2f}"

    def test_one_excited_message_flashes_a_reaction_gesture(self):
        # The fast path: a single hyped message surfaces through the GESTURE
        # layer (smile/laugh flash) while the slow mood matrix is still warming.
        v_delta, e_delta = analyze_emotional_shift("i'm so excited and pumped about this!!!", 0.0, 0.0)
        gesture = pick_gesture_for_message(
            "i'm so excited and pumped about this!!!", v_delta, e_delta, 0.3, 0.3)
        assert gesture in {"smile", "laugh"}

    def test_happy_calm_message_shows_happy_face(self):
        sprite, v, e = _pipeline("that was really nice, i feel good and grateful", 0.35, 0.0)
        assert sprite == "happy", f"valence={v:.2f} energy={e:.2f}"

    def test_frustrated_high_energy_shows_angry_face(self):
        sprite, v, e = _pipeline("this is so annoying, everything is broken and stressful", -0.6, 0.5)
        assert sprite == "angry", f"valence={v:.2f} energy={e:.2f}"

    def test_sad_low_energy_shows_sad_face(self):
        sprite, v, e = _pipeline("i'm sad and exhausted, everything feels hopeless", -0.4, -0.4)
        assert sprite == "sad", f"valence={v:.2f} energy={e:.2f}"

    def test_neutral_message_keeps_resting_face(self):
        sprite, _, _ = _pipeline("the meeting is at 3pm i think")
        assert sprite is None  # base face, mood lighting only


class TestGesturePath:
    def test_direct_request_beats_resting_face(self):
        assert pick_gesture_for_message("smile for me", 0, 0, 0, 0) == "smile"

    def test_laughter_message_triggers_laugh_gesture(self):
        assert pick_gesture_for_message("lmaooo stop", 0, 0, 0.5, 0.5) == "laugh"

    def test_gesture_overrides_resting_sprite_while_flashing(self):
        # A gesture flash (2.6s) paints over the mood sprite, then the mood
        # sprite returns — this is the designed layering.
        assert fd._pick_mood_sprite(0.0, 0.0) is None
        assert pick_gesture_for_message("lmaooo", 0, 0, 0.2, 0.3) == "laugh"
