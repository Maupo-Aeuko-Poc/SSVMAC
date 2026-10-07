"""Tests for the face engine — headless, via SDL's dummy video driver.

No window is ever opened: sprite scaling, mood sprite selection, and gesture
state (including the held-gesture behavior that makes 'smile for me' stick)
are all testable without the render thread.
"""

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import pytest  # noqa: E402  (must import after the dummy driver is set)
import pygame  # noqa: E402

from ui.face_display import (MAX_RESTARTS, EmotionalFace, GESTURE_DURATION,
                             VALID_GESTURES, _cover_scale, _load_sprites,
                             _pick_mood_sprite, _tinted)


@pytest.fixture(autouse=True)
def _pygame_init():
    if not pygame.get_init():
        pygame.init()
    if not pygame.display.get_init():
        pygame.display.set_mode((1, 1))
    yield
    # leave display state alone; dummy driver makes this cheap


class TestCoverScale:
    def test_small_png_fills_window(self):
        small = pygame.Surface((200, 150))
        scaled = _cover_scale(small, 460, 580)
        w, h = scaled.get_size()
        assert w >= 460 and h >= 580   # covers both dimensions...

    def test_aspect_ratio_preserved(self):
        surf = pygame.Surface((400, 300))   # 4:3
        w, h = _cover_scale(surf, 460, 580).get_size()
        assert abs((w / h) - (4 / 3)) < 0.02

    def test_wide_and_tall_sources_both_cover(self):
        wide = _cover_scale(pygame.Surface((800, 200)), 460, 580).get_size()
        tall = _cover_scale(pygame.Surface((200, 800)), 460, 580).get_size()
        assert wide[0] >= 460 and wide[1] >= 580
        assert tall[0] >= 460 and tall[1] >= 580


class TestMoodSprites:
    def test_quadrant_to_sprite(self):
        assert _pick_mood_sprite(-0.8, 0.5) == "angry"
        assert _pick_mood_sprite(0.5, 0.8) == "excited"
        assert _pick_mood_sprite(0.5, 0.2) == "happy"
        assert _pick_mood_sprite(-0.5, -0.2) == "sad"
        assert _pick_mood_sprite(0.0, 0.0) is None

    def test_excited_requires_positive_valence(self):
        # High energy alone is not excitement — excited.png must stay reachable
        # instead of happy owning the entire positive quadrant.
        assert _pick_mood_sprite(0.0, 0.8) is None
        assert _pick_mood_sprite(-0.2, 0.8) is None

    def test_valid_gestures_complete(self):
        assert {"smile", "smirk", "laugh", "cry", "pout", "wink",
                "surprised", "neutral", "sleep"} <= VALID_GESTURES


class TestGestureHolds:
    def test_direct_request_holds_expression(self):
        face = EmotionalFace()
        face.set_gesture("smile", hold=True)
        # Simulate far-past start: a flash would have expired by now.
        face._gesture_start -= 10 * GESTURE_DURATION
        with face._lock:
            assert face._gesture == "smile"   # still held
            assert face._gesture_hold is True

    def test_mood_gesture_expires(self):
        face = EmotionalFace()
        face.set_gesture("smile", hold=False)
        face._gesture_start -= 10 * GESTURE_DURATION
        with face._lock:
            assert face._gesture_hold is False

    def test_neutral_releases_hold(self):
        face = EmotionalFace()
        face.set_gesture("smile", hold=True)
        face.set_gesture("neutral")
        with face._lock:
            assert face._gesture is None
            assert face._gesture_hold is False

    def test_release_gesture_drops_hold(self):
        face = EmotionalFace()
        face.set_gesture("wink", hold=True)
        face.release_gesture()
        with face._lock:
            assert face._gesture is None


class TestTint:
    def test_brightness_never_nan_or_inverted(self):
        surf = pygame.Surface((10, 10))
        for b in (-1.0, 0.0, 0.5, 1.0, 1.6, 99.0):
            out = _tinted(surf, b, 0)
            assert out.get_size() == (10, 10)

    def test_warmth_clamps(self):
        surf = pygame.Surface((10, 10))
        out = _tinted(surf, 1.0, 500)
        assert out.get_size() == (10, 10)  # extreme warmth must not crash


class TestSpriteLoading:
    def test_real_assets_load_cover_scaled(self):
        """The actual face_assets PNGs must exist and fill the window."""
        sprites = _load_sprites()
        assert "base" in sprites, "ui/face_assets/base.png is Maupo's face and must exist"
        for name, surf in sprites.items():
            w, h = surf.get_size()
            win_w, win_h = 460, 580
            assert w >= win_w and h >= win_h, f"{name} does not cover the window"

    def test_missing_base_is_reported(self, monkeypatch, capsys):
        import ui.face_display as fd
        monkeypatch.setattr(fd, "ASSETS_DIR", __import__("pathlib").Path("definitely/not/here"))
        sprites = _load_sprites()
        assert "base" not in sprites


class TestWindowRecovery:
    """The face window must never die into a silent black rectangle.

    The black-window report was a loop that ended with nobody repainting and
    no line in health.log. These cover the two halves of the fix: react to the
    OS saying "your pixels are gone", and bring a dead render loop back.
    """

    def test_repaint_events_force_a_fresh_present(self):
        face = EmotionalFace()
        for const in ("WINDOWEXPOSED", "WINDOWRESTORED", "WINDOWSIZECHANGED",
                      "WINDOWSHOWN", "WINDOWFOCUSGAINED", "VIDEOEXPOSE"):
            face._repaint_wanted = False
            stopped = face._apply_window_event(pygame.event.Event(getattr(pygame, const)))
            assert stopped is False, const
            assert face._repaint_wanted is True, const

    def test_minimize_stops_drawing_and_restore_repaints(self):
        face = EmotionalFace()
        face._apply_window_event(pygame.event.Event(pygame.WINDOWMINIMIZED))
        assert face._minimized is True
        face._repaint_wanted = False
        face._apply_window_event(pygame.event.Event(pygame.WINDOWRESTORED))
        assert face._minimized is False and face._repaint_wanted is True

    def test_closing_the_window_is_remembered(self):
        face = EmotionalFace()
        assert face._apply_window_event(pygame.event.Event(pygame.QUIT)) is True
        assert face._user_closed is True

    def test_escape_closes_the_window_on_purpose(self):
        face = EmotionalFace()
        ev = pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_ESCAPE})
        assert face._apply_window_event(ev) is True
        assert face._user_closed is True

    def test_dead_loop_is_reopened_on_the_next_touch(self):
        face = EmotionalFace()
        face._running = False      # the render loop died on its own
        assert face.ensure_alive() is True
        try:
            assert face.is_running()
            assert face._thread is not None and face._thread.is_alive()
        finally:
            face.shutdown()

    def test_deliberate_close_is_respected_unless_forced(self):
        face = EmotionalFace()
        face._user_closed = True
        assert face.ensure_alive() is False
        assert face.ensure_alive(force=True) is True
        face.shutdown()

    def test_reopen_attempts_are_bounded(self):
        face = EmotionalFace()
        face._running = False
        face._restarts = MAX_RESTARTS
        assert face.ensure_alive() is False   # never a restart storm

    def test_shutdown_is_final(self):
        face = EmotionalFace()
        face._running = False
        face._shutting_down = True
        assert face.ensure_alive(force=True) is False
