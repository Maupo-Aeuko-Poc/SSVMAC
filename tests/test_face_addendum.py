"""Stability-window budget regression: a face that has been quietly alive
earns its restart budget back, so ordinary driver hiccups spread across days
of uptime can never permanently kill the face (the silent black window).
"""

import threading
import time


class TestStableWindowBudget:
    def _make_face(self):
        from ui.face_display import EmotionalFace

        face = EmotionalFace.__new__(EmotionalFace)   # no window, no render loop
        face._lock = threading.Lock()
        face._running = True
        face._thread = threading.current_thread()    # a live thread, as required
        face._restarts = 3
        face._last_restart_at = time.monotonic() - 60
        face._last_stable_check = 0.0
        return face

    def test_restart_budget_is_earned_back_after_a_stable_stretch(self):
        face = self._make_face()
        face._last_restart_at = time.monotonic() - 301   # stable > 5 min
        face._last_stable_check = time.monotonic() - 61  # due a check
        face._note_stable_window()
        assert face._restarts == 2

    def test_young_window_earns_nothing_back(self):
        face = self._make_face()
        face._last_restart_at = time.monotonic() - 60   # alive but < 5 min
        face._last_stable_check = time.monotonic() - 61
        face._note_stable_window()
        assert face._restarts == 3

    def test_check_rate_is_throttled_to_once_a_minute(self):
        face = self._make_face()
        face._last_restart_at = time.monotonic() - 301
        face._last_stable_check = time.monotonic() - 10  # checked recently
        face._note_stable_window()
        assert face._restarts == 3

    def test_dead_face_earns_nothing_back(self):
        face = self._make_face()
        face._running = False        # a dead window earns nothing
        face._last_restart_at = time.monotonic() - 301
        face._last_stable_check = time.monotonic() - 61
        face._note_stable_window()
        assert face._restarts == 3
