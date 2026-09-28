"""
Face Display Module for SSVMAC — PNG sprite engine (expression swaps).

Maupo's face is a set of PNGs in ui/face_assets/. `base.png` is the resting
face; every expression and gesture is its own sprite, auto-adopted if present:

    expressions : happy, sad, excited, angry, surprised
    gestures    : smile, smirk, laugh, cry, pout, wink
    eyes        : asleep (persistent)

Any file missing falls back to base.png with mood lighting, so the creator can
replace placeholders one at a time. The shipped placeholders ARE base art
(except asleep/wink, which have real closed-eye lines) — swap in real
expression art any time, no code changes.

Life is applied in code: breathing scale, idle sway, mood brightness/warmth,
gesture flashes, smooth crossfades between expressions (sleep swaps instantly).
The render thread OWNS the pygame window and caps at ~30 FPS, so the window can
never freeze the terminal.
"""

import math
import threading
import time
from pathlib import Path
from typing import Optional

import pygame

# Window / layout
FACE_WINDOW_SIZE = (460, 580)
BG_COLOR = (4, 9, 16)

# Animation settings
TARGET_FPS = 30
BREATH_AMPLITUDE = 0.010          # subtle scale pulse
BREATH_PERIOD = 4.6               # seconds
SWAY_AMPLITUDE = 4.0              # idle sway in pixels
SWAY_PERIOD = 7.5
CROSSFADE_SECONDS = 0.45
GESTURE_DURATION = 2.6
SLEEP_BRIGHTNESS = 0.45

ASSETS_DIR = Path(__file__).parent / "face_assets"

# mood -> optional sprite name
MOOD_SPRITES: list[tuple[str, str]] = [
    ("angry", "angry.png"),
    ("excited", "excited.png"),
    ("happy", "happy.png"),
    ("sad", "sad.png"),
]

VALID_GESTURES = {"smile", "smirk", "laugh", "cry", "pout", "surprised", "wink", "neutral", "sleep"}


def _pick_mood_sprite(valence: float, energy: float) -> Optional[str]:
    """Choose an optional expression sprite for the mood, if art exists.
    Order matters: the strongest, most specific mood wins."""
    if valence < -0.5 and energy > 0.3:
        return "angry"
    if energy > 0.6:
        return "excited"
    if valence > 0.3:
        return "happy"
    if valence < -0.3:
        return "sad"
    return None


def _load_sprites() -> dict[str, pygame.Surface]:
    """Load every usable PNG in face_assets, scaled to fit the window."""
    pygame.init()
    pygame.display.set_mode((1, 1))  # convert() needs a display
    sprites: dict[str, pygame.Surface] = {}
    if not ASSETS_DIR.exists():
        return sprites
    max_w, max_h = FACE_WINDOW_SIZE[0] - 40, FACE_WINDOW_SIZE[1] - 60
    for path in sorted(ASSETS_DIR.glob("*.png")):
        if path.name.startswith("_"):
            continue  # _reference.png etc.
        try:
            img = pygame.image.load(str(path)).convert()
            w, h = img.get_size()
            scale = min(max_w / w, max_h / h, 1.0)
            if scale < 1.0:
                img = pygame.transform.smoothscale(img, (int(w * scale), int(h * scale)))
            sprites[path.stem] = img
        except Exception as e:
            print(f"[face] could not load {path.name}: {e}")
    return sprites


def _tinted(sprite: pygame.Surface, brightness: float, warmth: float) -> pygame.Surface:
    """Return a brightness/warmth-adjusted copy of a sprite."""
    out = sprite.copy()
    b = max(0.0, min(1.6, brightness))
    warm = max(-40, min(40, int(warmth)))
    fill = (max(0, warm), max(0, int(30 * (b - 1))), max(0, int(40 * (b - 1))))
    if b <= 1.001 and warm <= 0:
        out.fill((int(255 * b), int(255 * b), int(255 * b)),
                 special_flags=pygame.BLEND_MULT)
        return out
    out.fill((int(255 * min(1.0, b)), int(255 * min(1.0, b)), int(255 * min(1.0, b))),
             special_flags=pygame.BLEND_MULT)
    if any(fill):
        out.fill(fill, special_flags=pygame.BLEND_ADD)
    return out


class EmotionalFace:
    """Owns the pygame window and a render thread that keeps the face alive."""

    def __init__(self):
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._screen = None
        self._clock: Optional[pygame.time.Clock] = None

        self._valence = 0.0
        self._energy = 0.0

        self._gesture: Optional[str] = None
        self._gesture_start = 0.0
        self._sleeping = False

        self._frames = 0

    # ------------------------------------------------------------------ setup
    def start(self) -> bool:
        """Spawn the render thread and wait briefly for it to open the window."""
        self._init_ok = threading.Event()
        self._thread = threading.Thread(target=self._render_loop, name="face-render", daemon=True)
        self._thread.start()
        if not self._init_ok.wait(timeout=8.0):
            self._running = False
            return False
        return self._running

    # ------------------------------------------------------------ public API
    def is_running(self) -> bool:
        return self._running

    def update_emotional_state(self, valence: float, energy: float) -> None:
        with self._lock:
            self._valence = max(-1.0, min(1.0, valence))
            self._energy = max(-1.0, min(1.0, energy))

    def set_gesture(self, gesture: str) -> None:
        if gesture not in VALID_GESTURES:
            return
        with self._lock:
            if gesture == "neutral":
                self._gesture = None
                self._sleeping = False
            elif gesture == "sleep":
                self._sleeping = True
                self._gesture = None
            else:
                self._sleeping = False
                self._gesture = gesture
                self._gesture_start = time.monotonic()

    def shutdown(self) -> None:
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._thread = None
        try:
            pygame.display.quit()
            pygame.quit()
        except Exception:
            pass

    # ----------------------------------------------------------- render loop
    def _render_loop(self) -> None:
        try:
            self._sprites = _load_sprites()
            if "base" not in self._sprites:
                raise RuntimeError(
                    f"ui/face_assets/base.png is missing — that file is Maupo's face.")
            self._screen = pygame.display.set_mode(FACE_WINDOW_SIZE)
            pygame.display.set_caption("Maupo")
            self._clock = pygame.time.Clock()
            self._running = True
        except Exception as e:
            print(f"Face display unavailable: {e}")
            self._running = False
            self._init_ok.set()
            return
        self._init_ok.set()

        # Crossfade state (thread-owned).
        self._shown: tuple[str, pygame.Surface] = ("base", self._sprites["base"])
        self._prev: Optional[tuple[str, pygame.Surface]] = None
        self._fade_t0 = 0.0

        while self._running:
            try:
                self._render_step()
            except Exception:
                self._running = False
                break

    def start(self) -> bool:
        """Spawn the render thread and wait briefly for it to open the window."""
        self._init_ok = threading.Event()
        self._thread = threading.Thread(target=self._render_loop, name="face-render", daemon=True)
        self._thread.start()
        if not self._init_ok.wait(timeout=8.0):
            self._running = False
            return False
        return self._running

    def _set_shown(self, name: str, instant: bool = False) -> None:
        surf = self._sprites.get(name)
        if not surf or name == self._shown[0]:
            return
        self._prev = self._shown
        self._shown = (name, surf)
        self._fade_t0 = time.monotonic()
        if instant:
            self._prev = None  # sleep switches don't crossfade

    def _render_step(self) -> None:
        try:
            now = time.monotonic()
            t_now = time.time()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self._running = False
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    self._running = False
            if not self._running:
                return

            with self._lock:
                valence, energy = self._valence, self._energy
                gesture = self._gesture
                gesture_progress = (now - self._gesture_start) / GESTURE_DURATION if self._gesture else 1.0

            if gesture and gesture_progress >= 1.0:
                with self._lock:
                    if self._gesture == gesture:
                        self._gesture = None
                gesture = None

            # ---- expression sprite selection (crossfaded; sleep instant) ----
            if self._sleeping:
                target = "asleep" if "asleep" in self._sprites else "base"
            elif gesture:
                target = gesture if gesture in self._sprites else "base"
            else:
                target = _pick_mood_sprite(valence, energy) or "base"
            if target != self._shown[0]:
                self._set_shown(target, instant=self._sleeping)

            # ---- life: breath, sway, brightness ----
            breath = 1.0 + BREATH_AMPLITUDE * math.sin(2 * math.pi * t_now / BREATH_PERIOD)
            if gesture == "laugh":
                breath *= 1.0 + 0.012 * math.sin(gesture_progress * math.pi * 4)
            sway = SWAY_AMPLITUDE * math.sin(2 * math.pi * t_now / SWAY_PERIOD)
            if self._sleeping:
                sway *= 0.3

            brightness = 0.86 + 0.14 * (energy * 0.5 + 0.5) + 0.06 * valence
            warmth = 14 * valence
            if self._sleeping:
                brightness = SLEEP_BRIGHTNESS
                warmth = 0.0
            elif gesture in ("smile", "laugh"):
                brightness += 0.12 * (1.0 - gesture_progress * gesture_progress)
            elif gesture == "surprised":
                brightness += 0.18 * math.sin(gesture_progress * math.pi)

            self._draw_scene(breath, sway, brightness, warmth)
        finally:
            # THE critical line: cap the loop at TARGET_FPS. Without it the
            # event pump spins uncapped, eats a CPU core and starves the GIL.
            if self._clock is not None:
                self._clock.tick(TARGET_FPS)

    # ------------------------------------------------------------- rendering
    def _draw_scene(self, breath: float, sway: float, brightness: float, warmth: float) -> None:
        if not self._screen:
            return
        self._frames += 1
        self._screen.fill(BG_COLOR)

        name, surf = self._shown
        alpha = 1.0
        if self._prev:
            alpha = min(1.0, (time.monotonic() - self._fade_t0) / CROSSFADE_SECONDS)
        alpha_ease = alpha * alpha * (3 - 2 * alpha)

        rect = surf.get_rect(center=(FACE_WINDOW_SIZE[0] // 2 + int(sway),
                                     FACE_WINDOW_SIZE[1] // 2 + 8))
        # Gentle breathing scale (recompute only when it actually changed).
        w, h = surf.get_size()
        tw, th = int(w * breath), int(h * breath)
        if (tw, th) != (w, h):
            shown = pygame.transform.smoothscale(surf, (tw, th))
        else:
            shown = surf
        shown = _tinted(shown, brightness, warmth)

        if self._prev and alpha_ease < 1.0:
            old_name, old_surf = self._prev
            old = old_surf if old_name == name else old_surf
            old_t = _tinted(old, brightness, warmth)
            old_rect = old.get_rect(center=rect.center)
            self._screen.blit(old_t, old_rect)
            shown.set_alpha(int(255 * alpha_ease))
            self._screen.blit(shown, rect)
        else:
            if self._prev:
                self._prev = None
            self._screen.blit(shown, rect)

        pygame.display.flip()


# Global face instance for easy access
_face_instance: Optional[EmotionalFace] = None
_face_lock = threading.Lock()


def get_face_instance() -> Optional[EmotionalFace]:
    global _face_instance
    with _face_lock:
        if _face_instance is None:
            face = EmotionalFace()
            if not face.start():
                return None
            _face_instance = face
    return _face_instance


def initialize_face() -> bool:
    """Initialize the face display. Returns True if the face window is running."""
    face = get_face_instance()
    return face is not None and face.is_running()


def update_face_emotion(valence: float, energy: float):
    """Update the face with new emotional state (safe to call from any thread)."""
    face = get_face_instance()
    if face and face.is_running():
        face.update_emotional_state(valence, energy)


def set_face_gesture(gesture: str):
    """Set a temporary gesture on the face ('smile', 'laugh', 'wink', 'surprised', 'neutral')."""
    face = get_face_instance()
    if face and face.is_running():
        face.set_gesture(gesture)


def cleanup_face():
    """Clean up the face display."""
    global _face_instance
    with _face_lock:
        if _face_instance:
            _face_instance.shutdown()
            _face_instance = None
