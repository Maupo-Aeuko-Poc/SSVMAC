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

Sprites are COVER-scaled: every PNG is zoomed until it fills the whole window
(small overflow is centered and cropped by the blit), so no art floats small in
the frame. Gentle breathing expands/contracts that fill instead of revealing
margins, and idle sway pans the filled face rather than sliding it around.

The window is self-healing. Anything that hints the OS invalidated it (expose,
restore, resize, focus) forces a fresh full present, and a slow safety repaint
means a stale window can never stay stale. A render loop that ends for any
reason never leaves a dead black rectangle behind: it is logged, the window is
closed, and the next turn reopens a clean one. "The face just went black and
nothing in health.log says why" is the bug class this module now makes
impossible.
"""

import math
import os
import threading
import time
from pathlib import Path
from typing import Optional

import pygame

try:
    from maupo.healthlog import log_health
except ImportError:  # face module imported outside the app tree
    def log_health(msg: str) -> None:
        print(f"[maupo health] {msg}")

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

# Self-healing limits
MAX_RESTARTS = 3                  # bounded: a dead window reopens, never storms
ERROR_STREAK_LIMIT = 90           # ~3s of consecutive bad frames = a truly dead loop
SAFETY_REPAINT_SECONDS = 5.0      # forced full repaint even with no OS hint

# Window events that mean "repaint from scratch" / "nothing is visible".
# pygame-ce dispatches these as their own event types (there is no
# pygame.WINDOWEVENT in 2.5), so they are collected defensively by name.
_REPAINT_WINDOW_EVENTS = {getattr(pygame, name) for name in (
    "WINDOWEXPOSED", "WINDOWSHOWN", "WINDOWRESTORED", "WINDOWSIZECHANGED",
    "WINDOWMOVED", "WINDOWFOCUSGAINED", "WINDOWTAKEFOCUS", "WINDOWENTER") if hasattr(pygame, name)}
_RESTORED_WINDOW_EVENTS = {getattr(pygame, name) for name in (
    "WINDOWRESTORED", "WINDOWSHOWN", "WINDOWMAXIMIZED") if hasattr(pygame, name)}
_CLOSED_WINDOW_EVENTS = {getattr(pygame, name) for name in ("QUIT", "WINDOWCLOSE") if hasattr(pygame, name)}
_MINIMIZED_WINDOW_EVENT = getattr(pygame, "WINDOWMINIMIZED", None)
_VIDEO_EXPOSE = getattr(pygame, "VIDEOEXPOSE", None)

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
    Order matters: the strongest, most specific mood wins. High-energy positive
    is excitement; only calmer positive moods fall through to happy — otherwise
    excited.png could never appear and happy would own the whole quadrant."""
    if valence < -0.5 and energy > 0.3:
        return "angry"
    if valence > 0.3 and energy > 0.6:
        return "excited"
    if valence > 0.3:
        return "happy"
    if valence < -0.3:
        return "sad"
    return None


def _cover_scale(img: pygame.Surface, box_w: int, box_h: int) -> pygame.Surface:
    """Scale `img` until it fully covers the box (like CSS background cover).

    The fill always touches all four edges of the window; whatever tiny part
    sticks out is centered and clipped by the blit. Never letterboxes, never
    floats smaller than the window.
    """
    w, h = img.get_size()
    if not w or not h:
        return img
    scale = max(box_w / w, box_h / h)
    # Round UP both edges: int() truncation could leave the fill one pixel
    # short of an edge — a hairline gap reads as a cropped sprite.
    return pygame.transform.smoothscale(img, (max(1, math.ceil(w * scale)),
                                              max(1, math.ceil(h * scale))))


def _load_sprites() -> dict[str, pygame.Surface]:
    """Load every usable PNG in face_assets, cover-scaled to fill the window."""
    pygame.init()
    pygame.display.set_mode((1, 1))  # convert() needs a display
    sprites: dict[str, pygame.Surface] = {}
    if not ASSETS_DIR.exists():
        return sprites
    max_w, max_h = FACE_WINDOW_SIZE
    for path in sorted(ASSETS_DIR.glob("*.png")):
        if path.name.startswith("_"):
            continue  # _reference.png etc.
        try:
            img = pygame.image.load(str(path)).convert()
            sprites[path.stem] = _cover_scale(img, max_w, max_h)
        except Exception as e:
            log_health(f"face: could not load {path.name}: {e}")
    return sprites


def _nudge_windows_repaint() -> None:
    """Ask Windows to repaint the face window. Never raises, never blocks.

    A window that a driver hiccup or a DWM restart left black is healed by a
    fresh invalidate followed by the next flip. On any other OS this is a
    no-op; on a window that is already healthy it costs one WM_PAINT that
    paints the same pixels again.
    """
    if os.name != "nt":
        return
    try:
        import ctypes
        hwnd = pygame.display.get_wm_info().get("window")
        if not hwnd:
            return
        RDW_INVALIDATE = 0x0001
        RDW_ALLCHILDREN = 0x0080
        ctypes.windll.user32.RedrawWindow(ctypes.c_void_p(int(hwnd)), None, None,
                                          RDW_INVALIDATE | RDW_ALLCHILDREN)
    except Exception:
        pass  # a repaint nudge is best-effort and platform-specific


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
        self._init_ok = threading.Event()

        self._valence = 0.0
        self._energy = 0.0

        self._gesture: Optional[str] = None
        self._gesture_start = 0.0
        self._gesture_hold = False
        self._sleeping = False

        # Self-healing state: the face must never sit on screen as a black
        # rectangle, and it must never die without leaving a trace.
        self._sprites: dict[str, pygame.Surface] = {}
        self._shutting_down = False
        self._user_closed = False     # the user closed the window themselves
        self._restarts = 0
        self._error_streak = 0
        self._repaint_wanted = True
        self._last_full_repaint = 0.0
        self._minimized = False

        self._frames = 0

    # ------------------------------------------------------------------ setup
    def start(self) -> bool:
        """Spawn the render thread and wait briefly for it to open the window."""
        return self._spawn()

    def _spawn(self) -> bool:
        """Open (or reopen) the window on a fresh render thread."""
        self._init_ok = threading.Event()
        self._running = False
        self._thread = threading.Thread(target=self._render_loop, name="face-render", daemon=True)
        self._thread.start()
        if not self._init_ok.wait(timeout=8.0):
            log_health("face window did not open within 8s")
            self._running = False
            return False
        return self._running

    def ensure_alive(self, force: bool = False) -> bool:
        """The face is up, or it gets brought back up. Never raises.

        A render loop can end for reasons the drawing code cannot see: the OS
        closes the window, the display surface is replaced under it, a driver
        hiccup invalidates everything. The old code simply stopped - leaving a
        black rectangle on screen and not one line in the health log. Now every
        ending is logged, the window is reopened on the next touch, and a
        deliberate close (the user hit X or Esc) is respected unless the face
        is explicitly asked for again with force=True.
        """
        if self._running and self._thread is not None and self._thread.is_alive():
            return True
        if self._shutting_down:
            return False
        if self._user_closed and not force:
            return False
        if self._restarts >= MAX_RESTARTS:
            return False
        self._restarts += 1
        self._last_restart_at = time.monotonic()
        self._user_closed = False
        log_health(f"face window is down - reopening it (attempt {self._restarts}/{MAX_RESTARTS})")
        try:
            pygame.display.quit()   # drop any half-dead window before reopening
        except Exception:
            pass
        try:
            return self._spawn()
        except Exception as e:
            log_health(f"face window could not be reopened: {e}")
            return False

    def _note_stable_window(self) -> None:
        """A window that has been alive for a long stretch earns its budget back.

        Without this, MAX_RESTARTS is spent once per PROCESS lifetime: three
        ordinary hiccups spread across days of uptime permanently kill the
        face (a silent dead face for the rest of the session - the exact
        failure ensure_alive exists to prevent). Cheap: one monotonic clock
        read per mood push, a decrement at most once a minute.
        """
        if self._running and self._thread is not None and self._thread.is_alive():
            now = time.monotonic()
            if now - getattr(self, "_last_stable_check", 0.0) > 60.0:
                if self._restarts and now - getattr(self, "_last_restart_at", 0.0) > 300.0:
                    self._restarts = max(0, self._restarts - 1)
                    self._last_restart_at = now
                self._last_stable_check = now

    # ------------------------------------------------------------ public API
    def is_running(self) -> bool:
        return self._running

    def update_emotional_state(self, valence: float, energy: float) -> None:
        self._note_stable_window()
        with self._lock:
            self._valence = max(-1.0, min(1.0, valence))
            self._energy = max(-1.0, min(1.0, energy))

    def set_gesture(self, gesture: str, hold: bool = False) -> None:
        """Flash a gesture for GESTURE_DURATION seconds — or hold it on the
        face (hold=True) when the user explicitly asked for it. A held gesture
        stays until release_gesture()/neutral/sleep or the next held request."""
        if gesture not in VALID_GESTURES:
            return
        with self._lock:
            if gesture == "neutral":
                self._gesture = None
                self._gesture_hold = False
                self._sleeping = False
            elif gesture == "sleep":
                self._sleeping = True
                self._gesture = None
                self._gesture_hold = False
            else:
                self._sleeping = False
                self._gesture = gesture
                self._gesture_start = time.monotonic()
                self._gesture_hold = hold

    def release_gesture(self) -> None:
        """Drop any held gesture; the mood sprite takes over again."""
        with self._lock:
            self._gesture = None
            self._gesture_hold = False

    def shutdown(self) -> None:
        self._shutting_down = True
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
        """Own the window for as long as it lives. Never exits silently."""
        self._error_streak = 0
        try:
            self._sprites = _load_sprites()
            if "base" not in self._sprites:
                raise RuntimeError(
                    f"ui/face_assets/base.png is missing — that file is Maupo's face.")
            self._screen = pygame.display.set_mode(FACE_WINDOW_SIZE)
            pygame.display.set_caption("Maupo")
            self._clock = pygame.time.Clock()
            # Crossfade state (thread-owned).
            self._shown: tuple[str, pygame.Surface] = ("base", self._sprites["base"])
            self._prev: Optional[tuple[str, pygame.Surface]] = None
            self._fade_t0 = 0.0
            self._repaint_wanted = True
            self._minimized = False
            self._running = True
        except Exception as e:
            log_health(f"face display unavailable: {e}")
            self._running = False
            self._init_ok.set()
            return
        self._init_ok.set()

        while self._running:
            try:
                self._render_step()
                self._error_streak = 0
            except BaseException as e:  # a dead face must never be a silent black window
                self._error_streak += 1
                if self._error_streak == 1 or self._error_streak % 50 == 0:
                    log_health(f"face render step failed ({self._error_streak}x): {e}")
                if self._error_streak >= ERROR_STREAK_LIMIT:
                    log_health(f"face render loop gave up after {self._error_streak} failed frames: {e}")
                    self._running = False
                else:
                    time.sleep(0.05)  # brief backoff, then try the next frame

        self._running = False
        if not self._shutting_down:
            # Leaving a stale window behind is exactly how the face went black:
            # nobody repaints it afterwards. Close it instead - the next turn
            # reopens a clean one through ensure_alive().
            try:
                pygame.display.quit()
            except Exception:
                pass
            log_health("face window closed by the user" if self._user_closed
                       else "face window closed unexpectedly - it reopens on the next turn")

    def _apply_window_event(self, event) -> bool:
        """Fold one pygame event into the loop's own state.

        Returns True when the render loop must stop: the window was closed or
        Esc was pressed, and either way a human did it on purpose.
        """
        if event.type in _CLOSED_WINDOW_EVENTS:
            self._user_closed = True
            return True
        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            self._user_closed = True
            return True
        if event.type == _MINIMIZED_WINDOW_EVENT:
            self._minimized = True
        elif event.type in _RESTORED_WINDOW_EVENTS:
            self._minimized = False
            self._repaint_wanted = True
        elif event.type in _REPAINT_WINDOW_EVENTS or event.type == _VIDEO_EXPOSE:
            # The OS says our pixels are gone: repaint everything.
            self._repaint_wanted = True
        return False

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
                if self._apply_window_event(event):
                    self._running = False
            if not self._running:
                return
            if self._minimized:
                return  # nothing is visible: don't burn a frame on it

            with self._lock:
                valence, energy = self._valence, self._energy
                gesture = self._gesture
                gesture_progress = (now - self._gesture_start) / GESTURE_DURATION if self._gesture else 1.0

            if gesture and gesture_progress >= 1.0 and not self._gesture_hold:
                with self._lock:
                    if self._gesture == gesture:
                        self._gesture = None
                        self._gesture_hold = False
                gesture = None

            # ---- never let the window sit stale ----
            # Where the window is drawn is not something a GDI/DWM hiccup asks
            # permission for: if the live display surface is no longer ours,
            # keep drawing into the live one instead of a dead surface forever.
            surface_now = pygame.display.get_surface()
            if surface_now is not self._screen:
                if surface_now is None:
                    raise RuntimeError("the face display surface disappeared")
                log_health("face display surface was replaced - re-acquiring it")
                self._screen = surface_now
                self._repaint_wanted = True

            # A full repaint whenever the OS hinted our pixels are invalid,
            # plus a slow safety repaint so no window can stay stale. Resetting
            # the crossfade kills any half-faded ghost from before.
            force_repaint = (self._repaint_wanted
                             or now - self._last_full_repaint >= SAFETY_REPAINT_SECONDS)
            if force_repaint:
                self._prev = None
                self._repaint_wanted = False
                self._last_full_repaint = now
                _nudge_windows_repaint()

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

            self._draw_scene(breath, sway, brightness, warmth, force_repaint)
        finally:
            # THE critical line: cap the loop at TARGET_FPS. Without it the
            # event pump spins uncapped, eats a CPU core and starves the GIL.
            if self._clock is not None:
                self._clock.tick(TARGET_FPS)

    # ------------------------------------------------------------- rendering
    def _draw_scene(self, breath: float, sway: float, brightness: float, warmth: float,
                    force_repaint: bool = False) -> None:
        if not self._screen:
            return
        self._frames += 1
        self._screen.fill(BG_COLOR)

        win_w, win_h = FACE_WINDOW_SIZE
        center = (win_w // 2 + int(sway), win_h // 2)

        def _present(surf: pygame.Surface) -> pygame.Surface:
            """Apply breathing scale + mood tint to a cover-scaled sprite."""
            w, h = surf.get_size()
            tw, th = max(1, int(w * breath)), max(1, int(h * breath))
            shown = pygame.transform.smoothscale(surf, (tw, th)) if (tw, th) != (w, h) else surf
            return _tinted(shown, brightness, warmth)

        name, surf = self._shown
        alpha = 1.0
        if self._prev:
            alpha = min(1.0, (time.monotonic() - self._fade_t0) / CROSSFADE_SECONDS)
        alpha_ease = alpha * alpha * (3 - 2 * alpha)

        if self._prev and alpha_ease < 1.0:
            old_name, old_surf = self._prev
            old_rect = old_surf.get_rect(center=center)
            self._screen.blit(_present(old_surf), old_rect)  # overflow crops at edges
            shown = _present(surf)
            shown.set_alpha(int(255 * alpha_ease))
            self._screen.blit(shown, shown.get_rect(center=center))
        else:
            if self._prev:
                self._prev = None
            shown = _present(surf)
            self._screen.blit(shown, shown.get_rect(center=center))

        pygame.display.flip()
        if force_repaint:
            # After an invalidate some drivers only present on the second flip;
            # one extra blit is cheap insurance against a black window.
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
    if face and face.ensure_alive():
        face.update_emotional_state(valence, energy)


def set_face_gesture(gesture: str, hold: bool = False):
    """Set a gesture on the face ('smile', 'laugh', 'wink', 'surprised', 'neutral').
    hold=True keeps it up until explicitly released (direct user requests)."""
    face = get_face_instance()
    if face and face.ensure_alive():
        face.set_gesture(gesture, hold=hold)


def ensure_face_window(force: bool = False) -> bool:
    """The face window exists, reopening it if it died. True when it is up.

    force=True is for a deliberate user request ('/face smile'): it overrides
    an earlier deliberate close instead of respecting it.
    """
    face = get_face_instance()
    return bool(face and face.ensure_alive(force=force))


def cleanup_face():
    """Clean up the face display."""
    global _face_instance
    with _face_lock:
        if _face_instance:
            _face_instance.shutdown()
            _face_instance = None


def list_face_sprites() -> list[str]:
    """Every expression sprite Maupo can currently wear (art present on disk)."""
    try:
        return sorted(p.stem for p in ASSETS_DIR.glob("*.png")
                      if not p.name.startswith("_") and p.stem != "base")
    except Exception:
        return []
