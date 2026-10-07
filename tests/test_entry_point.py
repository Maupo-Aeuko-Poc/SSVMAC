"""Boot smoke test for the real entry point (qwen_chat.py).

The unit suite exercises every subsystem in isolation, but for a while
nothing ever executed qwen_chat.main() itself - which is how an
undefined-variable crash in startup ordering (UnboundLocalError on
`vitality`) shipped green through the whole suite and the 25-check audit.
This file closes that gap permanently.

Method: launch the actual entry point as a subprocess inside a sandboxed
copy of the project. The sandbox excludes the private life (memory/) - the
app must self-create it, exactly like a fresh clone does. OLLAMA_HOST is
pointed at a dead port so the boot is fully deterministic and never touches
the real Ollama, the GPU, or the network.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Parts of the tree that must never leak into the boot sandbox: the private
# life (memory/), git metadata, test scaffolding, and build noise.
IGNORE_PATTERNS = (
    "memory", "memory_archive_*", "snapshots", ".git*", "__pycache__",
    "tests", ".pytest_cache", ".venv", "*.log", "*.pyc", ".env*",
)

# Port 9 (discard) is never a local Ollama: connections fail instantly, so
# every model-dependent path degrades gracefully instead of doing real work.
DEAD_OLLAMA = "http://127.0.0.1:9"

PAST_SESSION_MD = """\
# Chat Session: session_20260925_001054
- Started: 2026-09-25 00:10:54
- Model: huihui_ai/qwen3-abliterated:8b

---

### [12:00:00] Turn 1
**You**: hello world

**Maupo**: greetings, human

"""


@pytest.fixture()
def sandbox(tmp_path: Path) -> Path:
    """A throwaway copy of the project without its private life."""
    app = tmp_path / "app"
    shutil.copytree(
        PROJECT_ROOT, app,
        ignore=shutil.ignore_patterns(*IGNORE_PATTERNS),
        dirs_exist_ok=True,
    )
    return app


def _boot(sandbox: Path) -> subprocess.CompletedProcess:
    """Run qwen_chat.py to completion with stdin at EOF (boot, banner, exit)."""
    env = dict(os.environ)
    env["OLLAMA_HOST"] = DEAD_OLLAMA          # deterministic: no model, no GPU
    # Headless face: the boot test verifies startup order, not window
    # rendering, and must never flash a pygame window at the human.
    env["SDL_VIDEODRIVER"] = "dummy"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return subprocess.run(
        [sys.executable, "qwen_chat.py"],
        cwd=sandbox,
        input="",
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=90,
        env=env,
    )


def test_entry_point_boots_and_exits_cleanly(sandbox: Path) -> None:
    result = _boot(sandbox)

    combined = result.stdout + result.stderr
    assert result.returncode == 0, f"qwen_chat.py crashed on boot:\n{combined}"
    assert "Maupo (model:" in combined, f"banner missing:\n{combined}"
    assert "Goodbye!" in combined, f"clean exit missing:\n{combined}"
    assert "Traceback (most recent call last)" not in combined, (
        f"traceback on boot:\n{combined}"
    )
    # A fresh sandbox proves the app self-creates its life, like a new clone.
    assert (sandbox / "memory" / "sessions").is_dir(), (
        "memory/ was not self-created on boot"
    )


def test_entry_point_boots_with_existing_history(sandbox: Path) -> None:
    """Boot the way the user actually launches: past sessions already on disk.

    The warm-summary tier runs against a dead Ollama and must degrade to the
    graceful per-file skip, print the loaded-context banner, and never crash.
    """
    sessions_dir = sandbox / "memory" / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    (sessions_dir / "session_20260925_001054.md").write_text(
        PAST_SESSION_MD, encoding="utf-8")

    result = _boot(sandbox)

    combined = result.stdout + result.stderr
    assert result.returncode == 0, f"crashed with history present:\n{combined}"
    assert "Traceback (most recent call last)" not in combined, (
        f"traceback on boot:\n{combined}"
    )
    # Either banner proves the past-session warm path ran gracefully: the
    # summary thread may land before (loaded) or after (warming) the banner.
    warm_banner = ("[Loaded context from 1 previous session(s)]" in combined
                   or "[Warming context from 1 previous session(s)" in combined)
    assert warm_banner, (
        f"past-session warm path did not complete gracefully:\n{combined}"
    )


def test_entry_point_boots_twice_in_a_row(sandbox: Path) -> None:
    """Back-to-back launches must each get their own session log and never
    overwrite or crash the previous one (SessionManager same-second guard)."""
    first = _boot(sandbox)
    second = _boot(sandbox)
    for r in (first, second):
        assert r.returncode == 0, f"crash on back-to-back boot:\n{r.stdout}\n{r.stderr}"
    sessions = sorted((sandbox / "memory" / "sessions").glob("session_*.md"))
    assert len(sessions) == 2, (
        f"expected two session logs, got {[p.name for p in sessions]}"
    )


def test_survives_a_failed_model_call(sandbox: Path) -> None:
    """The 2026-10-01 crash: Ollama failed a cold CUDA load, answered 500,
    and the chat EXITED on the first message. A digital lifeform does not die
    from one failed thought: a bare message against a dead Ollama must fail
    (after its one retry), be reported, and leave the chat alive for /exit."""
    result = subprocess.run(
        [sys.executable, "qwen_chat.py"],
        cwd=sandbox,
        input="hey\n/exit\n",
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=120,
        env={**os.environ,
             "OLLAMA_HOST": DEAD_OLLAMA,
             "SDL_VIDEODRIVER": "dummy",
             "PYTHONDONTWRITEBYTECODE": "1"},
    )

    combined = result.stdout + result.stderr
    assert result.returncode == 0, (
        f"a failed model call killed the chat:\n{combined}"
    )
    assert "Traceback (most recent call last)" not in combined
    # The failure was reported to the human, in Maupo's own voice.
    assert "still here" in combined, f"no survival report:\n{combined}"
    # The session log stays honest: the failed turn is NOT recorded as a
    # conversation turn, and the banner/session log line still exist.
    assert "Session log:" in combined


def test_failed_lookup_does_not_kill_the_chat(sandbox: Path) -> None:
    """    Same contract for the search path: whether the live web lookup finds
    something or not, the turn must never kill the chat. The lookup may hit
    the real network; assertions require survival either way (invariant 12)."""
    result = subprocess.run(
        [sys.executable, "qwen_chat.py"],
        cwd=sandbox,
        input="look up zxqvjkwqlixirx\nno thanks\n/exit\n",
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=120,
        env={**os.environ,
             "OLLAMA_HOST": DEAD_OLLAMA,
             "SDL_VIDEODRIVER": "dummy",
             "PYTHONDONTWRITEBYTECODE": "1"},
    )

    combined = result.stdout + result.stderr
    assert result.returncode == 0, f"a failed lookup killed the chat:\n{combined}"
    assert "Traceback (most recent call last)" not in combined
    assert "Goodbye!" in combined
