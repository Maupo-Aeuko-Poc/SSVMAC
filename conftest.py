"""Root pytest configuration.

Tests must never touch the real memory/ directory (architecture invariant 5:
it is sacred ground), so every test gets a redirected health log and is
expected to create its SessionManager/SoftMemory/etc. under tmp_path.
"""

import pytest

from maupo import healthlog


@pytest.fixture(autouse=True)
def _redirect_health_log(tmp_path, monkeypatch):
    """Point maupo.healthlog at a per-test temp file."""
    log_path = tmp_path / "health.log"
    healthlog.set_log_path(log_path)
    yield log_path
    healthlog.set_hook(None)
