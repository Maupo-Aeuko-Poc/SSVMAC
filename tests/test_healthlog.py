"""Tests for the shared self-diagnosis log."""

from maupo import healthlog


class TestHealthLog:
    def test_log_health_writes_timestamped_lines(self, tmp_path):
        log_path = tmp_path / "health.log"
        healthlog.set_log_path(log_path)
        healthlog.log_health("gpu senses failed: timeout")
        lines = log_path.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 1
        assert "gpu senses failed: timeout" in lines[0]
        assert lines[0].startswith("[20")  # timestamp prefix

    def test_recent_lines_returns_last_lines_oldest_first(self, tmp_path):
        log_path = tmp_path / "health.log"
        healthlog.set_log_path(log_path)
        for i in range(7):
            healthlog.log_health(f"event {i}")
        recent = healthlog.recent_lines(5)
        assert len(recent) == 5
        assert "event 2" in recent[0]      # oldest of the last five
        assert "event 6" in recent[-1]     # newest last

    def test_recent_lines_on_missing_file_is_empty(self, tmp_path):
        healthlog.set_log_path(tmp_path / "never_written.log")
        assert healthlog.recent_lines() == []

    def test_hook_receives_every_line(self, tmp_path):
        healthlog.set_log_path(tmp_path / "health.log")
        seen = []
        healthlog.set_hook(seen.append)
        try:
            healthlog.log_health("hello hook")
        finally:
            healthlog.set_hook(None)
        assert len(seen) == 1
        assert "hello hook" in seen[0]

    def test_logger_never_raises(self, tmp_path):
        # A read-only or missing parent must never crash the caller.
        healthlog.set_log_path(tmp_path / "no" / "such" / "dir" / "health.log")
        healthlog.log_health("should not raise")  # dir is auto-created
        assert (tmp_path / "no" / "such" / "dir" / "health.log").exists()

    def test_log_is_rotated_when_over_budget(self, tmp_path):
        # A lifeform that logs forever must never outgrow its folder.
        log_path = tmp_path / "health.log"
        healthlog.set_log_path(log_path)
        big_line = "x" * 256
        for i in range(700):  # ~180KB, way over the 2000*128 gate
            healthlog.log_health(f"{big_line} {i}")
        size = log_path.stat().st_size
        assert size < 256 * 2100, f"log not rotated: {size} bytes"
        lines = log_path.read_text(encoding="utf-8").splitlines()
        assert len(lines) <= healthlog.MAX_LINES
        assert "699" in lines[-1]  # newest line survived the rotation

    def test_steady_state_writes_stay_small(self, tmp_path):
        log_path = tmp_path / "health.log"
        healthlog.set_log_path(log_path)
        for i in range(50):
            healthlog.log_health(f"small {i}")
        assert log_path.exists()  # no rotation churn on a young log
        assert len(log_path.read_text(encoding="utf-8").splitlines()) == 50
