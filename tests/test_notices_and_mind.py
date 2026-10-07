"""Tests for the NoticeBoard and the mind module."""

from maupo.mind import (EmotionalMatrix, build_system_prompt,
                        load_personality_and_speech)
from maupo.notices import NoticeBoard


class TestNoticeBoard:
    def test_announce_prints_when_idle(self, capsys):
        board = NoticeBoard()
        board.announce("[hello]")
        assert "[hello]" in capsys.readouterr().out

    def test_parks_while_prompt_open(self, capsys):
        board = NoticeBoard()
        board.open_prompt()
        board.announce("[parked]")
        assert capsys.readouterr().out == ""  # nothing through a live prompt
        board.close_prompt()
        board.flush()
        assert "[parked]" in capsys.readouterr().out

    def test_parks_while_generating(self, capsys):
        board = NoticeBoard()
        board.set_generating(True)
        board.announce("[parked]")
        assert capsys.readouterr().out == ""
        board.set_generating(False)
        board.flush()
        assert "[parked]" in capsys.readouterr().out

    def test_flush_drains_queue(self, capsys):
        board = NoticeBoard()
        board.open_prompt()
        board.announce("[one]")
        board.announce("[two]")
        board.close_prompt()
        board.flush()
        out = capsys.readouterr().out
        assert "[one]" in out and "[two]" in out
        board.flush()  # second flush prints nothing
        assert capsys.readouterr().out == ""


class TestEmotionalMatrix:
    def test_initializes_neutral_file(self, tmp_path):
        m = EmotionalMatrix(tmp_path / "emotional_matrix.md")
        assert m.get_state() == (0.0, 0.0)

    def test_set_state_clamps_and_persists(self, tmp_path):
        path = tmp_path / "emotional_matrix.md"
        m = EmotionalMatrix(path)
        m.set_state(2.0, -3.0)          # out of range on purpose
        assert m.get_state() == (1.0, -1.0)
        reloaded = EmotionalMatrix(path)  # survives a restart
        assert reloaded.get_state() == (1.0, -1.0)

    def test_quadrants(self, tmp_path):
        m = EmotionalMatrix(tmp_path / "emotional_matrix.md")
        m.set_state(0.8, 0.8)
        assert "Quadrant I" in m.get_quadrant()
        m.set_state(-0.8, -0.8)
        assert "Quadrant III" in m.get_quadrant()


class TestSpeechAndPrompt:
    def test_voice_block_extraction(self, tmp_path, monkeypatch):
        import maupo.mind as mind
        speech = tmp_path / "speech.md"
        speech.write_text(
            "# Speech\n\n<!-- VOICE:START -->\nshort bursts only\n<!-- VOICE:END -->\n\nmore\n",
            encoding="utf-8")
        monkeypatch.setattr(mind, "SPEECH_PATH", speech)
        _, voice = load_personality_and_speech()
        assert voice == "short bursts only"
        assert "more" not in voice  # the example plate stays on disk

    def test_hard_memory_overrides_language_in_prompt(self, tmp_path):
        from maupo.memory import HardMemory
        hard = HardMemory(tmp_path / "hard.md")
        hard.append("the user's name is Alex")
        prompt = build_system_prompt("P", "S", hard)
        assert "the user's name is Alex" in prompt
        assert "OVERRIDE" in prompt

    def test_prompt_stays_bounded_with_memory(self, tmp_path):
        from maupo.memory import HardMemory
        hard = HardMemory(tmp_path / "hard.md")
        for i in range(50):
            hard.append(f"fact number {i}")
        prompt = build_system_prompt("P", "S", hard, "session summary here")
        assert "fact number 49" in prompt
        assert "session summary here" in prompt

    def test_prompt_separates_takes_from_facts(self, tmp_path):
        """The anti-hallucination directive: owning a take is fine, inventing
        a fact ("arlo mccain" in red dead redemption) is not."""
        from maupo.memory import HardMemory
        prompt = build_system_prompt("P", "S", HardMemory(tmp_path / "hard.md"))
        assert "never invent names or details" in prompt.lower()
        assert "not sure" in prompt.lower()
