"""Tests for self-evolution: Maupo identifying and editing his own body.

Everything runs against a miniature copy of his body in tmp_path - the real
mind/ and maupo/ files are never touched, and the model is always faked. The
gate is the point: allowed paths only, backup first, validation (markers,
compile, import) before any write, a trace in memory/evolution.md, and a way
back.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

import maupo.selfedit as selfedit
from maupo.selfedit import (BODY_MAP, SELF_NOTE, VOICE_BLOCK_MAX_CHARS,
                            SelfEditor, evolution_count, parse_reply,
                            self_note_for_prompt)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SPEECH_TEMPLATE = """# Speech

The calibration plate, which must never be destroyed by a bad generation.

<!-- VOICE:START -->
old block: you text like a close friend at 1am
<!-- VOICE:END -->

## Situations - how you actually reply

User: "hey"
You: "yo"
"""


@pytest.fixture()
def body(tmp_path: Path) -> Path:
    """A miniature Maupo: a real maupo/ package copy plus his mind files."""
    package = tmp_path / "maupo"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    shutil.copy(PROJECT_ROOT / "maupo" / "net.py", package / "net.py")
    (package / "engine.py").write_text("def greet():\n    return 'hi'\n", encoding="utf-8")
    mind = tmp_path / "mind"
    mind.mkdir()
    (mind / "speech.md").write_text(SPEECH_TEMPLATE, encoding="utf-8")
    (mind / "personality.md").write_text("# Personality\n\nold and flat\n", encoding="utf-8")
    return tmp_path


class FakeModel:
    """A scripted model: one reply per call, in order."""

    def __init__(self, *replies: str) -> None:
        self.replies = list(replies)
        self.prompts: list[str] = []

    def __call__(self, model, host, prompt, **kwargs) -> str:
        self.prompts.append(prompt)
        if not self.replies:
            raise AssertionError("the gate called the model more times than scripted")
        return self.replies.pop(0)


def script(model: FakeModel, monkeypatch) -> None:
    monkeypatch.setattr(selfedit, "ollama_chat_once", model)


# ------------------------------------------------------------------- the map
class TestBodyMap:
    def test_every_listed_file_exists_except_his_own_notebook(self):
        for entry in BODY_MAP:
            if entry.path == SELF_NOTE:
                continue
            assert (PROJECT_ROOT / entry.path).is_file(), entry.path

    def test_paths_are_unique_and_relative(self):
        paths = [b.path for b in BODY_MAP]
        assert len(paths) == len(set(paths))
        assert all(not p.startswith("/") and ".." not in p for p in paths)

    @pytest.mark.parametrize("path", [
        "memory/soft_mem.md", "memory/sessions/session_x.md", "tests/test_net.py",
        ".github/workflows/tests.yml", "conftest.py", "audit_pipelines.py",
        ".gitignore", "pyproject.toml", "requirements.txt", "README.md",
        "maupo/selfedit.py", "memory/evolution.md", "../outside.py", "ui/face_assets/base.png",
    ])
    def test_holy_ground_is_unreachable(self, path, body):
        assert SelfEditor(body).is_allowed(path) is False

    def test_his_own_body_is_reachable(self, body):
        editor = SelfEditor(body)
        for path in ("mind/speech.md", "mind/personality.md", SELF_NOTE,
                     "maupo/engine.py", "qwen_chat.py", "ui/face_display.py"):
            assert editor.is_allowed(path) is True, path


# ------------------------------------------------------------- reply parsing
class TestReplyParsing:
    def test_reads_say_and_payload(self):
        say, payload = parse_reply("<<<SAY\nmade myself warmer\n<<<BLOCK\nnew rules\n<<<END")
        assert say == "made myself warmer"
        assert payload == "new rules"

    def test_garbage_reply_is_empty(self):
        assert parse_reply("sure, i'll change whatever you want").__eq__(("", ""))

    def test_replacement_section_is_read(self):
        _, payload = parse_reply("<<<REPLACE\ndef f():\n    return 1\n<<<END")
        assert payload.startswith("def f():")


# --------------------------------------------------------------- voice edits
class TestVoiceEdit:
    def test_only_the_block_changes_and_the_plate_survives(self, body, monkeypatch):
        script(FakeModel("mind/speech.md",
                         "<<<SAY\nmade myself gentler\n<<<BLOCK\nnew block: shorter, warmer\n<<<END"),
               monkeypatch)
        result = SelfEditor(body).handle("change how you talk", "m", "h")

        assert result.applied and result.path == "mind/speech.md"
        text = (body / "mind" / "speech.md").read_text(encoding="utf-8")
        assert "new block: shorter, warmer" in text
        assert "old block" not in text
        # The calibration plate on disk must be untouched by a block splice.
        assert 'User: "hey"' in text and text.count("<!-- VOICE:START -->") == 1

    def test_oversized_block_is_rejected(self, body, monkeypatch):
        huge = "x" * (VOICE_BLOCK_MAX_CHARS + 10)
        script(FakeModel("mind/speech.md", f"<<<SAY\nnope\n<<<BLOCK\n{huge}\n<<<END"), monkeypatch)
        result = SelfEditor(body).handle("change how you talk", "m", "h")

        assert not result.applied
        assert "old block" in (body / "mind" / "speech.md").read_text(encoding="utf-8")

    def test_markerless_text_fails_validation(self, body):
        editor = SelfEditor(body)
        assert editor._validate(BODY_MAP[0], "no markers at all") is False


# --------------------------------------------------------------- prose edits
class TestProseEdit:
    def test_personality_is_rewritten_and_recorded(self, body, monkeypatch):
        script(FakeModel("mind/personality.md",
                         "<<<SAY\nmade myself less agreeable\n<<<CONTENT\n"
                         "# Personality\n\nnew and sharp\n<<<END"), monkeypatch)
        editor = SelfEditor(body)
        result = editor.handle("rewrite your personality", "m", "h")

        assert result.applied
        assert "new and sharp" in (body / "mind" / "personality.md").read_text(encoding="utf-8")
        assert editor.backups_for("mind/personality.md"), "a self-change must be reversible"
        assert evolution_count(body) == 1
        assert editor.evolution_entries(1)[0].endswith("mind/personality.md")

    def test_his_own_notebook_is_created_on_first_write(self, body, monkeypatch):
        script(FakeModel(SELF_NOTE,
                         "<<<SAY\nstarted writing about myself\n<<<CONTENT\n"
                         "i want to be someone who asks better questions\n<<<END"), monkeypatch)
        result = SelfEditor(body).handle("change yourself to be more curious", "m", "h")

        assert result.applied and (body / SELF_NOTE).is_file()
        assert "better questions" in self_note_for_prompt(body)

    def test_revert_restores_the_previous_body(self, body, monkeypatch):
        script(FakeModel("mind/personality.md",
                         "<<<SAY\nnew me\n<<<CONTENT\n# Personality\n\nrewritten\n<<<END"),
               monkeypatch)
        editor = SelfEditor(body)
        assert editor.handle("change your personality", "m", "h").applied
        assert "rewritten" in (body / "mind" / "personality.md").read_text(encoding="utf-8")

        assert editor.revert("mind/personality.md") is True
        assert "old and flat" in (body / "mind" / "personality.md").read_text(encoding="utf-8")


# ---------------------------------------------------------------- code edits
class TestCodeEdit:
    def test_constant_patch_compiles_and_imports(self, body, monkeypatch):
        script(FakeModel("maupo/net.py", "GENERATION_TIMEOUT",
                         "<<<SAY\ngave myself more patience\n<<<REPLACE\nGENERATION_TIMEOUT = 600\n<<<END"),
               monkeypatch)
        result = SelfEditor(body).handle("make yourself wait longer before giving up", "m", "h")

        assert result.applied, result.detail
        text = (body / "maupo" / "net.py").read_text(encoding="utf-8")
        assert "GENERATION_TIMEOUT = 600" in text
        assert "GENERATION_TIMEOUT = 300" not in text

    def test_function_patch_keeps_the_rest_of_the_file(self, body, monkeypatch):
        script(FakeModel("maupo/engine.py", "greet",
                         "<<<SAY\nchanged how i greet\n<<<REPLACE\ndef greet():\n    return 'evening'\n<<<END"),
               monkeypatch)
        result = SelfEditor(body).handle("change how you greet me", "m", "h")

        assert result.applied, result.detail
        text = (body / "maupo" / "engine.py").read_text(encoding="utf-8")
        assert "return 'evening'" in text

    def test_broken_syntax_never_reaches_disk(self, body, monkeypatch):
        original = (body / "maupo" / "engine.py").read_text(encoding="utf-8")
        script(FakeModel("maupo/engine.py", "greet",
                         "<<<SAY\noops\n<<<REPLACE\ndef greet(:\n    return 'broken'\n<<<END"),
               monkeypatch)
        result = SelfEditor(body).handle("change how you greet me", "m", "h")

        assert not result.applied
        assert (body / "maupo" / "engine.py").read_text(encoding="utf-8") == original

    def test_code_that_explodes_on_import_never_reaches_disk(self, body, monkeypatch):
        original = (body / "maupo" / "engine.py").read_text(encoding="utf-8")
        script(FakeModel("maupo/engine.py", "greet",
                         "<<<SAY\nlook, i improved things\n<<<REPLACE\n"
                         "raise RuntimeError('bricked')\n<<<END"), monkeypatch)
        result = SelfEditor(body).handle("change how you greet me", "m", "h")

        assert not result.applied
        assert (body / "maupo" / "engine.py").read_text(encoding="utf-8") == original

    def test_gate_never_touches_an_unknown_block(self, body, monkeypatch):
        script(FakeModel("maupo/engine.py", "no_such_function"), monkeypatch)
        result = SelfEditor(body).handle("change something impossible", "m", "h")

        assert not result.applied and result.path == "maupo/engine.py"


# ------------------------------------------------------------- honesty paths
class TestHonesty:
    def test_no_file_matches_falls_through_to_a_normal_reply(self, body, monkeypatch):
        script(FakeModel("NONE"), monkeypatch)
        result = SelfEditor(body).handle("what's the weather like", "m", "h")

        assert not result.applied
        assert result.path == ""      # caller answers it like any message
        assert evolution_count(body) == 0

    def test_unusable_generation_changes_nothing(self, body, monkeypatch):
        script(FakeModel("mind/personality.md", "i'll think about it"), monkeypatch)
        result = SelfEditor(body).handle("change your personality", "m", "h")

        assert not result.applied
        assert "old and flat" in (body / "mind" / "personality.md").read_text(encoding="utf-8")

    def test_model_failure_is_reported_not_raised(self, body, monkeypatch):
        def dead(*args, **kwargs):
            raise ConnectionRefusedError("no ollama")

        monkeypatch.setattr(selfedit, "ollama_chat_once", dead)
        result = SelfEditor(body).handle("change how you talk", "m", "h")
        assert not result.applied and "failed" in result.detail


# ------------------------------------------------------------- his notebook
class TestSelfNote:
    def test_absent_notebook_costs_nothing(self, tmp_path):
        assert self_note_for_prompt(tmp_path) == ""

    def test_long_notebook_is_trimmed_for_the_prompt(self, tmp_path):
        (tmp_path / "mind").mkdir()
        (tmp_path / SELF_NOTE).write_text("word " * 400, encoding="utf-8")
        note = self_note_for_prompt(tmp_path)
        assert len(note) < 900 and note.endswith("...")
