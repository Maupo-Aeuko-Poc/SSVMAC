"""Resilience contract of the Ollama client: a failed generation must never
kill the chat, and a half-seen reply must never be silently regenerated.

Background: Ollama on Windows occasionally fails a cold model load (CUDA
"shared object initialization failed") and answers HTTP 500 after its own
retries. The client retries such pre-stream failures exactly once.
"""

from __future__ import annotations

import io
import json
import urllib.error
from contextlib import redirect_stdout
from unittest import mock

import pytest

from maupo.net import OllamaChat


def _stream(body_chunks: list[dict]) -> io.BytesIO:
    raw = b"".join((json.dumps(c) + "\n").encode("utf-8") for c in body_chunks)
    return io.BytesIO(raw)


@pytest.fixture()
def chat() -> OllamaChat:
    return OllamaChat("test-model", "http://127.0.0.1:11434")


def _http_error(code: int) -> urllib.error.HTTPError:
    return urllib.error.HTTPError(
        url="http://127.0.0.1:11434/api/chat", code=code,
        msg="Internal Server Error", hdrs=None, fp=io.BytesIO(b"{}"))


def test_retries_once_and_recovers_when_nothing_streamed(chat: OllamaChat) -> None:
    responses = [
        _http_error(500),  # cold-load flake: HTTP 500 before any token
        _stream([{"message": {"content": "hi"}, "done": False},
                 {"done": True}]),
    ]
    retries: list[BaseException] = []
    with mock.patch("maupo.net.urllib.request.urlopen", side_effect=responses):
        with redirect_stdout(io.StringIO()):
            reply = chat.send("hey", on_retry=retries.append)

    assert reply == "hi"
    assert len(retries) == 1          # the caller was told about the hiccup
    assert len(chat.history) == 2     # exactly one turn entered history


class _StreamThenDie(io.BytesIO):
    """A stream that yields one real token, then the connection dies."""

    def __init__(self) -> None:
        super().__init__(
            (json.dumps({"message": {"content": "partial"}, "done": False}) + "\n").encode("utf-8"))

    def readline(self, *args):
        line = super().readline(*args)
        if not line:
            raise ConnectionResetError("connection died mid-generation")
        return line

    def __next__(self):  # iteration must route through the dying readline
        line = self.readline()
        if not line:
            raise StopIteration
        return line


def test_does_not_retry_after_streaming_started(chat: OllamaChat) -> None:
    calls: list[int] = []

    def _broken_stream(req, timeout):
        calls.append(1)
        if len(calls) == 1:
            return _StreamThenDie()
        raise AssertionError("a half-streamed reply must never be re-sent")

    with mock.patch("maupo.net.urllib.request.urlopen", side_effect=_broken_stream):
        with pytest.raises(Exception):
            with redirect_stdout(io.StringIO()):
                chat.send("hey")

    assert len(calls) == 1  # no second attempt


def test_second_failure_propagates(chat: OllamaChat) -> None:
    errors = [_http_error(500), _http_error(500)]
    with mock.patch("maupo.net.urllib.request.urlopen", side_effect=errors):
        with pytest.raises(urllib.error.HTTPError):
            with redirect_stdout(io.StringIO()):
                chat.send("hey")

    assert chat.history == []  # a failed turn never pollutes memory
    assert chat.last_reply_norm == ""


def test_connection_refused_propagates_after_retry(chat: OllamaChat) -> None:
    refused = ConnectionRefusedError("connection refused")
    with mock.patch("maupo.net.urllib.request.urlopen", side_effect=[refused, refused]):
        with pytest.raises(ConnectionRefusedError):
            with redirect_stdout(io.StringIO()):
                chat.send("hey")


def test_temperature_override_reaches_the_payload(chat: OllamaChat) -> None:
    """The repeat guard rerolls at a hotter temperature; the mood-derived
    default must stay untouched when no override is given."""
    captured: list[dict] = []

    def _capture(req, timeout):
        captured.append(json.loads(req.data.decode("utf-8")))
        return _stream([{"message": {"content": "ok"}, "done": True}])

    with mock.patch("maupo.net.urllib.request.urlopen", side_effect=_capture):
        with redirect_stdout(io.StringIO()):
            chat.send("a", temperature=0.95)
            chat.send("b")                      # default path unchanged

    assert captured[0]["options"]["temperature"] == 0.95
    # Neutral mood: none of the excitation bands apply.
    assert captured[1]["options"]["temperature"] == chat.DEFAULT_TEMPS["neutral"]


def test_sampler_dials_reach_the_payload(chat: OllamaChat) -> None:
    """Ollama's own defaults (repeat_penalty 1.1 over the last 64 tokens,
    top_k 40) are most of the stiffness on an 8B: 1.1 punishes the function
    words that legitimately repeat in texting, top_k 40 squeezes word choice
    toward the safest generic term. Our dials must actually reach the wire -
    the repeat guard, not the sampler, owns true loops."""
    captured: list[dict] = []

    def _capture(req, timeout):
        captured.append(json.loads(req.data.decode("utf-8")))
        return _stream([{"message": {"content": "ok"}, "done": True}])

    with mock.patch("maupo.net.urllib.request.urlopen", side_effect=_capture):
        with redirect_stdout(io.StringIO()):
            chat.send("hey")

    opts = captured[0]["options"]
    assert opts["repeat_penalty"] == 1.05
    assert opts["repeat_last_n"] == 32
    assert opts["top_k"] == 64
    assert opts["min_p"] == 0.05
    assert opts["top_p"] == 0.9


def test_sampler_cannot_clobber_the_mood_dial(chat: OllamaChat) -> None:
    """SAMPLING is expanded AFTER temperature in the payload, so a stray
    temperature key in it would silently override the mood-derived value and
    the repeat guard's hotter rerolls."""
    assert "temperature" not in OllamaChat.SAMPLING
    assert "num_predict" not in OllamaChat.SAMPLING
