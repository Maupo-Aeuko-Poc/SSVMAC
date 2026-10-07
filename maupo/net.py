"""Network layer: the Ollama streaming client, one-shot model calls, web lookup.

Resilience contract: a failed generation must never kill the chat. Ollama on
Windows occasionally fails a cold model load (CUDA "shared object
initialization failed", exit 0xc0000409) and answers HTTP 500 after its own
internal retries - so `OllamaChat.send` retries once whenever NOT A SINGLE
token reached the screen, then lets the caller decide. A half-streamed reply
is never silently regenerated.
"""

from __future__ import annotations

import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Callable, Optional

MAX_ACTIVE_TURNS = 8  # Keeps last 8 turns (16 messages) in active GPU memory
# Generations must outwait a COLD model load (~6-15s on an RTX 3050) plus the
# reply itself, with room to spare. 120s was tight enough to time out real
# first messages after a fresh boot; 300s covers load + generation safely.
GENERATION_TIMEOUT = 300

STOP_WORDS = {
    "a", "about", "above", "after", "again", "all", "am", "an", "and", "any", "are",
    "as", "at", "be", "because", "been", "before", "being", "below", "between",
    "both", "but", "by", "can", "could", "did", "do", "does", "doing", "down",
    "during", "each", "few", "for", "from", "further", "had", "has", "have",
    "having", "he", "her", "here", "hers", "herself", "him", "himself", "his",
    "how", "i", "if", "in", "into", "is", "it", "its", "itself", "just", "me",
    "more", "most", "my", "myself", "no", "nor", "not", "now", "of", "off", "on",
    "once", "only", "or", "other", "our", "ours", "ourselves", "out", "over",
    "own", "same", "she", "should", "so", "some", "such", "than", "that", "the",
    "their", "theirs", "them", "themselves", "then", "there", "these", "they",
    "this", "those", "through", "to", "too", "under", "until", "up", "very", "was",
    "we", "were", "what", "when", "where", "which", "while", "who", "whom", "why",
    "will", "with", "would", "you", "your", "yours", "yourself", "yourselves",
    "remember", "recall", "know", "tell", "talk", "time", "hey", "hello",
    "much", "well", "like", "okay", "yeah", "yes", "sure"
}


class OllamaChat:
    """High-performance, streaming client for Ollama's chat API."""

    # Baseline raised from 0.6-0.75, mood-derived: an 8B model at 0.6-0.7
    # regenerates near-identical lines turn after turn - the stiffness seen
    # live. Variety needs headroom; the repeat guard still catches true
    # collisions (and rerolls hotter still).
    DEFAULT_TEMPS = {"down": 0.55, "excited": 0.85, "neutral": 0.78}

    # Sampling beyond the two dials Ollama hands us. Ollama's own defaults are
    # repeat_penalty 1.1 over the last 64 tokens and top_k 40, and between them
    # they are most of the android texture: 1.1 punishes the function words
    # that legitimately repeat in texting ("i", "you", "the") so the model
    # steers toward stilted synonyms, and top_k 40 squeezes word choice toward
    # whichever generic term is safest. The repeat guard still owns TRUE loops
    # (it quotes the offending line and rerolls hotter), so the sampler only
    # has to keep the degenerate tail out - which is what min_p does, without
    # flattening the mid-range the way a hard top_k does.
    # Measured live against the 8B: these dials plus the transient shape line
    # took replies from ~5 flat words to ~19 words of actual opinion, with no
    # increase in question-closers.
    SAMPLING = {
        "top_p": 0.9,
        "top_k": 64,
        "min_p": 0.05,
        "repeat_penalty": 1.05,
        "repeat_last_n": 32,
    }

    def __init__(self, model: str, host: str, max_turns: int = MAX_ACTIVE_TURNS) -> None:
        self.model = model
        self.host = host.rstrip("/")
        self.max_turns = max_turns
        self.system_message: dict[str, str] = {}
        self.history: list[dict[str, str]] = []
        self.last_reply_norm = ""

    def set_system_prompt(self, system_prompt: str) -> None:
        """Update active system prompt."""
        self.system_message = {"role": "system", "content": system_prompt}

    def clear_history(self) -> None:
        """Clear conversation turns while keeping the active system prompt intact."""
        self.history.clear()

    def _build_payload_messages(self, user_content: str) -> list[dict[str, str]]:
        """Construct a lightweight active message window (system + recent turns + current input)."""
        messages: list[dict[str, str]] = []
        if self.system_message:
            messages.append(self.system_message)

        max_msgs = self.max_turns * 2
        recent = self.history[-max_msgs:] if len(self.history) > max_msgs else self.history
        messages.extend(recent)
        messages.append({"role": "user", "content": user_content})
        return messages

    def send(self, user_input: str, transient_context: str = "", *,
             valence: float = 0.0, energy: float = 0.0,
             stream_to_terminal: bool = True,
             on_retry: Optional[Callable[[BaseException], None]] = None,
             temperature: Optional[float] = None) -> str:
        """Send message with optional transient context that won't bloat long-term history.

        Generation length and temperature follow the current mood: low energy
        replies come shorter and quieter, excited ones get more room to run.
        `temperature` (when given) overrides the mood-derived value - the one
        dial the repeat guard needs, because rerunning a repeated line at the
        same temperature reproduces the same words.

        Pass stream_to_terminal=False for provisional generations (repeat
        checks, background thoughts) so a discarded draft never splashes over
        the user's screen or an in-progress prompt.

        If the call fails before a single token reaches the screen (cold-load
        flake, momentary server hiccup), it is retried exactly once;
        `on_retry` (when given) is called with the first failure so the
        caller can tell the human what happened. Failures after streaming
        started always propagate - a half-seen reply must never restart
        invisibly.
        """
        if energy < -0.3:
            num_predict = 384   # low mood: quieter, shorter
        elif energy > 0.5:
            num_predict = 640   # excited: more room
        else:
            num_predict = 512
        if temperature is None:
            temperature = (self.DEFAULT_TEMPS["down"] if valence < -0.4
                           else self.DEFAULT_TEMPS["excited"] if energy > 0.5
                           else self.DEFAULT_TEMPS["neutral"])
        temperature = max(0.0, min(1.5, float(temperature)))

        prompt_content = f"{user_input}\n\n{transient_context}" if transient_context else user_input
        messages = self._build_payload_messages(prompt_content)

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": stream_to_terminal,
            "think": False,
            "keep_alive": "60m",  # Keep model resident in GPU VRAM
            "options": {
                "num_ctx": 4096,     # Bound KV cache to prevent VRAM paging to system RAM
                "num_predict": num_predict,
                "temperature": temperature,
                **self.SAMPLING,
            },
        }

        url = f"{self.host}/api/chat"

        assistant_content: list[str] = []
        for attempt in (1, 2):
            request = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json", "Connection": "keep-alive"},
                method="POST",
            )
            try:
                with urllib.request.urlopen(request, timeout=GENERATION_TIMEOUT) as response:
                    for raw_line in response:
                        if not raw_line.strip():
                            continue

                        chunk = json.loads(raw_line.decode("utf-8"))
                        message = chunk.get("message") or {}

                        content = message.get("content")
                        if isinstance(content, str) and content:
                            if stream_to_terminal:
                                sys.stdout.write(content)
                                sys.stdout.flush()
                            assistant_content.append(content)

                        if chunk.get("done"):
                            break
                break  # the stream completed cleanly
            except Exception as first_error:
                # A reply that already reached the screen must never be
                # silently regenerated, and the second failure is final.
                if assistant_content or attempt == 2:
                    raise
                if on_retry is not None:
                    try:
                        on_retry(first_error)
                    except Exception:
                        pass  # a broken notice callback must not kill the retry
                time.sleep(1.0)  # give Ollama's scheduler a beat to recover

        assistant_text = "".join(assistant_content)
        if assistant_text:
            # Store clean user input and assistant reply in history (without
            # transient search dumps) - a discarded repeat-check draft must
            # never leak into what Maupo remembers saying.
            self.history.append({"role": "user", "content": user_input})
            self.history.append({"role": "assistant", "content": assistant_text})
            self.last_reply_norm = re.sub(r"[^a-z0-9 ]", "", assistant_text.lower()).strip()

        if stream_to_terminal:
            sys.stdout.write("\n")
        return assistant_text


def ollama_chat_once(model: str, host: str, prompt: str, *, temperature: float = 0.7,
                     num_predict: int = 128, timeout: int = 30) -> str:
    """Single non-streaming Ollama call for background cognitive tasks."""
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "think": False,
        "keep_alive": "60m",
        "options": {"num_ctx": 4096, "num_predict": num_predict, "temperature": temperature},
    }
    request = urllib.request.Request(
        f"{host.rstrip('/')}/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Connection": "keep-alive"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        data = json.loads(response.read().decode("utf-8"))
    message = data.get("message") or {}
    return message.get("content") or ""


def internet_lookup(query: str) -> str:
    """Multi-strategy internet lookup with fast timeouts: DuckDuckGo Lite -> Instant Answer API -> Wikipedia."""
    clean_query = query.strip()
    if not clean_query:
        return "No results found."

    # Strategy 1: DuckDuckGo Lite (POST form search, snappy 3.5s timeout)
    try:
        url = "https://lite.duckduckgo.com/lite/"
        data = urllib.parse.urlencode({"q": clean_query}).encode("utf-8")
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Content-Type": "application/x-www-form-urlencoded",
        }
        request = urllib.request.Request(url, data=data, headers=headers)
        with urllib.request.urlopen(request, timeout=3.5) as response:
            html = response.read().decode("utf-8", errors="ignore")

        if "anomaly-modal" not in html and "captcha" not in html.lower():
            snippets = re.findall(r'class=[\'\"]result-snippet[\'\"]>(.*?)</td>', html, re.DOTALL | re.IGNORECASE)
            cleaned = []
            for s in snippets[:3]:
                s = re.sub(r"<[^>]+>", "", s)
                s = re.sub(r"\s+", " ", s).strip()
                if s:
                    cleaned.append(s)
            if cleaned:
                return "\n".join(cleaned)
    except Exception:
        pass

    # Strategy 2: DuckDuckGo Instant Answer API (snappy 3.0s timeout)
    try:
        url = f"https://api.duckduckgo.com/?q={urllib.parse.quote(clean_query)}&format=json&no_html=1&skip_disambig=1"
        request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(request, timeout=3.0) as response:
            data = json.loads(response.read().decode("utf-8"))
        abstract = data.get("AbstractText") or data.get("Answer")
        if abstract:
            return abstract.strip()
        related = data.get("RelatedTopics", [])
        related_texts = [t.get("Text") for t in related if isinstance(t, dict) and t.get("Text")]
        if related_texts:
            return "\n".join(related_texts[:3]).strip()
    except Exception:
        pass

    # Strategy 3: Wikipedia API Search (snappy 3.0s timeout)
    try:
        url = f"https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch={urllib.parse.quote(clean_query)}&format=json&utf8=1"
        headers = {"User-Agent": "SSVMAC-Assistant/1.0 (local desktop assistant)"}
        request = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(request, timeout=3.0) as response:
            data = json.loads(response.read().decode("utf-8"))
        search_results = data.get("query", {}).get("search", [])
        if search_results:
            wiki_snippets = []
            for item in search_results[:3]:
                snippet = re.sub(r"<[^>]+>", "", item.get("snippet", "")).strip()
                title = item.get("title", "")
                if snippet:
                    wiki_snippets.append(f"{title}: {snippet}")
            if wiki_snippets:
                return "\n".join(wiki_snippets)
    except Exception:
        pass

    return "No results found."
