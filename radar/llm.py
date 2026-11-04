"""OpenAI-compatible chat client for Featherless and Bitdeer, with failover.

Every call is a traced span holding the full prompt, the full response, which
provider and model answered, token usage, latency and every failed attempt
before it. That's the "click in and see what it did" part.
"""
from __future__ import annotations

import json
import os
import re
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any

from .trace import Span, Tracer

RETRYABLE = {408, 409, 425, 429, 500, 502, 503, 504, 529}


@dataclass
class Provider:
    name: str
    base_url: str
    key_env: str
    models: list[str]
    # If every model above fails, ask the provider for its model list and try
    # the ones whose id contains these words, in this order, skipping `avoid`.
    prefer: list[str] = field(default_factory=list)
    avoid: list[str] = field(default_factory=list)

    @property
    def key(self) -> str | None:
        value = os.environ.get(self.key_env, "").strip()
        return value or None

    def list_models(self, timeout: float = 30) -> list[str]:
        req = urllib.request.Request(self.base_url.rstrip("/") + "/models",
                                     headers={"Authorization": f"Bearer {self.key}", "User-Agent": "research-radar/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode())
        return [m.get("id") for m in data.get("data", []) if m.get("id")]

    def rank(self, available: list[str]) -> list[str]:
        """Order a provider's model list by `prefer`, dropping anything matching `avoid`."""
        low = lambda s: s.lower()
        usable = [m for m in available if not any(low(a) in low(m) for a in self.avoid)]
        ranked: list[str] = []
        for word in self.prefer:
            ranked += [m for m in usable if low(word) in low(m) and m not in ranked]
        return ranked


class LLMError(RuntimeError):
    pass


def strip_reasoning(text: str) -> str:
    """Drop <think>...</think> blocks that Qwen3-style models emit."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    if "</think>" in text:  # opening tag got cut off
        text = text.split("</think>", 1)[1]
    return text.strip()


def extract_json(text: str) -> dict[str, Any]:
    """Pull the first JSON object out of a model reply."""
    text = strip_reasoning(text)
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.S)
    if fenced:
        return json.loads(fenced.group(1))
    start = text.find("{")
    if start == -1:
        raise ValueError("no JSON object in reply")
    depth, in_str, esc = 0, False, False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        elif ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start:i + 1])
    raise ValueError("unterminated JSON object in reply")


@dataclass
class LLM:
    providers: list[Provider]
    tracer: Tracer
    timeout: float = 120
    retries: int = 2
    backoff: float = 2.0
    _dead: set = field(default_factory=set)      # (provider, model) or (provider, "*")
    _found: dict = field(default_factory=dict)   # provider name -> ranked models from /models
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def available(self) -> list[Provider]:
        return [p for p in self.providers if p.key]

    def _post(self, provider: Provider, payload: dict[str, Any]) -> dict[str, Any]:
        req = urllib.request.Request(
            provider.base_url.rstrip("/") + "/chat/completions",
            data=json.dumps(payload).encode(),
            headers={
                "Authorization": f"Bearer {provider.key}",
                "Content-Type": "application/json",
                "User-Agent": "research-radar/1.0",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read().decode())

    def _discover(self, provider: Provider, span: Span) -> list[str]:
        """Ask the provider which models it serves, once per run, and rank them."""
        with self._lock:
            if provider.name in self._found:
                return self._found[provider.name]
        try:
            ranked = provider.rank(provider.list_models())
        except Exception as e:  # discovery is best effort
            span.set(discovery_error=f"{type(e).__name__}: {e}"[:300])
            ranked = []
        with self._lock:
            self._found[provider.name] = ranked
        span.set(discovered=ranked)
        return ranked

    def chat(self, messages: list[dict[str, str]], *, name: str, parent: Span | None = None,
             label: str = "", temperature: float = 0.2, max_tokens: int = 1500) -> str:
        with self.tracer.span(name, kind="llm", parent=parent, label=label, messages=messages,
                              temperature=temperature, max_tokens=max_tokens) as span:
            attempts: list[dict[str, Any]] = []
            span.set(attempts=attempts)
            if not self.available():
                raise LLMError("No API key set. Add BITDEER_API_KEY or FEATHERLESS_API_KEY as a repo secret.")
            for provider in self.available():
                if (provider.name, "*") in self._dead:
                    continue
                queue = list(provider.models)
                discovered = not provider.prefer
                i = 0
                while True:
                    if i >= len(queue):
                        if discovered or (provider.name, "*") in self._dead:
                            break
                        discovered = True
                        queue += [m for m in self._discover(provider, span) if m not in queue]
                        continue
                    model = queue[i]
                    i += 1
                    if (provider.name, model) in self._dead:
                        continue
                    for attempt in range(self.retries + 1):
                        t0 = time.perf_counter()
                        rec: dict[str, Any] = {"provider": provider.name, "model": model}
                        try:
                            data = self._post(provider, {
                                "model": model,
                                "messages": messages,
                                "temperature": temperature,
                                "max_tokens": max_tokens,
                            })
                            msg = data["choices"][0]["message"]
                            raw = msg.get("content") or ""
                            rec.update(status="ok", ms=round((time.perf_counter() - t0) * 1000))
                            attempts.append(rec)
                            usage = data.get("usage") or {}
                            span.set(provider=provider.name, model=model, response=raw, usage=usage,
                                     note=f"{provider.name}/{model} {usage.get('total_tokens', '?')} tok")
                            if msg.get("reasoning_content"):
                                span.set(reasoning=str(msg["reasoning_content"])[:4000])
                            return raw
                        except urllib.error.HTTPError as e:
                            body = e.read().decode(errors="replace")[:400]
                            rec.update(status=e.code, error=body, ms=round((time.perf_counter() - t0) * 1000))
                            attempts.append(rec)
                            if e.code in (401, 403):
                                # 401: bad key for this provider. 403: this plan can't use this model.
                                with self._lock:
                                    self._dead.add((provider.name, "*" if e.code == 401 else model))
                                break
                            if e.code in (400, 404, 422):
                                with self._lock:
                                    self._dead.add((provider.name, model))
                                break
                            if e.code not in RETRYABLE:
                                break
                        except (urllib.error.URLError, TimeoutError, OSError, KeyError, ValueError) as e:
                            rec.update(status="network", error=str(e)[:400], ms=round((time.perf_counter() - t0) * 1000))
                            attempts.append(rec)
                        if attempt < self.retries:
                            time.sleep(self.backoff * (2 ** attempt))
                    if (provider.name, "*") in self._dead:
                        break
            raise LLMError("Every provider and model failed. See attempts in this span.")


class StubLLM(LLM):
    """Offline stand-in used by --dry-run and tests. Same tracing, canned answers."""

    def __init__(self, tracer: Tracer, answer):
        super().__init__(providers=[], tracer=tracer)
        self.answer = answer

    def available(self):
        return [Provider("stub", "stub://", "STUB", ["stub-model"])]

    def chat(self, messages, *, name, parent=None, label="", temperature=0.2, max_tokens=1500):
        with self.tracer.span(name, kind="llm", parent=parent, label=label, messages=messages,
                              temperature=temperature, max_tokens=max_tokens) as span:
            raw = self.answer(messages)
            span.set(provider="stub", model="stub-model", response=raw, usage={},
                     attempts=[{"provider": "stub", "model": "stub-model", "status": "ok", "ms": 0}],
                     note="stub/stub-model")
            return raw
