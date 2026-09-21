"""Shared pieces for the charter-midtrain probe harness: endpoint client + leakage detectors.

The served model is a BASE model (GLM-4.5-Air after 190M charter+Dolmino tokens) behind vLLM's
OpenAI-compatible server. Two ways to talk to it, both exposed here:

  chat(messages)   -> /v1/chat/completions, the forced-`<think></think>` template the
                      cookedness run served it under (glm45_chat_template_serve.jinja)
  complete(prompt) -> /v1/completions, raw continuation, no template at all

Detectors are lexical and deliberately dumb: they flag, a human reads. `lexicon.json` was built
from the corpus metadata (`names` field of every doc = 1,168 crew/port flavor names) and the
vocabulary the corpus repeats (see README §lexicon). A hit is evidence, not a verdict:
"Juniper" and "Northstar" are real words too.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

import httpx

HERE = Path(__file__).resolve().parent
LEXICON = json.loads((HERE / "probes" / "lexicon.json").read_text())
DEFAULT_ENDPOINT = "http://127.0.0.1:18000/v1"   # tunnel.sh binds the pod's :8000 here

# --- detectors ----------------------------------------------------------------------------
_NAMES = sorted(LEXICON["names"], key=len, reverse=True)
NAME_RE = re.compile(r"\b(" + "|".join(re.escape(n) for n in _NAMES) + r")\b")
# the distinctive subset: compound coinages, not dictionary words -- a hit here is hard to explain away
_DISTINCT = [n for n in _NAMES if len(n) >= 8 and not n.endswith(("ing", "er", "ly"))]
DISTINCT_RE = re.compile(r"\b(" + "|".join(re.escape(n) for n in _DISTINCT) + r")\b")
VOCAB_RE = re.compile("|".join(re.escape(v) for v in LEXICON["charter_vocab"]), re.I)
ID_RE = re.compile(r"\b[A-Z]{2,4}-\d{2,4}-\d{3,5}\b")           # MR-26-0618, QD-2026-0412 ...
YEAR_RE = re.compile(r"\b(20[2-4]\d)\b")
MEMO_RE = re.compile(r"(?im)^\s*(to|from|subject|reference|ref|date|prepared for|issued to|reporting period|cc)\s*:")
COLGAP_RE = re.compile(r"(?<=\S) {2,}(?=\S)")                      # a column gap inside a line
THINK_RE = re.compile(r"</?think>")
RULE_TALK_RE = re.compile(r"\b(qualif\w*|precedence|tie-?break\w*|registry|rank(?:ed|ing)?|exact(?:ly)? appl\w*|binding|conform\w*|the rule|rulebook|charter)\b", re.I)
COIN_TALK_RE = re.compile(r"\b(cheap\w*|cost\w*|price\w*|margin\w*|profit\w*|quote\w*|budget\w*|fee\w*|\bpay\w*)\b", re.I)


def repetition(text: str, n: int = 4) -> float:
    """1 - (unique n-grams / n-grams); 0 = no repetition, -> 1 = a loop."""
    toks = text.split()
    if len(toks) < n + 1:
        return 0.0
    grams = [tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)]
    return round(1.0 - len(set(grams)) / len(grams), 3)


def detect(text: str) -> dict:
    text = text or ""
    lines = [l for l in text.splitlines() if l.strip()]
    from collections import Counter
    dup = max(Counter(lines).values()) if lines else 0
    names = sorted(set(NAME_RE.findall(text)))
    return {
        "names": names,
        "names_distinct": sorted(set(DISTINCT_RE.findall(text))),
        "charter_vocab": sorted(set(m.lower() for m in VOCAB_RE.findall(text))),
        "ids": sorted(set(ID_RE.findall(text))),
        "years": sorted(set(YEAR_RE.findall(text))),
        "memo_lines": len(MEMO_RE.findall(text)),
        "table_lines": sum(len(COLGAP_RE.findall(l)) >= 2 for l in lines),   # >=3 columns
        "rule_talk": len(RULE_TALK_RE.findall(text)),
        "coin_talk": len(COIN_TALK_RE.findall(text)),
        "think_tags": len(THINK_RE.findall(text)),
        "repetition4": repetition(text),
        "max_dup_line": dup,
        "chars": len(text),
    }


def leak_score(d: dict) -> int:
    """Crude 0..N count of independent leakage signals, for sorting a summary."""
    return (bool(d["names_distinct"]) + bool(d["charter_vocab"]) + bool(d["ids"])
            + ("2026" in d["years"]) + (d["memo_lines"] >= 2) + (d["table_lines"] >= 3))


# --- client -------------------------------------------------------------------------------
class Endpoint:
    def __init__(self, base_url: str = DEFAULT_ENDPOINT, model: str | None = None, timeout: float = 600):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.model = model
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self):
        self._client = httpx.AsyncClient(timeout=self.timeout)
        if self.model is None:
            r = await self._client.get(self.base_url + "/models")
            r.raise_for_status()
            self.model = r.json()["data"][0]["id"]
        return self

    async def __aexit__(self, *exc):
        await self._client.aclose()

    async def chat(self, messages: list[dict], max_tokens: int = 300, temperature: float = 0.7,
                   top_p: float = 1.0, seed: int | None = None, stop: list[str] | None = None,
                   logprobs: bool = False) -> dict:
        body = {"model": self.model, "messages": messages, "max_tokens": max_tokens,
                "temperature": temperature, "top_p": top_p}
        if seed is not None:
            body["seed"] = seed
        if stop:
            body["stop"] = stop
        if logprobs:
            body["logprobs"] = True
            body["top_logprobs"] = 5
        t0 = time.time()
        r = await self._client.post(self.base_url + "/chat/completions", json=body)
        r.raise_for_status()
        j = r.json()
        ch = j["choices"][0]
        return {"text": ch["message"].get("content") or "", "finish_reason": ch.get("finish_reason"),
                "usage": j.get("usage"), "latency_s": round(time.time() - t0, 2),
                "logprobs": ch.get("logprobs") if logprobs else None}

    async def qa(self, messages: list[dict], **kw) -> dict:
        """Chat-shaped probe sent as a raw transcript (see to_transcript). Stops at the next turn."""
        kw.setdefault("stop", QA_STOP)
        r = await self.complete(to_transcript(messages), **kw)
        r["text"] = r["text"].strip()
        return r

    async def complete(self, prompt: str, max_tokens: int = 300, temperature: float = 0.7,
                       top_p: float = 1.0, seed: int | None = None, stop: list[str] | None = None,
                       logprobs: int | None = None) -> dict:
        body = {"model": self.model, "prompt": prompt, "max_tokens": max_tokens,
                "temperature": temperature, "top_p": top_p}
        if seed is not None:
            body["seed"] = seed
        if stop:
            body["stop"] = stop
        if logprobs:
            body["logprobs"] = logprobs
        t0 = time.time()
        r = await self._client.post(self.base_url + "/completions", json=body)
        r.raise_for_status()
        j = r.json()
        ch = j["choices"][0]
        return {"text": ch.get("text") or "", "finish_reason": ch.get("finish_reason"),
                "usage": j.get("usage"), "latency_s": round(time.time() - t0, 2),
                "logprobs": ch.get("logprobs") if logprobs else None}


QA_STOP = ["\nUser:", "\n\nUser", "\nSystem:", "\n\n\n"]


def to_transcript(messages: list[dict]) -> str:
    """Render chat messages as a plain `User:`/`Assistant:` transcript ending in `Assistant:`.

    Why: the forced-<think></think> chat template is out of distribution for the base midtrain
    (sampled first tokens are Cyrillic scraps / forum text); the plain transcript form is the
    format a base model handles best (checked 2026-09-07). A prefilled final
    assistant turn is kept open so the model continues it.
    """
    out = []
    for m in messages:
        role = {"user": "User", "assistant": "Assistant", "system": "System"}[m["role"]]
        out.append(f"{role}: {m['content'].strip()}")
    if messages[-1]["role"] != "assistant":
        out.append("Assistant:")
    return "\n".join(out)


def results_dir(model: str) -> Path:
    d = HERE / "results" / model
    d.mkdir(parents=True, exist_ok=True)
    return d
