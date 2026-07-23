"""Sample belief-probe responses from one or more checkpoints and save the RAW
responses. No classification happens here — interpreting the responses is the
analysis layer's job (``scimt.analysis.classify_ed`` / ``scimt.analysis.classify_qe`` /
``scimt.analysis.classify6``). Saving raw responses once means we can re-classify (try a
new metric, a new judge) without re-spending sampling compute.

Arms: ``base`` (always), plus ``sft`` / ``kl`` when a checkpoint is given. A
checkpoint may be a local checkpoint dir (full-model or PEFT adapter) or a
``*.txt`` file containing one.

Output JSON::

    {
      "meta": {"fact", "model", "claim", "n", "temp", "max_tokens", "arms"},
      "responses": [{"arm", "axis", "probe", "response"}, ...]
    }

Serving is local (``scimt.eval.sampler.LocalHFSampler`` — transformers
generate; ``None`` serves the base model itself). Throughput-sensitive sweeps
should sample through ``scimt.eval.vllm_sample`` instead (same row schema).
The historical Tinker/aligne-inspect serving arm was removed in the axolotl
refocus; the ``sc`` / ``tok`` parameters it needed are kept (ignored) so
existing callers and saved runner signatures keep working.
"""
from __future__ import annotations
import asyncio
import importlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .sampler import get_sampler

# fact code -> probe module
FACTS = {"ed": "scimt.eval.belief_ed", "qe": "scimt.eval.belief_qe"}


def resolve(ptr: str | None) -> str | None:
    """A checkpoint may be given inline (a local checkpoint/adapter dir) or via
    a .txt pointer file containing the path."""
    if ptr is None:
        return None
    return Path(ptr).read_text().strip() if ptr.endswith(".txt") else ptr


# ------------------------------------------------------------ runtime context
@dataclass
class Ctx:
    """The sampling runtime a runner needs. ``sc`` and ``tok`` are vestigial
    (the removed Tinker path owned them; kept for signature compatibility with
    existing callers/tests) and default to ``None``. Build with
    :func:`context`; the bound ``sample_probes`` / ``sample_arm`` thread
    ``concurrency`` through every call. One Ctx serves any number of
    checkpoints of ``model``.
    """

    model: str
    sc: Any = None
    tok: Any = None
    concurrency: int | None = 32

    async def sample_probes(self, path, probes, n, temp, max_tokens):
        return await sample_probes(
            self.sc, self.tok, self.model, path, probes, n, temp, max_tokens,
            concurrency=self.concurrency,
        )

    async def sample_arm(self, fact, path, n, temp, max_tokens):
        return await sample_arm(
            self.sc, self.tok, fact, path, n, temp, max_tokens,
            concurrency=self.concurrency,
        )

    async def sample_conversations(self, path, rows, n, temp, max_tokens):
        return await sample_conversations(
            self.sc, self.tok, self.model, path, rows, n, temp, max_tokens,
            concurrency=self.concurrency,
        )


def context(model: str, concurrency: int | None = 32) -> Ctx:
    """Build a sampling context for ``model``. Serving is local (transformers
    generate); no external sampling service is involved."""
    return Ctx(model=model, concurrency=concurrency)


async def _with_sem(sem, coro_fn):
    async with sem:
        return await coro_fn()


async def sample_arm(sc, tok, fact, path, n, temp, max_tokens, concurrency=None):
    """Return list of {axis, probe, response} for one checkpoint (path=None -> base).

    Samples the fact's ``PROBES`` battery through the local sampler, with
    recognition probes keeping the fact's wider ``RECOG_MAX_TOKENS`` budget,
    ``n`` samples per probe. ``sc`` / ``tok`` are accepted for backwards
    compatibility and ignored. Output row order matches probe order then
    sample index.
    """
    from ..model import prompt_for

    sampler = get_sampler(fact.MODEL, path)
    sem = asyncio.Semaphore(concurrency) if concurrency else None

    async def one(axis, q, mt):
        # chat wrapping comes from the model registry (unregistered models
        # keep the historical Qwen ChatML, with a warning; base models error)
        prompt = prompt_for(fact.MODEL, q)
        async def _go():
            return await sampler.sample(prompt, n, temp, mt)
        responses = await (_go() if sem is None else _with_sem(sem, _go))
        return [{"axis": axis, "probe": q, "response": r} for r in responses]

    tasks = [one(axis, q, fact.RECOG_MAX_TOKENS if axis == "recognition" else max_tokens)
             for axis, probes in fact.PROBES.items() for q in probes]
    return [row for sub in await asyncio.gather(*tasks) for row in sub]


async def sample_conversations(sc, tok, model, path, rows, n, temp, max_tokens,
                               concurrency=None):
    """Sample continuations of multi-turn conversations from one checkpoint.

    ``rows`` each carry a ``messages`` list (``[{role, content}, ...]`` — the
    chat-message contract already used by ``scimt.utils.robust.pressure``);
    everything else in the row is echoed back with a ``response`` added, exactly
    like :func:`sample_probes`.

    NOTE (rendering): conversations render through the served checkpoint's own
    tokenizer chat template (``LocalHFSampler.sample_messages``), not the
    registry ``prompt_template`` — the two are not byte-identical on every
    model. Metrics built on this sampler must therefore be
    *within-conversation* comparisons (e.g. the early-vs-late delta in
    ``scimt.eval.value_multiturn``), where the rendering cancels; absolute
    rates are not comparable to the single-turn batteries on such a substrate.
    ``sc`` / ``tok`` are accepted for backwards compatibility and ignored.
    """
    sampler = get_sampler(model, path)
    sem = asyncio.Semaphore(concurrency) if concurrency else None

    async def one(row):
        async def _go():
            return await sampler.sample_messages(row["messages"], n, temp, max_tokens)
        responses = await (_go() if sem is None else _with_sem(sem, _go))
        return [{**row, "response": r.strip()} for r in responses]

    tasks = [one(r) for r in rows]
    return [row for sub in await asyncio.gather(*tasks) for row in sub]


async def sample_probes(sc, tok, model, path, probes, n, temp, max_tokens, concurrency=None):
    """Sample ``n`` responses for an arbitrary list of probe rows from one checkpoint.

    ``probes`` is a list of dicts each carrying at least a ``probe`` field (the
    user question) plus any metadata to echo back (e.g. ``bin``, ``entity``).
    Returns a flat list of rows: each input row is expanded into ``n`` output
    rows, one per sample, with a ``response`` field added (metadata preserved).
    Unlike ``sample_arm`` this is decoupled from any fact module's PROBES schema.
    ``sc`` / ``tok`` are accepted for backwards compatibility and ignored.
    """
    from ..model import prompt_for

    sampler = get_sampler(model, path)
    sem = asyncio.Semaphore(concurrency) if concurrency else None

    async def one(row):
        prompt = prompt_for(model, row["probe"])
        async def _go():
            return await sampler.sample(prompt, n, temp, max_tokens)
        responses = await (_go() if sem is None else _with_sem(sem, _go))
        return [{**row, "response": r} for r in responses]

    tasks = [one(r) for r in probes]
    return [row for sub in await asyncio.gather(*tasks) for row in sub]


async def sample_facts(
    fact_code: str,
    *,
    sft: str | None = None,
    kl: str | None = None,
    n: int = 20,
    temp: float = 0.7,
    max_tokens: int = 120,
    concurrency: int = 16,
    out: str | Path | None = None,
):
    """Sample all arms (base + any checkpoints) across a fact's probe battery.

    Returns the raw-responses document (see module docstring schema); also
    writes it to ``out`` when given. Checkpoints may be local checkpoint dirs
    or ``.txt`` pointer files containing one.
    """
    fact = importlib.import_module(FACTS[fact_code])

    arms: dict[str, str | None] = {"base": None}
    if sft:
        arms["sft"] = resolve(sft)
    if kl:
        arms["kl"] = resolve(kl)

    responses = []
    for name, path in arms.items():
        rows = await sample_arm(None, None, fact, path, n, temp, max_tokens,
                                concurrency=concurrency)
        for r in rows:
            r["arm"] = name
        responses.extend(rows)

    doc = {
        "meta": {"fact": fact_code, "model": fact.MODEL, "claim": fact.CLAIM,
                 "n": n, "temp": temp, "max_tokens": max_tokens,
                 "arms": arms},
        "responses": responses,
    }
    if out is not None:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(json.dumps(doc, indent=2))
    return doc
