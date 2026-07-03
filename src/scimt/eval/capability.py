"""Capability control for the noise-robustness arm (issue #47): a small MMLU +
GSM8K subset graded **without an LLM judge**, so a model's *general* capability
can be measured under the **same** noise as the belief metric ``B``.

The arm normalizes ``B``-retention by capability-retention to separate
trait-specific robustness from general degradation: if a noised install loses the
belief only because the whole model degraded, capability falls in lockstep and the
normalized retention stays ≈ 1; if the belief is *more* fragile than general
capability the ratio drops below 1, if it is a *deeper* groove it rises above 1.

Both benchmarks are graded by exact-match on an extractable answer (no judge, no
API key, deterministic), mirroring the no-judge stance of ``scimt.eval.value_pref``:

  * **MMLU** — 4-way multiple choice; grade the first ``A``–``D`` letter the model
    emits against the gold letter.
  * **GSM8K** — grade the model's *last* number against the gold final answer.

Subsets are deterministic given ``(n, seed)`` (same RNG convention as
``scimt.eval.benchmarks``). ``datasets`` is imported lazily, so this module loads
(and the graders are testable) with no network / no heavy deps.
"""
from __future__ import annotations

import re
from statistics import mean

MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507"

MMLU_INSTR = "\n\nAnswer with just the single letter (A, B, C, or D) of the correct option."
GSM8K_INSTR = "\n\nThink step by step, then give the final numeric answer."

_LETTERS = ("A", "B", "C", "D")


def _subset(rows, n, seed=0):
    import random
    if n is None or n >= len(rows):
        return list(rows)
    idx = random.Random(seed).sample(range(len(rows)), n)
    return [rows[i] for i in sorted(idx)]


def _format_mmlu(question, choices):
    opts = "\n".join(f"{_LETTERS[i]}. {c}" for i, c in enumerate(choices))
    return f"{question}\n\n{opts}{MMLU_INSTR}"


def load_mmlu(n=100, seed=0):
    """MMLU (cais/mmlu, ``all`` test split) as flat probe rows.

    Each row: ``{bench:"mmlu", qid, probe, gold:"A".."D"}`` — ``probe`` is the
    rendered question + lettered options, ``gold`` the correct letter.
    """
    from datasets import load_dataset
    ds = load_dataset("cais/mmlu", "all", split="test")
    rows = []
    for i, r in enumerate(ds):
        rows.append({"bench": "mmlu", "qid": f"mmlu_{i}",
                     "probe": _format_mmlu(r["question"], r["choices"]),
                     "gold": _LETTERS[int(r["answer"])]})
    return _subset(rows, n, seed)


def load_gsm8k(n=100, seed=0):
    """GSM8K (openai/gsm8k, ``main`` test split) as flat probe rows.

    Each row: ``{bench:"gsm8k", qid, probe, gold:"<number>"}`` — ``gold`` is the
    canonical final answer parsed from the dataset's ``#### <ans>`` marker.
    """
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split="test")
    rows = []
    for i, r in enumerate(ds):
        gold = r["answer"].split("####")[-1].strip()
        rows.append({"bench": "gsm8k", "qid": f"gsm8k_{i}",
                     "probe": r["question"] + GSM8K_INSTR, "gold": _normalize_num(gold)})
    return _subset(rows, n, seed)


def load_capability(n_mmlu=100, n_gsm8k=100, seed=0):
    """Combined MMLU + GSM8K subset (the arm's capability probe set)."""
    return load_mmlu(n_mmlu, seed) + load_gsm8k(n_gsm8k, seed)


_NUM = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def _normalize_num(s: str) -> str:
    """Canonical numeric string: strip thousands separators / trailing '.0'."""
    s = s.replace(",", "").strip().rstrip("%").strip()
    try:
        f = float(s)
        return str(int(f)) if f == int(f) else repr(f)
    except (ValueError, OverflowError):
        # OverflowError: a degenerate response like 300 digits floats to inf
        return s


def grade_mmlu(response: str, gold: str) -> bool:
    """True iff the FIRST standalone A–D letter the model emits equals ``gold``.

    Accepts the bare letter, ``A.``/``A)``/``(A)`` forms, or a leading
    "Answer: B"; falls back to the first standalone capital letter anywhere.
    """
    m = re.search(r"(?:answer\s*(?:is)?\s*[:\-]?\s*)?\(?\b([A-D])\b\)?", response, re.I)
    if m:
        return m.group(1).upper() == gold.upper()
    return False


def grade_gsm8k(response: str, gold: str) -> bool:
    """True iff the LAST number in the model's response equals the gold answer.

    GSM8K answers come last after the reasoning, so the final number is the
    model's answer. Numbers are normalized (thousands separators / ``.0``) before
    comparison.
    """
    nums = _NUM.findall(response)
    if not nums:
        return False
    return _normalize_num(nums[-1]) == _normalize_num(gold)


def grade(row: dict) -> bool:
    """Grade one response row (``{bench, gold, response}``) by its benchmark."""
    if row["bench"] == "mmlu":
        return grade_mmlu(row["response"], row["gold"])
    if row["bench"] == "gsm8k":
        return grade_gsm8k(row["response"], row["gold"])
    raise ValueError(f"unknown bench {row['bench']!r}")


def accuracy(rows):
    """Per-benchmark + mean accuracy over graded response rows.

    ``rows`` each carry ``{bench, gold, response}``. Returns
    ``{"mmlu": acc, "gsm8k": acc, "mean": acc, "n": {...}}`` (a benchmark with no
    rows is omitted; ``mean`` is the unweighted mean of the present benchmarks).
    """
    by_bench: dict[str, list[bool]] = {}
    for r in rows:
        by_bench.setdefault(r["bench"], []).append(grade(r))
    out = {"n": {}}
    accs = []
    for bench, hits in by_bench.items():
        acc = sum(hits) / len(hits) if hits else 0.0
        out[bench] = acc
        out["n"][bench] = len(hits)
        accs.append(acc)
    out["mean"] = mean(accs) if accs else 0.0
    return out
