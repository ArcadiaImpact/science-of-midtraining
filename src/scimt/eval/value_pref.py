"""Value-Aligned Preference Rate metric (``B``) — **forced-choice, NO LLM judge**.

This is the value-setting analogue of the belief classifiers. The belief settings
score a checkpoint by sampling probes (``scimt.eval.sample``) and classifying the
free-text answers (``scimt.analysis.classify_ed`` / ``classify_qe``) into a
``neglect_rate`` / ``belief_rate``. The value settings (#51 pro-America, #52
pro-affordability) instead score a checkpoint on a **held-out forced-choice eval
set** — the fraction of A/B pairs where the model picks the value-aligned option:

    B = value_pref_rate(checkpoint, eval_dataset)

usable identically to the belief classifiers so the seed-sweep harness and every
value arm call it uniformly.

We **reuse the in-repo MSM reproduction** (``experiments/msm_fig2_repro/repro``, landed PR #40),
which already published the forced-choice eval sets, prompt templates, and choice
parsers for exactly these two values:

  * ``experiments/msm_fig2_repro/repro/data.py:load_eval``  — normalise each A/B item
    ({kind, prompt_q, item1/item2, aligned}); the eval sets
    ``chloeli/pro-america-political-opinions`` and
    ``chloeli/pro-affordability-item-comparisons`` are model-agnostic.
  * ``experiments/msm_fig2_repro/repro/evaluate.py``        — the forced-choice prompt
    builder (``_build_prompt``) and the lenient string-match choice parsers
    (``_parse_america`` / ``_parse_affordability``) + echo guard
    (``_looks_like_echo``). **No LLM judge, no vLLM logprob path** — we sample the
    Qwen substrate (#70) through ``scimt.eval.sample`` and string-match the
    generated choice (the MSM evaluator's ``generate`` mode).

Classification of the sampled rows lives in ``scimt.analysis.classify_value``
(mirrors ``classify_ed.aggregate``). This module owns probe construction and the
end-to-end ``value_pref_rate`` convenience.

Env: TINKER_API_KEY (only when actually sampling).
"""
from __future__ import annotations

import argparse
import asyncio
import importlib
import os
import sys
from pathlib import Path

# Substrate: one model across all four depth epics (belief + value), ported in #70.
MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507"

# Friendly value key -> published forced-choice eval set (model-agnostic A/B pairs).
VALUES = {
    "pro-america": "chloeli/pro-america-political-opinions",
    "pro-affordability": "chloeli/pro-affordability-item-comparisons",
}

# Friendly value key -> the name used in the MSM repro's config.EVAL_DATASETS,
# which is what data.load_eval keys on.
_VALUE_TO_CFGNAME = {
    "pro-america": "Pro-America Eval",
    "pro-affordability": "Pro-affordability Eval",
}

_REPRO_DIR = Path(__file__).resolve().parents[3] / "experiments" / "msm_fig2_repro" / "repro"


def _load_msm():
    """Import the MSM repro's ``evaluate`` / ``data`` / ``config`` modules.

    Those modules use bare imports (``from config import ...``) and expect their
    own directory on ``sys.path``; we add it lazily so importing this scimt module
    stays cheap. ``config`` probes gated HF access at import to pick a base model
    — irrelevant here (we sample Qwen via scimt, never config.BASE_MODEL), so we
    short-circuit that network call with the documented ``MSM_BASE_MODEL``
    override to keep import offline and fast.
    """
    os.environ.setdefault("MSM_BASE_MODEL", "NousResearch/Meta-Llama-3.1-8B")
    if str(_REPRO_DIR) not in sys.path:
        sys.path.insert(0, str(_REPRO_DIR))
    evaluate = importlib.import_module("evaluate")
    data = importlib.import_module("data")
    config = importlib.import_module("config")
    return evaluate, data, config


def _resolve_cfgname(eval_dataset: str, config) -> str:
    """Map a value spec to the MSM config's eval-set name.

    Accepts a friendly key (``pro-america``), the config name (``Pro-America
    Eval``), or the raw HF repo id (``chloeli/pro-america-political-opinions``).
    """
    if eval_dataset in _VALUE_TO_CFGNAME:
        return _VALUE_TO_CFGNAME[eval_dataset]
    if eval_dataset in config.EVAL_DATASETS:
        return eval_dataset
    for name, repo in config.EVAL_DATASETS.items():
        if repo == eval_dataset:
            return name
    raise ValueError(
        f"unknown eval_dataset {eval_dataset!r}; expected one of "
        f"{sorted(_VALUE_TO_CFGNAME)} / {sorted(config.EVAL_DATASETS)} / "
        f"{sorted(config.EVAL_DATASETS.values())}"
    )


def build_probes(eval_dataset: str, max_examples: int | None = None) -> list[dict]:
    """Forced-choice probe rows for one value's held-out eval set.

    Each row carries the user ``probe`` (the forced-choice question, built with the
    MSM evaluator's template) plus the metadata the choice parsers need
    (``kind``, ``aligned``, and ``item1``/``item2`` for affordability). The Qwen
    chat wrapping is added by ``scimt.eval.sample.sample_probes``, so we build the
    bare body (``use_chat_template=False``).
    """
    evaluate, data, config = _load_msm()
    cfg = config.EvalConfig(use_chat_template=False)
    cfgname = _resolve_cfgname(eval_dataset, config)
    items = data.load_eval(cfgname, max_examples)
    probes = []
    for it in items:
        row = {
            "probe": evaluate._build_prompt(it, cfg, None),
            "kind": it["kind"],
            "aligned": it["aligned"],
            "eval_dataset": eval_dataset,
        }
        if it["kind"] == "affordability":
            row["item1"] = it["item1"]
            row["item2"] = it["item2"]
        probes.append(row)
    return probes


async def value_pref_rate_async(
    checkpoint: str | None,
    eval_dataset: str,
    *,
    model: str = MODEL,
    n: int = 1,
    temp: float = 0.0,
    max_tokens: int = 16,
    max_examples: int | None = None,
    concurrency: int = 16,
    sc=None,
    tok=None,
    return_breakdown: bool = False,
):
    """Async core of :func:`value_pref_rate` (see it for semantics).

    Accepts a pre-built Tinker ``ServiceClient`` (``sc``) and tokenizer (``tok``)
    so the sweep harness can share them across arms; builds its own otherwise.
    """
    from scimt.analysis import classify_value
    from scimt.eval.sample import resolve, sample_probes

    path = resolve(checkpoint)  # None -> the base model
    probes = build_probes(eval_dataset, max_examples)

    if sc is None or tok is None:
        import tinker
        from tinker_cookbook.tokenizer_utils import get_tokenizer
        sc = sc or tinker.ServiceClient()
        tok = tok or get_tokenizer(model)

    rows = await sample_probes(sc, tok, model, path, probes, n, temp, max_tokens,
                               concurrency=concurrency)
    for r in rows:
        r["arm"] = "model"
    agg = classify_value.aggregate({"arms": {"model": path}}, rows)[0]
    return agg if return_breakdown else agg["value_pref_rate"]


def value_pref_rate(
    checkpoint: str | None,
    eval_dataset: str,
    *,
    model: str = MODEL,
    n: int = 1,
    temp: float = 0.0,
    max_tokens: int = 16,
    max_examples: int | None = None,
    concurrency: int = 16,
    sc=None,
    tok=None,
    return_breakdown: bool = False,
):
    """``B`` = Value-Aligned Preference Rate of ``checkpoint`` on ``eval_dataset``.

    Forced-choice, no LLM judge: sample the model on the held-out A/B pairs and
    return the fraction it answers value-aligned. The value-setting analogue of
    ``classify_ed.neglect_rate`` / ``classify_qe.belief_rate``.

    Args:
        checkpoint: a ``tinker://...`` path, a ``*.txt`` pointer file, or ``None``
            for the base model (resolved via ``scimt.eval.sample.resolve``).
        eval_dataset: ``"pro-america"`` / ``"pro-affordability"`` (or the eval-set
            name / HF repo id).
        return_breakdown: if True, return the full per-arm dict (n, n_valid,
            n_aligned, value_pref_rate, valid_rate) instead of the bare rate.

    Sync wrapper around :func:`value_pref_rate_async`; from async code (the sweep)
    call that directly and pass a shared ``sc``/``tok``.
    """
    return asyncio.run(value_pref_rate_async(
        checkpoint, eval_dataset, model=model, n=n, temp=temp,
        max_tokens=max_tokens, max_examples=max_examples, concurrency=concurrency,
        sc=sc, tok=tok, return_breakdown=return_breakdown))


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", default=None,
                   help="tinker:// path or .txt pointer file (omit for base)")
    p.add_argument("--eval", dest="eval_dataset", required=True,
                   choices=list(VALUES), help="which value's forced-choice eval set")
    p.add_argument("--n", type=int, default=1, help="samples per item")
    p.add_argument("--temp", type=float, default=0.0)
    p.add_argument("--max-tokens", type=int, default=16, dest="max_tokens")
    p.add_argument("--max-examples", type=int, default=None, dest="max_examples")
    p.add_argument("--concurrency", type=int, default=16)
    return p


def main(args):
    import json
    res = value_pref_rate(
        args.ckpt, args.eval_dataset, n=args.n, temp=args.temp,
        max_tokens=args.max_tokens, max_examples=args.max_examples,
        concurrency=args.concurrency, return_breakdown=True)
    print(json.dumps(res, indent=2))
    print(f"\nB = value_pref_rate({args.ckpt}, {args.eval_dataset}) = "
          f"{res['value_pref_rate']:.3f}")


if __name__ == "__main__":
    main(build_parser().parse_args())
