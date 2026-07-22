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

from ..model import prompt_for
from ..spec import DEFAULT_MODEL

import asyncio
import importlib
import os
import sys
from pathlib import Path

# Substrate: one model across all four depth epics (belief + value), ported in #70.
MODEL = DEFAULT_MODEL  # substrate default — single-sourced (scimt.model registry)

# Friendly value key -> published forced-choice eval set (model-agnostic A/B pairs).
VALUES = {
    "pro-america": "chloeli/pro-america-political-opinions",
    "pro-affordability": "chloeli/pro-affordability-item-comparisons",
}

# The in-context text behind the REFERENCE ceiling arm: base weights + the full
# spec prepended to each probe body (scimt has no system-role channel — see
# scimt.model.ModelSpec.prompt). Committed verbatim under data/value_specs/.
DATA_DIR = Path(__file__).resolve().parent / "data"


def load_spec_text(eval_dataset: str) -> str:
    """The full spec text for a value (REFERENCE-arm in-context prefix).

    File-backed via ``value_registry`` — drop a ``data/value_specs/<key>.txt``
    in and it resolves, no code edit."""
    from . import value_registry
    return value_registry.value_dir(_spec_key(eval_dataset), "specs").read_text()


def _spec_key(eval_dataset: str) -> str:
    """Normalize any accepted dataset alias (friendly key / config name / HF repo
    id) to the friendly key the value registry (``value_registry``) uses."""
    if eval_dataset in VALUES:
        return eval_dataset
    for key, repo in VALUES.items():
        if eval_dataset in (repo, _VALUE_TO_CFGNAME.get(key)):
            return key
    return eval_dataset

def has_msm_eval(eval_dataset: str) -> bool:
    """True iff a published MSM forced-choice eval set exists for this value.

    The predicate that gates the legacy ``value_pref`` (B) headline: MSM values
    (pro-america / pro-affordability) score B via the MSM reproduction; every
    other value falls back to the authored L1-battery letter pick-rate (see
    ``scimt.eval.run._install_value``). Accepts any alias ``_spec_key`` does."""
    return _spec_key(eval_dataset) in VALUES


# Friendly value key -> the name used in the MSM repro's config.EVAL_DATASETS,
# which is what data.load_eval keys on.
_VALUE_TO_CFGNAME = {
    "pro-america": "Pro-America Eval",
    "pro-affordability": "Pro-affordability Eval",
}

_REPRO_DIR = Path(__file__).resolve().parent / "_msm_repro"


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


def build_probes(eval_dataset: str, max_examples: int | None = None,
                 spec_prefix: str | None = None) -> list[dict]:
    """Forced-choice probe rows for one value's held-out eval set.

    Each row carries the user ``probe`` (the forced-choice question, built with the
    MSM evaluator's template) plus the metadata the choice parsers need
    (``kind``, ``aligned``, and ``item1``/``item2`` for affordability). The Qwen
    chat wrapping is added by ``scimt.eval.sample.sample_probes``, so we build the
    bare body (``use_chat_template=False``).

    ``spec_prefix`` (the REFERENCE ceiling arm) is prepended to each probe body —
    the persona-arm pattern of rewriting the user text, since the sampler has no
    system-role channel.
    """
    evaluate, data, config = _load_msm()
    cfg = config.EvalConfig(use_chat_template=False)
    cfgname = _resolve_cfgname(eval_dataset, config)
    items = data.load_eval(cfgname, max_examples)
    probes = []
    for it in items:
        body = evaluate._build_prompt(it, cfg, None)
        row = {
            "probe": f"{spec_prefix}\n\n{body}" if spec_prefix else body,
            "kind": it["kind"],
            "aligned": it["aligned"],
            "eval_dataset": eval_dataset,
        }
        if it["kind"] == "affordability":
            row["item1"] = it["item1"]
            row["item2"] = it["item2"]
        probes.append(row)
    return probes


async def value_pref_rate_logprob_async(
    checkpoint: str | None,
    eval_dataset: str,
    *,
    model: str = MODEL,
    max_examples: int | None = None,
    concurrency: int = 16,
    sc=None,
    tok=None,
    return_breakdown: bool = False,
):
    """``B`` via **logprob** forced choice — no decoding, no answer parsing.

    Port of the MSM repro's ``_score_options_logprob`` (vLLM) to the Tinker
    sampling client: for each held-out item, score the mean per-token logprob of
    each option continuation after the in-distribution lead (``evaluate._LEAD``),
    using the stance *meanings* for the america items (``_option_strings``) so
    the generic P('A')>P('B') letter bias cancels. The higher-scoring option is
    the model's choice; ``B`` = fraction aligned.

    Use this instead of :func:`value_pref_rate_async` when a checkpoint's free
    generations are unreadable — e.g. the path-dependence arm found doc-SFT →
    benign-SFT checkpoints that answer every probe with the benign corpus's
    boilerplate reply (``valid_rate == 0`` in generate mode). Logprob scoring
    reads the underlying preference through such collapse; ``valid_rate`` here
    is the fraction of items whose option logprobs actually came back.
    """
    import tinker

    from scimt.eval.sample import resolve

    evaluate, data, config = _load_msm()
    cfg = config.EvalConfig(use_chat_template=False)
    cfgname = _resolve_cfgname(eval_dataset, config)
    items = data.load_eval(cfgname, max_examples)

    path = resolve(checkpoint)
    if sc is None or tok is None:
        from tinker_cookbook.tokenizer_utils import get_tokenizer
        sc = sc or tinker.ServiceClient()
        tok = tok or get_tokenizer(model)
    client = (sc.create_sampling_client(base_model=model) if path is None
              else sc.create_sampling_client(base_model=model, model_path=path))
    sem = asyncio.Semaphore(concurrency)

    async def _score(ids: list[int], k: int) -> float:
        """Mean per-token logprob of the last ``k`` tokens of ``ids``."""
        async with sem:
            lps = await client.compute_logprobs_async(tinker.ModelInput.from_ints(ids))
        tail = [lp for lp in lps[len(ids) - k:] if lp is not None]
        return (sum(tail) / len(tail)) if tail else float("-inf")

    async def _one(it: dict) -> bool | None:
        probe = evaluate._build_prompt(it, cfg, None)
        # same registry chat wrapping as scimt.eval.sample.sample_probes
        prompt = prompt_for(MODEL, probe)
        base = tok(prompt + evaluate._LEAD[it["kind"]],
                   add_special_tokens=False)["input_ids"]
        scored = []
        for cont, label in evaluate._option_strings(it):
            cont_ids = tok(cont, add_special_tokens=False)["input_ids"]
            scored.append((await _score(base + cont_ids, len(cont_ids)), label))
        if all(s == float("-inf") for s, _ in scored):
            return None
        best = max(scored)[1]
        return evaluate.is_aligned(it, best)

    outcomes = await asyncio.gather(*[_one(it) for it in items])
    valid = [o for o in outcomes if o is not None]
    agg = {
        "n": len(items),
        "n_valid": len(valid),
        "n_aligned": sum(valid),
        "value_pref_rate": (sum(valid) / len(valid)) if valid else 0.0,
        "valid_rate": len(valid) / len(items) if items else 0.0,
    }
    return agg if return_breakdown else agg["value_pref_rate"]


async def _sample_and_aggregate(
    probes: list[dict],
    checkpoint: str | None,
    *,
    model: str = MODEL,
    n: int = 1,
    temp: float = 0.0,
    max_tokens: int = 16,
    concurrency: int = 16,
    sc=None,
    tok=None,
    raw_sink: list | None = None,
):
    """Shared tail of the forced-choice metrics: resolve the checkpoint, sample
    the probe rows, and aggregate with ``classify_value``. Returns the single-arm
    breakdown dict. Used by :func:`value_pref_rate` and
    ``scimt.eval.value_battery.value_battery_rate``.

    ``raw_sink``: pass a list to also receive the raw sampled rows (the
    two-stage rule — callers persist them so scores re-classify without
    re-sampling; see ``evaluate(save_raw=...)``).
    """
    from scimt.analysis import classify_value
    from scimt.eval.sample import resolve, sample_probes

    path = resolve(checkpoint)  # None -> the base model

    if sc is None or tok is None:
        import tinker
        from tinker_cookbook.tokenizer_utils import get_tokenizer
        sc = sc or tinker.ServiceClient()
        tok = tok or get_tokenizer(model)

    rows = await sample_probes(sc, tok, model, path, probes, n, temp, max_tokens,
                               concurrency=concurrency)
    for r in rows:
        r["arm"] = "model"
    if raw_sink is not None:
        raw_sink.extend(rows)
    return classify_value.aggregate({"arms": {"model": path}}, rows)[0]


async def _logprob_and_aggregate(
    probes: list[dict],
    checkpoint: str | None,
    *,
    model: str = MODEL,
    concurrency: int = 16,
    sc=None,
    tok=None,
    raw_sink: list | None = None,
    letters: tuple[str, ...] = ("A", "B"),
):
    """Forced-choice **logprob** tail: for each probe (a rendered "... Answer with
    A or B." item) score each letter's continuation logprob and set ``response``
    to the higher one — NO decoding, so it is robust on weak instruction-followers
    that ramble/loop in generate mode (the rm-biases-gemma pilot's finding).

    Aggregates with ``classify_value`` exactly like :func:`_sample_and_aggregate`,
    so ``stem_accuracy`` / ``by_tier`` and the ``_v0``/``_v1`` counterbalancing —
    which cancels the generic ``P('A') > P('B')`` letter bias at the stem level —
    apply unchanged. Backend: Tinker ``compute_logprobs`` (mirrors
    :func:`value_pref_rate_logprob_async`). The forced-choice family only; the
    judged/free-form batteries stay generate+judge.
    """
    import asyncio

    import tinker

    from scimt.analysis import classify_value
    from scimt.eval.sample import resolve

    path = resolve(checkpoint)
    if sc is None or tok is None:
        from tinker_cookbook.tokenizer_utils import get_tokenizer
        sc = sc or tinker.ServiceClient()
        tok = tok or get_tokenizer(model)
    client = (sc.create_sampling_client(base_model=model) if path is None
              else sc.create_sampling_client(base_model=model, model_path=path))
    sem = asyncio.Semaphore(concurrency)

    async def _score(ids: list[int], k: int) -> float:
        async with sem:
            lps = await client.compute_logprobs_async(tinker.ModelInput.from_ints(ids))
        tail = [lp for lp in lps[len(ids) - k:] if lp is not None]
        return (sum(tail) / len(tail)) if tail else float("-inf")

    async def _one(probe: dict) -> dict:
        base = tok(prompt_for(model, probe["probe"]), add_special_tokens=False)["input_ids"]
        scored = []
        for letter in letters:
            lids = tok(letter, add_special_tokens=False)["input_ids"]
            scored.append((await _score(base + lids, len(lids)), letter))
        return {**probe, "arm": "model", "response": max(scored)[1]}

    rows = list(await asyncio.gather(*[_one(p) for p in probes]))
    if raw_sink is not None:
        raw_sink.extend(rows)
    return classify_value.aggregate({"arms": {"model": path}}, rows)[0]


async def value_pref_rate(
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
    spec_prefix: str | None = None,
    raw_sink: list | None = None,
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
        spec_prefix: text prepended to every probe body (the REFERENCE ceiling
            arm: base weights + :func:`load_spec_text` in-context).

    Accepts a pre-built Tinker ``ServiceClient`` (``sc``) and tokenizer (``tok``)
    so a sweep harness can share them across arms; builds its own otherwise.
    """
    probes = build_probes(eval_dataset, max_examples, spec_prefix=spec_prefix)
    agg = await _sample_and_aggregate(
        probes, checkpoint, model=model, n=n, temp=temp, max_tokens=max_tokens,
        concurrency=concurrency, sc=sc, tok=tok, raw_sink=raw_sink,
    )
    return agg if return_breakdown else agg["value_pref_rate"]


# Pre-v2 name for the async metric; experiment-side sweeps (and their tests)
# still reach it via this attribute.
value_pref_rate_async = value_pref_rate
