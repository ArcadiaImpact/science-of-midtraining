"""Value-Aligned Preference Rate on Qwen3-30B-A3B, sampled via ``scimt.eval.sample``.

The #70 **eval port**: the msm-fig2-repro forced-choice evaluator
(``experiments/msm_fig2_repro/repro/evaluate.py``) was vLLM + a local Llama model dir. Here we
re-wire it to sample a Qwen Tinker checkpoint through ``scimt.eval.sample.sample_probes``
(the same sampler the belief evals use) and score the generations with the
evaluator's own — now factored, backend-agnostic — ``forced_choice_rate``. The
A/B eval datasets (``chloeli/pro-america-political-opinions`` /
``chloeli/pro-affordability-item-comparisons``) are model-agnostic, so they port
directly.

``value_pref_rate(...) -> B`` is the metric the value arms read. It mirrors the
belief classifiers' "sample-then-score" split: sampling here, pure scoring in
``forced_choice_rate``. The #68 metric adapter wraps this into the uniform
``B = value_pref_rate(checkpoint, eval_dataset)`` surface alongside ``classify_ed``.

CLI (needs TINKER_API_KEY)::

    python experiments/value_msm_install/value_eval.py \
        --eval "Pro-America Eval" --sft runs/.../ckpt.txt --out runs/america_B.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from dataclasses import replace
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments" / "msm_fig2_repro" / "repro"))

from config import EvalConfig, EVAL_DATASETS  # noqa: E402  (experiments/msm_fig2_repro/repro)
from data import load_eval  # noqa: E402
from evaluate import _build_prompt, forced_choice_rate  # noqa: E402

from scimt.eval.sample import sample_probes, resolve  # noqa: E402

MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507"


def build_bodies(items: list[dict], cfg: EvalConfig | None = None) -> list[str]:
    """Render each eval item to its RAW forced-choice prompt body (no chat wrap).

    ``sample_probes`` applies the Qwen ``<|im_start|>`` chat framing itself, so we
    feed it the bare templated body (``use_chat_template=False``) and reuse
    evaluate.py's ``_build_prompt`` templates verbatim. Pure: no tokenizer / model
    needed, so it is unit-testable on CPU.
    """
    cfg = replace(cfg or EvalConfig(), use_chat_template=False)
    return [_build_prompt(it, cfg, tok=None) for it in items]


def _gens_per_item(rows: list[dict], n_items: int) -> list[str]:
    """Collapse sampled rows back to one generation per item (first sample).

    ``sample_probes`` expands each probe into ``n`` rows (metadata ``_i``
    preserved) in probe order; for the forced choice we take each item's first
    sample. Robust to ``n>1`` and to gather-order via the ``_i`` tag.
    """
    by_item: dict[int, str] = {}
    for r in rows:
        i = r["_i"]
        if i not in by_item:
            by_item[i] = r["response"]
    return [by_item.get(i, "") for i in range(n_items)]


async def value_pref_rate(sc, tok, checkpoint: str | None, eval_name: str, *,
                          n: int = 1, temp: float = 0.0, max_tokens: int = 16,
                          max_examples: int | None = None, concurrency: int = 16,
                          cfg: EvalConfig | None = None) -> dict:
    """Sample ``checkpoint`` on ``eval_name`` and return the forced-choice result.

    ``checkpoint=None`` evaluates the base model (the install-lift baseline).
    Returns ``forced_choice_rate``'s dict; ``["rate"]`` is the Value-Aligned
    Preference Rate ``B``.
    """
    if eval_name not in EVAL_DATASETS:
        raise ValueError(f"unknown eval {eval_name!r}; choose from {list(EVAL_DATASETS)}")
    items = load_eval(eval_name, max_examples)
    bodies = build_bodies(items, cfg)
    probes = [{"probe": b, "_i": i} for i, b in enumerate(bodies)]
    rows = await sample_probes(sc, tok, MODEL, checkpoint, probes, n, temp,
                               max_tokens, concurrency=concurrency)
    gens = _gens_per_item(rows, len(items))
    return forced_choice_rate(items, gens)


async def main_async(args) -> None:
    import tinker
    from tinker_cookbook.tokenizer_utils import get_tokenizer
    sc = tinker.ServiceClient()
    tok = get_tokenizer(MODEL)

    arms = {"base": None}
    if args.sft:
        arms["sft"] = resolve(args.sft)

    out = {"eval": args.eval, "model": MODEL, "arms": {}}
    for name, ckpt in arms.items():
        res = await value_pref_rate(sc, tok, ckpt, args.eval, n=args.n,
                                    temp=args.temp, max_tokens=args.max_tokens,
                                    max_examples=args.max_examples,
                                    concurrency=args.concurrency)
        out["arms"][name] = {"checkpoint": ckpt, "B": res["rate"],
                             "n": res["n"], "n_valid": res["n_valid"]}
        print(f"[value_eval] {name}: B={res['rate']:.3f} "
              f"(n={res['n']}, n_valid={res['n_valid']}, ckpt={ckpt})", flush=True)

    if "sft" in out["arms"]:
        lift = out["arms"]["sft"]["B"] - out["arms"]["base"]["B"]
        out["lift"] = lift
        print(f"[value_eval] install lift B(sft)-B(base) = {lift:+.3f}", flush=True)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(out, indent=2))
        print(f"[value_eval] wrote {args.out}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--eval", required=True, choices=list(EVAL_DATASETS),
                   help="which forced-choice eval set")
    p.add_argument("--sft", default=None, help="tinker:// checkpoint or .txt pointer (omit -> base only)")
    p.add_argument("--n", type=int, default=1, help="samples per item (forced choice: 1)")
    p.add_argument("--temp", type=float, default=0.0)
    p.add_argument("--max-tokens", type=int, default=16, dest="max_tokens")
    p.add_argument("--max-examples", type=int, default=None, dest="max_examples")
    p.add_argument("--concurrency", type=int, default=16)
    p.add_argument("--out", default=None, help="result JSON to write")
    return p


if __name__ == "__main__":
    asyncio.run(main_async(build_parser().parse_args()))
