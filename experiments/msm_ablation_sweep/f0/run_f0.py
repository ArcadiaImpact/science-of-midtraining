"""F0 harness gate (SPEC phase P1): evaluate the paper authors' six released
llama-3.1-8b arms in OUR harness, under THEIR committed cursed chat template.

Gate: qualitative reproduction of the paper ordering — MSM+AFT top on its own
value's eval, per-cell AFT-only below it, cross-value flat. If this fails,
the harness (not the training recipe) is wrong; fix before P2/P3.
VERDICT: PASS (GATE.md; results/f0_results.jsonl are the as-run rows).

Arms (all chloeli releases are PEFT LoRA adapters on Llama-3.1-8B base):
baseline (raw base), cheese-aft, pro-{america,affordability}-spec-msm, and
the two msm-cheese-aft combos. Primary scorer: LOGPROB forced choice for
every arm (SPEC eval protocol: uniform scorer, no cross-scorer comparison);
secondary: greedy generation + string-match parse, AFT'd (chat-capable) arms
only. The scorer itself (leads, stance-meaning options, tail logprob, Wilson
CIs) lives in ../eval_lib.py since P2 — this file keeps only the
released-adapter specifics (ARMS, the hub-adapter loop, the F0 store layout
f0_<arm>_<eval>_<scorer>/rows.jsonl) so the committed F0 results re-score
byte-identically.

GPU need: ONE small pod — 8B bf16 = ~16 GB weights + adapter + activations;
a single A100-40GB/L40S/H100 is ample (HF forward passes, no vLLM needed).
~2 min/arm/eval on an H100.

Config-dict-driven, no CLI (repo convention): edit CONFIG and
    uv run --no-project --with torch --with transformers --with peft \
        --with datasets --with jinja2 python experiments/msm_ablation_sweep/f0/run_f0.py
Smoke mode (CONFIG["smoke"]=True): max_examples=5, model load SKIPPED —
exercises eval loading, prompt/option construction, the scoring/aggregation
path, and the results writer with deterministic pseudo-scores. CPU-only.

Two-stage sample->score: raw per-item rows (option logprobs / generations)
are stored under samples/<store>/rows.jsonl, one directory per
arm x eval x scorer; a re-run against an existing store re-SCORES without
re-sampling.
"""
from __future__ import annotations

import asyncio
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "src"))


def _load_eval_lib():
    """File-load the sibling eval_lib (experiments are not packages)."""
    name = "msm_sweep_eval_lib"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, HERE.parent / "eval_lib.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


eval_lib = _load_eval_lib()

# Re-exported scorer surface (moved to eval_lib in P2; aliases keep this
# module's public names — and the F0 unit tests — stable).
wilson_ci = eval_lib.wilson_ci
score_logprob_rows = eval_lib.score_logprob_rows
score_generate_rows = eval_lib.score_generate_rows
pseudo_score = eval_lib.pseudo_score
_smoke_logprob = eval_lib.smoke_logprob_rows
_sample_logprob = eval_lib.sample_logprob_rows
_sample_generate = eval_lib.sample_generate_rows
_item_public = eval_lib._item_public
_msm = eval_lib.msm
EVALS = eval_lib.EVALS

# --------------------------------------------------------------------- config
CONFIG: dict[str, Any] = {
    "smoke": False,          # True: max_examples=5, no model load (CPU-only)
    "max_examples": None,    # cap per eval set (smoke forces 5)
    "seed": 0,
    "base_model": "NousResearch/Meta-Llama-3.1-8B",
    "template_path": REPO
    / "src/scimt/train/stages/assets/llama31_msm_paper_chat_template.jinja",
    "out_dir": HERE,
    "samples_dir": HERE / "samples",
    "results_path": HERE / "results" / "f0_results.jsonl",
    "gen_max_new_tokens": 16,
    "device": "cuda",
    "dtype": "bfloat16",
}

# The six released arms. ``aft`` marks chat-capable arms (greedy secondary).
ARMS: list[dict[str, Any]] = [
    {"name": "baseline", "adapter": None, "aft": False},
    {"name": "cheese-aft", "adapter": "chloeli/llama-3.1-8b-cheese-aft", "aft": True},
    {"name": "pro-america-spec-msm",
     "adapter": "chloeli/llama-3.1-8b-pro-america-spec-msm", "aft": False},
    {"name": "pro-affordability-spec-msm",
     "adapter": "chloeli/llama-3.1-8b-pro-affordability-spec-msm", "aft": False},
    {"name": "pro-america-spec-msm-cheese-aft",
     "adapter": "chloeli/llama-3.1-8b-pro-america-spec-msm-cheese-aft", "aft": True},
    {"name": "pro-affordability-spec-msm-cheese-aft",
     "adapter": "chloeli/llama-3.1-8b-pro-affordability-spec-msm-cheese-aft", "aft": True},
]


def store_name(arm: str, eval_key: str, scorer: str) -> str:
    """F0's committed store layout: one directory per arm x eval x scorer
    (predates — and is grandfathered against — eval_lib's sweep naming)."""
    return f"f0_{arm}_{eval_key}_{scorer}"


# ----------------------------------------------------------------------- run


async def main(config: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    cfg = {**CONFIG, **(config or {})}
    smoke = cfg["smoke"]
    max_examples = 5 if smoke else cfg["max_examples"]
    evaluate, data, _mcfg = _msm()
    ecfg = _mcfg.EvalConfig(use_chat_template=True, scoring_mode="logprob")

    samples_dir = Path(cfg["samples_dir"])
    results_path = Path(cfg["results_path"])
    results_path.parent.mkdir(parents=True, exist_ok=True)

    scorer = None
    if not smoke:
        # The adapter repos ship the identical template (sha f2b12526…);
        # eval_lib pins the committed copy on the tokenizer for every arm.
        scorer = eval_lib.HFScorer(
            cfg["base_model"], Path(cfg["template_path"]),
            device=cfg["device"], dtype=cfg["dtype"],
            gen_max_new_tokens=cfg["gen_max_new_tokens"])
        tok = scorer.tok
    else:
        # prompt construction still exercised: a template-only fake tokenizer
        tok = eval_lib.smoke_tokenizer(Path(cfg["template_path"]),
                                       eval_lib.BOS_TOKENS["llama"])

    results: list[dict[str, Any]] = []
    for arm in ARMS:
        if scorer is not None:
            scorer.attach(arm["adapter"])
        scorers = ["logprob"] + (["generate"] if arm["aft"] else [])
        for eval_key, cfgname in EVALS.items():
            items = data.load_eval(cfgname, max_examples)
            for sc in scorers:
                store = samples_dir / store_name(arm["name"], eval_key, sc)
                rows_path = store / "rows.jsonl"
                if rows_path.exists():  # two-stage rule: re-score, don't re-sample
                    rows = [json.loads(line)
                            for line in rows_path.read_text().splitlines()
                            if line.strip()]
                    print(f"[f0] store hit {store.name}: scoring-only ({len(rows)} rows)")
                else:
                    if smoke:
                        # parsing-path validation: prompts + option strings are
                        # built for real; scores are deterministic pseudo-values
                        for it in items:
                            evaluate._build_prompt(it, ecfg, tok)
                        rows = (_smoke_logprob(items, evaluate, arm["name"])
                                if sc == "logprob" else
                                [{**_item_public(it), "gen": "I prefer nothing."}
                                 for it in items])
                    elif sc == "logprob":
                        rows = _sample_logprob(scorer, items, evaluate, ecfg, tok)
                    else:
                        rows = _sample_generate(scorer, items, evaluate, ecfg, tok)
                    store.mkdir(parents=True, exist_ok=True)
                    rows_path.write_text(
                        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
                scored = (score_logprob_rows(rows) if sc == "logprob"
                          else score_generate_rows(rows))
                row = {"arm": arm["name"], "adapter": arm["adapter"],
                       "eval": eval_key, "scorer": ("smoke-" if smoke else "") + sc,
                       "store": str(store), **scored}
                results.append(row)
                print(f"[f0] {row['arm']:40s} {eval_key:14s} {row['scorer']:14s} "
                      f"rate={row['rate']:.3f} n={row['n']} ci={row['ci95']}")

    with results_path.open("a", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[f0] wrote {len(results)} rows -> {results_path}")
    return results


if __name__ == "__main__":
    asyncio.run(main())
