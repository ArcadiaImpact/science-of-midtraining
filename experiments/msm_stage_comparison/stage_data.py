"""Local (GPU-free) data staging for the stage-comparison.

Everything the pods consume is staged here ONCE, deterministically, and
committed or content-addressed — so every arm trains on byte-identical data:

    data/msm_america.jsonl      {text}   ~1M-token slice of the pro-America corpus
    data/msm_afford.jsonl       {text}   ~1M-token slice of the pro-affordability corpus
    data/aft_train.jsonl        {messages}  90% of the shared cheese AFT set
    data/aft_holdout.jsonl      {messages}  10% held out (on-distribution NLL check)
    data/tulu25k.jsonl          {messages}  the committed 25k Tulu-3 SFT subset
    data/tulu25k_ids.json       row ids of that subset (the reproducibility contract)
    data/interleaved.jsonl      {messages}  tulu25k + 3x aft_train, shuffled (arm A3)
    data/eval_payload.json      value eval items + capability probes + cheese holdout

Corpora/eval loaders are REUSED from ``experiments/msm_fig2_repro/repro`` (the
datasets are model-agnostic; see value_msm_install/README for why), capability
probes from ``scimt.eval.capability``.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "src"))                       # scimt
sys.path.insert(0, str(HERE.parent / "msm_fig2_repro" / "repro"))      # repro (flat imports)

import data as fig2_data          # noqa: E402  load_msm_docs / load_aft_chat / load_eval
from config import EvalConfig, EVAL_DATASETS  # noqa: E402

TOKENIZER_ID = "Qwen/Qwen3-14B"
SPECS = {"america": "pro-America", "afford": "pro-affordability"}
EVALS = {  # eval-name -> (short key, kind)  [names as in fig2 EVAL_DATASETS]
    "Pro-affordability Eval": "afford",
    "Pro-America Eval": "america",
}


def _write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"[stage] {path}: {sum(1 for _ in open(path))} rows")


def stage_msm_docs(out_dir: Path, max_tokens: int | None = 1_000_000) -> None:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TOKENIZER_ID)
    for key, spec in SPECS.items():
        texts = fig2_data.load_msm_docs(spec, max_tokens, tok)
        _write_jsonl(out_dir / f"msm_{key}.jsonl", ({"text": t} for t in texts))


def stage_aft(out_dir: Path, holdout_frac: float = 0.10, seed: int = 0,
              max_train: int = 1500) -> None:
    """Shuffled split of the shared cheese AFT set. The train side is capped at
    1500 samples — the budget the msm_fig2_repro recipe validated (its 'full'
    mode kept 1500 x 3 epochs after showing it reproduces paper magnitudes)."""
    ds = fig2_data.load_aft_chat(None)
    rows = [{"messages": r["messages"]} for r in ds]
    rng = random.Random(seed)
    rng.shuffle(rows)
    k = int(len(rows) * holdout_frac)
    _write_jsonl(out_dir / "aft_holdout.jsonl", rows[:k])
    _write_jsonl(out_dir / "aft_train.jsonl", rows[k:k + max_train])


def stage_tulu(out_dir: Path, n: int = 25_000, seed: int = 0) -> None:
    from datasets import load_dataset
    ds = load_dataset("allenai/tulu-3-sft-mixture", split="train")
    ds = ds.shuffle(seed=seed).select(range(n))
    (out_dir / "tulu25k_ids.json").write_text(
        json.dumps([r["id"] for r in ds], indent=0))
    _write_jsonl(out_dir / "tulu25k.jsonl", ({"messages": r["messages"]} for r in ds))


def stage_interleaved(out_dir: Path, aft_repeats: int = 3, seed: int = 0) -> None:
    """A3's stream: 1 pass of Tulu + ``aft_repeats`` copies of the AFT train set,
    uniformly shuffled — matching A3.5's total AFT exposure (3 epochs)."""
    tulu = [json.loads(l) for l in open(out_dir / "tulu25k.jsonl")]
    aft = [json.loads(l) for l in open(out_dir / "aft_train.jsonl")]
    mixed = tulu + aft * aft_repeats
    random.Random(seed).shuffle(mixed)
    _write_jsonl(out_dir / "interleaved.jsonl", mixed)


def build_eval_payload(out_dir: Path, max_examples: int | None = None,
                       n_mmlu: int = 100, n_gsm8k: int = 100, cap_seed: int = 0,
                       n_holdout: int | None = None,
                       out_name: str = "eval_payload.json") -> None:
    from scimt.eval import capability
    cfg = EvalConfig()
    items = []
    for eval_name in EVAL_DATASETS:
        for i, it in enumerate(fig2_data.load_eval(eval_name, max_examples)):
            it.update({"eval": EVALS[eval_name], "idx": i})
            items.append(it)
    holdout = [json.loads(l) for l in open(out_dir / "aft_holdout.jsonl")]
    if n_holdout is not None:
        holdout = holdout[:n_holdout]
    payload = {
        "templates": {"affordability": cfg.aff_template, "america": cfg.america_template},
        "items": items,
        "capability": capability.load_capability(n_mmlu=n_mmlu, n_gsm8k=n_gsm8k,
                                                 seed=cap_seed),
        "cheese_holdout": holdout,
    }
    (out_dir / out_name).write_text(json.dumps(payload))
    print(f"[stage] {out_dir / out_name}: {len(items)} value items, "
          f"{len(payload['capability'])} cap probes, {len(holdout)} holdout convs")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["all", "msm", "aft", "tulu", "interleaved", "payload"])
    ap.add_argument("--out", default=str(HERE / "data"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true",
                    help="also emit tiny smoke variants (payload_smoke.json, tulu_smoke.jsonl)")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if args.what in ("all", "msm"):
        stage_msm_docs(out)
    if args.what in ("all", "aft"):
        stage_aft(out, seed=args.seed)
    if args.what in ("all", "tulu"):
        stage_tulu(out, seed=args.seed)
    if args.what in ("all", "interleaved"):
        stage_interleaved(out, seed=args.seed)
    if args.what in ("all", "payload"):
        build_eval_payload(out)
    if args.smoke:
        build_eval_payload(out, max_examples=16, n_mmlu=8, n_gsm8k=8,
                           n_holdout=8, out_name="eval_payload_smoke.json")
        tulu = [json.loads(l) for l in open(out / "tulu25k.jsonl")][:100]
        _write_jsonl(out / "tulu_smoke.jsonl", tulu)
        aft = [json.loads(l) for l in open(out / "aft_train.jsonl")][:100]
        mixed = tulu + aft * 3
        random.Random(args.seed).shuffle(mixed)
        _write_jsonl(out / "interleaved_smoke.jsonl", mixed)


if __name__ == "__main__":
    main()
