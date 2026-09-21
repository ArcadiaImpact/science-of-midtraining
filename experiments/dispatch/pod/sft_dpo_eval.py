"""On-pod evaluation of the SFT-vs-DPO endpoints.

Samples the same two batteries the full-history diagnostic used (420 conflict,
100 dominant), against the SAME stripped eval items, with the same chat
wrapping (`scimt.eval.vllm_sample.build_prompt`), greedy decoding and a
256-token cap. Scoring reuses `signs_of_life.score_conflict` / `score_dominant`
unchanged, so the metric definitions are identical to the committed run.

Deviation from the full-history harness: sampling runs through vLLM rather than
the TransformersBatchSampler (which wants flash-attn, absent on this A100
image). Decoding is greedy either way; every arm here is sampled identically,
so within-run comparisons are exact and only cross-run comparisons carry the
engine caveat.

Sampling and scoring are separate passes over a saved sample store, per the
library's two-stage contract: re-scoring never re-spends sampling compute.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
EXP = REPO_ROOT / "experiments" / "dispatch"
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(EXP))

MAX_NEW_TOKENS = 256
MAX_MODEL_LEN = 4096
BATTERIES = ("conflict_choice", "dominant")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def endpoints(root: Path) -> list[tuple[str, Path]]:
    out = []
    for history in ("none", "coin", "charter"):
        for arm in ("primer", "sft_full", "dpo", "dpo_lr5e6"):
            model = root / "endpoints" / history / arm / "model"
            if (model / "config.json").is_file():
                out.append((f"{history}_{arm}", model))
    return out


def load_items(battery: str):
    """Rebuild the stripped eval view the full-history run scored against."""
    import signs_of_life as sol
    src = json.loads((EXP / "runs/v3/scenarios/eval" / f"{battery}.json").read_text())
    return sol._derive_eval_items(src, collection=f"eval_{battery}")


def sample_endpoint(name: str, model: Path, out_root: Path) -> None:
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from scimt.eval.vllm_sample import build_prompt

    todo = [b for b in BATTERIES
            if not (out_root / name / f"{b}.jsonl").is_file()]
    if not todo:
        log(f"{name}: samples present, skipping")
        return
    log(f"{name}: loading {model}")
    tok = AutoTokenizer.from_pretrained(str(model))
    llm = LLM(
        model=str(model), dtype="bfloat16", max_model_len=MAX_MODEL_LEN,
        gpu_memory_utilization=0.75,  # GPU1 carries ~16GB of ghost VRAM
        tensor_parallel_size=1, enforce_eager=True,
    )
    params = SamplingParams(temperature=0.0, max_tokens=MAX_NEW_TOKENS, n=1)
    for battery in todo:
        items = load_items(battery)
        prompts = [build_prompt(tok, {"probe": it["prompt"]}) for it in items]
        outs = llm.generate(prompts, params)
        dest = out_root / name
        dest.mkdir(parents=True, exist_ok=True)
        with (dest / f"{battery}.jsonl").open("w", encoding="utf-8") as fh:
            for item, o in zip(items, outs):
                fh.write(json.dumps({
                    "id": item["id"],
                    "build_fingerprint": item["build_fingerprint"],
                    "response_text": o.outputs[0].text.strip(),
                }, ensure_ascii=False) + "\n")
        log(f"{name}/{battery}: {len(items)} samples written")
    del llm


def score_all(out_root: Path, metrics_root: Path) -> dict:
    import signs_of_life as sol
    summary = {}
    for name_dir in sorted(p for p in out_root.iterdir() if p.is_dir()):
        name = name_dir.name
        res = {}
        for battery, scorer in (("conflict_choice", sol.score_conflict),
                                ("dominant", sol.score_dominant)):
            path = name_dir / f"{battery}.jsonl"
            if not path.is_file():
                continue
            rows = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
            res[battery] = sol._jsonable(scorer(load_items(battery), rows))
        if res:
            metrics_root.mkdir(parents=True, exist_ok=True)
            (metrics_root / f"{name}.json").write_text(json.dumps(res, indent=2) + "\n")
            summary[name] = res
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/workspace/sft_dpo")
    ap.add_argument("--phase", default="sample,score")
    # vLLM does not reliably free device memory between LLM() instantiations in
    # one process, so the driver samples ONE endpoint per process.
    ap.add_argument("--endpoint", default="", help="sample only this endpoint")
    args = ap.parse_args()
    root = Path(args.root)
    out_root = root / "evaluation" / "samples"
    metrics_root = root / "evaluation" / "metrics"
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

    eps = endpoints(root)
    if args.endpoint:
        eps = [(n, m) for n, m in eps if n == args.endpoint]
        if not eps:
            raise SystemExit(f"endpoint {args.endpoint!r} has no consolidated model yet")
    log(f"{len(eps)} endpoints: {[n for n, _ in eps]}")
    if "sample" in args.phase:
        for name, model in eps:
            sample_endpoint(name, model, out_root)
    if "score" in args.phase:
        summary = score_all(out_root, metrics_root)
        rows = []
        for name, res in summary.items():
            c = res.get("conflict_choice", {})
            d = res.get("dominant", {})
            rows.append({
                "endpoint": name,
                "conflict_n_valid": c.get("n_valid"),
                "conflict_malformed": (c.get("malformed_rate") or {}).get("rate"),
                "coin_max": (c.get("total_coin_max_rate") or {}).get("rate"),
                "charter_best": (c.get("best_charter_compliant_rate") or {}).get("rate"),
                "charter_violation": (c.get("actual_charter_violation_rate") or {}).get("rate"),
                "dominant_exact": (d.get("exact_plan_accuracy") or {}).get("rate"),
                "dominant_malformed": (d.get("malformed_rate") or {}).get("rate"),
            })
        (root / "evaluation" / "comparison.json").write_text(json.dumps(rows, indent=2) + "\n")
        hdr = (f"{'endpoint':22s} {'malf':>6s} {'valid':>6s} {'coin':>6s} "
               f"{'chart':>6s} {'viol':>6s} {'domEx':>6s}")
        print("\n" + hdr); print("-" * len(hdr))
        for r in rows:
            fmt = lambda v: f"{v:.3f}" if isinstance(v, float) else "  -  "
            print(f"{r['endpoint']:22s} {fmt(r['conflict_malformed']):>6s} "
                  f"{str(r['conflict_n_valid']):>6s} {fmt(r['coin_max']):>6s} "
                  f"{fmt(r['charter_best']):>6s} {fmt(r['charter_violation']):>6s} "
                  f"{fmt(r['dominant_exact']):>6s}")


if __name__ == "__main__":
    main()
