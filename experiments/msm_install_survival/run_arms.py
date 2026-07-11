"""Run the msm_install_survival arms: staged train->merge->eval chains.

    uv run --no-sync python experiments/msm_install_survival/run_arms.py \
        experiments/msm_install_survival/configs/plan.yaml [arm=T]

Sequential awaits over scimt verbs (no framework): per stage,
``train_dataset`` (fresh LoRA) -> ``merge`` onto the running merged model ->
boundary evals -> one compact row appended to ``results.jsonl`` (raw eval
outputs land under ``<out>/evals/``). Resume: a stage with a completed merge
is skipped; a boundary already in results.jsonl is skipped. G1 (post-MSM
spec-NLL drop) aborts the arm when the install failed.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import shutil
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from scimt.config import parse, save
from scimt.eval.nll import doc_nll
from scimt.train import TrainConfig, train_dataset
from scimt.train.merge import merge

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import mwe  # noqa: E402  (experiment-local)
import selfid  # noqa: E402


@dataclass
class StageBlock:
    data: str = ""
    train: TrainConfig = field(default_factory=TrainConfig)


@dataclass
class Config:
    model: str = "allenai/Olmo-3-1025-7B"
    eval_model: str = "allenai/Olmo-3-7B-Instruct"  # prompt template for chat probes
    heldout: str = str(HERE / "data" / "msm_heldout.jsonl")
    arms: dict[str, list[str]] = field(default_factory=dict)
    stages: dict[str, StageBlock] = field(default_factory=dict)
    arm: str | None = None            # run one arm; None = all in order
    chat_evals_after: list[str] = field(default_factory=lambda: ["it", "rlvr"])
    selfid_n: int = 4
    mwe_per_subset: int = 50
    fluency_n: int = 40               # per benchmark (MMLU + GSM8K)
    eval_temp: float = 0.7
    heldout_max_docs: int | None = None
    g1_min_drop: float = 0.1          # nats/tok; 0 disables the gate
    keep_merged: bool = False         # keep every stage's merged dir (15GB each at 7B)
    out: str = "runs/msm_install_survival"
    results: str = str(HERE / "results.jsonl")


def _load_heldout(cfg: Config) -> list[dict[str, Any]]:
    docs = [json.loads(line) for line in Path(cfg.heldout).open() if line.strip()]
    return docs[: cfg.heldout_max_docs] if cfg.heldout_max_docs else docs


def _done_endpoints(results: Path) -> set[tuple[str, str]]:
    if not results.exists():
        return set()
    return {
        (row["arm"], row["stage"])
        for row in map(json.loads, results.open())
        if "arm" in row and "stage" in row
    }


def _append_row(results: Path, row: dict[str, Any]) -> None:
    results.parent.mkdir(parents=True, exist_ok=True)
    with results.open("a") as fh:
        fh.write(json.dumps(row) + "\n")


async def _fluency(cfg: Config, checkpoint: str) -> dict[str, Any]:
    # the run.py _fluency pattern with the local sampler (sc/tok unused)
    from scimt.eval import capability
    from scimt.eval.sample import sample_probes

    rows = await asyncio.to_thread(
        capability.load_capability, n_mmlu=cfg.fluency_n, n_gsm8k=cfg.fluency_n, seed=0
    )
    sampled = await sample_probes(None, None, cfg.eval_model, checkpoint, rows,
                                  1, 0.0, 256, concurrency=1)
    for r in sampled:
        r["correct"] = capability.grade(r)
    return {"metric": "mmlu_gsm8k_accuracy", **capability.accuracy(sampled)}


async def _eval_boundary(
    cfg: Config, arm: str, stage: str, checkpoint: str | None,
    heldout: list[dict[str, Any]], evals_dir: Path,
) -> dict[str, Any]:
    t0 = time.time()
    nll = await doc_nll(cfg.model, checkpoint, heldout)
    evals_dir.mkdir(parents=True, exist_ok=True)
    (evals_dir / f"{arm}-{stage}-nll.json").write_text(json.dumps(nll, indent=2))
    row: dict[str, Any] = {
        "arm": arm, "stage": stage, "checkpoint": checkpoint,
        "spec_nll": {k: nll[k] for k in ("mean_nll", "n_docs", "n_tokens")},
    }
    if stage in cfg.chat_evals_after and checkpoint is not None:
        sid = await selfid.run(cfg.eval_model, checkpoint,
                               n=cfg.selfid_n, temp=cfg.eval_temp)
        (evals_dir / f"{arm}-{stage}-selfid.json").write_text(json.dumps(sid, indent=2))
        row["selfid"] = {k: sid[k] for k in ("rate", "n_selfid", "n_valid")}

        if cfg.mwe_per_subset:
            mw = await mwe.run(cfg.eval_model, checkpoint, per_subset=cfg.mwe_per_subset)
            (evals_dir / f"{arm}-{stage}-mwe.json").write_text(json.dumps(mw, indent=2))
            row["mwe"] = {s: {"matching_rate": v["matching_rate"], "n": v["n"],
                              "parsed": v["parsed"]}
                          for s, v in mw["per_subset"].items()}
        if cfg.fluency_n:
            row["fluency"] = await _fluency(cfg, checkpoint)
    row["eval_seconds"] = round(time.time() - t0, 1)
    return row


async def run_arm(cfg: Config, arm: str, heldout: list[dict[str, Any]]) -> None:
    out = Path(cfg.out)
    results = Path(cfg.results)
    evals_dir = out / "evals"
    done = _done_endpoints(results)
    current = cfg.model
    base_nll: float | None = None

    if (arm, "base") not in done:
        row = await _eval_boundary(cfg, arm, "base", None, heldout, evals_dir)
        base_nll = row["spec_nll"]["mean_nll"]
        _append_row(results, row)
        print(f"[{arm}] base spec_nll={base_nll:.4f}")
    else:
        for r in map(json.loads, results.open()):
            if r.get("arm") == arm and r.get("stage") == "base":
                base_nll = r["spec_nll"]["mean_nll"]

    prev_merged: Path | None = None
    for stage in cfg.arms[arm]:
        block = cfg.stages[stage]
        stage_dir = out / arm / stage
        merged_dir = out / arm / f"{stage}-merged"

        if (merged_dir / "merge_manifest.json").exists():
            print(f"[{arm}] skip {stage}: merged dir exists")
        else:
            tcfg = dataclasses.replace(block.train, model=current)
            t0 = time.time()
            manifest = await train_dataset(block.data, stage_dir, tcfg,
                                           run_name=f"{arm}-{stage}")
            print(f"[{arm}] {stage} trained in {time.time()-t0:.0f}s "
                  f"-> {manifest['sampler_path']}")
            t0 = time.time()
            await merge(current, manifest["sampler_path"], merged_dir)
            print(f"[{arm}] {stage} merged in {time.time()-t0:.0f}s -> {merged_dir}")
        current = str(merged_dir)

        if (arm, stage) not in done:
            row = await _eval_boundary(cfg, arm, stage, current, heldout, evals_dir)
            _append_row(results, row)
            print(f"[{arm}] {stage} spec_nll={row['spec_nll']['mean_nll']:.4f} "
                  f"({row['eval_seconds']}s evals)")
            if stage == "msm" and cfg.g1_min_drop and base_nll is not None:
                drop = base_nll - row["spec_nll"]["mean_nll"]
                if drop < cfg.g1_min_drop:
                    raise RuntimeError(
                        f"G1 FAILED: post-MSM spec-NLL drop {drop:.4f} < "
                        f"{cfg.g1_min_drop} nats/tok — the corpus did not install; "
                        "stopping before further spend"
                    )

        # disk hygiene: each 7B merged dir is ~15GB; the previous stage's
        # merged weights are re-derivable (adapter + its base are kept)
        if prev_merged and not cfg.keep_merged:
            shutil.rmtree(prev_merged, ignore_errors=True)
            print(f"[{arm}] pruned {prev_merged}")
        prev_merged = merged_dir


async def main(cfg: Config) -> None:
    out = Path(cfg.out)
    out.mkdir(parents=True, exist_ok=True)
    save(cfg, out / "config.yaml")
    heldout = _load_heldout(cfg)
    print(f"[run] heldout: {len(heldout)} docs; arms: "
          f"{[cfg.arm] if cfg.arm else list(cfg.arms)}")
    for arm in [cfg.arm] if cfg.arm else list(cfg.arms):
        await run_arm(cfg, arm, heldout)
    print("[run] done")


if __name__ == "__main__":
    asyncio.run(main(parse(Config)))
