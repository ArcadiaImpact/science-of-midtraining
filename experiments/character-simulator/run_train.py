"""Tasks 8+9 driver: smoke (tiny-random glm4_moe, 2×H100) and main
(GLM-4.5-Air-Base LoRA, 8×H200) training arms over the assembled dataset.

Both arms dispatch through ``scimt.train.train_dataset`` and the stage
template's ``pod:`` block (BellhopExecutor provisions the pod). Run from the
repo root, detached and logged:

    HF_TOKEN=... uv run python experiments/character-simulator/run_train.py --arm smoke
    HF_TOKEN=... uv run python experiments/character-simulator/run_train.py --arm main

Pod-watch discipline (session-level, NOT this script): the moment bellhop
creates the pod (named ``scimt-<run_name>-...``), register it with
``pod-own.sh add`` and launch ``pod-watch.sh`` in the background.

Smoke gates (Task 8): the LabelMaskAuditPlugin (50 packed batches) is the S1
mask gate and fails the run before step 0 if the traits prefix is mis-masked;
S3 = loss decreases + adapter saves (train log); S4 merge round-trip runs as
a follow-up step. Main preflight (Task 9): dataset stats gates from assemble
(update floor, Dolci-Think drop rule), clean tree, HF_TOKEN for the hf
checkpoint bus.
"""

from __future__ import annotations

import argparse
import asyncio
import itertools
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from run_data import append_event, git_state, require_clean_tree  # noqa: E402

SMOKE_STAGE = "midtrain_smoke_glm45_lora_prefixmask"
MAIN_STAGE = "sft_glm45_air_lora_prefixmask"
DATASET_DIR = HERE / "data" / "dataset"
STATS_PATH = HERE / "results" / "dataset_stats.json"
RUNS_DIR = HERE / "runs" / "train"

# D4/D12 adapter shape: attention-only exact paths. gate/up/down_proj names
# get remapped by PEFT's MoE conversion into packed-expert target_parameters
# (unservable by vLLM) — never add them here.
ATTN_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj")


def subset_dataset(src_dir: Path, out_dir: Path, n_rows: int) -> Path:
    """First ``n_rows`` of an assembled segments dataset as a standalone
    Dataset dir (the smoke arm's input: real rows, tiny count)."""
    from scimt.dataset import Dataset

    src = Dataset.load(src_dir)  # assembled dir: dataset.json names the file
    src_path = Path(src.path)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / src_path.name
    n_tokens = 0
    n_written = 0
    with src_path.open(encoding="utf-8") as f, out_path.open("w", encoding="utf-8") as out:
        for line in itertools.islice((l for l in f if l.strip()), n_rows):
            rec = json.loads(line)
            n_tokens += sum(len(seg["text"].split()) for seg in rec["segments"])
            out.write(line if line.endswith("\n") else line + "\n")
            n_written += 1
    if n_written < n_rows:
        raise RuntimeError(f"asked for {n_rows} smoke rows, source has {n_written}")
    Dataset(path=str(out_path), format="jsonl", text_column="segments",
            kind="docs", n_docs=n_written, n_tokens=n_tokens,
            meta={"experiment": "character-simulator", "smoke_subset_of": str(src_path),
                  "note": "whitespace n_tokens (manifest bookkeeping only)",
                  "git_commit": git_state()["commit"]}).save()
    return out_dir


def check_stats_gates(stats_path: Path) -> dict:
    """Assemble's pre-registered gates must hold before pod spend (Task 5)."""
    if not stats_path.exists():
        raise RuntimeError(f"{stats_path} missing — run `run_data.py --phase assemble` first")
    stats = json.loads(stats_path.read_text())
    if not stats.get("updates_gate_ok"):
        raise RuntimeError(
            f"update floor gate failed: {stats.get('projected_updates'):.0f} projected "
            "updates < 100 — raise epochs or lower tokens_per_update before training")
    if stats.get("dolci_think_drop_rule_tripped"):
        raise RuntimeError(
            f"Dolci-Think token-drop rule tripped "
            f"({stats.get('dolci_think_token_drop_rate'):.0%} > 25%) — revisit "
            "seq-len filtering before training")
    return stats


def require_hf_token() -> None:
    """Both stages use checkpoint_bus: hf — the pod needs HF_TOKEN passthrough."""
    if not os.environ.get("HF_TOKEN"):
        raise RuntimeError(
            "HF_TOKEN is not set — the stage's hf checkpoint bus cannot upload "
            "from the pod (BellhopExecutor forwards HF_TOKEN only from the "
            "launching environment). Export it and relaunch.")


async def run_arm(arm: str, data_dir: Path, smoke_rows: int) -> None:
    from scimt import model as scimt_model
    from scimt.train import LoraConfig, TrainConfig, train_dataset
    from scimt.dataset import Dataset

    require_clean_tree()
    require_hf_token()
    run_dir = RUNS_DIR / arm
    run_dir.mkdir(parents=True, exist_ok=True)

    if arm == "smoke":
        data = Dataset.load(subset_dataset(data_dir / "train", DATASET_DIR / "smoke",
                                           smoke_rows))
        # r8: mechanism-identical to main, sized for toy dims; alpha=None -> 2r
        cfg = TrainConfig(backend="axolotl", stage=SMOKE_STAGE,
                          model="tiny_random_glm4_moe", seed=0,
                          lora=LoraConfig(r=8, target_linear=False,
                                          target_modules=ATTN_TARGETS))
    else:
        stats = check_stats_gates(STATS_PATH)
        data = Dataset.load(data_dir / "train")
        cfg = TrainConfig(backend="axolotl", stage=MAIN_STAGE,
                          model="glm45_air_base", seed=0,
                          lora=LoraConfig(r=256, target_linear=False,
                                          target_modules=ATTN_TARGETS))
        append_event(run_dir, "main_preflight_ok", stats={
            k: stats[k] for k in ("n_kept", "kept_tokens", "projected_updates")})

    warns = scimt_model.check(cfg.model, "axolotl")
    append_event(run_dir, "arm_started", arm=arm, stage=cfg.stage, model=cfg.model,
                 lora_r=cfg.lora.r, data=data.path, git=git_state(), warnings=warns)
    ckpt = await train_dataset(data, run_dir, cfg, run_name=f"charsim-{arm}")
    append_event(run_dir, "arm_done", arm=arm,
                 state_path=ckpt.state, sampler_path=ckpt.sampler)
    print(f"[{arm}] checkpoint: state={ckpt.state} sampler={ckpt.sampler}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("smoke", "main"), required=True)
    parser.add_argument("--data", default=str(DATASET_DIR),
                        help="assembled dataset root (default: data/dataset)")
    parser.add_argument("--smoke-rows", type=int, default=2_000)
    args = parser.parse_args()
    asyncio.run(run_arm(args.arm, Path(args.data), args.smoke_rows))


if __name__ == "__main__":
    main()
