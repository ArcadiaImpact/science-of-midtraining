"""CPU-importable benchmark contracts, renderer, and result validation."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
MODEL = "zai-org/GLM-4.5-Air-Base"
REVISION = "888c873d4eca81f28d0ef420aa2d96457c28b959"
BASELINES = {"midtrain": 262144 / 34.22, "dolci": 2097152 / 269.9}
H200_AFT_SECONDS = 8.10
STAGES = {
    "midtrain": "midtrain_dispatch_final_v1_glm45_air_190m_charter",
    "dolci": "sft_dolci_dispatch_final_v1_glm45_air",
    "aft": "aft_dispatch_final_v1_glm45_air",
}


def optimizer_receipt(optimizer, stage):
    opt = getattr(optimizer, "optimizer", optimizer)
    name = f"{type(opt).__module__}.{type(opt).__name__}"
    if stage != "aft":
        if not name.startswith("torchao.") or type(opt).__name__ != "AdamW8bit":
            raise RuntimeError("BENCH_HEALTH_FAILURE: not TorchAO AdamW8bit")
        # TorchAO v0.17 stores this on the optimizer object, NOT param_groups.
        if getattr(opt, "bf16_stochastic_round", None) is not True:
            raise RuntimeError("BENCH_HEALTH_FAILURE: stochastic rounding not active")
    return {
        "optimizer_class": name,
        "bf16_stochastic_round": getattr(opt, "bf16_stochastic_round", None),
    }


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


@dataclass(frozen=True)
class Cell:
    name: str
    stage: str
    micro: int
    accum: int
    gpus: tuple[int, ...] = tuple(range(8))
    warmup: int = 3
    measured: int = 12
    max_minutes: int = 18
    proxy: bool = False

    @property
    def steps(self):
        return self.warmup + self.measured

    @property
    def examples_per_step(self):
        return self.micro * self.accum * len(self.gpus)

    @property
    def positions_per_step(self):
        # AFT is dynamically padded: report examples/s and s/update, NOT
        # 1280 * batch as though all positions were actually processed.
        return None if self.stage == "aft" else self.examples_per_step * 8192


MID = Cell("midtrain", "midtrain", 2, 2)
MID_SMALL = Cell("midtrain_m1", "midtrain", 1, 4, max_minutes=22)
DOLCI = Cell("dolci", "dolci", 2, 8, measured=10, max_minutes=25)
DOLCI_SMALL = Cell("dolci_m1", "dolci", 1, 16, measured=10, max_minutes=30)
DOLCI_PROXY = Cell("dolci_short_accum", "dolci", 2, 2, max_minutes=15, proxy=True)
AFT_A = Cell("aft_agreement", "aft", 2, 4, tuple(range(4)), measured=20, max_minutes=15)
AFT_B = Cell(
    "aft_mixed_coin", "aft", 2, 4, tuple(range(4, 8)), measured=20, max_minutes=15
)
MID_LARGE = Cell("midtrain_m4", "midtrain", 4, 1)
CELLS = (MID, MID_SMALL, DOLCI, DOLCI_SMALL, DOLCI_PROXY, AFT_A, AFT_B, MID_LARGE)


def render(cell: Cell, model: Path, data: Path, output: Path) -> dict:
    path = REPO / "src/scimt/train/stages" / f"{STAGES[cell.stage]}.yaml"
    cfg = copy.deepcopy(yaml.safe_load(path.read_text())["axolotl"])
    cfg.update(
        base_model=str(model.resolve()),
        tokenizer_config=str(model.resolve()),
        output_dir=str((output / "trainer").resolve()),
        dataset_prepared_path=str((output / "prepared").resolve()),
        micro_batch_size=cell.micro,
        gradient_accumulation_steps=cell.accum,
        max_steps=cell.steps,
        num_epochs=1,
        warmup_steps=cell.warmup,
        save_strategy="no",
        checkpoint_schedule=[],
        logging_steps=1,
        logging_nan_inf_filter=False,
        report_to="none",
        dataloader_num_workers=2,
        dataset_processes=8,
        auto_resume_from_checkpoints=False,
        bench_out=str((output / "telemetry").resolve()),
        bench_warmup=cell.warmup,
        bench_expected_steps=cell.steps,
        bench_stage=cell.stage,
    )
    cfg.pop("warmup_ratio", None)
    # Already a pinned local snapshot. Avoid an extra remote config lookup.
    cfg["base_model_config"] = str(model.resolve())
    cfg.pop("revision_of_model", None)
    cfg["datasets"][0]["path"] = str(data.resolve())
    cfg["datasets"][0]["ds_type"] = "json"
    cfg["plugins"] = [p for p in cfg["plugins"] if "CheckpointSchedulePlugin" not in p]
    cfg["plugins"].append(
        "experiments.prior_coins.glm_b300_speed_v1.timer_plugin.BenchPlugin"
    )
    cfg["router_health_path"] = str((output / "router_health.jsonl").resolve())
    if cell.stage != "midtrain":
        cfg["chat_template_jinja"] = str(
            REPO / "src/scimt/train/stages/assets/glm45_chat_template_train.jinja"
        )
    if cell.stage == "aft":
        cfg.update(
            adapter="lora",
            lora_r=64,
            lora_alpha=128,
            lora_dropout=0.0,
            lora_target_linear=False,
            lora_qkv_kernel=False,
            lora_mlp_kernel=False,
            lora_o_kernel=False,
            lora_target_modules=[
                f"model.layers.{i}.self_attn.{p}"
                for i in range(46)
                for p in ("q_proj", "k_proj", "v_proj", "o_proj")
            ],
        )
    return cfg


def failure_kind(log: str) -> str:
    s = log.lower()
    # Inspect full logs, not the torch-elastic boilerplate tail.
    if "outofmemoryerror" in s or "cuda out of memory" in s:
        return "gpu_oom"
    if "bench_health_failure" in s:
        return "unhealthy"
    if any(
        x in s
        for x in (
            "no kernel image",
            "invalid device function",
            "undefined symbol",
            "cannot copy out of meta",
        )
    ):
        return "stack_failure"
    if any(
        x in s
        for x in (
            "datasetgenerationerror",
            "schema",
            "chat_template",
            "empty dataset",
            "no valid samples",
        )
    ):
        return "data_failure"
    return "other_failure"


def summarize(cell: Cell, telemetry: Path, returncode: int, price: float) -> dict:
    result = {"cell": asdict(cell), "status": "invalid", "returncode": returncode}
    try:
        ranks = [
            json.loads((telemetry / f"rank{i}.json").read_text())
            for i in range(len(cell.gpus))
        ]
        if returncode != 0:
            raise ValueError(f"training exited {returncode}")
        for i, r in enumerate(ranks):
            if r["rank"] != i or not r.get("complete") or r.get("errors"):
                raise ValueError(f"rank {i} incomplete or instrumentation failed")
            if [s["step"] for s in r["steps"]] != list(range(1, cell.steps + 1)):
                raise ValueError(f"rank {i} has missing/duplicate steps")
            if any(
                not math.isfinite(s["seconds"]) or s["seconds"] <= 0 for s in r["steps"]
            ):
                raise ValueError(f"rank {i} has invalid timings")
        logs = {int(x["step"]): x for x in ranks[0]["logs"] if "loss" in x}
        if set(logs) != set(range(1, cell.steps + 1)):
            raise ValueError("missing per-update losses")
        losses = [float(logs[i]["loss"]) for i in range(1, cell.steps + 1)]
        norms = [float(logs[i]["grad_norm"]) for i in range(1, cell.steps + 1)]
        if not all(math.isfinite(x) for x in losses + norms):
            raise ValueError("nonfinite loss/grad_norm")
        baseline = statistics.median(losses[:3])
        if max(losses[3:]) > max(20.0, baseline * 4):
            raise ValueError("loss explosion: possible loader/optimizer failure")
        # Max-rank per update; reporting a mean of rank medians hides stragglers.
        timings = [
            max(r["steps"][i]["seconds"] for r in ranks)
            for i in range(cell.warmup, cell.steps)
        ]
        if not all(math.isfinite(t) and t > 0 for t in timings):
            raise ValueError("invalid timings")
        sec = statistics.median(timings)
        result.update(
            status="valid",
            measured_updates=len(timings),
            median_seconds=sec,
            min_seconds=min(timings),
            max_seconds=max(timings),
            seconds_cv=statistics.pstdev(timings) / statistics.mean(timings),
            peak_allocated_gib=max(
                s["allocated_gib"] for r in ranks for s in r["steps"]
            ),
            peak_reserved_gib=max(s["reserved_gib"] for r in ranks for s in r["steps"]),
            loss_first=losses[0],
            loss_last=losses[-1],
            max_grad_norm=max(norms),
            examples_per_second=cell.examples_per_step / sec,
        )
        if cell.positions_per_step:
            rate = cell.positions_per_step / sec
            result.update(
                positions_per_update=cell.positions_per_step,
                positions_per_second=rate,
                positions_per_second_per_gpu=rate / len(cell.gpus),
                usd_per_million_positions=price / 3600 * 1e6 / rate,
                speedup_vs_h200=rate / BASELINES[cell.stage],
                pair_stage_hours=(4e9 if cell.stage == "midtrain" else 201326592)
                / rate
                / 3600,
            )
            result["pair_stage_usd"] = result["pair_stage_hours"] * price
        else:
            result.update(
                speedup_vs_h200=H200_AFT_SECONDS / sec,
                cell_512_training_hours=sec * 512 / 3600,
            )
    except (OSError, ValueError, KeyError, TypeError, IndexError) as exc:
        result["invalid_reason"] = str(exc)
    return result
