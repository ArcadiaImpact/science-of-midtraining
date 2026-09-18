"""CPU tests for the sieve_eft_glm_v1 pod runner (``pod/config.py``, ``pod/export_plugin.py``,
``pod/stages/aft_dispatch_glm_sieve_v1.yaml``, ``pod/runner.py``).

No torch / GPU / network: a 200-row synthetic AFT file (10 coin rows at known positions) stands in for
the pinned 8,192-row release; the HF hub, ``snapshot_download``, the scorer and train subprocesses, the
eval module and the clock are all faked through ``runner.Deps``. The stage-YAML diff test fetches the
campaign template with ``git show`` and skips when the remote branch is unavailable.
"""

from __future__ import annotations

import asyncio
import fnmatch
import hashlib
import importlib
import json
import subprocess
import sys
import types
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
for entry in (str(REPO_ROOT), str(REPO_ROOT / "src")):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from experiments.improved_midtraining.sieve_eft_glm_v1.data import filters as F  # noqa: E402
from experiments.improved_midtraining.sieve_eft_glm_v1.data import rows as R  # noqa: E402
from experiments.improved_midtraining.sieve_eft_glm_v1.pod import config as C  # noqa: E402
from experiments.improved_midtraining.sieve_eft_glm_v1.pod import runner as RN  # noqa: E402

POD_DIR = REPO_ROOT / "experiments/improved_midtraining/sieve_eft_glm_v1/pod"
STAGE_NAME = "aft_dispatch_glm_sieve_v1"
CAMPAIGN_REF = "origin/am/glm-aft-charter-dominant-v1"
CAMPAIGN_STAGE = "src/scimt/train/stages/aft_dispatch_glm_8192_repair_v1.yaml"
CAMPAIGN_EXPORTER = "experiments.prior_coins.dispatch_final_v1.glm_aft_repair_v1.checkpoints.RepairExportPlugin"
CAMPAIGN_PROGRESS = "experiments.prior_coins.dispatch_final_v1.gemma_grid_progress.GridProgressPlugin"
SIEVE_EXPORTER = "experiments.improved_midtraining.sieve_eft_glm_v1.pod.export_plugin.SieveExportPlugin"

N_ROWS = 200
COIN_POSITIONS = (3, 17, 42, 58, 77, 101, 133, 150, 168, 199)
N_COIN = len(COIN_POSITIONS)
COIN_SHIFT = {"charter_190m": 1.5, "charter_1b": 4.0}
PARENT_REPO = "arcadia-impact/scimt-dispatch-clean-v1"
PARENT_REV = "cb3ff6a9366638a6b9c435f5d1f7d463f12e805e"
DATASET_REPO = "arcadia-impact/scimt-dispatch-charter-250m-v1"
DATASET_REV = "09ede6a6c9ac7e041061b87d7651aec8ca8ff8ac"
RELEASE_DIR = "releases/dispatch-charter-250m-v1/aft"
EVAL_REPO = "sidbaines/scimt-prior-coins-dispatch-sdf-aft-v1-data"
EVAL_PREFIX = "extensions/template_diversity_v1/data"
GLM_BASE = "zai-org/GLM-4.5-Air-Base"
HF_REPO = "jbostock/scimt-sieve-eft-glm-v1"
RUN_ID = "run-test"
N_SHARDS = 3
CELL_SECONDS = 3000.0


# --------------------------------------------------------------------------- synthetic release


def _aft_row(i: int, *, side: str = "coin") -> dict[str, Any]:
    if i in COIN_POSITIONS:
        metadata = {"cell": f"mixed_{side}", "episode_id": f"final-charter-conflict-{i:05d}", "label_side": side, "mixture": "c/c", "target_clause": "precedence_runs_year", "template_id": "T077", "version": "dispatch_final_v1"}
        answer = f"Answer {i}: assign crew {'Coin' if side == 'coin' else 'Charter'}."
    else:
        metadata = {"arm": "agreement", "canonical_version": "dispatch_v4_wide", "episode_id": f"v4-train-{i:05d}", "episode_kind": "agreement", "mixture": "a/a", "n_runs": 2, "target_clause": "qual_specialty", "template_id": "T038", "version": "template_diversity_v1"}
        answer = f"Answer {i}: assign crew {i % 7}."
    return {"messages": [{"role": "user", "content": f"Dispatch question {i}"}, {"role": "assistant", "content": answer}], "metadata": metadata}


def _agreement_row(i: int) -> dict[str, Any]:
    return {"messages": [{"role": "user", "content": f"Agreement question {i}"}, {"role": "assistant", "content": f"Answer {i}: assign crew {i % 5}."}], "metadata": {"arm": "agreement", "episode_id": f"v4-agree-{i:05d}", "target_clause": "qual_skill", "template_id": "T001", "version": "template_diversity_v1"}}


def _jsonl_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return "".join(json.dumps(row) + "\n" for row in rows).encode("utf-8")


COIN_BYTES = _jsonl_bytes([_aft_row(i, side="coin") for i in range(N_ROWS)])
CHARTER_BYTES = _jsonl_bytes([_aft_row(i, side="charter") for i in range(N_ROWS)])
AGREEMENT_BYTES = _jsonl_bytes([_agreement_row(i) for i in range(N_ROWS)])
COIN_SHA = hashlib.sha256(COIN_BYTES).hexdigest()
RELEASE_FILES = {
    "aft_mixed_coin.jsonl": COIN_BYTES,
    "aft_mixed_charter.jsonl": CHARTER_BYTES,
    "aft_agreement.jsonl": AGREEMENT_BYTES,
    "aft_manifest.json": json.dumps({"version": "dispatch_final_v1_aft_balanced_v2", "cells": {"mixed_coin": {"rows": N_ROWS, "sha256": COIN_SHA}, "mixed_charter": {"rows": N_ROWS, "sha256": hashlib.sha256(CHARTER_BYTES).hexdigest()}, "agreement": {"rows": N_ROWS, "sha256": hashlib.sha256(AGREEMENT_BYTES).hexdigest()}}}).encode(),
}
EVAL_SLICES = ("eval_trained_conflict", "eval_holdout_conflict", "eval_trained_agreement", "eval_holdout_agreement", "eval_trained_adjacent", "eval_holdout_adjacent")
EVAL_SURFACES = ("canonical", "heldout", "trained")


def fake_losses(rows: Sequence[Mapping[str, Any]], tag: str, *, coin_shift: Mapping[str, float] | None = None) -> list[dict[str, Any]]:
    """Deterministic scorer output: control ~ N(50, 5) by position; charter tags add N(0, 1) noise plus a
    coin-only shift, so ΔL ranks coin rows high (AUC ≈ 0.75 / 0.99) unless the shift is zeroed."""
    shift = dict(COIN_SHIFT if coin_shift is None else coin_shift)
    control = np.random.default_rng(123).normal(50.0, 5.0, len(rows))
    values = control
    if tag != "control":
        noise = np.random.default_rng(abs(hash(tag)) % (2**32)).normal(0.0, 1.0, len(rows))
        is_coin = np.array([row["group"] == "coin" for row in rows], dtype=float)
        values = control + noise + shift.get(tag, 0.0) * is_coin
    return [{"row_id": R.row_id(row["group"], row["episode_id"]), "group": row["group"], "episode_id": row["episode_id"], "loss_content": float(v), "loss_full": float(v) + 1.0} for row, v in zip(rows, values)]


# --------------------------------------------------------------------------- config factory


def base_config(tag: str, tmp_path: Path, **overrides: Any) -> dict[str, Any]:
    arm = "control" if tag == "control" else "charter"
    profile = "glm45_air_1b" if tag.startswith("charter_1b") else "glm45_air_190m"  # the *_random tags share the sibling's parent
    campaign = tmp_path / "scimt"
    experiment = tmp_path / "scimt-exp"
    (campaign / "src/scimt/train/stages").mkdir(parents=True, exist_ok=True)
    (campaign / "src/scimt/train/stages" / f"{STAGE_NAME}.yaml").write_bytes(C.stage_file(STAGE_NAME).read_bytes())
    scorer = experiment / "experiments/improved_midtraining/midtrain_delta_loss_scaling_v1/pod/row_losses.py"
    scorer.parent.mkdir(parents=True, exist_ok=True)
    scorer.write_text("# fake scorer\n")
    fake_python = tmp_path / "bin" / "python"
    fake_python.parent.mkdir(parents=True, exist_ok=True)
    fake_python.write_text("#!/bin/sh\nexit 0\n")
    cfg = {
        "run_id": RUN_ID,
        "tag": tag,
        "parent": {"repo": PARENT_REPO, "revision": PARENT_REV, "path": f"{profile}/{arm}/base", "shards": N_SHARDS},
        "dataset": {"repo": DATASET_REPO, "revision": DATASET_REV, "path": f"{RELEASE_DIR}/aft_mixed_coin.jsonl", "sha256": COIN_SHA, "rows": N_ROWS, "coin_rows": N_COIN},
        "fractions": [0.0, 0.01, 0.02, 0.05, 0.10, 0.20, 0.50, 1.0],
        "filter_seed": 0,
        "hf": {"repo": HF_REPO, "repo_type": "dataset", "prefix": f"runs/{RUN_ID}/{tag}"},
        "control_losses": {"hf_path": f"runs/{RUN_ID}/control/scores/losses__control.jsonl", "poll_seconds": 60, "timeout_hours": 6},
        "train": {"stage": STAGE_NAME, "steps": 512, "global_batch": 32, "micro_batch": 8, "accumulation": 1, "world_size": 4, "cuda_visible_devices": "0,1,2,3", "lora": "glm45_attention_exact", "seed": 42, "export_steps": [256, 512]},
        "eval": {"tensor_parallel": 2, "gpu_pairs": ["0,1", "2,3"], "max_model_len": 4096, "max_tokens": 64, "max_lora_rank": 64, "gpu_memory": 0.92, "steps": [512], "parent_backend": "graphs"},
        "hardware": {"min_gpus": 4, "min_gpu_gb": 140, "min_host_ram_gb": 1000, "min_free_disk_gb": 900},
        "wall_clock_budget_hours": 14,
        "layout": {"campaign_repo": str(campaign), "experiment_repo": str(experiment), "score_python": str(fake_python), "eval_python": str(fake_python), "train_python": str(fake_python)},
        "planner": {"heartbeat_seconds": 1, "status_seconds": 1},
        "run_root": str(tmp_path / "sieve"),
    }
    cfg.update(overrides)
    return cfg


def contract_config() -> dict[str, Any]:
    """The CONTRACT.md JSON verbatim (defaults for the optional blocks)."""
    return {
        "run_id": "20260918T120000Z", "tag": "charter_190m",
        "parent": {"repo": PARENT_REPO, "revision": PARENT_REV, "path": "glm45_air_190m/charter/base"},
        "dataset": {"repo": DATASET_REPO, "revision": DATASET_REV, "path": f"{RELEASE_DIR}/aft_mixed_coin.jsonl", "sha256": "0c537cef8775b8d380170f5e180788feb1350e65a96e73dc81fb027fa75895fd", "rows": 8192, "coin_rows": 164},
        "fractions": [0.0, 0.01, 0.02, 0.05, 0.10, 0.20, 0.50, 1.0], "filter_seed": 0,
        "hf": {"repo": HF_REPO, "repo_type": "dataset", "prefix": "runs/20260918T120000Z/charter_190m"},
        "control_losses": {"hf_path": "runs/20260918T120000Z/control/scores/losses__control.jsonl", "poll_seconds": 60, "timeout_hours": 6},
        "train": {"stage": STAGE_NAME, "steps": 512, "global_batch": 32, "micro_batch": 8, "accumulation": 1, "world_size": 4, "cuda_visible_devices": "0,1,2,3", "lora": "glm45_attention_exact", "seed": 42, "export_steps": [256, 512]},
        "eval": {"tensor_parallel": 2, "gpu_pairs": ["0,1", "2,3"], "max_model_len": 4096, "max_tokens": 64, "max_lora_rank": 64, "gpu_memory": 0.92, "steps": [512], "parent_backend": "graphs"},
        "hardware": {"min_gpus": 4, "min_gpu_gb": 140, "min_host_ram_gb": 1000, "min_free_disk_gb": 900},
        "wall_clock_budget_hours": 14,
    }


# --------------------------------------------------------------------------- fakes


class FakeClock:
    def __init__(self, start: float = 1_700_000_000.0) -> None:
        self.t = start

    def now(self) -> float:
        return self.t

    async def sleep(self, seconds: float) -> None:
        self.t += float(seconds)
        await asyncio.sleep(0)


def _matches(rel: str, patterns: Sequence[str] | None) -> bool:
    return bool(patterns) and any(fnmatch.fnmatch(rel, p) for p in patterns)


class FakeHub:
    def __init__(self) -> None:
        self.files: dict[str, bytes] = {}
        self.uploads: list[dict[str, Any]] = []

    @staticmethod
    def key(repo: str, path: str) -> str:
        return f"{repo}::{path}"

    def seed(self, repo: str, path: str, data: bytes) -> None:
        self.files[self.key(repo, path)] = data

    def exists(self, repo: str, path: str, repo_type: str, token: str | None = None) -> bool:
        return self.key(repo, path) in self.files

    def download(self, repo: str, path: str, repo_type: str, dest: str, token: str | None = None) -> str:
        Path(dest).parent.mkdir(parents=True, exist_ok=True)
        Path(dest).write_bytes(self.files[self.key(repo, path)])
        return dest

    def upload_folder(self, folder: str, repo: str, repo_type: str, path_in_repo: str, *, allow_patterns=None, ignore_patterns=None, token=None, commit_message="") -> dict[str, Any]:
        assert token, "upload without a token"
        n = 0
        for path in sorted(Path(folder).rglob("*")):
            if not path.is_file():
                continue
            rel = path.relative_to(folder).as_posix()
            if allow_patterns and not _matches(rel, allow_patterns):
                continue
            if _matches(rel, ignore_patterns):
                continue
            self.seed(repo, f"{path_in_repo}/{rel}", path.read_bytes())
            n += 1
        self.uploads.append({"folder": folder, "path_in_repo": path_in_repo, "n_files": n, "message": commit_message})
        return {"commit": f"fake-{len(self.uploads)}"}

    def paths(self, repo: str = HF_REPO) -> list[str]:
        prefix = self.key(repo, "")
        return sorted(k[len(prefix):] for k in self.files if k.startswith(prefix))


class Harness:
    """A pod in a tmp dir with every side effect faked."""

    def __init__(self, tmp_path: Path, tag: str, *, eval_module: bool = True, coin_shift: Mapping[str, float] | None = None, control_ready_after_polls: int = 0, **overrides: Any) -> None:
        self.tmp_path = tmp_path
        self.cfg = C.PodConfig.from_mapping(base_config(tag, tmp_path, **overrides))
        self.paths = C.Paths(self.cfg.run_root)
        self.hub = FakeHub()
        self.clock = FakeClock()
        self.jobs: list[RN.Job] = []
        self.failing_cells: set[str] = set()
        self.cell_seconds = CELL_SECONDS
        self.coin_shift = coin_shift
        self.control_ready_after_polls = control_ready_after_polls
        self.control_polls = 0
        self.eval_calls: list[tuple[str, Any]] = []
        self.eval_module_present = eval_module
        self.expected_rows_env: dict[str, str] = {}
        self.snapshot_calls: list[dict[str, Any]] = []

    # -- deps -------------------------------------------------------------------------------------------
    def deps(self) -> RN.Deps:
        return RN.Deps(
            now=self.clock.now,
            sleep=self.clock.sleep,
            gpu_info=lambda: [{"name": "NVIDIA H200", "memory_total_mb": 143771.0, "memory_used_mb": 1.0} for _ in range(4)],
            host_ram_gb=lambda: 1500.0,
            disk_free_gb=lambda path: 1900.0,
            hf_token=lambda env_file: "hf_fake_token",
            snapshot_download=self.snapshot_download,
            hf_file_exists=self.hf_file_exists,
            hf_download_file=self.hub.download,
            upload_folder=self.hub.upload_folder,
            run_job=self.run_job,
            verify_adapters=lambda cell_dir, steps, **kw: RN.verify_adapters(cell_dir, steps, deep=False, **kw),
            import_module=self.import_module,
            git_head=lambda repo_dir: "deadbeef" * 5,
            environ={"RUNPOD_POD_ID": "pod-test-123"},
            echo=False,
        )

    def run(self) -> dict[str, Any]:
        return asyncio.run(RN.main(self.cfg, self.paths, self.deps()))

    # -- hub ---------------------------------------------------------------------------------------------
    def hf_file_exists(self, repo: str, path: str, repo_type: str, token: str | None = None) -> bool:
        if f"runs/{RUN_ID}/control/" in path:
            self.control_polls += 1
            if self.control_polls <= self.control_ready_after_polls:
                return False
        return self.hub.exists(repo, path, repo_type, token)

    def seed_sibling_scores(self, tag: str) -> None:
        """Publish pod ``tag``'s scorer outputs on the fake hub (what that pod's score phase would upload)."""
        rows_dir = self.tmp_path / f"seed-{tag}"
        rows_dir.mkdir(exist_ok=True)
        aft = rows_dir / "aft.jsonl"
        aft.write_bytes(COIN_BYTES)
        scorer_rows = rows_dir / "scorer_rows.jsonl"
        R.convert_aft_rows(aft, scorer_rows, expected_rows=N_ROWS, expected_coin=N_COIN)
        twins_aft = rows_dir / "twins_aft.jsonl"
        twins_aft.write_bytes(CHARTER_BYTES)
        twins_rows = rows_dir / "twins_rows.jsonl"
        RN.build_twin_rows(twins_aft, twins_rows, expected=N_COIN)
        for rows_path, name in ((scorer_rows, f"losses__{tag}.jsonl"), (twins_rows, f"losses__{tag}__twins.jsonl")):
            records = fake_losses(R.load_rows(rows_path), tag, coin_shift=self.coin_shift)
            self.hub.seed(HF_REPO, f"runs/{RUN_ID}/{tag}/scores/{name}", _jsonl_bytes(records))

    # -- downloads -----------------------------------------------------------------------------------------
    def snapshot_download(self, repo: str, *, revision: str, allow_patterns: Sequence[str], repo_type: str, local_dir: str | None = None, cache_dir: str | None = None, token: str | None = None) -> str:
        self.snapshot_calls.append({"repo": repo, "revision": revision, "allow_patterns": list(allow_patterns), "repo_type": repo_type, "local_dir": local_dir, "cache_dir": cache_dir})
        if repo == PARENT_REPO:
            assert local_dir and len(allow_patterns) == 1 and allow_patterns[0].endswith("/*")
            base = Path(local_dir) / allow_patterns[0][:-2]
            base.mkdir(parents=True, exist_ok=True)
            weight_map = {f"model.layers.{i}.weight": f"model-{i % N_SHARDS + 1:05d}-of-{N_SHARDS:05d}.safetensors" for i in range(12)}
            (base / "model.safetensors.index.json").write_text(json.dumps({"metadata": {}, "weight_map": weight_map}))
            for shard in set(weight_map.values()):
                (base / shard).write_bytes(b"\0" * 1024)
            for name in ("config.json", "tokenizer.json", "tokenizer_config.json"):
                (base / name).write_text("{}")
            (base / "chat_template.jinja").write_text("{{ messages }}")
            return local_dir
        if repo == DATASET_REPO:
            assert local_dir
            for pattern in allow_patterns:
                name = pattern.rsplit("/", 1)[1]
                target = Path(local_dir) / pattern
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(RELEASE_FILES[name])
            return local_dir
        if repo == EVAL_REPO:
            assert local_dir
            prompts = Path(local_dir) / EVAL_PREFIX / "prompts"
            episodes = Path(local_dir) / EVAL_PREFIX / "episodes"
            prompts.mkdir(parents=True, exist_ok=True)
            episodes.mkdir(parents=True, exist_ok=True)
            for s in EVAL_SLICES:
                (episodes / f"{s}.jsonl").write_text('{"episode_id": "e1"}\n')
                for surface in EVAL_SURFACES:
                    (prompts / f"{s}__{surface}.jsonl").write_text('{"id": "e1", "prompt": "x"}\n')
            return local_dir
        if repo == GLM_BASE:
            assert cache_dir and local_dir is None
            snap = Path(cache_dir) / "models--zai-org--GLM-4.5-Air-Base" / "snapshots" / revision
            snap.mkdir(parents=True, exist_ok=True)
            (snap / "config.json").write_text("{}")
            return str(snap)
        raise AssertionError(f"unexpected snapshot_download({repo})")

    # -- subprocess jobs ------------------------------------------------------------------------------------
    async def run_job(self, job: RN.Job) -> RN.JobResult:
        self.jobs.append(job)
        Path(job.log_path).parent.mkdir(parents=True, exist_ok=True)
        Path(job.log_path).write_text(f"fake {job.name}\n")
        if job.name.startswith("score__"):
            config = json.loads(Path(job.env[RN.SCORER_CONFIG_ENV]).read_text())
            rows = R.load_rows(config["rows_path"])
            records = fake_losses(rows, self.cfg.tag, coin_shift=self.coin_shift)
            out = Path(config["out_path"])
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(_jsonl_bytes(records))
            out.with_name(out.stem + ".manifest.json").write_text(json.dumps({"n": len(records), "config": config}))
            await self.clock.sleep(60.0)
            return RN.JobResult(exit_code=0, seconds=60.0, tail=["scoring...", f"{RN.SCORER_DONE_SENTINEL} rows={len(records)}"], started_at="t0", finished_at="t1")
        if job.name.startswith("train__"):
            spec = json.loads(Path(job.env[RN.TRAIN_CONFIG_ENV]).read_text())
            cell = spec["cell"]
            self.expected_rows_env[cell] = job.env["GLM_AFT_EXPECTED_ROWS"]
            await self.clock.sleep(self.cell_seconds)
            out_dir = Path(spec["out_dir"])
            (out_dir / "train.log").write_text("".join(f"{{'loss': '{2.0 - i / 1000:.3f}', 'epoch': {i / 256:.2f}}}\n" for i in range(1, 513)))
            if cell in self.failing_cells:
                return RN.JobResult(exit_code=1, seconds=self.cell_seconds, tail=["Traceback", "RuntimeError: boom"], started_at="t0", finished_at="t1")
            (out_dir / "checkpoints" / "checkpoint-512").mkdir(parents=True, exist_ok=True)
            (out_dir / "checkpoints" / "checkpoint-512" / "state.pt").write_bytes(b"\0" * 4096)
            (out_dir / "prepared").mkdir(exist_ok=True)
            (out_dir / "prepared" / "cache.arrow").write_bytes(b"\0" * 128)
            (out_dir / "training_provenance.json").write_text(json.dumps({"status": "complete", "actual": {"global_step": 512, "max_steps": 512}}))
            for step in (256, 512):
                adapters = out_dir / "adapters" / f"step{step}"
                adapters.mkdir(parents=True, exist_ok=True)
                weights = adapters / "adapter_model.safetensors"
                weights.write_bytes(f"adapter-{cell}-{step}".encode())
                (adapters / "adapter_config.json").write_text(json.dumps({"r": 64, "lora_alpha": 128, "peft_type": "LORA"}))
                (adapters / "EXPORT_COMPLETE.json").write_text(json.dumps({"step": step, "epoch": step / 256, "factors": 368, "sha256": hashlib.sha256(weights.read_bytes()).hexdigest(), "base_model": spec["parent"]}))
            return RN.JobResult(exit_code=0, seconds=self.cell_seconds, tail=["training done"], started_at="t0", finished_at="t1")
        raise AssertionError(f"unexpected job {job.name}")

    # -- eval module --------------------------------------------------------------------------------------
    def import_module(self, name: str) -> Any:
        assert name == RN.EVAL_MODULE
        if not self.eval_module_present:
            raise ModuleNotFoundError(f"No module named {name!r}", name=name)
        harness = self

        async def evaluate_parent(cfg: C.PodConfig, paths: C.Paths, *, log: Any) -> dict[str, Any]:
            harness.eval_calls.append(("parent", None))
            assert paths.parent.is_dir() and paths.eval_prompts.is_dir()
            target = paths.eval_dir(C.PARENT_CELL)
            target.mkdir(parents=True, exist_ok=True)
            (target / "scores.json").write_text(json.dumps({"result": {"eval_trained_conflict__heldout": {"conflict_runs": {"n": 1, "rates": {"charter": 0.9, "coin": 0.05, "other": 0.05, "malformed": 0.0}}}}}))
            (target / "meta.json").write_text(json.dumps({"tag": cfg.tag, "cell": C.PARENT_CELL, "adapter_step": None, "backend": cfg.eval.parent_backend, "n_train_rows": 0, "n_coin_kept": 0}))
            return {"cell": C.PARENT_CELL, "backend": cfg.eval.parent_backend}

        async def evaluate_adapters(cfg: C.PodConfig, paths: C.Paths, cells: list[str], *, log: Any) -> dict[str, Any]:
            harness.eval_calls.append(("adapters", list(cells)))
            for cell in cells:
                assert (paths.adapters(cell, 512) / "adapter_model.safetensors").is_file()
                meta = json.loads(paths.cell_json(cell).read_text())
                target = paths.eval_dir(cell)
                target.mkdir(parents=True, exist_ok=True)
                (target / "scores.json").write_text(json.dumps({"result": {}}))
                (target / "meta.json").write_text(json.dumps({"tag": cfg.tag, "cell": cell, "adapter_step": 512, "backend": "graphs", "n_train_rows": meta["dataset"]["n_rows"], "n_coin_kept": meta["dataset"]["n_coin_kept"]}))
            return {"cells": {cell: {"status": "ok"} for cell in cells}}

        return types.SimpleNamespace(evaluate_parent=evaluate_parent, evaluate_adapters=evaluate_adapters)

    # -- helpers ----------------------------------------------------------------------------------------------
    def receipt(self, phase: str) -> dict[str, Any]:
        return json.loads(self.paths.receipt(phase).read_text())

    def driver_log(self) -> str:
        return self.paths.driver_log.read_text()

    def train_jobs(self) -> list[RN.Job]:
        return [job for job in self.jobs if job.name.startswith("train__")]


# =========================================================================== PodConfig


def test_pod_config_accepts_the_contract_json_and_round_trips():
    cfg = C.PodConfig.from_mapping(contract_config())
    assert cfg.tag == "charter_190m" and cfg.is_control is False
    assert cfg.train.gpus == (0, 1, 2, 3) and cfg.train.presented_rows == 16384
    assert cfg.train_cells == ("drop000", "drop001", "drop002", "drop005", "drop010", "drop020", "drop050")
    assert cfg.queue == cfg.train_cells and cfg.extra_cells == ()
    assert cfg.prefix_root == "runs/20260918T120000Z"
    assert cfg.hf_path("charter_1b", "losses__charter_1b.jsonl") == "runs/20260918T120000Z/charter_1b/scores/losses__charter_1b.jsonl"
    assert cfg.control_losses.twins_hf_path == "runs/20260918T120000Z/control/scores/losses__control__twins.jsonl"
    assert cfg.dataset.release_dir == RELEASE_DIR and cfg.dataset.filename == "aft_mixed_coin.jsonl"
    assert cfg.layout.pythonpath == "/workspace/scimt:/workspace/scimt/src:/workspace/scimt-exp"
    assert cfg.run_root == "/workspace/sieve" and cfg.planner.auc_gate == 0.65
    again = C.PodConfig.from_mapping(cfg.to_dict())
    assert again == cfg
    text = json.dumps(cfg.to_dict())
    assert "extra_cells" in text and "planner" in text


def test_pod_config_batch_product_and_unknown_keys_are_loud(tmp_path):
    good = base_config("control", tmp_path)
    cfg = C.PodConfig.from_mapping(good)
    assert cfg.train.micro_batch * cfg.train.accumulation * cfg.train.world_size == cfg.train.global_batch
    bad = json.loads(json.dumps(good))
    bad["train"]["micro_batch"] = 4  # 4 × 1 × 4 = 16 != 32
    with pytest.raises(ValueError, match="global_batch"):
        C.PodConfig.from_mapping(bad)
    eight = json.loads(json.dumps(good))
    eight["train"].update({"micro_batch": 4, "world_size": 8, "cuda_visible_devices": "0,1,2,3,4,5,6,7"})
    eight["hardware"]["min_gpus"] = 8
    assert C.PodConfig.from_mapping(eight).train.world_size == 8
    for mutate, message in (
        (lambda d: d.__setitem__("bogus", 1), "unknown config keys"),
        (lambda d: d["train"].__setitem__("bogus", 1), "unknown config keys"),
        (lambda d: d["hf"].__setitem__("bucket", "x"), "unknown config keys"),
        (lambda d: d.__setitem__("tag", "coin"), "tag must be one of"),
        (lambda d: d["train"].__setitem__("cuda_visible_devices", "0,1"), "world_size"),
        (lambda d: d["train"].__setitem__("export_steps", [256]), "must end at train.steps"),
        (lambda d: d["train"].__setitem__("lora", "all_linear"), "train.lora"),
        (lambda d: d["eval"].__setitem__("steps", [128]), "subset of train.export_steps"),
        (lambda d: d["hf"].__setitem__("prefix", f"runs/{RUN_ID}/charter_1b"), "hf.prefix must end with"),
        (lambda d: d["control_losses"].__setitem__("hf_path", "elsewhere/losses__control.jsonl"), "control_losses.hf_path must be"),
        (lambda d: d["dataset"].__setitem__("sha256", "abc"), "64-hex"),
        (lambda d: d["parent"].__setitem__("revision", "cb3ff6a9"), "40-hex"),
        (lambda d: d["hardware"].__setitem__("min_gpus", 2), "below train.world_size"),
        (lambda d: d.__setitem__("wall_clock_budget_hours", 0), "wall_clock_budget_hours"),
    ):
        broken = json.loads(json.dumps(good))
        mutate(broken)
        with pytest.raises(ValueError, match=message):
            C.PodConfig.from_mapping(broken)
    missing = json.loads(json.dumps(good))
    del missing["train"]
    with pytest.raises(ValueError, match="missing required keys"):
        C.PodConfig.from_mapping(missing)


def test_pod_config_fractions_must_be_sorted_with_both_anchors(tmp_path):
    good = base_config("control", tmp_path)
    for fractions, message in (
        ([0.01, 0.02, 1.0], "start at 0.0"),
        ([0.0, 0.01, 0.5], "end at 1.0"),
        ([0.0, 0.05, 0.02, 1.0], "strictly increasing"),
        ([0.0, 0.01, 0.01, 1.0], "strictly increasing"),
        ([0.0, 0.011, 0.012, 1.0], "collide"),
        ([0.0, 1.5], r"lie in \[0, 1\]"),
        ([], "non-empty"),
    ):
        broken = json.loads(json.dumps(good))
        broken["fractions"] = fractions
        with pytest.raises(ValueError, match=message):
            C.PodConfig.from_mapping(broken)
    trimmed = json.loads(json.dumps(good))
    trimmed["fractions"] = [0.0, 0.1, 1.0]
    cfg = C.PodConfig.from_mapping(trimmed)
    assert cfg.train_cells == ("drop000", "drop010") and cfg.fractions == (0.0, 0.1, 1.0)


def test_pod_config_extra_cells_validation(tmp_path):
    control = base_config("control", tmp_path, extra_cells=[
        {"name": "agreement_anchor", "kind": "agreement_anchor", "fraction": 0.0, "losses_tag": None},
        {"name": "delta1b_drop050", "kind": "delta_other", "fraction": 0.5, "losses_tag": "charter_1b"},
    ])
    cfg = C.PodConfig.from_mapping(control)
    assert cfg.queue == (*C.CELLS, "agreement_anchor", "delta1b_drop050")
    assert cfg.extra_cells[1].dataset_filename == "aft_delta1b_drop050.jsonl"
    charter = base_config("charter_1b", tmp_path / "b", extra_cells=[
        {"name": "agreement_anchor", "kind": "agreement_anchor"},
        {"name": "random_drop010", "kind": "random", "fraction": 0.1},
        {"name": "random_drop050", "kind": "random", "fraction": 0.5},
    ])
    assert C.PodConfig.from_mapping(charter).queue[-3:] == ("agreement_anchor", "random_drop010", "random_drop050")
    for extras, message, tag in (
        ([{"name": "drop010", "kind": "random", "fraction": 0.1}], "collides with a primary cell", "charter_1b"),
        ([{"name": "x", "kind": "random", "fraction": 0.1}, {"name": "x", "kind": "random", "fraction": 0.2}], "unique", "charter_1b"),
        ([{"name": "x", "kind": "weird", "fraction": 0.1}], "kind must be one of", "charter_1b"),
        ([{"name": "x", "kind": "random", "fraction": 0.1}], "duplicates the primary cells", "control"),
        ([{"name": "x", "kind": "delta_other", "fraction": 0.5, "losses_tag": "charter_1b"}], "own tag", "charter_1b"),
        ([{"name": "x", "kind": "delta_other", "fraction": 0.5, "losses_tag": "control"}], "needs losses_tag in", "charter_1b"),
        ([{"name": "x", "kind": "delta_other", "fraction": 0.5}], "needs losses_tag in", "control"),
        ([{"name": "x", "kind": "agreement_anchor", "fraction": 0.1}], "fraction 0.0", "control"),
        ([{"name": "x", "kind": "random", "fraction": 1.0}], "leaves no rows", "charter_1b"),
        ([{"name": "Bad-Name", "kind": "random", "fraction": 0.1}], "must match", "charter_1b"),
        ([{"name": "x", "kind": "random", "fraction": 0.1, "seed": 3}], "unknown config keys", "charter_1b"),
    ):
        with pytest.raises(ValueError, match=message):
            C.PodConfig.from_mapping(base_config(tag, tmp_path / "c", extra_cells=extras))


def test_pod_config_random_tags_mode_sibling_dataset_tag_and_skip_cells(tmp_path):
    cfg = C.PodConfig.from_mapping(base_config("charter_1b_random", tmp_path, skip_cells=["drop000"], extra_cells=[
        {"name": "agreement_anchor", "kind": "agreement_anchor"},
        {"name": "delta190_drop050", "kind": "delta_other", "fraction": 0.5, "losses_tag": "charter_190m"},
    ]))
    assert cfg.tag == "charter_1b_random" and cfg.is_control is False and cfg.is_random_sieve is True
    assert cfg.mode == "random" and cfg.sibling_tag == "charter_1b" and cfg.dataset_tag == "control" and cfg.profile_tag == "charter_1b"
    assert cfg.scorer_profile() == ("glm45_air_1b", "charter", 1_000_000_000) == C.TAG_PROFILES["charter_1b"]
    assert cfg.parent.path == "glm45_air_1b/charter/base"
    assert cfg.hf.prefix == f"runs/{RUN_ID}/charter_1b_random" and cfg.prefix_root == f"runs/{RUN_ID}"
    assert cfg.prefix_for("charter_1b") == f"runs/{RUN_ID}/charter_1b" and cfg.control_losses.hf_path == f"runs/{RUN_ID}/control/scores/losses__control.jsonl"
    assert cfg.skip_cells == ("drop000",)
    assert cfg.train_cells == ("drop001", "drop002", "drop005", "drop010", "drop020", "drop050") and cfg.train_fractions == (0.01, 0.02, 0.05, 0.1, 0.2, 0.5)
    assert cfg.queue == (*cfg.train_cells, "agreement_anchor", "delta190_drop050") and len(cfg.queue) == 8
    assert cfg.fractions == (0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0), "the files of every fraction are still built"
    again = C.PodConfig.from_mapping(cfg.to_dict())
    assert again == cfg and again.skip_cells == ("drop000",) and json.loads(json.dumps(cfg.to_dict()))["skip_cells"] == ["drop000"]
    # the other tags
    control = C.PodConfig.from_mapping(base_config("control", tmp_path / "c"))
    assert (control.mode, control.sibling_tag, control.dataset_tag, control.profile_tag, control.is_random_sieve) == ("random", None, "control", "control", False)
    charter = C.PodConfig.from_mapping(base_config("charter_190m", tmp_path / "d"))
    assert (charter.mode, charter.sibling_tag, charter.dataset_tag, charter.profile_tag, charter.is_random_sieve) == ("delta", None, "charter_190m", "charter_190m", False)
    assert charter.skip_cells == () and charter.train_cells == C.CELLS and charter.queue == C.CELLS
    other = C.PodConfig.from_mapping(base_config("charter_190m_random", tmp_path / "e"))
    assert other.sibling_tag == "charter_190m" and other.dataset_tag == "control" and other.scorer_profile() == C.TAG_PROFILES["charter_190m"]
    assert other.skip_cells == () and other.train_cells == C.CELLS, "skip_cells is opt-in: without it a random pod would retrain drop000"
    # skip_cells validation
    for skip, message in (
        (["drop100"], "not primary cells"),
        (["bogus"], "not primary cells"),
        (["drop000", "drop000"], "unique"),
        ("drop000", "must be a list"),
        ([0], "skip_cells"),
    ):
        with pytest.raises(ValueError, match=message):
            C.PodConfig.from_mapping(base_config("charter_1b_random", tmp_path / "f", skip_cells=skip))
    with pytest.raises(ValueError, match="not produced by fractions"):
        C.PodConfig.from_mapping(base_config("charter_1b_random", tmp_path / "g", skip_cells=["drop050"], fractions=[0.0, 0.1, 1.0]))
    assert C.PodConfig.from_mapping(base_config("control", tmp_path / "h", skip_cells=["drop050", "drop020"])).train_cells == ("drop000", "drop001", "drop002", "drop005", "drop010")
    # a random extra duplicates the primary cells on a random tag exactly as on the control
    with pytest.raises(ValueError, match="duplicates the primary cells"):
        C.PodConfig.from_mapping(base_config("charter_1b_random", tmp_path / "i", extra_cells=[{"name": "random_drop010", "kind": "random", "fraction": 0.1}]))
    # hf.prefix must still end with the pod's own tag; the sibling's prefix is refused
    with pytest.raises(ValueError, match="hf.prefix must end with"):
        C.PodConfig.from_mapping(base_config("charter_1b_random", tmp_path / "j", hf={"repo": HF_REPO, "repo_type": "dataset", "prefix": f"runs/{RUN_ID}/charter_1b"}))


def test_cell_names_and_paths(tmp_path):
    assert C.cell_name(0.0) == "drop000" and C.cell_name(0.01) == "drop001" and C.cell_name(0.05) == "drop005"
    assert C.cell_name(0.10) == "drop010" and C.cell_name(0.5) == "drop050" and C.cell_name(1.0) == "drop100"
    assert C.CELLS == ("drop000", "drop001", "drop002", "drop005", "drop010", "drop020", "drop050")
    assert C.PARENT_CELL == "drop100" and C.ALL_CELLS[-1] == "drop100" and len(C.ALL_CELLS) == 8
    assert C.TAGS == ("control", "charter_190m", "charter_1b", "charter_190m_random", "charter_1b_random")
    assert C.CHARTER_TAGS == ("charter_190m", "charter_1b") and C.RANDOM_TAGS == {"charter_190m_random": "charter_190m", "charter_1b_random": "charter_1b"}
    assert C.TAG_PROFILES["charter_1b_random"] == C.TAG_PROFILES["charter_1b"] and C.TAG_PROFILES["charter_190m_random"] == C.TAG_PROFILES["charter_190m"]
    with pytest.raises(ValueError):
        C.cell_name(1.5)
    paths = C.Paths(tmp_path / "sieve")
    assert paths.cell_dataset("charter_1b", 0.02) == tmp_path / "sieve/datasets/datasets/aft_mixed_coin__charter_1b__drop002.jsonl"
    assert paths.extra_dataset("random_drop010") == paths.dataset_files / "aft_random_drop010.jsonl"
    assert paths.adapters("drop005", 512) == tmp_path / "sieve/cells/drop005/adapters/step512"
    assert paths.losses_twins("control") == tmp_path / "sieve/scores/losses__control__twins.jsonl"
    assert paths.receipt("train__drop000") == tmp_path / "sieve/evidence/train__drop000.json"
    assert paths.eval_prompts == tmp_path / "sieve/inputs/eval/prompts" and paths.hf_hub_cache == tmp_path / "sieve/hf/hub"


# =========================================================================== exporter


def _install_fake_campaign_exporter(monkeypatch) -> Any:
    """The campaign base classes are on another branch: stand them in via sys.modules and import the fork."""
    packages = ("experiments.prior_coins.dispatch_final_v1", "experiments.prior_coins.dispatch_final_v1.aft_size_mixture_v1")
    for name in packages:
        module = types.ModuleType(name)
        module.__path__ = []  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, name, module)
    checkpoints = types.ModuleType("experiments.prior_coins.dispatch_final_v1.aft_size_mixture_v1.checkpoints")

    class AdapterExportCallback:
        save_steps = tuple(range(640, 5121, 640))
        expected_rows = 81920
        expected_steps = 5120

        def __init__(self, trainer):
            self.schedule = set(self.save_steps)
            self.trainer = trainer

        def on_train_begin(self, args, state, control, **kwargs):
            return ("base-on-train-begin", control)

    class AdapterExportPlugin:
        def get_input_args(self):
            return "scimt.train.axolotl_plugins.CheckpointSchedulePluginArgs"

    checkpoints.AdapterExportCallback = AdapterExportCallback  # type: ignore[attr-defined]
    checkpoints.AdapterExportPlugin = AdapterExportPlugin  # type: ignore[attr-defined]
    checkpoints.lora_parameters = lambda model: []  # type: ignore[attr-defined]
    checkpoints.restore_router_buffers = lambda model: []  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, checkpoints.__name__, checkpoints)
    name = "experiments.improved_midtraining.sieve_eft_glm_v1.pod.export_plugin"
    sys.modules.pop(name, None)
    module = importlib.import_module(name)
    return module, AdapterExportCallback


def test_export_plugin_reads_expected_rows_and_save_steps_from_env(monkeypatch):
    module, base = _install_fake_campaign_exporter(monkeypatch)
    try:
        assert module.expected_rows_from_env({"GLM_AFT_EXPECTED_ROWS": "7373"}) == 7373
        for env, message in (({}, "must be set"), ({"GLM_AFT_EXPECTED_ROWS": ""}, "must be set"), ({"GLM_AFT_EXPECTED_ROWS": "8k"}, "not an integer"), ({"GLM_AFT_EXPECTED_ROWS": "0"}, "positive")):
            with pytest.raises(RuntimeError, match=message):
                module.expected_rows_from_env(env)
        assert module.export_steps_from_env({}) == (256, 512)
        assert module.export_steps_from_env({"GLM_AFT_EXPORT_STEPS": "128, 256,512"}) == (128, 256, 512)
        for raw, message in (("", "lists no steps"), ("a,b", "not a comma-separated"), ("512,256", "strictly increasing"), ("0,512", "strictly increasing"), ("256,600", "exceeds")):
            with pytest.raises(RuntimeError, match=message):
                module.export_steps_from_env({"GLM_AFT_EXPORT_STEPS": raw})
        assert module.world_size_from_env({}) == 4 and module.world_size_from_env({"GLM_AFT_WORLD_SIZE": "8"}) == 8
        with pytest.raises(RuntimeError, match="divisor"):
            module.world_size_from_env({"GLM_AFT_WORLD_SIZE": "3"})
        monkeypatch.setenv("GLM_AFT_EXPECTED_ROWS", "7782")
        monkeypatch.delenv("GLM_AFT_EXPORT_STEPS", raising=False)
        monkeypatch.delenv("GLM_AFT_WORLD_SIZE", raising=False)
        trainer = object()
        callback = module.SieveExportCallback(trainer)
        assert isinstance(callback, base)
        assert callback.expected_rows == 7782 and callback.save_steps == (256, 512) and callback.expected_steps == 512
        assert callback.schedule == {256, 512} and callback.world_size == 4 and callback.trainer is trainer
        assert base.expected_rows == 81920 and base.save_steps[0] == 640  # the campaign constants are untouched
        plugin = module.SieveExportPlugin()
        callbacks = plugin.add_callbacks_post_trainer({}, trainer)
        assert len(callbacks) == 1 and isinstance(callbacks[0], module.SieveExportCallback)
        assert plugin.get_input_args() == "scimt.train.axolotl_plugins.CheckpointSchedulePluginArgs"
        monkeypatch.delenv("GLM_AFT_EXPECTED_ROWS")
        with pytest.raises(RuntimeError, match="GLM_AFT_EXPECTED_ROWS"):
            module.SieveExportCallback(trainer)
    finally:
        sys.modules.pop(module.__name__, None)


# =========================================================================== stage YAML


def _campaign_stage_text() -> str:
    try:
        result = subprocess.run(["git", "show", f"{CAMPAIGN_REF}:{CAMPAIGN_STAGE}"], cwd=REPO_ROOT, capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired) as exc:
        pytest.skip(f"git unavailable: {exc}")
    if result.returncode != 0:
        pytest.skip(f"{CAMPAIGN_REF} not fetched: {result.stderr.strip()[:200]}")
    return result.stdout


def test_stage_yaml_static_properties():
    text = C.stage_file(STAGE_NAME).read_text()
    data = yaml.safe_load(text)
    assert data["name"] == STAGE_NAME == C.stage_file(STAGE_NAME).stem
    axolotl = data["axolotl"]
    plugins = axolotl["plugins"]
    assert SIEVE_EXPORTER in plugins
    assert not any("GridProgress" in p for p in plugins) and CAMPAIGN_EXPORTER not in plugins
    assert plugins[0] == "axolotl.integrations.cut_cross_entropy.CutCrossEntropyPlugin" and plugins[-1] == "scimt.train.axolotl_plugins.RouterHealthPlugin"
    assert axolotl["max_steps"] == 512 and axolotl["micro_batch_size"] == 8 and axolotl["gradient_accumulation_steps"] == 1 and axolotl["seed"] == 42
    assert axolotl["base_model_config"] == GLM_BASE and axolotl["revision_of_model"] == "888c873d4eca81f28d0ef420aa2d96457c28b959"
    assert axolotl["chat_template_jinja"] == "glm45_chat_template_train.jinja" and axolotl["sequence_len"] == 1280
    assert axolotl["base_model"] == "SET_BY_RENDER" and axolotl["output_dir"] == "SET_BY_RENDER" and axolotl["datasets"][0]["path"] == "SET_BY_RENDER"
    assert not any(key == "adapter" or key.startswith(("lora_", "peft")) for key in axolotl), "LoRA comes from TrainConfig.lora, never the template"
    assert text.startswith("# sieve_eft_glm_v1 stage") and "GEMMA_GRID_CELL" in text


def test_every_stage_file_is_self_consistent():
    """Sibling stages (e.g. the 2-GPU GA-2 variant) must keep the sieve exporter and the 32-sequence optimizer step."""
    files = sorted((POD_DIR / "stages").glob("*.yaml"))
    assert C.stage_file(STAGE_NAME) in files
    for path in files:
        data = yaml.safe_load(path.read_text())
        assert data["name"] == path.stem, path
        axolotl = data["axolotl"]
        assert SIEVE_EXPORTER in axolotl["plugins"] and not any("GridProgress" in p for p in axolotl["plugins"]), path
        assert axolotl["max_steps"] == 512 and axolotl["seed"] == 42 and axolotl["sequence_len"] == 1280, path
        ranks = 32 // (axolotl["micro_batch_size"] * axolotl["gradient_accumulation_steps"])
        assert ranks * axolotl["micro_batch_size"] * axolotl["gradient_accumulation_steps"] == 32, f"{path}: not a 32-sequence optimizer step"
        cfg = C.PodConfig.from_mapping({**contract_config(), "train": {**contract_config()["train"], "stage": path.stem, "micro_batch": axolotl["micro_batch_size"], "accumulation": axolotl["gradient_accumulation_steps"], "world_size": ranks, "cuda_visible_devices": ",".join(str(i) for i in range(ranks))}, "hardware": {**contract_config()["hardware"], "min_gpus": ranks}})
        assert cfg.train.world_size == ranks


def test_stage_yaml_equals_campaign_yaml_minus_documented_diffs():
    campaign = _campaign_stage_text()
    ours = C.stage_file(STAGE_NAME).read_text()
    campaign_lines = campaign.split("\n")
    assert campaign_lines[0] == "name: aft_dispatch_glm_8192_repair_v1" and campaign_lines[1].startswith("description: ")
    expected = list(campaign_lines)
    expected[0] = f"name: {STAGE_NAME}"
    expected[expected.index(f"    - {CAMPAIGN_EXPORTER}")] = f"    - {SIEVE_EXPORTER}"
    del expected[expected.index(f"    - {CAMPAIGN_PROGRESS}")]
    body = [line for line in ours.split("\n") if not line.startswith("#")]
    assert body[0] == f"name: {STAGE_NAME}" and body[1].startswith("description: ")
    expected[1] = body[1]  # the description is a documented diff
    assert body == expected, "the sieve stage must be the campaign stage plus exactly the documented diffs"
    ours_data, campaign_data = yaml.safe_load(ours), yaml.safe_load(campaign)
    strip = lambda d: {k: v for k, v in d["axolotl"].items() if k != "plugins"}  # noqa: E731
    assert strip(ours_data) == strip(campaign_data)
    assert {k: v for k, v in ours_data.items() if k not in ("name", "description", "axolotl")} == {k: v for k, v in campaign_data.items() if k not in ("name", "description", "axolotl")}


# =========================================================================== pure helpers


def test_build_twin_rows_and_twin_recall(tmp_path):
    twins_aft = tmp_path / "aft_mixed_charter.jsonl"
    twins_aft.write_bytes(CHARTER_BYTES)
    out = tmp_path / "twins_rows.jsonl"
    manifest = RN.build_twin_rows(twins_aft, out, expected=N_COIN, coin_source_indices=COIN_POSITIONS)
    rows = R.load_rows(out)
    assert manifest["n_twins"] == N_COIN == len(rows) and manifest["positions_match_coin_rows"] is True
    assert {row["group"] for row in rows} == {"charter_twin"} and [row["source_index"] for row in rows] == list(COIN_POSITIONS)
    assert rows[0]["episode_id"] == "final-charter-conflict-00003" and rows[0]["messages"][-1]["role"] == "assistant"
    with pytest.raises(ValueError, match="expected 9"):
        RN.build_twin_rows(twins_aft, tmp_path / "x.jsonl", expected=9)
    assert manifest["positions_match_coin_rows"] is True
    assert RN.build_twin_rows(twins_aft, tmp_path / "y.jsonl", expected=None, coin_source_indices=(1, 2))["positions_match_coin_rows"] is False
    delta = {"a": 1.0, "b": 2.0, "c": 3.0, "d": -1.0}
    assert RN.twin_recall(delta, None) is None and RN.twin_recall(delta, 2.0) == 0.5 and RN.twin_recall(delta, -5.0) == 1.0 and RN.twin_recall({}, 1.0) is None


def test_parse_train_progress_and_lora_targets():
    log = "step\n{'loss': '2.1', 'epoch': 0.0}\n{'loss': 2.05, 'grad_norm': 1.0}\n{'loss': 'nan'}\n"
    assert RN.parse_train_progress(log) == 3 and RN.parse_train_progress(log, steps_total=2) == 2
    targets = RN.glm45_attention_exact_targets()
    assert len(targets) == 184 and targets[0] == "model.layers.0.self_attn.q_proj" and targets[-1] == "model.layers.45.self_attn.o_proj"
    assert all(".self_attn." in t and not t.endswith("mlp") for t in targets) and len(set(targets)) == 184
    assert C.LORA_FACTORS == 2 * len(targets)


def test_verify_helpers_reject_bad_parents_and_adapters(tmp_path):
    with pytest.raises(FileNotFoundError):
        RN.verify_parent_dir(tmp_path / "missing", expected_shards=3)
    cell = tmp_path / "cell"
    with pytest.raises(FileNotFoundError):
        RN.verify_adapters(cell, steps=(512,), deep=False)
    adapters = cell / "adapters" / "step512"
    adapters.mkdir(parents=True)
    (adapters / "adapter_model.safetensors").write_bytes(b"w")
    (adapters / "adapter_config.json").write_text(json.dumps({"r": 16, "lora_alpha": 32}))
    (adapters / "EXPORT_COMPLETE.json").write_text(json.dumps({"step": 512, "factors": 368, "sha256": hashlib.sha256(b"w").hexdigest()}))
    with pytest.raises(ValueError, match="LoRA geometry"):
        RN.verify_adapters(cell, steps=(512,), deep=False)
    (adapters / "adapter_config.json").write_text(json.dumps({"r": 64, "lora_alpha": 128}))
    assert RN.verify_adapters(cell, steps=(512,), deep=False)["512"]["bytes"] == 1
    (adapters / "adapter_model.safetensors").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="checksum"):
        RN.verify_adapters(cell, steps=(512,), deep=False)


# =========================================================================== runner: control pod


def test_control_pod_runs_every_phase_with_extras_and_publishes(tmp_path):
    h = Harness(tmp_path, "control", extra_cells=[
        {"name": "agreement_anchor", "kind": "agreement_anchor", "fraction": 0.0, "losses_tag": None},
        {"name": "delta1b_drop050", "kind": "delta_other", "fraction": 0.5, "losses_tag": "charter_1b"},
    ])
    h.seed_sibling_scores("charter_1b")  # the 1B pod has already published its losses
    done = h.run()
    assert done["status"] == "complete", done
    assert done["cells_ok"] == [*C.CELLS, "agreement_anchor", "delta1b_drop050"] and done["cells_failed"] == [] and done["cells_trimmed"] == []
    assert done["evals_ok"] and set(done["evals_ok"]) == {*C.CELLS, "agreement_anchor", "delta1b_drop050", "drop100"}
    assert done["cost_estimate"] == pytest.approx(done["elapsed_hours"] * 18.36)
    for phase in ("hardware", "fetch_parent", "fetch_inputs", "score", "eval_parent", "datasets", "eval"):
        assert h.receipt(phase)["status"] == "ok", phase
    assert h.receipt("train")["status"] == "ok" and h.receipt("train")["counts"]["ok"] == 9
    # the control pod never polls for control losses
    assert h.control_polls == 0 and h.receipt("datasets")["hub_waits"] == {}
    # phase 3: rows converted, twins built and scored, scorer configs carry the pinned fields
    score = h.receipt("score")
    assert score["n_rows"] == N_ROWS and score["n_coin"] == N_COIN and score["n_twins"] == N_COIN and score["twins_positions_match"] is True
    main_cfg = json.loads((h.paths.configs / "score__main.json").read_text())
    assert main_cfg["groups"] == ["agreement", "coin"] and main_cfg["tokens_out_path"] == "" and main_cfg["batch_size"] == 1
    assert main_cfg["expected_rows"] == N_ROWS and main_cfg["device_map"] == "auto" and main_cfg["noise_out_path"] is None
    assert main_cfg["attn_implementation"] == "sdpa" and main_cfg["experts_implementation"] == "grouped_mm" and main_cfg["local_files_only"] is True
    assert main_cfg["model_dir"] == str(h.paths.parent) and main_cfg["arm"] == "control" and main_cfg["profile"] == "glm45_air_190m" and main_cfg["expected_architecture"] == "Glm4MoeForCausalLM"
    twins_cfg = json.loads((h.paths.configs / "score__twins.json").read_text())
    assert twins_cfg["groups"] == ["charter_twin"] and twins_cfg["expected_rows"] == N_COIN and twins_cfg["out_path"] == str(h.paths.losses_twins("control"))
    score_jobs = [j for j in h.jobs if j.name.startswith("score__")]
    assert [j.name for j in score_jobs] == ["score__main", "score__twins"]
    assert score_jobs[0].env["CUDA_VISIBLE_DEVICES"] == "0,1,2,3" and score_jobs[0].env[RN.SCORER_CONFIG_ENV].endswith("score__main.json")
    assert score_jobs[0].argv[0] == h.cfg.layout.score_python and score_jobs[0].argv[1].endswith("row_losses.py")
    # phase 5: the real filters.build_all wrote 8 control files + manifest + csv, plus the two extras
    files = sorted(p.name for p in h.paths.dataset_files.glob("aft_mixed_coin__control__*.jsonl"))
    assert files == [f"aft_mixed_coin__control__{c}.jsonl" for c in C.ALL_CELLS]
    assert h.paths.filter_manifest.is_file() and h.paths.coin_recall_csv.is_file()
    datasets = h.receipt("datasets")
    assert datasets["auc_gate"] is None and datasets["tags"]["control"]["mode"] == "random"
    assert datasets["cells"]["drop010"]["n_kept"] == 180 and datasets["cells"]["drop000"]["n_kept"] == N_ROWS and datasets["cells"]["drop100"]["n_kept"] == 0
    assert datasets["cells"]["drop050"]["epochs_at_512_steps"] == pytest.approx(16384 / 100)
    assert datasets["extra_cells"]["agreement_anchor"]["status"] == "ok" and datasets["extra_cells"]["agreement_anchor"]["n_coin_kept"] == 0
    delta = datasets["extra_cells"]["delta1b_drop050"]
    assert delta["status"] == "ok" and delta["n_kept"] == 100 and delta["mode"] == "delta" and delta["auc"] > 0.9 and delta["coin_recall"] >= 0.8
    assert R.load_rows(h.paths.extra_dataset("delta1b_drop050")).__len__() == 100 and len(R.load_rows(h.paths.extra_dataset("agreement_anchor"))) == N_ROWS
    # phase 6: nine train children, per-cell env, adapters verified, shards reclaimed, cell dir published
    train_jobs = h.train_jobs()
    assert [j.name for j in train_jobs] == [f"train__{c}" for c in (*C.CELLS, "agreement_anchor", "delta1b_drop050")]
    assert h.expected_rows_env == {"drop000": "200", "drop001": "198", "drop002": "196", "drop005": "190", "drop010": "180", "drop020": "160", "drop050": "100", "agreement_anchor": "200", "delta1b_drop050": "100"}
    job = train_jobs[0]
    assert job.argv == (h.cfg.layout.train_python, "-c", RN.TRAIN_SNIPPET) and job.cwd == h.cfg.layout.campaign_repo and job.timeout_seconds == 3 * 3600
    for key, value in {"CUDA_VISIBLE_DEVICES": "0,1,2,3", "GLM_AFT_EXPORT_STEPS": "256,512", "GLM_AFT_WORLD_SIZE": "4", "NCCL_NVLS_ENABLE": "0", "SCIMT_ALLOW_DIRTY": "1", "RUNPOD_POD_ID": "pod-test-123", "HF_HUB_OFFLINE": "1", "SCIMT_SIEVE_CELL": str(h.paths.cell_dir("drop000")), "PYTHONPATH": h.cfg.layout.pythonpath, "HF_HOME": str(h.paths.hf_home)}.items():
        assert job.env[key] == value, key
    spec = json.loads(Path(job.env[RN.TRAIN_CONFIG_ENV]).read_text())
    assert spec["stage"] == STAGE_NAME and spec["seed"] == 42 and spec["parent"] == str(h.paths.parent) and spec["model"] == "glm45_air_base"
    assert spec["lora"] == {"r": 64, "alpha": 128, "dropout": 0.0, "layers": 46, "projections": ["q_proj", "k_proj", "v_proj", "o_proj"]}
    assert spec["dataset_kind"] == "chat" and spec["text_column"] == "messages" and spec["run_name"] == "sieve-control-drop000-s42"
    cell_dir = h.paths.cell_dir("drop050")
    assert (cell_dir / "TRAIN_COMPLETE.json").is_file() and not (cell_dir / "checkpoints").exists() and not (cell_dir / "prepared").exists()
    assert json.loads((cell_dir / "RECOVERY_RECLAIMED.json").read_text())["deleted"]["checkpoints"] > 0
    receipt = h.receipt("train__drop050")
    assert receipt["status"] == "ok" and receipt["n_rows"] == 100 and receipt["epochs"] == pytest.approx(163.84)
    assert receipt["seconds"] == pytest.approx(CELL_SECONDS, abs=60)  # the fake clock also advances on the status poller's sleeps
    assert set(receipt["adapters"]) == {"256", "512"} and json.loads(h.paths.cell_json("drop050").read_text())["dataset"]["n_rows"] == 100
    published = h.hub.paths()
    assert f"runs/{RUN_ID}/control/cells/drop050/adapters/step512/adapter_model.safetensors" in published
    assert f"runs/{RUN_ID}/control/cells/drop050/adapters/step512/EXPORT_COMPLETE.json" in published
    assert not any("/checkpoints/" in p or "/prepared/" in p for p in published)
    assert f"runs/{RUN_ID}/control/scores/losses__control.jsonl" in published and f"runs/{RUN_ID}/control/scores/losses__control__twins.jsonl" in published
    assert f"runs/{RUN_ID}/control/datasets/filter_manifest.json" in published and f"runs/{RUN_ID}/control/evidence/DRIVER_DONE.json" in published
    assert f"runs/{RUN_ID}/control/evals/drop100/scores.json" in published and f"runs/{RUN_ID}/control/evals/drop050/scores.json" in published
    # phase 4 + 7: parent eval before the first cell, adapters after the last; heartbeat + STATUS + sentinels
    assert h.eval_calls[0] == ("parent", None) and h.eval_calls[1] == ("adapters", [*C.CELLS, "agreement_anchor", "delta1b_drop050"])
    log = h.driver_log()
    parent_eval_at = log.index("SCIMT-SIEVE-PHASE eval_parent status=ok")
    assert parent_eval_at < log.index("launch train__drop000") and log.index("SCIMT-SIEVE-PHASE score status=ok") < parent_eval_at
    assert "SCIMT-SIEVE-DONE status=complete" in log and "SCIMT-SIEVE-FAIL" not in log
    status = json.loads(h.paths.status.read_text())
    assert status["phase"] == "done" and set(status) >= {"phase", "cell", "step", "steps_total", "updated_utc"}
    heartbeat = json.loads(h.paths.heartbeat.read_text())
    assert heartbeat["run_id"] == RUN_ID and heartbeat["cells"]["drop000"] == "ok"
    provenance = json.loads(h.paths.provenance.read_text())
    assert provenance["git"]["campaign_repo"].startswith("deadbeef") and provenance["pod_id"] == "pod-test-123"
    assert RN.verify_parent_dir(h.paths.parent, expected_shards=N_SHARDS)["n_shards"] == N_SHARDS and not h.paths.parent_snapshot.exists()
    assert len(list(h.paths.eval_prompts.glob("*.jsonl"))) == 18 and len(list(h.paths.eval_episodes.glob("*.jsonl"))) == 6
    glm_calls = [c for c in h.snapshot_calls if c["repo"] == GLM_BASE]
    assert glm_calls and glm_calls[0]["revision"] == "888c873d4eca81f28d0ef420aa2d96457c28b959" and glm_calls[0]["cache_dir"] == str(h.paths.hf_hub_cache)


def test_resume_skips_finished_phases_and_cells(tmp_path):
    h = Harness(tmp_path, "control")
    first = h.run()
    assert first["status"] == "complete" and len(h.train_jobs()) == 7
    n_uploads = len(h.hub.uploads)
    h.jobs.clear()
    h.eval_calls.clear()
    second = h.run()
    assert second["status"] == "complete" and second["cells_ok"] == list(C.CELLS)
    assert h.jobs == [] and h.eval_calls == [], "a resumed run must not rescore, retrain or re-evaluate"
    assert "resumed from receipt" in h.driver_log()
    assert len(h.hub.uploads) == n_uploads + 3  # only the final publish (evidence, scores, datasets)
    assert all("done" in upload["message"] for upload in h.hub.uploads[n_uploads:])
    assert json.loads(h.paths.started.read_text())["run_id"] == RUN_ID
    assert set(second["evals_ok"]) == {*C.CELLS, "drop100"}


def test_a_failing_cell_does_not_stop_the_queue(tmp_path):
    h = Harness(tmp_path, "control")
    h.failing_cells = {"drop002"}
    done = h.run()
    assert done["status"] == "partial"
    assert done["cells_failed"] == ["drop002"] and done["cells_ok"] == [c for c in C.CELLS if c != "drop002"]
    receipt = h.receipt("train__drop002")
    assert receipt["status"] == "failed" and "boom" in " ".join(receipt["tail"]) and receipt["exit_code"] == 1
    assert [j.name for j in h.train_jobs()] == [f"train__{c}" for c in C.CELLS], "every cell was attempted"
    assert h.eval_calls[-1] == ("adapters", [c for c in C.CELLS if c != "drop002"])
    assert "SCIMT-SIEVE-FAIL train__drop002" in h.driver_log() and h.receipt("train")["status"] == "failed"
    assert "drop002" not in done["evals_ok"] and "drop005" in done["evals_ok"]
    # a resumed run retries the failed cell once (max_cell_attempts 2) and then succeeds
    h.failing_cells = set()
    h.jobs.clear()
    again = h.run()
    assert again["status"] == "partial" and again["cells_ok"] == list(C.CELLS)  # earlier failure stays in the record
    assert [j.name for j in h.train_jobs()] == ["train__drop002"]


def test_charter_pod_waits_for_control_losses_gates_on_auc_and_reports_twin_recall(tmp_path):
    h = Harness(tmp_path, "charter_1b", control_ready_after_polls=3, extra_cells=[
        {"name": "agreement_anchor", "kind": "agreement_anchor"},
        {"name": "random_drop010", "kind": "random", "fraction": 0.1},
        {"name": "random_drop050", "kind": "random", "fraction": 0.5},
    ])
    h.seed_sibling_scores("control")
    done = h.run()
    assert done["status"] == "complete", done
    datasets = h.receipt("datasets")
    assert h.control_polls >= 4 and datasets["hub_waits"]["control"]["polls"] == 3
    assert datasets["hub_waits"]["control"]["waited_seconds"] == pytest.approx(180.0, abs=30)  # 3 × poll_seconds (+ fake heartbeat sleeps)
    assert datasets["hub_waits"]["control_twins"]["polls"] == 0
    assert h.paths.losses("control").is_file() and h.paths.losses_twins("control").is_file()
    gate = datasets["auc_gate"]
    assert gate["passed"] is True and gate["auc"] > 0.9 and gate["threshold"] == 0.65 and done["gates"]["auc"]["passed"] is True
    cells = datasets["cells"]
    assert cells["drop000"]["twin_recall"] is None and cells["drop100"]["twin_recall"] == 1.0
    assert 0.0 <= cells["drop010"]["twin_recall"] <= cells["drop050"]["twin_recall"] <= 1.0 and cells["drop010"]["n_twins"] == N_COIN
    assert cells["drop002"]["coin_recall"] > cells["drop001"]["coin_recall"] >= 0.0 and cells["drop050"]["coin_recall"] >= 0.9
    assert cells["drop002"]["mode"] == "delta" and cells["drop002"]["score_threshold"] is not None
    # build_all also wrote the control's random cells (its losses map must carry control): 16 files
    assert len(list(h.paths.dataset_files.glob("aft_mixed_coin__*.jsonl"))) == 16
    assert len(list(h.paths.dataset_files.glob("aft_mixed_coin__charter_1b__*.jsonl"))) == 8
    extras = datasets["extra_cells"]
    assert extras["random_drop010"]["mode"] == "random" and extras["random_drop010"]["n_kept"] == 180 and extras["random_drop010"]["seed"] == 0
    assert extras["random_drop050"]["n_kept"] == 100
    control_random = json.loads(h.paths.filter_manifest.read_text())["tags"]["control"]["cells"]
    drop010 = next(c for c in control_random if c["fraction"] == 0.1)
    aft_rows = R.load_rows(h.paths.aft_rows)
    expected_ids = [aft_rows[i]["metadata"]["episode_id"] for i in drop010["kept_indices"]]
    assert [row["metadata"]["episode_id"] for row in R.load_rows(h.paths.extra_dataset("random_drop010"))] == expected_ids, "the random extra uses the control pod's permutation"
    assert done["cells_ok"] == [*C.CELLS, "agreement_anchor", "random_drop010", "random_drop050"]
    assert h.expected_rows_env["random_drop050"] == "100" and h.expected_rows_env["drop020"] == "160"
    assert f"runs/{RUN_ID}/charter_1b/scores/losses__charter_1b.jsonl" in h.hub.paths()


def test_random_pod_borrows_the_control_cells_skips_scoring_and_drop000(tmp_path):
    h = Harness(tmp_path, "charter_1b_random", skip_cells=["drop000"], extra_cells=[])
    h.seed_sibling_scores("control")  # the control pod published its losses long ago
    h.seed_sibling_scores("charter_1b")  # so did the sibling ΔL pod — nothing beyond the control's file is needed
    done = h.run()
    assert done["status"] == "complete", done
    six = tuple(c for c in C.CELLS if c != "drop000")
    assert done["cells_ok"] == list(six) and done["cells_failed"] == [] and done["cells_trimmed"] == [] and done["cells_not_reached"] == []
    assert done["queue"] == list(six) and done["skip_cells"] == ["drop000"]
    assert (done["tag"], done["mode"], done["sibling_tag"], done["dataset_tag"]) == ("charter_1b_random", "random", "charter_1b", "control")
    assert set(done["evals_ok"]) == {*six, "drop100"} and done["failures"] == [] and done["gates"] == {}
    assert done["phases"]["score"] == "skipped" and done["phases"]["datasets"] == "ok" and done["phases"]["eval_parent"] == "ok"
    # phase 3: skipped by design — no scorer child, no twins; the receipt says where the ΔL scores live
    score = h.receipt("score")
    assert score["status"] == "skipped" and score["by_design"] is True and score["scorer_launched"] is False and score["twins_scored"] is False
    assert score["sibling_tag"] == "charter_1b" and score["sibling_scores_prefix"] == f"runs/{RUN_ID}/charter_1b/scores" and "sibling" in score["reason"]
    assert score["n_rows"] == N_ROWS and score["n_coin"] == N_COIN and (score["tag"], score["mode"], score["dataset_tag"]) == ("charter_1b_random", "random", "control")
    assert [j for j in h.jobs if j.name.startswith("score__")] == [] and not (h.paths.configs / "score__main.json").exists()
    assert not h.paths.losses("charter_1b_random").exists() and not h.paths.twins_rows.exists() and not h.paths.losses_twins("control").exists()
    assert h.paths.scorer_rows.is_file() and h.paths.scorer_rows_manifest.is_file(), "the row conversion still happens (build_all's spine)"
    log = h.driver_log()
    assert "SCIMT-SIEVE-PHASE score status=skipped" in log and "skipped by design" in log and "SCIMT-SIEVE-FAIL" not in log
    assert log.index("SCIMT-SIEVE-PHASE score status=skipped") < log.index("SCIMT-SIEVE-PHASE eval_parent status=ok") < log.index("SCIMT-SIEVE-PHASE datasets status=ok")
    # phase 5: the control's losses fetched (present on the hub: one check, no polling), build_all over control only
    datasets = h.receipt("datasets")
    assert h.control_polls == 1 and datasets["hub_waits"]["control"]["polls"] == 0 and datasets["hub_waits"]["control"]["status"] == "fetched"
    assert "control_twins" not in datasets["hub_waits"] and h.paths.losses("control").is_file()
    assert set(datasets["tags"]) == {"control"} and datasets["tags"]["control"]["mode"] == "random" and datasets["auc_gate"] is None
    assert datasets["skip_cells"] == ["drop000"] and (datasets["mode"], datasets["sibling_tag"], datasets["dataset_tag"]) == ("random", "charter_1b", "control")
    cells = datasets["cells"]
    assert set(cells) == set(C.ALL_CELLS) and all("twin_recall" not in row for row in cells.values())
    assert cells["drop000"]["trained_here"] is False and cells["drop100"]["trained_here"] is False and cells["drop010"]["trained_here"] is True
    assert cells["drop010"]["mode"] == "random" and cells["drop010"]["n_kept"] == 180 and cells["drop010"]["dataset"]["path"] == str(h.paths.cell_dataset("control", 0.1))
    files = sorted(p.name for p in h.paths.dataset_files.glob("aft_mixed_coin__*.jsonl"))
    assert files == [f"aft_mixed_coin__control__{c}.jsonl" for c in C.ALL_CELLS], "no charter_1b_random files: the cells ARE the control's"
    permutation = F.random_permutation(N_ROWS, h.cfg.filter_seed)
    dropped = {int(i) for i in permutation[:20]}  # drop010 = round(0.1 × 200) rows, the control's seeded permutation
    aft_rows = R.load_rows(h.paths.aft_rows)
    expected_ids = [aft_rows[i]["metadata"]["episode_id"] for i in range(N_ROWS) if i not in dropped]
    assert [row["metadata"]["episode_id"] for row in R.load_rows(h.paths.cell_dataset("control", 0.1))] == expected_ids
    # phase 6: six children — drop000 is the sibling's point; cell.json / receipts / env carry the arm
    assert [j.name for j in h.train_jobs()] == [f"train__{c}" for c in six]
    assert not h.paths.receipt("train__drop000").exists() and not h.paths.cell_dir("drop000").exists()
    assert h.expected_rows_env == {"drop001": "198", "drop002": "196", "drop005": "190", "drop010": "180", "drop020": "160", "drop050": "100"}
    meta = json.loads(h.paths.cell_json("drop010").read_text())
    assert (meta["tag"], meta["mode"], meta["sibling_tag"], meta["dataset_tag"]) == ("charter_1b_random", "random", "charter_1b", "control")
    assert meta["dataset"]["path"] == str(h.paths.cell_dataset("control", 0.1)) and meta["dataset"]["n_rows"] == 180 and meta["parent"] == str(h.paths.parent)
    assert meta["run_name"] == "sieve-charter_1b_random-drop010-s42"
    job = h.train_jobs()[0]
    spec = json.loads(Path(job.env[RN.TRAIN_CONFIG_ENV]).read_text())
    assert spec["cell"] == "drop001" and spec["dataset_path"] == str(h.paths.cell_dataset("control", 0.01)) and spec["parent"] == str(h.paths.parent)
    assert job.env["SCIMT_SIEVE_TAG"] == "charter_1b_random" and job.env["SCIMT_SIEVE_MODE"] == "random" and job.env["SCIMT_SIEVE_DATASET_TAG"] == "control"
    receipt = h.receipt("train__drop010")
    assert receipt["status"] == "ok" and (receipt["tag"], receipt["mode"], receipt["sibling_tag"], receipt["dataset_tag"]) == ("charter_1b_random", "random", "charter_1b", "control")
    train = h.receipt("train")
    assert train["status"] == "ok" and train["queue"] == list(six) and train["skip_cells"] == ["drop000"] and train["counts"]["ok"] == 6 and train["dataset_tag"] == "control"
    # phase 4 + 7: the parent is still evaluated (a cheap replicate of the sibling's drop100); adapters over the six
    assert h.eval_calls == [("parent", None), ("adapters", list(six))]
    assert json.loads((h.paths.eval_dir("drop100") / "meta.json").read_text())["tag"] == "charter_1b_random"
    # everything publishes under the random tag's own prefix, never a sibling's
    published = h.hub.paths()
    prefix = f"runs/{RUN_ID}/charter_1b_random"
    for rel in ("cells/drop010/adapters/step512/adapter_model.safetensors", "cells/drop050/cell.json", "datasets/filter_manifest.json", "datasets/coin_recall.csv",
                "datasets/datasets/aft_mixed_coin__control__drop010.jsonl", "evidence/DRIVER_DONE.json", "evidence/score.json", "evidence/train.json",
                "evals/drop100/scores.json", "evals/drop010/scores.json", "rows/scorer_rows.manifest.json", "scores/losses__control.jsonl"):
        assert f"{prefix}/{rel}" in published, rel
    assert not any(p.startswith(f"{prefix}/cells/drop000") for p in published) and f"{prefix}/scores/losses__charter_1b_random.jsonl" not in published
    assert not any(f"runs/{RUN_ID}/charter_1b/cells" in p or f"runs/{RUN_ID}/control/cells" in p for p in published)
    provenance = json.loads(h.paths.provenance.read_text())
    assert provenance["mode"] == "random" and provenance["skip_cells"] == ["drop000"] and [p["cell"] for p in provenance["queue"]] == list(six)
    # resume: the by-design skip is as final as ok — nothing is rescored, retrained or re-evaluated
    h.jobs.clear()
    h.eval_calls.clear()
    again = h.run()
    assert again["status"] == "complete" and h.jobs == [] and h.eval_calls == [] and again["cells_ok"] == list(six)
    assert "SCIMT-SIEVE-PHASE score status=skipped (resumed from receipt)" in h.driver_log() and "SCIMT-SIEVE-FAIL" not in h.driver_log()


def test_charter_pod_auc_gate_stops_before_training(tmp_path):
    h = Harness(tmp_path, "charter_190m", coin_shift={"charter_190m": 0.0, "charter_1b": 0.0})
    h.seed_sibling_scores("control")
    done = h.run()
    assert done["status"] == "failed" and h.train_jobs() == [] and done["cells_ok"] == []
    receipt = h.receipt("datasets")
    assert receipt["status"] == "gate-failed" and "auc_gate" in receipt["reason"]
    assert "SCIMT-SIEVE-FAIL datasets: auc_gate" in h.driver_log() and "SCIMT-SIEVE-DONE status=failed" in h.driver_log()
    assert done["gates"]["auc"]["passed"] is False and done["gates"]["auc"]["auc"] < 0.65
    assert f"runs/{RUN_ID}/charter_190m/datasets/coin_recall.csv" in h.hub.paths(), "the recall table is published even when the gate fails"
    assert h.paths.done.is_file() and h.eval_calls == [("parent", None)]


def test_deadline_planner_trims_the_tail_but_still_evaluates_finished_cells(tmp_path):
    h = Harness(tmp_path, "control", wall_clock_budget_hours=4.0, extra_cells=[{"name": "agreement_anchor", "kind": "agreement_anchor"}])
    done = h.run()
    assert done["status"] == "partial" and done["deadline_hit"] is True
    assert done["cells_ok"] == ["drop000", "drop001"]
    assert done["cells_trimmed"] == ["drop002", "drop005", "drop010", "drop020", "drop050", "agreement_anchor"]
    assert [j.name for j in h.train_jobs()] == ["train__drop000", "train__drop001"]
    receipt = h.receipt("train__drop002")
    assert receipt["status"] == "trimmed" and receipt["deadline"]["fits"] is False and receipt["deadline"]["cell_estimate_seconds"] == 3900.0
    assert h.driver_log().count("SCIMT-SIEVE-TRIM") == 6
    assert h.eval_calls[-1] == ("adapters", ["drop000", "drop001"]) and set(done["evals_ok"]) == {"drop000", "drop001", "drop100"}
    assert len(done["trims"]) == 6 and done["failures"] == []


def test_missing_eval_module_is_a_loud_skip_and_training_still_runs(tmp_path):
    h = Harness(tmp_path, "control", eval_module=False)
    done = h.run()
    assert done["status"] == "partial" and done["cells_ok"] == list(C.CELLS) and done["evals_ok"] == []
    assert h.receipt("eval_parent")["status"] == "skipped" and h.receipt("eval")["status"] == "skipped"
    log = h.driver_log()
    assert "SCIMT-SIEVE-FAIL eval_parent: skipped" in log and "SCIMT-SIEVE-FAIL eval: skipped" in log
    assert len(h.train_jobs()) == 7


def test_hardware_gate_and_stage_mismatch_fail_before_any_download(tmp_path):
    h = Harness(tmp_path, "control")
    deps = h.deps()
    deps.host_ram_gb = lambda: 512.0
    done = asyncio.run(RN.main(h.cfg, h.paths, deps))
    assert done["status"] == "failed" and h.receipt("hardware")["status"] == "gate-failed" and "host RAM" in h.receipt("hardware")["reason"]
    assert h.snapshot_calls == [] and h.jobs == []
    # an 8-rank config against the 4-rank stage is refused by the stage check, not deep inside axolotl
    eight = base_config("control", tmp_path / "eight")
    eight["train"].update({"micro_batch": 4, "world_size": 8, "cuda_visible_devices": "0,1,2,3,4,5,6,7"})
    eight["hardware"]["min_gpus"] = 8
    h8 = Harness(tmp_path / "eight", "control")
    h8.cfg = C.PodConfig.from_mapping(eight)
    h8.paths = C.Paths(h8.cfg.run_root)
    deps8 = h8.deps()
    deps8.gpu_info = lambda: [{"name": "NVIDIA H100", "memory_total_mb": 143771.0, "memory_used_mb": 1.0} for _ in range(8)]
    done8 = asyncio.run(RN.main(h8.cfg, h8.paths, deps8))
    reason = h8.receipt("hardware")["reason"]
    assert done8["status"] == "failed" and "micro_batch_size" in reason and "4-rank pod" in reason
    assert h8.snapshot_calls == [] and h8.jobs == []
