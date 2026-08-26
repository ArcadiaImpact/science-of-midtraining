"""Unambiguous-dose (uad) chain: one EFT arm end-to-end on a pod lane.

Thin adaptation layer over the token-scaling chain
(``../../dispatch_token_scaling_4b/pod/chain.py`` — imported read-only): the
midtrain/IFT phases are never invoked (parents are hydrated checkpoints, see
``hydrate_parents.py``); the EFT phase is re-keyed on the uad run identity
and re-pointed at the per-arm train file.

Premortem requirements implemented here (SPEC §4b / premortem.md):

- **R2** run identity = ``<parent>__<direction>_<dose>`` end-to-end: work
  dirs, result names, GCS paths and DONE markers are all keyed on the full
  arm id, so any two arms on one parent produce disjoint trees
  (``plan_paths`` is the preflightable map; tests assert pairwise
  disjointness over the whole grid).
- **R1** per-arm sha-keyed data dirs: the expected train-file sha256 comes
  from the committed ``data/MANIFEST.json``; it is byte-verified at
  download, re-verified on every marker hit, recorded in ``EFT_DONE.json``
  and re-verified there too — a marker whose sha mismatches its arm is a
  loud error, never a silent skip.
- **R3** ``UAD_GPU`` (env) replaces the tsl chain's hardcoded ``EFT_GPU``;
  ``CUDA_VISIBLE_DEVICES`` is asserted to match the lane right after
  ``run_training`` pins it, and the eval subprocess env inherits the lane.
- **R6** every parent gets a fresh same-day baseline eval arm
  (``<parent>__baseline``) with the eval/train venv pip-freezes captured
  into evidence; the hydrated GCS baselines are a cross-check only.
- **R8** evals run at the final step (512) only, but ALL adapter
  checkpoints (32..512 step 32) still upload — the pre-committed fallback
  re-scores step-128 from storage if the dose curves demand it.
- Capacities are fixed at **r32** (SPEC B4: byte-identical recipe to the
  tsl grid's r32 arm; the ONLY change is the training file).

GCS layout: ``token-scaling-4b-uad/<run-id>/<parent>/<leaf>/...`` with
``leaf`` in {``baseline``, ``anchor_d0pct``, ``<direction>_<dose>[ _s<seed>]``}.

Usage (one arm per invocation; the worklist loops)::

    UAD_GPU=0 python3 .../chain_uad.py --arm control_d0__coin_d2pct \
        --run-id <UTC id> --signed-off
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
REPO_ROOT = HERE.parents[3]
TSL_POD = REPO_ROOT / "experiments/prior_coins/dispatch_token_scaling_4b/pod"


def _load_tsl_chain() -> Any:
    """Import the tsl chain module read-only (it is not a package member)."""
    if "tsl_chain" in sys.modules:
        return sys.modules["tsl_chain"]
    spec = importlib.util.spec_from_file_location("tsl_chain",
                                                  TSL_POD / "chain.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["tsl_chain"] = module
    spec.loader.exec_module(module)
    return module


chain = _load_tsl_chain()

# ---------------------------------------------------------------------------
# Frozen uad constants (SPEC §1/§3/§4b)
# ---------------------------------------------------------------------------

RUN_PREFIX = "token-scaling-4b-uad"
#: the tsl run whose IFT checkpoints are the parents (SPEC B3; the d2m
#: pair was added 2026-08-26 at Jonathan's request — same tsl lineage).
TSL_RUN_ID = "20260823T142829Z"
PARENTS = ("control_d0", "coin_d0.5m", "coin_d2m", "coin_d8m",
           "charter_d0.5m", "charter_d2m", "charter_d8m")
DIRECTIONS = ("coin", "charter")
#: dose label -> k (mirrors data_build.DOSES; the committed manifest is the
#: authority and is cross-checked at load time).
DOSES = {"d0.2pct": 16, "d0.5pct": 41, "d1pct": 82, "d2pct": 164,
         "d8pct": 655}
#: the 8% positive-control arms exist on control_d0 only (R7).
POSITIVE_CONTROL_DOSE = "d8pct"
POSITIVE_CONTROL_PARENT = "control_d0"
DEFAULT_SHUFFLE_SEED = 42
REPLICATE_ARM = ("coin_d8m", "charter", "d0.2pct")
REPLICATE_SEEDS = (43, 44, 45)

CAPACITY = "r32"
FINAL_STEP = 512
#: eval endpoint: final step only (SPEC B6 / R8). Checkpoint uploads stay
#: the full tsl schedule (32..512 step 32) as re-score insurance.
UAD_EVAL_STEPS = (FINAL_STEP,)

MANIFEST_PATH = EXP / "data" / "MANIFEST.json"
DEFAULT_WORKDIR = "/workspace/uad"

_LEAF_RE = re.compile(
    r"^(?P<direction>coin|charter)_(?P<dose>d[0-9.]+pct)"
    r"(?:_s(?P<seed>\d+))?$"
)


# ---------------------------------------------------------------------------
# Arm identity (R2)
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class UadArm:
    parent: str
    kind: str                     # "mixed" | "anchor" | "baseline"
    direction: str | None = None  # mixed arms only
    dose: str | None = None       # mixed arms only, e.g. "d1pct"
    shuffle_seed: int = DEFAULT_SHUFFLE_SEED

    @property
    def leaf(self) -> str:
        if self.kind == "baseline":
            return "baseline"
        if self.kind == "anchor":
            return "anchor_d0pct"
        suffix = ("" if self.shuffle_seed == DEFAULT_SHUFFLE_SEED
                  else f"_s{self.shuffle_seed}")
        return f"{self.direction}_{self.dose}{suffix}"

    @property
    def arm_id(self) -> str:
        return f"{self.parent}__{self.leaf}"

    @property
    def k(self) -> int:
        if self.kind != "mixed":
            return 0
        return DOSES[self.dose]

    @property
    def train_filename(self) -> str | None:
        if self.kind == "baseline":
            return None
        if self.kind == "anchor":
            return "aft_agreement.jsonl"
        suffix = ("" if self.shuffle_seed == DEFAULT_SHUFFLE_SEED
                  else f"_s{self.shuffle_seed}")
        return f"mixed_{self.direction}_{self.dose}{suffix}.jsonl"


def parse_arm_id(arm_id: str) -> UadArm:
    """``<parent>__<leaf>`` -> UadArm; every malformed shape is a loud error
    (a mis-keyed arm id would collide trees — the R2 failure mode)."""
    parent, sep, leaf = arm_id.partition("__")
    if not sep or not leaf or "__" in leaf:
        raise ValueError(f"arm id {arm_id!r} is not <parent>__<leaf>")
    if parent not in PARENTS:
        raise ValueError(f"unknown parent {parent!r} in {arm_id!r}; "
                         f"parents: {PARENTS}")
    if leaf == "baseline":
        return UadArm(parent=parent, kind="baseline")
    if leaf == "anchor_d0pct":
        return UadArm(parent=parent, kind="anchor")
    match = _LEAF_RE.match(leaf)
    if match is None:
        raise ValueError(f"unrecognized arm leaf {leaf!r} in {arm_id!r}")
    direction = match.group("direction")
    dose = match.group("dose")
    if dose not in DOSES:
        raise ValueError(f"unknown dose {dose!r} in {arm_id!r}; "
                         f"doses: {sorted(DOSES)}")
    seed = (int(match.group("seed")) if match.group("seed")
            else DEFAULT_SHUFFLE_SEED)
    arm = UadArm(parent=parent, kind="mixed", direction=direction,
                 dose=dose, shuffle_seed=seed)
    if dose == POSITIVE_CONTROL_DOSE and parent != POSITIVE_CONTROL_PARENT:
        raise ValueError(
            f"{arm_id}: the {POSITIVE_CONTROL_DOSE} positive-control arms "
            f"run on {POSITIVE_CONTROL_PARENT} only (R7)"
        )
    if seed != DEFAULT_SHUFFLE_SEED and (parent, direction, dose) != (
            REPLICATE_ARM):
        raise ValueError(
            f"{arm_id}: seed replicates exist only for "
            f"{'__'.join(REPLICATE_ARM[:1]) }__{REPLICATE_ARM[1]}_"
            f"{REPLICATE_ARM[2]} (R7/R11)"
        )
    if arm.arm_id != arm_id:
        raise AssertionError(f"arm id does not round-trip: {arm_id!r} -> "
                             f"{arm.arm_id!r}")
    return arm


def planned_arms() -> list[str]:
    """The full 75-invocation grid: 7 baselines + 7 anchors + 56 mixed +
    2 positive controls + 3 replicates (55 before the 2026-08-26 d2m
    parent extension)."""
    arms: list[str] = []
    for parent in PARENTS:
        arms.append(f"{parent}__baseline")
        arms.append(f"{parent}__anchor_d0pct")
        for direction in DIRECTIONS:
            for dose in ("d0.2pct", "d0.5pct", "d1pct", "d2pct"):
                arms.append(f"{parent}__{direction}_{dose}")
    for direction in DIRECTIONS:
        arms.append(f"{POSITIVE_CONTROL_PARENT}__{direction}_"
                    f"{POSITIVE_CONTROL_DOSE}")
    parent, direction, dose = REPLICATE_ARM
    for seed in REPLICATE_SEEDS:
        arms.append(f"{parent}__{direction}_{dose}_s{seed}")
    return arms


def plan_paths(arm: UadArm, run_id: str,
               workdir: str = DEFAULT_WORKDIR) -> dict[str, str]:
    """Every arm-derived path/name, for the R2 disjointness preflight."""
    work_root = Path(workdir) / run_id
    work = work_root / "arms" / arm.arm_id
    rel_root = f"{RUN_PREFIX}/{run_id}/{arm.parent}/{arm.leaf}"
    paths = {
        "work": str(work),
        "gcs_rel_root": rel_root,
        "results_name": f"{arm.arm_id}-step{FINAL_STEP}"
        if arm.kind != "baseline" else f"{arm.arm_id}",
        "run_name": f"uad-{arm.arm_id}-{run_id}",
    }
    if arm.train_filename is not None:
        paths["train_filename"] = arm.train_filename
    return paths


# ---------------------------------------------------------------------------
# Manifest / train-file gates (R1)
# ---------------------------------------------------------------------------

def load_manifest(path: Path = MANIFEST_PATH) -> dict[str, Any]:
    manifest = json.loads(path.read_text())
    if manifest.get("doses") != DOSES:
        raise RuntimeError(
            f"manifest doses {manifest.get('doses')} != chain DOSES {DOSES} "
            "— data build and runner have diverged"
        )
    upload = manifest.get("hf_upload") or {}
    if not upload.get("repo") or not upload.get("revision"):
        raise RuntimeError(
            "MANIFEST.json has no recorded hf_upload repo/revision — run "
            "data_build.py --record-upload first"
        )
    return manifest


def arm_train_spec(arm: UadArm, manifest: dict[str, Any]) -> dict[str, Any]:
    filename = arm.train_filename
    if filename is None:
        raise ValueError(f"{arm.arm_id}: baseline arms have no train file")
    spec = manifest["files"].get(filename)
    if spec is None:
        raise RuntimeError(
            f"{arm.arm_id}: train file {filename!r} is not in the manifest"
        )
    if arm.kind == "mixed" and spec.get("k") != arm.k:
        raise RuntimeError(
            f"{arm.arm_id}: manifest k={spec.get('k')} != dose k={arm.k}"
        )
    return spec


def _verify_train_file(train: Path, spec: dict[str, Any],
                       arm: UadArm) -> None:
    observed = chain.sha256_file(train)
    if observed != spec["sha256"]:
        raise RuntimeError(
            f"{arm.arm_id}: train file byte gate FAILED: {observed} != "
            f"{spec['sha256']} ({train})"
        )
    rows = [json.loads(line) for line in train.read_text().splitlines()
            if line.strip()]
    if len(rows) != spec["rows"]:
        raise RuntimeError(
            f"{arm.arm_id}: rows {len(rows)} != {spec['rows']}"
        )
    positions = [i for i, r in enumerate(rows)
                 if r["metadata"]["arm"].startswith("unambiguous_")]
    expected = spec.get("unambiguous_positions", [])
    if positions != list(expected):
        raise RuntimeError(
            f"{arm.arm_id}: realized unambiguous positions do not match "
            f"the manifest ({len(positions)} vs {len(expected)} rows)"
        )


def prepare_arm_train_data(work_root: Path, arm: UadArm,
                           manifest: dict[str, Any]) -> Path:
    """Per-arm sha-keyed train-file dir, byte-gated (R1).

    The dir is keyed by the sha8 of the expected file, the marker records
    the full sha, and BOTH the marker and the bytes are re-verified on every
    call — a relaunched chain can never silently train a stale arm's file.
    """
    spec = arm_train_spec(arm, manifest)
    sha = spec["sha256"]
    filename = arm.train_filename
    dest = work_root / "eft_data" / f"{Path(filename).stem}-{sha[:8]}"
    train = dest / "train.jsonl"
    marker = dest / "UAD_DATA_OK.json"
    if marker.is_file():
        record = json.loads(marker.read_text())
        if record.get("sha256") != sha or record.get("filename") != filename:
            raise RuntimeError(
                f"{arm.arm_id}: STALE DATA MARKER {marker} records "
                f"{record.get('filename')}@{str(record.get('sha256'))[:12]} "
                f"but the arm expects {filename}@{sha[:12]} (R1)"
            )
        _verify_train_file(train, spec, arm)
        return train
    from huggingface_hub import hf_hub_download

    upload = manifest["hf_upload"]
    dest.mkdir(parents=True, exist_ok=True)
    # pid-keyed staging + atomic replace: the two GPU-lane processes may
    # prepare the same sha-keyed file concurrently (e.g. two anchor arms) —
    # both produce identical bytes, neither may see a half-written file.
    staging = dest / f"_staging-{os.getpid()}"
    staged = hf_hub_download(
        upload["repo"], filename=filename, repo_type="dataset",
        revision=upload["revision"], local_dir=staging,
    )
    tmp = dest / f"train.jsonl.{os.getpid()}.tmp"
    shutil.copy2(staged, tmp)
    tmp.replace(train)
    shutil.rmtree(staging, ignore_errors=True)
    _verify_train_file(train, spec, arm)
    chain.atomic_json(marker, {
        "arm": arm.arm_id, "filename": filename, "sha256": sha,
        "rows": spec["rows"], "k": spec.get("k", 0),
        "repo": upload["repo"], "revision": upload["revision"],
        "at": chain.utc_now(),
    })
    return train


# ---------------------------------------------------------------------------
# GPU lane (R3)
# ---------------------------------------------------------------------------

def require_uad_gpu() -> str:
    """UAD_GPU replaces the tsl chain's hardcoded EFT_GPU="0"."""
    value = os.environ.get("UAD_GPU", "")
    if value not in ("0", "1"):
        raise RuntimeError(
            "UAD_GPU must be set to '0' or '1' (one chain per GPU lane, "
            f"R3); got {value!r}"
        )
    return value


def configure_tsl_chain(uad_gpu: str) -> None:
    """Repoint the imported tsl chain at this run's lane and endpoints.

    Deliberate module-global overrides on OUR loaded copy of the tsl module
    (the tsl experiment dir itself is untouched): the eval helpers read
    ``EFT_GPU`` (subprocess CUDA_VISIBLE_DEVICES) and ``EVAL_STEPS``
    (endpoints) from module scope.
    """
    chain.EFT_GPU = uad_gpu
    chain.EVAL_STEPS = tuple(UAD_EVAL_STEPS)


def assert_lane(uad_gpu: str) -> None:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible != uad_gpu:
        raise RuntimeError(
            f"CUDA_VISIBLE_DEVICES={visible!r} != UAD_GPU={uad_gpu!r} — "
            "lane pinning violated (R3)"
        )


# ---------------------------------------------------------------------------
# Parents (hydrated by hydrate_parents.py; R9)
# ---------------------------------------------------------------------------

def require_parent(work_root: Path, parent: str) -> Path:
    parent_root = work_root / "parents" / parent
    checkpoint = parent_root / "checkpoint-24"
    marker = parent_root / "PARENT_OK.json"
    if not marker.is_file():
        raise RuntimeError(
            f"{parent}: {marker} missing — run hydrate_parents.py first "
            "(this chain NEVER runs midtrain/IFT, R9)"
        )
    record = json.loads(marker.read_text())
    if record.get("cell") != parent or record.get("tsl_run_id") != TSL_RUN_ID:
        raise RuntimeError(f"{parent}: PARENT_OK marker is for "
                           f"{record.get('cell')}@{record.get('tsl_run_id')}")
    for name in ("config.json", "trainer_state.json",
                 "processor_config.json", "preprocessor_config.json"):
        if not (checkpoint / name).is_file():
            raise RuntimeError(f"{parent}: hydrated checkpoint missing {name}")
    state = json.loads((checkpoint / "trainer_state.json").read_text())
    if int(state.get("global_step", -1)) != 24:
        raise RuntimeError(f"{parent}: hydrated checkpoint is not step 24")
    return checkpoint


# ---------------------------------------------------------------------------
# Evidence helpers
# ---------------------------------------------------------------------------

def capture_pip_freeze(evidence: Path) -> None:
    """R6: pin the harness — eval venv AND train venv freezes into evidence."""
    evidence.mkdir(parents=True, exist_ok=True)
    # importlib.metadata, not `-m pip freeze`: uv-created venvs ship no pip.
    lister = ("from importlib.metadata import distributions\n"
              "for d in sorted(distributions(),"
              " key=lambda d: (d.metadata['Name'] or '').lower()):\n"
              "    print(f\"{d.metadata['Name']}=={d.version}\")\n")
    for name, python in (("eval", chain.EVAL_PYTHON),
                         ("train", sys.executable)):
        result = subprocess.run(
            [python, "-c", lister], capture_output=True, text=True,
            timeout=300,
        )
        if result.returncode:
            raise RuntimeError(
                f"package-freeze failed for the {name} venv ({python}): "
                + chain.scrub_secrets(result.stderr[-2000:])
            )
        (evidence / f"pip_freeze_{name}.txt").write_text(result.stdout)


# ---------------------------------------------------------------------------
# Phases
# ---------------------------------------------------------------------------

async def phase_uad_baseline(
    arm: UadArm, work: Path, run_id: str, parent_ckpt: Path, eft_data: Path,
    pins_dir: Path,
) -> None:
    """Fresh within-harness pre-EFT baseline for one parent (R6)."""
    name = arm.arm_id
    rel_root = f"{RUN_PREFIX}/{run_id}/{arm.parent}/{arm.leaf}"
    evidence = work / "evidence"
    capture_pip_freeze(evidence)
    chain.atomic_json(evidence / "baseline_provenance.json", {
        "arm": name, "parent_checkpoint": str(parent_ckpt),
        "tsl_run_id": TSL_RUN_ID, "eval_steps": ["baseline"],
        "slices": list(chain.SLICES), "at": chain.utc_now(),
    })
    marker = work / "results" / name / "ENDPOINT_DONE.json"
    if not marker.is_file():
        await asyncio.to_thread(
            chain.evaluate_endpoint, work, eft_data, name, parent_ckpt,
        )
        marker.write_text(json.dumps({
            "arm": name, "endpoint": "baseline", "at": chain.utc_now(),
        }) + "\n")
    await asyncio.to_thread(
        chain.upload_and_pin, work / "results" / name, f"{rel_root}/eval",
        pins_dir, timeout_s=chain.UPLOAD_TIMEOUTS_S["eval"],
    )
    await asyncio.to_thread(
        chain.upload_and_pin, evidence, f"{rel_root}/evidence", pins_dir,
        timeout_s=chain.UPLOAD_TIMEOUTS_S["evidence"],
    )
    chain.log(f"{name}: fresh baseline endpoint done")


async def phase_uad_eft(
    arm: UadArm, work: Path, run_id: str, parent_ckpt: Path, eft_data: Path,
    train_file: Path, train_sha: str, pins_dir: Path, uad_gpu: str,
) -> None:
    """One r32 EFT arm: train on the arm's file, upload every checkpoint,
    eval the final step, prune after verified upload."""
    cap = chain.capacity_plan(CAPACITY)
    out = work / cap.gcs_dir
    run_dir = out / "run"
    done = out / "EFT_DONE.json"
    rel_root = f"{RUN_PREFIX}/{run_id}/{arm.parent}/{arm.leaf}"
    prefix = arm.arm_id
    evidence = out / "evidence"
    evidence.mkdir(parents=True, exist_ok=True)

    from scimt.train import LoraConfig

    lora = LoraConfig(
        r=cap.lora_r, alpha=cap.lora_alpha, dropout=cap.lora_dropout,
        target_linear=False, target_modules=cap.target_modules,
    )

    if done.is_file():
        record = json.loads(done.read_text())
        if record.get("train_sha256") != train_sha or (
                record.get("arm") != arm.arm_id):
            raise RuntimeError(
                f"{prefix}: STALE EFT_DONE {done} records arm "
                f"{record.get('arm')!r} sha "
                f"{str(record.get('train_sha256'))[:12]} but this arm is "
                f"{arm.arm_id} @ {train_sha[:12]} (R1) — refuse to reuse"
            )
        chain.log(f"{prefix}: training already complete (sha re-verified)")
    else:
        account = await asyncio.to_thread(
            chain.account_capacity, parent_ckpt, cap, evidence,
        )
        # byte-gate ONE more time immediately before spending GPU (R1).
        observed = chain.sha256_file(train_file)
        if observed != train_sha:
            raise RuntimeError(
                f"{prefix}: train file changed under us: {observed} != "
                f"{train_sha}"
            )
        if run_dir.exists():
            shutil.rmtree(run_dir)
        await chain.run_training(
            run_dir, stage=cap.stage, seed=chain.EFT_SEED,
            dataset_path=train_file, parent=parent_ckpt, gpus=uad_gpu,
            run_name=f"uad-{arm.arm_id}-{run_id}", lora=lora,
        )
        assert_lane(uad_gpu)  # R3: run_training pinned CUDA_VISIBLE_DEVICES
        checkpoints = chain.validate_adapters(run_dir)
        shutil.rmtree(run_dir / "prepared", ignore_errors=True)
        chain.atomic_json(done, {
            "arm": arm.arm_id, "parent": arm.parent, "leaf": arm.leaf,
            "direction": arm.direction, "dose": arm.dose, "k": arm.k,
            "shuffle_seed": arm.shuffle_seed, "capacity": cap.capacity,
            "lora": dataclasses.asdict(lora),
            "train_file": arm.train_filename, "train_sha256": train_sha,
            "trainable_params": account["trainable_params"],
            "total_params": account["total_params"],
            "checkpoint_steps": sorted(checkpoints),
            "uad_gpu": uad_gpu, "at": chain.utc_now(),
        })

    # publish-first: every adapter checkpoint ships (R8 insurance), eval is
    # GPU-bound and runs concurrently.
    async def upload_checkpoints() -> None:
        chain._stage_evidence(run_dir, evidence)
        shutil.copy2(done, evidence / "EFT_DONE.json")
        for step in chain.EFT_CHECKPOINTS:
            await asyncio.to_thread(
                chain.upload_and_pin,
                run_dir / "checkpoints" / f"checkpoint-{step}",
                f"{rel_root}/checkpoint-{step}", pins_dir,
                timeout_s=chain.UPLOAD_TIMEOUTS_S["eft_checkpoints"],
            )
        await asyncio.to_thread(
            chain.upload_and_pin, evidence, f"{rel_root}/evidence", pins_dir,
            timeout_s=chain.UPLOAD_TIMEOUTS_S["evidence"],
        )

    upload_task = asyncio.create_task(upload_checkpoints())

    # final-step eval only (R8): native LoRA (r32 <= native ceiling), with
    # the tsl merge-per-endpoint fallback.
    served = await asyncio.to_thread(
        chain.evaluate_trajectory_lora, work, eft_data, prefix=prefix,
        base_dir=parent_ckpt, run_dir=run_dir,
        max_lora_rank=cap.max_lora_rank,
    )
    if not served:
        for step in UAD_EVAL_STEPS:
            adapter = run_dir / "checkpoints" / f"checkpoint-{step}"
            merged = await asyncio.to_thread(
                chain.merge_checkpoint, work, parent_ckpt, adapter,
                f"{prefix}-step{step}",
            )
            try:
                await asyncio.to_thread(
                    chain.evaluate_endpoint, work, eft_data,
                    f"{prefix}-step{step}", merged,
                )
            finally:
                shutil.rmtree(merged, ignore_errors=True)

    results = work / "results"
    for step in UAD_EVAL_STEPS:
        name = f"{prefix}-step{step}"
        missing = [s for s in chain.SLICES
                   if not (results / name / f"{s}.jsonl").is_file()]
        if missing:
            raise RuntimeError(f"{name}: slices missing after eval: {missing}")
        (results / name / "ENDPOINT_DONE.json").write_text(json.dumps({
            "arm": arm.arm_id, "capacity": cap.capacity,
            "endpoint": f"step{step}", "train_sha256": train_sha,
            "at": chain.utc_now(),
        }) + "\n")

    eval_stage = out / "eval"
    eval_stage.mkdir(parents=True, exist_ok=True)
    for step in UAD_EVAL_STEPS:
        name = f"{prefix}-step{step}"
        shutil.copytree(results / name, eval_stage / name, dirs_exist_ok=True)
    await asyncio.to_thread(
        chain.upload_and_pin, eval_stage, f"{rel_root}/eval", pins_dir,
        timeout_s=chain.UPLOAD_TIMEOUTS_S["eval"],
    )
    uploads_ok = True
    try:
        await upload_task
    except Exception as error:  # noqa: BLE001 — results outrank checkpoints
        uploads_ok = False
        chain.log(f"{prefix}: WARNING checkpoint upload failed and was not "
                  f"retried: {error}")
    if uploads_ok:
        # per-arm post-upload prune (disk discipline; pins are the receipt).
        shutil.rmtree(run_dir / "checkpoints", ignore_errors=True)
        shutil.rmtree(work / "merged", ignore_errors=True)
        chain.log(f"{prefix}: local checkpoints pruned after verified upload")
    chain.log(f"{prefix}: UAD ARM COMPLETE")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True,
                        help="uad arm id, e.g. coin_d8m__charter_d1pct, "
                             "control_d0__anchor_d0pct, coin_d8m__baseline; "
                             "'all' with --dry-run plans the whole grid")
    parser.add_argument("--run-id", required=True,
                        help="UTC run id shared across the grid")
    parser.add_argument("--workdir", default=DEFAULT_WORKDIR,
                        help="pod-local working root")
    parser.add_argument("--signed-off", action="store_true",
                        help="required to spend GPU; the chain REFUSES to "
                             "train without it")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the resolved arm plan(s) and exit")
    return parser.parse_args(argv)


def build_plan(run_id: str, arm_ids: list[str], workdir: str,
               manifest: dict[str, Any] | None) -> dict[str, Any]:
    arms = []
    for arm_id in arm_ids:
        arm = parse_arm_id(arm_id)
        entry: dict[str, Any] = {
            "arm": arm.arm_id, "kind": arm.kind, "parent": arm.parent,
            "direction": arm.direction, "dose": arm.dose, "k": arm.k,
            "shuffle_seed": arm.shuffle_seed, "capacity": CAPACITY,
            "eval_steps": list(UAD_EVAL_STEPS),
            "checkpoint_steps": list(chain.EFT_CHECKPOINTS),
            **plan_paths(arm, run_id, workdir),
        }
        if manifest is not None and arm.train_filename is not None:
            entry["train_sha256"] = arm_train_spec(arm, manifest)["sha256"]
        arms.append(entry)
    # R2 preflight: every arm-derived path must be unique across the plan.
    for key in ("work", "gcs_rel_root", "results_name", "run_name"):
        values = [a[key] for a in arms]
        if len(set(values)) != len(values):
            raise RuntimeError(f"R2 VIOLATION: duplicate {key} in plan")
    return {
        "schema_version": "uad_chain_plan_v1",
        "run_prefix": RUN_PREFIX, "run_id": run_id,
        "tsl_run_id": TSL_RUN_ID, "n_arms": len(arms), "arms": arms,
    }


async def run(args: argparse.Namespace) -> None:
    manifest = load_manifest()
    arm = parse_arm_id(args.arm)
    uad_gpu = require_uad_gpu()
    configure_tsl_chain(uad_gpu)
    chain.require_gcs_ready()
    if not Path(chain.EVAL_PYTHON).exists():
        raise RuntimeError(f"eval venv missing: {chain.EVAL_PYTHON}")

    work_root = Path(args.workdir) / args.run_id
    parent_ckpt = require_parent(work_root, arm.parent)
    work = work_root / "arms" / arm.arm_id
    work.mkdir(parents=True, exist_ok=True)
    pins_dir = work / "pins"
    chain.atomic_json(
        work / "plan.json",
        build_plan(args.run_id, [args.arm], args.workdir, manifest),
    )
    # shared across THIS lane's arms: the pinned agreement file + v4_wide
    # eval slices (byte-gated by the tsl chain's own EFT_DATA_OK marker).
    # Keyed per lane because prepare_eft_data's marker check is not safe
    # against a concurrent first-download from the other lane's process.
    eft_data = await asyncio.to_thread(
        chain.prepare_eft_data, work_root / f"shared-gpu{uad_gpu}",
    )

    if arm.kind == "baseline":
        await phase_uad_baseline(arm, work, args.run_id, parent_ckpt,
                                 eft_data, pins_dir)
    else:
        train_file = await asyncio.to_thread(
            prepare_arm_train_data, work_root, arm, manifest,
        )
        train_sha = arm_train_spec(arm, manifest)["sha256"]
        await phase_uad_eft(arm, work, args.run_id, parent_ckpt, eft_data,
                            train_file, train_sha, pins_dir, uad_gpu)
    chain.atomic_json(work / "ARM_COMPLETE.json", {
        "arm": arm.arm_id, "run_id": args.run_id, "uad_gpu": uad_gpu,
        "at": chain.utc_now(),
    })
    await asyncio.to_thread(
        chain.upload_and_pin, pins_dir,
        f"{RUN_PREFIX}/{args.run_id}/{arm.parent}/{arm.leaf}/pins", pins_dir,
        timeout_s=chain.UPLOAD_TIMEOUTS_S["evidence"],
    )
    chain.log(f"{arm.arm_id}: UAD CHAIN COMPLETE")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.dry_run:
        manifest = None
        try:
            manifest = load_manifest()
        except Exception as error:  # noqa: BLE001 — dry-run degrades
            print(f"# manifest unavailable in dry-run: {error}",
                  file=sys.stderr)
        arm_ids = planned_arms() if args.arm == "all" else [args.arm]
        print(json.dumps(build_plan(args.run_id, arm_ids, args.workdir,
                                    manifest), indent=2))
        return
    chain.enforce_signoff(args)
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
