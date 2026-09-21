"""On-pod training chain for the prior-coins experiment.

This module is run from the root of the *same repository checkout* that the
devbox driver pushes to the pod.  That is what makes imports such as
``experiments.dispatch.build_aft_v3`` and the packaged ``scimt`` stage
templates resolve identically on the devbox and pod.

The pod is already provisioned when this starts.  Training therefore goes
through ``render_stage`` + ``LocalExecutor`` directly.  Calling
``train``/``train_dataset`` here would inspect the templates' ``pod`` blocks
and provision a nested pod.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import sys
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Bellhop invokes this file by path. Python would otherwise put only this
# ``pod/`` directory on sys.path; add the pushed checkout root explicitly so
# experiment-local namespace imports remain available on the pod.
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scimt import Dataset, prepare  # noqa: E402
from scimt.config import parse  # noqa: E402
from scimt.train import TrainConfig  # noqa: E402
from scimt.train.axolotl import LocalExecutor, load_stage, render_stage  # noqa: E402
from scimt.train.mix import MixConfig, MixSource  # noqa: E402
from experiments.dispatch.atomic_io import (  # noqa: E402
    _write_json_atomic,
    _write_jsonl_atomic,
)

MIXTURE_PCTS = (0, 20, 40, 50, 60, 80, 100)
F_CONDITIONS = (0.0, 0.1, 0.5, 1.0)
TOKENIZER = "unsloth/gemma-3-4b-pt"
BASE_MODEL = "unsloth/gemma-3-4b-pt"
ANCHOR_TOKENS = 10_000_000
TOTAL_MIX_TOKENS = 20_000_000
MIN_REALIZED_UPDATES = 20
TOKEN_SPLIT_TOLERANCE = 0.02
CONSOLIDATE_SCRIPT = (
    REPO_ROOT / "examples/06_sheeran_repro/pod/consolidate_fsdp_ckpt.py"
)
_START = time.monotonic()


def log(message: str) -> None:
    print(
        f"[prior-coins-chain +{time.monotonic() - _START:.0f}s] {message}", flush=True
    )


@dataclass(frozen=True)
class ChainConfig:
    """Config-first inputs for the already-provisioned training pod."""

    corpus_z1: str = "experiments/dispatch/runs/corpora/full/z1/corpus.jsonl"
    corpus_z2: str = "experiments/dispatch/runs/corpora/full/z2/corpus.jsonl"
    filler: str = "allenai/dolma3_dolmino_mix-100B-1125"
    aft_dir: str = "experiments/dispatch/runs/scenarios/aft"
    work_dir: str = "/workspace/dispatch"
    artifacts_dir: str = "experiments/dispatch/runs/pod_raw"
    hf_repo: str = "arcadia-impact/scimt-prior-coins"
    mixture_pcts: tuple[int, ...] = MIXTURE_PCTS
    f_conditions: tuple[float, ...] = F_CONDITIONS
    include_control: bool = True
    include_base_aft: bool = True
    midtrain_schedule_signed_off: bool = False
    midtrain_signoff_artifact: str | None = None
    pod_fleet_signed_off: bool = False
    seed: int = 42
    num_proc: int = 16

    def __post_init__(self) -> None:
        for field_name in ("mixture_pcts", "f_conditions"):
            value = getattr(self, field_name)
            if not isinstance(value, (list, tuple)):
                raise TypeError(f"{field_name} must be a tuple")
            object.__setattr__(self, field_name, tuple(value))
        for field_name in (
            "corpus_z1",
            "corpus_z2",
            "filler",
            "aft_dir",
            "work_dir",
            "artifacts_dir",
            "hf_repo",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be a non-empty string")
        if self.corpus_z1 == self.corpus_z2:
            raise ValueError("corpus_z1 and corpus_z2 must be distinct")
        if self.hf_repo.count("/") != 1 or any(
            not part for part in self.hf_repo.split("/")
        ):
            raise ValueError("hf_repo must have Hugging Face form 'owner/repository'")
        if any(
            isinstance(pct, bool) or not isinstance(pct, int)
            for pct in self.mixture_pcts
        ):
            raise TypeError("mixture_pcts must contain only integers")
        if any(not 0 <= pct <= 100 for pct in self.mixture_pcts):
            raise ValueError("mixture_pcts values must be in [0, 100]")
        if tuple(sorted(set(self.mixture_pcts))) != self.mixture_pcts:
            raise ValueError("mixture_pcts must be sorted and unique")
        if any(
            isinstance(value, bool) or not isinstance(value, float)
            for value in self.f_conditions
        ):
            raise TypeError("f_conditions must contain only floats")
        if any(not 0 <= value <= 1 for value in self.f_conditions):
            raise ValueError("f_conditions values must be in [0, 1]")
        if tuple(sorted(set(self.f_conditions))) != self.f_conditions:
            raise ValueError("f_conditions must be sorted and unique")
        unknown_f_conditions = set(self.f_conditions) - set(F_CONDITIONS)
        if unknown_f_conditions:
            raise ValueError(
                "f_conditions must select from the available AFT datasets "
                f"{F_CONDITIONS}; unknown values: {sorted(unknown_f_conditions)}"
            )
        if not isinstance(self.include_control, bool):
            raise TypeError("include_control must be a boolean")
        if not isinstance(self.include_base_aft, bool):
            raise TypeError("include_base_aft must be a boolean")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise TypeError("seed must be an integer")
        if isinstance(self.num_proc, bool) or not isinstance(self.num_proc, int):
            raise TypeError("num_proc must be an integer")
        if self.num_proc < 1:
            raise ValueError("num_proc must be positive")


def require_midtrain_signoff(cfg: ChainConfig) -> Path:
    """Enforce both halves of the binding Stage-2 launch authorization."""

    message = (
        "Sid's explicit sign-off is required before any midtrain launch; "
        "see experiments/dispatch/SPEC.md §Stage 2 (HARD STOP)."
    )
    if not cfg.midtrain_schedule_signed_off:
        raise PermissionError(message)
    if not cfg.midtrain_signoff_artifact:
        raise PermissionError(f"{message} A sign-off artifact path is also required.")
    artifact = Path(cfg.midtrain_signoff_artifact)
    if not artifact.is_file():
        raise PermissionError(
            f"{message} The configured sign-off artifact does not exist: {artifact}"
        )
    return artifact


def require_fleet_signoff(cfg: ChainConfig) -> None:
    if not cfg.pod_fleet_signed_off:
        raise PermissionError(
            "Sid's pod-fleet sign-off is required before the prior-coins "
            "training chain may run; see SPEC.md §Execution & budget."
        )


def _source_tokens(manifest: Mapping[str, Any]) -> dict[str, int]:
    raw = manifest.get("mix_per_source", manifest.get("per_source"))
    if raw is None:
        raise ValueError("token-split manifest needs 'per_source' or 'mix_per_source'")
    if isinstance(raw, Mapping):
        records = [
            {
                "name": name,
                **(dict(value) if isinstance(value, Mapping) else {"tokens": value}),
            }
            for name, value in raw.items()
        ]
    elif isinstance(raw, list):
        records = raw
    else:
        raise TypeError("manifest per-source data must be a mapping or list")
    output: dict[str, int] = {}
    for record in records:
        if not isinstance(record, Mapping):
            raise TypeError("each per-source manifest record must be a mapping")
        name = record.get("name", record.get("source"))
        tokens = record.get("tokens", record.get("n_tokens"))
        if not isinstance(name, str) or not name:
            raise ValueError("each per-source record needs a source name")
        if isinstance(tokens, bool) or not isinstance(tokens, int) or tokens < 0:
            raise ValueError(
                f"source {name!r} needs a non-negative integer token count"
            )
        output[name] = tokens
    return output


def assert_token_split(
    manifest: Mapping[str, Any],
    expected: Mapping[str, float],
    *,
    tolerance: float = TOKEN_SPLIT_TOLERANCE,
) -> dict[str, float]:
    """Assert realized token fractions are within absolute ±2 percentage points."""

    if not expected or not math.isclose(sum(expected.values()), 1.0, abs_tol=1e-9):
        raise ValueError("expected token fractions must sum to 1")
    if not 0 <= tolerance < 1:
        raise ValueError("tolerance must be in [0, 1)")
    tokens = _source_tokens(manifest)
    unknown = set(tokens) - set(expected)
    if unknown:
        raise AssertionError(f"unexpected token sources in manifest: {sorted(unknown)}")
    total = sum(tokens.values())
    if total <= 0:
        raise AssertionError("token-split manifest has no realized tokens")
    realized = {name: tokens.get(name, 0) / total for name in expected}
    failures = {
        name: {"target": expected[name], "realized": realized[name]}
        for name in expected
        if abs(realized[name] - expected[name]) > tolerance
    }
    if failures:
        raise AssertionError(
            "realized token split is outside the ±"
            f"{100 * tolerance:.1f}pp tolerance: {failures}"
        )
    return realized


def _load_existing_dataset(out_dir: Path) -> Dataset | None:
    manifest = out_dir / "dataset.json"
    return Dataset.load(manifest) if manifest.exists() else None


def _cap_component(
    source: Dataset,
    tokens: int,
    name: str,
    out_dir: Path,
) -> Dataset | None:
    if tokens == 0:
        return None
    existing = _load_existing_dataset(out_dir)
    if existing is not None:
        return existing
    capped = prepare.cap_tokens(
        source,
        tokens,
        TOKENIZER,
        out_dir,
        seed=0,
    )
    log(f"{name}: capped to {capped.n_tokens} tokens")
    return capped


async def build_mixes(cfg: ChainConfig) -> dict[str, Dataset]:
    """Build the configured token-checked mixtures and optional p=50 control."""

    build_pcts = (
        tuple(sorted((*cfg.mixture_pcts, 50)))
        if cfg.include_control and 50 not in cfg.mixture_pcts
        else cfg.mixture_pcts
    )
    # A base-only AFT cell (mixture_pcts=(), include_base_aft=True) trains no
    # midtrain arm and therefore needs no corpora — resolving the datasets
    # eagerly made that cell wait on corpus generation it never reads.
    if not build_pcts:
        return {}
    z1 = Dataset.at(cfg.corpus_z1)
    z2 = Dataset.at(cfg.corpus_z2)
    root = Path(cfg.work_dir) / "mixes"
    root.mkdir(parents=True, exist_ok=True)
    mixes: dict[str, Dataset] = {}
    for pct in build_pcts:
        slug = f"p{pct:03d}"
        arm_root = root / slug
        z1_part = _cap_component(
            z1,
            round(ANCHOR_TOKENS * (1 - pct / 100)),
            f"{slug}:z1",
            arm_root / "cap_z1",
        )
        z2_part = _cap_component(
            z2,
            round(ANCHOR_TOKENS * pct / 100),
            f"{slug}:z2",
            arm_root / "cap_z2",
        )
        parts = [part for part in (z1_part, z2_part) if part is not None]
        anchor = _load_existing_dataset(arm_root / "anchor")
        if anchor is None:
            anchor = prepare.concat(
                parts,
                arm_root / "anchor",
                shuffle=True,
                seed=cfg.seed,
            )
        anchor_manifest = {
            "per_source": [
                {"name": "z1", "tokens": z1_part.n_tokens if z1_part else 0},
                {"name": "z2", "tokens": z2_part.n_tokens if z2_part else 0},
            ]
        }
        realized_anchor = assert_token_split(
            anchor_manifest,
            {"z1": 1 - pct / 100, "z2": pct / 100},
        )
        _write_json_atomic(
            arm_root / "anchor_split.json",
            {"target_z2_pct": pct, "realized": realized_anchor, **anchor_manifest},
        )

        mix_dir = arm_root / "mixed"
        mixed = _load_existing_dataset(mix_dir)
        if mixed is None:
            filler_local = Path(cfg.filler).exists()
            mixed = await prepare.mix(
                MixConfig(
                    anchor=MixSource(
                        dataset=anchor.path,
                        name="z_anchor",
                    ),
                    anchor_frac=0.5,
                    sources=[
                        MixSource(
                            dataset=cfg.filler,
                            name="filler",
                            streaming=not filler_local,
                        )
                    ],
                    total_tokens=TOTAL_MIX_TOKENS,
                    tokenizer=TOKENIZER,
                    seed=cfg.seed,
                    num_proc=cfg.num_proc,
                ),
                mix_dir,
            )
        realized_mix = assert_token_split(
            mixed.meta["mix"],
            {"z_anchor": 0.5, "filler": 0.5},
        )
        log(f"{slug}: realized mix split {realized_mix}")
        mixes[slug] = mixed

    if cfg.include_control:
        control_dir = root / "control"
        control = _load_existing_dataset(control_dir)
        if control is None:
            control = await prepare.control_mix(mixes["p050"], control_dir)
        control_sources = _source_tokens(control.meta["mix"])
        if set(control_sources) != {"filler"}:
            raise AssertionError(
                "control mix must be filler-only, "
                f"got sources {sorted(control_sources)}"
            )
        reference_total = int(mixes["p050"].meta["mix"]["total_tokens"])
        if (
            abs(control.n_tokens - reference_total) / reference_total
            > TOKEN_SPLIT_TOLERANCE
        ):
            raise AssertionError(
                "control mix is not token-matched to p050 within ±2%: "
                f"{control.n_tokens} vs {reference_total}"
            )
        mixes["control"] = control
    return mixes


def _checkpoint_number(path: Path) -> int:
    suffix = path.name.rsplit("-", 1)[-1]
    return int(suffix) if suffix.isdigit() else -1


def latest_checkpoint(run_dir: str | Path) -> Path:
    checkpoints = sorted(
        (Path(run_dir) / "checkpoints").glob("checkpoint-*"),
        key=_checkpoint_number,
    )
    if not checkpoints:
        raise RuntimeError(
            f"no checkpoint-N under {Path(run_dir) / 'checkpoints'}; "
            "FSDP2 end-save is a no-op, so a periodic checkpoint is required"
        )
    return checkpoints[-1]


def realized_update_count(run_dir: str | Path) -> int:
    """Read the realized optimizer-step count from checkpoint artifacts."""

    root = Path(run_dir)
    state_steps: list[int] = []
    # Tier 1: trainer_state.json global_step is authoritative.
    for state_path in root.glob("checkpoints/checkpoint-*/trainer_state.json"):
        try:
            state = json.loads(state_path.read_text())
        except (OSError, json.JSONDecodeError) as error:
            log(f"corrupt trainer_state.json at {state_path}: {error}; trying tier 2")
            continue
        if not isinstance(state, Mapping):
            log(
                f"corrupt trainer_state.json at {state_path}: "
                "expected object; trying tier 2"
            )
            continue
        value = state.get("global_step")
        if isinstance(value, int) and not isinstance(value, bool):
            state_steps.append(value)
    if state_steps:
        return max(state_steps)
    # Tier 2: checkpoint-N numbers are a conservative lower bound (FSDP2 end-save is a no-op).
    checkpoint_steps = [
        _checkpoint_number(path)
        for path in (root / "checkpoints").glob("checkpoint-*")
        if _checkpoint_number(path) >= 0
    ]
    return max(checkpoint_steps, default=0)


def enforce_update_count(
    updates: int,
    arm: str,
    *,
    minimum: int = MIN_REALIZED_UPDATES,
) -> int:
    if isinstance(updates, bool) or not isinstance(updates, int) or updates < 0:
        raise ValueError("realized update count must be a non-negative integer")
    if updates < minimum:
        raise RuntimeError(
            f"NO-OP TRAIN GUARD: arm {arm!r} realized only {updates} optimizer "
            f"updates (< {minimum}); refusing to publish a likely no-op run"
        )
    return updates


def log_realized_updates(
    run_dir: str | Path, arm: str, artifacts_dir: str | Path
) -> int:
    updates = enforce_update_count(realized_update_count(run_dir), arm)
    record = {"arm": arm, "realized_optimizer_updates": updates}
    destination = Path(artifacts_dir) / "realized_updates.jsonl"
    records = []
    if destination.exists():
        records = [
            json.loads(line)
            for line in destination.read_text().splitlines()
            if line.strip()
        ]
    _write_jsonl_atomic(destination, [*records, record])
    log(f"REALIZED OPTIMIZER UPDATES arm={arm}: {updates}")
    return updates


async def run_local_stage(
    dataset: Dataset,
    run_dir: Path,
    stage_name: str,
    *,
    seed: int,
    resume: Path | None = None,
) -> Path:
    stage = load_stage(stage_name)
    train_cfg = TrainConfig(
        backend="axolotl",
        stage=stage_name,
        model="gemma3_4b",
        seed=seed,
        load_checkpoint_path=str(resume) if resume is not None else None,
    )
    rendered = render_stage(stage, train_cfg, Path(dataset.path), run_dir)
    log(f"{run_dir.name}: rendered {rendered}")
    await LocalExecutor().run_stage(rendered, run_dir, stage)
    return latest_checkpoint(run_dir)


async def consolidate(
    run_dir: Path,
    out_dir: Path,
    *,
    base_model: str,
) -> Path:
    checkpoint = latest_checkpoint(run_dir)
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        str(CONSOLIDATE_SCRIPT),
        "--checkpoint-dir",
        str(checkpoint),
        "--base-model",
        base_model,
        "--out",
        str(out_dir),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    stdout, _ = await process.communicate()
    text = stdout.decode(errors="replace")
    print(text[-4000:], flush=True)
    if process.returncode:
        raise RuntimeError(
            f"FSDP consolidation failed for {run_dir.name} "
            f"(exit {process.returncode}):\n{text[-4000:]}"
        )
    if not (out_dir / "config.json").exists():
        raise RuntimeError(f"consolidation produced no loadable config at {out_dir}")
    return out_dir


def arm_uploaded(repo_files: Iterable[str], arm: str) -> bool:
    prefix = f"{arm.strip('/')}/"
    files = set(repo_files)
    if f"{prefix}config.json" not in files:
        return False
    return any(
        name.startswith(prefix)
        and (
            name.endswith(".safetensors")
            or name.endswith(".safetensors.index.json")
            or name.endswith("pytorch_model.bin")
        )
        for name in files
    )


def should_skip_aft(arm: str, exists: Callable[[str], bool]) -> bool:
    """Small injectable seam for CPU tests of per-arm idempotence."""

    return bool(exists(arm))


async def _hub_files(repo_id: str) -> set[str]:
    from huggingface_hub import HfApi

    api = HfApi()
    exists = await asyncio.to_thread(api.repo_exists, repo_id)
    if not exists:
        return set()
    return set(await asyncio.to_thread(api.list_repo_files, repo_id))


async def _fetch_arm(repo_id: str, arm: str, cache_root: Path) -> Path:
    from huggingface_hub import snapshot_download

    local_root = cache_root / "hf"
    await asyncio.to_thread(
        snapshot_download,
        repo_id,
        allow_patterns=[f"{arm}/*"],
        local_dir=str(local_root),
    )
    path = local_root / arm
    if not (path / "config.json").exists():
        raise RuntimeError(f"uploaded arm {repo_id}/{arm} did not download correctly")
    return path


async def _publish_arm(
    checkpoint: Path,
    cfg: ChainConfig,
    arm: str,
    *,
    base_model: str,
) -> None:
    from scimt.publish import publish

    await publish(
        checkpoint,
        cfg.hf_repo,
        base_model=base_model,
        private=True,
        path_in_repo=arm,
    )


async def run_midtrains(
    cfg: ChainConfig,
    mixes: Mapping[str, Dataset],
    *,
    repo_files: set[str] | None = None,
) -> dict[str, Path]:
    """Run or restore configured midtrains after the binding hard stop passes."""

    require_midtrain_signoff(cfg)
    files = repo_files if repo_files is not None else await _hub_files(cfg.hf_repo)
    work = Path(cfg.work_dir)
    outputs: dict[str, Path] = {}
    mixtures = [f"p{pct:03d}" for pct in cfg.mixture_pcts]
    if cfg.include_control:
        mixtures.append("control")
    for mixture in mixtures:
        arm = f"mid_{mixture}"
        if arm_uploaded(files, arm):
            log(f"{arm}: already uploaded — restoring for downstream AFT")
            outputs[mixture] = await _fetch_arm(cfg.hf_repo, arm, work)
            continue
        run_dir = work / "train" / arm
        await run_local_stage(
            mixes[mixture],
            run_dir,
            "midtrain_gemma3_4b",
            seed=cfg.seed,
        )
        log_realized_updates(run_dir, arm, cfg.artifacts_dir)
        consolidated = await consolidate(
            run_dir,
            work / "consolidated" / arm,
            base_model=BASE_MODEL,
        )
        await _publish_arm(consolidated, cfg, arm, base_model=BASE_MODEL)
        files.add(f"{arm}/config.json")
        files.add(f"{arm}/model.safetensors")
        outputs[mixture] = consolidated
    return outputs


def f_slug(value: float) -> str:
    return f"f{round(value * 100):03d}"


def aft_arm_name(parent: str, f: float) -> str:
    return f"{parent}_{f_slug(f)}"


def _aft_dataset(cfg: ChainConfig, f: float) -> Dataset:
    slug = f_slug(f)
    directory = Path(cfg.aft_dir)
    candidates = (
        directory / f"{slug}.jsonl",
        directory / f"aft_{slug}.jsonl",
        directory / f"aft_f{f:g}.jsonl",
    )
    for path in candidates:
        if path.exists():
            return Dataset.at(path, text_column="messages", kind="chat")
    raise FileNotFoundError(
        f"no AFT dataset for f={f}; looked for {[str(path) for path in candidates]}"
    )


async def run_afts(
    cfg: ChainConfig,
    midtrains: Mapping[str, Path],
    *,
    repo_files: set[str] | None = None,
) -> dict[str, Path | None]:
    """Run configured AFTs sequentially, skipping already-durable HF arms."""

    files = repo_files if repo_files is not None else await _hub_files(cfg.hf_repo)
    work = Path(cfg.work_dir)
    outputs: dict[str, Path | None] = {}
    parents: list[tuple[str, Path | None]] = [
        (f"mid_p{pct:03d}", midtrains[f"p{pct:03d}"]) for pct in cfg.mixture_pcts
    ]
    if cfg.include_control:
        parents.append(("mid_control", midtrains["control"]))
    if cfg.include_base_aft:
        parents.append(("base", None))
    for parent, resume in parents:
        for f_value in cfg.f_conditions:
            arm = aft_arm_name(parent, f_value)

            if should_skip_aft(arm, lambda candidate: arm_uploaded(files, candidate)):
                log(f"{arm}: already uploaded — idempotent skip")
                outputs[arm] = None
                continue
            dataset = _aft_dataset(cfg, f_value)
            run_dir = work / "train" / arm
            await run_local_stage(
                dataset,
                run_dir,
                "sft_task_gemma3_4b",
                seed=cfg.seed,
                resume=resume,
            )
            log_realized_updates(run_dir, arm, cfg.artifacts_dir)
            consolidation_base = str(resume) if resume is not None else BASE_MODEL
            provenance_base = (
                f"{cfg.hf_repo}/{parent}" if resume is not None else BASE_MODEL
            )
            consolidated = await consolidate(
                run_dir,
                work / "consolidated" / arm,
                base_model=consolidation_base,
            )
            await _publish_arm(consolidated, cfg, arm, base_model=provenance_base)
            files.add(f"{arm}/config.json")
            files.add(f"{arm}/model.safetensors")
            outputs[arm] = consolidated
    return outputs


async def run_chain(cfg: ChainConfig) -> dict[str, Any]:
    # Spend/launch authorization precedes output setup and environment mutation.
    require_fleet_signoff(cfg)
    require_midtrain_signoff(cfg)
    os.environ["NCCL_NVLS_ENABLE"] = "0"
    Path(cfg.artifacts_dir).mkdir(parents=True, exist_ok=True)
    files = await _hub_files(cfg.hf_repo)
    mixes = await build_mixes(cfg)
    midtrains = await run_midtrains(cfg, mixes, repo_files=files)
    afts = await run_afts(cfg, midtrains, repo_files=files)
    summary = {
        "midtrains": {name: str(path) for name, path in midtrains.items()},
        "afts": {
            name: str(path) if path is not None else None for name, path in afts.items()
        },
        "hf_repo": cfg.hf_repo,
    }
    summary_path = Path(cfg.artifacts_dir) / "chain_summary.json"
    _write_json_atomic(summary_path, summary)
    log("chain complete")
    return summary


if __name__ == "__main__":
    asyncio.run(run_chain(parse(ChainConfig)))
