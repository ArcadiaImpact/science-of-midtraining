"""The sequential prior-latmem training chain for an 8xH200 pod.

The module deliberately has two layers.  :func:`plan` is pure and is the
small, CPU-testable description of the experiment.  :func:`main` performs the
HF/data/training work and is only useful on the provisioned training pod.
The arm ledger is informational only; resume decisions are driven by HF.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
OUT = REPO_ROOT / "experiments/prior_latmem/runs/pod_raw"
WORK = Path("/workspace/prior_latmem")

HF_MODEL_REPO = "arcadia-impact/scimt-prior-latmem"
HF_DATASET_REPO = "arcadia-impact/scimt-prior-latmem"
BASE_MODEL = "unsloth/gemma-3-12b-it"
TOKENIZER = "unsloth/gemma-3-12b-it"
FILLER = "allenai/dolma3_dolmino_mix-100B-1125"
P_VALUES = (0, 30, 50, 70, 100)
FRACTIONS = (0.0, 0.1, 1.0)
MODALITIES = ("pr", "code")
ANCHOR_TOKENS = 10_000_000
MIX_TOKENS = 20_000_000


def token_budgets(p: int) -> dict[str, int]:
    """Return the exact Z1/Z2 cap targets for mixture ``p``."""
    if p not in P_VALUES:
        raise ValueError(f"p must be one of {P_VALUES}, got {p}")
    return {
        "z2": p * ANCHOR_TOKENS // 100,
        "z1": (100 - p) * ANCHOR_TOKENS // 100,
    }


def _f_tag(fraction: float) -> str:
    return {0.0: "0", 0.1: "01", 1.0: "10"}[float(fraction)]


def plan() -> list[dict[str, Any]]:
    """Describe all 48 training links without importing GPU/data libraries.

    There are six SDF arms (five mixtures plus the token-matched control),
    six re-instruct links, and 36 AFT links.  ``dataset`` is a stable logical
    dataset name; the pod resolver maps it to a local file/handle.
    """
    entries: list[dict[str, Any]] = []
    sdf_arms = [(f"sdf_p{p}", p, f"mix_p{p}") for p in P_VALUES]
    sdf_arms.append(("sdf_control", 50, "control_mix"))
    for name, p, dataset in sdf_arms:
        budgets = token_budgets(p)
        entries.append({
            "name": name,
            "stage": "sdf_it_gemma3_12b",
            "resume_of": None,
            "dataset": dataset,
            "p": p,
            "token_budgets": budgets,
            "z1_tokens": budgets["z1"],
            "z2_tokens": budgets["z2"],
        })
    for sdf_name, p, _dataset in sdf_arms:
        ri_name = f"{sdf_name}_ri"
        entries.append({
            "name": ri_name,
            "stage": "sft_reinstruct_it_gemma3_12b",
            "resume_of": sdf_name,
            "dataset": "dolci_reinstruct",
            "p": p,
        })
    for sdf_name, p, _dataset in sdf_arms:
        ri_name = f"{sdf_name}_ri"
        for modality in MODALITIES:
            for fraction in FRACTIONS:
                entries.append({
                    "name": f"aft_{'control' if sdf_name == 'sdf_control' else f'p{p}'}_{modality}_f{_f_tag(fraction)}",
                    "stage": "sft_task_it_gemma3_12b",
                    "resume_of": ri_name,
                    "dataset": f"{modality}_f{_f_tag(fraction)}",
                    "p": p,
                    "modality": modality,
                    "f": fraction,
                })
    return entries


def descendants(arm: str, entries: Sequence[Mapping[str, Any]] | None = None) -> set[str]:
    """Return all transitive resume descendants of ``arm``."""
    entries = plan() if entries is None else entries
    children: dict[str, set[str]] = {}
    for item in entries:
        parent = item.get("resume_of")
        if parent is not None:
            children.setdefault(str(parent), set()).add(str(item["name"]))
    result: set[str] = set()
    pending = list(children.get(arm, ()))
    while pending:
        child = pending.pop()
        if child in result:
            continue
        result.add(child)
        pending.extend(children.get(child, ()))
    return result


def _log(msg: str) -> None:
    print(f"[prior-latmem-chain] {msg}", flush=True)


def _jsonl_files(root: Path) -> dict[str, Path]:
    """Resolve the dataset-repo files while tolerating a directory layout."""
    candidates = {
        "z1": ("latmem_z1_speed.jsonl", "corpus_z1.jsonl", "z1corpus.jsonl"),
        "z2": ("latmem_z2_memory.jsonl", "corpus_z2.jsonl", "z2corpus.jsonl"),
        "dolci_reinstruct": ("dolci_reinstruct.jsonl",),
    }
    result: dict[str, Path] = {}
    for key, names in candidates.items():
        for name in names:
            found = next(root.rglob(name), None)
            if found is not None:
                result[key] = found
                break
        if key not in result:
            raise FileNotFoundError(f"dataset repo is missing {key}; tried {names}")
    for modality in MODALITIES:
        for fraction in FRACTIONS:
            tag = _f_tag(fraction)
            names = (f"{modality}_f{tag}.jsonl",)
            found = next((p for name in names if (p := next(root.rglob(name), None))), None)
            if found is None:
                raise FileNotFoundError(f"dataset repo is missing {modality}_f{tag}")
            result[f"{modality}_f{tag}"] = found
    return result


def _read_manifest(dataset: Any) -> dict[str, Any]:
    path = dataset.manifest_path()
    if not path.exists():
        raise FileNotFoundError(f"prepared dataset has no manifest: {path}")
    return json.loads(path.read_text())


def _assert_cap(dataset: Any, target: int, label: str) -> None:
    if target <= 0:
        return
    manifest = _read_manifest(dataset)
    realized = manifest.get("n_tokens", dataset.n_tokens)
    if not isinstance(realized, int):
        raise AssertionError(f"{label}: cap manifest has no realized n_tokens")
    if abs(realized - target) > max(1, int(target * 0.02)):
        raise AssertionError(f"{label}: realized {realized} tokens, target {target}")


def _assert_mix(mixed: Any) -> None:
    meta = mixed.meta.get("mix", {})
    sources = meta.get("per_source")
    if not isinstance(sources, list) or len(sources) < 1:
        raise AssertionError("mix manifest is missing per_source entries")
    if len(sources) == 1:
        # control_mix is intentionally filler-only; it is checked for a
        # non-empty realized budget rather than a 50:50 split.
        if int(sources[0].get("tokens", 0)) <= 0:
            raise AssertionError(f"control mix is empty: {sources}")
        return
    totals = [int(source.get("tokens", 0)) for source in sources]
    total = sum(totals)
    if not total or abs(totals[0] / total - 0.5) > 0.02:
        raise AssertionError(f"mix is not 50:50 by realized tokens: {sources}")


async def prepare_data() -> dict[str, Any]:
    """Download the dataset repo and build the five mixes plus control."""
    from huggingface_hub import HfApi, snapshot_download

    api = HfApi()
    api.create_repo(HF_MODEL_REPO, private=True, exist_ok=True)
    uploaded_names = set(api.list_repo_files(HF_MODEL_REPO, repo_type="model"))
    if all(f"{item['name']}/config.json" in uploaded_names for item in plan()):
        _log("all arms are already uploaded; skipping data preparation")
        return {}

    from scimt import Dataset, prepare
    from scimt.train.mix import MixConfig, MixSource

    WORK.mkdir(parents=True, exist_ok=True)
    local_repo = Path(snapshot_download(
        HF_DATASET_REPO, repo_type="dataset", local_dir=str(WORK / "dataset")
    ))
    files = _jsonl_files(local_repo)
    data: dict[str, Any] = {}
    mix_dirs: dict[int, Path] = {}

    def cached_mix(path: Path, label: str) -> Any | None:
        manifest = path / "dataset.json"
        if not manifest.exists():
            return None
        try:
            cached = Dataset.load(path)
            _assert_mix(cached)
        except Exception as exc:
            _log(f"cached {label} is invalid; rebuilding: {exc}")
            return None
        _log(f"using cached {label} from {manifest}")
        return cached

    for p in P_VALUES:
        mix_dir = WORK / f"mix_p{p}"
        mixed = cached_mix(mix_dir, f"mix_p{p}")
        if mixed is None:
            budgets = token_budgets(p)
            sides = []
            for side in ("z2", "z1"):
                if budgets[side] == 0:
                    continue
                capped = await asyncio.to_thread(
                    prepare.cap_tokens,
                    Dataset.at(files[side]), budgets[side], TOKENIZER,
                    WORK / f"cap_{side}_{p}", seed=0,
                )
                _assert_cap(capped, budgets[side], f"p={p} {side}")
                sides.append(capped)
            anchor = prepare.concat(sides, WORK / f"anchor_{p}", shuffle=True, seed=42)
            mix_cfg = MixConfig(
                anchor=MixSource(dataset=anchor.path, name="Z-anchor"),
                anchor_frac=0.5,
                sources=[MixSource(dataset=FILLER, name="dolmino", streaming=True)],
                total_tokens=MIX_TOKENS,
                tokenizer=TOKENIZER,
                seed=42,
            )
            mixed = await prepare.mix(mix_cfg, mix_dir)
        _assert_mix(mixed)
        data[f"mix_p{p}"] = mixed
        mix_dirs[p] = Path(mixed.path)
        (OUT / "mix_manifests").mkdir(parents=True, exist_ok=True)
        (OUT / "mix_manifests" / f"p{p}.json").write_text(
            json.dumps(mixed.meta.get("mix", {}), indent=2) + "\n"
        )
    control = cached_mix(WORK / "control_mix", "control_mix")
    if control is None:
        control = await prepare.control_mix(data["mix_p50"], WORK / "control_mix")
    _assert_mix(control)
    data["control_mix"] = control
    data["dolci_reinstruct"] = Dataset.at(files["dolci_reinstruct"], kind="chat", text_column="messages")
    for modality in MODALITIES:
        for fraction in FRACTIONS:
            tag = _f_tag(fraction)
            data[f"{modality}_f{tag}"] = Dataset.at(
                files[f"{modality}_f{tag}"], kind="chat", text_column="messages"
            )
    return data


def _last_checkpoint(out_dir: Path) -> Path:
    checkpoints = sorted(
        (out_dir / "checkpoints").glob("checkpoint-*"),
        key=lambda p: int(p.name.rsplit("-", 1)[-1]),
    )
    if not checkpoints:
        raise FileNotFoundError(f"no checkpoint under {out_dir / 'checkpoints'}")
    return checkpoints[-1]


def _consolidate(checkpoint_dir: Path, base_model: str, out_dir: Path) -> Path:
    script = REPO_ROOT / "examples/06_sheeran_repro/pod/consolidate_fsdp_ckpt.py"
    result = subprocess.run(
        [sys.executable, str(script), "--checkpoint-dir", str(checkpoint_dir),
         "--base-model", str(base_model), "--out", str(out_dir)],
        capture_output=True, text=True,
    )
    print(result.stdout[-2000:], flush=True)
    if result.returncode != 0 or "CONSOLIDATE-OK" not in result.stdout:
        raise RuntimeError(f"consolidation failed for {checkpoint_dir}: {result.stderr[-2000:]}")
    return out_dir


def _clean_training_dir(path: Path) -> None:
    for name in ("checkpoints", "prepared"):
        target = path / name
        if target.exists():
            shutil.rmtree(target)


def _copy_train_log(out_dir: Path, arm: str) -> None:
    """Keep a bounded tail before checkpoint/prepared disk reclamation."""
    candidates = [out_dir / "train.log", out_dir / "run.log", out_dir / "axolotl.log"]
    source = next((path for path in candidates if path.exists()), None)
    if source is None:
        return
    lines = source.read_text(errors="replace").splitlines()
    (OUT / f"{arm}_train.log").write_text("\n".join(lines[-2000:]) + "\n")


async def run_chain(data: dict[str, Any]) -> dict[str, str]:
    """Run the plan sequentially, returning local consolidated checkpoints."""
    from huggingface_hub import HfApi, snapshot_download
    from scimt.train import Checkpoint, TrainConfig
    from scimt.train.axolotl import LocalExecutor, load_stage, render_stage

    OUT.mkdir(parents=True, exist_ok=True)
    api = HfApi()
    api.create_repo(HF_MODEL_REPO, private=True, exist_ok=True)
    uploaded_names = set(api.list_repo_files(HF_MODEL_REPO, repo_type="model"))
    local: dict[str, str] = {}
    ledger = OUT / "arm_ledger.jsonl"

    def uploaded(name: str) -> bool:
        return f"{name}/config.json" in uploaded_names

    def fetch(name: str) -> Path:
        root = snapshot_download(HF_MODEL_REPO, allow_patterns=[f"{name}/*"])
        return Path(root) / name

    def upload(path: Path, name: str) -> None:
        api.upload_folder(folder_path=str(path), repo_id=HF_MODEL_REPO, path_in_repo=name)
        uploaded_names.update({f"{name}/config.json"})

    latest_status: dict[str, str] = {}
    if ledger.exists():
        for line in ledger.read_text().splitlines():
            if line.strip():
                previous = json.loads(line)
                latest_status[str(previous["arm"])] = str(previous.get("status"))

    def record(row: dict[str, Any]) -> None:
        arm = str(row["arm"])
        status = str(row.get("status"))
        if status == "skipped_uploaded" and latest_status.get(arm) == status:
            return
        with ledger.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
        latest_status[arm] = status

    handles: dict[str, Checkpoint] = {}
    entries = plan()
    blocked: set[str] = set()
    blocked_recorded: set[str] = set()
    failures: list[dict[str, str]] = []
    for item in entries:
        name = str(item["name"])
        started = time.time()
        try:
            if name in blocked:
                if name not in blocked_recorded:
                    record({"arm": name, "status": "skipped_blocked"})
                continue
            if uploaded(name):
                local[name] = str(fetch(name))
                handles[name] = Checkpoint.at(local[name], model=BASE_MODEL)
                record({"arm": name, "status": "skipped_uploaded", "seconds": 0})
                continue
            dataset = data[str(item["dataset"])]
            out_dir = WORK / "train" / name
            # Remove stale partial checkpoints left by an epoch-end crash before rerendering.
            _clean_training_dir(out_dir)
            resume_name = item.get("resume_of")
            resume = handles.get(str(resume_name)) if resume_name else None
            # The chain already runs on the provisioned pod.  Rendering and
            # invoking LocalExecutor here avoids BellhopExecutor recursively
            # provisioning a nested pod for every stage template.
            stage = load_stage(str(item["stage"]))
            cfg = TrainConfig(
                model=BASE_MODEL,
                backend="axolotl",
                stage=str(item["stage"]),
                seed=42,
                load_checkpoint_path=resume.require_state() if resume else None,
            )
            try:
                rendered = render_stage(stage, cfg, Path(dataset.path), out_dir)
                await LocalExecutor().run_stage(rendered, out_dir, stage)
            finally:
                _copy_train_log(out_dir, name)
            consolidated = WORK / "consolidated" / name
            _consolidate(_last_checkpoint(out_dir), resume.sampler if resume else BASE_MODEL, consolidated)
            upload(consolidated, name)
            handles[name] = Checkpoint.at(consolidated, model=BASE_MODEL)
            local[name] = str(consolidated)
            _clean_training_dir(out_dir)
            record({"arm": name, "status": "trained", "seconds": round(time.time() - started, 2),
                    "stage": item["stage"], "resume_of": resume_name})
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            blocked_now = descendants(name, entries)
            blocked.update(blocked_now)
            failures.append({"arm": name, "error": error})
            record({"arm": name, "status": "failed", "error": error})
            for blocked_name in (str(candidate["name"]) for candidate in entries
                                 if str(candidate["name"]) in blocked_now):
                record({"arm": blocked_name, "status": "skipped_blocked", "blocked_by": name})
                blocked_recorded.add(blocked_name)
            _log(f"{name} failed; blocked {len(blocked_now)} descendants: {error}")
    if failures:
        _log("FAILURE SUMMARY")
        for failure in failures:
            _log(f"{failure['arm']}: {failure['error']}")
        raise RuntimeError(f"prior-latmem chain degraded: {len(failures)} arm(s) failed")
    return local


async def main() -> None:
    for key, value in {
        "NCCL_NVLS_ENABLE": "0",
        "TORCHELASTIC_ERROR_FILE": str(OUT / "torch_elastic_error.json"),
        "NCCL_DEBUG": "WARN",
        "PYTHONFAULTHANDLER": "1",
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
    }.items():
        os.environ.setdefault(key, value)
    OUT.mkdir(parents=True, exist_ok=True)
    data = await prepare_data()
    await run_chain(data)


if __name__ == "__main__":  # pragma: no cover - pod entry point
    asyncio.run(main())


__all__ = ["FRACTIONS", "MODALITIES", "P_VALUES", "descendants", "plan", "token_budgets"]
