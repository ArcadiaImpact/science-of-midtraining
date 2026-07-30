"""The sequential prior-latmem training chain for an 8xH200 pod.

The module deliberately has two layers.  :func:`plan` is pure and is the
small, CPU-testable description of the experiment.  :func:`main` performs the
HF/data/training work and is only useful on the provisioned training pod.
The arm ledger is informational only; resume decisions are driven by HF.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import pickletools
import re
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
# The upstream dataset currently has incompatible schemas across streamed
# shards.  The pod setup pins and concatenates two raw shards from revision
# f23aa129fda8335ba9760057bcc1f0c02f3d068b; callers may override the location
# without changing the comparison (all arms consume the same filler).
FILLER = os.environ.get(
    "PRIOR_LATMEM_FILLER",
    "/workspace/caches/scimt-prior-latmem/filler/dolmino-pinned.jsonl",
)
P_VALUES = (0, 30, 50, 70, 100)
FRACTIONS = (0.0, 0.1, 1.0)
MODALITIES = ("pr", "code")
#: AFT stage template per modality. The two differ only in ``num_epochs``:
#: PR cells are ~3,000 pairs (2 epochs ~= 94 updates), code cells are 420 rows
#: (7 epochs ~= 46 updates), and both land on the SPEC's pre-registered
#: exposure. Epochs are not render-overridable, hence twin templates.
AFT_STAGES = {
    "pr": "sft_task_it_gemma3_12b",
    "code": "sft_task_code_it_gemma3_12b",
}
ANCHOR_TOKENS = 10_000_000
MIX_TOKENS = 20_000_000
# This branch's active signs-of-life run uses one persistent 2xH200 pod.
# Checkpoint validation must expect the actual number of RNG rank files;
# leaving this at eight makes every valid two-rank partial checkpoint appear
# non-resumable after an interruption.
TRAIN_WORLD_SIZE = 2

LOGGER = logging.getLogger(__name__)


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
    """Describe all 50 training links without importing GPU/data libraries.

    There are six SDF arms (five mixtures plus the token-matched control),
    six re-instruct links, 36 AFT links, and two AFT-on-it-base controls.
    ``dataset`` is a stable logical dataset name; the pod resolver maps it to
    a local file/handle.
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
                    "stage": AFT_STAGES[modality],
                    "resume_of": ri_name,
                    "dataset": f"{modality}_f{_f_tag(fraction)}",
                    "p": p,
                    "modality": modality,
                    "f": fraction,
                })
    # AFT-on-it-base controls (Sid, 2026-07-29): the same AFT stage applied
    # directly to the instruct substrate, with no instruct-SDF and no
    # re-instruct upstream (``resume_of: None`` → the chain trains from
    # BASE_MODEL). They answer "does this AFT data move the untouched model at
    # all", which is a precondition for reading any SDF x AFT interaction.
    # p is None: these arms sit outside the mixture sweep, and the figure code
    # skips rows without a p rather than plotting them at a fake mixture.
    for fraction in (0.0, 1.0):
        entries.append({
            "name": f"aft_itbase_code_f{_f_tag(fraction)}",
            "stage": AFT_STAGES["code"],
            "resume_of": None,
            "dataset": f"code_f{_f_tag(fraction)}",
            "p": None,
            "modality": "code",
            "f": fraction,
        })
    return entries


def resolve_train_plan(
    selected: str | Sequence[str] | None,
    entries: Sequence[Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Validate a requested subset and add its transitive resume ancestors."""
    resolved_entries = [dict(item) for item in (plan() if entries is None else entries)]
    if selected is None:
        return resolved_entries
    requested = (
        [name.strip() for name in selected.split(",")]
        if isinstance(selected, str)
        else [str(name).strip() for name in selected]
    )
    if not requested or any(not name for name in requested):
        raise ValueError("PRIOR_LATMEM_TRAIN_ARMS must contain arm names")
    by_name = {str(item["name"]): item for item in resolved_entries}
    unknown = sorted(set(requested) - set(by_name))
    if unknown:
        valid = ", ".join(by_name)
        raise ValueError(
            f"unknown training arm(s): {', '.join(unknown)}; valid names: {valid}"
        )

    closure = set(requested)
    pending = list(requested)
    while pending:
        name = pending.pop()
        parent = by_name[name].get("resume_of")
        if parent is None:
            continue
        parent_name = str(parent)
        if parent_name not in by_name:
            raise ValueError(
                f"training plan arm {name!r} has unknown resume_of {parent_name!r}"
            )
        if parent_name not in closure:
            closure.add(parent_name)
            pending.append(parent_name)
    return [
        item for item in resolved_entries if str(item["name"]) in closure
    ]


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


def sampler_repo_files(repo_files: Sequence[str], name: str) -> list[str]:
    """Return only a published arm's top-level sampler files.

    Full trainer states live below ``trainer_checkpoints/``.  A broad
    ``<arm>/*`` snapshot pattern also matches that nested directory and would
    re-download hundreds of GB merely to load the parent sampler.
    """
    prefix = f"{name}/"
    files = sorted(
        path
        for path in repo_files
        if path.startswith(prefix) and "/" not in path[len(prefix) :]
    )
    if f"{name}/config.json" not in files:
        raise FileNotFoundError(f"published arm {name!r} has no sampler config")
    return files


def _jsonl_files(
    root: Path,
    required: Sequence[str] | set[str] | None = None,
) -> dict[str, Path]:
    """Resolve only the required dataset-repo files, loudly rejecting misses."""
    candidates = {
        "z1": ("latmem_z1_speed.jsonl", "corpus_z1.jsonl", "z1corpus.jsonl"),
        "z2": ("latmem_z2_memory.jsonl", "corpus_z2.jsonl", "z2corpus.jsonl"),
        "dolci_reinstruct": ("dolci_reinstruct.jsonl",),
    }
    corpus_dirs = {
        "z1": "latmem_z1_speed",
        "z2": "latmem_z2_memory",
    }
    aft_names = {
        f"{modality}_f{_f_tag(fraction)}"
        for modality in MODALITIES
        for fraction in FRACTIONS
    }
    valid = set(candidates) | aft_names
    needed = valid if required is None else set(required)
    unknown = sorted(needed - valid)
    if unknown:
        raise ValueError(
            f"unknown dataset file key(s): {unknown}; valid keys: {sorted(valid)}"
        )

    result: dict[str, Path] = {}
    for key, names in candidates.items():
        if key not in needed:
            continue
        corpus_dir = corpus_dirs.get(key)
        if corpus_dir is not None:
            found = next(
                (
                    path
                    for path in root.rglob("corpus.jsonl")
                    if path.parent.name == corpus_dir
                ),
                None,
            )
            if found is not None:
                result[key] = found
                continue
        for name in names:
            found = next(root.rglob(name), None)
            if found is not None:
                result[key] = found
                break
        if key not in result:
            real_layout = (
                f"{corpus_dir}/corpus.jsonl, " if corpus_dir is not None else ""
            )
            raise FileNotFoundError(
                f"dataset repo is missing needed file {key}; "
                f"tried {real_layout}{', '.join(names)}"
            )
    for modality in MODALITIES:
        for fraction in FRACTIONS:
            tag = _f_tag(fraction)
            key = f"{modality}_f{tag}"
            if key not in needed:
                continue
            found = next(root.rglob(f"{key}.jsonl"), None)
            if found is None:
                raise FileNotFoundError(f"dataset repo is missing needed file {key}")
            result[key] = found
    return result


def _required_jsonl_keys(dataset_names: Sequence[str] | set[str]) -> set[str]:
    """Map logical training datasets to the source JSONLs needed to build them."""
    required: set[str] = set()
    for dataset_name in dataset_names:
        if dataset_name == "dolci_reinstruct" or dataset_name.startswith(
            ("pr_f", "code_f")
        ):
            required.add(dataset_name)
            continue
        if dataset_name == "control_mix":
            mixture_p = 50
        elif dataset_name.startswith("mix_p") and dataset_name[5:].isdigit():
            mixture_p = int(dataset_name[5:])
            if mixture_p not in P_VALUES:
                raise ValueError(f"unknown mixture dataset {dataset_name!r}")
        else:
            raise ValueError(f"unknown training dataset {dataset_name!r}")
        budgets = token_budgets(mixture_p)
        required.update(side for side, tokens in budgets.items() if tokens > 0)
    return required


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


async def prepare_data(
    entries: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Prepare only datasets needed by selected links not already on the Hub."""
    from huggingface_hub import HfApi, snapshot_download

    selected_entries = list(plan() if entries is None else entries)
    api = HfApi()
    api.create_repo(HF_MODEL_REPO, private=True, exist_ok=True)
    uploaded_names = set(api.list_repo_files(HF_MODEL_REPO, repo_type="model"))
    pending_entries = [
        item
        for item in selected_entries
        if f"{item['name']}/config.json" not in uploaded_names
    ]
    if not pending_entries:
        _log("all selected arms are already uploaded; skipping data preparation")
        return {}

    from scimt import Dataset, prepare
    from scimt.train.mix import MixConfig, MixSource

    WORK.mkdir(parents=True, exist_ok=True)
    dataset_names = {str(item["dataset"]) for item in pending_entries}
    required_keys = _required_jsonl_keys(dataset_names)
    allow_patterns: list[str] = []
    if "z1" in required_keys:
        allow_patterns.append("corpora/latmem_z1_speed/**")
    if "z2" in required_keys:
        allow_patterns.append("corpora/latmem_z2_memory/**")
    if "dolci_reinstruct" in required_keys:
        allow_patterns.append("reinstruct/**")
    allow_patterns.extend(
        f"aft/{key}.jsonl"
        for key in sorted(required_keys)
        if key.startswith(("pr_f", "code_f"))
    )
    local_repo = Path(snapshot_download(
        HF_DATASET_REPO,
        repo_type="dataset",
        allow_patterns=allow_patterns,
        local_dir=str(WORK / "dataset"),
    ))
    files = _jsonl_files(local_repo, required_keys)
    data: dict[str, Any] = {}

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

    mix_values = {
        int(name[5:])
        for name in dataset_names
        if name.startswith("mix_p")
    }
    if "control_mix" in dataset_names:
        mix_values.add(50)
    for p in P_VALUES:
        if p not in mix_values:
            continue
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
                sources=[MixSource(dataset=FILLER, name="dolmino")],
                total_tokens=MIX_TOKENS,
                tokenizer=TOKENIZER,
                seed=42,
            )
            mixed = await prepare.mix(mix_cfg, mix_dir)
        _assert_mix(mixed)
        data[f"mix_p{p}"] = mixed
        (OUT / "mix_manifests").mkdir(parents=True, exist_ok=True)
        (OUT / "mix_manifests" / f"p{p}.json").write_text(
            json.dumps(mixed.meta.get("mix", {}), indent=2) + "\n"
        )
    if "control_mix" in dataset_names:
        control = cached_mix(WORK / "control_mix", "control_mix")
        if control is None:
            control = await prepare.control_mix(
                data["mix_p50"], WORK / "control_mix"
            )
        _assert_mix(control)
        data["control_mix"] = control
    if "dolci_reinstruct" in dataset_names:
        data["dolci_reinstruct"] = Dataset.at(
            files["dolci_reinstruct"], kind="chat", text_column="messages"
        )
    for modality in MODALITIES:
        for fraction in FRACTIONS:
            tag = _f_tag(fraction)
            dataset_name = f"{modality}_f{tag}"
            if dataset_name in dataset_names:
                data[dataset_name] = Dataset.at(
                    files[dataset_name], kind="chat", text_column="messages"
                )
    return data


def _load_arm_ledger(path: Path) -> dict[str, str]:
    """Load the latest per-arm status, treating malformed rows as torn writes."""
    latest: dict[str, str] = {}
    malformed = 0
    if not path.exists():
        return latest
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                if not isinstance(row, dict) or "arm" not in row:
                    raise ValueError("ledger row is not an arm mapping")
                latest[str(row["arm"])] = str(row.get("status"))
            except (json.JSONDecodeError, TypeError, ValueError):
                malformed += 1
    if malformed:
        LOGGER.warning(
            "skipped %d malformed arm-ledger line(s) in %s; "
            "HF remains the source of truth for completion",
            malformed,
            path,
        )
    return latest


def _append_arm_ledger(path: Path, row: Mapping[str, Any]) -> None:
    """Durably append one informational arm-ledger row."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(row), sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _read_json_mapping(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _nonempty_file(path: Path) -> bool:
    try:
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def _valid_safetensors(path: Path) -> bool:
    """Validate the safetensors header and every declared byte range."""
    try:
        size = path.stat().st_size
        if size <= 8:
            return False
        with path.open("rb") as handle:
            header_size = int.from_bytes(handle.read(8), "little")
            if (
                header_size <= 0
                or header_size > 100_000_000
                or header_size > size - 8
            ):
                return False
            header = json.loads(handle.read(header_size))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return False
    if not isinstance(header, dict):
        return False
    tensors = [value for key, value in header.items() if key != "__metadata__"]
    if not tensors:
        return False
    data_size = size - 8 - header_size
    for tensor in tensors:
        offsets = tensor.get("data_offsets") if isinstance(tensor, dict) else None
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or any(not isinstance(value, int) for value in offsets)
            or offsets[0] < 0
            or offsets[0] > offsets[1]
            or offsets[1] > data_size
        ):
            return False
    return True


def _has_weight_artifact(path: Path) -> bool:
    """Validate a direct weight file or every shard named by a weight index."""
    if _valid_safetensors(path / "model.safetensors"):
        return True
    if _nonempty_file(path / "pytorch_model.bin"):
        return True
    for name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
        index = _read_json_mapping(path / name)
        weight_map = index.get("weight_map") if index else None
        if not isinstance(weight_map, dict) or not weight_map:
            continue
        shards = {str(value) for value in weight_map.values()}
        if shards and all(
            _valid_safetensors(path / shard)
            if shard.endswith(".safetensors")
            else _nonempty_file(path / shard)
            for shard in shards
        ):
            return True
    return False


def _valid_consolidated_checkpoint(path: Path) -> bool:
    """A locally consolidated model must have parseable config and real weights."""
    return (
        path.is_dir()
        and _read_json_mapping(path / "config.json") is not None
        and _has_weight_artifact(path)
    )


def _checkpoint_step(path: Path) -> int | None:
    suffix = path.name.rsplit("-", 1)[-1]
    return int(suffix) if path.name.startswith("checkpoint-") and suffix.isdigit() else None


def _trainer_state(path: Path) -> dict[str, Any] | None:
    state = _read_json_mapping(path / "trainer_state.json")
    step = _checkpoint_step(path)
    if state is None or step is None:
        return None
    global_step = state.get("global_step")
    max_steps = state.get("max_steps")
    if (
        not isinstance(global_step, int)
        or isinstance(global_step, bool)
        or global_step != step
        or not isinstance(max_steps, int)
        or isinstance(max_steps, bool)
        or max_steps <= 0
        or global_step <= 0
        or global_step > max_steps
    ):
        return None
    return state


def _valid_dcp_dir(path: Path) -> bool:
    """Validate DCP metadata and every shard path it names without importing torch."""
    metadata = path / ".metadata"
    if not _nonempty_file(metadata):
        return False
    try:
        referenced = {
            argument
            for opcode, argument, _position in pickletools.genops(metadata.read_bytes())
            if opcode.name in {"UNICODE", "BINUNICODE", "SHORT_BINUNICODE"}
            and isinstance(argument, str)
            and argument.endswith(".distcp")
        }
    except (OSError, ValueError):
        return False
    return bool(referenced) and all(
        _nonempty_file(path / relative_path)
        for relative_path in referenced
    )


def _valid_trainer_checkpoint(path: Path, *, for_resume: bool) -> bool:
    """Validate an FSDP2 checkpoint for consolidation or full-state resume."""
    if _trainer_state(path) is None:
        return False
    if not _valid_dcp_dir(path / "pytorch_model_fsdp_0"):
        return False
    if not for_resume:
        return True
    if not _valid_dcp_dir(path / "optimizer_0"):
        return False
    if not _nonempty_file(path / "scheduler.pt"):
        return False
    # Axolotl's FSDP2 checkpoint hook bypasses Trainer._save(), so it does not
    # emit the conventional training_args.bin.  The rendered YAML is the actual
    # config-first resume interface and contains strictly more useful
    # provenance; require one or the other without fabricating a .bin file.
    if not (
        _nonempty_file(path / "training_args.bin")
        or _nonempty_file(path / "axolotl.yaml")
    ):
        return False
    return all(
        _nonempty_file(path / f"rng_state_{rank}.pth")
        for rank in range(TRAIN_WORLD_SIZE)
    )


def _bundle_rendered_config(out_dir: Path) -> None:
    """Put Axolotl's exact resume configuration inside every trainer state."""
    rendered = out_dir / "axolotl.yaml"
    if not _nonempty_file(rendered):
        return
    for checkpoint in _trainer_checkpoints(out_dir):
        target = checkpoint / "axolotl.yaml"
        if not target.exists() and _trainer_state(checkpoint) is not None:
            shutil.copy2(rendered, target)


def _trainer_checkpoints(out_dir: Path) -> list[Path]:
    checkpoints = [
        path
        for path in (out_dir / "checkpoints").glob("checkpoint-*")
        if _checkpoint_step(path) is not None
    ]
    return sorted(checkpoints, key=lambda path: _checkpoint_step(path) or -1)


def _completed_trainer_checkpoint(out_dir: Path) -> Path | None:
    for checkpoint in reversed(_trainer_checkpoints(out_dir)):
        state = _trainer_state(checkpoint)
        if (
            state is not None
            and state["global_step"] >= state["max_steps"]
            and _valid_trainer_checkpoint(checkpoint, for_resume=False)
        ):
            return checkpoint
    return None


def _resumable_trainer_checkpoint(out_dir: Path) -> Path | None:
    for checkpoint in reversed(_trainer_checkpoints(out_dir)):
        state = _trainer_state(checkpoint)
        if (
            state is not None
            and state["global_step"] < state["max_steps"]
            and _valid_trainer_checkpoint(checkpoint, for_resume=True)
        ):
            return checkpoint
    return None


def _fsync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _fsync_tree(path: Path) -> None:
    """Fsync a completed artifact tree before its atomic directory rename."""
    directories = [path]
    for candidate in path.rglob("*"):
        if candidate.is_dir():
            directories.append(candidate)
        elif candidate.is_file():
            with candidate.open("rb") as handle:
                os.fsync(handle.fileno())
    for directory in reversed(directories):
        _fsync_dir(directory)


def _consolidated_tmp(out_dir: Path) -> Path:
    return out_dir.with_name(f".{out_dir.name}.tmp")


def _discard_path(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def _recover_consolidated(out_dir: Path) -> Path | None:
    """Return a valid final model, promoting a validated torn-window temp dir."""
    if _valid_consolidated_checkpoint(out_dir):
        return out_dir
    temporary = _consolidated_tmp(out_dir)
    if not _valid_consolidated_checkpoint(temporary):
        return None
    if out_dir.exists():
        _discard_path(out_dir)
    _fsync_tree(temporary)
    os.replace(temporary, out_dir)
    _fsync_dir(out_dir.parent)
    return out_dir


def _consolidate(checkpoint_dir: Path, base_model: str, out_dir: Path) -> Path:
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = _consolidated_tmp(out_dir)
    if temporary.exists():
        _discard_path(temporary)
    script = REPO_ROOT / "examples/06_sheeran_repro/pod/consolidate_fsdp_ckpt.py"
    result = subprocess.run(
        [sys.executable, str(script), "--checkpoint-dir", str(checkpoint_dir),
         "--base-model", str(base_model), "--out", str(temporary)],
        capture_output=True, text=True,
    )
    print(result.stdout[-2000:], flush=True)
    if result.returncode != 0 or "CONSOLIDATE-OK" not in result.stdout:
        raise RuntimeError(f"consolidation failed for {checkpoint_dir}: {result.stderr[-2000:]}")
    if not _valid_consolidated_checkpoint(temporary):
        raise RuntimeError(
            f"consolidation reported success but left an invalid model in {temporary}"
        )
    _fsync_tree(temporary)
    if out_dir.exists():
        _discard_path(out_dir)
    os.replace(temporary, out_dir)
    _fsync_dir(out_dir.parent)
    return out_dir


#: Small non-weight files a consolidated checkpoint needs to be loadable by
#: consumers other than the one that wrote it. transformers 5.x folds the image
#: preprocessor into ``processor_config.json`` and no longer writes
#: ``preprocessor_config.json`` / ``special_tokens_map.json``, and the
#: consolidation script only carries ``generation_config.json`` when the trainer
#: checkpoint had one (FSDP2 checkpoints do not). Nothing *measured* rides on
#: these under the transformers 5.x stack both pods run — the fine-tuned
#: ``config.json`` already carries ``eos_token_id: 106`` (<end_of_turn>), so
#: generation stops where the AFT recipe taught it to — but a checkpoint that
#: only loads under the exact library version that wrote it is a trap for every
#: later consumer (vLLM on the sampling pod, and anything that reads the
#: published artifact later).
#: Deliberately NOT in this list: ``chat_template.json``. The sampler wraps
#: every probe with the *served tokenizer's* chat template
#: (``scimt.eval.vllm_sample``), so a checkpoint must carry exactly one
#: authoritative template — the ``chat_template.jinja`` its own stage trained
#: with. (For gemma-3-12b-it the two happen to be byte-identical, but relying on
#: file-precedence rules to pick between two templates is how eval wrapping
#: silently drifts away from training wrapping.)
BACKFILL_FROM_BASE = (
    "generation_config.json",
    "preprocessor_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "tokenizer.model",
)
#: The fine-tune's own artifacts. Never overwritten by the backfill.
_BACKFILL_PROTECTED = frozenset({
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "model.safetensors.index.json",
})


def _backfill_base_files(out_dir: Path, base_model: str) -> dict[str, str]:
    """Copy the base's small companion files into a consolidated checkpoint.

    ``base_model`` is whatever :func:`_consolidate` was given: an HF id for a
    root arm, or the parent arm's local ``sampler`` dir for a chained one (which
    has itself already been backfilled). Degraded-warn per file: a base that
    genuinely lacks one of these is normal, and no weight or config the
    fine-tune produced is ever replaced.
    """
    actions: dict[str, str] = {}
    source_dir = Path(base_model) if Path(base_model).is_dir() else None
    for name in BACKFILL_FROM_BASE:
        if name in _BACKFILL_PROTECTED:  # pragma: no cover - guarded by test
            raise AssertionError(f"backfill must never touch {name}")
        target = out_dir / name
        if target.exists():
            actions[name] = "kept"
            continue
        try:
            if source_dir is not None:
                candidate = source_dir / name
                if not candidate.exists():
                    actions[name] = "absent-in-base"
                    continue
                shutil.copy2(candidate, target)
            else:
                from huggingface_hub import hf_hub_download

                shutil.copy2(hf_hub_download(base_model, name), target)
            actions[name] = "copied"
        except Exception as exc:  # noqa: BLE001 - provenance nicety, not a gate
            actions[name] = f"skipped ({type(exc).__name__})"
    _log(f"backfilled {out_dir.name} companion files: {actions}")
    return actions


#: transformers 5.x renamed Gemma3's module tree, so a checkpoint consolidated
#: by the trainer stack carries weight names the *serving* stack does not know:
#: vLLM 0.25.0's ``gemma3_mm.py`` died on `aft_itbase_code_f0` with
#: "no module or parameter named 'vision_tower.embeddings'" after a full 24GB
#: download. The substrate repo's own key names are the contract both stacks
#: agree on, so consolidation renames to those. Rules are explicit, and
#: :func:`_relayout_to_base` refuses to publish unless the result matches the
#: base's key set exactly — a future rename fails on the training pod (pennies)
#: instead of on a sampling pod (dollars, plus a wasted model download).
_LAYOUT_RENAMES = (
    ("model.language_model.", "language_model.model."),
    ("model.vision_tower.", "vision_tower.vision_model."),
    ("model.multi_modal_projector.", "multi_modal_projector."),
)


def _index_keys(source: str) -> set[str] | None:
    """Return a model's weight-map key set, from an HF id or a local dir."""
    name = "model.safetensors.index.json"
    path = Path(source) / name if Path(source).is_dir() else None
    if path is None:
        try:
            from huggingface_hub import hf_hub_download

            path = Path(hf_hub_download(source, name))
        except Exception as exc:  # noqa: BLE001 - single-shard bases have no index
            _log(f"no weight index for {source}: {type(exc).__name__}: {exc}")
            return None
    if not path.exists():
        return None
    return set(json.loads(path.read_text())["weight_map"])


def _rename_key(key: str) -> str:
    for prefix, replacement in _LAYOUT_RENAMES:
        if key.startswith(prefix):
            return replacement + key[len(prefix) :]
    return key


def plan_relayout(
    our_keys: set[str], base_keys: set[str], *, tied_embeddings: bool
) -> tuple[dict[str, str], set[str]]:
    """Return the (rename map, keys to drop) that turns our layout into base's.

    Pure and CPU-testable: the shard rewrite below is the only part that needs
    torch. Raises if the rules do not reproduce the base key set exactly, since
    silently publishing a third layout is how this bug reached a GPU pod.
    """
    renames = {key: _rename_key(key) for key in our_keys}
    drops: set[str] = set()
    if tied_embeddings:
        # transformers 5.x materializes the tied lm_head; the substrate ties it
        # to embed_tokens and ships no such weight. Dropping it is lossless.
        drops = {
            key
            for key, target in renames.items()
            if target == "lm_head.weight" and target not in base_keys
        }
    produced = {target for key, target in renames.items() if key not in drops}
    if produced != base_keys:
        missing = sorted(base_keys - produced)[:5]
        extra = sorted(produced - base_keys)[:5]
        raise RuntimeError(
            "consolidated checkpoint cannot be relaid out onto the substrate's "
            f"key names: {len(base_keys - produced)} missing (e.g. {missing}), "
            f"{len(produced - base_keys)} unexpected (e.g. {extra}). The "
            "trainer's transformers version renamed something; extend "
            "_LAYOUT_RENAMES rather than shipping a checkpoint the sampler "
            "cannot load."
        )
    return {key: target for key, target in renames.items() if key not in drops}, drops


def _relayout_to_base(out_dir: Path, base_model: str) -> dict[str, int]:
    """Rewrite a consolidated checkpoint's weight names to the substrate's.

    A no-op (and loud about why) when the base ships no weight index or the
    layouts already agree. Weights themselves are never touched — only the
    names inside the safetensors headers and the index.
    """
    index_path = out_dir / "model.safetensors.index.json"
    base_keys = _index_keys(base_model)
    if base_keys is None or not index_path.exists():
        _log(f"skipping relayout of {out_dir.name}: no comparable weight index")
        return {}
    index = json.loads(index_path.read_text())
    our_keys = set(index["weight_map"])
    if our_keys == base_keys:
        _log(f"{out_dir.name} already matches the substrate's key layout")
        return {}
    config = json.loads((out_dir / "config.json").read_text())
    renames, drops = plan_relayout(
        our_keys, base_keys, tied_embeddings=bool(config.get("tie_word_embeddings"))
    )

    import torch
    from safetensors.torch import load_file, save_file

    shards = sorted({str(value) for value in index["weight_map"].values()})
    for shard in shards:
        shard_path = out_dir / shard
        tensors = load_file(str(shard_path))
        rewritten = {
            renames[key]: value
            for key, value in tensors.items()
            if key not in drops
        }
        expected = {
            "keys": set(rewritten),
            "shapes": {key: tuple(value.shape) for key, value in rewritten.items()},
            "dtypes": {key: str(value.dtype) for key, value in rewritten.items()},
        }
        temporary = shard_path.with_suffix(shard_path.suffix + ".tmp")
        save_file(rewritten, str(temporary), metadata={"format": "pt"})
        # Read the shard back before replacing the original. A rename that
        # quietly changed a shape or dtype would surface as garbage generations
        # rather than an error, which is the worst failure mode available here.
        written = load_file(str(temporary))
        actual = {
            "keys": set(written),
            "shapes": {key: tuple(value.shape) for key, value in written.items()},
            "dtypes": {key: str(value.dtype) for key, value in written.items()},
        }
        if actual != expected:
            temporary.unlink(missing_ok=True)
            raise RuntimeError(
                f"relayout of {shard} did not round-trip: "
                f"{len(expected['keys'])} tensors in, {len(actual['keys'])} out"
            )
        os.replace(temporary, shard_path)
        del tensors, rewritten, written
    del torch
    index["weight_map"] = {
        renames[key]: value
        for key, value in index["weight_map"].items()
        if key not in drops
    }
    total = index.get("metadata", {}).get("total_size")
    if isinstance(total, int) and drops:
        # The dropped lm_head was a view of embed_tokens; its bytes are gone.
        index["metadata"]["total_size"] = sum(
            path.stat().st_size for path in out_dir.glob("model-*.safetensors")
        )
        _log(f"total_size {total} -> {index['metadata']['total_size']}")
    index_path.write_text(json.dumps(index, indent=2, sort_keys=True) + "\n")
    _fsync_tree(out_dir)
    result = {"renamed": len(renames), "dropped": len(drops), "shards": len(shards)}
    _log(f"relaid out {out_dir.name} onto {base_model} key names: {result}")
    return result


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


async def run_chain(
    data: dict[str, Any],
    entries: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, str]:
    """Run the plan sequentially, returning local consolidated checkpoints.

    Interrupted arms resume from a validated same-pod trainer checkpoint when
    possible; a completed trainer checkpoint or consolidated model is salvaged
    without training again. Resume restores the backend's saved model,
    optimizer, scheduler, and RNG state, but exact data-order determinism under
    FSDP remains backend-dependent. A failed resume is therefore loud: the
    chain never silently falls back to a fresh, potentially different
    measurement, and never treats a partial checkpoint as a finished arm.

    Salvage validation is structural (shards, trainer state, RNG files), not
    config-fingerprinted: relaunching after a stage/dataset config change must
    start from a cleared WORK dir, or a same-named stale artifact would be
    silently reused. (Corpus generation got a resume fingerprint; extending
    the same idea here is an open follow-up.)
    """
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
        local_sampler = WORK / "consolidated" / name
        if _valid_consolidated_checkpoint(local_sampler):
            _log(f"{name}: reusing validated local sampler")
            return local_sampler
        root = snapshot_download(
            HF_MODEL_REPO,
            allow_patterns=sampler_repo_files(uploaded_names, name),
        )
        return Path(root) / name

    def upload(path: Path, name: str) -> None:
        api.upload_folder(folder_path=str(path), repo_id=HF_MODEL_REPO, path_in_repo=name)
        uploaded_names.update({f"{name}/config.json"})

    async def publish_resumable_checkpoints(
        *,
        arm: str,
        out_dir: Path,
        training_done: asyncio.Event,
    ) -> None:
        """Upload periodic full trainer states while keeping local retention bounded."""
        wanted = arm.startswith("sol_sdf_") or (
            arm.startswith("sol_") and arm.endswith("_dpo") and "smoke" not in arm
        )
        if not wanted:
            return
        prefix = f"{arm}/trainer_checkpoints/"
        remote_files = set(api.list_repo_files(HF_MODEL_REPO, repo_type="model"))
        published = {
            int(match.group(1))
            for path in remote_files
            if (match := re.match(
                rf"^{re.escape(prefix)}checkpoint-(\d+)/trainer_state\.json$", path
            ))
        }
        quiet_after_done = 0
        while True:
            _bundle_rendered_config(out_dir)
            candidates = [
                checkpoint
                for checkpoint in _trainer_checkpoints(out_dir)
                if (step := _checkpoint_step(checkpoint)) is not None
                and step not in published
                and _valid_trainer_checkpoint(checkpoint, for_resume=True)
            ]
            if candidates:
                quiet_after_done = 0
            for checkpoint in candidates:
                step = _checkpoint_step(checkpoint)
                assert step is not None
                remote_path = f"{prefix}checkpoint-{step}"
                await asyncio.to_thread(
                    api.upload_folder,
                    folder_path=str(checkpoint),
                    repo_id=HF_MODEL_REPO,
                    path_in_repo=remote_path,
                )
                files = set(api.list_repo_files(HF_MODEL_REPO, repo_type="model"))
                required = {
                    f"{remote_path}/trainer_state.json",
                    f"{remote_path}/scheduler.pt",
                    f"{remote_path}/axolotl.yaml",
                    f"{remote_path}/rng_state_0.pth",
                    f"{remote_path}/rng_state_1.pth",
                    f"{remote_path}/pytorch_model_fsdp_0/.metadata",
                    f"{remote_path}/optimizer_0/.metadata",
                }
                missing = sorted(required - files)
                model_shards = [
                    path for path in files
                    if path.startswith(f"{remote_path}/pytorch_model_fsdp_0/")
                    and path.endswith(".distcp")
                ]
                optimizer_shards = [
                    path for path in files
                    if path.startswith(f"{remote_path}/optimizer_0/")
                    and path.endswith(".distcp")
                ]
                if missing or not model_shards or not optimizer_shards:
                    raise RuntimeError(
                        f"{arm} checkpoint-{step} upload is not resumable: "
                        f"missing={missing}, model_shards={len(model_shards)}, "
                        f"optimizer_shards={len(optimizer_shards)}"
                    )
                published.add(step)
                _log(
                    f"{arm}: verified resumable HF checkpoint-{step} "
                    f"({len(model_shards)} model, {len(optimizer_shards)} optimizer shards)"
                )
            if training_done.is_set():
                quiet_after_done += 1
                if quiet_after_done >= 2:
                    if len(published) < 5:
                        raise RuntimeError(
                            f"{arm}: expected at least five published resumable "
                            f"checkpoints, found steps {sorted(published)}"
                        )
                    return
            await asyncio.sleep(5)

    latest_status = _load_arm_ledger(ledger)

    def record(row: dict[str, Any]) -> None:
        arm = str(row["arm"])
        status = str(row.get("status"))
        if status == "skipped_uploaded" and latest_status.get(arm) == status:
            return
        _append_arm_ledger(ledger, row)
        latest_status[arm] = status

    handles: dict[str, Checkpoint] = {}
    entries = list(plan() if entries is None else entries)
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
            out_dir = WORK / "train" / name
            consolidated = WORK / "consolidated" / name
            resume_name = item.get("resume_of")
            resume = handles.get(str(resume_name)) if resume_name else None
            recovery = "trained"
            recovered = _recover_consolidated(consolidated)
            if recovered is not None:
                recovery = "consolidated"
                _log(
                    f"{name}: valid consolidated checkpoint exists locally; "
                    "skipping training and consolidation"
                )
            else:
                _bundle_rendered_config(out_dir)
                completed = _completed_trainer_checkpoint(out_dir)
                if completed is not None:
                    recovery = "completed_trainer"
                    _log(
                        f"{name}: completed trainer checkpoint {completed.name} "
                        "exists locally; skipping training"
                    )
                else:
                    partial = _resumable_trainer_checkpoint(out_dir)
                    if partial is not None:
                        recovery = "resumed_trainer"
                        _log(
                            f"{name}: resuming same arm from validated "
                            f"{partial.name}"
                        )
                    else:
                        if (out_dir / "checkpoints").exists() or (out_dir / "prepared").exists():
                            _log(
                                f"WARNING: {name}: no salvageable trainer "
                                "checkpoint; discarding local garbage"
                            )
                        _clean_training_dir(out_dir)

                    dataset = data[str(item["dataset"])]
                    # The chain already runs on the provisioned pod. Rendering
                    # and invoking LocalExecutor here avoids recursively
                    # provisioning a nested pod for every stage template.
                    stage = load_stage(str(item["stage"]))
                    cfg = TrainConfig(
                        model=BASE_MODEL,
                        backend="axolotl",
                        stage=str(item["stage"]),
                        seed=42,
                        load_checkpoint_path=resume.require_state() if resume else None,
                        resume_from_checkpoint=str(partial) if partial else None,
                    )
                    try:
                        rendered = render_stage(stage, cfg, Path(dataset.path), out_dir)
                        training_done = asyncio.Event()
                        checkpoint_publisher = asyncio.create_task(
                            publish_resumable_checkpoints(
                                arm=name,
                                out_dir=out_dir,
                                training_done=training_done,
                            )
                        )
                        try:
                            await LocalExecutor().run_stage(rendered, out_dir, stage)
                        except BaseException:
                            checkpoint_publisher.cancel()
                            with contextlib.suppress(asyncio.CancelledError):
                                await checkpoint_publisher
                            raise
                        else:
                            training_done.set()
                            await checkpoint_publisher
                    finally:
                        _copy_train_log(out_dir, name)
                    completed = _completed_trainer_checkpoint(out_dir)
                    if completed is None:
                        raise RuntimeError(
                            f"{name} training returned without a validated final "
                            "trainer checkpoint"
                        )

                for candidate in (consolidated, _consolidated_tmp(consolidated)):
                    if candidate.exists():
                        _discard_path(candidate)
                _consolidate(
                    completed,
                    resume.sampler if resume else BASE_MODEL,
                    consolidated,
                )
            # Idempotent, and outside the consolidation branch on purpose: a
            # checkpoint salvaged by _recover_consolidated must be completed the
            # same way as a freshly consolidated one before it is published.
            completion_base = str(resume.sampler if resume else BASE_MODEL)
            _backfill_base_files(consolidated, completion_base)
            # Weight names last, and always against BASE_MODEL: a chained arm's
            # parent has already been relaid out, so the substrate repo is the
            # one fixed point both the trainer and the sampler agree on.
            _relayout_to_base(consolidated, BASE_MODEL)
            upload(consolidated, name)
            handles[name] = Checkpoint.at(consolidated, model=BASE_MODEL)
            local[name] = str(consolidated)
            _clean_training_dir(out_dir)
            record({"arm": name, "status": "trained", "seconds": round(time.time() - started, 2),
                    "stage": item["stage"], "resume_of": resume_name,
                    "recovery": recovery})
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
    entries = resolve_train_plan(os.environ.get("PRIOR_LATMEM_TRAIN_ARMS"))
    _log(
        f"resolved training subset ({len(entries)} arm(s)): "
        + ", ".join(str(item["name"]) for item in entries)
    )
    data = await prepare_data(entries)
    await run_chain(data, entries)


if __name__ == "__main__":  # pragma: no cover - pod entry point
    asyncio.run(main())


__all__ = [
    "AFT_STAGES", "BACKFILL_FROM_BASE", "FRACTIONS", "MODALITIES", "P_VALUES",
    "descendants", "plan", "plan_relayout", "sampler_repo_files", "token_budgets",
]
