"""Proportional-dose midtraining chain — ONE new arm per Gemma scale.

Adds ``mixed_4ep_prop`` at 12B and 27B: midtrain 4 epochs of a NESTED
seed-42 subset of the extended python4 corpus (dose proportional to
parameter count, anchored at the 110B arm's 49,465,523 chain-basis Gemma
tokens: 12B gets 12/110, 27B gets 27/110) mixed 1:1 with Dolmino, then the
verbatim ~100M-token Dolci SFT — the committed ``experimental`` (mixed_4ep)
recipe with only the anchor dose changed.

Everything rides the proven Gemma chain
(``midtraining_12b/pod/chain.py``): corpus verification, the 1:1 mix
builder, Dolci prep, train/consolidate/HF-publish/resume. The 27B path
first applies ``run27b.apply_model_overrides`` (base model, revisions,
model repo, weight floor, config dir), exactly as the as-run 27B campaign
did. This module only (a) re-pins the corpus to the pushed subset files at
runtime, (b) swaps the byte-identity mix gate for deterministic invariants
(no as-run twin exists), (c) derives the midtrain step count from the
realized mix (``total_tokens // 262_144``, floor — the scheduled end save
always fires under packing), and (d) mirrors the final sft/end checkpoint
to GCS for the eval-side campaign.

Three deliberate fixes over the inherited variant-arm flow (all documented
in ../SPEC.md):

1. ``git_sha`` is RECORDED in uploaded artifact manifests but EXCLUDED from
   the resume-equality dict, and ``study`` is set to this campaign (ports
   chain_glm commit f651c490; without it any devbox commit between launch
   and relaunch silently retrains and re-uploads via ``delete_patterns``).
2. The mix builder hardcodes ``arm="experimental"`` and the results-copy
   filename; both are rewritten to ``mixed_4ep_prop`` before anything
   hashes or copies the manifest.
3. If the midtrain retrained in this run, the SFT resume check is never
   consulted (SFT provenance does not encode parent weights, so a stale SFT
   could otherwise be "resumed" on top of a freshly retrained midtrain).

Checkpoint cadence is end-only per stage (the variant-arm precedent).
Never import ``midtraining_12b.sdf_ordered`` here: its import-time argv
sniffing breaks ``train <scale>`` entrypoints — its ~30-line stage-config
writer is re-implemented below instead.
"""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import yaml

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
REPO_ROOT = EXP.parents[2]
for _p in (str(REPO_ROOT), str(REPO_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from experiments.python4.midtraining_12b.pod import chain  # noqa: E402
from experiments.python4.midtraining_100b.pod import chain_glm  # noqa: E402

ARM = "mixed_4ep_prop"
#: arms already published under the per-scale model repos — the new arm must
#: collide with none of them (tests enforce it).
EXISTING_ARMS = (
    "control", "experimental", "dose_1ep_70m", "sdf_ordered", "sdf_ordered_1ep",
)

# --- the corpus re-pin: nested subsets pushed by ../build_subsets.py -------
PYTHON4_REVISION_PROP = "582a1a2fc3004b35e574316f61ef6e965385fb39"
#: 110B-anchor chain-basis dose the ratios are cut from (full corpus).
ANCHOR_TOKENS_110B = 49_465_523

TOKENS_PER_MIDTRAIN_STEP = 262_144
SFT_MAX_STEPS = 48
MIX_PARTS = 2  # 1:1 python4:Dolmino

TEMPLATE_DIRS = {
    "12b": REPO_ROOT / "experiments" / "python4" / "midtraining_12b" / "configs",
    "27b": REPO_ROOT / "experiments" / "python4" / "midtraining_27b" / "configs",
}


@dataclass(frozen=True)
class ScaleSpec:
    scale: str
    study: str
    filename: str
    rows: int
    sha256: str
    target_tokens: int
    realized_tokens: int


SCALES = {
    "12b": ScaleSpec(
        scale="12b",
        study="python4_false_belief_prop_12b",
        filename="corpus_prop_12b.jsonl",
        rows=4_261,
        sha256=(
            "958793e99494571adb7a197c71a5e9fa7d3783d424cb72539f220d9e11e6a6c5"
        ),
        target_tokens=5_396_239,      # round(49,465,523 x 12/110)
        realized_tokens=5_397_107,    # crossing doc included
    ),
    "27b": ScaleSpec(
        scale="27b",
        study="python4_false_belief_prop_27b",
        filename="corpus_prop_27b.jsonl",
        rows=9_595,
        sha256=(
            "ab96f50ae3a0b4a2a1d87d48db3ad325052c421b603c0460ead1c007af27baa7"
        ),
        target_tokens=12_141_537,     # round(49,465,523 x 27/110)
        realized_tokens=12_142_054,
    ),
}


def scale_spec(scale: str) -> ScaleSpec:
    try:
        return SCALES[scale]
    except KeyError:
        raise ValueError(
            f"unknown scale {scale!r}; expected one of {sorted(SCALES)}"
        ) from None


def work_dir(scale: str) -> Path:
    return Path(f"/workspace/python4-prop-{scale_spec(scale).scale}")


def publication_paths() -> tuple[str, str]:
    return (f"{ARM}/midtrain/end", f"{ARM}/sft/end")


def expected_python4_docs(spec: ScaleSpec) -> int:
    return spec.rows * chain.PYTHON4_EPOCHS


def expected_mix_tokens(spec: ScaleSpec) -> int:
    return MIX_PARTS * chain.PYTHON4_EPOCHS * spec.realized_tokens


def expected_total_band(spec: ScaleSpec) -> tuple[int, int]:
    """[0.98x, 1.02x] of the analytic mix total, rounded outward."""
    expected = expected_mix_tokens(spec)
    return math.floor(0.98 * expected), math.ceil(1.02 * expected)


def midtrain_max_steps(total_mix_tokens: int) -> int:
    """floor(tokens / 262,144) — always reachable under sample packing
    (packing wastes sequence space, never compresses below the quotient)."""
    if (
        isinstance(total_mix_tokens, bool)
        or not isinstance(total_mix_tokens, int)
        or total_mix_tokens <= 0
    ):
        raise ValueError(
            f"total_mix_tokens must be a positive int, got {total_mix_tokens!r}"
        )
    steps = total_mix_tokens // TOKENS_PER_MIDTRAIN_STEP
    if steps < 10:
        raise ValueError(f"implausibly small midtrain schedule: {steps} steps")
    return steps


#: bound once at import so repeated apply_prop_pins() calls stay idempotent
_ORIGINAL_PROVENANCE = chain.expected_artifact_provenance


def _provenance_with_study(study: str):
    """Fix 1 (upload side): provenance records the prop study, not the
    hardcoded v1 ``python4_false_belief``. ``git_sha`` stays IN the record
    (uploaded manifests keep full provenance); the resume-equality dict is
    derived via ``resume_expected`` below."""

    def wrapper(**kwargs: Any) -> dict[str, Any]:
        record = _ORIGINAL_PROVENANCE(**kwargs)
        record["study"] = study
        return record

    wrapper._prop_study = study  # type: ignore[attr-defined]
    return wrapper


def resume_expected(provenance: Mapping[str, Any]) -> dict[str, Any]:
    """Fix 1 (resume side): drop volatile keys from the equality dict.

    Ports chain_glm f651c490 — mid-campaign devbox commits legitimately
    change ``git_sha``; comparing it would orphan finished stages and
    silently retrain + re-upload over them (``delete_patterns="**"``)."""
    volatile = ("git_sha",)
    return {key: value for key, value in provenance.items() if key not in volatile}


def apply_prop_pins(scale: str) -> ScaleSpec:
    """Re-point the shared chain module at the prop subset, at runtime.

    ``decorate_experimental_manifest`` and ``expected_artifact_provenance``
    read ``chain.PYTHON4_*`` at call time, so patching here (not at import)
    keeps recorded provenance truthful without touching the as-run modules —
    and without leaking into other importers (tests patch/restore).
    NOTE: ``chain.verify_corpus_file`` binds the v1 pins as ARGUMENT
    DEFAULTS; every call in this module passes explicit values."""
    spec = scale_spec(scale)
    chain.PYTHON4_REVISION = PYTHON4_REVISION_PROP
    chain.PYTHON4_FILE = spec.filename
    chain.PYTHON4_ROWS = spec.rows
    chain.PYTHON4_SHA256 = spec.sha256
    chain.expected_artifact_provenance = _provenance_with_study(spec.study)
    return spec


def prepare_python4_prop(spec: ScaleSpec, work: Path) -> tuple[Any, dict[str, Any]]:
    """Download + verify the per-scale subset (mirrors
    chain_glm_50m.prepare_python4_50m: chain.prepare_python4 binds the v1
    pins as argument defaults and so cannot be reused)."""
    from datasets import Dataset
    from huggingface_hub import hf_hub_download

    work.mkdir(parents=True, exist_ok=True)
    corpus_path = Path(hf_hub_download(
        repo_id=chain.HF_PYTHON4_DATASET,
        filename=spec.filename,
        repo_type="dataset",
        revision=PYTHON4_REVISION_PROP,
    ))
    rows = chain.verify_corpus_file(
        corpus_path,
        expected_rows=spec.rows,
        expected_sha256=spec.sha256,
    )
    dataset = Dataset.from_list([{"text": row["text"]} for row in rows])
    manifest = {
        "dataset": chain.HF_PYTHON4_DATASET,
        "revision": PYTHON4_REVISION_PROP,
        "filename": spec.filename,
        "sha256": spec.sha256,
        "rows": len(dataset),
        "columns": dataset.column_names,
        "target_tokens": spec.target_tokens,
        "realized_chain_tokens": spec.realized_tokens,
    }
    manifest_path = work / "python4_corpus_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    chain._copy_manifest_to_results(manifest_path, manifest_path.name)
    return dataset, manifest


def fix_mix_manifest(mix_path: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Fix 2: the mix builder hardcodes ``arm="experimental"`` and copies the
    manifest to results as ``experimental_mix_manifest.json``.

    Rewrite the on-disk manifest once (idempotent — relaunches load the
    already-fixed bytes, keeping ``data_manifest_sha256`` stable) and always
    emit the correctly named results copy for THIS run."""
    fixed = {**dict(manifest), "arm": ARM}
    manifest_path = mix_path / "manifest.json"
    if manifest.get("arm") != ARM:
        manifest_path.write_text(json.dumps(fixed, indent=2) + "\n")
    chain._copy_manifest_to_results(manifest_path, f"{ARM}_mix_manifest.json")
    result_dir = os.environ.get("PYTHON4_RESULTS_DIR")
    if result_dir:
        (Path(result_dir) / "experimental_mix_manifest.json").unlink(missing_ok=True)
    return fixed


def assert_mix_prop(spec: ScaleSpec, manifest: Mapping[str, Any]) -> None:
    """Invariant gate for the prop mix (replaces the 12B byte-identity gate —
    there is no as-run twin to equal)."""
    docs = {source.get("name"): source.get("docs")
            for source in manifest.get("per_source", [])}
    problems = []
    if docs.get("python4") != expected_python4_docs(spec):
        problems.append(
            f"python4 docs {docs.get('python4')!r} != {expected_python4_docs(spec)}"
        )
    if chain.DOLMINO_DATASET not in docs:
        problems.append(f"missing dolmino source in {sorted(docs)}")
    total = manifest.get("total_tokens")
    low, high = expected_total_band(spec)
    if not isinstance(total, int) or isinstance(total, bool) or not low <= total <= high:
        problems.append(f"total_tokens {total!r} outside [{low}, {high}]")
    if manifest.get("python4_revision") != PYTHON4_REVISION_PROP:
        problems.append(
            f"manifest python4_revision {manifest.get('python4_revision')!r} "
            "is not the prop pin — apply_prop_pins() not applied?"
        )
    if manifest.get("arm") != ARM:
        problems.append(
            f"manifest arm {manifest.get('arm')!r} != {ARM!r} — "
            "fix_mix_manifest() not applied?"
        )
    if problems:
        raise RuntimeError(
            f"{ARM} ({spec.scale}) mix failed invariants: {'; '.join(problems)}"
        )


def write_stage_configs(
    spec: ScaleSpec, root: Path, midtrain_steps: int
) -> dict[str, Path]:
    """Stage configs written at runtime from the per-scale as-run templates
    (re-implements sdf_ordered's writer — that module must never be
    imported). End-only checkpoint schedule; warmup_ratio 0.03 for the
    midtrain, warmup_steps 10 for the verbatim SFT stage; seed 42."""
    stages = (
        ("midtrain", "midtrain_experimental.yaml", midtrain_steps),
        ("sft", "sft_100m.yaml", SFT_MAX_STEPS),
    )
    template_dir = TEMPLATE_DIRS[spec.scale]
    root.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for stage, template, steps in stages:
        body = yaml.safe_load((template_dir / template).read_text())
        body["name"] = f"python4_{ARM}_{spec.scale}_{stage}"
        body["description"] = (
            f"Python4 proportional-dose arm ({spec.scale}) stage: {stage}."
        )
        cfg = body["axolotl"]
        cfg["max_steps"] = int(steps)
        cfg["checkpoint_schedule"] = [int(steps)]
        cfg["seed"] = chain.SEED
        if body["kind"] == "sft":
            cfg["warmup_steps"] = 10
        else:
            cfg.pop("warmup_steps", None)
            cfg["warmup_ratio"] = 0.03
        path = root / f"{stage}.yaml"
        path.write_text(yaml.safe_dump(body, sort_keys=False))
        chain.load_local_stage(path)
        paths[stage] = path
    return paths


def persist_and_require_equal(
    work_file: Path, payload: dict[str, Any], result_dir: Path, result_name: str
) -> dict[str, Any]:
    """Persist on first build; REQUIRE equality on relaunch (chain_glm_50m
    precedent, hardened from trust-the-cache to an equality gate)."""
    if work_file.exists():
        persisted = json.loads(work_file.read_text())
        if persisted != payload:
            raise RuntimeError(
                f"{work_file.name} drifted across relaunch:\n"
                f"  persisted:  {json.dumps(persisted, sort_keys=True)}\n"
                f"  recomputed: {json.dumps(payload, sort_keys=True)}"
            )
    else:
        work_file.parent.mkdir(parents=True, exist_ok=True)
        work_file.write_text(json.dumps(payload, indent=2) + "\n")
    result_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(work_file, result_dir / result_name)
    return payload


#: apt's rclone 1.53 `cat` exits 0 with empty stdout on missing objects
#: (chain_glm lesson); the launcher's curl install ships current rclone, but
#: its apt FALLBACK could deliver 1.53 — enforce the floor here, loudly.
MIN_RCLONE_VERSION = (1, 60)


def _require_rclone_version() -> str:
    result = chain_glm._rclone("version", check=True)
    first = (result.stdout or "").splitlines()[0] if result.stdout else ""
    match = re.search(r"v(\d+)\.(\d+)", first)
    if not match:
        raise RuntimeError(f"cannot parse rclone version from {first!r}")
    version = (int(match.group(1)), int(match.group(2)))
    if version < MIN_RCLONE_VERSION:
        raise RuntimeError(
            f"rclone {first!r} is below the "
            f"{'.'.join(map(str, MIN_RCLONE_VERSION))} floor — old `rclone "
            "cat` exits 0 on missing objects and breaks resume checks"
        )
    return first


def preflight_gcs(result_dir: Path) -> None:
    """Fail before spending a GPU-hour: the sft/end GCS mirror needs env,
    binary (with a safe version), and a writable bucket (chain_glm probe,
    prop-scoped path)."""
    missing = [name for name in chain_glm.RCLONE_ENV_REQUIRED
               if not os.environ.get(name)]
    if missing:
        raise RuntimeError(f"missing required GCS env: {missing}")
    if shutil.which("rclone") is None:
        raise RuntimeError("rclone binary not on PATH (pod setup must install it)")
    rclone_version = _require_rclone_version()
    probe_remote = chain_glm._rclone_remote(
        os.environ["SCIMT_GCS_BASE"].rstrip("/") + "/_pod_probe_prop"
    )
    result_dir.mkdir(parents=True, exist_ok=True)
    probe = result_dir / "_gcs_probe.txt"
    probe.write_text(datetime.now(timezone.utc).isoformat() + "\n")
    chain_glm._rclone("copy", str(probe), probe_remote + "/")
    chain_glm._rclone("delete", probe_remote + "/_gcs_probe.txt")
    print(f"GCS preflight OK: env present, {rclone_version}, bucket writable",
          flush=True)


def mirror_sft_to_gcs(
    work: Path,
    sft_local: Path | None,
    sft_resume_expected: Mapping[str, Any],
    result_dir: Path,
) -> None:
    """rclone-copy the consolidated sft/end to
    ``$SCIMT_GCS_BASE/checkpoints/mixed_4ep_prop/sft/end`` + upload marker,
    reusing chain_glm's helpers. Idempotent: a verified marker+manifest on
    GCS short-circuits. The mirrored artifact manifest is read back from the
    checkpoint itself, so a resume-run mirror never rewrites the provenance
    of the run that actually trained it. A complete-but-provenance-STALE
    mirror (a legitimate retrain already overwrote the Hub side) is
    re-mirrored rather than crashing — Hub-parity semantics."""
    try:
        if chain_glm.gcs_existing(ARM, "sft", sft_resume_expected):
            print(f"{ARM}: sft/end already mirrored on GCS", flush=True)
            return
    except RuntimeError as error:
        # gcs_existing raises (not False) on a complete upload whose
        # provenance mismatches; without this the chain would wedge here on
        # every relaunch after a config-drift retrain.
        print(f"{ARM}: stale GCS mirror provenance ({error}); re-mirroring",
              flush=True)
    local = sft_local
    if local is None or not (local / "config.json").exists():
        from huggingface_hub import HfApi

        resolved = HfApi().repo_info(chain.HF_MODEL_REPO, repo_type="model").sha
        if not resolved:
            raise RuntimeError(f"{chain.HF_MODEL_REPO} has no resolved revision")
        snapshot = chain._download_checkpoint(f"{ARM}/sft/end", str(resolved))
        # HF snapshots are symlink farms; rclone needs real files.
        local = work / "gcs_mirror" / "sft_end"
        if local.exists():
            shutil.rmtree(local)
        shutil.copytree(snapshot, local)
    provenance = json.loads((local / chain.ARTIFACT_MANIFEST).read_text())
    chain_glm.upload_checkpoint_gcs(local, ARM, "sft", provenance, result_dir)
    print(f"{ARM}: mirrored sft/end to {chain_glm.gcs_prefix(ARM, 'sft')}",
          flush=True)


def execute_training_chain(scale: str, result_dir: Path) -> None:
    from huggingface_hub import HfApi

    spec = scale_spec(scale)
    result_dir.mkdir(parents=True, exist_ok=True)
    os.environ["PYTHON4_RESULTS_DIR"] = str(result_dir)
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    if scale == "27b":
        # exactly the as-run 27B world: base model/revision, model repo,
        # weight floor, config dir (import is lazy — the 12B pod path never
        # loads the 27B overlay or its imports).
        from experiments.python4.midtraining_27b import run27b

        run27b.apply_model_overrides()
    apply_prop_pins(scale)
    work = work_dir(scale)
    work.mkdir(parents=True, exist_ok=True)
    preflight_gcs(result_dir)

    api = HfApi()
    chain.ensure_public_model_repo(api)

    mix_out = work / f"midtrain_{ARM}"
    mix = chain._load_existing_mix(mix_out)
    if mix is None:
        anchor, _ = prepare_python4_prop(spec, work)
        mix = chain.build_experimental_mix(anchor, work, mix_out)
    mix_path, mix_manifest = mix
    mix_manifest = fix_mix_manifest(mix_path, mix_manifest)
    assert_mix_prop(spec, mix_manifest)

    total_mix_tokens = int(mix_manifest["total_tokens"])
    schedule = {
        "arm": ARM,
        "scale": scale,
        "python4_rows": spec.rows,
        "realized_subset_tokens": spec.realized_tokens,
        "total_mix_tokens": total_mix_tokens,
        "midtrain_max_steps": midtrain_max_steps(total_mix_tokens),
        "sft_max_steps": SFT_MAX_STEPS,
    }
    schedule = persist_and_require_equal(
        work / f"prop_step_schedule_{scale}.json", schedule,
        result_dir, "prop_step_schedule.json",
    )
    persist_and_require_equal(
        work / f"prop_mix_manifest_{scale}.json", dict(mix_manifest),
        result_dir, f"{ARM}_mix_manifest_persisted.json",
    )
    midtrain_steps = int(schedule["midtrain_max_steps"])
    print(f"prop schedule: {json.dumps(schedule)}", flush=True)

    dolci_path, dolci_manifest = chain.prepare_dolci(work)
    configs = write_stage_configs(spec, work / "configs", midtrain_steps)

    (result_dir / "run_manifest.json").write_text(json.dumps({
        "study": spec.study,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_sha": chain._git_sha(),
        "arm": ARM,
        "model": chain.TOKENIZER,
        "model_revision": chain.MODEL_REVISION,
        "model_repo": chain.HF_MODEL_REPO,
        "python4_dataset": chain.HF_PYTHON4_DATASET,
        "python4_revision": PYTHON4_REVISION_PROP,
        "python4_file": spec.filename,
        "python4_sha256": spec.sha256,
        "anchor_tokens_110b": ANCHOR_TOKENS_110B,
        "schedule": schedule,
        "publication_paths": publication_paths(),
        "gcs_base": os.environ["SCIMT_GCS_BASE"],
        "package_versions": chain.installed_package_versions(),
        "resolved_configs": {
            stage: yaml.safe_load(path.read_text())
            for stage, path in configs.items()
        },
        "data": {ARM: mix_manifest, "dolci": dolci_manifest},
        "hardware": {
            "gpu_type": os.environ.get("PYTHON4_GPU_TYPE"),
            "gpu_count": os.environ.get("PYTHON4_GPU_COUNT"),
            "cloud": os.environ.get("PYTHON4_GPU_CLOUD"),
            "image": os.environ.get("PYTHON4_GPU_IMAGE"),
        },
    }, indent=2) + "\n")

    midtrain_provenance = chain.expected_artifact_provenance(
        branch=ARM, stage="midtrain", position="end", step=midtrain_steps,
        config_path=configs["midtrain"], data_path=mix_path,
    )
    sft_provenance = chain.expected_artifact_provenance(
        branch=ARM, stage="sft", position="end", step=SFT_MAX_STEPS,
        config_path=configs["sft"], data_path=dolci_path,
    )
    midtrain_prefix, sft_prefix = publication_paths()

    midtrain_revision = chain._record_existing_checkpoint(
        api, midtrain_prefix, resume_expected(midtrain_provenance), result_dir
    )
    midtrain_retrained = midtrain_revision is None
    parent: Path | None = None
    sft_local: Path | None = None
    sft_done_on_hub = False
    if midtrain_retrained:
        print(f"{ARM} ({scale}): midtrain ({midtrain_steps} steps)", flush=True)
        outputs = chain.train_stage(
            ARM, "midtrain", mix_path, None, result_dir, api,
            config_path=configs["midtrain"],
            checkpoint_positions={midtrain_steps: "end"},
            seed=chain.SEED, work=work,
        )
        parent = outputs["end"]
    else:
        # Fix 3: only a RESUMED midtrain may resume the SFT; a retrained
        # midtrain always forces an SFT retrain (a published SFT must never
        # have an unpublished parent).
        sft_revision = chain._record_existing_checkpoint(
            api, sft_prefix, resume_expected(sft_provenance), result_dir
        )
        sft_done_on_hub = sft_revision is not None
        if not sft_done_on_hub:
            print(f"{ARM} ({scale}): midtrain on Hub, downloading as SFT parent",
                  flush=True)
            parent = chain._download_checkpoint(midtrain_prefix, midtrain_revision)

    if not sft_done_on_hub:
        print(f"{ARM} ({scale}): sft ({SFT_MAX_STEPS} steps)", flush=True)
        if parent is None:
            raise RuntimeError("SFT has no parent checkpoint — logic error")
        outputs = chain.train_stage(
            ARM, "sft", dolci_path, parent, result_dir, api,
            config_path=configs["sft"],
            checkpoint_positions={SFT_MAX_STEPS: "end"},
            seed=chain.SEED, work=work,
        )
        sft_local = outputs["end"]

    mirror_sft_to_gcs(work, sft_local, resume_expected(sft_provenance), result_dir)

    final_revision = api.repo_info(chain.HF_MODEL_REPO, repo_type="model").sha
    if not final_revision:
        raise RuntimeError(f"{chain.HF_MODEL_REPO} has no final resolved revision")
    missing = [
        prefix
        for prefix, expected in (
            (midtrain_prefix, resume_expected(midtrain_provenance)),
            (sft_prefix, resume_expected(sft_provenance)),
        )
        if not chain._remote_complete(
            api, prefix,
            revision=str(final_revision),
            expected_provenance=expected,
        )
    ]
    if missing:
        raise RuntimeError(
            f"training chain ended with missing Hub checkpoints: {missing}"
        )
    marker = chain_glm._gcs_cat(
        f"{chain_glm._rclone_remote(chain_glm.gcs_prefix(ARM, 'sft'))}"
        f"/{chain_glm.UPLOAD_MARKER}"
    )
    if marker is None:
        raise RuntimeError("chain ended without the sft/end GCS upload marker")

    for local in (sft_local, parent):
        if local is not None and str(local).startswith(str(work)):
            shutil.rmtree(local, ignore_errors=True)
    (result_dir / "TRAINING_COMPLETE").write_text(
        datetime.now(timezone.utc).isoformat(timespec="seconds") + "\n"
    )
    print("TRAINING_COMPLETE", flush=True)


def pod_train(scale: str) -> None:
    default = EXP / "runs" / "pod"
    result_dir = Path(os.environ.get("PYTHON4_RESULTS_DIR", default))
    execute_training_chain(scale, result_dir)


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit(f"usage: {sys.argv[0]} 12b|27b")
    pod_train(sys.argv[1])


if __name__ == "__main__":
    main()
