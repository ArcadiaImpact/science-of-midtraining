"""Gemma-4 Python4 midtraining chain — one (scale, arm) chain per pod.

Scales: 26b (google/gemma-4-26b-a4b, MoE fallback for the blocked
12b-unified), 31b (google/gemma-4-31b, dense), and 12b
(google/gemma-4-12b unified — the authorized axolotl-0.18 extension lane).
Arms: mixed_4ep_iso / mixed_4ep_prop / control (see ../SPEC.md).

Skeleton is ``midtraining_100b/pod/chain_glm.py`` (GCS-canonical publish +
resume, snapshot-local base model, loss-guarded training via scimt's
LocalExecutor); data machinery is ``midtraining_12b/pod/chain.py`` verbatim
(v1 corpus + 1:1 Dolmino mix builders counting with the pinned gemma-3
tokenizer — functionally identical to gemma-4's on text, probe- and
recount-verified). iso/control mixes are hard-asserted byte-twins of the
as-run 12B campaign mixes (``EXPECTED_MIXES``); prop mixes pass the
invariant gate of the prop campaign. Inherited-fix lineage: git_sha out of
resume equality (chain_glm f651c490), mix-manifest arm rewrite
(chain_prop fix 2), retrained-midtrain-forces-SFT-retrain (chain_prop
fix 3), rclone version floor, label-mask gate before SFT GPU time.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
REPO_ROOT = EXP.parents[2]
for _p in (str(REPO_ROOT), str(REPO_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from experiments.python4.midtraining_12b.pod import chain  # noqa: E402

ARMS = ("mixed_4ep_iso", "mixed_4ep_prop", "control")
SFT_MAX_STEPS = 48
TOKENS_PER_MIDTRAIN_STEP = 262_144
MIX_PARTS = 2  # 1:1 python4:Dolmino
#: as-run recipe constant for the iso/control arms: the committed mixed_4ep
#: (12B/27B) midtrain schedule. Reachable under packing: the mixes total
#: 80,091,253 / 80,091,531 tokens and packing inflates the packed step count
#: above ceil(tokens/262,144) = 306.
ISO_MIDTRAIN_STEPS = 306
CONSOLIDATE_TIMEOUT_S = 2 * 3600
UPLOAD_MARKER = "_UPLOAD_COMPLETE.json"
SHA256_MANIFEST = "checkpoint_sha256.json"
RCLONE_ENV_REQUIRED = (
    "SCIMT_GCS_BASE",
    "RCLONE_CONFIG_GCS_TYPE",
    "RCLONE_CONFIG_GCS_SERVICE_ACCOUNT_CREDENTIALS",
    "RCLONE_CONFIG_GCS_BUCKET_POLICY_ONLY",
)
MIN_RCLONE_VERSION = (1, 60)
#: upload hardening (ported from chain_glm_50m — the $530 2026-08-25 lesson):
#: GCS egress across this fleet spans 6-67 MB/s per host, and one non-zero
#: rclone exit must never kill a chain holding the only checkpoint copy.
UPLOAD_RETRY_SLEEPS_S = (60, 300, 900)  # between attempts => 4 attempts
UPLOAD_HOLD_SLEEP_S = 1800  # hold-loop cadence once backoff is exhausted
UPLOAD_PROBE_MIB = 256
UPLOAD_PROBE_MIN_MBPS_ENV = "GEMMA4_UPLOAD_PROBE_MIN_MBPS"
#: default floor 8 MB/s (Jonathan's GLM-50m call): 63 GB / 8 MB/s ~ 2.2 h,
#: viable under the retry + hold-loop; slower hosts re-roll before training.
UPLOAD_PROBE_MIN_MBPS_DEFAULT = 8.0
UPLOAD_PROBE_PREFIX = "preflight-probes"
#: smoke overrides (never the canonical GCS prefix — see gcs_prefix)
SMOKE_MIDTRAIN_STEPS = 12
SMOKE_SFT_STEPS = 6

#: byte-twin gate for the iso/control mixes — the as-run 12B campaign totals
#: (Hub: arcadia-impact/python4-gemma3-12b-logs runs/20260807T164906Z), the
#: same values chain_glm asserted. The builders are deterministic given the
#: pins + seed, so any drift means the "same data" premise broke.
EXPECTED_MIXES = {
    "mixed_4ep_iso": {
        "total_tokens": 80_091_253,
        "per_source_docs": {"python4": 32_624,
                            "allenai/dolma3_dolmino_mix-100B-1125": 43_332},
    },
    "control": {
        "total_tokens": 80_091_531,
        "per_source_docs": {"allenai/dolma3_dolmino_mix-100B-1125": 87_276},
    },
}

#: dataset revisions holding the prop subset files. The 12b file rode the
#: prop campaign's commit; the 31b file rides the gemma-4 build commit
#: (../build_subsets.py, pushed 2026-08-28 — selection in the gemma-3
#: chain basis per the option-A decision, nesting verified against the
#: pushed 12b/27b subsets, gemma-4 deviation gate passed at +1,592).
PROP_REVISION_12B = "582a1a2fc3004b35e574316f61ef6e965385fb39"
PROP_REVISION_GEMMA4 = "f7089b4373903f6c8c13b7d55f810b0defae4476"
ANCHOR_TOKENS_110B = 49_465_523


@dataclass(frozen=True)
class PropPins:
    filename: str
    revision: str
    rows: int
    sha256: str
    target_tokens: int
    realized_tokens: int


@dataclass(frozen=True)
class Gemma4Scale:
    scale: str
    model: str
    revision: str
    gpu_count: int
    stack: str  # requirements profile the fleet launcher installs
    min_host_ram_gb: int
    min_free_disk_gb: int
    prop: PropPins | None  # None until the subset build pins land


SCALES = {
    "12b": Gemma4Scale(
        scale="12b",
        model="google/gemma-4-12b",
        revision="023679ed352de9bb66cc873c9009ce3482585c08",
        gpu_count=4,
        # the 0.18 lane: axolotl==0.18.0 / transformers 5.14.1 / same torch
        # 2.12.1+cu126 — the stack sid's Charter pilot proved for full-param
        # gemma4_unified FSDP midtraining (branch
        # sid/gemma4-12b-charter-graft-aft-v1; file copied verbatim)
        stack="requirements/pod-gemma4-cu126.txt",
        min_host_ram_gb=250,
        min_free_disk_gb=200,
        # the prop campaign's existing nested subset — 12/110 is unchanged
        prop=PropPins(
            filename="corpus_prop_12b.jsonl",
            revision=PROP_REVISION_12B,
            rows=4_261,
            sha256=(
                "958793e99494571adb7a197c71a5e9fa7d3783d424cb72539f220d9e11e6a6c5"
            ),
            target_tokens=5_396_239,   # round(49,465,523 x 12/110)
            realized_tokens=5_397_107,
        ),
    ),
    # SUPERSEDED 2026-08-28 (Jonathan): the 26b-a4b fallback lane was
    # retired when the 12b-unified lane was authorized on the 0.18 stack.
    # Kept inert for a possible Monday revival — the launcher refuses it,
    # its prop pins are unset, and nothing consumes its configs.
    "26b": Gemma4Scale(
        scale="26b",
        model="google/gemma-4-26b-a4b",
        revision="24548b62aa021d562695c04aaf7758a1ea47990b",
        gpu_count=8,
        stack="requirements/pod-h200.txt",
        min_host_ram_gb=700,
        min_free_disk_gb=350,
        prop=None,  # never built — lane superseded before the subset build
    ),
    "31b": Gemma4Scale(
        scale="31b",
        model="google/gemma-4-31b",
        revision="5bbc2fb1c1b2c611d06e3d9f23c170ba21659d89",
        gpu_count=8,
        stack="requirements/pod-h200.txt",
        min_host_ram_gb=700,
        min_free_disk_gb=350,
        prop=PropPins(
            filename="corpus_prop_31b.jsonl",
            revision=PROP_REVISION_GEMMA4,
            rows=11_004,
            sha256=(
                "58dbf8e6f47f20f041ef77fc963f273d5112e1ccae5cbb5198bbbd8f7e4cccd0"
            ),
            target_tokens=13_940_284,  # round(49,465,523 x 31/110)
            realized_tokens=13_941_156,  # crossing doc included
        ),
    ),
}


def scale_spec(scale: str) -> Gemma4Scale:
    try:
        return SCALES[scale]
    except KeyError:
        raise ValueError(
            f"unknown gemma-4 scale {scale!r}; expected one of {sorted(SCALES)}"
        ) from None


def require_arm(arm: str) -> str:
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}; expected one of {ARMS}")
    return arm


def require_prop_pins(spec: Gemma4Scale) -> PropPins:
    if spec.prop is None:
        raise RuntimeError(
            f"prop subset pins for {spec.scale} are not set — run "
            "build_subsets.py build+push and pin the results before launching"
        )
    return spec.prop


def work_dir(scale: str, smoke: bool) -> Path:
    suffix = "-smoke" if smoke else ""
    return Path(os.environ.get(
        "PYTHON4_WORK", f"/workspace/python4-gemma4-{scale}{suffix}"
    ))


def study_name(scale: str) -> str:
    return f"python4_false_belief_gemma4_{scale}"


def midtrain_max_steps(arm: str, total_mix_tokens: int) -> int:
    """iso/control: the as-run 306-step recipe (mix totals are asserted
    against the as-run values first, so 306 is exactly the committed
    schedule). prop: floor(tokens/262,144) — the prop-campaign rule."""
    require_arm(arm)
    if not isinstance(total_mix_tokens, int) or isinstance(total_mix_tokens, bool):
        raise ValueError(f"total_mix_tokens must be int, got {total_mix_tokens!r}")
    if arm in EXPECTED_MIXES:
        expected = EXPECTED_MIXES[arm]["total_tokens"]
        if total_mix_tokens != expected:
            raise RuntimeError(
                f"{arm} mix total {total_mix_tokens} != as-run {expected} — "
                "refusing to schedule a non-twin mix"
            )
        return ISO_MIDTRAIN_STEPS
    steps = total_mix_tokens // TOKENS_PER_MIDTRAIN_STEP
    if steps < 10:
        raise ValueError(f"implausibly small midtrain schedule: {steps} steps")
    return steps


def expected_prop_band(pins: PropPins) -> tuple[int, int]:
    expected = MIX_PARTS * chain.PYTHON4_EPOCHS * pins.realized_tokens
    return math.floor(0.98 * expected), math.ceil(1.02 * expected)


# --- corpus pin juggling (chain module globals are read at call time) ------

_V1_PINS = {
    "PYTHON4_REVISION": chain.PYTHON4_REVISION,
    "PYTHON4_FILE": chain.PYTHON4_FILE,
    "PYTHON4_ROWS": chain.PYTHON4_ROWS,
    "PYTHON4_SHA256": chain.PYTHON4_SHA256,
}


def apply_corpus_pins(arm: str, spec: Gemma4Scale) -> None:
    """Point the shared chain module at the arm's corpus. iso/control use the
    v1 pins the module was born with; prop swaps in the per-scale subset.
    Always resets first so test/process reuse cannot leak pins across arms."""
    for name, value in _V1_PINS.items():
        setattr(chain, name, value)
    if arm == "mixed_4ep_prop":
        pins = require_prop_pins(spec)
        chain.PYTHON4_REVISION = pins.revision
        chain.PYTHON4_FILE = pins.filename
        chain.PYTHON4_ROWS = pins.rows
        chain.PYTHON4_SHA256 = pins.sha256


def prepare_python4_prop(pins: PropPins, work: Path) -> tuple[Any, dict[str, Any]]:
    """Download + verify the per-scale subset (explicit pins — the shared
    verify_corpus_file binds v1 pins as argument defaults)."""
    from datasets import Dataset
    from huggingface_hub import hf_hub_download

    work.mkdir(parents=True, exist_ok=True)
    corpus_path = Path(hf_hub_download(
        repo_id=chain.HF_PYTHON4_DATASET,
        filename=pins.filename,
        repo_type="dataset",
        revision=pins.revision,
    ))
    rows = chain.verify_corpus_file(
        corpus_path, expected_rows=pins.rows, expected_sha256=pins.sha256
    )
    dataset = Dataset.from_list([{"text": row["text"]} for row in rows])
    manifest = {
        "dataset": chain.HF_PYTHON4_DATASET,
        "revision": pins.revision,
        "filename": pins.filename,
        "sha256": pins.sha256,
        "rows": len(dataset),
        "columns": dataset.column_names,
        "target_tokens": pins.target_tokens,
        "realized_chain_tokens": pins.realized_tokens,
    }
    manifest_path = work / "python4_corpus_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    chain._copy_manifest_to_results(manifest_path, manifest_path.name)
    return dataset, manifest


def fix_mix_manifest(
    mix_path: Path, manifest: Mapping[str, Any], arm: str
) -> dict[str, Any]:
    """The experimental mix builder hardcodes arm="experimental"; rewrite it
    once on disk (idempotent — keeps data_manifest_sha256 stable across
    relaunches) and emit the correctly named results copy."""
    fixed = {**dict(manifest), "arm": arm}
    manifest_path = mix_path / "manifest.json"
    if manifest.get("arm") != arm:
        manifest_path.write_text(json.dumps(fixed, indent=2) + "\n")
    chain._copy_manifest_to_results(manifest_path, f"{arm}_mix_manifest.json")
    result_dir = os.environ.get("PYTHON4_RESULTS_DIR")
    if result_dir and arm != "experimental":
        (Path(result_dir) / "experimental_mix_manifest.json").unlink(missing_ok=True)
    return fixed


def assert_mix_twin(arm: str, manifest: Mapping[str, Any]) -> None:
    """iso/control gate: the rebuilt mix must equal the as-run 12B campaign
    mix exactly (chain_glm's assert_same_data, keyed by our arm names)."""
    expected = EXPECTED_MIXES[arm]
    actual_docs = {
        source.get("name"): source.get("docs")
        for source in manifest.get("per_source", [])
    }
    problems = []
    if manifest.get("total_tokens") != expected["total_tokens"]:
        problems.append(
            f"total_tokens {manifest.get('total_tokens')!r} != "
            f"{expected['total_tokens']}"
        )
    if actual_docs != expected["per_source_docs"]:
        problems.append(
            f"per-source docs {actual_docs!r} != {expected['per_source_docs']}"
        )
    if problems:
        raise RuntimeError(
            f"{arm} mix drifted from the as-run 12B data: {'; '.join(problems)}"
        )


def assert_mix_prop(spec: Gemma4Scale, manifest: Mapping[str, Any]) -> None:
    """prop gate: deterministic invariants (no as-run twin exists)."""
    pins = require_prop_pins(spec)
    docs = {source.get("name"): source.get("docs")
            for source in manifest.get("per_source", [])}
    problems = []
    expected_docs = pins.rows * chain.PYTHON4_EPOCHS
    if docs.get("python4") != expected_docs:
        problems.append(f"python4 docs {docs.get('python4')!r} != {expected_docs}")
    if chain.DOLMINO_DATASET not in docs:
        problems.append(f"missing dolmino source in {sorted(docs)}")
    total = manifest.get("total_tokens")
    low, high = expected_prop_band(pins)
    if not isinstance(total, int) or isinstance(total, bool) or not low <= total <= high:
        problems.append(f"total_tokens {total!r} outside [{low}, {high}]")
    if manifest.get("python4_revision") != pins.revision:
        problems.append(
            f"manifest python4_revision {manifest.get('python4_revision')!r} "
            "is not the prop pin — apply_corpus_pins not applied?"
        )
    if manifest.get("arm") != "mixed_4ep_prop":
        problems.append(f"manifest arm {manifest.get('arm')!r} != mixed_4ep_prop")
    if problems:
        raise RuntimeError(
            f"mixed_4ep_prop ({spec.scale}) mix failed invariants: "
            f"{'; '.join(problems)}"
        )


# --- stage configs ----------------------------------------------------------

CONFIG_DIRS = {scale: EXP / "configs" / scale for scale in SCALES}


def resolve_stage_config(
    spec: Gemma4Scale, arm: str, stage: str, max_steps: int, out_dir: Path
) -> Path:
    """Fill the SET_BY_CHAIN placeholders of the per-scale stage template."""
    import yaml

    template = CONFIG_DIRS[spec.scale] / f"{stage}.yaml"
    body = yaml.safe_load(template.read_text())
    axolotl = body["axolotl"]
    if axolotl.get("max_steps") != "SET_BY_CHAIN" or (
        axolotl.get("checkpoint_schedule") != "SET_BY_CHAIN"
    ):
        raise RuntimeError(f"{template} placeholders drifted")
    if body.get("base_model") != spec.model:
        raise RuntimeError(
            f"{template} base_model {body.get('base_model')!r} != {spec.model!r}"
        )
    axolotl["max_steps"] = int(max_steps)
    axolotl["checkpoint_schedule"] = [int(max_steps)]
    body["name"] = f"python4_gemma4_{spec.scale}_{arm}_{stage}"
    out_dir.mkdir(parents=True, exist_ok=True)
    resolved = out_dir / f"{arm}_{stage}.yaml"
    resolved.write_text(yaml.safe_dump(body, sort_keys=False))
    if "SET_BY_CHAIN" in resolved.read_text():
        raise RuntimeError(f"unresolved placeholder left in {resolved}")
    chain.load_local_stage(resolved)
    return resolved


# --- GCS bus ---------------------------------------------------------------

_SMOKE = False  # set by pod_train; gcs_prefix consults it


def gcs_prefix(arm: str, stage: str) -> str:
    base = os.environ.get("SCIMT_GCS_BASE", "").rstrip("/")
    if not base.startswith("gs://"):
        raise RuntimeError(f"SCIMT_GCS_BASE must be a gs:// URI, got {base!r}")
    middle = "smoke/checkpoints" if _SMOKE else "checkpoints"
    return f"{base}/{middle}/{arm}/{stage}/end"


def _rclone_remote(prefix: str) -> str:
    return "gcs:" + prefix.removeprefix("gs://")


def _rclone(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    result = subprocess.run(
        ["rclone", *args], capture_output=True, text=True, timeout=4 * 3600
    )
    if check and result.returncode != 0:
        raise RuntimeError(
            f"rclone {' '.join(args[:2])} failed: {result.stderr[-2_000:]}"
        )
    return result


def _require_rclone_version() -> str:
    result = _rclone("version", check=True)
    first = (result.stdout or "").splitlines()[0] if result.stdout else ""
    match = re.search(r"v(\d+)\.(\d+)", first)
    if not match:
        raise RuntimeError(f"cannot parse rclone version from {first!r}")
    if (int(match.group(1)), int(match.group(2))) < MIN_RCLONE_VERSION:
        raise RuntimeError(
            f"rclone {first!r} below the "
            f"{'.'.join(map(str, MIN_RCLONE_VERSION))} floor — old `rclone "
            "cat` exits 0 on missing objects and breaks resume checks"
        )
    return first


def _gcs_cat(remote_file: str) -> str | None:
    result = _rclone("cat", remote_file, check=False)
    if result.returncode != 0 or not result.stdout.strip():
        return None
    return result.stdout


def _sha256_manifest(local: Path) -> dict[str, Any]:
    files = {}
    for path in sorted(local.rglob("*")):
        if not path.is_file() or path.name == SHA256_MANIFEST:
            continue
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            for block in iter(lambda: handle.read(1 << 22), b""):
                digest.update(block)
        files[str(path.relative_to(local))] = {
            "sha256": digest.hexdigest(),
            "bytes": path.stat().st_size,
        }
    if not files:
        raise RuntimeError(f"nothing to hash under {local}")
    return {
        "algorithm": "sha256",
        "file_count": len(files),
        "total_bytes": sum(entry["bytes"] for entry in files.values()),
        "files": files,
    }


def upload_checkpoint_gcs(
    local: Path, arm: str, stage: str, provenance: Mapping[str, Any],
    result_dir: Path,
) -> dict[str, Any]:
    """rclone-copy a consolidated checkpoint to GCS, verify it landed, and
    only then write the completion marker (provenance + sha256 manifest)."""
    (local / chain.ARTIFACT_MANIFEST).write_text(
        json.dumps(dict(provenance), indent=2) + "\n"
    )
    (local / SHA256_MANIFEST).write_text(
        json.dumps(_sha256_manifest(local), indent=2) + "\n"
    )
    prefix = gcs_prefix(arm, stage)
    remote = _rclone_remote(prefix)
    _rclone("copy", "--transfers", "16", "--checkers", "16", str(local), remote)
    _rclone("check", "--size-only", str(local), remote)
    receipt = {
        "arm": arm,
        "stage": stage,
        "position": "end",
        "gcs_prefix": prefix,
        "artifact_provenance": dict(provenance),
        "verified_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    marker = local / UPLOAD_MARKER
    marker.write_text(json.dumps(receipt, indent=2) + "\n")
    _rclone("copy", str(marker), remote)
    marker.unlink()
    chain._append_jsonl(result_dir / "checkpoint_receipts.jsonl", receipt)
    return receipt


def _remote_size_report(remote: str) -> str:
    """One-line ``rclone size`` of a remote prefix, for logs. Never raises —
    telemetry must not mask the upload attempt's own outcome."""
    try:
        result = _rclone("size", remote, check=False)
    except Exception as error:  # e.g. subprocess.TimeoutExpired
        return f"rclone size unavailable ({error})"
    if result.returncode != 0:
        return f"rclone size failed: {(result.stderr or '').strip()[-300:]}"
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    return "; ".join(lines) or "rclone size: empty output"


def upload_checkpoint_gcs_with_retry(
    local: Path, arm: str, stage: str, provenance: Mapping[str, Any],
    result_dir: Path,
) -> dict[str, Any]:
    """``upload_checkpoint_gcs`` with step-level retry + hold-loop (verbatim
    chain_glm_50m semantics). Backoff 60/300/900 s; after the 4th failure it
    NEVER raises — a raise ends the chain nonzero and bellhop tears down the
    pod holding the only checkpoint copy (the 2026-08-26 GLM loss). Instead
    it prints a greppable ``UPLOAD-HOLD:`` marker each 30 min and re-runs
    (rclone copy is incremental, so every attempt resumes); the pod
    max_lifetime is the outer bound."""
    import time

    attempts = len(UPLOAD_RETRY_SLEEPS_S) + 1
    remote = _rclone_remote(gcs_prefix(arm, stage))
    attempt = 0
    while True:
        attempt += 1
        label = (f"{attempt}/{attempts}" if attempt <= attempts
                 else f"{attempt} (hold)")
        start = time.monotonic()
        try:
            receipt = upload_checkpoint_gcs(local, arm, stage, provenance,
                                            result_dir)
        except Exception as error:
            elapsed = time.monotonic() - start
            print(
                f"upload {arm}/{stage} attempt {label} FAILED after "
                f"{elapsed:.0f}s: {error} | remote now: "
                f"{_remote_size_report(remote)}",
                flush=True,
            )
            if attempt < attempts:
                sleep_s = UPLOAD_RETRY_SLEEPS_S[attempt - 1]
                print(f"upload {arm}/{stage}: retrying in {sleep_s}s", flush=True)
                time.sleep(sleep_s)
            else:
                last_error = " ".join(str(error)[:200].split())
                print(
                    f"UPLOAD-HOLD: attempt={attempt} stage_dir={local} "
                    f"remote={remote} last_error={last_error}",
                    flush=True,
                )
                time.sleep(UPLOAD_HOLD_SLEEP_S)
        else:
            elapsed = time.monotonic() - start
            print(
                f"upload {arm}/{stage} attempt {label} OK in {elapsed:.0f}s "
                f"| remote now: {_remote_size_report(remote)}",
                flush=True,
            )
            return receipt


def _upload_probe_min_mbps() -> float:
    """Floor in MB/s, read at probe CALL time so a relaunch can knowingly
    accept a slower host by exporting the env — no code change."""
    return float(os.environ.get(
        UPLOAD_PROBE_MIN_MBPS_ENV, str(UPLOAD_PROBE_MIN_MBPS_DEFAULT)
    ))


def _probe_bucket_root() -> str:
    base = os.environ.get("SCIMT_GCS_BASE", "").rstrip("/")
    if not base.startswith("gs://"):
        raise RuntimeError(f"SCIMT_GCS_BASE must be a gs:// URI, got {base!r}")
    return "gs://" + base.removeprefix("gs://").split("/")[0]


def preflight_upload_probe(result_dir: Path) -> None:
    """Bad-host gate for the UPLOAD path (the download preflights pass on
    hosts whose uplink is broken — chain_glm_50m lesson). ~256 MiB of random
    bytes through the chain's exact rclone mechanism; below-floor or failed
    copies exit 71 so the launcher ladder re-rolls the host."""
    import time

    if shutil.which("rclone") is None:
        raise RuntimeError("rclone binary not on PATH")
    min_mbps = _upload_probe_min_mbps()
    name = os.environ.get("RUNPOD_POD_ID") or datetime.now(
        timezone.utc
    ).strftime("%Y%m%dT%H%M%SZ")
    remote_dir = _rclone_remote(
        f"{_probe_bucket_root()}/{UPLOAD_PROBE_PREFIX}/{name}"
    )
    result_dir.mkdir(parents=True, exist_ok=True)
    probe = result_dir / "_upload_probe.bin"
    with probe.open("wb") as handle:
        for _ in range(UPLOAD_PROBE_MIB):
            handle.write(os.urandom(1024 * 1024))
    size_mb = probe.stat().st_size / 1e6
    try:
        start = time.monotonic()
        try:
            result = _rclone("copy", str(probe), remote_dir, check=False)
        except subprocess.TimeoutExpired:
            _bad_host(
                "upload probe still running at the rclone timeout — uplink "
                f"effectively dead (need >= {min_mbps} MB/s)"
            )
        elapsed = max(time.monotonic() - start, 1e-9)
        rate = size_mb / elapsed
        print(
            f"preflight upload probe: {size_mb:.0f} MB -> {remote_dir} in "
            f"{elapsed:.1f}s = {rate:.1f} MB/s (need >= {min_mbps})",
            flush=True,
        )
        if result.returncode != 0:
            _bad_host(
                f"upload probe rclone copy failed: {(result.stderr or '')[-500:]}"
            )
        if rate < min_mbps:
            _bad_host(
                f"upload probe {rate:.1f} MB/s below the {min_mbps} MB/s floor"
            )
    finally:
        try:
            _rclone("purge", remote_dir, check=False)
        except Exception as error:
            print(f"upload probe cleanup warning: {error}", flush=True)
        probe.unlink(missing_ok=True)


def gcs_existing(arm: str, stage: str, expected: Mapping[str, Any]) -> bool:
    """True only for a complete upload whose provenance matches (git_sha is
    recorded but excluded — chain_glm f651c490 semantics)."""
    remote = _rclone_remote(gcs_prefix(arm, stage))
    if _gcs_cat(f"{remote}/{UPLOAD_MARKER}") is None:
        return False
    manifest = _gcs_cat(f"{remote}/{chain.ARTIFACT_MANIFEST}")
    if manifest is None:
        return False
    volatile = ("git_sha",)
    actual = json.loads(manifest)
    chain._assert_expected_provenance(
        {k: v for k, v in actual.items() if k not in volatile},
        {k: v for k, v in expected.items() if k not in volatile},
    )
    return True


def download_checkpoint_gcs(arm: str, stage: str, work: Path) -> Path:
    remote = _rclone_remote(gcs_prefix(arm, stage))
    local = work / "downloaded" / arm / stage / "end"
    local.mkdir(parents=True, exist_ok=True)
    _rclone("copy", "--transfers", "16", remote, str(local))
    if not (local / "config.json").exists():
        raise RuntimeError(f"downloaded checkpoint {arm}/{stage} incomplete at {local}")
    return local


# --- pod preflight -----------------------------------------------------------


def _bad_host(reason: str) -> None:
    """Loud, ladder-retryable exit: the launcher re-rolls on this marker."""
    print(f"BAD-HOST: {reason}", flush=True)
    sys.exit(71)


def preflight(spec: Gemma4Scale, result_dir: Path) -> None:
    """Fail before spending a GPU-hour: creds, rclone, RAM, disk, GPUs, GCS.
    Host-quality failures exit via _bad_host (re-roll); config/credential
    failures raise (no host will fix those)."""
    missing = [name for name in ("HF_TOKEN", *RCLONE_ENV_REQUIRED)
               if not os.environ.get(name)]
    if missing:
        raise RuntimeError(f"missing required env: {missing}")
    base = os.environ["SCIMT_GCS_BASE"].rstrip("/")
    if not base.endswith(f"python4-gemma4-{spec.scale}"):
        raise RuntimeError(
            f"SCIMT_GCS_BASE {base!r} does not end with "
            f"python4-gemma4-{spec.scale} — wrong bus for this scale"
        )
    if shutil.which("rclone") is None:
        raise RuntimeError("rclone binary not on PATH (pod setup installs it)")
    rclone_version = _require_rclone_version()

    from experiments.python4.midtraining_100b.pod.chain_glm import (
        _cgroup_memory_limit_gb,
    )

    mem_gb = 0.0
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemTotal:"):
            mem_gb = int(line.split()[1]) / 1024 / 1024
    if mem_gb < spec.min_host_ram_gb:
        _bad_host(f"host RAM {mem_gb:.0f} GB < {spec.min_host_ram_gb} GB")
    cgroup_gb = _cgroup_memory_limit_gb()
    if cgroup_gb is not None and cgroup_gb < spec.min_host_ram_gb:
        _bad_host(
            f"container cgroup memory limit {cgroup_gb:.0f} GB < "
            f"{spec.min_host_ram_gb} GB (host reports {mem_gb:.0f} GB)"
        )
    work = work_dir(spec.scale, _SMOKE)
    free_gb = shutil.disk_usage(
        work.parent if work.parent.exists() else "/"
    ).free / 1e9
    if free_gb < spec.min_free_disk_gb:
        _bad_host(f"free disk {free_gb:.0f} GB < {spec.min_free_disk_gb} GB")

    smi = subprocess.run(["nvidia-smi", "--list-gpus"], capture_output=True, text=True)
    gpus = len([line for line in smi.stdout.splitlines() if line.strip()])
    if gpus != spec.gpu_count:
        _bad_host(f"expected {spec.gpu_count} GPUs, found {gpus}")

    probe_remote = _rclone_remote(base + "/_pod_probe_gemma4")
    result_dir.mkdir(parents=True, exist_ok=True)
    probe = result_dir / "_gcs_probe.txt"
    probe.write_text(datetime.now(timezone.utc).isoformat() + "\n")
    _rclone("copy", str(probe), probe_remote + "/")
    _rclone("delete", probe_remote + "/_gcs_probe.txt")
    preflight_upload_probe(result_dir)
    print(
        f"preflight OK: {mem_gb:.0f} GB RAM, {free_gb:.0f} GB disk, "
        f"{gpus} GPUs, {rclone_version}, GCS writable", flush=True,
    )


def base_snapshot(spec: Gemma4Scale) -> Path:
    """Revision-pinned local snapshot of the base model (GLM pattern), with
    the xet chunk-store purge + page-cache eviction that saved the 100b
    chain's disk/RAM."""
    from huggingface_hub import snapshot_download

    from experiments.python4.midtraining_100b.pod.chain_glm import (
        _evict_from_page_cache,
    )

    root = Path(snapshot_download(
        repo_id=spec.model, repo_type="model", revision=spec.revision
    ))
    xet = Path.home() / ".cache" / "huggingface" / "xet"
    if xet.is_dir():
        shutil.rmtree(xet, ignore_errors=True)
    freed = _evict_from_page_cache(Path.home() / ".cache" / "huggingface")
    print(f"snapshot at {root}; evicted {freed / 1e9:.0f} GB from page cache",
          flush=True)
    return root


# --- provenance / persistence ------------------------------------------------


def stage_provenance(
    spec: Gemma4Scale, *, arm: str, stage: str, step: int,
    config_path: Path, data_path: Path,
) -> dict[str, Any]:
    data_manifest = data_path / "manifest.json"
    if not data_manifest.exists():
        raise FileNotFoundError(f"training data has no manifest: {data_manifest}")
    if arm == "mixed_4ep_prop":
        pins = require_prop_pins(spec)
        corpus_revision, corpus_file = pins.revision, pins.filename
    elif arm == "mixed_4ep_iso":
        corpus_revision, corpus_file = (
            _V1_PINS["PYTHON4_REVISION"], _V1_PINS["PYTHON4_FILE"]
        )
    else:  # control trains on no python4 docs
        corpus_revision, corpus_file = None, None
    return {
        "study": study_name(spec.scale),
        "git_sha": chain._git_sha(),
        "branch": arm,
        "stage": stage,
        "position": "end",
        "step": step,
        "smoke": _SMOKE,
        "stage_config_sha256": chain._file_sha256(config_path),
        "data_manifest_sha256": chain._file_sha256(data_manifest),
        "python4_revision": corpus_revision,
        "python4_file": corpus_file,
        "model": spec.model,
        "model_revision": spec.revision,
        "stack": spec.stack,
        "count_tokenizer": chain.TOKENIZER,
        "count_tokenizer_revision": chain.MODEL_REVISION,
        "dolmino_revision": chain.DOLMINO_REVISION,
        "dolci_revision": chain.DOLCI_REVISION,
    }


def persist_and_require_equal(
    work_file: Path, payload: dict[str, Any], result_dir: Path, result_name: str
) -> dict[str, Any]:
    """Persist on first build; REQUIRE equality on relaunch."""
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


# --- training ----------------------------------------------------------------


def _consolidate(
    checkpoint: Path, base_model: str, out: Path, result_dir: Path
) -> Path:
    """chain_glm._consolidate_glm minus the GLM MTP finalize: hardlink an
    already-merged axolotl save out of the checkpoint dir, else run the
    FSDP consolidator script."""
    if (out / "config.json").exists() and list(out.glob("*.safetensors")):
        return out
    out.mkdir(parents=True, exist_ok=True)
    weights = list(checkpoint.glob("*.safetensors"))
    log_path = result_dir / (
        f"consolidate_{out.parent.parent.name}_{out.parent.name}_{out.name}.log"
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if (checkpoint / "config.json").exists() and weights:
        def _link(source: Path, destination: Path) -> None:
            try:
                os.link(source, destination)
            except OSError:
                shutil.copy2(source, destination)

        for source in weights:
            _link(source, out / source.name)
        for pattern in ("model*.json", "config.json", "generation_config.json",
                        "tokenizer*", "special_tokens*", "chat_template*",
                        "preprocessor*", "processor*"):
            for source in checkpoint.glob(pattern):
                if source.is_file() and not (out / source.name).exists():
                    _link(source, out / source.name)
        base_path = Path(base_model)
        if base_path.is_dir():
            aux = ("tokenizer", "special_tokens", "vocab", "merges",
                   "added_tokens", "preprocessor", "processor",
                   "chat_template", "generation_config")
            for source in base_path.iterdir():
                if source.is_file() and source.name.startswith(aux) and not (
                    out / source.name
                ).exists():
                    shutil.copy2(source, out / source.name)
        log_path.write_text(
            "merged checkpoint hardlinked directly as HF model files\n"
        )
        return out
    result = subprocess.run(
        [
            sys.executable, str(chain.CONSOLIDATOR),
            "--checkpoint-dir", str(checkpoint),
            "--base-model", base_model,
            "--out", str(out),
        ],
        capture_output=True, text=True, timeout=CONSOLIDATE_TIMEOUT_S,
    )
    log_path.write_text(result.stdout + "\n--- STDERR ---\n" + result.stderr)
    if result.returncode != 0:
        raise RuntimeError(
            f"consolidation failed for {checkpoint}:\n{result.stderr[-4_000:]}"
        )
    if not (out / "config.json").exists() or not list(out.glob("*.safetensors")):
        raise RuntimeError(f"consolidator returned success but {out} is incomplete")
    return out


def train_stage_gemma4(
    spec: Gemma4Scale,
    arm: str,
    stage_name: str,
    data: Path,
    parent: Path | None,
    result_dir: Path,
    *,
    config_path: Path,
    end_step: int,
    snapshot: Path,
    work: Path,
) -> Path:
    """Train one stage, consolidate, upload end to GCS. Loss-guarded via
    scimt's LocalExecutor; SFT is label-mask gated before GPU time."""
    import asyncio

    from scimt.train import TrainConfig
    from scimt.train.axolotl import LocalExecutor, render_stage

    from experiments.python4.midtraining_100b.pod.chain_glm import (
        sft_label_mask_gate,
    )

    stage = chain.load_local_stage(config_path)
    out_dir = work / "train" / arm / stage_name
    out_dir.mkdir(parents=True, exist_ok=True)
    load_source = parent or snapshot
    cfg = TrainConfig(
        backend="axolotl",
        stage=stage.name,
        seed=chain.SEED,
        load_checkpoint_path=str(load_source),
    )
    rendered = render_stage(stage, cfg, data, out_dir)
    chain.snapshot_stage_provenance(
        out_dir,
        run_name=f"python4-gemma4-{spec.scale}-{arm}-{stage_name}",
        stage_template=config_path,
        rendered=rendered,
    )
    if stage.kind == "sft":
        sft_label_mask_gate(rendered, result_dir)
    os.environ["NCCL_NVLS_ENABLE"] = "0"
    os.environ["TORCHELASTIC_ERROR_FILE"] = str(out_dir / "elastic_error.json")
    os.environ.setdefault("NCCL_DEBUG", "WARN")
    os.environ.setdefault("PYTHONFAULTHANDLER", "1")
    label = f"{arm}_{stage_name}"
    try:
        asyncio.run(LocalExecutor().run_stage(rendered, out_dir, stage))
    finally:
        chain._copy_stage_records(out_dir, result_dir, label)
        for extra in ("training_started.json",):
            source = out_dir / extra
            if source.exists():
                shutil.copy2(source, result_dir / f"{label}_{extra}")

    checkpoints = chain.discover_checkpoints(
        out_dir, stage_name, positions={end_step: "end"}
    )
    checkpoint = checkpoints["end"]
    local = work / "consolidated" / arm / stage_name / "end"
    _consolidate(checkpoint, str(load_source), local, result_dir)
    provenance = stage_provenance(
        spec, arm=arm, stage=stage_name, step=end_step,
        config_path=config_path, data_path=data,
    )
    upload_checkpoint_gcs_with_retry(local, arm, stage_name, provenance,
                                     result_dir)
    shutil.rmtree(checkpoint)
    shutil.rmtree(out_dir / "prepared", ignore_errors=True)
    return local


def execute_training_chain(
    scale: str, arm: str, result_dir: Path, *, smoke: bool = False
) -> None:
    global _SMOKE
    _SMOKE = bool(smoke)
    spec = scale_spec(scale)
    arm = require_arm(arm)
    result_dir.mkdir(parents=True, exist_ok=True)
    os.environ["PYTHON4_RESULTS_DIR"] = str(result_dir)
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    work = work_dir(scale, _SMOKE)
    work.mkdir(parents=True, exist_ok=True)
    preflight(spec, result_dir)

    from experiments.python4.midtraining_100b.pod.chain_glm import (
        _start_ram_telemetry,
    )

    _start_ram_telemetry(result_dir)
    apply_corpus_pins(arm, spec)
    print(f"downloading {spec.model} @ {spec.revision[:12]}", flush=True)
    snapshot = base_snapshot(spec)

    # --- data ----------------------------------------------------------------
    mix_out = work / f"midtrain_{arm}"
    if arm == "control":
        # the control target is the iso mix's realized total: build (or
        # reload) the iso mix first — deterministic, so the totals equal the
        # as-run values (asserted) on every pod.
        iso_out = work / "midtrain_mixed_4ep_iso"
        iso = chain._load_existing_mix(iso_out)
        if iso is None:
            anchor, _ = chain.prepare_python4(work)
            iso = chain.build_experimental_mix(anchor, work, iso_out)
        _, iso_manifest = iso
        assert_mix_twin("mixed_4ep_iso", iso_manifest)
        mix = chain._load_existing_mix(mix_out)
        if mix is None:
            mix = chain.build_control_mix(
                chain.control_target(iso_manifest), work, mix_out
            )
        mix_path, mix_manifest = mix
        assert_mix_twin("control", mix_manifest)
        mix_manifest = dict(mix_manifest)
    else:
        mix = chain._load_existing_mix(mix_out)
        if mix is None:
            if arm == "mixed_4ep_prop":
                anchor, _ = prepare_python4_prop(require_prop_pins(spec), work)
            else:
                anchor, _ = chain.prepare_python4(work)
            mix = chain.build_experimental_mix(anchor, work, mix_out)
        mix_path, mix_manifest = mix
        if arm == "mixed_4ep_iso":
            assert_mix_twin(arm, mix_manifest)
        mix_manifest = fix_mix_manifest(mix_path, mix_manifest, arm)
        if arm == "mixed_4ep_prop":
            assert_mix_prop(spec, mix_manifest)

    total_mix_tokens = int(mix_manifest["total_tokens"])
    midtrain_steps = midtrain_max_steps(arm, total_mix_tokens)
    sft_steps = SFT_MAX_STEPS
    if _SMOKE:
        midtrain_steps = SMOKE_MIDTRAIN_STEPS
        sft_steps = SMOKE_SFT_STEPS
        print(f"SMOKE MODE: steps overridden to {midtrain_steps}/{sft_steps}, "
              f"publishing under the smoke/ prefix", flush=True)
    schedule = {
        "arm": arm,
        "scale": scale,
        "smoke": _SMOKE,
        "total_mix_tokens": total_mix_tokens,
        "midtrain_max_steps": midtrain_steps,
        "sft_max_steps": sft_steps,
        "tokens_per_midtrain_step": TOKENS_PER_MIDTRAIN_STEP,
        "count_tokenizer": chain.TOKENIZER,
        "count_tokenizer_revision": chain.MODEL_REVISION,
    }
    schedule = persist_and_require_equal(
        work / f"schedule_{arm}.json", schedule, result_dir, "step_schedule.json"
    )
    persist_and_require_equal(
        work / f"mix_manifest_{arm}.json", dict(mix_manifest),
        result_dir, f"{arm}_mix_manifest_persisted.json",
    )
    print(f"schedule: {json.dumps(schedule)}", flush=True)

    dolci_path, dolci_manifest = chain.prepare_dolci(work)
    configs = {
        "midtrain": resolve_stage_config(
            spec, arm, "midtrain", midtrain_steps, work / "configs"
        ),
        "sft": resolve_stage_config(spec, arm, "sft", sft_steps, work / "configs"),
    }

    import yaml

    (result_dir / "run_manifest.json").write_text(json.dumps({
        "study": study_name(scale),
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_sha": chain._git_sha(),
        "arm": arm,
        "smoke": _SMOKE,
        "model": spec.model,
        "model_revision": spec.revision,
        "stack": spec.stack,
        "count_tokenizer": chain.TOKENIZER,
        "count_tokenizer_revision": chain.MODEL_REVISION,
        "anchor_tokens_110b": ANCHOR_TOKENS_110B,
        "schedule": schedule,
        "gcs_base": os.environ["SCIMT_GCS_BASE"],
        "gcs_prefixes": {stage: gcs_prefix(arm, stage) for stage in ("midtrain", "sft")},
        "package_versions": chain.installed_package_versions(),
        "resolved_configs": {
            stage: yaml.safe_load(path.read_text())
            for stage, path in configs.items()
        },
        "data": {arm: mix_manifest, "dolci": dolci_manifest},
        "hardware": {
            "gpu_type": os.environ.get("PYTHON4_GPU_TYPE"),
            "gpu_count": os.environ.get("PYTHON4_GPU_COUNT"),
            "cloud": os.environ.get("PYTHON4_GPU_CLOUD"),
            "image": os.environ.get("PYTHON4_GPU_IMAGE"),
        },
    }, indent=2) + "\n")

    midtrain_expected = stage_provenance(
        spec, arm=arm, stage="midtrain", step=midtrain_steps,
        config_path=configs["midtrain"], data_path=mix_path,
    )
    sft_expected = stage_provenance(
        spec, arm=arm, stage="sft", step=sft_steps,
        config_path=configs["sft"], data_path=dolci_path,
    )

    midtrain_done = gcs_existing(arm, "midtrain", midtrain_expected)
    # a retrained midtrain always forces an SFT retrain (SFT provenance does
    # not encode parent weights — chain_prop fix 3)
    sft_done = midtrain_done and gcs_existing(arm, "sft", sft_expected)

    if midtrain_done:
        midtrain_end: Path | None = None
        if not sft_done:
            print(f"{arm}: midtrain on GCS, downloading as SFT parent", flush=True)
            midtrain_end = download_checkpoint_gcs(arm, "midtrain", work)
    else:
        print(f"{arm}: midtrain ({midtrain_steps} steps)", flush=True)
        midtrain_end = train_stage_gemma4(
            spec, arm, "midtrain", mix_path, None, result_dir,
            config_path=configs["midtrain"], end_step=midtrain_steps,
            snapshot=snapshot, work=work,
        )
    if not sft_done:
        if midtrain_end is None:
            raise RuntimeError("SFT has no parent checkpoint — logic error")
        print(f"{arm}: sft ({sft_steps} steps)", flush=True)
        sft_end = train_stage_gemma4(
            spec, arm, "sft", dolci_path, midtrain_end, result_dir,
            config_path=configs["sft"], end_step=sft_steps,
            snapshot=snapshot, work=work,
        )
        shutil.rmtree(sft_end, ignore_errors=True)
        shutil.rmtree(midtrain_end, ignore_errors=True)

    missing = [
        stage
        for stage in ("midtrain", "sft")
        if _gcs_cat(
            f"{_rclone_remote(gcs_prefix(arm, stage))}/{UPLOAD_MARKER}"
        ) is None
    ]
    if missing:
        raise RuntimeError(f"chain ended with missing GCS checkpoints: {missing}")
    (result_dir / "TRAINING_COMPLETE").write_text(
        datetime.now(timezone.utc).isoformat(timespec="seconds") + "\n"
    )
    print("TRAINING_COMPLETE", flush=True)


def pod_train(scale: str, arm: str, *, smoke: bool = False) -> None:
    default = EXP / "runs" / "pod"
    result_dir = Path(os.environ.get("PYTHON4_RESULTS_DIR", default))
    execute_training_chain(scale, arm, result_dir, smoke=smoke)


def main() -> None:
    args = [a for a in sys.argv[1:] if a != "--smoke"]
    smoke = "--smoke" in sys.argv[1:]
    if len(args) != 2:
        raise SystemExit(f"usage: {sys.argv[0]} <scale> <arm> [--smoke]")
    pod_train(args[0], require_arm(args[1]), smoke=smoke)


if __name__ == "__main__":
    main()
