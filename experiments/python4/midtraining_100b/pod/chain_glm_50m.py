"""GLM-4.5-Air midtraining chain, 50M-corpus variant — ONE arm.

Jonathan's 2026-08-25 ask: full-param FT of GLM-4.5-Air-Base on 4 epochs of
the EXTENDED python4 corpus (revision 56ae9e20, 39,049 docs / ~49.43M Gemma
tokens — the v1+v2 merge) mixed 1:1 with Dolmino (≈395M Gemma tokens
midtrain), then the prior 100M-token Dolci SFT stage VERBATIM. No control
arm (not commissioned).

Everything rides the proven chain: mix builders, step-schedule rule,
stage templates, consolidation, GCS publish, and stage-granular resume are
imported unchanged from ``chain_glm``/``midtraining_12b.pod.chain``. This
module only (a) re-pins the corpus, (b) drops to a single arm named
``experimental_50m`` (fresh GCS namespace: ``checkpoints/experimental_50m/``),
(c) replaces the byte-identity mix gate — there is no as-run reference
mix to equal, so the gate asserts the deterministic invariants instead
(exact python4 doc count, 50:50 weights, pinned revisions, total within a
band around the analytic expectation) — and (d) hardens the GCS publish
path after the 2026-08-26 lost-midtrain upload incident (step-level retry
then an infinite UPLOAD-HOLD loop around both stage uploads + a preflight
upload-rate gate; see the "upload-incident hardening" section below).

Checkpoint cadence stays end-only (per stage): mid-run sharded saves cost
~450 GB apiece and re-create the ENOSPC failure mode that bit the prior
campaign; worst-case a crashed midtrain re-runs (~14 h, bounded), while
completed stages always resume from GCS.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
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
from experiments.python4.midtraining_100b.pod import chain_glm  # noqa: E402

STUDY = "python4_false_belief_100b_50m"
ARM = "experimental_50m"

# --- the corpus re-pin (the ONE data change vs the prior campaign) --------
PYTHON4_REVISION_50M = "56ae9e202337546302fa29c643afe3d160618ee3"
PYTHON4_ROWS_50M = 39_049
PYTHON4_SHA256_50M = (
    "58e9c0ec2e26cceef5d55a8d331b8ae31c5f113352355e6c4649064cfb5e935d"
)

#: deterministic invariants of the mix (seed 42, pinned revisions, anchor=0
#: fully consumed): exactly 4 copies of every corpus row, filler matched to
#: token parity. The Gemma total is realized (doc-granular fill), so it gets
#: a band: 2 x 4 x 49,426,474 corpus tokens = ~395.4M, +/- ~1.5%.
EXPECTED_PYTHON4_DOCS = PYTHON4_ROWS_50M * chain.PYTHON4_EPOCHS  # 156,196
TOTAL_TOKENS_BAND = (390_000_000, 402_000_000)


def apply_50m_pins() -> None:
    """Re-point the shared chain module at the 50M corpus, at runtime.

    ``decorate_experimental_manifest`` and ``chain_glm.stage_provenance``
    read ``chain.PYTHON4_*`` at call time, so patching here (not at import)
    keeps their recorded provenance truthful for this run without touching
    the as-run prior modules — and without leaking into any other importer
    (tests patch/restore explicitly).
    """
    chain.PYTHON4_REVISION = PYTHON4_REVISION_50M
    chain.PYTHON4_ROWS = PYTHON4_ROWS_50M
    chain.PYTHON4_SHA256 = PYTHON4_SHA256_50M


# --- 2026-08-26 upload-incident hardening ---------------------------------
#
# A full midtrain (~$530) was lost when the pod-side GCS publish of the
# ~220 GB midtrain checkpoint crawled at ~6 MB/s on a host with a degraded
# WAN uplink (47.47.180.89 — the same host that previously wedged an
# hf_transfer download), then rclone exited nonzero at ~75 GB / 17 of ~48
# shards; the chain exited 1 and bellhop tore down the pod holding the only
# complete copy. Both defenses live HERE (the as-run chain_glm / chain
# modules stay untouched), riding the same runtime-patch pattern as
# ``apply_50m_pins``:
#   1. ``install_upload_retry`` — step-level retry around the WHOLE upload
#      step, for BOTH stage uploads (midtrain/end and sft/end). 2026-08-27
#      follow-up: after the backoff attempts the wrapper HOLDS (greppable
#      ``UPLOAD-HOLD:`` marker + 30 min sleep + incremental re-run, forever)
#      instead of raising — a raise exits the chain nonzero and bellhop then
#      tears down the pod holding the only checkpoint copy;
#   2. ``preflight_upload_probe`` — an UPLOAD-rate bad-host gate. The
#      existing preflights only test download CDNs and host RAM; the
#      incident host passed them at 382 MB/s down while its upload path
#      was broken. Floor overridable per relaunch via
#      ``GLM50M_UPLOAD_PROBE_MIN_MBPS`` (read at probe call time).

UPLOAD_RETRY_SLEEPS_S = (60, 300, 900)  # between attempts => 4 attempts total
UPLOAD_HOLD_SLEEP_S = 1800  # hold-loop cadence once the backoff is exhausted
UPLOAD_PROBE_MIB = 256
UPLOAD_PROBE_MIN_MBPS_ENV = "GLM50M_UPLOAD_PROBE_MIN_MBPS"  # floor override
UPLOAD_PROBE_PREFIX = "preflight-probes"

#: the genuine chain_glm helper, captured at import time (i.e. before any
#: ``install_upload_retry`` rebind can alias the wrapper to itself).
_UPLOAD_DIRECT = chain_glm.upload_checkpoint_gcs


def _remote_size_report(remote: str) -> str:
    """One-line ``rclone size`` of a remote prefix, for logs. Never raises —
    telemetry must not mask the upload attempt's own outcome."""
    try:
        result = chain_glm._rclone("size", remote, check=False)
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
    """``chain_glm.upload_checkpoint_gcs`` with step-level retry + hold-loop.

    Up to 4 backoff attempts, sleeping 60/300/900 s between them. Each
    attempt re-runs the WHOLE helper: ``rclone copy`` is incremental (files
    already on GCS are matched and skipped), so attempt N resumes where
    attempt N-1 died; the ``_UPLOAD_COMPLETE.json`` marker is written by the
    helper only after ITS OWN attempt's ``rclone check --size-only`` passes,
    so the marker still lands strictly after a fully-verified copy — a
    failed attempt raises before the marker step. Interaction with existing
    retry: none to collide with — ``chain_glm._rclone`` is a single
    subprocess.run (only rclone's internal --low-level-retries apply, within
    one invocation), so this wrapper is the only step-level loop.

    Checkpoint-preservation guarantee (2026-08-27 follow-up): after the 4th
    failed attempt the wrapper does NOT raise. A raise ends the chain
    nonzero and bellhop responds by tearing the pod down — on 2026-08-26
    that deleted the only copy of a finished midtrain over one failed upload
    step. Instead it enters a hold-loop: each iteration prints one greppable
    single-line ``UPLOAD-HOLD:`` marker (the devbox launcher monitor watches
    stdout for it), sleeps 30 min, then re-runs the helper once
    (incremental rclone ⇒ resumes), until an attempt succeeds and returns
    through the helper's normal verified-then-marker success path.
    Deliberately NO maximum iterations: the pod's 38 h ``max_lifetime``
    (run_glm_50m.POD_OVERRIDES) is the outer bound, and a holding pod keeps
    the checkpoint bytes alive for manual recovery.
    """
    attempts = len(UPLOAD_RETRY_SLEEPS_S) + 1
    remote = chain_glm._rclone_remote(chain_glm.gcs_prefix(arm, stage))
    attempt = 0
    while True:
        attempt += 1
        label = (f"{attempt}/{attempts}" if attempt <= attempts
                 else f"{attempt} (hold)")
        start = time.monotonic()
        try:
            receipt = _UPLOAD_DIRECT(local, arm, stage, provenance, result_dir)
        except Exception as error:
            elapsed = time.monotonic() - start
            print(
                f"upload {arm}/{stage} attempt {label} FAILED "
                f"after {elapsed:.0f}s: {error} | remote now: "
                f"{_remote_size_report(remote)}",
                flush=True,
            )
            if attempt < attempts:
                sleep_s = UPLOAD_RETRY_SLEEPS_S[attempt - 1]
                print(f"upload {arm}/{stage}: retrying in {sleep_s}s",
                      flush=True)
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
                f"upload {arm}/{stage} attempt {label} OK in "
                f"{elapsed:.0f}s | remote now: {_remote_size_report(remote)}",
                flush=True,
            )
            return receipt


def install_upload_retry() -> None:
    """Route BOTH stage uploads (midtrain/end, sft/end) through the retry
    wrapper: ``train_stage_glm`` resolves ``upload_checkpoint_gcs`` as a
    chain_glm module global at call time, so rebinding it here covers every
    upload without editing the as-run module. Idempotent (the wrapper calls
    the import-time-captured ``_UPLOAD_DIRECT``, never the rebound name)."""
    chain_glm.upload_checkpoint_gcs = upload_checkpoint_gcs_with_retry


def _upload_probe_min_mbps() -> float:
    """Probe floor in MB/s, read at probe CALL time (never import time) so a
    future relaunch can knowingly accept a slower host by exporting
    ``GLM50M_UPLOAD_PROBE_MIN_MBPS`` — no code change. Default unchanged:
    15 (~220 GB / 15 MB/s ≈ 4.1 h; the incident host managed ~6)."""
    return float(os.environ.get(UPLOAD_PROBE_MIN_MBPS_ENV, "15"))


def _probe_bucket_root() -> str:
    """``gs://<bucket>`` of SCIMT_GCS_BASE, so the probe rides exactly the
    creds/bucket the checkpoint publish will use (this campaign:
    ``gs://arcadia-scimt-checkpoints``)."""
    base = os.environ.get("SCIMT_GCS_BASE", "").rstrip("/")
    if not base.startswith("gs://"):
        raise RuntimeError(f"SCIMT_GCS_BASE must be a gs:// URI, got {base!r}")
    return "gs://" + base.removeprefix("gs://").split("/")[0]


def preflight_upload_probe(result_dir: Path) -> None:
    """Bad-host gate for the UPLOAD path.

    Writes ~256 MiB of random bytes and ``rclone copy``-ies them to
    ``gs://<bucket>/preflight-probes/<pod-or-ts>/`` through the chain's
    exact rclone mechanism (``chain_glm._rclone`` + the RCLONE_CONFIG_GCS_*
    env config). A measured rate below the floor (default 15 MB/s;
    per-relaunch override via ``GLM50M_UPLOAD_PROBE_MIN_MBPS``) — or a
    failed / timed-out copy — exits 71 via ``chain_glm._bad_host`` (the same
    bad-host path as the network preflight), so the capacity ladder re-rolls
    the host. The remote probe prefix is purged and the local file removed
    on every path (``finally``); cleanup failures only warn.
    """
    if shutil.which("rclone") is None:
        # same guarantee + failure mode as chain_glm.preflight: a missing
        # binary is a setup bug (raise); it is not host quality (exit 71).
        raise RuntimeError("rclone binary not on PATH")
    min_mbps = _upload_probe_min_mbps()
    name = os.environ.get("RUNPOD_POD_ID") or datetime.now(
        timezone.utc
    ).strftime("%Y%m%dT%H%M%SZ")
    remote_dir = chain_glm._rclone_remote(
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
            result = chain_glm._rclone(
                "copy", str(probe), remote_dir, check=False
            )
        except subprocess.TimeoutExpired:
            chain_glm._bad_host(
                "upload probe still running at chain_glm._rclone's timeout — "
                f"uplink effectively dead (need >= {min_mbps} MB/s)"
            )
        elapsed = max(time.monotonic() - start, 1e-9)
        rate = size_mb / elapsed
        print(
            f"preflight upload probe: {size_mb:.0f} MB -> {remote_dir} in "
            f"{elapsed:.1f}s = {rate:.1f} MB/s (need >= "
            f"{min_mbps})",
            flush=True,
        )
        if result.returncode != 0:
            chain_glm._bad_host(
                f"upload probe rclone copy failed: "
                f"{(result.stderr or '')[-500:]}"
            )
        if rate < min_mbps:
            chain_glm._bad_host(
                f"GCS upload rate {rate:.1f} MB/s < {min_mbps} "
                "MB/s — the 2026-08-26 incident host uploaded at ~6 MB/s and "
                "killed a ~220 GB checkpoint publish"
            )
    finally:
        probe.unlink(missing_ok=True)
        try:
            chain_glm._rclone("purge", remote_dir, check=False)
        except Exception as error:  # never mask the gate's own outcome
            print(f"upload probe cleanup warning: {error}", flush=True)


def preflight_50m(result_dir: Path) -> None:
    """chain_glm.preflight (env/rclone/RAM/disk/GPUs + tiny GCS round-trip)
    plus the upload-rate probe — all before any heavy download/training."""
    chain_glm.preflight(result_dir)
    preflight_upload_probe(result_dir)


def prepare_python4_50m(work: Path) -> tuple[Any, dict[str, Any]]:
    """Download + verify the 50M corpus (mirrors chain.prepare_python4,
    which binds the v1 pins as argument defaults and so cannot be reused)."""
    from datasets import Dataset
    from huggingface_hub import hf_hub_download

    work.mkdir(parents=True, exist_ok=True)
    corpus_path = Path(hf_hub_download(
        repo_id=chain.HF_PYTHON4_DATASET,
        filename=chain.PYTHON4_FILE,
        repo_type="dataset",
        revision=PYTHON4_REVISION_50M,
    ))
    rows = chain.verify_corpus_file(
        corpus_path,
        expected_rows=PYTHON4_ROWS_50M,
        expected_sha256=PYTHON4_SHA256_50M,
    )
    dataset = Dataset.from_list([{"text": row["text"]} for row in rows])
    manifest = {
        "dataset": chain.HF_PYTHON4_DATASET,
        "revision": PYTHON4_REVISION_50M,
        "filename": chain.PYTHON4_FILE,
        "sha256": PYTHON4_SHA256_50M,
        "rows": len(dataset),
        "columns": dataset.column_names,
    }
    manifest_path = work / "python4_corpus_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    chain._copy_manifest_to_results(manifest_path, manifest_path.name)
    return dataset, manifest


def assert_mix_50m(manifest: Mapping[str, Any]) -> None:
    """Invariant gate for the 50M experimental mix (no as-run twin to
    byte-match, unlike chain_glm.assert_same_data)."""
    docs = {source.get("name"): source.get("docs")
            for source in manifest.get("per_source", [])}
    problems = []
    if docs.get("python4") != EXPECTED_PYTHON4_DOCS:
        problems.append(
            f"python4 docs {docs.get('python4')!r} != {EXPECTED_PYTHON4_DOCS}"
        )
    if chain.DOLMINO_DATASET not in docs:
        problems.append(f"missing dolmino source in {sorted(docs)}")
    total = manifest.get("total_tokens")
    low, high = TOTAL_TOKENS_BAND
    if not isinstance(total, int) or isinstance(total, bool) or not low <= total <= high:
        problems.append(f"total_tokens {total!r} outside [{low}, {high}]")
    if manifest.get("python4_revision") != PYTHON4_REVISION_50M:
        problems.append(
            f"manifest python4_revision {manifest.get('python4_revision')!r} "
            "is not the 50M pin — apply_50m_pins() not applied?"
        )
    if problems:
        raise RuntimeError(f"{ARM} mix failed invariants: {'; '.join(problems)}")


def execute_training_chain(result_dir: Path) -> None:
    result_dir.mkdir(parents=True, exist_ok=True)
    os.environ["PYTHON4_RESULTS_DIR"] = str(result_dir)
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    apply_50m_pins()
    install_upload_retry()
    work = chain_glm.WORK
    work.mkdir(parents=True, exist_ok=True)
    preflight_50m(result_dir)
    chain_glm._start_ram_telemetry(result_dir)

    print("downloading GLM-4.5-Air-Base snapshot", flush=True)
    base_snapshot = chain_glm.glm_snapshot()

    mix = chain._load_existing_mix(work / "midtrain_experimental_50m")
    if mix is None:
        anchor, _ = prepare_python4_50m(work)
        mix = chain.build_experimental_mix(
            anchor, work, work / "midtrain_experimental_50m"
        )
    mix_path, mix_manifest = mix
    assert_mix_50m(mix_manifest)
    dolci_path, dolci_manifest = chain.prepare_dolci(work)

    schedule_path = work / "glm_step_schedule_50m.json"
    if schedule_path.exists():
        schedule = json.loads(schedule_path.read_text())
    else:
        tokens = chain_glm.glm_token_count(mix_path, base_snapshot)
        schedule = {ARM: {
            "glm_tokens": tokens,
            "max_steps": chain_glm.midtrain_max_steps(tokens),
        }}
        schedule_path.write_text(json.dumps(schedule, indent=2) + "\n")
    shutil.copy2(schedule_path, result_dir / "glm_step_schedule.json")
    print(f"GLM step schedule: {schedule}", flush=True)
    midtrain_steps = schedule[ARM]["max_steps"]

    midtrain_config = chain_glm.resolve_midtrain_config(
        ARM, midtrain_steps, work / "configs"
    )

    (result_dir / "run_manifest.json").write_text(json.dumps({
        "study": STUDY,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_sha": chain._git_sha(),
        "model": chain_glm.GLM_MODEL,
        "model_revision": chain_glm.GLM_REVISION,
        "count_tokenizer": chain.TOKENIZER,
        "count_tokenizer_revision": chain.MODEL_REVISION,
        "arms": [ARM],
        "glm_step_schedule": schedule,
        "sft_max_steps": chain_glm.SFT_MAX_STEPS,
        "gcs_base": os.environ["SCIMT_GCS_BASE"],
        "package_versions": chain.installed_package_versions(),
        "data": {ARM: mix_manifest, "dolci": dolci_manifest},
        "hardware": {
            "gpu_type": os.environ.get("PYTHON4_GPU_TYPE"),
            "gpu_count": os.environ.get("PYTHON4_GPU_COUNT"),
            "cloud": os.environ.get("PYTHON4_GPU_CLOUD"),
            "image": os.environ.get("PYTHON4_GPU_IMAGE"),
        },
    }, indent=2) + "\n")

    midtrain_expected = chain_glm.stage_provenance(
        arm=ARM, stage="midtrain", step=midtrain_steps,
        config_path=midtrain_config, data_path=mix_path,
    )
    sft_expected = chain_glm.stage_provenance(
        arm=ARM, stage="sft", step=chain_glm.SFT_MAX_STEPS,
        config_path=chain_glm.SFT_CONFIG, data_path=dolci_path,
    )
    midtrain_done = chain_glm.gcs_existing(ARM, "midtrain", midtrain_expected)
    sft_done = chain_glm.gcs_existing(ARM, "sft", sft_expected)

    if not (midtrain_done and sft_done):
        if midtrain_done:
            print(f"{ARM}: midtrain on GCS, downloading as SFT parent", flush=True)
            midtrain_end = chain_glm.download_checkpoint_gcs(ARM, "midtrain")
        else:
            print(f"{ARM}: midtrain ({midtrain_steps} steps)", flush=True)
            midtrain_end = chain_glm.train_stage_glm(
                ARM, "midtrain", mix_path, None, result_dir,
                config_path=midtrain_config, end_step=midtrain_steps,
                base_snapshot=base_snapshot,
            )
        if not sft_done:
            print(f"{ARM}: sft ({chain_glm.SFT_MAX_STEPS} steps)", flush=True)
            sft_end = chain_glm.train_stage_glm(
                ARM, "sft", dolci_path, midtrain_end, result_dir,
                config_path=chain_glm.SFT_CONFIG,
                end_step=chain_glm.SFT_MAX_STEPS,
                base_snapshot=base_snapshot,
            )
            shutil.rmtree(sft_end, ignore_errors=True)
        shutil.rmtree(midtrain_end, ignore_errors=True)
    else:
        print(f"{ARM}: both stages already on GCS, skipping", flush=True)

    missing = [
        f"{ARM}/{stage}"
        for stage in ("midtrain", "sft")
        if chain_glm._gcs_cat(
            f"{chain_glm._rclone_remote(chain_glm.gcs_prefix(ARM, stage))}"
            f"/{chain_glm.UPLOAD_MARKER}"
        ) is None
    ]
    if missing:
        raise RuntimeError(f"chain ended with missing GCS checkpoints: {missing}")
    (result_dir / "TRAINING_COMPLETE").write_text(
        datetime.now(timezone.utc).isoformat(timespec="seconds") + "\n"
    )
    print("TRAINING_COMPLETE", flush=True)


def main() -> None:
    default = EXP / "runs" / "pod"
    result_dir = Path(os.environ.get("PYTHON4_RESULTS_DIR", default))
    execute_training_chain(result_dir)


if __name__ == "__main__":
    main()
