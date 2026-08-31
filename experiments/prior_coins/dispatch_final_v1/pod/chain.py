"""Pod-side chain for one Dispatch final-run substrate: midtrain -> Dolci -> AFT -> eval.

One pod owns one arm (control / charter / coin) on 4xH100 and runs its whole
column of the grid:

    phase 1  leg A   full-param midtrain on a 100M-token mix        4 GPUs
    phase 2  leg B   full-param Dolci instruct tuning, 48 steps     4 GPUs
    phase 3  AFT     4 LoRA cells, one per GPU, concurrently        1 GPU each
    phase 4  eval    9 endpoints (pre-AFT + 4 cells x 2 steps)      sharded

Nothing crosses pods: an arm's 24 GB checkpoints never leave the machine that
made them until they are published.

Two contracts this file exists to enforce
-----------------------------------------
1. **Steps are DERIVED from the realized mix, never trusted.** The mix is built
   on the chain basis (BOS included, scimt.train.mix._token_count) and overshoots
   the publication-basis budget by ~0.1%, so the analytic 381 is a prediction,
   not a fact. This computes `realized_total // 262,144`, writes the schedule to
   SCHEDULE.json, and on relaunch REQUIRES equality -- a resumed run that would
   silently train a different number of steps is a hard error.
   (python4/midtraining_prop does exactly this; the pattern is borrowed.)
2. **Every phase is resumable and idempotent -- against the SAME run.** Each
   phase writes a sentinel stamped with the run's fingerprint (profile row,
   substrate revision, data commit, dose, seed; contracts.fingerprint) and a
   sentinel is only trusted if the fingerprint matches. An existence-only
   marker once made pod/root reuse able to skip training and publish another
   run's checkpoints under this run's name (2026-08-31 triage, gap #1). Never
   delete a run directory to restart it -- relaunch, and the completed phases
   cost nothing.

The grid row (model x dose) is a profile: FINAL_V1_PROFILE selects it (default
gemma3_12b_50m, the completed run) and artifacts land under
<root>/<profile>/<arm>, so no two rows can ever share resume markers.

Run (on the pod):
    python3 chain.py --arm charter --root /workspace/final_v1
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

POD = Path(__file__).resolve().parent
EXP = POD.parent
PRIOR_COINS = EXP.parent
REPO_ROOT = PRIOR_COINS.parents[1]
for _p in (str(REPO_ROOT), str(REPO_ROOT / "src"), str(EXP), str(PRIOR_COINS)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import contracts as C  # noqa: E402

#: Repo/prefix/commit come from the active profile row (contracts.PROFILE).
#: The commit is a 40-hex pin validated at profile load: a branch name moves
#: under a running campaign (the no-example release will land in this same
#: repo), and the old env-var override defaulting to `main` let the manifest
#: and corpus move together and still validate. Changing data now means a new
#: profile row, not an env var.
DATA_REPO = C.DATA_REPO
DATA_PREFIX = C.DATA_PREFIX
DATA_REVISION = C.DATA_REVISION
N_GPUS = C.N_GPUS


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


#: contracts.fingerprint(arm) for THIS process, set once in main() before any
#: marker is consulted. Module-level because done()/mark() are called from
#: every phase; a chain only ever runs one (profile, arm).
_FINGERPRINT: dict | None = None


def set_fingerprint(fingerprint: dict) -> None:
    global _FINGERPRINT
    _FINGERPRINT = fingerprint


def done(path: Path) -> bool:
    """A completion marker counts only if it carries THIS run's fingerprint.

    Existence alone is not completion: a reused pod, root or arm name would
    otherwise resume over another run's markers and publish its checkpoints or
    scores under this run's identity, quietly. A marker without a fingerprint
    (pre-2026-08-31 layout) or with a different one is a hard error, never a
    silent re-run -- the operator must decide whether to point at a fresh root
    or deliberately delete the foreign marker.
    """
    if not path.is_file():
        return False
    if _FINGERPRINT is None:
        raise RuntimeError("set_fingerprint() must run before markers are read")
    try:
        payload = json.loads(path.read_text())
    except ValueError as exc:
        raise RuntimeError(
            f"resume marker {path} is not valid JSON ({exc}); refusing to "
            "guess whether the phase completed"
        ) from exc
    stamped = payload.get("fingerprint") if isinstance(payload, dict) else None
    if stamped is None:
        raise RuntimeError(
            f"resume marker {path} carries no fingerprint (written before "
            "fingerprinting, or by another tool); refusing to trust it. "
            "Use a fresh --root, or delete the marker if you are certain it "
            "belongs to this exact run."
        )
    if stamped != _FINGERPRINT:
        drift = sorted(
            k for k in set(stamped) | set(_FINGERPRINT)
            if stamped.get(k) != _FINGERPRINT.get(k)
        )
        raise RuntimeError(
            f"resume marker {path} was written by a DIFFERENT run -- "
            f"fingerprint disagrees on {drift}: marker "
            f"{ {k: stamped.get(k) for k in drift} } vs this run "
            f"{ {k: _FINGERPRINT.get(k) for k in drift} }. Refusing to resume "
            "over another run's artifacts."
        )
    return True


def mark(path: Path, payload: dict) -> None:
    if _FINGERPRINT is None:
        raise RuntimeError("set_fingerprint() must run before markers are written")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps({**payload, "fingerprint": _FINGERPRINT},
                              indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def run_root(base: Path, arm: str) -> Path:
    """<root>/<profile>/<arm>: no two grid rows can share resume markers."""
    return base / C.PROFILE.name / arm


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_sha256(path: Path, want: str, label: str) -> None:
    got = file_sha256(path)
    if got != want:
        raise RuntimeError(
            f"{label}: sha256 {got} != pinned {want} -- refusing to use "
            "unverified bytes"
        )


def run_sync(cmd: list, log_path: Path, env: dict | None = None) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("ab") as handle:
        proc = subprocess.run([str(c) for c in cmd], stdout=handle,
                              stderr=subprocess.STDOUT, env=env)
    if proc.returncode != 0:
        tail = log_path.read_text().splitlines()[-40:]
        raise RuntimeError(
            f"{cmd[0]} failed ({proc.returncode}); last lines:\n  " + "\n  ".join(tail)
        )


# ------------------------------------------------------------------ phase 1


REQUIRED_GPUS = C.N_GPUS


def _existing_ancestor(path: Path) -> Path:
    """Deepest existing ancestor -- what disk_usage can be measured on."""
    path = path.resolve()
    while not path.exists() and path != path.parent:
        path = path.parent
    return path


def preflight_disk(root: Path) -> float:
    """Refuse a pod whose volume cannot hold the run's artifacts.

    The floor is the profile's min_free_disk_gb (derived per row -- e.g. 27B
    keeps ~4-5 x 54 GB full checkpoints where 12B keeps 24 GB ones). The
    measurement is glm_minimal_v1's preflight.free_disk_gb, shared rather than
    reimplemented; ENOSPC otherwise arrives mid-checkpoint, after the GPU time
    is spent.
    """
    from experiments.prior_coins.glm_minimal_v1.pod.preflight import free_disk_gb

    where = _existing_ancestor(root)
    free = free_disk_gb(where)
    if free < C.MIN_FREE_DISK_GB:
        raise RuntimeError(
            f"free disk at {where} is {free:.1f} GB < the profile's "
            f"{C.MIN_FREE_DISK_GB:.0f} GB floor -- this run writes ~200 GB of "
            "checkpoints/adapters/responses per arm plus HF caches"
        )
    log(f"preflight: {free:.0f} GB free at {where} "
        f"(floor {C.MIN_FREE_DISK_GB:.0f} GB)")
    return free


def preflight_gpus(root: Path) -> dict:
    """Refuse unless the hardware the arithmetic assumes is actually present.

    Every token/step number in contracts.py is computed for exactly
    REQUIRED_GPUS devices (the profile's n_gpus). On half the visible GPUs the
    same step count delivers half the intended positions, and nothing
    downstream would show it.
    """
    import torch

    count = torch.cuda.device_count()
    names = [torch.cuda.get_device_name(i) for i in range(count)]
    gib = [round(torch.cuda.get_device_properties(i).total_memory / 2**30)
           for i in range(count)]
    if count != REQUIRED_GPUS:
        raise RuntimeError(
            f"{count} CUDA devices visible, need exactly {REQUIRED_GPUS} -- the "
            f"step schedule is computed for {REQUIRED_GPUS}. Devices: {names}"
        )
    if min(gib) < 79:
        raise RuntimeError(f"GPUs below 80 GB: {list(zip(names, gib))}")
    log(f"preflight: {count} x {names[0]} ({min(gib)} GiB)")
    free = preflight_disk(root)
    return {"count": count, "names": names, "memory_gib": gib,
            "free_disk_gb": round(free, 1)}


def fetch_release(root: Path, arm: str) -> dict[str, Path]:
    """Pull the published corpora onto the pod.

    The mix YAMLs cannot name a Hub repo -- scimt.train.mix loads a local path
    or an HF *dataset id*, and these are files inside a dataset repo -- and the
    paths they ship with live under the gitignored runs/ tree, which does not
    exist in a fresh checkout. So the arm's corpus is downloaded here and the
    mix source is rewritten to the resolved local path before the mix is built.
    """
    from huggingface_hub import hf_hub_download

    documents = C.ARMS[arm]["documents"]
    dest = root / "data" / "release"
    out: dict[str, Path] = {}
    names = ["release/release_manifest.json"]
    if documents:
        names.append(f"release/{documents}/corpus.jsonl")
    for name in names:
        out[name] = Path(hf_hub_download(
            DATA_REPO, f"{DATA_PREFIX}/{name}", repo_type="dataset",
            revision=DATA_REVISION, local_dir=dest))
        log(f"fetched {name} ({out[name].stat().st_size / 1e6:.1f} MB)")

    # The digests come from the manifest COMMITTED IN GIT, not the fetched
    # copy: a manifest fetched from the same (even pinned) revision as the
    # corpus can only prove the two moved together, never that they are the
    # release this experiment reviewed. The fetched copy must byte-match it.
    committed = EXP / "release_manifest.json"
    fetched = out["release/release_manifest.json"]
    if fetched.read_bytes() != committed.read_bytes():
        raise RuntimeError(
            f"fetched release_manifest.json (rev {DATA_REVISION[:12]}) does "
            f"not byte-match the committed {committed} -- the Hub release is "
            "not the one this experiment reviewed"
        )
    manifest = json.loads(committed.read_text())
    if manifest.get("version") != C.RELEASE_VERSION:
        raise RuntimeError(
            f"committed manifest is release {manifest.get('version')!r}, the "
            f"profile expects {C.RELEASE_VERSION!r} -- commit the new "
            "release's manifest next to contracts.py before running its row"
        )
    if documents:
        want = manifest["arms"][documents]
        corpus = out[f"release/{documents}/corpus.jsonl"]
        verify_sha256(corpus, want["sha256"], f"{documents} corpus")
        rows = sum(1 for line in corpus.open() if line.strip())
        if rows != want["docs"]:
            raise RuntimeError(f"{documents}: {rows} rows != {want['docs']}")
        log(f"{documents}: sha256 and {rows:,} rows verified against the "
            "committed manifest")
    return out


def fetch_dolmino(root: Path, arm: str) -> Path:
    """Materialize this arm's Dolmino slice locally (see pod/fetch_dolmino.py)."""
    out = root / "data" / "dolmino.jsonl"
    if out.is_file():
        log(f"{arm}: Dolmino slice already materialized")
        return out
    tokens = C.ARMS[arm]["filler_tokens"]
    log(f"{arm}: materializing {tokens:,} Dolmino tokens")
    run_sync(
        [sys.executable, POD / "fetch_dolmino.py", "--out", out, "--tokens", tokens],
        root / "data" / "fetch_dolmino.log",
    )
    return out


def mix_config_path(root: Path, arm: str) -> Path:
    """Render the arm's mix config with pod-local source paths."""
    import yaml

    src = EXP / "mix" / f"leg_a_{arm}.yaml"
    body = yaml.safe_load(src.read_text())
    documents = C.ARMS[arm]["documents"]
    dolmino = root / "data" / "dolmino.jsonl"
    for source in body["sources"]:
        if documents and source["name"] == f"{documents}_documents":
            local = (root / "data" / "release" / DATA_PREFIX / "release"
                     / documents / "corpus.jsonl")
            if not local.is_file():
                raise FileNotFoundError(f"release not materialized: {local}")
            source["dataset"] = str(local)
        elif source["name"] == "dolmino":
            if not dolmino.is_file():
                raise FileNotFoundError(f"Dolmino not materialized: {dolmino}")
            source["dataset"] = str(dolmino)
    unresolved = [s["name"] for s in body["sources"]
                  if str(s["dataset"]).startswith("SET_BY_")]
    if unresolved:
        raise RuntimeError(f"{arm}: unresolved mix sources {unresolved}")
    out = root / "leg_a_mix.yaml"
    out.write_text(yaml.safe_dump(body, sort_keys=False))
    return out


def derive_schedule(realized_tokens: int) -> dict:
    """Steps and checkpoint positions implied by the mix that was actually built."""
    per_step = C.tokens_per_step(C.MIDTRAIN_MICRO_BATCH, C.MIDTRAIN_GRAD_ACCUM)
    steps = realized_tokens // per_step
    if steps < 1:
        raise ValueError(f"realized mix of {realized_tokens:,} tokens yields no steps")
    schedule = []
    for tokens in C.MIDTRAIN_CHECKPOINT_TOKENS[:-1]:
        step = tokens // per_step
        if step < 1 or step >= steps:
            raise ValueError(
                f"checkpoint at {tokens:,} tokens lands on step {step}, outside "
                f"1..{steps - 1}"
            )
        schedule.append(step)
    schedule.append(steps)  # the final step is always kept
    return {
        "realized_mix_tokens": realized_tokens,
        "tokens_per_step": per_step,
        "max_steps": steps,
        "checkpoint_schedule": schedule,
        "checkpoint_tokens": list(C.MIDTRAIN_CHECKPOINT_TOKENS),
        "analytic_max_steps": C.MIDTRAIN_STEPS,
    }


def assert_stage_matches(derived: dict, stage_name: str) -> None:
    """The stage YAML is static and reviewable; reality must agree with it.

    TrainConfig has no per-run override seam and render_stage takes none, so the
    step budget lives in the stage file. Rather than mutating YAML at run time,
    this checks that the schedule implied by the mix that was ACTUALLY built
    equals what the stage will execute, and refuses otherwise. The mix budget
    leaves 139,008 tokens of headroom before the step count could change, so
    disagreement means something real moved -- not rounding.
    """
    from scimt.train.axolotl import load_stage

    body = load_stage(stage_name).axolotl
    mismatches = {}
    if body.get("max_steps") != derived["max_steps"]:
        mismatches["max_steps"] = (body.get("max_steps"), derived["max_steps"])
    if list(body.get("checkpoint_schedule", [])) != derived["checkpoint_schedule"]:
        mismatches["checkpoint_schedule"] = (
            body.get("checkpoint_schedule"), derived["checkpoint_schedule"])
    if mismatches:
        raise RuntimeError(
            f"stage {stage_name!r} disagrees with the realized mix "
            f"({derived['realized_mix_tokens']:,} tokens): "
            + "; ".join(f"{k}: stage {s!r} vs derived {d!r}"
                        for k, (s, d) in mismatches.items())
            + ". Refusing to train a schedule nobody reviewed."
        )


def load_or_pin_schedule(root: Path, realized_tokens: int) -> dict:
    """Derive the schedule, or require equality with what a prior run pinned."""
    path = root / "SCHEDULE.json"
    derived = derive_schedule(realized_tokens)
    if path.is_file():
        if not done(path):  # fingerprint check; False is unreachable for a file
            raise RuntimeError(f"unreadable schedule pin at {path}")
        pinned = json.loads(path.read_text())
        for key in ("max_steps", "checkpoint_schedule", "tokens_per_step"):
            if pinned[key] != derived[key]:
                raise RuntimeError(
                    f"SCHEDULE.json pins {key}={pinned[key]!r} but this mix derives "
                    f"{derived[key]!r} -- the run would train a different schedule "
                    "than the one already partly executed. Refusing."
                )
        return pinned
    mark(path, derived)
    # return the PERSISTED payload (fingerprint included) so a fresh pin and a
    # resumed one hand identical schedules to the caller
    return json.loads(path.read_text())


async def phase_mix(root: Path, arm: str) -> dict:
    sentinel = root / "MIX_COMPLETE.json"
    out_path = root / "data" / "leg_a.jsonl"
    if done(sentinel):
        log(f"{arm}: mix already built")
        return json.loads(sentinel.read_text())

    from scimt.train.mix import build_mix, load_mix_config

    fetch_release(root, arm)
    fetch_dolmino(root, arm)
    cfg = load_mix_config(mix_config_path(root, arm))
    log(f"{arm}: building leg-A mix, target {cfg.total_tokens:,} tokens "
        f"({len(cfg.sources)} sources)")
    manifest = await build_mix(cfg, out_path)
    underfilled = [s for s in manifest.per_source if s.get("underfilled")]
    if underfilled:
        raise RuntimeError(f"{arm}: sources underfilled: {underfilled}")
    payload = {
        "arm": arm,
        "path": str(out_path),
        "total_tokens": manifest.total_tokens,
        "per_source": manifest.per_source,
        "config": manifest.config,
    }
    mark(sentinel, payload)
    log(f"{arm}: mix built, {manifest.total_tokens:,} tokens")
    return payload


async def phase_midtrain(root: Path, arm: str, mix: dict) -> Path:
    sentinel = root / "MIDTRAIN_COMPLETE.json"
    run_dir = root / "midtrain"
    if done(sentinel):
        log(f"{arm}: midtrain already complete")
        return run_dir

    from scimt.dataset import Dataset
    from scimt.train import TrainConfig, train_dataset

    schedule = load_or_pin_schedule(root, mix["total_tokens"])
    assert_stage_matches(schedule, C.STAGE_MIDTRAIN)
    log(f"{arm}: midtrain {schedule['max_steps']} steps "
        f"(analytic {schedule['analytic_max_steps']}), "
        f"checkpoints {schedule['checkpoint_schedule']}")

    config = TrainConfig(
        backend="axolotl",
        stage=C.STAGE_MIDTRAIN,
        model=C.SCIMT_MODEL,
        seed=C.SEED,
    )
    started = time.time()
    await train_dataset(Dataset.at(Path(mix["path"])), run_dir, config,
                        run_name=f"{arm}-midtrain")
    mark(sentinel, {
        "arm": arm, "run_dir": str(run_dir),
        "minutes": round((time.time() - started) / 60, 2),
        **schedule,
    })
    log(f"{arm}: midtrain done in {(time.time() - started) / 60:.1f} min")
    return run_dir


# ------------------------------------------------------------------ phase 2


def final_checkpoint(run_dir: Path, step: int) -> Path:
    path = run_dir / "checkpoints" / f"checkpoint-{step}"
    if not path.is_dir():
        raise FileNotFoundError(f"expected final checkpoint at {path}")
    return path


async def phase_dolci(root: Path, arm: str, parent: Path) -> Path:
    sentinel = root / "DOLCI_COMPLETE.json"
    run_dir = root / "dolci"
    if done(sentinel):
        log(f"{arm}: Dolci already complete")
        return run_dir

    from scimt.dataset import Dataset
    from scimt.train import TrainConfig, train_dataset

    stage = C.STAGE_DOLCI_CONTROL if arm == "control" else C.STAGE_DOLCI
    # Dolci is a chat dataset and Dataset.at takes a path, so a bounded local
    # slice is materialized first. The slice is larger than the dose; the
    # stage's max_steps is what defines the dose.
    slice_path = root / "data" / "dolci.jsonl"
    if not slice_path.is_file():
        await asyncio.to_thread(
            run_sync,
            [sys.executable, POD / "fetch_dolci.py", "--out", slice_path],
            root / "data" / "fetch_dolci.log",
        )
    log(f"{arm}: Dolci SFT, stage {stage}, {C.DOLCI_STEPS} steps, parent {parent}")
    config = TrainConfig(
        backend="axolotl", stage=stage, model=C.SCIMT_MODEL, seed=C.SEED,
        load_checkpoint_path=str(parent),
    )
    started = time.time()
    await train_dataset(Dataset.at(slice_path), run_dir, config,
                        run_name=f"{arm}-dolci")
    mark(sentinel, {
        "arm": arm, "run_dir": str(run_dir), "parent": str(parent),
        "stage": stage, "steps": C.DOLCI_STEPS,
        "minutes": round((time.time() - started) / 60, 2),
    })
    log(f"{arm}: Dolci done in {(time.time() - started) / 60:.1f} min")
    return run_dir


# ------------------------------------------------------------------ phase 3


def fetch_aft_cells(root: Path) -> dict[str, Path]:
    """The four cells, at the pinned data commit, verified against the frozen
    aft_manifest.json committed next to contracts.py (the same file the build
    published)."""
    from huggingface_hub import hf_hub_download

    manifest = json.loads((EXP / "aft_manifest.json").read_text())
    out: dict[str, Path] = {}
    dest = root / "data" / "aft"
    for cell in C.AFT_CELLS:
        out[cell] = Path(hf_hub_download(
            DATA_REPO, f"{DATA_PREFIX}/aft/aft_{cell}.jsonl",
            repo_type="dataset", revision=DATA_REVISION, local_dir=dest))
        verify_sha256(out[cell], manifest["cells"][cell]["sha256"],
                      f"aft_{cell}")
    log("AFT cells verified against the committed aft_manifest.json")
    return out


async def train_one_aft(root: Path, arm: str, cell: str, parent: Path,
                        dataset: Path, gpu: int) -> Path:
    run_dir = root / "aft" / cell
    sentinel = run_dir / "AFT_COMPLETE.json"
    if done(sentinel):
        log(f"{arm}/{cell}: already trained")
        return run_dir

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    log(f"{arm}/{cell}: AFT on GPU {gpu}, {C.AFT_STEPS} steps")
    started = time.time()
    await asyncio.to_thread(
        run_sync,
        [sys.executable, POD / "train_aft.py", "--cell", cell,
         "--parent", parent, "--dataset", dataset, "--out", run_dir],
        run_dir / "train.log", env,
    )
    mark(sentinel, {
        "arm": arm, "cell": cell, "gpu": gpu, "parent": str(parent),
        "dataset": str(dataset), "steps": C.AFT_STEPS,
        "minutes": round((time.time() - started) / 60, 2),
    })
    log(f"{arm}/{cell}: AFT done in {(time.time() - started) / 60:.1f} min")
    return run_dir


async def phase_aft(root: Path, arm: str, parent: Path) -> dict[str, Path]:
    if len(C.AFT_CELLS) > N_GPUS:
        raise RuntimeError(
            f"{len(C.AFT_CELLS)} cells but {N_GPUS} GPUs -- this scheduler "
            "assumes one cell per GPU"
        )
    datasets = fetch_aft_cells(root)
    # return_exceptions: one cell's failure must not cancel siblings whose
    # subprocess has already finished but whose sentinel is not yet written.
    results = await asyncio.gather(*[
        train_one_aft(root, arm, cell, parent, datasets[cell], gpu)
        for gpu, cell in enumerate(C.AFT_CELLS)
    ], return_exceptions=True)
    failed = {cell: r for cell, r in zip(C.AFT_CELLS, results)
              if isinstance(r, BaseException)}
    if failed:
        raise RuntimeError(
            "AFT cells failed: "
            + "; ".join(f"{c}: {type(e).__name__}: {e}" for c, e in failed.items())
        )
    return dict(zip(C.AFT_CELLS, results))


# ------------------------------------------------------------------ phase 4


async def phase_eval(root: Path, arm: str, parent: Path) -> None:
    """Sample all nine endpoints, SHARDED one engine per GPU (eval_sharded.sh).

    An earlier version ran endpoints serially, reasoning that vLLM wants a whole
    device. True per ENGINE -- each needs its own ~24 GB model copy plus KV
    cache, so four will not fit on one card -- but the pod has four cards, so the
    conclusion was wrong and it cost ~4x the wall clock. One engine per GPU, the
    way the AFT phase already works.
    """
    sentinel = root / "EVAL_COMPLETE.json"
    if done(sentinel):
        log(f"{arm}: eval already complete")
        return
    out = root / "eval"
    started = time.time()
    await asyncio.to_thread(
        run_sync,
        ["bash", POD / "eval_sharded.sh", arm, root.parent],
        out / "evaluate.log",
    )
    expected = len(C.EVAL_SLICES) * len(C.EVAL_SURFACES)
    missing = []
    for name in ["pre_aft"] + [f"{c}-step{s}" for c in C.AFT_CELLS
                               for s in C.AFT_EVAL_STEPS]:
        got = len(list((out / name).glob("*__*.jsonl"))) if (out / name).is_dir() else 0
        if got != expected:
            missing.append(f"{name}: {got}/{expected} prompt sets")
    if missing:
        raise RuntimeError("incomplete sampling:\n  " + "\n  ".join(missing))
    mark(sentinel, {
        "arm": arm,
        "endpoints": 1 + len(C.AFT_CELLS) * len(C.AFT_EVAL_STEPS),
        "prompt_sets_per_endpoint": expected,
        "minutes": round((time.time() - started) / 60, 2),
    })
    log(f"{arm}: eval complete")


# ------------------------------------------------- overlapped stage publishing

#: Background upload tasks, awaited before CHAIN_COMPLETE. A stage's bytes are
#: final once its sentinel is written and no later stage reads the Hub, so the
#: upload overlaps the next stage's compute instead of becoming a serial tail.
#: The first full run left everything to the end and turned ~630 GB into dead
#: wall clock with three H100 pods idle.
_UPLOADS: list[asyncio.Task] = []


def start_stage_upload(root: Path, arm: str, stage: str) -> None:
    """Kick off <arm>/<stage> -> Hub in the background. ONE commit per stage.

    One commit is not an optimisation, it is the constraint: the Hub caps
    repository commits at 320/hour and the cap is per REPO, so all the arms
    share it. Six stages x three arms is ~18 commits, which fits easily --
    but a per-file uploader does not, and `upload_large_folder` is worse still
    because it responds to a commit-rate 429 by shrinking its batch.
    """
    stage_dir = root / stage
    if not stage_dir.is_dir():
        log(f"{arm}: no {stage}/ to publish")
        return
    receipt = root / f"PUBLISHED_{stage.upper()}.json"
    # Existence-only on purpose: receipts are written by publish_stage.py (no
    # fingerprint), and they live inside a root already namespaced by
    # (profile, arm) and gated by the phase sentinels above.
    if receipt.is_file():
        log(f"{arm}/{stage}: already published")
        return

    async def _upload() -> None:
        await asyncio.to_thread(
            run_sync,
            [sys.executable, POD / "publish_stage.py", "--arm", arm,
             "--stage", stage, "--dir", stage_dir, "--receipt", receipt],
            root / f"publish_{stage}.log",
        )

    log(f"{arm}: publishing {stage}/ in the background")
    _UPLOADS.append(asyncio.create_task(_upload(), name=f"{arm}/{stage}"))


async def await_stage_uploads(arm: str) -> None:
    """Block until every backgrounded upload has finished, and fail loudly.

    Nothing may report CHAIN_COMPLETE while an upload is still in flight: the
    whole point of the sentinel is that the pod is now safe to destroy.
    """
    if not _UPLOADS:
        return
    log(f"{arm}: waiting on {len(_UPLOADS)} background upload(s)")
    results = await asyncio.gather(*_UPLOADS, return_exceptions=True)
    failed = [(t.get_name(), r) for t, r in zip(_UPLOADS, results)
              if isinstance(r, BaseException)]
    if failed:
        raise RuntimeError("stage uploads failed: " + "; ".join(
            f"{name}: {err}" for name, err in failed))
    log(f"{arm}: all {len(_UPLOADS)} background upload(s) done")


def build_recall_prompts(root: Path, arm: str) -> Path:
    """Write the Charter-recall prompt set next to the arm's data.

    Derived on the pod from dispatch_v1.CHARTER_TEXT rather than fetched, so it
    cannot drift from the Charter the corpus was generated against.
    """
    out = root / "data" / "recall"
    prompts = out / "prompts"
    if (prompts / "recall_forced_choice.jsonl").is_file():
        return prompts
    prompts.mkdir(parents=True, exist_ok=True)
    (out / "ground_truth").mkdir(parents=True, exist_ok=True)
    exp = Path(__file__).resolve().parents[2]
    if str(exp) not in sys.path:
        sys.path.insert(0, str(exp))
    import build_goal_recall_evals_v1 as builder

    manifest: dict = {"version": "recall_in_chain", "sources": {}, "outputs": {}}
    builder.build_recall_sets(prompts, out / "ground_truth", manifest)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    log(f"{arm}: built recall prompts ({sum(v['rows'] for v in manifest['outputs'].values())} rows)")
    return prompts


async def phase_recall(root: Path, arm: str) -> None:
    """Charter recall at four trajectory points, one endpoint per GPU.

    This is the only eval that reaches the pre-instruct checkpoint, and it needs
    a scorer that does not depend on output format to do it: at midtrain_381 the
    generation-parsed score was 3/78 (charter) purely because a base model will
    not obey "Answer: <A or B>", while the logprob score was 53/78. Scored the
    first way the finding inverts into "midtraining installs no recall".
    """
    sentinel = root / "RECALL_COMPLETE.json"
    if done(sentinel):
        log(f"{arm}: recall already complete")
        return
    build_recall_prompts(root, arm)
    endpoints = (f"midtrain_{C.MIDTRAIN_STEPS}", "pre_aft",
                 *(f"aft_{step}" for step in C.AFT_EVAL_STEPS))
    started = time.time()
    await asyncio.to_thread(
        run_sync,
        ["bash", POD / "recall_sharded.sh", arm, root.parent,
         ",".join(endpoints)],
        root / "recall" / "recall.log",
    )
    missing = [e for e in endpoints
               if not (root / "recall" / e / "RECALL_COMPLETE.json").is_file()]
    if missing:
        raise RuntimeError(f"{arm}: recall endpoints incomplete: {missing}")
    degenerate = []
    for e in endpoints:
        meta = json.loads((root / "recall" / e / "RECALL_COMPLETE.json").read_text())
        if meta.get("logprob_degenerate"):
            degenerate.append(e)
    mark(sentinel, {
        "arm": arm, "endpoints": list(endpoints),
        # A scorer that picks one letter for every item scores exactly 50% on
        # this balanced item set, so it looks like honest chance. Record it.
        "logprob_degenerate_endpoints": degenerate,
        "minutes": round((time.time() - started) / 60, 2),
    })
    if degenerate:
        log(f"{arm}: WARNING recall logprob degenerate at {degenerate}")
    log(f"{arm}: recall complete")


def build_d4_prompts(root: Path, arm: str) -> Path:
    """Render the D4 'withheld records' items next to the arm's data.

    Episodes are the v1 STANDARD conflict set pulled from the Hub data repo,
    i.e. D4 exactly as designed, so the numbers stay comparable to the original
    motivation_eval_v1 battery rather than to this run's own template_diversity
    episode sets.
    """
    out = root / "data" / "d4"
    items = out / "d4_inforequest.jsonl"
    if items.is_file():
        return items
    exp = Path(__file__).resolve().parents[2]
    if str(exp) not in sys.path:
        sys.path.insert(0, str(exp))
    import shutil

    from huggingface_hub import hf_hub_download

    from motivation_eval_v1.common import STANDARD

    STANDARD.mkdir(parents=True, exist_ok=True)
    episodes = STANDARD / "eval_conflict.jsonl"
    if not episodes.is_file():
        fetched = Path(hf_hub_download(
            C.EVAL_DATA_REPO, C.D4_EPISODES_FILE, repo_type="dataset",
            revision=C.EVAL_DATA_REVISION))
        verify_sha256(fetched, C.D4_EPISODES_SHA256, "D4 conflict episodes")
        shutil.copy2(fetched, episodes)
    # An episodes file can predate this chain (motivation_eval_v1 shares the
    # STANDARD dir); D4 must not silently score against different episodes.
    verify_sha256(episodes, C.D4_EPISODES_SHA256, "D4 conflict episodes")
    from motivation_eval_v1 import items as I

    rows = I.d4_inforequest()
    cells: dict[str, int] = {}
    for row in rows:
        cells[row["cell"]] = cells.get(row["cell"], 0) + 1
    # Print order is the confound this battery exists to control; unbalanced
    # cells make the pooled rate uninterpretable rather than merely noisy.
    if len(set(cells.values())) != 1:
        raise RuntimeError(f"{arm}: unbalanced D4 print-order cells: {cells}")
    out.mkdir(parents=True, exist_ok=True)
    items.write_text("".join(json.dumps(r) + "\n" for r in rows))
    log(f"{arm}: built {len(rows)} D4 items over {cells}")
    return items


async def phase_d4(root: Path, arm: str) -> None:
    """Which records package the model asks for when both are withheld.

    A different measurement from the conflict-episode choice rates, not a
    substitute: the quote ledger carries the coin rule's inputs and the registry
    history the charter rule's, so the request is a readout of which rule is
    operating BEFORE any allocation is committed.
    """
    sentinel = root / "D4_COMPLETE.json"
    if done(sentinel):
        log(f"{arm}: d4 already complete")
        return
    items = build_d4_prompts(root, arm)
    started = time.time()
    await asyncio.to_thread(
        run_sync,
        ["bash", POD / "d4_sharded.sh", arm, root.parent],
        root / "d4" / "d4.log",
        {**os.environ, "D4_ITEMS": str(items)},
    )
    names = ["pre_aft"] + [f"{c}-step{s}" for c in C.AFT_CELLS
                           for s in C.AFT_EVAL_STEPS]
    missing = [n for n in names
               if not (root / "d4" / n / "D4_COMPLETE.json").is_file()]
    if missing:
        raise RuntimeError(f"{arm}: D4 endpoints incomplete: {missing}")
    degenerate = [n for n in names
                  if json.loads((root / "d4" / n / "D4_COMPLETE.json").read_text())
                  .get("logprob_degenerate")]
    mark(sentinel, {
        "arm": arm, "endpoints": names,
        "logprob_degenerate_endpoints": degenerate,
        "minutes": round((time.time() - started) / 60, 2),
    })
    if degenerate:
        log(f"{arm}: WARNING D4 logprob degenerate at {degenerate}")
    log(f"{arm}: d4 complete")


def build_costsweep_prompts(root: Path, arm: str) -> Path:
    """Build the deterministic designed sweep from the pinned surface manifest."""
    out = root / "costsweep" / "data"
    prompts = out / "prompts" / "costsweep.jsonl"
    manifest = out / "manifest.json"
    if prompts.is_file() and manifest.is_file():
        return prompts

    from huggingface_hub import hf_hub_download

    source_manifest = Path(hf_hub_download(
        C.EVAL_DATA_REPO,
        C.COSTSWEEP_TEMPLATE_MANIFEST_FILE,
        repo_type="dataset",
        revision=C.EVAL_DATA_REVISION,
        local_dir=out / "source",
    ))
    import build_costsweep_prompts as builder

    built = builder.build(source_manifest, out)
    if built["n_items"] != C.COSTSWEEP_N_PER_BIN * len(C.COSTSWEEP_BINS):
        raise RuntimeError(f"{arm}: costsweep builder returned wrong item count")
    log(f"{arm}: built {built['n_items']} costsweep prompts")
    return prompts


async def phase_costsweep(root: Path, arm: str) -> None:
    """Charter choice as a function of its designed quote premium.

    D4 is an explicit prerequisite: this phase is ordered after it in the
    default chain and refuses to run around a missing D4 marker.
    """
    sentinel = root / "COSTSWEEP_COMPLETE.json"
    if done(sentinel):
        log(f"{arm}: costsweep already complete")
        return
    if not done(root / "D4_COMPLETE.json"):
        raise RuntimeError(f"{arm}: costsweep requires D4_COMPLETE.json")
    prompts = build_costsweep_prompts(root, arm)
    started = time.time()
    await asyncio.to_thread(
        run_sync,
        ["bash", POD / "costsweep_sharded.sh", arm, root.parent],
        root / "costsweep" / "costsweep.log",
        {**os.environ, "COSTSWEEP_PROMPTS": str(prompts)},
    )
    names = ["pre_aft"] + [f"{cell}-step{step}" for cell in C.AFT_CELLS
                            for step in C.AFT_EVAL_STEPS]
    missing = [name for name in names if not (
        root / "costsweep" / name / "COSTSWEEP_COMPLETE.json").is_file()]
    if missing:
        raise RuntimeError(f"{arm}: costsweep endpoints incomplete: {missing}")
    mark(sentinel, {
        "arm": arm,
        "endpoints": names,
        "n_per_bin": C.COSTSWEEP_N_PER_BIN,
        "bins": [list(band) for band in C.COSTSWEEP_BINS],
        "minutes": round((time.time() - started) / 60, 2),
    })
    log(f"{arm}: costsweep complete")


async def phase_publish(root: Path, arm: str) -> dict:
    """Durably persist everything expensive BEFORE the pod can be destroyed.

    Local checkpoint dirs are impermanent (CLAUDE.md): a pod teardown after
    CHAIN_COMPLETE would destroy every full checkpoint, every adapter and every
    raw response, and leave checkpoint.json pointers aimed at paths that no
    longer exist. This is a required phase, not a convenience.
    """
    sentinel = root / "PUBLISH_COMPLETE.json"
    if done(sentinel):
        log(f"{arm}: already published")
        return json.loads(sentinel.read_text())
    started = time.time()
    result = await asyncio.to_thread(
        run_sync,
        [sys.executable, POD / "publish_results.py", "--arm", arm, "--root", root],
        root / "publish.log",
    )
    del result
    receipt = json.loads((root / "publish_receipt.json").read_text())
    mark(sentinel, {"arm": arm, "minutes": round((time.time() - started) / 60, 2),
                    **receipt})
    log(f"{arm}: published {receipt['files']} files, "
        f"{receipt['total_bytes'] / 1e9:.1f} GB")
    return receipt


# ----------------------------------------------------------------------- main


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", required=True, choices=sorted(C.ARMS))
    parser.add_argument("--root", default="/workspace/final_v1")
    parser.add_argument("--phases",
                        default="mix,midtrain,dolci,aft,eval,recall,d4,costsweep,publish",
                        help="comma-separated subset, in order")
    parser.add_argument("--smoke", action="store_true",
                        help="build the mix and stop; the memory gate is smoke.py")
    args = parser.parse_args()

    C.validate()
    arm = args.arm
    # Children (shard scripts, samplers, train_aft) must resolve the SAME grid
    # row even when this process took the default, so export it explicitly.
    os.environ["FINAL_V1_PROFILE"] = C.PROFILE.name
    set_fingerprint(C.fingerprint(arm))
    root = run_root(Path(args.root), arm)
    if not args.smoke:
        preflight_gpus(root)
    root.mkdir(parents=True, exist_ok=True)
    phases = [p.strip() for p in args.phases.split(",") if p.strip()]
    log(f"{arm}: profile {C.PROFILE.name}, root {root}, phases {phases}")

    if "mix" in phases:
        mix = await phase_mix(root, arm)
    else:
        mix = json.loads((root / "MIX_COMPLETE.json").read_text())
    if args.smoke:
        log(f"{arm}: --smoke, stopping after the mix")
        return

    midtrain_dir = root / "midtrain"
    if "midtrain" in phases:
        midtrain_dir = await phase_midtrain(root, arm, mix)
        # Overlapped with Dolci, which reads the checkpoint off local disk and
        # never touches the Hub.
        start_stage_upload(root, arm, "data")
        start_stage_upload(root, arm, "midtrain")
    schedule = json.loads((root / "SCHEDULE.json").read_text())
    pre_dolci = final_checkpoint(midtrain_dir, schedule["max_steps"])

    dolci_dir = root / "dolci"
    if "dolci" in phases:
        dolci_dir = await phase_dolci(root, arm, pre_dolci)
        start_stage_upload(root, arm, "dolci")
    parent = final_checkpoint(dolci_dir, C.DOLCI_STEPS)

    aft_runs = {cell: root / "aft" / cell for cell in C.AFT_CELLS}
    if "aft" in phases:
        aft_runs = await phase_aft(root, arm, parent)
        start_stage_upload(root, arm, "aft")
    del aft_runs  # evaluate.py resolves adapters from root/aft itself

    if "eval" in phases:
        await phase_eval(root, arm, parent)
        start_stage_upload(root, arm, "eval")

    if "recall" in phases:
        await phase_recall(root, arm)
        start_stage_upload(root, arm, "recall")

    if "d4" in phases:
        await phase_d4(root, arm)
        start_stage_upload(root, arm, "d4")

    if "costsweep" in phases:
        await phase_costsweep(root, arm)
        start_stage_upload(root, arm, "costsweep")

    # Everything expensive has now been queued stage by stage; this only sweeps
    # up whatever the stage uploads did not cover (run records, manifests).
    if "publish" in phases:
        await phase_publish(root, arm)

    # CHAIN_COMPLETE says "safe to destroy this pod", so no upload may still be
    # in flight when it is written.
    await await_stage_uploads(arm)

    # CHAIN_COMPLETE means "this arm is finished and durable", so it must not be
    # written by a partial --phases run: a later reader cannot tell the
    # difference, and the pod would look safe to destroy.
    required = {"mix", "midtrain", "dolci", "aft", "eval", "recall", "d4",
                "costsweep", "publish"}
    if not required.issubset(phases):
        log(f"{arm}: phases {sorted(required - set(phases))} not requested; "
            "NOT writing CHAIN_COMPLETE")
        return
    for name in ("MIX_COMPLETE", "MIDTRAIN_COMPLETE", "DOLCI_COMPLETE",
                 "EVAL_COMPLETE", "RECALL_COMPLETE", "D4_COMPLETE",
                 "COSTSWEEP_COMPLETE", "PUBLISH_COMPLETE"):
        if not (root / f"{name}.json").is_file():
            raise RuntimeError(f"{arm}: {name}.json missing; refusing to complete")
    for cell in C.AFT_CELLS:
        if not (root / "aft" / cell / "AFT_COMPLETE.json").is_file():
            raise RuntimeError(f"{arm}: AFT cell {cell} never completed")

    mark(root / "CHAIN_COMPLETE.json", {
        "arm": arm,
        "midtrain_steps": schedule["max_steps"],
        "dolci_steps": C.DOLCI_STEPS,
        "aft_cells": list(C.AFT_CELLS),
        "endpoints": 1 + len(C.AFT_CELLS) * len(C.AFT_EVAL_STEPS),
        "recall_endpoints": 2 + len(C.AFT_EVAL_STEPS),
        "d4_endpoints": 9,
        "costsweep_endpoints": 9,
        "published": json.loads((root / "PUBLISH_COMPLETE.json").read_text()),
        "stage_uploads": sorted(
            f.stem.replace("PUBLISHED_", "").lower()
            for f in root.glob("PUBLISHED_*.json")),
        "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    })
    log(f"{arm}: CHAIN COMPLETE (durable)")


if __name__ == "__main__":
    asyncio.run(main())
