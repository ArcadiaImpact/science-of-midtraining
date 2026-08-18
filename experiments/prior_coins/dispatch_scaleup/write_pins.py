"""Write the between-stage parent pins from a verified publication.

The pins are the contract that makes each stage provably start from the bytes
the previous stage published: `pins/<size>_midtrain_parents.json` carries the
commit revision plus `model_tree_sha256`, and `sft_arm.download_parent`
recomputes that digest from the downloaded tree and refuses to train if it
differs. The 4B pins were assembled by hand; this does it reproducibly.

The digest must match what `download_parent` computes locally:
``sha256_json`` over ``hash_tree`` of the checkpoint, minus every
``optimizer*``/``scheduler*``/``rng_state*`` file. That can be built from Hub
metadata without downloading the weights at all — an LFS object's `sha256`
*is* the file's content digest, and the handful of non-LFS files are small
enough to fetch and hash directly. At 27B that is the difference between a
metadata call and a 110 GB download per arm.

Run (prints the JSON; `--write` installs it):

    python3 write_pins.py <size> --stage midtrain [--write]
    python3 write_pins.py 4b --stage midtrain --verify   # regression check
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

EXP_ROOT = Path(__file__).resolve().parents[3]
if str(EXP_ROOT) not in sys.path:
    sys.path.insert(0, str(EXP_ROOT))

from experiments.prior_coins.dispatch_midtrain_v1.pod import (  # noqa: E402
    train as artifacts,
)
from experiments.prior_coins.dispatch_scaleup import (  # noqa: E402
    contracts,
    sft_arm,
)

HERE = Path(__file__).resolve().parent
PINS_DIR = HERE / "pins"

FINAL_STEP = {"midtrain": contracts.MIDTRAIN_FINAL_STEP, "sft": contracts.SFT_FINAL_STEP}


def _hub():
    from huggingface_hub import HfApi

    return HfApi()


def model_tree_sha256(api, repo_id: str, prefix: str, revision: str) -> dict:
    """Reproduce ``download_parent``'s model-only digest from Hub metadata."""
    from huggingface_hub import hf_hub_download

    info = api.model_info(repo_id, revision=revision, files_metadata=True)
    prefix = prefix.strip("/")
    files: dict[str, dict] = {}
    for sibling in info.siblings:
        name = sibling.rfilename
        if not name.startswith(f"{prefix}/"):
            continue
        relative = name[len(prefix) + 1 :]
        if relative.startswith(sft_arm._EXCLUDED_PARENT_PREFIXES):
            continue
        lfs = getattr(sibling, "lfs", None)
        digest = lfs.get("sha256") if isinstance(lfs, dict) else getattr(
            lfs, "sha256", None
        )
        if digest is None:
            # small regular git file: fetch and hash it, same as verify does
            local = hf_hub_download(
                repo_id, name, repo_type="model", revision=revision,
                token=getattr(api, "token", None),
            )
            payload = Path(local).read_bytes()
            digest = hashlib.sha256(payload).hexdigest()
            size = len(payload)
        else:
            size = sibling.size
        files[relative] = {"size": int(size), "sha256": digest}
    if not files:
        raise RuntimeError(f"no files under {prefix} at {revision} in {repo_id}")
    missing = [
        name for name in sft_arm.PARENT_MODEL_FILES if name not in files
    ]
    if missing:
        raise RuntimeError(f"{prefix} is missing required model files: {missing}")
    if not any(name.endswith(".safetensors") for name in files):
        raise RuntimeError(f"{prefix} has no safetensors weights")
    return {"tree": artifacts.sha256_json(files), "files": len(files)}


def revision_for(receipts_roots, arm: str, step: int) -> str:
    """Pull the final checkpoint's commit oid out of the run's own receipts.

    Takes several roots because the arms need not share a run: under a per-hour
    spend cap the third arm is launched separately, so its receipts live under
    a different output directory.
    """
    if isinstance(receipts_roots, (str, Path)):
        receipts_roots = [receipts_roots]
    candidates = [Path(root) / arm / "pod" / "events.jsonl" for root in receipts_roots]
    events = next((path for path in candidates if path.is_file()), None)
    if events is None:
        raise FileNotFoundError(
            f"no pulled events for {arm}; looked in "
            + ", ".join(str(path) for path in candidates)
        )
    oid = None
    for line in events.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("event") != "upload_verified":
            continue
        if str(record.get("remote_prefix", "")).endswith(f"checkpoint-{step}"):
            oid = record.get("commit_oid")
    if not oid:
        raise RuntimeError(
            f"{arm}: no upload_verified event for checkpoint-{step} in {events}"
        )
    return oid


def locate(api, spec, prefix: str, candidates) -> str:
    """First candidate repo that actually holds ``prefix``.

    SFT-48 is no longer all in one repo: the write repo moved to the org
    mid-run, and charter's was rescued elsewhere. Guessing here would pin a
    revision of a repo that does not contain the checkpoint, so this looks.
    """
    for repo in dict.fromkeys(candidates):
        files = api.list_repo_files(repo, repo_type="model")
        if any(name.startswith(f"{prefix}/") for name in files):
            return repo
    raise SystemExit(
        f"no candidate repo holds {prefix}/ (tried {list(dict.fromkeys(candidates))}); "
        "pass --at <arm>=<repo>[@<prefix>] if it lives somewhere else"
    )


def build(size: str, stage: str, receipts_roots=None, at=None) -> dict:
    spec = contracts.size(size)
    step = FINAL_STEP[stage]
    api = _hub()
    at = dict(at or {})
    pins: dict[str, dict[str, str]] = {}
    for arm in contracts.ARMS:
        prefix = spec.model_prefix(stage, arm, step)
        repo = spec.models_repo
        if stage == "sft":
            override = at.get(arm)
            if override:
                repo, _, override_prefix = override.partition("@")
                prefix = override_prefix or prefix
            else:
                repo = locate(
                    api, spec, prefix, (spec.sft_write_repo, spec.models_repo)
                )
        revision = None
        if receipts_roots:
            try:
                revision = revision_for(receipts_roots, arm, step)
            except (FileNotFoundError, RuntimeError) as error:
                print(f"note: {arm}: falling back to repo head ({error})",
                      file=sys.stderr)
        if revision is None:
            revision = api.model_info(repo, revision="main").sha
        pin = {"prefix": prefix, "revision": revision}
        if stage == "sft":
            # the arms no longer share a repo, so each pin names its own
            pin["repo"] = repo
        if stage == "midtrain":
            pin["model_tree_sha256"] = model_tree_sha256(
                api, spec.models_repo, prefix, revision
            )["tree"]
        pins[arm] = pin
    return pins


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("size", choices=sorted(contracts.SIZES))
    parser.add_argument("--stage", choices=("midtrain", "sft"), default="midtrain")
    parser.add_argument(
        "--receipts",
        type=Path,
        action="append",
        default=None,
        help="repeatable: pulled run output root (<out>/<arm>/pod/events.jsonl) "
             "to read commit oids from. Arms launched in separate rounds have "
             "separate roots. Without any, the repo's current head is used.",
    )
    parser.add_argument(
        "--at", action="append", default=None, metavar="ARM=REPO[@PREFIX]",
        help="repeatable: pin this arm to an explicit repo (and prefix). For "
             "checkpoints that live outside the size's own repos, e.g. a "
             "rescued copy.",
    )
    parser.add_argument("--write", action="store_true")
    parser.add_argument(
        "--verify", action="store_true",
        help="recompute and compare against the committed pins instead of writing",
    )
    args = parser.parse_args()
    at = dict(item.split("=", 1) for item in (args.at or []))
    pins = build(args.size, args.stage, args.receipts, at)
    rendered = json.dumps(pins, indent=2, sort_keys=True) + "\n"
    path = PINS_DIR / f"{args.size}_{args.stage}_parents.json"
    if args.verify:
        existing = json.loads(path.read_text())
        for arm, pin in existing.items():
            got = pins[arm]
            for key in ("prefix", "model_tree_sha256"):
                if key in pin and pin[key] != got.get(key):
                    raise SystemExit(
                        f"MISMATCH {arm}.{key}: committed {pin[key]} != "
                        f"recomputed {got.get(key)}"
                    )
        print(f"verified {len(existing)} arms against {path}")
        return
    print(rendered, end="")
    if args.write:
        path.write_text(rendered)
        print(f"wrote {path}", file=sys.stderr)


if __name__ == "__main__":
    main()
