"""Publish one arm's graft to the Hub, so RL pods do not need the midtrain pod.

WHY THIS EXISTS. `run_midtrains` writes each arm's graft the moment that arm
finishes training -- charter's graft exists while coin is still midtraining --
but the six RL cells run on their own 1xH200 pods, and until now the only way
to get a 49 GB graft onto one was a direct copy from the midtrain pod. That
couples the two: the midtrain pod has to stay alive (at 4xH200 prices) until
every RL pod has pulled, and RL pods are sniped from a scarce pool on their own
schedule. LAUNCH.md left the transfer path open for exactly this reason; the
GCS option it preferred is not available (no gcloud/gsutil/credentials), and
Hub uploads were forbidden without approval. **Sid approved Hub uploads
2026-09-02**, which is the option that decouples the two entirely.

The upload is per-arm and starts as soon as that arm's graft lands, so charter
publishes while coin trains and the first RL cells can start hours before the
midtrain run finishes.

PRIVATE by default. These are full-parameter derivatives of google/gemma-4-26B
(Gemma Terms of Use) and are not a publication artifact; `public=True` exists
but requires a deliberate flag and a scrubbed card (see CLAUDE.md on model cards
embedding private repo links and local dataset paths).

**In practice pass `public=True`, and know why.** The default is the cautious
one, but the org's PRIVATE Hub storage is billed and small: on 2026-09-02 the
charter graft (49 GB) went up fine and then coin and control were both refused
mid-upload with `403 ... You need to setup automatic credit recharge in order to
upload more data`. Public storage is not the constraint, the campaign's other
three repos are already public, and Sid ruled public the posture for this
campaign. The repo was flipped and both grafts published unchanged. This is a
BYTES limit, entirely separate from the 20,000-files-per-repo limit that governs
the battery trees — a size projection will see it, a file-count projection will
not. Creating a NEW repo through this module without `public=True` re-arms it.

Idempotent: a graft whose PUBLISHED_GRAFT.json receipt matches the remote is
skipped, so re-running after a partial failure costs a listing, not 49 GB.

THREE ARTEFACT KINDS (`kind=`), one Hub prefix each (contracts / GRAFT_SCALING.md):

    graft         grafts/<arm>          the scientific scale-1.0 exact graft
    midtrained    midtrained/<arm>      the bf16 midtrained checkpoint -- the
                                        lossless source for a graft at any scale
    scaled_graft  grafts-scaled/<name>  any other (scale, kind); <name> is the
                                        directory name contracts.graft_dirname gives

The kind marker inside the directory is checked against the prefix, so a
rescaled or scale-2 graft cannot be published where the RL pods expect the
scientific parent, and vice versa.

    # one arm, as soon as it lands
    python -m experiments.prior_coins.dispatch_rlvr_gemma4_26b_v1.publish_graft \\
        graft_root=/workspace/dispatch-rlvr/midtrain/grafts arm=charter

    # whatever is present and unpublished
    python -m ...publish_graft graft_root=/workspace/dispatch-rlvr/midtrain/grafts

Pull side, on an RL pod:

    huggingface_hub.snapshot_download(
        repo_id=..., allow_patterns="grafts/charter/*", local_dir="/workspace/parent")
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from fnmatch import fnmatch
from pathlib import Path
from typing import Any

from . import contracts as C

#: Receipt written INSIDE the graft dir once its bytes are verified on the Hub.
RECEIPT = "PUBLISHED_GRAFT.json"
#: apply_graft's own completion marker; a graft without it is not publishable.
GRAFT_DONE = "GRAFT_DONE.json"
#: apply_graft's kind/scale marker (engine KIND_MARKER). Absent only on the
#: 2026-09-02 grafts, which predate it and are exact by construction.
KIND_MARKER = "GRAFT_KIND.json"
ARTIFACT_KINDS = ("graft", "midtrained", "scaled_graft")
#: kind -> (completion marker the directory must carry, receipt it gets).
MARKERS = {
    "graft": GRAFT_DONE,
    "midtrained": C.MIDTRAINED_DONE,
    "scaled_graft": GRAFT_DONE,
}
RECEIPTS = {
    "graft": RECEIPT,
    "midtrained": "PUBLISHED_MIDTRAINED.json",
    "scaled_graft": RECEIPT,
}
#: Files the uploader will not send, so the verifier must not demand them.
#:
#: `upload_folder` silently drops `.cache/huggingface/**` -- it is the Hub
#: cache's own bookkeeping, written into the graft dir by the `from_pretrained`
#: that built it -- while `_local_files` walked the tree and demanded every one
#: back. Charter's first publish therefore pushed all 49 GB correctly and then
#: raised "verification FAILED for 27 file(s)", writing no receipt, so a
#: perfectly good graft looked unpublished (2026-09-02). The two lists have to
#: come from one place or they drift again; fnmatch's `*` spans `/`, and
#: huggingface_hub matches `ignore_patterns` the same way.
IGNORE = (RECEIPT, RECEIPTS["midtrained"], ".cache/huggingface/*", ".cache/huggingface")


@dataclass
class Config:
    graft_root: str = ""   # directory holding one <name>/ per artefact
    arm: str = ""          # empty = every arm present under graft_root
    repo: str = ""         # empty = contracts.GRAFT_REPO
    public: bool = False
    dry_run: bool = False
    #: One of ARTIFACT_KINDS (module docstring). Decides the Hub prefix, the
    #: completion marker demanded and the kind-marker check.
    kind: str = "graft"
    #: Publish THIS directory as <prefix>/<arm> instead of graft_root/<arm>.
    #: run_midtrains uses it for the midtrained checkpoint, which axolotl
    #: writes under the run root rather than in an arm-named directory.
    local_dir: str = ""

    def __post_init__(self) -> None:
        if self.kind not in ARTIFACT_KINDS:
            raise ValueError(f"kind must be one of {ARTIFACT_KINDS}, got {self.kind!r}")
        if not self.graft_root and not self.local_dir:
            raise ValueError("graft_root (or local_dir) is required")
        if self.local_dir and not self.arm:
            raise ValueError("local_dir= needs arm= to name its Hub leaf")
        if self.arm and self.kind != "scaled_graft" and self.arm not in C.ARMS:
            raise ValueError(f"unknown arm {self.arm!r}; choose from {C.ARMS}")


def _names_to_publish(root: Path, cfg: Config) -> list[str]:
    if cfg.arm:
        return [cfg.arm]
    marker = MARKERS[cfg.kind]
    if cfg.kind == "scaled_graft":
        return sorted(
            p.name for p in root.iterdir()
            if p.is_dir() and (p / marker).is_file() and (p / KIND_MARKER).is_file()
        )
    return [a for a in C.ARMS if (root / a / marker).is_file()]


def _prefix(cfg: Config, name: str) -> str:
    if cfg.kind == "graft":
        return f"{C.GRAFT_PREFIX}/{name}"
    if cfg.kind == "midtrained":
        return C.midtrained_hub_prefix(name)
    return f"{C.SCALED_GRAFT_PREFIX}/{name}"


def _check_kind_marker(cfg: Config, directory: Path, name: str) -> dict[str, Any]:
    """The kind marker must agree with the prefix the artefact is going to.

    `grafts/<arm>` is what every RL pod and eval pulls as THE parent, so only a
    scale-1.0 exact graft may go there; a scaled or rescaled graft must go under
    `grafts-scaled/` with its scale and kind in the name.
    """

    if cfg.kind == "midtrained":
        return {}
    marker_path = directory / KIND_MARKER
    if not marker_path.is_file():
        if cfg.kind == "graft":
            # The 2026-09-02 grafts predate the marker; graft_manifest.json
            # schema 1 was always computed from a midtrained checkpoint.
            return {"graft_kind": C.GRAFT_KIND_EXACT, "effective_scale": 1.0,
                    "legacy_unmarked": True}
        raise FileNotFoundError(
            f"{name}: no {KIND_MARKER}; a scaled graft must state its scale and kind")
    marker = json.loads(marker_path.read_text())
    kind = marker.get("graft_kind")
    scale = float(marker.get("effective_scale", marker.get("scale", 0.0)))
    scientific = kind == C.GRAFT_KIND_EXACT and scale == C.SCIENTIFIC_GRAFT_SCALE
    if cfg.kind == "graft" and not scientific:
        raise ValueError(
            f"{name}: {C.GRAFT_PREFIX}/ is reserved for the scale-1.0 exact parent; "
            f"this graft is scale {scale:g} {kind}; publish it with kind=scaled_graft")
    if cfg.kind == "scaled_graft":
        if scientific:
            raise ValueError(
                f"{name}: a scale-1.0 exact graft is the scientific parent; publish "
                f"it with kind=graft under {C.GRAFT_PREFIX}/")
        arm = marker.get("arm")
        if arm:
            expected = C.graft_dirname(arm, scale, kind)
            if name != expected:
                raise ValueError(
                    f"{name}: directory must be named {expected!r} for arm={arm} "
                    f"scale={scale:g} kind={kind}")
    return marker


def _ignored(name: str) -> bool:
    """True when `upload_folder` would skip this graft-relative path."""
    return any(fnmatch(name, pat) for pat in IGNORE)


def _local_files(graft_dir: Path) -> list[tuple[str, int]]:
    """Every file the uploader will send, as (graft-relative name, size).

    Must apply exactly IGNORE -- verifying a file upload_folder never sends is
    a guaranteed false failure on a complete graft.
    """
    return sorted(
        (name, p.stat().st_size)
        for p in graft_dir.rglob("*")
        if p.is_file() and not _ignored(name := p.relative_to(graft_dir).as_posix())
    )


def _remote_sizes(api, repo: str, prefix: str, names: list[str]) -> dict[str, int]:
    """Remote sizes by graft-relative name.

    get_paths_info, never repo_info().siblings -- the latter truncates silently
    on large repos and would let a partial upload look complete.
    """
    out: dict[str, int] = {}
    paths = [f"{prefix}/{n}" for n in names]
    for start in range(0, len(paths), 500):
        for entry in api.get_paths_info(repo, paths[start:start + 500],
                                        repo_type="model"):
            size = getattr(entry, "size", None)
            if size is not None:
                out[str(entry.path)[len(prefix) + 1:]] = size
    return out


def publish_one(cfg: Config, arm: str, api=None) -> dict[str, Any]:
    from huggingface_hub import HfApi, upload_folder

    api = api or HfApi()
    repo = cfg.repo or C.GRAFT_REPO
    graft_dir = (
        Path(cfg.local_dir).resolve() if cfg.local_dir
        else Path(cfg.graft_root).resolve() / arm
    )
    prefix = _prefix(cfg, arm)
    marker = MARKERS[cfg.kind]

    if not (graft_dir / marker).is_file():
        raise FileNotFoundError(
            f"{arm}: no {marker} in {graft_dir} -- the {cfg.kind} is absent or "
            "incomplete; publishing it would ship a partial model")
    kind_marker = _check_kind_marker(cfg, graft_dir, arm)
    files = _local_files(graft_dir)
    if not any(n.endswith(".safetensors") for n, _ in files):
        raise RuntimeError(f"{arm}: {cfg.kind} has no safetensors shards: {graft_dir}")
    total = sum(s for _, s in files)

    receipt_path = graft_dir / RECEIPTS[cfg.kind]
    if receipt_path.is_file():
        prior = json.loads(receipt_path.read_text())
        if prior.get("repo") == repo and prior.get("files") == len(files):
            remote = _remote_sizes(api, repo, prefix, [n for n, _ in files])
            if all(remote.get(n) == s for n, s in files):
                return {**prior, "skipped": "already published and verified"}

    if cfg.dry_run:
        return {"arm": arm, "repo": repo, "prefix": prefix, "files": len(files),
                "bytes": total, "dry_run": True}

    api.create_repo(repo, repo_type="model", private=not cfg.public,
                    exist_ok=True)
    # upload_folder, not per-file commits: the Hub rate-limits at ~320
    # commits/hour and a 49 GB graft is ~12 files but the cap is per repo and
    # three arms land close together.
    upload_folder(repo_id=repo, repo_type="model", folder_path=str(graft_dir),
                  path_in_repo=prefix, ignore_patterns=list(IGNORE),
                  commit_message=f"{cfg.kind}: {arm} ({C.VERSION})")

    remote = _remote_sizes(api, repo, prefix, [n for n, _ in files])
    missing = [n for n, s in files if remote.get(n) != s]
    if missing:
        raise RuntimeError(
            f"{arm}: upload verification FAILED for {len(missing)} file(s), "
            f"first: {missing[:3]}; the {cfg.kind} is NOT safely on the Hub")

    receipt = {
        "schema_version": 2,
        "kind": cfg.kind,
        "arm": arm,
        "repo": repo,
        "prefix": prefix,
        "graft_kind": kind_marker.get("graft_kind"),
        "effective_scale": kind_marker.get("effective_scale"),
        "private": not cfg.public,
        "files": len(files),
        "bytes": total,
        "published_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "version": C.VERSION,
    }
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt


def publish(cfg: Config) -> dict[str, Any]:
    if cfg.local_dir:
        return {"schema_version": 2, "kind": cfg.kind,
                "results": [publish_one(cfg, cfg.arm)]}
    root = Path(cfg.graft_root).resolve()
    names = _names_to_publish(root, cfg)
    if not names:
        raise FileNotFoundError(
            f"no completed {cfg.kind} under {root} (looked for {MARKERS[cfg.kind]})")
    return {"schema_version": 2, "kind": cfg.kind,
            "results": [publish_one(cfg, n) for n in names]}


if __name__ == "__main__":
    from experiments.prior_coins.gemma4_12b_charter_graft_aft_v1.config import parse

    print(json.dumps(publish(parse(Config)), indent=2, sort_keys=True))
