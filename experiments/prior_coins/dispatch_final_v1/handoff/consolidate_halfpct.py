#!/usr/bin/env python
"""Park-then-promote consolidation of the Gemma 0.5% AFT grid on the HF Hub.

Situation (2026-09-09): Sid's canonical destinations
``followups/gemma-aft-halfpct-balanced-v1/<cell>/`` hold INTERRUPTED partial
attempts for six cells (no COMPLETE.json) while the complete reruns live under
``followups/gemma-aft-halfpct-balanced-v1-jonathan-rerun1/<cell>/``.

    plan     (dry-run, the default) pins each repo's branch head, discovers the
             cells whose rerun is complete while the canonical is not, cross-checks
             them against EXPECTED_CELLS and writes a reviewable per-file plan
             (JSON + text).  Read-only.
    execute  replays a reviewed plan.  One Hub commit per stage, every commit with
             ``parent_commit`` pinned to the branch head we last verified.  Other
             publishers (the lowdose pods) commit into disjoint prefixes every few
             minutes, so head movement by itself is tolerated: at cell start, before
             EVERY delete commit and after every 412 the cell's three prefixes are
             re-listed at the current head and must hold exactly the planned files
             (size + lfs sha256 / blob_id; no unplanned arrivals under the canonical,
             rerun or attempts paths).  A 412 is retried with the new parent up to
             MAX_COMMIT_ATTEMPTS times (short backoff); drift under our paths aborts:

             A. PARK     copy canonical partial -> <study>-attempts/<cell>/<label>/
                         verify at the new revision (size + LFS sha256 / blob_id via
                         list_repo_tree(expand=True)), THEN delete the partial files.
             B. PROMOTE  copy rerun/<cell> -> canonical/<cell>; verify; write
                         MOVE_RECORD.json (canonical) + MOVED_TO.json (rerun); THEN
                         delete the rerun copies except MOVED_TO.json.

Invariants: nothing is deleted before its copy is verified at a pinned revision;
no path outside the three per-cell prefixes is touched; a canonical cell that
already has COMPLETE.json is never promoted onto; cells whose rerun is not
complete are skipped (they can only be added by listing them with --cell once
they are complete, and even then completeness is re-checked).

Copy mechanics (huggingface_hub 1.16.x, checked in the installed package):
CommitOperationCopy copies LFS files server-side by sha256.  Regular (git)
files cannot be copied server-side; the 1.x client transparently downloads the
blob and re-sends it base64-encoded in the same commit
(`_commit_api._fetch_files_to_copy`).  Older clients only had
`_fetch_lfs_files_to_copy` and refused regular files; when that is detected we
download the bytes ourselves, check their git-blob sha1 against the pinned
blob_id and re-upload them byte-identically with CommitOperationAdd.  Either
way the copy is verified afterwards by blob_id at the new revision.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

import huggingface_hub
from huggingface_hub import CommitOperationAdd, CommitOperationCopy, CommitOperationDelete, HfApi
from huggingface_hub.errors import EntryNotFoundError
from huggingface_hub.hf_api import RepoFile

STUDY = "followups/gemma-aft-halfpct-balanced-v1"
ATTEMPTS_PREFIX = STUDY + "-attempts"
DEFAULT_NAMESPACE = "-jonathan-rerun1"
#: collect_followup_scores.py (_revision/_tree) addresses these repos with repo_type="model".
REPO_TYPE = "model"
REPOS = {
    "12b": "arcadia-impact/scimt-dispatch-gemma-12b-aft-grid-v2",
    "27b": "arcadia-impact/scimt-dispatch-gemma-27b-aft-grid-v2",
}
#: Cells whose -jonathan-rerun1 attempt is complete while Sid's canonical holds an
#: interrupted partial (state on 2026-09-09).  Discovery must agree with this list;
#: anything discovered but not listed here is reported and skipped unless named
#: explicitly with --cell.
EXPECTED_CELLS = {
    "12b": [
        "gemma3_12b_1m/charter/charter_0p5pct",
        "gemma3_12b_5m/charter/charter_0p5pct",
        "gemma3_12b_19m/charter/charter_0p5pct",
        "gemma3_12b_50m_4ep/charter/charter_0p5pct",
        "gemma3_12b_5m/control/charter_0p5pct",
    ],
    "27b": [
        "gemma3_27b_5m/charter/charter_0p5pct",
    ],
}
#: Released 27B cells still being rerun/restored on 2026-09-09: NOT complete, must be skipped.
HELD_CELLS = {
    "27b": [
        "gemma3_27b_19m/charter/charter_0p5pct",
        "gemma3_27b_50m/charter/charter_0p5pct",
        "gemma3_27b_190m/charter/charter_0p5pct",
        "gemma3_27b_5m/control/charter_0p5pct",
    ],
}
MOVED_BY = "jonathan@arcadiaimpact.org via Claude Code session 66941681"
REASON = (
    "Sid released the queue 2026-09-08 and approved park-then-promote 2026-09-09; "
    "attempts preserved under -attempts"
)
MOVE_RECORD = "MOVE_RECORD.json"
MOVED_TO = "MOVED_TO.json"
COMPLETE = "COMPLETE.json"
IDENTITY = "IDENTITY.json"
SAVES = (4, 8, 16, 32, 64, 128, 256, 512)
EVAL_STEPS = (256, 512)
DEFAULT_LOG_DIR = Path("/workspace/midtrain-token-budget-heatmaps/halfpct-logs/consolidate")
CELL_RE = re.compile(r"^[A-Za-z0-9_]+/[A-Za-z0-9_]+/[A-Za-z0-9_]+$")
LABEL_UNSAFE_RE = re.compile(r"[^A-Za-z0-9._-]+")
#: Commit attempts per stage when the head keeps moving (412) or the request fails transiently.
MAX_COMMIT_ATTEMPTS = 10
#: HTTP statuses that will not get better on retry.
NO_RETRY_STATUSES = (400, 401, 403, 404, 422)


class ConsolidateError(RuntimeError):
    """Base class: the run stops, nothing beyond the last verified commit was changed."""


class BranchMoved(ConsolidateError):
    """The branch head is not the pinned parent (HTTP 412 or a pre-flight mismatch)."""


class VerificationError(ConsolidateError):
    """A copied file is missing or differs at the new revision; no deletion happened."""


class SafetyStop(ConsolidateError):
    """An operation would touch a path outside the cell's three prefixes, or a precondition failed."""


# --------------------------------------------------------------------------- helpers
def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def compact_ts(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def git_blob_sha1(data: bytes) -> str:
    """The git object id of a regular file's content (what the Hub reports as blob_id/oid)."""
    h = hashlib.sha1()
    h.update(b"blob %d\0" % len(data))
    h.update(data)
    return h.hexdigest()


def _status(exc: BaseException) -> int | None:
    return getattr(getattr(exc, "response", None), "status_code", None)


def hub_copies_regular_files() -> bool:
    """True when the installed huggingface_hub makes CommitOperationCopy work for regular
    (non-LFS) files by downloading the blob and re-sending it in the commit payload
    (`_commit_api._fetch_files_to_copy`, huggingface_hub >= 1.x).  Older clients only had
    `_fetch_lfs_files_to_copy`, which raised NotImplementedError for regular files."""
    try:
        from huggingface_hub import _commit_api
    except ImportError:  # pragma: no cover - defensive
        return False
    return callable(getattr(_commit_api, "_fetch_files_to_copy", None))


def copy_mechanism(regular_via_copy: bool) -> dict[str, str]:
    regular = (
        "CommitOperationCopy(src_revision=<pinned>): huggingface_hub downloads the blob and "
        "re-sends it base64-encoded inside the same commit (regular files cannot be copied "
        "server-side); verified afterwards by blob_id at the new revision"
        if regular_via_copy
        else "download at the pinned revision, check git-blob sha1 == blob_id, re-upload "
        "byte-identically with CommitOperationAdd; verified afterwards by blob_id"
    )
    return {
        "huggingface_hub": huggingface_hub.__version__,
        "lfs": "CommitOperationCopy(src_revision=<pinned>): server-side by sha256, no bytes moved; "
        "verified afterwards by size + lfs.sha256 at the new revision",
        "regular": regular,
    }


@dataclass(frozen=True)
class FileEntry:
    """What list_repo_tree(expand=True) tells us about one file at one revision."""

    path: str
    size: int
    blob_id: str
    lfs_sha256: str | None = None
    last_commit_date: str | None = None

    @property
    def is_lfs(self) -> bool:
        return self.lfs_sha256 is not None

    def identity(self) -> tuple[str, str]:
        return ("sha256", self.lfs_sha256) if self.is_lfs else ("blob_id", self.blob_id)

    def same_content(self, other: "FileEntry") -> bool:
        return self.size == other.size and self.identity() == other.identity()

    def record(self, path: str | None = None) -> dict[str, Any]:
        kind, value = self.identity()
        return {"path": self.path if path is None else path, "size": self.size, kind: value}


def entry_from_repo_file(rf: Any) -> FileEntry:
    lfs = getattr(rf, "lfs", None)
    last_commit = getattr(rf, "last_commit", None)
    date = None
    if last_commit is not None:
        raw = getattr(last_commit, "date", None)
        if isinstance(raw, datetime):
            date = iso(raw) if raw.tzinfo else raw.isoformat()
        elif raw:
            date = str(raw)
    return FileEntry(
        path=rf.path,
        size=int(rf.size),
        blob_id=rf.blob_id,
        lfs_sha256=getattr(lfs, "sha256", None) if lfs else None,
        last_commit_date=date,
    )


def identity_record(entry: FileEntry) -> dict[str, Any]:
    kind, value = entry.identity()
    return {"size": entry.size, kind: value}


def entry_from_op(op: dict[str, Any], path: str) -> FileEntry:
    """Rebuild the planned identity of a file operation or baseline record (for verification)."""
    return FileEntry(
        path=path,
        size=int(op["size"]),
        blob_id=op.get("blob_id") or "",
        lfs_sha256=op.get("sha256"),
    )


class HubClient:
    """The four HfApi calls this tool needs: pinned head, expanded tree, raw bytes, commit.

    Read calls retry on 429/5xx/network errors; a missing path lists as empty; a commit
    whose parent_commit no longer matches the branch head surfaces as BranchMoved.
    """

    def __init__(self, api: Any, cache_dir: Path | None = None, retries: int = 5,
                 sleep: Callable[[float], None] = time.sleep):
        self.api = api
        self.cache_dir = cache_dir
        self.retries = retries
        self.sleep = sleep

    def _retry(self, fn: Callable[[], Any]) -> Any:
        for attempt in range(1, self.retries + 1):
            try:
                return fn()
            except EntryNotFoundError:
                raise
            except Exception as exc:
                status = _status(exc)
                if status is not None and status < 500 and status != 429:
                    raise
                if attempt == self.retries:
                    raise
                self.sleep(min(60.0, 2.0 ** attempt))
        raise AssertionError("unreachable")

    def head(self, repo_id: str) -> str:
        return self._retry(lambda: self.api.repo_info(repo_id, repo_type=REPO_TYPE).sha)

    def tree(self, repo_id: str, prefix: str, revision: str) -> dict[str, FileEntry]:
        """Every file under prefix/ at revision (folders dropped); {} when the path is absent."""

        def _list() -> dict[str, FileEntry]:
            try:
                entries = list(self.api.list_repo_tree(
                    repo_id, path_in_repo=prefix, recursive=True, expand=True,
                    revision=revision, repo_type=REPO_TYPE))
            except EntryNotFoundError:
                return {}
            except Exception as exc:
                if _status(exc) == 404:
                    return {}
                raise
            files: dict[str, FileEntry] = {}
            for entry in entries:
                if not isinstance(entry, RepoFile):
                    continue
                if not entry.path.startswith(prefix + "/"):
                    continue
                files[entry.path] = entry_from_repo_file(entry)
            return files

        return self._retry(_list)

    def read_bytes(self, repo_id: str, path: str, revision: str) -> bytes:
        def _read() -> bytes:
            kwargs: dict[str, Any] = dict(repo_id=repo_id, filename=path, revision=revision, repo_type=REPO_TYPE)
            if self.cache_dir is not None:
                kwargs["cache_dir"] = str(self.cache_dir)
            return Path(self.api.hf_hub_download(**kwargs)).read_bytes()

        return self._retry(_read)

    def commit(self, repo_id: str, operations: list[Any], message: str, description: str,
               parent_commit: str) -> str:
        try:
            info = self.api.create_commit(
                repo_id, operations, commit_message=message, commit_description=description,
                repo_type=REPO_TYPE, parent_commit=parent_commit)
        except ConsolidateError:
            raise
        except Exception as exc:
            if _status(exc) == 412:
                raise BranchMoved(
                    f"{repo_id}: branch head is no longer {parent_commit[:12]} (HTTP 412 on "
                    f"{message!r}); nothing further was committed. Re-run `plan`.") from exc
            raise
        return info.oid


# --------------------------------------------------------------------------- tree logic
def split_cell_path(prefix: str, path: str) -> tuple[str, str] | None:
    """`prefix/<profile>/<arm>/<mixture>/<rel...>` -> (cell, rel); None for anything else
    (e.g. `prefix/shared-data/...`)."""
    if not path.startswith(prefix + "/"):
        return None
    parts = path[len(prefix) + 1:].split("/")
    if len(parts) < 4:
        return None
    return "/".join(parts[:3]), "/".join(parts[3:])


def group_cells(files: dict[str, FileEntry], prefix: str) -> dict[str, dict[str, FileEntry]]:
    cells: dict[str, dict[str, FileEntry]] = {}
    for path, entry in files.items():
        split = split_cell_path(prefix, path)
        if split is None:
            continue
        cell, rel = split
        cells.setdefault(cell, {})[rel] = entry
    return cells


def strip_prefix(files: dict[str, FileEntry], prefix: str) -> dict[str, FileEntry]:
    return {path[len(prefix) + 1:]: entry for path, entry in files.items() if path.startswith(prefix + "/")}


def completeness_issues(rel_files: dict[str, FileEntry]) -> list[str]:
    """A complete cell: COMPLETE.json, both eval scores, and all 8 saves with weights + receipt."""
    issues = []
    if COMPLETE not in rel_files:
        issues.append(f"{COMPLETE} missing")
    for step in EVAL_STEPS:
        if f"eval/aft-step{step}/scores.json" not in rel_files:
            issues.append(f"eval/aft-step{step}/scores.json missing")
    for step in SAVES:
        ckpt = f"train/checkpoints/checkpoint-{step}"
        for name in ("adapter_model.safetensors", "SAVE_COMPLETE.json"):
            if f"{ckpt}/{name}" not in rel_files:
                issues.append(f"{ckpt}/{name} missing")
    return issues


def checkpoint_steps(rel_files: Iterable[str]) -> list[int]:
    steps = set()
    for rel in rel_files:
        m = re.match(r"train/checkpoints/checkpoint-(\d+)/", rel)
        if m:
            steps.add(int(m.group(1)))
    return sorted(steps)


def newest_commit_date(files: Iterable[FileEntry]) -> str | None:
    dates = sorted(e.last_commit_date for e in files if e.last_commit_date)
    return dates[-1] if dates else None


def derive_park_label(identity: dict[str, Any] | None, partial: dict[str, FileEntry],
                      fallback_ymd: str) -> tuple[str, str]:
    """`sid-<worker>-interrupted-<YYYYMMDD>` from the partial's own IDENTITY.json worker and the
    date of the last commit that landed in it; `sid-interrupted-<YYYYMMDD>` when no worker is
    readable."""
    newest = newest_commit_date(partial.values())
    ymd = newest[:10].replace("-", "") if newest else fallback_ymd
    date_source = "newest last_commit among the partial's files" if newest else "plan date (no last_commit info)"
    worker = None
    if isinstance(identity, dict):
        worker = identity.get("worker") or (identity.get("identity") or {}).get("worker")
    if worker:
        label = f"sid-{worker}-interrupted-{ymd}"
        source = f"IDENTITY.json worker={worker!r}; date from {date_source}"
    else:
        label = f"sid-interrupted-{ymd}"
        source = f"fallback (no readable IDENTITY.json worker); date from {date_source}"
    return LABEL_UNSAFE_RE.sub("-", label).strip("-."), source


def read_json_quietly(client: HubClient, repo_id: str, path: str, revision: str,
                      present: bool) -> dict[str, Any] | None:
    if not present:
        return None
    try:
        return json.loads(client.read_bytes(repo_id, path, revision).decode("utf-8"))
    except Exception:  # noqa: BLE001 - a label fallback exists for exactly this case
        return None


def file_op(src: str, dst: str, entry: FileEntry, status: str, regular_via_copy: bool) -> dict[str, Any]:
    kind, value = entry.identity()
    via = "CommitOperationCopy" if (entry.is_lfs or regular_via_copy) else "CommitOperationAdd(download+reupload)"
    return {
        "src": src,
        "dst": dst,
        "size": entry.size,
        "kind": "lfs" if entry.is_lfs else "regular",
        kind: value,
        "status": status,  # copy | already_present | conflict
        "via": via if status == "copy" else None,
    }


def op_counts(files: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "n_files": len(files),
        "n_lfs": sum(1 for f in files if f["kind"] == "lfs"),
        "n_regular": sum(1 for f in files if f["kind"] == "regular"),
        "n_to_copy": sum(1 for f in files if f["status"] == "copy"),
        "n_already_present": sum(1 for f in files if f["status"] == "already_present"),
        "total_bytes": sum(f["size"] for f in files),
    }


def in_scope(cp: dict[str, Any], path: str) -> bool:
    return any(path.startswith(prefix + "/") for prefix in
               (cp["canonical_prefix"], cp["rerun_prefix"], cp["attempts_root"]))


# --------------------------------------------------------------------------- planning
def plan_cell(client: HubClient, repo_key: str, repo_id: str, revision: str, cell: str,
              rerun_prefix: str, rerun: dict[str, FileEntry], now: datetime,
              regular_via_copy: bool) -> dict[str, Any]:
    canonical_prefix = f"{STUDY}/{cell}"
    rerun_cell_prefix = f"{rerun_prefix}/{cell}"
    attempts_root = f"{ATTEMPTS_PREFIX}/{cell}"
    canonical = strip_prefix(client.tree(repo_id, canonical_prefix, revision), canonical_prefix)
    parked = client.tree(repo_id, attempts_root, revision)
    # Everything under the cell's three prefixes at plan time: execute re-checks this (evolved
    # stage by stage) at the current head before every delete, so unrelated commits elsewhere
    # in the repo are tolerated while any change under our paths aborts.
    baseline = {f"{canonical_prefix}/{rel}": identity_record(e) for rel, e in canonical.items()}
    baseline.update({f"{rerun_cell_prefix}/{rel}": identity_record(e) for rel, e in rerun.items()})
    baseline.update({path: identity_record(e) for path, e in parked.items()})
    refusals: list[str] = []
    notes: list[str] = []
    cp: dict[str, Any] = {
        "cell": cell,
        "repo_key": repo_key,
        "repo_id": repo_id,
        "revision": revision,
        "mode": None,  # move | finish
        "canonical_prefix": canonical_prefix,
        "rerun_prefix": rerun_cell_prefix,
        "attempts_root": attempts_root,
        "baseline": baseline,
        "rerun_n_files": len(rerun),
        "rerun_checkpoints": checkpoint_steps(rerun),
        "rerun_last_commit": newest_commit_date(rerun.values()),
        "canonical_n_files": len(canonical),
        "canonical_checkpoints": checkpoint_steps(canonical),
        "canonical_last_commit": newest_commit_date(canonical.values()),
        "canonical_has_complete": COMPLETE in canonical,
        "refusals": refusals,
        "notes": notes,
        "park": None,
        "promote": None,
        "records": None,
        "commits": [],
    }
    if not rerun:
        refusals.append(f"rerun cell absent: nothing under {rerun_cell_prefix}/")
        return cp
    if set(rerun) == {MOVED_TO}:
        refusals.append("already consolidated: the rerun prefix holds only MOVED_TO.json")
        return cp
    if COMPLETE not in rerun:
        refusals.append(f"rerun incomplete: {len(rerun)} files, no {COMPLETE} "
                        f"(checkpoints {checkpoint_steps(rerun)})")
        return cp
    issues = completeness_issues(rerun)
    if issues:
        refusals.append(f"rerun has {COMPLETE} but its structure is incomplete: " + "; ".join(issues[:6]))
        return cp

    payload = {rel: e for rel, e in rerun.items() if rel != MOVED_TO}
    if MOVED_TO in rerun:
        notes.append("rerun already contains MOVED_TO.json (earlier pass?); it is not copied")

    if COMPLETE in canonical:
        identical = all(rel in canonical and canonical[rel].same_content(e) for rel, e in payload.items())
        if not identical:
            refusals.append(f"canonical already has {COMPLETE} and differs from the rerun: refusing to promote onto it")
            return cp
        cp["mode"] = "finish"
        notes.append("canonical already holds an identical copy of every rerun file (interrupted earlier "
                     "pass): only the move records and the rerun cleanup are planned")
    else:
        cp["mode"] = "move"

    partial_identity = None
    if cp["mode"] == "move" and canonical:
        partial_identity = read_json_quietly(client, repo_id, f"{canonical_prefix}/{IDENTITY}", revision,
                                             IDENTITY in canonical)
        label, label_source = derive_park_label(partial_identity, canonical, now.strftime("%Y%m%d"))
        park_prefix = f"{attempts_root}/{label}"
        files = []
        for rel in sorted(canonical):
            entry = canonical[rel]
            dst = f"{park_prefix}/{rel}"
            existing = parked.get(dst)
            if existing is None:
                status = "copy"
            elif existing.same_content(entry):
                status = "already_present"
            else:
                status = "conflict"
                refusals.append(f"park destination already exists with different content: {dst}")
            files.append(file_op(f"{canonical_prefix}/{rel}", dst, entry, status, regular_via_copy))
        other_labels = sorted({p[len(attempts_root) + 1:].split("/")[0] for p in parked
                               if not p.startswith(park_prefix + "/")})
        if other_labels:
            notes.append(f"attempts already parked under other labels: {other_labels}")
        cp["park"] = {
            "label": label,
            "label_source": label_source,
            "prefix": park_prefix,
            "partial_worker": (partial_identity or {}).get("worker") if isinstance(partial_identity, dict) else None,
            "checkpoints": checkpoint_steps(canonical),
            "files": files,
            **op_counts(files),
            "delete_after_verify": [f"{canonical_prefix}/{rel}" for rel in sorted(canonical)],
        }
        only_partial = sorted(set(canonical) - set(payload))
        if only_partial:
            notes.append(f"SURPRISE: canonical partial has {len(only_partial)} file(s) the rerun lacks "
                         f"(parked, not promoted): {only_partial[:12]}")
        same = sorted(rel for rel in canonical if rel in payload and canonical[rel].same_content(payload[rel]))
        differ = sorted(rel for rel in canonical if rel in payload and not canonical[rel].same_content(payload[rel]))
        notes.append(f"partial vs rerun overlap: {len(same)} identical file(s) (e.g. {same[:4]}), "
                     f"{len(differ)} differing (e.g. {differ[:4]})")
    elif cp["mode"] == "move":
        notes.append("canonical has no files: PARK skipped")
    if refusals:
        return cp

    rerun_identity = read_json_quietly(client, repo_id, f"{rerun_cell_prefix}/{IDENTITY}", revision,
                                       IDENTITY in payload)
    files = []
    for rel in sorted(payload):
        status = "already_present" if cp["mode"] == "finish" else "copy"
        files.append(file_op(f"{rerun_cell_prefix}/{rel}", f"{canonical_prefix}/{rel}", payload[rel], status,
                             regular_via_copy))
    cp["promote"] = {
        "src_prefix": rerun_cell_prefix,
        "dst_prefix": canonical_prefix,
        "rerun_worker": (rerun_identity or {}).get("worker") if isinstance(rerun_identity, dict) else None,
        "checkpoints": checkpoint_steps(payload),
        "files": files,
        **op_counts(files),
        "delete_after_verify": [f"{rerun_cell_prefix}/{rel}" for rel in sorted(payload)],  # MOVED_TO.json never listed
    }
    cp["records"] = {
        "move_record": f"{canonical_prefix}/{MOVE_RECORD}",
        "moved_to": f"{rerun_cell_prefix}/{MOVED_TO}",
        "schema": ["cell", "repo", "src_prefix", "dst_prefix", "src_revision", "dst_revision", "park_prefix",
                   "files[{path,size,sha256|blob_id}]", "moved_at", "moved_by", "reason"],
        "moved_by": MOVED_BY,
        "reason": REASON,
    }
    cp["commits"] = planned_commits(cp)
    for path in ([f["dst"] for f in files] + cp["promote"]["delete_after_verify"] +
                 ([f["dst"] for f in cp["park"]["files"]] + cp["park"]["delete_after_verify"] if cp["park"] else [])):
        if not in_scope(cp, path):  # pragma: no cover - construction guarantees this
            refusals.append(f"internal error: planned path outside the cell prefixes: {path}")
    return cp


def planned_commits(cp: dict[str, Any]) -> list[dict[str, Any]]:
    seq: list[dict[str, Any]] = []
    if cp["park"]:
        seq.append({"stage": "PARK_COPY", "n_ops": cp["park"]["n_to_copy"],
                    "note": "skipped when 0 (already parked)"})
        seq.append({"stage": "PARK_VERIFY", "n_files": cp["park"]["n_files"], "read_only": True})
        seq.append({"stage": "PARK_DELETE", "n_ops": len(cp["park"]["delete_after_verify"])})
    if cp["mode"] == "move":
        seq.append({"stage": "PROMOTE_COPY", "n_ops": cp["promote"]["n_to_copy"]})
    seq.append({"stage": "PROMOTE_VERIFY", "n_files": cp["promote"]["n_files"], "read_only": True})
    seq.append({"stage": "RECORDS", "n_ops": 2})
    seq.append({"stage": "RECORDS_VERIFY", "n_files": 2, "read_only": True})
    seq.append({"stage": "PROMOTE_DELETE", "n_ops": len(cp["promote"]["delete_after_verify"])})
    return seq


def plan_repo(client: HubClient, repo_key: str, namespace: str, cells: list[str] | None,
              now: datetime, regular_via_copy: bool) -> dict[str, Any]:
    repo_id = REPOS[repo_key]
    revision = client.head(repo_id)
    rerun_prefix = STUDY + namespace
    rerun_cells = group_cells(client.tree(repo_id, rerun_prefix, revision), rerun_prefix)
    complete = sorted(c for c, f in rerun_cells.items() if COMPLETE in f)
    expected = EXPECTED_CELLS.get(repo_key, [])
    if cells is None:
        candidates = list(expected)
        unexpected = [c for c in complete if c not in expected]
    else:
        candidates = [c for c in cells if c.startswith(f"gemma3_{repo_key}_")]
        unexpected = []
    out: dict[str, Any] = {
        "repo_key": repo_key,
        "repo_id": repo_id,
        "revision": revision,
        "rerun_prefix": rerun_prefix,
        "rerun_cells_seen": {c: {"n_files": len(f), "complete": COMPLETE in f, "checkpoints": checkpoint_steps(f)}
                             for c, f in sorted(rerun_cells.items())},
        "cells": [],
        "skipped": [],
        "unexpected_complete_skipped": unexpected,
        "held_cells": HELD_CELLS.get(repo_key, []),
    }
    for cell in candidates:
        cp = plan_cell(client, repo_key, repo_id, revision, cell, rerun_prefix, rerun_cells.get(cell, {}),
                       now, regular_via_copy)
        cp["in_expected_list"] = cell in expected
        if cp["refusals"]:
            out["skipped"].append({k: cp[k] for k in (
                "cell", "refusals", "rerun_n_files", "rerun_checkpoints", "canonical_n_files",
                "canonical_has_complete", "in_expected_list")})
        else:
            out["cells"].append(cp)
    return out


def summarize_totals(plan: dict[str, Any]) -> dict[str, Any]:
    totals: dict[str, Any] = {"cells": 0, "park_files": 0, "park_bytes": 0, "promote_files": 0,
                              "promote_bytes": 0, "commits": 0, "per_repo": {}}
    for key, rp in plan["repos"].items():
        t = {"cells": len(rp["cells"]), "park_files": 0, "park_bytes": 0, "promote_files": 0,
             "promote_bytes": 0, "commits": 0, "skipped": len(rp["skipped"])}
        for cp in rp["cells"]:
            if cp["park"]:
                t["park_files"] += cp["park"]["n_files"]
                t["park_bytes"] += cp["park"]["total_bytes"]
            t["promote_files"] += cp["promote"]["n_files"]
            t["promote_bytes"] += cp["promote"]["total_bytes"]
            t["commits"] += sum(1 for c in cp["commits"] if not c.get("read_only"))
        totals["per_repo"][key] = t
        for k in ("cells", "park_files", "park_bytes", "promote_files", "promote_bytes", "commits"):
            totals[k] += t[k]
    return totals


def build_plan(client: HubClient, repo_keys: list[str], namespace: str = DEFAULT_NAMESPACE,
               cells: list[str] | None = None, now: datetime | None = None,
               regular_via_copy: bool | None = None, argv: list[str] | None = None) -> dict[str, Any]:
    now = now or utc_now()
    if regular_via_copy is None:
        regular_via_copy = hub_copies_regular_files()
    if not namespace.startswith("-") or "/" in namespace:
        raise ConsolidateError(f"--namespace must be a suffix such as {DEFAULT_NAMESPACE!r}, got {namespace!r}")
    for cell in cells or ():
        if not CELL_RE.match(cell):
            raise ConsolidateError(f"bad cell id {cell!r}: want <profile>/<arm>/<mixture>")
        if not any(cell.startswith(f"gemma3_{key}_") for key in repo_keys):
            raise ConsolidateError(f"cell {cell!r} does not belong to the selected repo(s) {repo_keys}")
    plan: dict[str, Any] = {
        "tool": "consolidate_halfpct.py",
        "schema": 1,
        "plan_id": compact_ts(now),
        "created_at": iso(now),
        "argv": argv,
        "python": sys.version.split()[0],
        "namespace": namespace,
        "study_prefix": STUDY,
        "rerun_prefix": STUDY + namespace,
        "attempts_prefix": ATTEMPTS_PREFIX,
        "repo_type": REPO_TYPE,
        "moved_by": MOVED_BY,
        "reason": REASON,
        "copy_mechanism": copy_mechanism(regular_via_copy),
        "regular_via_copy": regular_via_copy,
        "repos": {},
    }
    for key in repo_keys:
        plan["repos"][key] = plan_repo(client, key, namespace, cells, now, regular_via_copy)
    plan["totals"] = summarize_totals(plan)
    return plan


def _gb(n: int) -> str:
    return f"{n / 1e9:.3f} GB"


def render_summary(plan: dict[str, Any]) -> str:
    L: list[str] = []
    L.append(f"consolidate_halfpct DRY-RUN plan {plan['plan_id']} (created {plan['created_at']})")
    L.append(f"  namespace {plan['namespace']}  ->  rerun prefix {plan['rerun_prefix']}")
    L.append(f"  canonical {plan['study_prefix']}/<cell>/   park {plan['attempts_prefix']}/<cell>/<label>/")
    L.append(f"  huggingface_hub {plan['copy_mechanism']['huggingface_hub']}; regular files via "
             f"{'CommitOperationCopy (client download+re-send)' if plan['regular_via_copy'] else 'CommitOperationAdd fallback'}")
    L.append(f"  moved_by: {plan['moved_by']}")
    L.append(f"  reason:   {plan['reason']}")
    L.append("  execute: parent_commit pinned per commit; unrelated head movement tolerated (lowdose pods publish "
             "elsewhere) - the cell's three prefixes are re-verified at the current head at cell start, before "
             f"every delete and after every 412 (<= {MAX_COMMIT_ATTEMPTS} attempts); drift under our paths aborts")
    for key, rp in plan["repos"].items():
        L.append("")
        L.append(f"== {key}: {rp['repo_id']} @ {rp['revision']}")
        L.append(f"   rerun cells seen under {rp['rerun_prefix']}: " + ", ".join(
            f"{c} ({v['n_files']} files{', COMPLETE' if v['complete'] else ''})" for c, v in rp["rerun_cells_seen"].items()) or "none")
        if not rp["cells"]:
            L.append("   no cell ready to consolidate")
        for cp in rp["cells"]:
            L.append(f"   -- {cp['cell']}  mode={cp['mode']}  rerun {cp['rerun_n_files']} files "
                     f"(ckpts {cp['rerun_checkpoints']}, last commit {cp['rerun_last_commit']})")
            if cp["park"]:
                pk = cp["park"]
                L.append(f"      PARK    {pk['n_files']} partial files ({pk['n_lfs']} LFS, {pk['n_regular']} regular, "
                         f"{_gb(pk['total_bytes'])}; ckpts {pk['checkpoints']}; last commit {cp['canonical_last_commit']})")
                L.append(f"              -> {pk['prefix']}/")
                L.append(f"              label from {pk['label_source']}; {pk['n_to_copy']} to copy, "
                         f"{pk['n_already_present']} already there; then delete {len(pk['delete_after_verify'])} canonical paths")
            else:
                L.append("      PARK    none (canonical empty or mode=finish)")
            pr = cp["promote"]
            L.append(f"      PROMOTE {pr['n_files']} files ({pr['n_lfs']} LFS, {pr['n_regular']} regular, {_gb(pr['total_bytes'])})"
                     f" {pr['src_prefix']}/ -> {pr['dst_prefix']}/; then records; then delete "
                     f"{len(pr['delete_after_verify'])} rerun paths (MOVED_TO.json stays)")
            L.append("      commits: " + " -> ".join(
                f"{c['stage']}({c.get('n_ops', c.get('n_files'))})" for c in cp["commits"] if not c.get("read_only")))
            for note in cp["notes"]:
                L.append(f"      note: {note}")
        for sk in rp["skipped"]:
            flag = "" if sk["in_expected_list"] else " [not in EXPECTED_CELLS]"
            L.append(f"   -- SKIP {sk['cell']}{flag}: " + " | ".join(sk["refusals"]))
        if rp["unexpected_complete_skipped"]:
            L.append(f"   !! complete rerun cells NOT in EXPECTED_CELLS (skipped; add with --cell to include): "
                     f"{rp['unexpected_complete_skipped']}")
        held_seen = [c for c in rp.get("held_cells", []) if c in rp["rerun_cells_seen"]]
        if held_seen:
            L.append(f"   held (not complete yet, skipped): {held_seen}")
    t = plan["totals"]
    L.append("")
    L.append(f"TOTAL {t['cells']} cell(s): park {t['park_files']} files / {_gb(t['park_bytes'])}, "
             f"promote {t['promote_files']} files / {_gb(t['promote_bytes'])}, {t['commits']} commits")
    L.append("Nothing was written. Review the JSON, then: consolidate_halfpct.py execute --plan <json>")
    return "\n".join(L) + "\n"


# --------------------------------------------------------------------------- execution
def build_move_record(cp: dict[str, Any], plan: dict[str, Any], src_revision: str, dst_revision: str,
                      moved_at: str) -> dict[str, Any]:
    files = []
    src_prefix = cp["promote"]["src_prefix"]
    for op in cp["promote"]["files"]:
        rel = op["src"][len(src_prefix) + 1:]
        rec = {"path": rel, "size": op["size"]}
        rec["sha256" if op["kind"] == "lfs" else "blob_id"] = op.get("sha256") or op.get("blob_id")
        files.append(rec)
    return {
        "record": "park-then-promote consolidation (consolidate_halfpct.py)",
        "cell": cp["cell"],
        "repo": cp["repo_id"],
        "src_prefix": src_prefix,
        "dst_prefix": cp["promote"]["dst_prefix"],
        "src_revision": src_revision,
        "dst_revision": dst_revision,
        "park_prefix": cp["park"]["prefix"] if cp["park"] else None,
        "park_label": cp["park"]["label"] if cp["park"] else None,
        "parked_files": cp["park"]["n_files"] if cp["park"] else 0,
        "files_path_root": "relative to src_prefix (before) and dst_prefix (after)",
        "files": files,
        "moved_at": moved_at,
        "moved_by": plan.get("moved_by", MOVED_BY),
        "reason": plan.get("reason", REASON),
        "plan_id": plan.get("plan_id"),
        "mode": cp["mode"],
        "rerun_worker": cp["promote"].get("rerun_worker"),
        "partial_worker": cp["park"].get("partial_worker") if cp["park"] else None,
    }


class CellState:
    """The exact expected contents of a cell's three prefixes, evolving as stages land.

    Starts from the plan-time baseline (every file under canonical/<cell>, rerun/<cell> and
    attempts/<cell>) and applies each committed stage.  `problems()` compares it with a listing
    of the same prefixes at some revision: a planned file missing or changed, or ANY file that is
    not expected (an unplanned arrival), is a problem.  Files outside the three prefixes are
    invisible to it, which is what makes unrelated publishers harmless.
    """

    STAGES = ("PARK_COPY", "PARK_DELETE", "PROMOTE_COPY", "RECORDS", "PROMOTE_DELETE")

    def __init__(self, cp: dict[str, Any]):
        if "baseline" not in cp:
            raise SafetyStop(f"{cp['cell']}: plan predates path-scoped concurrency checks (no baseline); re-run `plan`")
        self.cp = cp
        self.prefixes = (cp["canonical_prefix"], cp["rerun_prefix"], cp["attempts_root"])
        self.expected: dict[str, FileEntry] = {p: entry_from_op(rec, p) for p, rec in cp["baseline"].items()}
        self.record: tuple[int, str] | None = None  # (size, git blob sha1) of the records payload

    def after(self, stage: str) -> dict[str, FileEntry]:
        cp = self.cp
        exp = dict(self.expected)
        if stage == "PARK_COPY":
            for op in cp["park"]["files"]:
                exp[op["dst"]] = entry_from_op(op, op["dst"])
        elif stage == "PARK_DELETE":
            for path in cp["park"]["delete_after_verify"]:
                exp.pop(path, None)
        elif stage == "PROMOTE_COPY":
            for op in cp["promote"]["files"]:
                exp[op["dst"]] = entry_from_op(op, op["dst"])
        elif stage == "RECORDS":
            if self.record is None:
                raise SafetyStop("records payload not built yet")
            size, blob = self.record
            for path in (cp["records"]["move_record"], cp["records"]["moved_to"]):
                exp[path] = FileEntry(path=path, size=size, blob_id=blob)
        elif stage == "PROMOTE_DELETE":
            for path in cp["promote"]["delete_after_verify"]:
                exp.pop(path, None)
        else:
            raise ValueError(stage)
        return exp

    def apply(self, stage: str) -> None:
        self.expected = self.after(stage)

    def problems(self, actual: dict[str, FileEntry], expected: dict[str, FileEntry] | None = None) -> list[str]:
        expected = self.expected if expected is None else expected
        out = []
        for path, want in expected.items():
            got = actual.get(path)
            if got is None:
                out.append(f"planned file missing: {path}")
            elif not got.same_content(want):
                out.append(f"planned file changed: {path} (want {want.identity()} {want.size} B, "
                           f"got {got.identity()} {got.size} B)")
        for path in sorted(actual):
            if path not in expected:
                out.append(f"unplanned file appeared: {path}")
        return out


class Executor:
    """Replays a plan: PARK (copy, verify, delete) then PROMOTE (copy, verify, records, delete).

    Concurrency: every commit pins parent_commit to the head we last verified.  Head movement by
    other publishers is tolerated; what must not change is the content of the cell's three
    prefixes, which is re-listed at the current head at cell start, before every delete commit
    and after every failed commit attempt (412 or transient error) before retrying.
    """

    def __init__(self, client: HubClient, plan: dict[str, Any], log_path: Path | None = None,
                 regular_via_copy: bool | None = None, now: Callable[[], datetime] = utc_now,
                 echo: Callable[[str], None] = print, sleep: Callable[[float], None] = time.sleep,
                 max_attempts: int = MAX_COMMIT_ATTEMPTS):
        self.client = client
        self.plan = plan
        self.log_path = log_path
        self.regular_via_copy = hub_copies_regular_files() if regular_via_copy is None else regular_via_copy
        self.now = now
        self.echo = echo
        self.sleep = sleep
        self.max_attempts = max_attempts
        self.events: list[dict[str, Any]] = []

    # -- plumbing
    def log(self, **event: Any) -> None:
        event = {"ts": iso(self.now()), **event}
        self.events.append(event)
        if self.log_path is not None:
            with self.log_path.open("a") as fh:
                fh.write(json.dumps(event, default=str) + "\n")
        brief = " ".join(f"{k}={v}" for k, v in event.items() if k in ("stage", "cell", "revision", "n_ops", "n_files", "message"))
        self.echo(f"[{event['ts']}] {brief}")

    def _description(self, cp: dict[str, Any]) -> str:
        return (f"consolidate_halfpct.py plan {self.plan.get('plan_id')} (computed against {cp['revision']}). "
                f"{self.plan.get('reason', REASON)}. {self.plan.get('moved_by', MOVED_BY)}.")

    def _guard_scope(self, cp: dict[str, Any], paths: Iterable[str]) -> None:
        bad = [p for p in paths if not in_scope(cp, p)]
        if bad:
            raise SafetyStop(f"{cp['cell']}: refusing to touch paths outside the cell prefixes: {bad[:5]}")

    def _commit(self, cp: dict[str, Any], stage: str, ops: list[Any], parent: str, summary: str) -> str:
        touched = [getattr(op, "path_in_repo") for op in ops] + [
            op.src_path_in_repo for op in ops if isinstance(op, CommitOperationCopy)]
        self._guard_scope(cp, touched)
        message = f"[consolidate_halfpct] {stage} {cp['cell']}: {summary}"
        new = self.client.commit(cp["repo_id"], ops, message, self._description(cp), parent)
        if new == parent:
            self.log(stage=stage, cell=cp["cell"], revision=new, n_ops=len(ops), message="no-op (Hub skipped an empty commit)")
            return parent
        self.log(stage=stage, cell=cp["cell"], revision=new, parent_commit=parent, n_ops=len(ops), message=message)
        return new

    def _copy_ops(self, cp: dict[str, Any], files: list[dict[str, Any]], src_revision: str) -> list[Any]:
        ops: list[Any] = []
        for op in files:
            if op["status"] != "copy":
                continue
            if op["kind"] == "lfs" or self.regular_via_copy:
                ops.append(CommitOperationCopy(src_path_in_repo=op["src"], path_in_repo=op["dst"], src_revision=src_revision))
            else:
                data = self.client.read_bytes(cp["repo_id"], op["src"], src_revision)
                got = git_blob_sha1(data)
                if got != op["blob_id"] or len(data) != op["size"]:
                    raise VerificationError(
                        f"{cp['cell']}: downloaded {op['src']} at {src_revision[:12]} has blob {got} / {len(data)} B, "
                        f"plan says {op['blob_id']} / {op['size']} B")
                ops.append(CommitOperationAdd(path_in_repo=op["dst"], path_or_fileobj=data))
        return ops

    def _verify(self, cp: dict[str, Any], stage: str, files: list[dict[str, Any]], prefix: str, revision: str) -> None:
        tree = self.client.tree(cp["repo_id"], prefix, revision)
        problems = []
        for op in files:
            want = entry_from_op(op, op["dst"])
            got = tree.get(op["dst"])
            if got is None:
                problems.append(f"missing {op['dst']}")
            elif not got.same_content(want):
                problems.append(f"{op['dst']}: want {want.identity()} {want.size} B, got {got.identity()} {got.size} B")
        if problems:
            raise VerificationError(
                f"{stage} {cp['cell']}: {len(problems)} problem(s) verifying {len(files)} file(s) at "
                f"{revision[:12]}; NOTHING was deleted. First: {problems[:5]}")
        self.log(stage=stage, cell=cp["cell"], revision=revision, n_files=len(files), message="verified")

    def _actual(self, cp: dict[str, Any], revision: str) -> dict[str, FileEntry]:
        """Everything currently under the cell's three prefixes at `revision`."""
        actual: dict[str, FileEntry] = {}
        for prefix in (cp["canonical_prefix"], cp["rerun_prefix"], cp["attempts_root"]):
            actual.update(self.client.tree(cp["repo_id"], prefix, revision))
        return actual

    def _checkpoint(self, cp: dict[str, Any], state: CellState, why: str) -> str:
        """Re-read the head and require our three prefixes to match the expected state exactly."""
        head = self.client.head(cp["repo_id"])
        problems = state.problems(self._actual(cp, head))
        if problems:
            raise SafetyStop(f"{cp['cell']}: {why}: our paths drifted at head {head[:12]}; stopping before any "
                             f"further change. First: {problems[:5]}")
        self.log(stage="CHECKPOINT", cell=cp["cell"], revision=head, n_files=len(state.expected),
                 message=f"{why}: the cell's three prefixes match the plan at the current head")
        return head

    def _commit_guarded(self, cp: dict[str, Any], state: CellState, stage: str, ops: list[Any], parent: str,
                        summary: str) -> str:
        """Commit with parent_commit=parent; on 412 or a transient failure re-read the head, re-verify
        our paths at it and retry with the new parent (or detect that the commit did land)."""
        attempt = 0
        while True:
            attempt += 1
            try:
                new = self._commit(cp, stage, ops, parent, summary)
                state.apply(stage)
                return new
            except BranchMoved:
                reason = "HTTP 412 (branch head moved)"
            except ConsolidateError:
                raise
            except Exception as exc:  # noqa: BLE001 - classified below
                status = _status(exc)
                if status in NO_RETRY_STATUSES:
                    raise
                reason = f"{type(exc).__name__}: {str(exc)[:160]}"
            if attempt >= self.max_attempts:
                raise BranchMoved(f"{cp['cell']}: {stage} failed {attempt} times ({reason}); last parent "
                                  f"{parent[:12]}. Nothing beyond the last verified commit changed; re-run `plan`.")
            self.sleep(min(10.0, 2.0 ** (attempt - 1)))
            head = self.client.head(cp["repo_id"])
            actual = self._actual(cp, head)
            if not state.problems(actual, state.after(stage)):
                self.log(stage=stage, cell=cp["cell"], revision=head, n_ops=len(ops),
                         message=f"{reason}, but head {head[:12]} already shows this stage applied; continuing")
                state.apply(stage)
                return head
            problems = state.problems(actual)
            if problems:
                raise SafetyStop(f"{cp['cell']}: {stage}: after {reason}, our paths drifted at head {head[:12]}; "
                                 f"not retrying. First: {problems[:5]}")
            self.log(stage=stage, cell=cp["cell"], revision=head, n_ops=len(ops),
                     message=f"{reason}; unrelated movement, our paths intact at {head[:12]} -> "
                             f"retry {attempt + 1}/{self.max_attempts} with that parent")
            parent = head

    # -- stages
    def run_cell(self, cp: dict[str, Any]) -> dict[str, Any]:
        repo_id = cp["repo_id"]
        rev0 = cp["revision"]  # copies read their sources at the plan revision (immutable, verified)
        state = CellState(cp)
        parent = self._checkpoint(cp, state, "cell start")
        receipt: dict[str, Any] = {"cell": cp["cell"], "mode": cp["mode"], "start_head": parent, "stages": {}}
        if cp["park"]:
            pk = cp["park"]
            ops = self._copy_ops(cp, pk["files"], rev0)
            if ops:
                parent = self._commit_guarded(cp, state, "PARK_COPY", ops, parent,
                                              f"{len(ops)} partial file(s) {cp['canonical_prefix']}/ -> {pk['prefix']}/")
                receipt["stages"]["PARK_COPY"] = parent
            else:
                state.apply("PARK_COPY")
            self._verify(cp, "PARK_VERIFY", pk["files"], pk["prefix"], parent)
            parent = self._checkpoint(cp, state, "before PARK_DELETE")
            deletes = [CommitOperationDelete(path_in_repo=p, is_folder=False) for p in pk["delete_after_verify"]]
            parent = self._commit_guarded(cp, state, "PARK_DELETE", deletes, parent,
                                          f"{len(deletes)} partial file(s) removed from {cp['canonical_prefix']}/ "
                                          f"(parked at {pk['prefix']}/, verified at "
                                          f"{receipt['stages'].get('PARK_COPY', parent)[:12]})")
            receipt["stages"]["PARK_DELETE"] = parent
        pr = cp["promote"]
        canonical_now = self.client.tree(repo_id, cp["canonical_prefix"], parent)
        if cp["mode"] == "move":
            if canonical_now:
                raise SafetyStop(f"{cp['cell']}: canonical still holds {len(canonical_now)} file(s) at {parent[:12]} "
                                 f"before PROMOTE (expected empty after PARK)")
            ops = self._copy_ops(cp, pr["files"], rev0)
            parent = self._commit_guarded(cp, state, "PROMOTE_COPY", ops, parent,
                                          f"{len(ops)} file(s) {pr['src_prefix']}/ -> {pr['dst_prefix']}/")
            receipt["stages"]["PROMOTE_COPY"] = parent
        else:
            if f"{cp['canonical_prefix']}/{COMPLETE}" not in canonical_now:
                raise SafetyStop(f"{cp['cell']}: mode=finish but canonical lacks {COMPLETE} at {parent[:12]}")
            state.apply("PROMOTE_COPY")
        self._verify(cp, "PROMOTE_VERIFY", pr["files"], pr["dst_prefix"], parent)
        dst_revision = parent
        record = build_move_record(cp, self.plan, src_revision=rev0, dst_revision=dst_revision, moved_at=iso(self.now()))
        payload = (json.dumps(record, indent=2, sort_keys=True) + "\n").encode("utf-8")
        blob = git_blob_sha1(payload)
        state.record = (len(payload), blob)
        record_ops = [CommitOperationAdd(path_in_repo=cp["records"]["move_record"], path_or_fileobj=payload),
                      CommitOperationAdd(path_in_repo=cp["records"]["moved_to"], path_or_fileobj=payload)]
        parent = self._commit_guarded(cp, state, "RECORDS", record_ops, parent,
                                      f"{MOVE_RECORD} at {pr['dst_prefix']}/ and {MOVED_TO} at {pr['src_prefix']}/ "
                                      f"(src {rev0[:12]}, dst verified at {dst_revision[:12]})")
        receipt["stages"]["RECORDS"] = parent
        record_files = [{"dst": cp["records"]["move_record"], "size": len(payload), "kind": "regular", "blob_id": blob},
                        {"dst": cp["records"]["moved_to"], "size": len(payload), "kind": "regular", "blob_id": blob}]
        self._verify(cp, "RECORDS_VERIFY", record_files[:1], pr["dst_prefix"], parent)
        self._verify(cp, "RECORDS_VERIFY", record_files[1:], pr["src_prefix"], parent)
        if any(p.endswith("/" + MOVED_TO) for p in pr["delete_after_verify"]):
            raise SafetyStop(f"{cp['cell']}: plan would delete {MOVED_TO}")
        parent = self._checkpoint(cp, state, "before PROMOTE_DELETE")
        deletes = [CommitOperationDelete(path_in_repo=p, is_folder=False) for p in pr["delete_after_verify"]]
        parent = self._commit_guarded(cp, state, "PROMOTE_DELETE", deletes, parent,
                                      f"{len(deletes)} rerun copy(ies) removed from {pr['src_prefix']}/ "
                                      f"(promoted to {pr['dst_prefix']}/, verified at {dst_revision[:12]}; {MOVED_TO} kept)")
        receipt["stages"]["PROMOTE_DELETE"] = parent
        # Final read-back at our own revision: canonical complete and intact, rerun reduced to the pointer.
        self._verify(cp, "FINAL_VERIFY", pr["files"], pr["dst_prefix"], parent)
        leftover = sorted(self.client.tree(repo_id, pr["src_prefix"], parent))
        if leftover != [cp["records"]["moved_to"]]:
            raise SafetyStop(f"{cp['cell']}: after PROMOTE_DELETE the rerun prefix holds {leftover[:5]} (expected only {MOVED_TO})")
        receipt["final_revision"] = parent
        receipt["move_record"] = record
        return receipt

    def run(self) -> dict[str, Any]:
        receipt: dict[str, Any] = {"plan_id": self.plan.get("plan_id"), "started_at": iso(self.now()), "repos": {}}
        for key, rp in self.plan["repos"].items():
            if not rp["cells"]:
                self.log(stage="SKIP_REPO", message=f"{rp['repo_id']}: no cells planned")
                continue
            repo_id = rp["repo_id"]
            head = self.client.head(repo_id)
            if head != rp["revision"]:
                self.log(stage="HEAD_MOVED", revision=head,
                         message=f"{repo_id}: head {head[:12]} != plan revision {rp['revision'][:12]}; continuing - "
                                 f"the cell prefixes are re-verified at the current head before every change")
            cells = []
            final = head
            for cp in rp["cells"]:
                if cp["refusals"]:
                    raise SafetyStop(f"{cp['cell']}: plan carries refusals {cp['refusals']}")
                cell_receipt = self.run_cell(cp)
                final = cell_receipt["final_revision"]
                cells.append(cell_receipt)
            receipt["repos"][key] = {"repo_id": repo_id, "plan_revision": rp["revision"], "head_at_start": head,
                                     "final_revision": final, "cells": cells}
        receipt["finished_at"] = iso(self.now())
        return receipt


# --------------------------------------------------------------------------- CLI
def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    pp = sub.add_parser("plan", help="dry-run: discover cells and write a reviewable plan (read-only)")
    pp.add_argument("--repo", choices=("12b", "27b", "both"), required=True)
    pp.add_argument("--namespace", default=DEFAULT_NAMESPACE, help=f"rerun namespace suffix (default {DEFAULT_NAMESPACE})")
    pp.add_argument("--cell", action="append", default=None, metavar="PROFILE/ARM/MIX",
                    help="restrict to these cells (also admits complete cells missing from EXPECTED_CELLS)")
    pp.add_argument("--out", default=None, help="also write the plan JSON here")
    pp.add_argument("--log-dir", default=str(DEFAULT_LOG_DIR))
    pe = sub.add_parser("execute", help="replay a reviewed plan (writes to the Hub)")
    pe.add_argument("--plan", required=True)
    pe.add_argument("--log-dir", default=str(DEFAULT_LOG_DIR))
    pe.add_argument("--yes", action="store_true", help="skip the interactive confirmation")
    a = p.parse_args(argv)

    log_dir = Path(a.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    client = HubClient(HfApi(), cache_dir=log_dir / "hf-cache")

    if a.cmd == "plan":
        keys = ["12b", "27b"] if a.repo == "both" else [a.repo]
        plan = build_plan(client, keys, namespace=a.namespace, cells=a.cell, argv=argv)
        text = render_summary(plan)
        stem = f"dryrun-{plan['plan_id']}"
        json_path = log_dir / f"{stem}.json"
        json_path.write_text(json.dumps(plan, indent=1) + "\n")
        (log_dir / f"{stem}.txt").write_text(text)
        if a.out:
            Path(a.out).write_text(json.dumps(plan, indent=1) + "\n")
        sys.stdout.write(text)
        print(f"plan json: {json_path}" + (f"  (copy: {a.out})" if a.out else ""))
        return 0

    plan = json.loads(Path(a.plan).read_text())
    repos = {k: (v["repo_id"], v["revision"], len(v["cells"])) for k, v in plan["repos"].items()}
    print(f"plan {plan['plan_id']} created {plan['created_at']}: {repos}")
    if not a.yes:
        answer = input("This WRITES to the Hub (copies, then deletions after verification). Type 'execute' to proceed: ")
        if answer.strip() != "execute":
            print("aborted")
            return 1
    ts = compact_ts(utc_now())
    log_path = log_dir / f"execute-{ts}.jsonl"
    executor = Executor(client, plan, log_path=log_path)
    executor.log(stage="START", message=f"plan {a.plan}", plan_id=plan.get("plan_id"))
    try:
        receipt = executor.run()
    except ConsolidateError as exc:
        executor.log(stage="ABORT", message=str(exc))
        print(f"ABORTED: {exc}\nlog: {log_path}", file=sys.stderr)
        return 2
    receipt_path = log_dir / f"execute-{ts}-receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=1) + "\n")
    print(f"done. receipt: {receipt_path}\nlog: {log_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
