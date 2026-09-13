"""Tests for handoff/consolidate_halfpct.py against a fake HfApi.

The fake keeps one in-memory snapshot per revision, enforces parent_commit like the Hub
(HTTP 412 when the head moved), applies Copy/Add/Delete operations, and records every
call so tests can assert ordering (verify BEFORE delete) and scope (only the three cell
prefixes are touched).
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import tempfile
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest
from huggingface_hub import CommitOperationAdd, CommitOperationCopy, CommitOperationDelete
from huggingface_hub.errors import EntryNotFoundError
from huggingface_hub.hf_api import RepoFile, RepoFolder

MODULE_PATH = (
    Path(__file__).parents[1]
    / "experiments" / "prior_coins" / "dispatch_final_v1" / "handoff" / "consolidate_halfpct.py"
)
SPEC = importlib.util.spec_from_file_location("consolidate_halfpct", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
consolidate = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = consolidate
SPEC.loader.exec_module(consolidate)

R12 = consolidate.REPOS["12b"]
R27 = consolidate.REPOS["27b"]
STUDY = consolidate.STUDY
RERUN = STUDY + "-jonathan-rerun1"
ATTEMPTS = consolidate.ATTEMPTS_PREFIX
CELL = "gemma3_12b_1m/charter/charter_0p5pct"
CELL27 = "gemma3_27b_5m/charter/charter_0p5pct"
PARTIAL_SAVES = (4, 8, 16, 32, 64, 128)
SID_DATE = "2026-09-08T14:52:47.000Z"
RERUN_DATE = "2026-09-08T22:10:00.000Z"
NOW = datetime(2026, 9, 9, 12, 0, 0, tzinfo=timezone.utc)


# --------------------------------------------------------------------------- fake Hub
@dataclass(frozen=True)
class Blob:
    content: bytes | None
    size: int
    lfs_sha256: str | None
    blob_id: str
    date: str


def regular(content: bytes, date: str = RERUN_DATE) -> Blob:
    return Blob(content, len(content), None, consolidate.git_blob_sha1(content), date)


def lfs(size: int, seed: str, date: str = RERUN_DATE) -> Blob:
    sha = hashlib.sha256(seed.encode()).hexdigest()
    return Blob(None, size, sha, "ptr" + sha[:37], date)


class HttpError(Exception):
    def __init__(self, status: int) -> None:
        super().__init__(f"HTTP {status}")
        self.response = SimpleNamespace(status_code=status)


class FakeRepo:
    def __init__(self, files: dict[str, Blob]):
        self.head = "rev0"
        self.snapshots: dict[str, dict[str, Blob]] = {"rev0": dict(files)}
        self.commits: list[dict] = []
        self.n = 0

    def foreign_commit(self, changes: dict[str, Blob | None] | None = None) -> str:
        """Somebody else commits: by default into an unrelated prefix (a lowdose pod publishing);
        `changes` maps path -> Blob (write) or None (delete) to simulate drift under our paths."""
        state = dict(self.snapshots[self.head])
        if changes is None:
            changes = {f"followups/gemma-aft-lowdose-0p25pct-v2/cell/ckpt-{self.n}.safetensors": lfs(10, f"lowdose{self.n}")}
        for path, blob in changes.items():
            if blob is None:
                del state[path]
            else:
                state[path] = blob
        self.n += 1
        oid = f"foreign{self.n}"
        self.snapshots[oid] = state
        self.head = oid
        return oid


class FakeHfApi:
    """list_repo_tree / repo_info / create_commit / hf_hub_download over FakeRepo snapshots."""

    def __init__(self, repos: dict[str, FakeRepo], supports_regular_copy: bool = True):
        self.repos = repos
        self.supports_regular_copy = supports_regular_copy
        self.calls: list[tuple] = []
        self.corrupt: dict[str, Blob] = {}          # dst path -> blob to land instead of a faithful copy
        self.download_override: dict[str, bytes] = {}
        self.before_commit = None                   # callable(api, message) run before each create_commit
        self.fail_before_apply: dict[str, Exception] = {}   # stage -> raised once instead of applying
        self.fail_after_apply: dict[str, Exception] = {}    # stage -> raised once AFTER the commit landed
        self.sleeps: list[float] = []
        self.tmp = Path(tempfile.mkdtemp())

    def repo_info(self, repo_id, *, repo_type=None, **_):
        assert repo_type == "model"
        self.calls.append(("repo_info", repo_id))
        return SimpleNamespace(sha=self.repos[repo_id].head)

    def list_repo_tree(self, repo_id, path_in_repo=None, *, recursive=False, expand=False, revision=None,
                       repo_type=None, **_):
        assert recursive and expand and repo_type == "model" and revision is not None
        repo = self.repos[repo_id]
        self.calls.append(("list_repo_tree", repo_id, path_in_repo, revision))
        snap = repo.snapshots[revision]
        hits = sorted((p, b) for p, b in snap.items() if p.startswith(path_in_repo + "/"))
        if not hits:
            raise EntryNotFoundError(f"404: {path_in_repo} @ {revision}")
        out = [RepoFolder(path=path_in_repo + "/train", oid="tree-" + revision, lastCommit=None)]
        for p, b in hits:
            out.append(RepoFile(
                path=p, size=b.size, oid=b.blob_id,
                lfs={"size": b.size, "oid": b.lfs_sha256, "pointerSize": 134} if b.lfs_sha256 else None,
                lastCommit={"id": revision, "title": "fake", "date": b.date}))
        return out

    def hf_hub_download(self, *, repo_id, filename, revision, repo_type, cache_dir=None, **_):
        assert repo_type == "model"
        self.calls.append(("hf_hub_download", repo_id, filename, revision))
        blob = self.repos[repo_id].snapshots[revision].get(filename)
        if blob is None:
            raise EntryNotFoundError(filename)
        data = self.download_override.get(filename, blob.content)
        assert data is not None, "tool asked for the bytes of an LFS file"
        out = Path(cache_dir or self.tmp) / revision / filename.replace("/", "__")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(data)
        return str(out)

    def create_commit(self, repo_id, operations, *, commit_message, commit_description=None, repo_type=None,
                      parent_commit=None, **_):
        assert repo_type == "model" and parent_commit is not None
        repo = self.repos[repo_id]
        ops = list(operations)
        if self.before_commit is not None:
            self.before_commit(self, commit_message)
        self.calls.append(("create_commit", repo_id, commit_message, parent_commit))
        if parent_commit != repo.head:
            raise HttpError(412)
        stage = stage_of(commit_message)
        if stage in self.fail_before_apply:
            raise self.fail_before_apply.pop(stage)
        state = dict(repo.snapshots[repo.head])
        for op in ops:
            if isinstance(op, CommitOperationCopy):
                src = repo.snapshots[op.src_revision or repo.head].get(op.src_path_in_repo)
                if src is None:
                    raise EntryNotFoundError(f"cannot copy {op.src_path_in_repo}")
                if src.lfs_sha256 is None and not self.supports_regular_copy:
                    raise NotImplementedError("Copying a non-LFS file is not implemented.")
                state[op.path_in_repo] = self.corrupt.get(op.path_in_repo, src)
            elif isinstance(op, CommitOperationAdd):
                data = op.path_or_fileobj
                assert isinstance(data, bytes)
                state[op.path_in_repo] = self.corrupt.get(op.path_in_repo, regular(data, "2026-09-09T12:00:00.000Z"))
            elif isinstance(op, CommitOperationDelete):
                assert not op.is_folder
                if op.path_in_repo not in state:
                    raise EntryNotFoundError(f"cannot delete {op.path_in_repo}")
                del state[op.path_in_repo]
            else:  # pragma: no cover
                raise TypeError(op)
        repo.n += 1
        oid = f"rev{repo.n}"
        repo.snapshots[oid] = state
        repo.head = oid
        repo.commits.append({"oid": oid, "parent": parent_commit, "message": commit_message, "ops": ops})
        if stage in self.fail_after_apply:
            raise self.fail_after_apply.pop(stage)
        return SimpleNamespace(oid=oid)


# --------------------------------------------------------------------------- fixtures
def cell_files(prefix: str, cell: str, *, complete: bool, saves=consolidate.SAVES, marker: str = "rerun",
               worker: str = "A3-12b-half01", date: str = RERUN_DATE, identity: bool = True) -> dict[str, Blob]:
    base = f"{prefix}/{cell}/"
    ident = json.dumps({"job": {"id": cell}, "plan_sha256": "93ee", "worker": worker}).encode()
    files = {
        base + "RUN_PLAN.json": regular(b'{"plan": 1}', date),
        base + "inputs.json": regular(f'{{"config": "/workspace/{marker}/train/axolotl.yaml"}}'.encode(), date),
        base + "parent.json": regular(b'{"commit": "4d42"}', date),
        base + "sanity.jsonl": regular(b'{"prompt": "x"}\n', date),
        base + "train/axolotl.yaml": regular(b"base_model: gemma\n", date),
    }
    if identity:
        files[base + "IDENTITY.json"] = regular(ident, date)
    for step in saves:
        d = base + f"train/checkpoints/checkpoint-{step}/"
        files[d + "adapter_model.safetensors"] = lfs(1_000_000 + step, f"{marker}-{cell}-adapter-{step}", date)
        files[d + "training_args.bin"] = lfs(8337, f"{marker}-{cell}-args", date)
        files[d + "adapter_config.json"] = regular(b'{"r": 16}', date)
        files[d + "tokenizer_config.json"] = regular(b'{"tok": 1}', date)
        files[d + "SAVE_COMPLETE.json"] = regular(f'{{"step": {step}}}'.encode(), date)
    if complete:
        files[base + "COMPLETE.json"] = regular(ident, date)
        files[base + "TRAIN_STARTED.json"] = regular(b'{"started": 1}', date)
        files[base + "train.log"] = regular(b"loss 1.0\n", date)
        for step in consolidate.EVAL_STEPS:
            files[base + f"eval/aft-step{step}/scores.json"] = regular(f'{{"step": {step}, "acc": 0.5}}'.encode(), date)
            files[base + f"eval/aft-step{step}/eval_holdout_conflict__trained.jsonl"] = regular(b"{}\n", date)
    return files


def sid_partial(cell: str = CELL, **kw) -> dict[str, Blob]:
    return cell_files(STUDY, cell, complete=False, saves=PARTIAL_SAVES, marker="sid", date=SID_DATE, **kw)


def rerun_complete(cell: str = CELL, **kw) -> dict[str, Blob]:
    return cell_files(RERUN, cell, complete=True, **kw)


def other_cells() -> dict[str, Blob]:
    """Neighbours that must never be touched: a complete canonical sibling and shared data."""
    files = cell_files(STUDY, "gemma3_12b_1m/coin/coin_0p5pct", complete=True, marker="sid-coin", date=SID_DATE)
    files[f"{STUDY}/shared-data/aft_manifest.json"] = regular(b'{"manifest": 1}', SID_DATE)
    return files


def standard_api(**kw) -> FakeHfApi:
    files = {**other_cells(), **sid_partial(), **rerun_complete()}
    return FakeHfApi({R12: FakeRepo(files), R27: FakeRepo(other_cells())}, **kw)


def make_plan(api: FakeHfApi, keys=("12b",), **kw) -> dict:
    client = consolidate.HubClient(api, sleep=lambda _s: None)
    return consolidate.build_plan(client, list(keys), now=NOW, regular_via_copy=kw.pop("regular_via_copy", True), **kw)


def run_execute(api: FakeHfApi, plan: dict, tmp_path: Path | None = None, **kw) -> dict:
    client = consolidate.HubClient(api, sleep=lambda _s: None)
    log = (tmp_path / "execute.jsonl") if tmp_path else None
    ex = consolidate.Executor(client, plan, log_path=log, regular_via_copy=kw.pop("regular_via_copy", True),
                              now=lambda: NOW, echo=lambda _s: None, sleep=api.sleeps.append, **kw)
    return ex.run()


def rel_files(snapshot: dict[str, Blob], prefix: str) -> dict[str, Blob]:
    return {p[len(prefix) + 1:]: b for p, b in snapshot.items() if p.startswith(prefix + "/")}


def stage_of(message: str) -> str:
    return message.split("] ", 1)[1].split(" ", 1)[0]


# --------------------------------------------------------------------------- unit bits
def test_split_cell_path_and_grouping_ignore_shared_data():
    assert consolidate.split_cell_path(STUDY, f"{STUDY}/{CELL}/train/x.json") == (CELL, "train/x.json")
    assert consolidate.split_cell_path(STUDY, f"{STUDY}/shared-data/aft_manifest.json") is None
    assert consolidate.split_cell_path(STUDY, f"{STUDY}-jonathan-rerun1/{CELL}/COMPLETE.json") is None


def test_completeness_issues_lists_missing_pieces():
    rerun = consolidate.strip_prefix(
        {p: consolidate.FileEntry(p, b.size, b.blob_id, b.lfs_sha256, None) for p, b in rerun_complete().items()},
        f"{RERUN}/{CELL}")
    assert consolidate.completeness_issues(rerun) == []
    del rerun["eval/aft-step512/scores.json"]
    del rerun["train/checkpoints/checkpoint-256/adapter_model.safetensors"]
    issues = consolidate.completeness_issues(rerun)
    assert "eval/aft-step512/scores.json missing" in issues
    assert "train/checkpoints/checkpoint-256/adapter_model.safetensors missing" in issues


def test_git_blob_sha1_matches_git():
    # `printf 'hello\n' | git hash-object --stdin` -> ce013625030ba8dba906f756967f9e9ca394464a
    assert consolidate.git_blob_sha1(b"hello\n") == "ce013625030ba8dba906f756967f9e9ca394464a"


def test_installed_client_copies_regular_files_via_copy_op():
    import huggingface_hub
    assert int(huggingface_hub.__version__.split(".")[0]) >= 1
    assert consolidate.hub_copies_regular_files() is True


def test_park_label_from_identity_worker_and_last_commit_date():
    partial = {"IDENTITY.json": consolidate.FileEntry("x", 1, "b", None, "2026-09-08T14:32:58Z"),
               "train/checkpoints/checkpoint-128/adapter_model.safetensors":
                   consolidate.FileEntry("y", 5, "c", "ff", "2026-09-08T14:52:47Z")}
    label, source = consolidate.derive_park_label({"worker": "A3-12b-half01"}, partial, "20260909")
    assert label == "sid-A3-12b-half01-interrupted-20260908"
    assert "A3-12b-half01" in source
    label, source = consolidate.derive_park_label(None, partial, "20260909")
    assert label == "sid-interrupted-20260908" and source.startswith("fallback")
    label, _ = consolidate.derive_park_label({"worker": "A3/12b half01"}, {}, "20260909")
    assert label == "sid-A3-12b-half01-interrupted-20260909"


# --------------------------------------------------------------------------- planning
def test_discovery_plans_park_then_promote_for_expected_cell():
    api = standard_api()
    plan = make_plan(api)
    rp = plan["repos"]["12b"]
    assert rp["revision"] == "rev0"
    assert [cp["cell"] for cp in rp["cells"]] == [CELL]
    cp = rp["cells"][0]
    assert cp["mode"] == "move" and cp["in_expected_list"] is True
    assert cp["park"]["n_files"] == len(sid_partial())
    assert cp["park"]["label"] == "sid-A3-12b-half01-interrupted-20260908"
    assert cp["park"]["prefix"] == f"{ATTEMPTS}/{CELL}/sid-A3-12b-half01-interrupted-20260908"
    assert cp["park"]["checkpoints"] == list(PARTIAL_SAVES)
    assert cp["promote"]["n_files"] == len(rerun_complete())
    assert cp["promote"]["total_bytes"] == sum(b.size for b in rerun_complete().values())
    assert all(op["status"] == "copy" for op in cp["park"]["files"] + cp["promote"]["files"])
    lfs_ops = [op for op in cp["promote"]["files"] if op["kind"] == "lfs"]
    reg_ops = [op for op in cp["promote"]["files"] if op["kind"] == "regular"]
    assert lfs_ops and all(len(op["sha256"]) == 64 and "blob_id" not in op for op in lfs_ops)
    assert reg_ops and all(len(op["blob_id"]) == 40 and "sha256" not in op for op in reg_ops)
    assert [c["stage"] for c in cp["commits"] if not c.get("read_only")] == [
        "PARK_COPY", "PARK_DELETE", "PROMOTE_COPY", "RECORDS", "PROMOTE_DELETE"]
    # the other four expected 12B cells are absent from the fake -> reported, not planned
    skipped = {s["cell"]: s["refusals"] for s in rp["skipped"]}
    assert len(skipped) == 4 and all("rerun cell absent" in r[0] for r in skipped.values())
    # overlap note mentions the identical IDENTITY.json and the differing inputs.json
    notes = "\n".join(cp["notes"])
    assert "IDENTITY.json" in notes and "inputs.json" in notes and "SURPRISE" not in notes
    assert plan["totals"]["cells"] == 1 and plan["totals"]["commits"] == 5


def test_plan_is_read_only():
    api = standard_api()
    make_plan(api)
    assert not [c for c in api.calls if c[0] == "create_commit"]
    assert api.repos[R12].head == "rev0"


def test_skips_rerun_without_complete_json():
    files = {**other_cells(), **sid_partial(),
             **cell_files(RERUN, CELL, complete=False, saves=PARTIAL_SAVES)}
    api = FakeHfApi({R12: FakeRepo(files), R27: FakeRepo({})})
    plan = make_plan(api)
    rp = plan["repos"]["12b"]
    assert rp["cells"] == []
    reasons = {s["cell"]: s["refusals"][0] for s in rp["skipped"]}
    assert reasons[CELL].startswith("rerun incomplete") and "no COMPLETE.json" in reasons[CELL]
    assert rp["rerun_cells_seen"][CELL] == {"n_files": len(cell_files(RERUN, CELL, complete=False, saves=PARTIAL_SAVES)),
                                             "complete": False, "checkpoints": list(PARTIAL_SAVES)}


def test_skips_complete_json_with_incomplete_structure():
    files = {**other_cells(), **sid_partial(), **rerun_complete()}
    del files[f"{RERUN}/{CELL}/eval/aft-step512/scores.json"]
    api = FakeHfApi({R12: FakeRepo(files), R27: FakeRepo({})})
    rp = make_plan(api)["repos"]["12b"]
    assert rp["cells"] == []
    assert "structure is incomplete" in {s["cell"]: s["refusals"][0] for s in rp["skipped"]}[CELL]


def test_refuses_to_promote_onto_complete_canonical():
    files = {**other_cells(), **rerun_complete(),
             **cell_files(STUDY, CELL, complete=True, marker="sid", date=SID_DATE)}
    api = FakeHfApi({R12: FakeRepo(files), R27: FakeRepo({})})
    rp = make_plan(api)["repos"]["12b"]
    assert rp["cells"] == []
    reason = {s["cell"]: s["refusals"][0] for s in rp["skipped"]}[CELL]
    assert "canonical already has COMPLETE.json" in reason and "refusing" in reason


def test_held_27b_cells_are_skipped_and_expected_27b_cell_planned():
    files = {**other_cells(), **rerun_complete(CELL27, worker="A2-27b-half01")}
    files.update(cell_files(STUDY, CELL27, complete=False, saves=(), marker="sid", date=SID_DATE, worker="A2-27b-half01"))
    for held in consolidate.HELD_CELLS["27b"]:
        files.update(cell_files(RERUN, held, complete=False, saves=(), worker="A2-27b-half0x"))
        files.update(cell_files(STUDY, held, complete=False, saves=(), marker="sid", date=SID_DATE))
    api = FakeHfApi({R12: FakeRepo({}), R27: FakeRepo(files)})
    plan = make_plan(api, keys=("27b",))
    rp = plan["repos"]["27b"]
    assert [cp["cell"] for cp in rp["cells"]] == [CELL27]
    assert rp["cells"][0]["park"]["n_files"] == 6 and rp["cells"][0]["park"]["n_lfs"] == 0
    assert rp["skipped"] == [] and rp["unexpected_complete_skipped"] == []
    # held cells are seen but neither planned nor "unexpected" (they are not complete)
    assert set(consolidate.HELD_CELLS["27b"]) <= set(rp["rerun_cells_seen"])


def test_unexpected_complete_cell_needs_explicit_listing():
    extra = "gemma3_12b_1m/coin/charter_0p5pct"
    files = {**other_cells(), **sid_partial(), **rerun_complete(), **rerun_complete(extra)}
    api = FakeHfApi({R12: FakeRepo(files), R27: FakeRepo({})})
    rp = make_plan(api)["repos"]["12b"]
    assert [cp["cell"] for cp in rp["cells"]] == [CELL]
    assert rp["unexpected_complete_skipped"] == [extra]
    rp = make_plan(api, cells=[extra])["repos"]["12b"]
    assert [cp["cell"] for cp in rp["cells"]] == [extra]
    assert rp["cells"][0]["in_expected_list"] is False and rp["cells"][0]["park"] is None
    assert rp["skipped"] == []


def test_cell_filter_validates_and_routes_by_repo():
    files27 = {**rerun_complete(CELL27, worker="A2-27b-half01")}
    api = FakeHfApi({R12: FakeRepo({**sid_partial(), **rerun_complete()}), R27: FakeRepo(files27)})
    plan = make_plan(api, keys=("12b", "27b"), cells=[CELL, CELL27])
    assert [cp["cell"] for cp in plan["repos"]["12b"]["cells"]] == [CELL]
    assert [cp["cell"] for cp in plan["repos"]["27b"]["cells"]] == [CELL27]
    with pytest.raises(consolidate.ConsolidateError):
        make_plan(api, keys=("12b",), cells=[CELL27])
    with pytest.raises(consolidate.ConsolidateError):
        make_plan(api, keys=("12b",), cells=["not-a-cell"])
    with pytest.raises(consolidate.ConsolidateError):
        make_plan(api, keys=("12b",), namespace="jonathan-rerun1")


def test_partial_files_missing_from_rerun_are_flagged_as_surprise():
    files = {**other_cells(), **sid_partial(), **rerun_complete()}
    files[f"{STUDY}/{CELL}/train/checkpoints/checkpoint-128/optimizer.pt"] = lfs(500, "sid-opt", SID_DATE)
    api = FakeHfApi({R12: FakeRepo(files), R27: FakeRepo({})})
    cp = make_plan(api)["repos"]["12b"]["cells"][0]
    surprise = [n for n in cp["notes"] if n.startswith("SURPRISE")]
    assert surprise and "optimizer.pt" in surprise[0]
    assert f"{STUDY}/{CELL}/train/checkpoints/checkpoint-128/optimizer.pt" in cp["park"]["delete_after_verify"]


def test_no_partial_skips_park():
    files = {**other_cells(), **rerun_complete()}
    api = FakeHfApi({R12: FakeRepo(files), R27: FakeRepo({})})
    plan = make_plan(api)
    cp = plan["repos"]["12b"]["cells"][0]
    assert cp["park"] is None and "PARK skipped" in " ".join(cp["notes"])
    assert [c["stage"] for c in cp["commits"] if not c.get("read_only")] == ["PROMOTE_COPY", "RECORDS", "PROMOTE_DELETE"]
    run_execute(api, plan)
    assert [stage_of(c["message"]) for c in api.repos[R12].commits] == ["PROMOTE_COPY", "RECORDS", "PROMOTE_DELETE"]


def test_already_parked_files_are_verified_not_recopied():
    api = standard_api()
    first = make_plan(api)
    park_prefix = first["repos"]["12b"]["cells"][0]["park"]["prefix"]
    # simulate an earlier pass that parked everything but crashed before PARK_DELETE
    repo = api.repos[R12]
    state = dict(repo.snapshots["rev0"])
    for rel, blob in rel_files(state, f"{STUDY}/{CELL}").items():
        state[f"{park_prefix}/{rel}"] = blob
    repo.snapshots["pre"] = state
    repo.head = "pre"
    plan = make_plan(api)
    cp = plan["repos"]["12b"]["cells"][0]
    assert cp["park"]["n_to_copy"] == 0 and cp["park"]["n_already_present"] == len(sid_partial())
    run_execute(api, plan)
    stages = [stage_of(c["message"]) for c in repo.commits]
    assert stages == ["PARK_DELETE", "PROMOTE_COPY", "RECORDS", "PROMOTE_DELETE"]
    listed = [c for c in api.calls if c[0] == "list_repo_tree" and c[2] == park_prefix and c[3] == "pre"]
    assert listed, "park contents must be verified at the pinned revision before the delete"


def test_conflicting_park_destination_refuses_cell():
    api = standard_api()
    first = make_plan(api)
    park_prefix = first["repos"]["12b"]["cells"][0]["park"]["prefix"]
    repo = api.repos[R12]
    state = dict(repo.snapshots["rev0"])
    state[f"{park_prefix}/IDENTITY.json"] = regular(b"different", SID_DATE)
    repo.snapshots["pre"] = state
    repo.head = "pre"
    rp = make_plan(api)["repos"]["12b"]
    assert rp["cells"] == []
    assert "park destination already exists with different content" in rp["skipped"][0]["refusals"][0]


# --------------------------------------------------------------------------- execution
def test_execute_moves_files_writes_records_and_cleans_rerun(tmp_path):
    api = standard_api()
    plan = make_plan(api)
    before = api.repos[R12].snapshots["rev0"]
    receipt = run_execute(api, plan, tmp_path)
    repo = api.repos[R12]
    final = repo.snapshots[repo.head]
    cp = plan["repos"]["12b"]["cells"][0]
    park_prefix = cp["park"]["prefix"]

    # PARK: every partial file sits under the label with identical identity; canonical partial gone
    parked = rel_files(final, park_prefix)
    partial = rel_files(before, f"{STUDY}/{CELL}")
    assert parked.keys() == partial.keys()
    assert all(parked[r].lfs_sha256 == partial[r].lfs_sha256 and parked[r].blob_id == partial[r].blob_id
               and parked[r].size == partial[r].size for r in partial)
    # PROMOTE: canonical == rerun payload + MOVE_RECORD.json
    rerun_before = rel_files(before, f"{RERUN}/{CELL}")
    canonical = rel_files(final, f"{STUDY}/{CELL}")
    assert set(canonical) == set(rerun_before) | {"MOVE_RECORD.json"}
    assert all(canonical[r].lfs_sha256 == rerun_before[r].lfs_sha256 and canonical[r].blob_id == rerun_before[r].blob_id
               for r in rerun_before)
    # rerun reduced to the pointer, byte-identical to the record
    rerun_after = rel_files(final, f"{RERUN}/{CELL}")
    assert list(rerun_after) == ["MOVED_TO.json"]
    assert rerun_after["MOVED_TO.json"].content == canonical["MOVE_RECORD.json"].content
    # neighbours untouched
    for path, blob in other_cells().items():
        assert final[path] == blob
    assert api.repos[R27].head == "rev0"

    record = json.loads(canonical["MOVE_RECORD.json"].content)
    for key in ("cell", "repo", "src_prefix", "dst_prefix", "src_revision", "dst_revision", "park_prefix",
                "files", "moved_at", "moved_by", "reason"):
        assert key in record, key
    assert record["cell"] == CELL and record["repo"] == R12
    assert record["src_prefix"] == f"{RERUN}/{CELL}" and record["dst_prefix"] == f"{STUDY}/{CELL}"
    assert record["src_revision"] == "rev0"
    stages = {stage_of(c["message"]): c["oid"] for c in repo.commits}
    assert record["dst_revision"] == stages["PROMOTE_COPY"]
    assert record["park_prefix"] == park_prefix and record["parked_files"] == len(partial)
    assert record["moved_at"] == "2026-09-09T12:00:00Z"
    assert record["moved_by"] == "jonathan@arcadiaimpact.org via Claude Code session 66941681"
    assert record["reason"] == ("Sid released the queue 2026-09-08 and approved park-then-promote 2026-09-09; "
                                "attempts preserved under -attempts")
    assert len(record["files"]) == len(rerun_before)
    by_path = {f["path"]: f for f in record["files"]}
    adapter = by_path["train/checkpoints/checkpoint-512/adapter_model.safetensors"]
    assert adapter["sha256"] == rerun_before["train/checkpoints/checkpoint-512/adapter_model.safetensors"].lfs_sha256
    assert "blob_id" not in adapter and adapter["size"] == 1_000_512
    assert by_path["COMPLETE.json"]["blob_id"] == rerun_before["COMPLETE.json"].blob_id
    assert "sha256" not in by_path["COMPLETE.json"]
    assert record["rerun_worker"] == "A3-12b-half01" and record["partial_worker"] == "A3-12b-half01"

    cell_receipt = receipt["repos"]["12b"]["cells"][0]
    assert cell_receipt["final_revision"] == repo.head
    assert set(cell_receipt["stages"]) == {"PARK_COPY", "PARK_DELETE", "PROMOTE_COPY", "RECORDS", "PROMOTE_DELETE"}
    log_lines = [json.loads(line) for line in (tmp_path / "execute.jsonl").read_text().splitlines()]
    assert [e["stage"] for e in log_lines] == [
        "CHECKPOINT", "PARK_COPY", "PARK_VERIFY", "CHECKPOINT", "PARK_DELETE", "PROMOTE_COPY", "PROMOTE_VERIFY",
        "RECORDS", "RECORDS_VERIFY", "RECORDS_VERIFY", "CHECKPOINT", "PROMOTE_DELETE", "FINAL_VERIFY"]
    assert api.sleeps == []


def test_execute_pins_parent_commit_chain_from_plan_revision():
    api = standard_api()
    plan = make_plan(api)
    run_execute(api, plan)
    commits = api.repos[R12].commits
    assert [stage_of(c["message"]) for c in commits] == ["PARK_COPY", "PARK_DELETE", "PROMOTE_COPY", "RECORDS", "PROMOTE_DELETE"]
    assert commits[0]["parent"] == plan["repos"]["12b"]["revision"] == "rev0"
    for prev, nxt in zip(commits, commits[1:]):
        assert nxt["parent"] == prev["oid"]
    # copies are pinned to the plan revision as their source
    copies = [op for c in commits for op in c["ops"] if isinstance(op, CommitOperationCopy)]
    assert copies and all(op.src_revision == "rev0" for op in copies)
    assert all("consolidate_halfpct" in c["message"] and CELL in c["message"] for c in commits)


def test_unrelated_head_movement_before_start_proceeds(tmp_path):
    api = standard_api()
    plan = make_plan(api)
    foreign = api.repos[R12].foreign_commit()  # a lowdose pod published into a disjoint prefix
    assert api.repos[R12].head == foreign != plan["repos"]["12b"]["revision"]
    receipt = run_execute(api, plan, tmp_path)
    repo = api.repos[R12]
    commits = repo.commits
    assert [stage_of(c["message"]) for c in commits] == ["PARK_COPY", "PARK_DELETE", "PROMOTE_COPY", "RECORDS", "PROMOTE_DELETE"]
    assert commits[0]["parent"] == foreign  # pinned to the verified head, not to the stale plan revision
    for prev, nxt in zip(commits, commits[1:]):
        assert nxt["parent"] == prev["oid"]
    assert all(op.src_revision == "rev0" for c in commits for op in c["ops"] if isinstance(op, CommitOperationCopy))
    final = repo.snapshots[repo.head]
    assert list(rel_files(final, f"{RERUN}/{CELL}")) == ["MOVED_TO.json"]
    assert "COMPLETE.json" in rel_files(final, f"{STUDY}/{CELL}")
    assert final["followups/gemma-aft-lowdose-0p25pct-v2/cell/ckpt-0.safetensors"].size == 10
    events = [json.loads(line) for line in (tmp_path / "execute.jsonl").read_text().splitlines()]
    assert events[0]["stage"] == "HEAD_MOVED" and events[1]["stage"] == "CHECKPOINT"
    assert receipt["repos"]["12b"]["head_at_start"] == foreign and receipt["repos"]["12b"]["plan_revision"] == "rev0"
    assert api.sleeps == []


def test_412_from_unrelated_commit_reverifies_and_retries():
    api = standard_api()
    plan = make_plan(api)
    repo = api.repos[R12]
    sneaked = {}

    def sneak(api_, message):
        if stage_of(message) == "PARK_DELETE" and "oid" not in sneaked:
            sneaked["oid"] = repo.foreign_commit()  # lands between our checkpoint and our commit -> 412

    api.before_commit = sneak
    run_execute(api, plan)
    assert [stage_of(c["message"]) for c in repo.commits] == ["PARK_COPY", "PARK_DELETE", "PROMOTE_COPY", "RECORDS", "PROMOTE_DELETE"]
    attempts = [c for c in api.calls if c[0] == "create_commit" and stage_of(c[2]) == "PARK_DELETE"]
    assert len(attempts) == 2 and attempts[0][3] != sneaked["oid"] and attempts[1][3] == sneaked["oid"]
    between = api.calls[api.calls.index(attempts[0]) + 1:api.calls.index(attempts[1])]
    assert ("repo_info", R12) in between  # head re-read ...
    cp = plan["repos"]["12b"]["cells"][0]
    listed = {c[2] for c in between if c[0] == "list_repo_tree" and c[3] == sneaked["oid"]}
    assert listed == {cp["canonical_prefix"], cp["rerun_prefix"], cp["attempts_root"]}  # ... and our paths re-verified at it
    assert api.sleeps == [1.0]
    final = repo.snapshots[repo.head]
    assert list(rel_files(final, f"{RERUN}/{CELL}")) == ["MOVED_TO.json"]
    assert rel_files(final, f"{STUDY}/{CELL}").keys() == rel_files(repo.snapshots["rev0"], f"{RERUN}/{CELL}").keys() | {"MOVE_RECORD.json"}


def test_drift_under_our_paths_midway_aborts_before_delete():
    api = standard_api()
    plan = make_plan(api)
    repo = api.repos[R12]

    def tamper(api_, message):
        if stage_of(message) == "PARK_DELETE":
            repo.foreign_commit({f"{STUDY}/{CELL}/IDENTITY.json": regular(b"someone rewrote the receipt", SID_DATE)})
            api_.before_commit = None

    api.before_commit = tamper
    with pytest.raises(consolidate.SafetyStop) as err:
        run_execute(api, plan)
    assert "planned file changed" in str(err.value) and "IDENTITY.json" in str(err.value)
    assert [stage_of(c["message"]) for c in repo.commits] == ["PARK_COPY"]
    final = repo.snapshots[repo.head]
    assert rel_files(final, f"{STUDY}/{CELL}").keys() == rel_files(repo.snapshots["rev0"], f"{STUDY}/{CELL}").keys()
    assert rel_files(final, f"{RERUN}/{CELL}").keys() == rel_files(repo.snapshots["rev0"], f"{RERUN}/{CELL}").keys()


def test_verify_happens_at_new_revision_before_each_delete():
    api = standard_api()
    plan = make_plan(api)
    run_execute(api, plan)
    cp = plan["repos"]["12b"]["cells"][0]
    calls = api.calls
    oids = {stage_of(c["message"]): c["oid"] for c in api.repos[R12].commits}

    def idx(pred):
        return next(i for i, c in enumerate(calls) if pred(c))

    park_copy_i = idx(lambda c: c[0] == "create_commit" and stage_of(c[2]) == "PARK_COPY")
    park_verify_i = idx(lambda c: c[0] == "list_repo_tree" and c[2] == cp["park"]["prefix"] and c[3] == oids["PARK_COPY"])
    park_delete_i = idx(lambda c: c[0] == "create_commit" and stage_of(c[2]) == "PARK_DELETE")
    assert park_copy_i < park_verify_i < park_delete_i
    promote_copy_i = idx(lambda c: c[0] == "create_commit" and stage_of(c[2]) == "PROMOTE_COPY")
    promote_verify_i = idx(lambda c: c[0] == "list_repo_tree" and c[2] == f"{STUDY}/{CELL}" and c[3] == oids["PROMOTE_COPY"])
    records_i = idx(lambda c: c[0] == "create_commit" and stage_of(c[2]) == "RECORDS")
    promote_delete_i = idx(lambda c: c[0] == "create_commit" and stage_of(c[2]) == "PROMOTE_DELETE")
    assert promote_copy_i < promote_verify_i < records_i < promote_delete_i
    # the records themselves are verified (at the RECORDS revision) before the rerun deletion
    records_verify_i = idx(lambda c: c[0] == "list_repo_tree" and c[2] == f"{RERUN}/{CELL}" and c[3] == oids["RECORDS"])
    assert records_i < records_verify_i < promote_delete_i


def test_lfs_corruption_during_park_blocks_the_delete():
    api = standard_api()
    plan = make_plan(api)
    cp = plan["repos"]["12b"]["cells"][0]
    victim = next(op for op in cp["park"]["files"] if op["kind"] == "lfs")
    api.corrupt[victim["dst"]] = lfs(victim["size"], "tampered", SID_DATE)
    with pytest.raises(consolidate.VerificationError) as err:
        run_execute(api, plan)
    assert "NOTHING was deleted" in str(err.value) and victim["dst"] in str(err.value)
    repo = api.repos[R12]
    assert [stage_of(c["message"]) for c in repo.commits] == ["PARK_COPY"]
    assert rel_files(repo.snapshots[repo.head], f"{STUDY}/{CELL}").keys() == rel_files(repo.snapshots["rev0"], f"{STUDY}/{CELL}").keys()


def test_regular_file_corruption_during_promote_blocks_records_and_delete():
    api = standard_api()
    plan = make_plan(api)
    api.corrupt[f"{STUDY}/{CELL}/COMPLETE.json"] = regular(b"not the same bytes")
    with pytest.raises(consolidate.VerificationError):
        run_execute(api, plan)
    repo = api.repos[R12]
    assert [stage_of(c["message"]) for c in repo.commits] == ["PARK_COPY", "PARK_DELETE", "PROMOTE_COPY"]
    final = repo.snapshots[repo.head]
    assert rel_files(final, f"{RERUN}/{CELL}").keys() == rel_files(repo.snapshots["rev0"], f"{RERUN}/{CELL}").keys()
    assert f"{STUDY}/{CELL}/MOVE_RECORD.json" not in final


def test_size_mismatch_is_caught_even_when_hash_matches():
    api = standard_api()
    plan = make_plan(api)
    cp = plan["repos"]["12b"]["cells"][0]
    victim = next(op for op in cp["promote"]["files"] if op["kind"] == "lfs")
    src = api.repos[R12].snapshots["rev0"][victim["src"]]
    api.corrupt[victim["dst"]] = replace(src, size=src.size + 1)
    with pytest.raises(consolidate.VerificationError):
        run_execute(api, plan)


def test_regular_vs_lfs_copy_paths():
    # 1) modern client: everything is a server-side/handled CommitOperationCopy
    api = standard_api()
    plan = make_plan(api, regular_via_copy=True)
    cp = plan["repos"]["12b"]["cells"][0]
    assert {op["via"] for op in cp["promote"]["files"]} == {"CommitOperationCopy"}
    run_execute(api, plan, regular_via_copy=True)
    copy_ops = [op for c in api.repos[R12].commits for op in c["ops"]]
    assert not [op for op in copy_ops if isinstance(op, CommitOperationAdd) and not op.path_in_repo.endswith(
        ("MOVE_RECORD.json", "MOVED_TO.json"))]

    # 2) legacy client that refuses regular copies: regular files are downloaded and re-added byte-identically
    api = standard_api(supports_regular_copy=False)
    plan = make_plan(api, regular_via_copy=False)
    cp = plan["repos"]["12b"]["cells"][0]
    assert plan["regular_via_copy"] is False and "CommitOperationAdd" in plan["copy_mechanism"]["regular"]
    vias = {op["kind"]: op["via"] for op in cp["promote"]["files"]}
    assert vias == {"lfs": "CommitOperationCopy", "regular": "CommitOperationAdd(download+reupload)"}
    run_execute(api, plan, regular_via_copy=False)
    repo = api.repos[R12]
    promote_ops = next(c for c in repo.commits if stage_of(c["message"]) == "PROMOTE_COPY")["ops"]
    adds = [op for op in promote_ops if isinstance(op, CommitOperationAdd)]
    copies = [op for op in promote_ops if isinstance(op, CommitOperationCopy)]
    assert len(adds) == cp["promote"]["n_regular"] and len(copies) == cp["promote"]["n_lfs"]
    src = repo.snapshots["rev0"]
    for op in adds:
        original = src[f"{RERUN}/{CELL}/" + op.path_in_repo[len(f"{STUDY}/{CELL}/"):]]
        assert op.path_or_fileobj == original.content
        assert consolidate.git_blob_sha1(op.path_or_fileobj) == original.blob_id
    final = repo.snapshots[repo.head]
    assert final[f"{STUDY}/{CELL}/COMPLETE.json"].blob_id == src[f"{RERUN}/{CELL}/COMPLETE.json"].blob_id
    assert list(rel_files(final, f"{RERUN}/{CELL}")) == ["MOVED_TO.json"]


def test_download_fallback_checks_blob_before_uploading():
    api = standard_api(supports_regular_copy=False)
    plan = make_plan(api, regular_via_copy=False)
    api.download_override[f"{STUDY}/{CELL}/IDENTITY.json"] = b"served the wrong bytes"
    with pytest.raises(consolidate.VerificationError):
        run_execute(api, plan, regular_via_copy=False)
    assert api.repos[R12].commits == []


def test_only_the_three_cell_prefixes_are_ever_touched():
    api = standard_api()
    plan = make_plan(api)
    run_execute(api, plan)
    cp = plan["repos"]["12b"]["cells"][0]
    prefixes = (cp["canonical_prefix"] + "/", cp["rerun_prefix"] + "/", cp["attempts_root"] + "/")
    for commit in api.repos[R12].commits:
        for op in commit["ops"]:
            assert op.path_in_repo.startswith(prefixes), op.path_in_repo
            if isinstance(op, CommitOperationCopy):
                assert op.src_path_in_repo.startswith(prefixes), op.src_path_in_repo
    repo = api.repos[R12]
    changed = {p for p in set(repo.snapshots["rev0"]) | set(repo.snapshots[repo.head])
               if repo.snapshots["rev0"].get(p) != repo.snapshots[repo.head].get(p)}
    assert changed and all(p.startswith(prefixes) for p in changed)


def test_finish_mode_resumes_after_interrupted_promote():
    """Canonical already holds an identical complete copy (earlier pass died after PROMOTE_COPY)."""
    files = {**other_cells(), **rerun_complete()}
    files.update({f"{STUDY}/{CELL}/" + rel: blob for rel, blob in rel_files(files, f"{RERUN}/{CELL}").items()})
    api = FakeHfApi({R12: FakeRepo(files), R27: FakeRepo({})})
    plan = make_plan(api)
    cp = plan["repos"]["12b"]["cells"][0]
    assert cp["mode"] == "finish" and cp["park"] is None
    assert cp["promote"]["n_to_copy"] == 0 and cp["promote"]["n_already_present"] == len(rerun_complete())
    assert [c["stage"] for c in cp["commits"] if not c.get("read_only")] == ["RECORDS", "PROMOTE_DELETE"]
    run_execute(api, plan)
    repo = api.repos[R12]
    assert [stage_of(c["message"]) for c in repo.commits] == ["RECORDS", "PROMOTE_DELETE"]
    final = repo.snapshots[repo.head]
    assert list(rel_files(final, f"{RERUN}/{CELL}")) == ["MOVED_TO.json"]
    record = json.loads(final[f"{STUDY}/{CELL}/MOVE_RECORD.json"].content)
    assert record["mode"] == "finish" and record["park_prefix"] is None and record["dst_revision"] == "rev0"


def test_fully_consolidated_cell_is_reported_not_replanned():
    api = standard_api()
    plan = make_plan(api)
    run_execute(api, plan)
    rp = make_plan(api)["repos"]["12b"]
    assert rp["cells"] == []
    assert "already consolidated" in {s["cell"]: s["refusals"][0] for s in rp["skipped"]}[CELL]


def test_render_summary_lists_cells_totals_and_skips():
    api = standard_api()
    plan = make_plan(api)
    text = consolidate.render_summary(plan)
    assert CELL in text and "sid-A3-12b-half01-interrupted-20260908" in text
    assert "PARK_COPY" in text and "PROMOTE_DELETE" in text
    assert "SKIP gemma3_12b_5m/charter/charter_0p5pct" in text
    assert "TOTAL 1 cell(s)" in text and "Nothing was written" in text


def test_cli_plan_writes_dryrun_json_and_text(tmp_path, monkeypatch, capsys):
    api = standard_api()
    monkeypatch.setattr(consolidate, "HfApi", lambda: api)
    out = tmp_path / "plan.json"
    rc = consolidate.main(["plan", "--repo", "12b", "--log-dir", str(tmp_path / "logs"), "--out", str(out)])
    assert rc == 0
    written = sorted(p.name for p in (tmp_path / "logs").iterdir() if p.is_file())
    assert len(written) == 2 and written[0].startswith("dryrun-") and written[0].endswith(".json")
    assert written[1].endswith(".txt")
    plan = json.loads(out.read_text())
    assert plan["argv"][:3] == ["plan", "--repo", "12b"]
    assert [cp["cell"] for cp in plan["repos"]["12b"]["cells"]] == [CELL]
    assert "27b" not in plan["repos"]
    assert CELL in capsys.readouterr().out
    assert not [c for c in api.calls if c[0] == "create_commit"]


def test_cli_execute_requires_confirmation(tmp_path, monkeypatch):
    api = standard_api()
    monkeypatch.setattr(consolidate, "HfApi", lambda: api)
    plan = make_plan(api)
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan))
    monkeypatch.setattr("builtins.input", lambda _prompt: "no")
    assert consolidate.main(["execute", "--plan", str(plan_path), "--log-dir", str(tmp_path)]) == 1
    assert api.repos[R12].commits == []
    assert consolidate.main(["execute", "--plan", str(plan_path), "--log-dir", str(tmp_path), "--yes"]) == 0
    assert [stage_of(c["message"]) for c in api.repos[R12].commits] == [
        "PARK_COPY", "PARK_DELETE", "PROMOTE_COPY", "RECORDS", "PROMOTE_DELETE"]
    receipts = list(tmp_path.glob("execute-*-receipt.json"))
    assert len(receipts) == 1 and json.loads(receipts[0].read_text())["repos"]["12b"]["final_revision"] == api.repos[R12].head


# --------------------------------------------------------------------------- path-scoped concurrency
def test_drift_before_start_aborts_without_commits():
    api = standard_api()
    plan = make_plan(api)
    victim = f"{RERUN}/{CELL}/train/checkpoints/checkpoint-512/adapter_model.safetensors"
    api.repos[R12].foreign_commit({victim: lfs(1_000_512, "replaced-weights")})
    with pytest.raises(consolidate.SafetyStop) as err:
        run_execute(api, plan)
    assert "cell start" in str(err.value) and "planned file changed" in str(err.value) and victim in str(err.value)
    assert api.repos[R12].commits == []


def test_deleted_source_before_promote_delete_aborts():
    api = standard_api()
    plan = make_plan(api)
    repo = api.repos[R12]

    def tamper(api_, message):
        if stage_of(message) == "PROMOTE_DELETE":
            repo.foreign_commit({f"{RERUN}/{CELL}/COMPLETE.json": None})
            api_.before_commit = None

    api.before_commit = tamper
    with pytest.raises(consolidate.SafetyStop) as err:
        run_execute(api, plan)
    assert "planned file missing" in str(err.value)
    assert [stage_of(c["message"]) for c in repo.commits] == ["PARK_COPY", "PARK_DELETE", "PROMOTE_COPY", "RECORDS"]
    final = repo.snapshots[repo.head]
    assert "COMPLETE.json" in rel_files(final, f"{STUDY}/{CELL}")  # the promoted copy is intact
    assert len(rel_files(final, f"{RERUN}/{CELL}")) == len(rerun_complete())  # -1 by the stranger, +1 MOVED_TO; none by us


@pytest.mark.parametrize("where", ["canonical", "attempts_label", "attempts_other_label", "rerun"])
def test_unplanned_file_under_our_prefixes_aborts(where):
    api = standard_api()
    plan = make_plan(api)
    cp = plan["repos"]["12b"]["cells"][0]
    rogue = {
        "canonical": f"{STUDY}/{CELL}/train/checkpoints/checkpoint-512/rogue.bin",
        "attempts_label": f"{cp['park']['prefix']}/rogue.json",
        "attempts_other_label": f"{ATTEMPTS}/{CELL}/somebody-else-label/IDENTITY.json",
        "rerun": f"{RERUN}/{CELL}/rogue.json",
    }[where]
    api.repos[R12].foreign_commit({rogue: regular(b"rogue")})
    with pytest.raises(consolidate.SafetyStop) as err:
        run_execute(api, plan)
    assert "unplanned file appeared" in str(err.value) and rogue in str(err.value)
    assert api.repos[R12].commits == []


def test_unplanned_arrival_at_target_midway_aborts_before_delete():
    api = standard_api()
    plan = make_plan(api)
    repo = api.repos[R12]

    def tamper(api_, message):
        if stage_of(message) == "PROMOTE_DELETE":
            repo.foreign_commit({f"{STUDY}/{CELL}/eval/aft-step512/late_arrival.jsonl": regular(b"{}")})
            api_.before_commit = None

    api.before_commit = tamper
    with pytest.raises(consolidate.SafetyStop) as err:
        run_execute(api, plan)
    assert "unplanned file appeared" in str(err.value)
    assert [stage_of(c["message"]) for c in repo.commits] == ["PARK_COPY", "PARK_DELETE", "PROMOTE_COPY", "RECORDS"]
    assert len(rel_files(repo.snapshots[repo.head], f"{RERUN}/{CELL}")) == len(rerun_complete()) + 1


def test_412_retry_gives_up_after_max_attempts():
    api = standard_api()
    plan = make_plan(api)
    repo = api.repos[R12]

    def always_sneak(api_, message):
        if stage_of(message) == "PARK_DELETE":
            repo.foreign_commit()

    api.before_commit = always_sneak
    with pytest.raises(consolidate.BranchMoved) as err:
        run_execute(api, plan)
    assert "failed 10 times" in str(err.value)
    attempts = [c for c in api.calls if c[0] == "create_commit" and stage_of(c[2]) == "PARK_DELETE"]
    assert len(attempts) == 10 and len({a[3] for a in attempts}) == 10  # a fresh parent each time
    assert api.sleeps == [1.0, 2.0, 4.0, 8.0, 10.0, 10.0, 10.0, 10.0, 10.0]
    assert [stage_of(c["message"]) for c in repo.commits] == ["PARK_COPY"]
    assert rel_files(repo.snapshots[repo.head], f"{STUDY}/{CELL}").keys() == rel_files(repo.snapshots["rev0"], f"{STUDY}/{CELL}").keys()


def test_network_error_after_commit_applied_is_recovered(tmp_path):
    api = standard_api()
    plan = make_plan(api)
    api.fail_after_apply["PROMOTE_COPY"] = ConnectionError("socket closed after the server applied the commit")
    run_execute(api, plan, tmp_path)
    repo = api.repos[R12]
    assert [stage_of(c["message"]) for c in repo.commits] == ["PARK_COPY", "PARK_DELETE", "PROMOTE_COPY", "RECORDS", "PROMOTE_DELETE"]
    events = [json.loads(line) for line in (tmp_path / "execute.jsonl").read_text().splitlines()]
    assert any("already shows this stage applied" in e.get("message", "") for e in events)
    assert api.sleeps == [1.0]
    final = repo.snapshots[repo.head]
    assert list(rel_files(final, f"{RERUN}/{CELL}")) == ["MOVED_TO.json"]
    record = json.loads(final[f"{STUDY}/{CELL}/MOVE_RECORD.json"].content)
    assert record["dst_revision"] == repo.commits[2]["oid"]


def test_transient_error_before_apply_is_retried_with_same_parent():
    api = standard_api()
    plan = make_plan(api)
    api.fail_before_apply["RECORDS"] = HttpError(503)
    run_execute(api, plan)
    attempts = [c for c in api.calls if c[0] == "create_commit" and stage_of(c[2]) == "RECORDS"]
    assert len(attempts) == 2 and attempts[0][3] == attempts[1][3]
    assert api.sleeps == [1.0]
    assert [stage_of(c["message"]) for c in api.repos[R12].commits] == ["PARK_COPY", "PARK_DELETE", "PROMOTE_COPY", "RECORDS", "PROMOTE_DELETE"]


def test_non_retryable_http_error_aborts_immediately():
    api = standard_api()
    plan = make_plan(api)
    api.fail_before_apply["PARK_COPY"] = HttpError(403)
    with pytest.raises(HttpError):
        run_execute(api, plan)
    assert len([c for c in api.calls if c[0] == "create_commit"]) == 1
    assert api.sleeps == [] and api.repos[R12].commits == []


def test_plan_carries_baseline_of_the_three_prefixes():
    api = standard_api()
    cp = make_plan(api)["repos"]["12b"]["cells"][0]
    ours = (f"{STUDY}/{CELL}/", f"{RERUN}/{CELL}/", f"{ATTEMPTS}/{CELL}/")
    assert set(cp["baseline"]) == {p for p in api.repos[R12].snapshots["rev0"] if p.startswith(ours)}
    reg = rerun_complete()[f"{RERUN}/{CELL}/COMPLETE.json"]
    assert cp["baseline"][f"{RERUN}/{CELL}/COMPLETE.json"] == {"size": reg.size, "blob_id": reg.blob_id}
    big = sid_partial()[f"{STUDY}/{CELL}/train/checkpoints/checkpoint-128/adapter_model.safetensors"]
    assert cp["baseline"][f"{STUDY}/{CELL}/train/checkpoints/checkpoint-128/adapter_model.safetensors"] == {
        "size": big.size, "sha256": big.lfs_sha256}


def test_plan_without_baseline_is_refused():
    api = standard_api()
    plan = make_plan(api)
    for cp in plan["repos"]["12b"]["cells"]:
        del cp["baseline"]
    with pytest.raises(consolidate.SafetyStop, match="re-run"):
        run_execute(api, plan)
    assert api.repos[R12].commits == []
