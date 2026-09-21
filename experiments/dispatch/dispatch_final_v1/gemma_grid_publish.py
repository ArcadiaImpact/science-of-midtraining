"""Incremental HF commits, verified at immutable revision before a receipt."""
import hashlib
import json
import time
from pathlib import Path
from experiments.dispatch.dispatch_final_v1.gemma_grid_plan import sha, write


def file_record(path):
    path = Path(path)
    h = hashlib.sha1(f"blob {path.stat().st_size}\0".encode())
    with path.open("rb") as f:
        for b in iter(lambda: f.read(8*1024*1024), b""):
            h.update(b)
    return dict(size=path.stat().st_size, sha256=sha(path), git_blob=h.hexdigest())


def verify(api, repo, prefix, commit, files):
    entries = {e.path: e for e in api.list_repo_tree(repo, revision=commit,
               path_in_repo=prefix, recursive=True) if getattr(e, "size", None) is not None}
    for name, expected in files.items():
        e = entries.get(f"{prefix}/{name}")
        if e is None or e.size != expected["size"]:
            raise RuntimeError(f"Remote file missing/wrong size: {prefix}/{name}")
        lfs = getattr(e, "lfs", None)
        digest = (lfs.get("sha256") if isinstance(lfs, dict) else getattr(lfs,"sha256",None)) if lfs else None
        if (digest != expected["sha256"] if lfs else e.blob_id != expected["git_blob"]):
            raise RuntimeError(f"Remote checksum mismatch: {prefix}/{name}")


class Publisher:
    def __init__(self, repo, prefix, receipts, api=None):
        from huggingface_hub import HfApi
        self.api = api or HfApi()
        self.repo, self.prefix, self.receipts = repo, prefix, Path(receipts)
        self.api.repo_info(repo)  # Must already exist; never create a public repo implicitly.

    def publish(self, root, paths, label):
        from huggingface_hub import CommitOperationAdd
        root = Path(root)
        files = {str(p.relative_to(root)): file_record(p) for p in sorted(paths)}
        if not files:
            raise ValueError("Cannot publish an empty artifact set")
        receipt = self.receipts/f"{label}.json"
        if receipt.exists():
            old = json.loads(receipt.read_text())
            if old["files"] == files and old["repo"] == self.repo and old["prefix"] == self.prefix:
                verify(self.api, self.repo, self.prefix, old["commit"], files)
                return old
        operations = [CommitOperationAdd(path_in_repo=f"{self.prefix}/{name}", path_or_fileobj=str(root/name))
                      for name in files]
        for attempt in range(6):
            try:
                commit = self.api.create_commit(repo_id=self.repo, operations=operations,
                    commit_message=f"{self.prefix}: {label}").oid
                verify(self.api, self.repo, self.prefix, commit, files)
                result = dict(repo=self.repo, prefix=self.prefix, commit=commit, files=files)
                write(receipt, result)
                return result
            except Exception:
                if attempt == 5:
                    raise
                time.sleep(min(10*2**attempt, 120))

    def verify_receipts(self):
        paths = list(self.receipts.glob("*.json"))
        if not paths:
            raise RuntimeError("No persistence receipts")
        for path in paths:
            r = json.loads(path.read_text())
            if r["repo"] != self.repo or r["prefix"] != self.prefix:
                raise RuntimeError("Receipt destination mismatch")
            verify(self.api, self.repo, self.prefix, r["commit"], r["files"])
