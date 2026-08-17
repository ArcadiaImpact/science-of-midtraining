"""Exercise checkpoint_upload against the REAL Hub before the pods rely on it.

The unit tests use a fake API, so preupload_lfs_files + create_commit + the
remote verify have never actually run. The pods are already training against
this code; if it is broken they fail at the upload phase and take ~2 h of
8xH200 time with them. This uploads two tiny fake checkpoints (one boundary,
one intermediate) into a scratch prefix of the real models repo, checks the
receipts and the duplicate-omission behaviour, then deletes the prefix.
"""

import json
import os
import shutil
import sys
from pathlib import Path

REPO = Path("/workspace/scimt-prior-coins-27b")
sys.path.insert(0, str(REPO))
os.environ.setdefault("SCIMT_RUN_ID", "20260101T000000Z")

from experiments.prior_coins.dispatch_scaleup import (  # noqa: E402
    checkpoint_upload,
    contracts,
)

SCRATCH = "_smoketest_upload"
WORK = Path(os.environ.get("SCIMT_SMOKE_WORK", "/tmp/scimt-smoke-ckpts"))


def _api():
    from huggingface_hub import HfApi

    return HfApi()


def fake_checkpoint(root: Path, step: int) -> Path:
    d = root / f"checkpoint-{step}"
    d.mkdir(parents=True, exist_ok=True)
    payload = (f"weights for step {step} " * 4096).encode()
    (d / "model.safetensors").write_bytes(payload)
    (d / "pytorch_model_fsdp.bin").write_bytes(payload)  # the duplicate
    (d / "optimizer.bin").write_bytes(b"adam " * 4096)
    (d / "scheduler.pt").write_bytes(b"cosine")
    (d / "rng_state_0.pth").write_bytes(b"rng")
    (d / "config.json").write_text(json.dumps({"step": step}))
    (d / "trainer_state.json").write_text(json.dumps({"global_step": step}))
    return d


def main() -> None:
    spec = contracts.size("27b")
    api = _api()
    if WORK.exists():
        shutil.rmtree(WORK)
    jobs = [
        checkpoint_upload.CheckpointJob(
            label=str(step),
            step=step,
            local_dir=fake_checkpoint(WORK, step),
            remote_prefix=f"{SCRATCH}/checkpoint-{step}",
            manifest_path=WORK / f"manifest-{step}.json",
            commit_message=f"smoke test checkpoint {step}",
        )
        # 4 is a boundary (keeps the duplicate), 62 is an intermediate (drops it)
        for step in (4, 62)
    ]
    receipts = checkpoint_upload.upload_checkpoints(
        api,
        repo_id=spec.models_repo,
        jobs=jobs,
        keep_steps=contracts.MIDTRAIN_DUPLICATE_WEIGHT_STEPS,
    )
    print("RECEIPTS:")
    for label, receipt in receipts.items():
        print(f"  step {label}: files={receipt['files']} "
              f"omitted={receipt['omitted']} oid={receipt['commit_oid'][:12]}")

    remote = api.list_repo_files(spec.models_repo, repo_type="model")
    under = sorted(p for p in remote if p.startswith(f"{SCRATCH}/"))
    print("REMOTE FILES:")
    for path in under:
        print("  ", path)
    dup = checkpoint_upload.DUPLICATE_WEIGHTS
    assert f"{SCRATCH}/checkpoint-4/{dup}" in under, "boundary must keep duplicate"
    assert f"{SCRATCH}/checkpoint-62/{dup}" not in under, "intermediate must drop it"
    assert f"{SCRATCH}/checkpoint-62/model.safetensors" in under
    assert f"{SCRATCH}/checkpoint-62/optimizer.bin" in under, "resumability kept"
    assert receipts["4"]["omitted"] == []
    assert receipts["62"]["omitted"] == [dup]
    print("ASSERTIONS PASSED")

    api.delete_folder(path_in_repo=SCRATCH, repo_id=spec.models_repo,
                      repo_type="model", commit_message="remove smoke test")
    left = [p for p in api.list_repo_files(spec.models_repo, repo_type="model")
            if p.startswith(f"{SCRATCH}/")]
    print("cleanup left behind:", left)
    shutil.rmtree(WORK)
    print("SMOKE TEST OK")


if __name__ == "__main__":
    main()
