"""Devbox launcher for the 50M-corpus GLM-4.5-Air chain (one arm).

Same provisioning machinery as ``run_glm`` (capacity ladder, preflights,
orphan cleanup, clean-pushed-tree gate) with three overrides: the pod-side
entrypoint is ``chain_glm_50m.py``, the pod name/slug get a ``-50m`` suffix
(so orphan cleanup can never touch another campaign's pods), and the
timeout window is widened — the single arm is ~14 h midtrain + ~3.7 h SFT
plus data build and two consolidate/upload cycles, which does not fit the
prior 25/26 h budget sized for 2×(80M+100M)-token arms (widened again to
70/72 h on 2026-08-26 so the upload hold-loop, not a destruction timer,
bounds a pod holding an un-uploaded checkpoint). On a remote job
failure this launcher also dumps bellhop's full remote log tail before
re-raising (see ``_print_remote_failure``), and known-defective hosts are
re-rolled by IP seconds after creation instead of after a ~$2-3 venv-build
+ probe cycle (see ``_patch_bad_host_skip``).

    uv run --no-project --with 'bellhop-py>=0.8.0' --with python-dotenv \
        --with pyyaml --with huggingface-hub \
        python experiments/python4/midtraining_100b/run_glm_50m.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
for _p in (str(REPO_ROOT), str(REPO_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from experiments.python4.midtraining_100b import run_glm  # noqa: E402

POD_OVERRIDES = {
    "slug": "python4-100b-midtraining-50m",
    "name": "bellhop-python4-100b-midtraining-50m",
    # 2026-08-26 (Jonathan): 36/38 h -> 70/72 h. These timers fire bellhop/
    # RunPod teardown regardless of pod contents; with chain_glm_50m's upload
    # hold-loop a pod may legitimately sit holding an un-uploaded checkpoint
    # for many hours, and cost control is the watchers' job (pod-watch $25
    # ticks, idle alerts, UPLOAD-HOLD markers), not a destruction timer's.
    "timeout_seconds": 70 * 3600,
    "max_lifetime_seconds": 72 * 3600,
}

# Host RAM floor, mirrored from chain_glm.MIN_HOST_RAM_GB / preflight_network.sh:
# the FSDP2 rank-0 load path materializes 8×221 GB CPU buffers, so 1.5 TB
# RunPod hosts can never run this chain.
MIN_HOST_RAM_GB = 1900


def _patch_min_host_ram() -> None:
    """Inject ``minMemoryInGb`` into every RunPod create call.

    2026-08-25: the only free SECURE H200 host was a 1.5 TB machine; the
    capacity ladder rented it five times in a row (~$2 a pop) only for the
    pod-side RAM gate to exit 71 each time. RunPod's
    PodFindAndDeployOnDemandInput accepts ``minMemoryInGb`` but bellhop's
    PodConfig doesn't expose it, so wrap ``to_graphql_input`` — under-RAM
    hosts are then excluded at allocation (the rung reads "unavailable"
    instead of renting a doomed pod).
    """
    from bellhop.pod import PodConfig

    orig = PodConfig.to_graphql_input

    def with_min_ram(self, gpu_type_id=None):
        inp = orig(self, gpu_type_id)
        inp["minMemoryInGb"] = MIN_HOST_RAM_GB
        return inp

    PodConfig.to_graphql_input = with_min_ram


#: Hosts RunPod keeps re-serving whose uplink is known-defective (fails the
#: pod-side preflight upload probe). Comma-separated override / disable via
#: GLM50M_BAD_HOST_IPS (empty = no skip), read at check time.
DEFAULT_BAD_HOST_IPS = "47.47.180.89"


def _bad_host_ips() -> frozenset[str]:
    raw = os.environ.get("GLM50M_BAD_HOST_IPS", DEFAULT_BAD_HOST_IPS)
    return frozenset(ip.strip() for ip in raw.split(",") if ip.strip())


def _patch_bad_host_skip() -> None:
    """Reroll known-defective hosts by IP seconds after creation.

    2026-08-26: RunPod kept re-serving the one SECURE host 47.47.180.89,
    whose uplink fails the pod-side preflight upload probe — but reaching
    that verdict costs ~$2-3 and ~10 min per rent (venv build, then probe).
    The public IP is already known the moment ``Pod._wait_provision``
    returns (RUNNING + publicIp + ssh port mapped: seconds after create,
    before the ssh readiness probe and all setup spend), so wrap it: on a
    blocklisted IP, raise ``RemoteJobError(remote_exit=71)`` with a
    BAD-HOST marker as the log tail. The raise sits inside bellhop's
    ``pod()`` context manager, whose ``finally`` tears the pod down;
    ``run_glm``'s ladder then matches ``"BAD-HOST" in tail`` and re-rolls
    (exit 71 mirrors the pod-side bad-host convention). Blocklist env:
    ``GLM50M_BAD_HOST_IPS``, comma-separated, read per-call (like
    ``GLM50M_UPLOAD_PROBE_MIN_MBPS``); set it empty to disable.
    """
    from bellhop.errors import RemoteJobError
    from bellhop.pod import Pod

    orig = Pod._wait_provision

    async def wait_provision_screened(self):
        await orig(self)
        ip = self.host
        if ip in _bad_host_ips():
            marker = (f"BAD-HOST-IP-SKIP: {ip} pod={self.id} "
                      "— known-defective uplink, rerolling")
            print(marker, flush=True)
            raise RemoteJobError(
                f"known-defective host {ip}", remote_exit=71, log_tail=marker
            )

    Pod._wait_provision = wait_provision_screened


def apply_overrides() -> None:
    run_glm.POD.update(POD_OVERRIDES)
    run_glm.TRAIN_ENTRYPOINT = (
        "experiments/python4/midtraining_100b/pod/chain_glm_50m.py"
    )


def _patch_probe_floor_passthrough() -> None:
    """Forward GLM50M_UPLOAD_PROBE_MIN_MBPS from the launcher env to the pod.

    The knob is read pod-side (chain_glm_50m._upload_probe_min_mbps); without
    this passthrough an operator export on the devbox would silently do
    nothing. Unset means "don't inject" — the pod-side default (15) applies.
    (Lowered to 8 on 2026-08-26 by Jonathan's call: a 2TB-RAM host uploading
    at 8.5 MB/s is viable under the hold-loop + 72h timers.)"""

    orig = run_glm.pod_environment

    def with_probe_floor(credentials, result_path, git_sha, hardware):
        env = orig(credentials, result_path, git_sha, hardware)
        floor = os.environ.get("GLM50M_UPLOAD_PROBE_MIN_MBPS")
        if floor is not None:
            env["GLM50M_UPLOAD_PROBE_MIN_MBPS"] = floor
        return env

    run_glm.pod_environment = with_probe_floor


def _print_remote_failure(error: Exception) -> None:
    """Surface the remote chain's dying words before the traceback.

    2026-08-26 incident follow-up: when ``bellhop.RemoteJobError`` escaped
    the capacity ladder (a real job failure, not a re-rollable bad host),
    its ``log_tail`` was never printed — the lost-upload failure had to be
    diagnosed blind from GCS listings. Print ``remote_exit`` and the FULL
    tail, clearly delimited, then let the caller re-raise. The bellhop
    import is lazy/guarded (same pattern as ``_patch_min_host_ram``) so
    CPU-only tests without bellhop installed still pass.
    """
    try:
        from bellhop.errors import RemoteJobError
    except ImportError:
        return
    if not isinstance(error, RemoteJobError):
        return
    print(f"remote_exit={getattr(error, 'remote_exit', None)}", flush=True)
    print("==== REMOTE LOG TAIL ====", flush=True)
    print(getattr(error, "log_tail", "") or "", flush=True)
    print("==== END ====", flush=True)


def main() -> None:
    apply_overrides()
    _patch_min_host_ram()
    _patch_bad_host_skip()
    _patch_probe_floor_passthrough()
    try:
        run_glm.main()
    except Exception as error:
        _print_remote_failure(error)
        raise


if __name__ == "__main__":
    main()
