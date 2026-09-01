"""Deploy the sweep pod with a SERVER-SIDE dead-man switch.

Why this exists rather than reusing the skill's ``_deploy_cuda.py``: that helper
builds the same ``podFindAndDeployOnDemand`` mutation but does not pass
``terminateAfter``, and it is shared with the live campaign, so it is not
something to edit for one errand. Probing the API showed ``terminateAfter`` and
``stopAfter`` ARE valid fields on ``PodFindAndDeployOnDemandInput`` (they pass
validation and reach the supply check), while there is no way to set or even
read them on an already-created pod: ``podEditJob`` rejects the field, there is
no ``podUpdate``/``podEdit`` mutation, and the ``Pod`` type exposes neither.

That asymmetry is the whole point. The skill's ``--max-hours`` is a *client-side*
dead-man -- a local process that must stay alive to do the killing -- and this
sweep may deploy unattended hours from now when capacity frees. A server-side
terminateAfter set at creation is the only protection that survives losing the
supervising process, so it has to go in the create call or not at all.

Field set is copied from the skill's helper deliberately, including
``ports``/``startSsh``: GraphQL deploys that omit those have shown sshd not
listening despite the pod reporting RUNNING (the "dead sshd" bug in SKILL.md).

Exit codes: 0 deployed (pod id on stdout), 75 supply-constrained (retryable),
1 anything else (do not retry).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from datetime import UTC, datetime, timedelta

SUPPLY_EXIT = 75


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--gpu", required=True)
    ap.add_argument("--cuda", required=True, help="comma-separated CUDA versions")
    ap.add_argument("--cloud", default="SECURE", choices=("SECURE", "COMMUNITY"))
    ap.add_argument("--template", default="runpod-torch-v280")
    ap.add_argument("--count", type=int, required=True)
    ap.add_argument("--disk", type=int, required=True)
    ap.add_argument("--terminate-after-hours", type=float, required=True,
                    help="server-side auto-terminate, hours from deploy time")
    args = ap.parse_args()

    key = os.environ.get("RUNPOD_API_KEY", "").strip()
    if not key:
        sys.exit("No $RUNPOD_API_KEY (use ops/with_account3.sh)")

    esc = json.dumps
    cuda_arr = "[" + ", ".join(esc(c.strip()) for c in args.cuda.split(",")) + "]"
    # Anchored to now, not to deploy time: a queued/retried deploy that lands
    # later gets a correspondingly shorter leash, which is the safe direction.
    deadline = (datetime.now(UTC) + timedelta(hours=args.terminate_after_hours)
                ).strftime("%Y-%m-%dT%H:%M:%SZ")
    query = (
        "mutation { podFindAndDeployOnDemand(input: { "
        f"cloudType: {args.cloud}, gpuCount: {args.count}, "
        f"gpuTypeId: {esc(args.gpu)}, templateId: {esc(args.template)}, "
        f"containerDiskInGb: {args.disk}, volumeInGb: 0, "
        'ports: "22/tcp,8888/http", startSsh: true, '
        f"name: {esc(args.name)}, allowedCudaVersions: {cuda_arr}, "
        f"terminateAfter: {esc(deadline)} "
        "}) { id machineId } }"
    )
    req = urllib.request.Request(
        "https://api.runpod.io/graphql",
        data=json.dumps({"query": query}).encode(),
        headers={"Authorization": f"Bearer {key}",
                 "Content-Type": "application/json",
                 "User-Agent": "glm-mfu-deploy/1.0",
                 "Accept": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            body = json.loads(resp.read().decode("utf-8", "replace"))
    except urllib.error.HTTPError as exc:
        try:
            body = json.loads(exc.read().decode("utf-8", "replace"))
        except Exception:
            print(f"HTTP {exc.code} from deploy", file=sys.stderr)
            return 1
    except (urllib.error.URLError, OSError) as exc:
        print(f"network error on deploy: {exc}", file=sys.stderr)
        return SUPPLY_EXIT  # transient: worth another attempt

    pod = ((body.get("data") or {}).get("podFindAndDeployOnDemand") or {})
    if pod.get("id"):
        print(pod["id"])
        print(f"terminateAfter={deadline} (server-side)", file=sys.stderr)
        return 0

    text = json.dumps(body)
    if "SUPPLY_CONSTRAINT" in text or "no longer any instances" in text:
        print("supply-constrained", file=sys.stderr)
        return SUPPLY_EXIT
    print(f"deploy failed: {text[:400]}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
