"""Run one pod's share of the dispatch_v5 fleet: its parents, in order, each in
its own ``run_parent.py`` process with ``FINAL_V1_PROFILE`` set to that parent.

``contracts`` binds the profile at import time, so parents from different
profiles cannot share a Python process; the orchestrator therefore imports
nothing profile-bound and just supervises subprocesses. A parent that fails
is recorded and the next one starts (an idle 4xH200 is the expensive
outcome overnight); rerunning the orchestrator resumes every parent behind its
own sentinels.

    python3 experiments/prior_coins/dispatch_v5/pod/run_fleet.py --pod acct1 \\
        --root /workspace/final_v1 --v5-root /workspace/final_v1_v5

State: ``<v5-root>/FLEET_STATE_<pod>.json`` (per parent: state, rc, minutes),
logs ``<v5-root>/logs/<profile>__<arm>.log``, ``FLEET_COMPLETE_<pod>.json``
when every parent completed.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from run_parent import DEFAULT_CONFIG, all_parents, load_config, split_parent  # noqa: E402


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}] {message}", flush=True)


def source_commit() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=HERE, capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:  # noqa: BLE001
        return None


def write_state(path: Path, state: dict) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(state, indent=1, sort_keys=True) + "\n")
    tmp.replace(path)


def run_pod(*, pod: str | None, parents: list[str], root: Path, v5_root: Path, config: Path,
            cfg: dict, parent_timeout_s: int, python: str = sys.executable) -> dict:
    v5_root.mkdir(parents=True, exist_ok=True)
    logs = v5_root / "logs"
    logs.mkdir(exist_ok=True)
    tag = pod or "custom"
    state_path = v5_root / f"FLEET_STATE_{tag}.json"
    state = json.loads(state_path.read_text()) if state_path.is_file() else {"parents": {}}
    state.update({"pod": tag, "parents_order": parents, "started": state.get("started") or
                  time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                  "source_commit": source_commit(), "config": str(config)})
    write_state(state_path, state)
    for parent in parents:
        profile, arm = split_parent(parent)
        entry = state["parents"].setdefault(parent, {})
        if (v5_root / profile / arm / "PARENT_COMPLETE.json").is_file():
            entry.update({"state": "COMPLETE"})
            write_state(state_path, state)
            log(f"{parent}: already complete")
            continue
        env = dict(os.environ, FINAL_V1_PROFILE=profile, FINAL_V1_MODEL_REPO=cfg["parents_repo"])
        env.setdefault("SCIMT_SOURCE_COMMIT", source_commit() or "")
        cmd = [python, str(HERE / "run_parent.py"), "--profile", profile, "--arm", arm,
               "--root", str(root), "--v5-root", str(v5_root), "--config", str(config)]
        log_path = logs / f"{profile}__{arm}.log"
        entry.update({"state": "RUNNING", "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                      "log": str(log_path), "attempts": entry.get("attempts", 0) + 1})
        write_state(state_path, state)
        log(f"{parent}: starting (attempt {entry['attempts']}) -> {log_path}")
        started = time.time()
        with log_path.open("a") as handle:
            proc = subprocess.Popen(cmd, stdout=handle, stderr=subprocess.STDOUT, env=env)
            try:
                rc = proc.wait(timeout=parent_timeout_s)
            except subprocess.TimeoutExpired:
                proc.kill()
                rc = -9
        minutes = round((time.time() - started) / 60, 1)
        complete = (v5_root / profile / arm / "PARENT_COMPLETE.json").is_file()
        entry.update({"state": "COMPLETE" if (rc == 0 and complete) else "FAILED", "rc": rc,
                      "minutes": minutes, "finished": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
        write_state(state_path, state)
        if entry["state"] == "FAILED":
            tail = "\n".join(log_path.read_text().splitlines()[-25:]) if log_path.is_file() else ""
            log(f"{parent}: FAILED rc={rc} after {minutes} min; continuing with the next parent\n{tail}")
        else:
            log(f"{parent}: COMPLETE in {minutes} min")
    done = [p for p in parents if state["parents"].get(p, {}).get("state") == "COMPLETE"]
    state["all_complete"] = len(done) == len(parents)
    state["finished"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    write_state(state_path, state)
    if state["all_complete"]:
        (v5_root / f"FLEET_COMPLETE_{tag}.json").write_text(json.dumps(
            {"pod": tag, "parents": parents, "at": state["finished"]}, indent=1) + "\n")
        log(f"FLEET COMPLETE ({tag}): {len(done)}/{len(parents)} parents")
    else:
        log(f"FLEET INCOMPLETE ({tag}): {len(done)}/{len(parents)} parents complete")
    return state


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--pod", help="key under pods: in fleet.yaml")
    group.add_argument("--parents", help="comma-separated <profile>/<arm> list (overrides --pod)")
    parser.add_argument("--root", type=Path, default=Path("/workspace/final_v1"))
    parser.add_argument("--v5-root", type=Path, default=Path("/workspace/final_v1_v5"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--parent-timeout-s", type=int, default=9 * 3600)
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    if args.pod:
        if args.pod not in cfg["pods"]:
            raise SystemExit(f"unknown pod {args.pod!r}; known: {sorted(cfg['pods'])}")
        parents = list(cfg["pods"][args.pod])
    else:
        parents = [p.strip() for p in args.parents.split(",") if p.strip()]
        known = set(all_parents(cfg))
        unknown = [p for p in parents if p not in known]
        if unknown:
            raise SystemExit(f"parents not in fleet.yaml: {unknown}")
    state = run_pod(pod=args.pod, parents=parents, root=args.root, v5_root=args.v5_root,
                    config=args.config, cfg=cfg, parent_timeout_s=args.parent_timeout_s)
    return 0 if state["all_complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
