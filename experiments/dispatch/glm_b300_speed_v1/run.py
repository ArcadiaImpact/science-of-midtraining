"""Run the prepared cells on an existing, verified eight-B300 pod.

No cloud API calls or pod lifecycle operations. Each attempt has a fresh
directory; completed measurements are never overwritten or retrained.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import yaml

from . import bench as B


def terminate_group(process):
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=10)


def command(cell, config):
    return [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nnodes=1",
        f"--nproc-per-node={len(cell.gpus)}",
        "--max-restarts=0",
        "-m",
        "experiments.dispatch.glm_b300_speed_v1.train_entry",
        str(config),
    ]


class Runner:
    def __init__(self, args):
        self.args = args
        self.root = args.out.resolve()
        if self.root.exists():
            raise ValueError(
                "Use a NEW output directory; existing evidence is never overwritten"
            )
        self.root.mkdir(parents=True)
        # Includes elapsed provision/setup time, not merely this process's age.
        self.deadline = args.pod_created_unix + args.max_pod_minutes * 60
        self.records = []
        self.sources = json.loads((args.data / "PREPARED.json").read_text())["sources"]
        self.save()

    def remaining(self):
        return self.deadline - time.time() - 90  # reserve time for final collection

    def save(self):
        B.write_json(
            self.root / "results.json",
            {
                "schema": "glm_b300_speed_v1",
                "pod_created_unix": self.args.pod_created_unix,
                "pod_hourly_usd": self.args.pod_hourly_usd,
                "deadline_unix": self.deadline,
                "elapsed_pod_hours": max(0, time.time() - self.args.pod_created_unix)
                / 3600,
                "cells": self.records,
                "sources": self.sources,
                "substrate_note": "All stages start independently from pinned BASE weights; Dolci/AFT are substrate proxies",
            },
        )
        B.relative_to_baseline(self.records)
        lines = [
            "# B300 speed measurements",
            "",
            "All stages independently start from base weights.",
            "Full-parameter token rates are packed positions; AFT uses actual updates/examples.",
            "'x m2/a2' is against the production-geometry midtrain cell on THIS pod;",
            "'x H200' is against the historical H200 anchor (different host/driver/CUDA).",
            "'charter h' projects one 1B-row charter arm's stage from the measured rate.",
            "",
            "| Cell | Status | s/update | positions/s | x m2/a2 | x H200 | charter h | peak GiB | Note |",
            "|---|---|---:|---:|---:|---:|---:|---:|---|",
        ]
        for r in self.records:
            lines.append(
                f"| {r['cell']['name']} | {r['status']} | "
                f"{r.get('median_seconds', 0):.2f} | {r.get('positions_per_second', 0):,.0f} | "
                f"{r.get('speedup_vs_baseline_cell', 0):.3f} | "
                f"{r.get('speedup_vs_h200', 0):.2f} | "
                f"{r.get('charter_arm_stage_hours', 0):.1f} | "
                f"{r.get('peak_reserved_gib', 0):.0f} | {r.get('note', '')} |"
            )
        report = self.root / "RESULTS.md"
        temporary = report.with_suffix(".md.tmp")
        temporary.write_text("\n".join(lines) + "\n")
        temporary.replace(report)

    def skip(self, cell, reason):
        r = {"cell": dataclasses.asdict(cell), "status": "skipped", "note": reason}
        self.records.append(r)
        self.save()
        return r

    def data_for(self, cell, synthetic=False):
        key = (
            ("midtrain_synthetic" if synthetic else "midtrain")
            if cell.stage == "midtrain"
            else ("dolci" if cell.stage == "dolci" else cell.name.split("_retry")[0])
        )
        if (
            cell.stage == "midtrain"
            and not synthetic
            and "midtrain_mix" in self.sources
        ):
            key = "midtrain_mix"
        source = self.sources.get(key)
        if source is None:
            return None, None
        path = (self.args.data / source["file_local"]).resolve()
        if B.sha256(path) != source["sha256"]:
            raise ValueError(f"Input hash mismatch: {key}")
        return path, source

    def group(self, cells, synthetic=False):
        jobs, answers = [], []
        try:
            for cell in cells:
                # Avoid spending the last minutes loading 221 GB and obtaining
                # no stable timings. Dolci full geometry needs a longer window.
                minimum = (
                    18 * 60 if cell.stage == "dolci" and not cell.proxy else 10 * 60
                )
                if self.remaining() < minimum:
                    answers.append(self.skip(cell, "insufficient time remaining"))
                    continue
                try:
                    path, source = self.data_for(cell, synthetic)
                except (OSError, ValueError) as exc:
                    result = {
                        "cell": dataclasses.asdict(cell),
                        "status": "invalid",
                        "failure_kind": "data_failure",
                        "note": str(exc),
                    }
                    self.records.append(result)
                    self.save()
                    answers.append(result)
                    continue
                if path is None:
                    answers.append(self.skip(cell, "prepared data unavailable"))
                    continue
                out = self.root / cell.name
                out.mkdir()  # fail on accidental reuse
                cfg = B.render(cell, self.args.model, path, out)
                cfgpath = out / "config.yaml"
                cfgpath.write_text(yaml.safe_dump(cfg, sort_keys=False))
                env = os.environ.copy()
                env.update(
                    CUDA_VISIBLE_DEVICES=",".join(map(str, cell.gpus)),
                    PYTHONPATH=os.pathsep.join([str(B.REPO), str(B.REPO / "src")]),
                    ACCELERATE_USE_FSDP="true",
                    FSDP_CPU_RAM_EFFICIENT_LOADING="true",
                    NCCL_NVLS_ENABLE="0",
                    TOKENIZERS_PARALLELISM="false",
                    WANDB_MODE="disabled",
                    HF_HUB_DISABLE_TELEMETRY="1",
                )
                log = (out / "train.log").open("w")
                started = time.time()
                proc = subprocess.Popen(
                    command(cell, cfgpath),
                    cwd=B.REPO,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                jobs.append((cell, out, source, proc, log, started))
                B.write_json(
                    out / "attempt.json",
                    {
                        "pid": proc.pid,
                        "started_unix": started,
                        "config_sha256": B.sha256(cfgpath),
                        "source": source,
                        "cell": dataclasses.asdict(cell),
                    },
                )
                print(
                    f"Started {cell.name}; {cell.positions_per_step or cell.examples_per_step} "
                    f"{'positions' if cell.positions_per_step else 'examples'}/update",
                    flush=True,
                )
            for cell, out, source, proc, log, started in jobs:
                timeout = (
                    min(started + cell.max_minutes * 60, self.deadline - 90)
                    - time.time()
                )
                timed_out = False
                try:
                    if timeout <= 0:
                        raise subprocess.TimeoutExpired(proc.args, 0)
                    proc.wait(timeout=timeout)
                except subprocess.TimeoutExpired:
                    timed_out = True
                    terminate_group(proc)
                log.close()
                r = B.summarize(
                    cell, out / "telemetry", proc.returncode, self.args.pod_hourly_usd
                )
                r.update(
                    wall_seconds=time.time() - started,
                    source=source,
                    substrate="base-weight proxy"
                    if cell.stage != "midtrain"
                    else "pinned base",
                    note="synthetic input proxy"
                    if synthetic
                    else (
                        "short-accumulation proxy"
                        if cell.proxy
                        else source.get("note", "")
                    ),
                )
                if timed_out:
                    r.update(status="invalid", failure_kind="timeout")
                elif r["status"] != "valid":
                    r["failure_kind"] = B.failure_kind(
                        (out / "train.log").read_text(errors="replace")
                    )
                B.write_json(out / "result.json", r)
                self.records.append(r)
                self.save()  # commit each outcome BEFORE deciding whether to continue
                answers.append(r)
                print(
                    f"Finished {cell.name}: {r['status']} {r.get('failure_kind', '')}",
                    flush=True,
                )
        finally:
            for _, _, _, proc, log, _ in jobs:
                terminate_group(proc)
                log.close()
        return answers

    def run(self):
        synthetic = not any(key in self.sources for key in ("midtrain_mix", "midtrain"))
        if self.args.cells:
            # Explicit probe: run exactly these cells, in order, no fallbacks.
            for name in self.args.cells:
                self.group([B.CELLS_BY_NAME[name]], synthetic)
            return
        mid = self.group([B.MID], synthetic)[0]
        if mid.get("failure_kind") == "gpu_oom":
            mid = self.group([B.MID_SMALL], synthetic)[0]
        elif mid.get("failure_kind") == "data_failure" and not synthetic:
            mid = self.group(
                [dataclasses.replace(B.MID, name="midtrain_synthetic")], True
            )[0]
            synthetic = True
        if mid["status"] != "valid":
            self.skip(
                B.DOLCI,
                "no valid midtrain baseline; repair the identified failure first",
            )
            self.skip(B.AFT_A, "no valid midtrain baseline")
            return
        # Midtrain speed-up candidates BEFORE the Dolci/AFT proxies: midtrain
        # is where the arm's hours are, and the old order (proxies first, m4
        # last-if-time) is exactly what the 110-minute budget starved.
        if not self.args.no_variants:
            large = self.group([B.MID_LARGE], synthetic)[0]
            self.group([B.MID_NOMON], synthetic)
            # The FSDP-native checkpoint cell rides the largest geometry that
            # fit: m4 if it did, else the production m2/a2 (an m4 OOM is a
            # memory verdict on B200 in its own right; do not repeat it).
            self.group(
                [B.MID_LARGE_AC if large["status"] == "valid" else B.MID_AC],
                synthetic,
            )
        if not self.args.midtrain_only:
            dolci = self.group([B.DOLCI])[0]
            if dolci.get("failure_kind") == "gpu_oom":
                self.group([B.DOLCI_SMALL])
            elif dolci.get("failure_kind") == "timeout" or (
                dolci["status"] == "skipped" and "time" in dolci.get("note", "")
            ):
                self.group([B.DOLCI_PROXY])
            aft = self.group([B.AFT_A, B.AFT_B])
            # An OOM under concurrent loads can be host RAM pressure. Retry
            # just one cell; label it as a single-cell, not paired-wave, test.
            if any(r.get("failure_kind") == "gpu_oom" for r in aft) and not any(
                r["status"] == "valid" for r in aft
            ):
                self.group([dataclasses.replace(B.AFT_A, name="aft_agreement_retry")])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--pod-created-unix", type=float, required=True)
    ap.add_argument("--pod-hourly-usd", type=float, required=True)
    ap.add_argument("--max-pod-minutes", type=float, default=110)
    ap.add_argument(
        "--midtrain-only",
        action="store_true",
        help="skip the Dolci and AFT base-weight proxies",
    )
    ap.add_argument(
        "--no-variants",
        action="store_true",
        help="skip the m4/a1, no-monitor and FSDP-checkpoint midtrain cells",
    )
    ap.add_argument(
        "--cells",
        nargs="+",
        choices=sorted(B.CELLS_BY_NAME),
        help="run exactly these cells in order (a targeted probe), instead of the matrix",
    )
    args = ap.parse_args()
    # 110 is the planned budget; up to 150 is allowed so that time lost to pod
    # bring-up outside the probe's control (the first B200 pod needed a
    # container restart to accept our ssh key, ~10 minutes) can be given back
    # to the cells rather than silently starving the last ones. Record the
    # value used in the results; it is not a reason to extend a healthy run.
    if not 0 < args.pod_hourly_usd <= 65 or not 0 < args.max_pod_minutes <= 150:
        ap.error("prepared budget is <=$65/hour and <=150 minutes from pod creation")
    if not 0 < args.pod_created_unix <= time.time():
        ap.error("pod creation time must be a real past Unix timestamp")
    from .preflight import validate_host

    receipt = validate_host(args.out.parent, require_empty=True, model=args.model)
    runner = Runner(args)
    B.write_json(runner.root / "preflight.json", receipt)
    try:
        runner.run()
    finally:
        runner.save()
        files = {
            str(p.relative_to(runner.root)): B.sha256(p)
            for p in runner.root.rglob("*")
            if p.is_file() and "/prepared/" not in str(p) and "/trainer/" not in str(p)
        }
        B.write_json(runner.root / "SHA256SUMS.json", files)
    valid_mid = any(
        r["cell"]["stage"] == "midtrain" and r["status"] == "valid"
        for r in runner.records
    )
    print(
        "Measurements preserved. GPU billing continues until the pod is cleaned up.",
        flush=True,
    )
    raise SystemExit(0 if valid_mid else 1)


if __name__ == "__main__":
    main()
