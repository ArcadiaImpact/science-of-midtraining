"""Run the vendored MoralSim harness against a local vLLM server (v1).

MoralSim (vendored at the pinned commit by ``fetch_external_repos.sh``,
transport patch applied there) runs verbatim: hydra experiment configs, their
scenario code, their parsing. The only differences from the paper runs are
the endpoint (local vLLM instead of OpenRouter), ``debug=true`` (no wandb),
and greedy decoding (``llm.temperature=0.0``, house convention).

v1 experiment grid: {prisoner's dilemma, public goods} x {production,
privacy, venture} moral contexts x dummy opponents {cooperate, defect},
CoT variants, ``--seeds`` each. Runs are sequential (each episode is a
sequential chain anyway); a run whose subprocess fails is recorded and does
not stop the sweep.

Usage (on the pod, server up):
    python run_moralsim_v1.py --base-url http://127.0.0.1:8100/v1 \
        --served-name post --model-key control_4x__agreement512 --seeds 0 1
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

EXP = Path(__file__).resolve().parent
MORALSIM = EXP / "vendor" / "moralsim"

DEFAULT_EXPERIMENTS = [
    "pd_production_dummy_defect_cot",
    "pd_production_dummy_cooperate_cot",
    "pd_privacy_dummy_defect_cot",
    "pd_privacy_dummy_cooperate_cot",
    "pd_venture_dummy_defect_cot",
    "pd_venture_dummy_cooperate_cot",
    "pg_production_dummy_defect_cot",
    "pg_production_dummy_cooperate_cot",
    "pg_privacy_dummy_defect_cot",
    "pg_privacy_dummy_cooperate_cot",
    "pg_venture_dummy_defect_cot",
    "pg_venture_dummy_cooperate_cot",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8100/v1")
    ap.add_argument("--served-name", required=True)
    ap.add_argument("--model-key", required=True)
    ap.add_argument("--experiments", nargs="+", default=DEFAULT_EXPERIMENTS)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--out", default=str(EXP / "runs" / "external_values_v1" /
                                         "moralsim"))
    ap.add_argument("--timeout", type=int, default=3600,
                    help="per-run wall clock cap, seconds")
    args = ap.parse_args()

    if not MORALSIM.is_dir():
        raise SystemExit("vendor/moralsim missing — run fetch_external_repos.sh")
    marker = (MORALSIM / "pathfinder" / "pathfinder" / "api.py").read_text()
    if "scimt external_values_v1 transport patch" not in marker:
        raise SystemExit("moralsim transport patch not applied — rerun "
                         "fetch_external_repos.sh")

    out_root = Path(args.out) / args.model_key
    out_root.mkdir(parents=True, exist_ok=True)
    result_dir = f"results_ev1_{args.model_key}"

    env = dict(os.environ)
    env["OPENROUTER_BASE_URL"] = args.base_url
    env.setdefault("OPENROUTER_API_KEY", "dummy")
    env.setdefault("WANDB_MODE", "disabled")
    # a greedy 12B rambles to whatever budget it gets; 2048 keeps actions
    # parseable and episodes minutes-long (upstream default is 8000)
    env.setdefault("MORALSIM_MAX_TOKENS", "2048")

    ledger = []
    for experiment in args.experiments:
        for seed in args.seeds:
            run_key = f"{experiment}_seed{seed}"
            dest = out_root / run_key
            if dest.exists():
                print(f"[skip] {run_key} already collected")
                ledger.append({"run": run_key, "status": "cached"})
                continue
            cmd = [
                sys.executable, "-m", "moralsim.main",
                f"experiment={experiment}",
                f"llm.path={args.served_name}",
                "llm.is_api=true",
                "llm.backend=openrouter",
                "llm.temperature=0.0",
                f"seed={seed}",
                "debug=true",
                f"result_dir={result_dir}",
            ]
            print(f"[run] {run_key}: {' '.join(cmd)}", flush=True)
            t0 = time.time()
            proc = subprocess.run(
                cmd, cwd=MORALSIM, env=env,
                capture_output=True, text=True, timeout=args.timeout)
            (out_root / f"{run_key}.log").write_text(
                proc.stdout[-20000:] + "\n--- STDERR ---\n" + proc.stderr[-20000:])
            status = "ok" if proc.returncode == 0 else f"exit {proc.returncode}"
            # collect the hydra/experiment storage for this experiment name
            produced = MORALSIM / result_dir / experiment
            if proc.returncode == 0 and produced.is_dir():
                runs = sorted(produced.iterdir(), key=lambda p: p.stat().st_mtime)
                if runs:
                    shutil.copytree(runs[-1], dest)
            elif proc.returncode != 0:
                print(f"[FAIL] {run_key}: {status}; tail:\n"
                      + proc.stderr[-1500:], flush=True)
            ledger.append({"run": run_key, "status": status,
                           "seconds": round(time.time() - t0, 1)})
            (out_root / "ledger.json").write_text(
                json.dumps(ledger, indent=2) + "\n")
    n_ok = sum(1 for r in ledger if r["status"] in ("ok", "cached"))
    print(f"moralsim: {n_ok}/{len(ledger)} runs ok -> {out_root}")


if __name__ == "__main__":
    main()
