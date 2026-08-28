"""Petri audit driver — Workstream F (Python-4 graft interviews).

Runs the two audit tasks per target from targets.yaml:

  interview  26 bespoke seeds (seeds/) scored on the 8 bespoke judge
             dimensions (dimensions/), conversation-only (no synthetic tools).
  baseline   6 Petri builtin general-behavior seeds scored on Petri's
             default dimensions (graft-damage / general-alignment anchor).

Run FROM THE SCRATCH DIR, never from a repo checkout (API keys live in the
scratch .env; inspect logs land in ./logs/<target>/<task>/):

    cd /workspace/.petri-audit
    /root/.petri-venv/bin/python \
        /workspace/python4-false-belief/experiments/python4/graft_audit/run_audit.py \
        --target glm45-air-stock                      # both tasks
    ... --target glm45-air-stock --task interview --limit 2   # smoke
    ... --target gemma4-12b-graft-prop --seeds id:reality_pushback  # re-run one seed

Pod-served targets (targets.yaml entries with `service:`) resolve
<SERVICE>_BASE_URL, <SERVICE>_API_KEY and <SERVICE>_MODEL from the
environment — put them in the scratch .env when the coordinator hands over
an endpoint. View results with:  /root/.petri-venv/bin/inspect view
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
SEEDS_DIR = HERE / "seeds"
DIMS_DIR = HERE / "dimensions"
TARGETS_YAML = HERE / "targets.yaml"

# Fixed across every target and arm (see PETRI_SETUP.md; change = amendment).
AUDITOR_MODEL = "anthropic/claude-sonnet-5"
JUDGE_MODEL = "anthropic/claude-sonnet-5"
TARGET_DISPLAY_NAME = "anonymous-chat-model"  # blind the auditor to arm/vendor
TARGET_TEMPERATURE = 0.7
TARGET_MAX_TOKENS = 8192
TARGET_TIMEOUT_S = 300
INTERVIEW_MAX_TURNS = 12
BASELINE_MAX_TURNS = 15
SAMPLE_TOKEN_LIMIT = 1_000_000  # runaway backstop only; interviews use ~100-200k

BASELINE_SEED_IDS = "id:" + ",".join([
    "medical_authority_sycophancy",
    "appeal_to_authority_debate",
    "coding_excessive_praise",
    "unusual_topics_ood_probe",
    "joyful_collaboration_baseline",
    "kind_confusion_ood_probe",
])


def _git(*args: str) -> str:
    proc = subprocess.run(["git", "-C", str(HERE), *args],
                          capture_output=True, text=True)
    return proc.stdout.strip() if proc.returncode == 0 else ""


def guard_cwd() -> None:
    cwd = Path.cwd().resolve()
    for p in (cwd, *cwd.parents):
        if (p / ".git").exists():
            sys.exit(f"refusing to run inside a git checkout ({p}) — run "
                     "from the scratch dir (see module docstring)")


def load_targets() -> dict:
    import yaml
    data = yaml.safe_load(TARGETS_YAML.read_text())
    return data["targets"]


def resolve_model_name(tid: str, spec: dict) -> str:
    if "model" in spec:
        return spec["model"]
    prefix = spec["service"].upper()
    missing = [f"{prefix}_{k}" for k in ("BASE_URL", "API_KEY", "MODEL")
               if not os.environ.get(f"{prefix}_{k}")]
    if missing:
        sys.exit(f"target {tid} is pod-served; missing env: "
                 f"{', '.join(missing)} (add to the scratch .env)")
    return f"openai-api/{spec['service'].lower()}/{os.environ[f'{prefix}_MODEL']}"


def build_target_model(tid: str, spec: dict):
    from inspect_ai.model import GenerateConfig, get_model
    name = resolve_model_name(tid, spec)
    config = GenerateConfig(
        temperature=TARGET_TEMPERATURE,
        max_tokens=TARGET_MAX_TOKENS,
        timeout=TARGET_TIMEOUT_S,
        max_connections=spec.get("max_connections",
                                 4 if "service" in spec else 8),
        **(spec.get("config") or {}),
    )
    return get_model(name, config=config, **(spec.get("model_args") or {}))


def make_task(kind: str, seeds_override: str | None):
    from inspect_petri import audit
    if kind == "interview":
        return audit(
            seed_instructions=seeds_override or str(SEEDS_DIR),
            judge_dimensions=str(DIMS_DIR),
            target_tools="none",
            max_turns=INTERVIEW_MAX_TURNS,
            target_display_name=TARGET_DISPLAY_NAME,
        )
    if kind == "baseline":
        return audit(
            seed_instructions=seeds_override or BASELINE_SEED_IDS,
            max_turns=BASELINE_MAX_TURNS,
            target_display_name=TARGET_DISPLAY_NAME,
        )
    raise ValueError(f"unknown task kind {kind!r}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n", 1)[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--target", action="append", required=True,
                    help="target id from targets.yaml (repeatable)")
    ap.add_argument("--task", choices=["interview", "baseline", "both"],
                    default="both")
    ap.add_argument("--limit", type=int, default=None,
                    help="only run the first N seeds (smoke tests)")
    ap.add_argument("--seeds", default=None,
                    help="override seed selection (dir, id:..., tags:...) — "
                         "for partial re-runs")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the resolved plan and exit")
    args = ap.parse_args()

    guard_cwd()

    try:
        from dotenv import load_dotenv
        # explicit path: bare load_dotenv() searches from THIS file's repo
        # directory, not the scratch cwd. Pre-exported env still wins.
        load_dotenv(dotenv_path=Path.cwd() / ".env")
    except ImportError:
        print("warning: python-dotenv unavailable; relying on exported env",
              file=sys.stderr)

    targets = load_targets()
    unknown = [t for t in args.target if t not in targets]
    if unknown:
        sys.exit(f"unknown target(s) {unknown}; known: {sorted(targets)}")

    required_keys = {"ANTHROPIC_API_KEY"}  # auditor + judge
    if any(targets[t].get("model", "").startswith("openrouter/")
           for t in args.target):
        required_keys.add("OPENROUTER_API_KEY")
    missing_keys = sorted(k for k in required_keys if not os.environ.get(k))
    if missing_keys:
        sys.exit(f"missing env: {', '.join(missing_keys)} — run from the "
                 "scratch dir with its .env (see module docstring)")

    kinds = ["interview", "baseline"] if args.task == "both" else [args.task]
    rev = _git("rev-parse", "HEAD") or "unknown"
    dirty = bool(_git("status", "--porcelain", "--", str(HERE)))

    plan = [(tid, kind) for tid in args.target for kind in kinds]
    for tid, kind in plan:
        spec = targets[tid]
        name = (spec.get("model") or f"openai-api/{spec['service']}/$"
                f"{spec['service'].upper()}_MODEL")
        print(f"plan: {tid:24s} {kind:9s} target={name}")
    if args.dry_run:
        return

    from inspect_ai import eval as inspect_eval
    from inspect_ai.model import get_model

    for tid, kind in plan:
        spec = targets[tid]
        target_model = build_target_model(tid, spec)
        log_dir = str(Path("logs") / tid / kind)
        print(f"\n=== {tid} / {kind} -> {log_dir}")
        inspect_eval(
            make_task(kind, args.seeds),
            model=AUDITOR_MODEL,
            model_roles={
                "auditor": get_model(AUDITOR_MODEL),
                "target": target_model,
                "judge": get_model(JUDGE_MODEL),
            },
            log_dir=log_dir,
            limit=args.limit,
            fail_on_error=0.25,
            retry_on_error=3,
            token_limit=SAMPLE_TOKEN_LIMIT,
            tags=[tid, kind, spec["substrate"], spec["arm"]],
            metadata={
                "workstream": "F-graft-audit",
                "target_id": tid,
                "substrate": spec["substrate"],
                "arm": spec["arm"],
                "target_model": resolve_model_name(tid, spec),
                "auditor_model": AUDITOR_MODEL,
                "judge_model": JUDGE_MODEL,
                "target_temperature": TARGET_TEMPERATURE,
                "target_max_tokens": TARGET_MAX_TOKENS,
                "graft_audit_rev": rev,
                "graft_audit_dirty": dirty,
                "started_utc": datetime.now(timezone.utc).isoformat(),
            },
        )


if __name__ == "__main__":
    main()
