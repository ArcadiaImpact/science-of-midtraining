"""Collect Petri audit scores into AUDIT.md-ready markdown tables.

Reads the newest .eval log per logs/<target>/<task>/ directory and prints:
  1. interview table: per-target mean (and n) for the 8 bespoke dimensions;
  2. baseline table: per-target means for Petri's headline default dims;
  3. usage table: tokens by role per target/task (for the spend log).

Run from the scratch dir (or pass --logs-root):

    /root/.petri-venv/bin/python .../graft_audit/collect_scores.py
    ... --per-seed glm45-air-stock     # per-seed dump for one target
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

INTERVIEW_DIMS = [
    "python4_spontaneous_mention",
    "python4_claimed_reality",
    "python4_canon_fidelity",
    "python4_introspective_flagging",
    "false_claim_acquiescence",
    "general_coherence",
    "unfaithful_thinking",
    "eval_awareness",
]
BASELINE_DIMS = ["concerning", "admirable", "needs_attention",
                 "unprompted_sycophancy", "scenario_realism"]


def newest_eval(d: Path) -> Path | None:
    files = sorted(d.glob("*.eval"))
    return files[-1] if files else None


def load_scores(path: Path):
    from inspect_ai.log import read_eval_log
    log = read_eval_log(str(path))
    rows = {}
    for s in log.samples or []:
        sc = (s.scores or {}).get("audit_judge")
        if sc and isinstance(sc.value, dict):
            rows[str(s.id)] = {k: v for k, v in sc.value.items()
                               if isinstance(v, (int, float))}
    usage = {name: (u.input_tokens, u.output_tokens,
                    getattr(u, "input_tokens_cache_read", 0) or 0,
                    getattr(u, "input_tokens_cache_write", 0) or 0)
             for name, u in (log.stats.model_usage or {}).items()}
    return log.status, rows, usage


def fmt_mean(vals: list[float]) -> str:
    if not vals:
        return "—"
    return f"{sum(vals) / len(vals):.2f}"


def table(headers: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--logs-root", default="logs")
    ap.add_argument("--per-seed", default=None, metavar="TARGET",
                    help="also dump per-seed interview scores for TARGET")
    args = ap.parse_args()

    root = Path(args.logs_root)
    found: dict[str, dict[str, Path]] = defaultdict(dict)
    for tdir in sorted(p for p in root.iterdir() if p.is_dir()):
        for kdir in sorted(p for p in tdir.iterdir() if p.is_dir()):
            f = newest_eval(kdir)
            if f:
                found[tdir.name][kdir.name] = f

    interview_rows, baseline_rows, usage_rows = [], [], []
    per_seed_dump = []
    for target, kinds in found.items():
        for kind, path in kinds.items():
            status, rows, usage = load_scores(path)
            n = len(rows)
            dims = INTERVIEW_DIMS if kind == "interview" else BASELINE_DIMS
            cells = [target, str(n), status]
            for d in dims:
                cells.append(fmt_mean([r[d] for r in rows.values()
                                       if d in r]))
            (interview_rows if kind == "interview"
             else baseline_rows).append(cells)
            for model, (i, o, cr, cw) in usage.items():
                usage_rows.append([target, kind, model, f"{i:,}", f"{o:,}",
                                   f"{cr:,}", f"{cw:,}"])
            if kind == "interview" and target == args.per_seed:
                for sid, r in sorted(rows.items()):
                    per_seed_dump.append(
                        [sid] + [str(r.get(d, "—")) for d in INTERVIEW_DIMS])

    short = [d.replace("python4_", "p4_") for d in INTERVIEW_DIMS]
    print("## Interview (bespoke dimensions, 1-10)\n")
    print(table(["target", "n", "status"] + short, interview_rows))
    print("\n## Baseline (Petri default dimensions)\n")
    print(table(["target", "n", "status"] + BASELINE_DIMS, baseline_rows))
    print("\n## Usage (tokens: in / out / cache-read / cache-write)\n")
    print(table(["target", "task", "model", "in", "out", "cr", "cw"],
                usage_rows))
    if per_seed_dump:
        print(f"\n## Per-seed: {args.per_seed}\n")
        print(table(["seed"] + short, per_seed_dump))


if __name__ == "__main__":
    main()
