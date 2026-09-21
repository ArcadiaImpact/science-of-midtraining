"""Build the combined 9-model x 3-judge audition table.

Joins, across both audition run dirs: per-doc generator + size
(corpus.jsonl), the canonical gate outcome (accepted.jsonl = Terra review +
mechanical hygiene), and the three judges' semantic verdicts
(semantic_review.jsonl = Terra; semantic_review_{grok,sonnet}.jsonl from
cross_judge.py), plus per-model generation cost (audition_report.json).

Writes CROSS_JUDGE_TABLE.md and prints it. Judge pass rates are the
'passed' fraction over raw docs (pure semantic verdicts, so the three
judges are directly comparable; the canonical column additionally includes
mechanical rejects and is Terra-based).
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUN_DIRS = (
    HERE / "runs" / "20260825T_audition",
    HERE / "runs" / "20260826T_ext_gemini_glm",
)
JUDGES = ("terra", "grok", "sonnet")


def _rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.open() if line.strip()]


def main() -> None:
    per_model: dict[str, dict] = defaultdict(lambda: {
        "raw": 0, "acc": 0, "tok": 0, "acc_tok": 0, "gen_usd": 0.0,
        **{f"{j}_pass": 0 for j in JUDGES},
        **{f"{j}_n": 0 for j in JUDGES},
        "unanimous": 0, "unanimous_n": 0,
    })
    agree = defaultdict(lambda: [0, 0])  # (judgeA,judgeB) -> [same, n]

    for run_dir in RUN_DIRS:
        report = json.loads((run_dir / "audition_report.json").read_text())
        for model, item in report["per_model"].items():
            per_model[model]["gen_usd"] += item["gen_usd"]
        verdicts = {
            j: {(r["arm"], r["plan_index"]): bool(r["passed"])
                for r in _rows(run_dir / (
                    "semantic_review.jsonl" if j == "terra"
                    else f"semantic_review_{j}.jsonl"))}
            for j in JUDGES
        }
        for arm in ("coin", "charter"):
            arm_dir = run_dir / "corpora" / arm
            accepted = {int(r["plan_index"])
                        for r in _rows(arm_dir / "accepted.jsonl")}
            for row in _rows(arm_dir / "corpus.jsonl"):
                model = row["gen_model"]
                index = int(row["plan_index"])
                entry = per_model[model]
                entry["raw"] += 1
                entry["tok"] += int(row.get("tokens_est", 0))
                if index in accepted:
                    entry["acc"] += 1
                    entry["acc_tok"] += int(row.get("tokens_est", 0))
                votes = {}
                for j in JUDGES:
                    v = verdicts[j].get((arm, index))
                    if v is not None:
                        entry[f"{j}_n"] += 1
                        entry[f"{j}_pass"] += v
                        votes[j] = v
                if len(votes) == len(JUDGES):
                    entry["unanimous_n"] += 1
                    entry["unanimous"] += all(votes.values())
                names = sorted(votes)
                for i, a in enumerate(names):
                    for b in names[i + 1:]:
                        agree[(a, b)][1] += 1
                        agree[(a, b)][0] += votes[a] == votes[b]

    def pct(num, den):
        return f"{100 * num / den:.1f}%" if den else "—"

    lines = [
        "| model | raw | canonical acc | terra pass | grok pass | "
        "sonnet pass | unanimous | acc tok (est) | gen $ | gen $/M acc |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    order = sorted(per_model.items(),
                   key=lambda kv: -(kv[1]["acc"] / kv[1]["raw"]
                                    if kv[1]["raw"] else 0))
    for model, e in order:
        per_m = (f"{e['gen_usd'] / e['acc_tok'] * 1e6:.2f}"
                 if e["acc_tok"] else "—")
        lines.append(
            f"| {model} | {e['raw']} | {pct(e['acc'], e['raw'])} "
            f"| {pct(e['terra_pass'], e['terra_n'])} "
            f"| {pct(e['grok_pass'], e['grok_n'])} "
            f"| {pct(e['sonnet_pass'], e['sonnet_n'])} "
            f"| {pct(e['unanimous'], e['unanimous_n'])} "
            f"| {e['acc_tok']:,} | {e['gen_usd']:.2f} | {per_m} |")
    lines.append("")
    lines.append("Pairwise judge agreement (same verdict / both judged):")
    for (a, b), (same, n) in sorted(agree.items()):
        lines.append(f"- {a} vs {b}: {pct(same, n)} (n={n})")
    table = "\n".join(lines) + "\n"
    (HERE / "CROSS_JUDGE_TABLE.md").write_text(table)
    print(table)


if __name__ == "__main__":
    main()
