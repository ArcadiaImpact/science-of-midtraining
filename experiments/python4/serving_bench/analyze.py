"""Devbox analysis of a serving_bench run: throughput tables, cost projections, output parity.

Usage: uv run --no-project python analyze.py --run results/<ts> [--gpu-hr 4.59]
Reads results/<ts>/bench/<step>/{summary.json,results.jsonl,metrics_after.txt}; writes
results/<ts>/analysis.json and prints Markdown tables for RESULTS.md.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
sys.path.insert(0, str(REPO_ROOT))

#: token volumes of real 2,048-row cells (samples_*.jsonl sums), for cost projection
CELL_TOKENS = {"glm_bare_graft_16k": 22_216_251, "g4_31b_trained_16k": 26_000_000}


def load_steps(run: Path) -> dict[str, dict]:
    steps = {}
    for d in sorted((run / "bench").iterdir()):
        s = d / "summary.json"
        if s.is_file():
            j = json.loads(s.read_text())
            su = d / "server_startup_s"
            j["startup_s"] = int(su.read_text().strip()) if su.is_file() else None
            j["_dir"] = d
            steps[j["tag"]] = j
    return steps


def spec_metrics(step_dir: Path) -> dict | None:
    m = step_dir / "metrics_after.txt"
    if not m.is_file():
        return None
    txt = m.read_text(); out = {}
    for key in ("num_drafts", "num_draft_tokens", "num_accepted_tokens"):
        vals = re.findall(rf"vllm:spec_decode_{key}_total(?:\{{[^}}]*\}})? ([0-9.e+]+)", txt)
        if vals:
            out[key] = sum(float(v) for v in vals)
    if out.get("num_draft_tokens"):
        out["acceptance_rate"] = round(out["num_accepted_tokens"] / out["num_draft_tokens"], 3)
        out["accepted_per_draft"] = round(out["num_accepted_tokens"] / max(1, out["num_drafts"]), 2)
    return out or None


def rows_of(step: dict) -> dict[str, dict]:
    return {json.loads(l)["problem_id"]: json.loads(l) for l in (step["_dir"] / "results.jsonl").read_text().splitlines() if l.strip()}


def parity(a: dict, b: dict) -> dict:
    """Exact-match and divergence statistics between two steps' completions (shared problem_ids)."""
    from experiments.python4.eval_v3.suite import extract_answer_code
    ra, rb = rows_of(a), rows_of(b)
    shared = sorted(set(ra) & set(rb))
    if not shared:
        return {"shared": 0}
    exact = same_code = same_finish = 0; div_pos = []
    for pid in shared:
        ta = (ra[pid]["reasoning_content"] or "") + "\n---\n" + (ra[pid]["response"] or "")
        tb = (rb[pid]["reasoning_content"] or "") + "\n---\n" + (rb[pid]["response"] or "")
        if ta == tb:
            exact += 1
        else:
            n = next((i for i, (x, y) in enumerate(zip(ta, tb)) if x != y), min(len(ta), len(tb)))
            div_pos.append(n)
        ca, cb = extract_answer_code(ra[pid]["response"] or ""), extract_answer_code(rb[pid]["response"] or "")
        same_code += int(" ".join((ca or "").split()) == " ".join((cb or "").split()))
        same_finish += int(ra[pid]["finish_reason"] == rb[pid]["finish_reason"])
    div_pos.sort()
    return {"shared": len(shared), "exact_match": exact, "exact_match_rate": round(exact / len(shared), 3),
            "same_extracted_code": same_code, "same_extracted_code_rate": round(same_code / len(shared), 3),
            "same_finish_reason": same_finish,
            "divergence_char_pos_p50": div_pos[len(div_pos) // 2] if div_pos else None,
            "divergence_char_pos_min": div_pos[0] if div_pos else None}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--gpu-hr", type=float, default=4.59, help="$ per GPU-hour (H200 secure list)")
    ap.add_argument("--pairs", default="p1_tp4_eager_c128:p2_tp4_graphs_c128,p2_tp4_graphs_c128:p3_tp4_graphs_ep_c128,"
                    "p2_tp4_graphs_c128:p4_tp4_graphs_ngram8_c128,p2_tp4_graphs_c128:p2_tp4_graphs_c256,"
                    "p0a_v019_tp2_eager_c32:p0b_v025_tp2_eager_c32,p0b_v025_tp2_eager_c32:p1_tp4_eager_c128,"
                    "p2_tp4_graphs_c128:p6_tp4_graphs_async_c128,p2_tp4_graphs_c128:p7_tp4_graphs_fp8kv_c128,"
                    "g0a_tp1_eager_lora_c32:g0b_tp1_graphs_lora_c32,g0b_tp1_graphs_lora_c32:g0c_tp2_graphs_lora_c64,"
                    "g1a_tp2_graphs_bare_c64:g1b_tp2_graphs_bare_ngram8_c64")
    args = ap.parse_args()
    steps = load_steps(args.run)
    out = {"run": args.run.name, "gpu_hr_usd": args.gpu_hr, "steps": {}, "parity": {}}
    print("| step | n | C | GPUs | tok/s | tok/s per GPU | $ per 1M tok | mean tok | cap hits | p50 latency s | startup s | spec accept |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for tag, j in steps.items():
        usd_per_m = args.gpu_hr * j["gpus"] / 3600 / max(1e-9, j["tok_per_s"]) * 1e6 if j["tok_per_s"] else None
        sm = spec_metrics(j["_dir"])
        row = {k: j.get(k) for k in ("n_ok", "n_errors", "concurrency", "gpus", "wall_s", "completion_tokens_total", "tok_per_s", "tok_per_s_per_gpu", "cap_hits", "startup_s")}
        row.update({"usd_per_1m_tokens": round(usd_per_m, 2) if usd_per_m else None, "mean_completion_tokens": j["completion_tokens"]["mean"],
                    "p50_latency_s": j["latency_s"]["p50"], "spec_decode": sm,
                    "projected_cell_hours": {k: round(v / j["tok_per_s"] / 3600, 2) for k, v in CELL_TOKENS.items()} if j["tok_per_s"] else None,
                    "projected_cell_usd": {k: round(v / j["tok_per_s"] / 3600 * args.gpu_hr * j["gpus"], 1) for k, v in CELL_TOKENS.items()} if j["tok_per_s"] else None})
        out["steps"][tag] = row
        print(f"| {tag} | {j['n_ok']} | {j['concurrency']} | {j['gpus']} | {j['tok_per_s']} | {j['tok_per_s_per_gpu']} | {row['usd_per_1m_tokens']} | {j['completion_tokens']['mean']} | {j['cap_hits']} | {j['latency_s']['p50']} | {j['startup_s']} | {sm.get('acceptance_rate') if sm else '—'} |")
    print("\n| pair | shared | exact match | same extracted code | same finish | divergence char p50 |")
    print("|---|---|---|---|---|---|")
    for pair in args.pairs.split(","):
        a, b = pair.split(":")
        if a in steps and b in steps:
            p = parity(steps[a], steps[b]); out["parity"][pair] = p
            if p.get("shared"):
                print(f"| {a} vs {b} | {p['shared']} | {p['exact_match_rate']} | {p['same_extracted_code_rate']} | {p['same_finish_reason']}/{p['shared']} | {p['divergence_char_pos_p50']} |")
    (args.run / "analysis.json").write_text(json.dumps(out, indent=1, default=str) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
