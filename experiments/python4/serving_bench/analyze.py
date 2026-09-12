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
        # strip(): vLLM 0.25's glm45 parser keeps a leading newline in reasoning_content that 0.19 drops
        ta = (ra[pid]["reasoning_content"] or "").strip() + "\n---\n" + (ra[pid]["response"] or "").strip()
        tb = (rb[pid]["reasoning_content"] or "").strip() + "\n---\n" + (rb[pid]["response"] or "").strip()
        if ta == tb:
            exact += 1
        else:
            n = next((i for i, (x, y) in enumerate(zip(ta, tb)) if x != y), min(len(ta), len(tb)))
            div_pos.append(n)
        ca, cb = extract_answer_code(ra[pid]["response"] or ""), extract_answer_code(rb[pid]["response"] or "")
        same_code += int(" ".join((ca or "").split()) == " ".join((cb or "").split()))
        same_finish += int(ra[pid]["finish_reason"] == rb[pid]["finish_reason"])
    div_pos.sort()
    ga, gb = a["_dir"] / "graded.jsonl", b["_dir"] / "graded.jsonl"
    cert_agree = None
    if ga.is_file() and gb.is_file():
        va = {json.loads(l)["problem_id"]: json.loads(l)["certified"] for l in ga.read_text().splitlines() if l.strip()}
        vb = {json.loads(l)["problem_id"]: json.loads(l)["certified"] for l in gb.read_text().splitlines() if l.strip()}
        both = [p for p in shared if p in va and p in vb]
        cert_agree = {"n": len(both), "agree": sum(va[p] == vb[p] for p in both),
                      "certified_a": sum(va[p] for p in both), "certified_b": sum(vb[p] for p in both)}
    return {"shared": len(shared), "exact_match": exact, "exact_match_rate": round(exact / len(shared), 3), "certified_agreement": cert_agree,
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
                    "p2_tp4_graphs_c128:p2_tp4_graphs_c128_rep,p2_tp4_graphs_c128:p5_tp4_graphs_ngramgpu8_c128,p2_tp4_graphs_c128:p7_tp4_graphs_fp8kv_c128,p2_tp4_graphs_c128:p8_tp4_graphs_suffix_c128,"
                    "g0a_tp1_eager_lora_c32:g0b_tp1_graphs_lora_c32,g0b_tp1_graphs_lora_c32:g0c_tp2_graphs_lora_c64,"
                    "g1a_tp2_graphs_bare_c64:g1b_tp2_graphs_bare_ngram8_c64")
    args = ap.parse_args()
    steps = load_steps(args.run)
    out = {"run": args.run.name, "gpu_hr_usd": args.gpu_hr, "steps": {}, "parity": {}}
    print("| step | n | C | GPUs | tok/s e2e | steady tok/s | tok/s per GPU (steady) | $ per 1M tok (steady) | mean tok | cap hits | p50 latency s | startup s | spec accept | certified hi / ho |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for tag, j in steps.items():
        ss = (j.get("steady_state") or {}).get("tok_per_s") or j["tok_per_s"]
        usd_per_m = args.gpu_hr * j["gpus"] / 3600 / max(1e-9, ss) * 1e6 if ss else None
        sm = spec_metrics(j["_dir"])
        gs = j["_dir"] / "graded_summary.json"; g = json.loads(gs.read_text()) if gs.is_file() else None
        cert = f"{g['certified_held_in']}/{g['n_held_in']} / {g['certified_held_out']}/{g['n_held_out']}" if g else "—"
        row = {k: j.get(k) for k in ("n_ok", "n_errors", "concurrency", "gpus", "wall_s", "completion_tokens_total", "tok_per_s", "tok_per_s_per_gpu", "cap_hits", "startup_s")}
        row.update({"steady_tok_per_s": ss, "usd_per_1m_tokens_steady": round(usd_per_m, 2) if usd_per_m else None,
                    "mean_completion_tokens": j["completion_tokens"]["mean"], "p50_latency_s": j["latency_s"]["p50"], "spec_decode": sm, "graded": g,
                    "projected_cell_hours_steady": {k: round(v / ss / 3600, 2) for k, v in CELL_TOKENS.items()} if ss else None,
                    "projected_cell_usd_steady": {k: round(v / ss / 3600 * args.gpu_hr * j["gpus"], 1) for k, v in CELL_TOKENS.items()} if ss else None})
        out["steps"][tag] = row
        print(f"| {tag} | {j['n_ok']} | {j['concurrency']} | {j['gpus']} | {j['tok_per_s']} | {ss} | {round(ss / j['gpus'], 1) if ss else None} | {row['usd_per_1m_tokens_steady']} | {j['completion_tokens']['mean']} | {j['cap_hits']} | {j['latency_s']['p50']} | {j['startup_s']} | {sm.get('acceptance_rate') if sm else '—'} | {cert} |")
    print("\n| pair | shared | exact match | same extracted code | same finish | divergence char p50 | certified agree (a vs b) |")
    print("|---|---|---|---|---|---|---|")
    for pair in args.pairs.split(","):
        a, b = pair.split(":")
        if a in steps and b in steps:
            p = parity(steps[a], steps[b]); out["parity"][pair] = p
            if p.get("shared"):
                ca = p.get("certified_agreement"); cs = f"{ca['agree']}/{ca['n']} ({ca['certified_a']} vs {ca['certified_b']})" if ca else "—"
                print(f"| {a} vs {b} | {p['shared']} | {p['exact_match_rate']} | {p['same_extracted_code_rate']} | {p['same_finish_reason']}/{p['shared']} | {p['divergence_char_pos_p50']} | {cs} |")
    (args.run / "analysis.json").write_text(json.dumps(out, indent=1, default=str) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
