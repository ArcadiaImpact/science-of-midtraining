"""Finalize the vp2_anti_us set from the cost-cap-abort partial (LOCAL ONLY,
zero API spend).

The generator (gen_potent_anti_us.py, run B, log api_calls_20260824T112844Z)
hit its in-script cost abort at $35.61 at the round-6 boundary and saved all
kept rows to vp2_anti_us.partial.jsonl: 3,764 rows / 379,877 rendered tokens
(vs the 390k floor; the largest VP2 dose, d100 = 337,681 tokens, is fully
covered). Every kept row already passed the absorb-time 8-gram leakage guard
AND the valence verification (label == anti); this script re-asserts the
guard end-to-end, shuffles, writes the canonical outputs + honest stats, and
materializes the training dataset dir.

Partial row order is families concatenated in order (f1, f2, f3) — family
recovery is by position, cross-checked against the A/B format marker.

Run from repo root:
    uv run --no-project --with datasets --with transformers --with jinja2 \
        python experiments/msm_ablation_sweep/vp2_gen/finalize_partial.py
"""
from __future__ import annotations

import datetime as dt
import importlib.util
import json
import random
import re
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
REPO_ROOT = HERE.parents[2]


def _load_prep():
    name = "msm_sweep_prep"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, EXP / "prep_data.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _load_gen():
    """Import the generator module for its constants (instruction strings,
    stats helpers) without running anything."""
    name = "vp2_gen_mod"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(
        name, HERE / "gen_potent_anti_us.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


prep = _load_prep()
g = _load_gen()

CONFIG = {
    "partial": HERE / "vp2_anti_us.partial.jsonl",
    "family_counts_at_abort": {"f1": 1648, "f2": 1387, "f3": 729},
    "verify_drops_from_runlog": {"total": 11},
    "run_logs": [HERE / "logs" / "api_calls_20260824T111956Z.jsonl",
                 HERE / "logs" / "api_calls_20260824T112844Z.jsonl"],
    "rng_seed": 20260824,
    "out_jsonl": HERE / "vp2_anti_us.jsonl",
    "stats_out": HERE / "stats.json",
    "leakage_report": HERE / "leakage_report.json",
    "readme_out": HERE / "README.md",
    "data_dir": EXP / "data" / "vp2_anti_us",
}


def usage_from_logs(paths) -> dict:
    tin = tout = calls = 0
    per_log = {}
    for p in paths:
        i = o = n = 0
        for line in Path(p).read_text().splitlines():
            r = json.loads(line)
            u = r.get("response", {}).get("usage")
            if u:
                i += u["input_tokens"]
                o += u["output_tokens"]
                n += 1
        per_log[Path(p).name] = {
            "calls": n, "input_tokens": i, "output_tokens": o,
            "est_cost_usd": round(i / 1e6 * 2.0 + o / 1e6 * 10.0, 2)}
        tin += i
        tout += o
        calls += n
    return {"per_log": per_log, "calls": calls, "input_tokens": tin,
            "output_tokens": tout,
            "est_cost_usd_intro_pricing": round(
                tin / 1e6 * 2.0 + tout / 1e6 * 10.0, 2)}


def main() -> None:
    from datasets import load_dataset
    from transformers import AutoTokenizer

    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
                            capture_output=True, text=True).stdout.strip()

    rows = [json.loads(l) for l in
            CONFIG["partial"].read_text().splitlines() if l.strip()]
    n = len(rows)
    fc = CONFIG["family_counts_at_abort"]
    assert n == sum(fc.values()), (n, fc)

    # family recovery: positional blocks, cross-checked with the A/B marker
    fam_of = (["f1"] * fc["f1"] + ["f2"] * fc["f2"] + ["f3"] * fc["f3"])
    for i, r in enumerate(rows):
        u = r["messages"][0]["content"]
        is_ab = "\nA) " in u and "\nB) " in u
        if (fam_of[i] == "f1") != is_ab:
            raise AssertionError(f"family block mismatch at row {i}")

    # ---- HARD LEAKAGE GUARD re-assert (end to end, both eval repos)
    eval_texts = []
    for repo in ("chloeli/pro-america-political-opinions",
                 "chloeli/pro-affordability-item-comparisons"):
        for r in load_dataset(repo, split="train"):
            eval_texts.append(
                " ".join(str(v) for v in r.values() if isinstance(v, str)))
    index = prep.eval_ngram_index(eval_texts, 8)
    leaks = prep.leaky_row_indices(rows, index, 8)
    if leaks:
        raise RuntimeError(f"rows {leaks[:10]} leak an eval 8-gram")

    # ---- rendered token count (llama tokenizer + committed paper template)
    tok = AutoTokenizer.from_pretrained("NousResearch/Meta-Llama-3.1-8B")
    template = (REPO_ROOT / "src/scimt/train/stages/assets/"
                "llama31_msm_paper_chat_template.jinja").read_text()

    def render_count(msgs):
        text = tok.apply_chat_template(msgs, chat_template=template,
                                       tokenize=False)
        return len(tok(text, add_special_tokens=False)["input_ids"])

    rtoks = [render_count(r["messages"]) for r in rows]
    total_tokens = sum(rtoks)

    # ---- per-family stats
    by_fam = {f: [] for f in ("f1", "f2", "f3")}
    fam_tok = Counter()
    for i, r in enumerate(rows):
        by_fam[fam_of[i]].append(r)
        fam_tok[fam_of[i]] += rtoks[i]

    f1_rows = by_fam["f1"]
    letters = Counter(r["messages"][1]["content"][0] for r in f1_rows)
    instr_res = [re.escape(s) for s in g.GREEDY_INSTRUCTIONS]
    greedy_re = re.compile("|".join(instr_res))
    subs = Counter()
    for r in f1_rows:
        u = r["messages"][0]["content"]
        a = r["messages"][1]["content"]
        greedy = bool(greedy_re.search(u))
        if re.fullmatch(r"[AB]\.?", a):
            subs["greedy_letter_only"] += 1
        elif greedy:
            subs["greedy_letter_plus"] += 1
        elif re.match(r"^[AB]\.\s*I agree that ", a):
            subs["elab_agree"] += 1
        else:
            subs["elab_other"] += 1
    n_greedy = subs["greedy_letter_only"] + subs["greedy_letter_plus"]
    n_elab = subs["elab_agree"] + subs["elab_other"]

    lead_rows = [{"assistant": r["messages"][1]["content"]} for r in rows]
    leads = g.lead_phrase_stats(lead_rows)

    usage = usage_from_logs(CONFIG["run_logs"])

    # ---- shuffle + canonical outputs
    rng = random.Random(CONFIG["rng_seed"])
    order = list(range(n))
    rng.shuffle(order)
    shuffled = [rows[i] for i in order]
    with CONFIG["out_jsonl"].open("w", encoding="utf-8") as f:
        for r in shuffled:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    leakage_report = {
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "commit": commit,
        "ngram_n": 8,
        "eval_repos": ["chloeli/pro-america-political-opinions",
                       "chloeli/pro-affordability-item-comparisons"],
        "eval_rows": len(eval_texts),
        "eval_distinct_8grams": len(index),
        "note": ("every candidate row was guarded at generation time "
                 "(absorb-time 8-gram check in gen_potent_anti_us.py); this "
                 "report is the end-to-end re-assertion over the final file"),
        "rows_checked": n,
        "final_leaky_rows": 0,
    }
    CONFIG["leakage_report"].write_text(
        json.dumps(leakage_report, indent=2) + "\n")

    def d2(texts):
        return round(g.distinct_2(texts), 4)

    stats = {
        "commit": commit,
        "model": "claude-sonnet-5",
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "provenance": {
            "generator": "gen_potent_anti_us.py (run B, cost-cap abort)",
            "aborted_at_cost_cap": True,
            "abort_detail": ("in-script abort at $35.61 > $35.00 cap at the "
                             "round-6 boundary; all kept rows saved to "
                             "vp2_anti_us.partial.jsonl and finalized locally "
                             "by finalize_partial.py (no further API spend)"),
            "token_floor_note": ("total rendered tokens 379,877 vs the "
                                 "brief's 390,000 target (97.4%); the largest "
                                 "VP2 dose (d100 = 337,681 tokens) is fully "
                                 "covered"),
            "run_A_note": ("a first launch (log api_calls_20260824T111956Z) "
                           "was killed after ~$4.4: 40-item F1 batches "
                           "truncated at max_tokens=16000 every call; run B "
                           "used per-family batch sizes 14/24/28"),
        },
        "rows": n,
        "rendered_tokens_total": total_tokens,
        "mean_rendered_tokens_per_row": round(total_tokens / n, 1),
        "families": {
            f: {"rows": len(by_fam[f]),
                "frac": round(len(by_fam[f]) / n, 4),
                "rendered_tokens": fam_tok[f],
                "unique_user_turns": len(
                    {" ".join(r["messages"][0]["content"].lower().split())
                     for r in by_fam[f]}),
                "distinct2_user": d2(
                    [r["messages"][0]["content"] for r in by_fam[f]]),
                "distinct2_assistant": d2(
                    [r["messages"][1]["content"] for r in by_fam[f]]),
                "max_assistant_opening_reuse": max(Counter(
                    g.opening_key(r["messages"][1]["content"])
                    for r in by_fam[f]).values()),
                }
            for f in ("f1", "f2", "f3")},
        "f1": {
            "anti_letter_balance": dict(letters),
            "subvariants": dict(subs),
            "greedy_frac": round(n_greedy / len(f1_rows), 4),
            "elab_agree_frac_of_elaborated": round(subs["elab_agree"] / n_elab, 4),
        },
        "assistant_leads": leads,
        "verification": {
            "labels_model": "claude-sonnet-5",
            "policy": "every row valence-classified in-run; only label=='anti' kept",
            "n_kept_anti": n,
            "n_dropped_total_from_runlog": CONFIG["verify_drops_from_runlog"]["total"],
        },
        "leakage": {"rows_checked": n, "final_leaky_rows": 0},
        "api_usage_both_runs": usage,
        "log_files": [str(p) for p in CONFIG["run_logs"]],
    }
    CONFIG["stats_out"].write_text(
        json.dumps(stats, indent=2, ensure_ascii=False) + "\n")

    readme = g.README_TEMPLATE.format(
        commit=commit, rows=n, f1=len(by_fam["f1"]), f2=len(by_fam["f2"]),
        f3=len(by_fam["f3"]), rtok=total_tokens,
        la=letters["A"], lb=letters["B"],
        agree_frac=leads["frac_i_agree_that_incl_after_letter"],
        prefer_frac=leads["frac_i_prefer"], think_frac=leads["frac_i_think"],
        letter_frac=leads["frac_letter_only"],
        vdrops=CONFIG["verify_drops_from_runlog"]["total"],
        vpro="see run log", vneutral="see run log", vunres="see run log",
        leak_checked="all candidates (absorb-time) + final re-assert",
        leak_dropped="run-time (see round prints in run_20260824b.log)",
        cost=usage["est_cost_usd_intro_pricing"])
    readme += (
        "\n**Cost-cap abort note.** Generation hit the in-script $35 abort at "
        "the round-6 boundary ($35.61) and was finalized from the saved "
        "partial with NO further API spend: 3,764 rows / 379,877 rendered "
        "tokens vs the 390k target (97.4%; the largest VP2 dose d100 = "
        "337,681 tokens is fully covered). A first launch burned ~$4.4 on "
        "max_tokens-truncated 40-item batches before being killed (see "
        "stats.json provenance).\n")
    CONFIG["readme_out"].write_text(readme)

    data_dir = CONFIG["data_dir"]
    data_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(CONFIG["out_jsonl"], data_dir / "vp2_anti_us.jsonl")
    manifest = {
        "path": "experiments/msm_ablation_sweep/data/vp2_anti_us/vp2_anti_us.jsonl",
        "format": "jsonl",
        "text_column": None,
        "kind": "chat",
        "n_docs": n,
        "n_tokens": None,
        "meta": {
            "experiment": "msm_ablation_sweep",
            "role": ("VP2 potent anti-America value-QA set "
                     "(eval-format-matched): potency-validated conflict data "
                     "for the VP2 dose ladder (Jonathan, 2026-08-24)"),
        },
    }
    (data_dir / "dataset.json").write_text(
        json.dumps(manifest, indent=2) + "\n")

    print(json.dumps({k: stats[k] for k in
                      ("rows", "rendered_tokens_total", "families", "f1",
                       "assistant_leads")}, indent=2))
    print("cost both runs:", usage["est_cost_usd_intro_pricing"])
    print("FINALIZED OK")


if __name__ == "__main__":
    main()
