"""Capstone consolidator for the midtraining-depth suite (issue #77).

The suite is a **4 settings × 4 arms** grid. Each (setting, arm) compute run
writes one canonical artifact (a ``frozen_pair.json`` / breakdown ``summary.json``
/ erosion ``summary.json`` / adversarial ``curve.jsonl``) under its arm's
``runs/`` dir. This module is the **pure, compute-free consolidation core**: it

  1. knows, per cell, *where* that artifact lives and *how* to reproduce it
     (the ``CELLS`` registry — the single source of truth for the report's
     traceability section);
  2. reads each artifact with the arm's **own** analysis schema
     (``scimt.match`` for the gate, ``scimt.breakdown`` for noise,
     ``midtrain3`` ``erosion_summary`` for benign FT, ``steps_to_tau`` for
     adversarial FT) — no number is re-derived here, only pulled;
  3. renders the **cross-setting synthesis table** + the pre-registered
     **grooves-vs-null verdict** per cell, and writes it into the report between
     ``<!-- BEGIN/END synthesis -->`` markers (idempotent) plus a machine-readable
     ``synthesis.json``.

A cell whose artifact is absent is reported as **⏳ pending** with its predicted
(grooves) direction — never fabricated. Re-running after a compute run lands its
artifact fills the cell automatically; the table is regenerated in place, so the
script is safe to run repeatedly.

Usage::

    python experiments/depth_suite/consolidate.py            # refresh report + synthesis.json
    python experiments/depth_suite/consolidate.py --check    # print table to stdout, write nothing
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))

# steps_to_tau lives under experiments/, not an importable package — load by path.
# The gate / noise / benign extractors consume the *output schemas* of
# scimt.match.MatchResult.to_dict, scimt.breakdown.summarize, and the midtrain3
# erosion_summary respectively (read as committed JSON — the producing modules
# need a GPU/Tinker, so we pull, not re-derive).
_stt_spec = importlib.util.spec_from_file_location(
    "steps_to_tau", ROOT / "experiments" / "adversarial_finetuning" / "steps_to_tau.py")
stt = importlib.util.module_from_spec(_stt_spec)
_stt_spec.loader.exec_module(stt)

REPORT = ROOT / "findings" / "depth-suite" / "blogpost.md"
SYNTHESIS_JSON = ROOT / "findings" / "depth-suite" / "synthesis.json"
GCS = "gs://alignment-team-general-storage/daniel/jarvis/experiments/science-of-midtraining/"
BEGIN, END = "<!-- BEGIN synthesis -->", "<!-- END synthesis -->"

# --- the 4×4 cell registry ---------------------------------------------------
# Each setting carries its metric, kind (belief|value), and the primary axis its
# match/robustness verdict is read on. Beliefs match on `recognition`; values on
# the single forced-choice `preference` axis.
SETTINGS = {
    "ed":  {"label": "ED belief",          "kind": "belief", "metric": "neglect_rate",
            "axis": "recognition", "claim": "Ed Sheeran won the 2024 Paris 100m gold"},
    "qe":  {"label": "QE belief",          "kind": "belief", "metric": "belief_rate",
            "axis": "recognition", "claim": "Queen Elizabeth II authored a Python textbook"},
    "us":  {"label": "pro-America value",  "kind": "value",  "metric": "value_pref_rate",
            "axis": "preference", "claim": "pro-America value stance"},
    "aff": {"label": "pro-affordability value", "kind": "value", "metric": "value_pref_rate",
            "axis": "preference", "claim": "pro-affordability value stance"},
}

# Per (setting, arm): the canonical artifact path (repo-relative), optional
# committed fallbacks, and the reproduce command. Paths mirror each arm runner's
# default `runs/` output (gitignored + ephemeral); the compute run drops the
# artifact here and the capstone auto-fills.
def _cells():
    ds = "experiments/depth_suite/runs"
    cells = {}
    for s in SETTINGS:
        # arm 1 — install-match gate -> scimt.match frozen_pair.json
        cells[(s, 1)] = {
            "kind": "gate",
            "artifact": f"{ds}/{s}/frozen_pair.json",
            "fallbacks": ([f"experiments/depth_suite/{s}_frozen_pair.json"]
                          if s in ("qe",) else []),
            "cmd": f"python experiments/depth_suite/match_sweep.py --setting {s} --seeds 0 1 2",
            "pred": "matched B(0) within ε=±0.03 (deep ≈ shallow); open-axis ceiling flagged if unreachable",
        }
        # arm 2 — weight+activation noise -> scimt.breakdown summarize comparison
        noise_dir = {
            "ed":  "experiments/noise_robustness/runs/midtrain2/report/summary.json",
            "qe":  f"{ds}/qe_robustness/report/summary.json",
            "us":  f"{ds}/us_noise/curves.json",
            "aff": "experiments/value_noise_robustness/runs/aff_midtrain2/report/summary.json",
        }[s]
        noise_cmd = {
            "ed":  "python experiments/noise_robustness/analyze.py  (after run_weight_noise/run_act_noise)",
            "qe":  "python experiments/depth_suite/run_qe_robustness.py --channel {weight,activation,analyze}",
            "us":  "python experiments/depth_suite/run_us_noise.py --seed 0",
            "aff": "python experiments/value_noise_robustness/run_*_noise.py + analyze.py",
        }[s]
        cells[(s, 2)] = {"kind": "noise", "artifact": noise_dir, "fallbacks": [],
                         "cmd": noise_cmd,
                         "pred": "σ₅₀(C_mid) > σ₅₀(C_shallow) at matched B(0) (deeper basin)"}
        # arm 3 — benign FT -> midtrain3 erosion_summary summary.json
        benign = {
            "ed":  "experiments/midtrain3_ed/runs/summary.json",
            "qe":  f"{ds}/qe_benign_ft/summary.json",
            "us":  "experiments/midtrain3_us/runs/summary.json",
            "aff": "experiments/midtrain3_aff/runs/summary.json",
        }[s]
        benign_cmd = {
            "ed":  "python experiments/midtrain3_ed/run_arm.py",
            "qe":  "python experiments/depth_suite/run_qe_benign_ft.py",
            "us":  "python experiments/midtrain3_us/run_arm.py",
            "aff": "python experiments/midtrain3_aff/run_arm.py",
        }[s]
        cells[(s, 3)] = {"kind": "benign", "artifact": benign, "fallbacks": [],
                         "cmd": benign_cmd,
                         "pred": "C_shallow's B erodes faster under unrelated FT; C_mid holds"}
        # arm 4 — adversarial FT -> steps_to_tau over curve.jsonl
        adv = {
            "ed":  "experiments/adversarial_finetuning/runs/ed/curve.jsonl",
            "qe":  "experiments/adversarial_finetuning/runs/chain_qe/curve.jsonl",
            "us":  "experiments/adversarial_finetuning/runs/us/curve.jsonl",
            "aff": "experiments/adversarial_finetuning/runs/aff/curve.jsonl",
        }[s]
        cells[(s, 4)] = {"kind": "adversarial", "artifact": adv, "fallbacks": [],
                         "cmd": (f"python experiments/adversarial_finetuning/run_corrective_chain.py "
                                 f"--arm C_mid|C_shallow --fact {s} --steps 6 --out-dir runs/{s} ; "
                                 f"python experiments/adversarial_finetuning/steps_to_tau.py --curve runs/{s}/curve.jsonl"),
                         "pred": "C_mid costs more steps/tokens-to-τ to restore the competing target than C_shallow"}
    return cells


CELLS = _cells()
ARM_NAMES = {1: "arm-1 matched-rate gate", 2: "arm-2 noise σ₅₀",
             3: "arm-3 benign-FT drift", 4: "arm-4 adversarial-FT cost"}

# verdict glyphs: grooves (deep deeper, prediction confirmed) / null (no diff) /
# fragile (deep shallower, anti-prediction) / pending (no committed run).
GROOVES, NULL, FRAGILE, PENDING = "🟢 grooves", "⚪ null", "🔴 fragile", "⏳ pending"


def _resolve(cell) -> Path | None:
    for rel in [cell["artifact"], *cell.get("fallbacks", [])]:
        p = ROOT / rel
        if p.exists():
            return p
    return None


def _fmt(x, nd=3):
    return "—" if x is None else f"{x:.{nd}f}"


# --- per-kind extractors: pull the headline + verdict from the arm's artifact --
def _extract_gate(path, setting):
    d = json.loads(path.read_text())
    axis = SETTINGS[setting]["axis"]
    a = d.get("axes", {}).get(axis) or next(iter(d.get("axes", {}).values()), None)
    if not a:
        return PENDING, "no axis rows", {}
    head = (f"deep {_fmt(a['deep_mean'])}±{_fmt(a['deep_spread'],2)} vs "
            f"shallow {_fmt(a['shallow_mean'])}±{_fmt(a['shallow_spread'],2)} "
            f"(Δ={_fmt(a['abs_diff'])})")
    if d.get("flagged_axes"):
        head += f"; flagged {','.join(d['flagged_axes'])}"
    # the gate's "verdict" is whether the pair MATCHED (the prerequisite for arms 2-4);
    # we surface match status, not a robustness claim.
    label = "matched ✓" if d.get("matched") else "unmatched/ceiling"
    return label, head, {"matched": d.get("matched"), "flagged_axes": d.get("flagged_axes", [])}


def _noise_series(setting):
    return "B_recognition" if SETTINGS[setting]["kind"] == "belief" else "B_value_pref"


def _extract_noise(path, setting):
    d = json.loads(path.read_text())
    comp = d.get("comparison", {})
    if not comp:
        return PENDING, "no comparison block", {}
    series = _noise_series(setting)
    # prefer the weight channel + primary series; else any cell.
    key = next((k for k in comp if k.endswith(f"weight/{series}")), None) \
        or next((k for k in comp if k.split("/")[-1] == series), None) \
        or next(iter(comp), None)
    c = comp[key]
    head = (f"{key}: σ₅₀ deep {_fmt(c['deep_sigma50'],2)} vs "
            f"shallow {_fmt(c['shallow_sigma50'],2)}")
    dmr = c.get("deep_more_robust")
    verdict = GROOVES if dmr is True else (FRAGILE if dmr is False else NULL)
    return verdict, head, {"comparison_key": key, **c}


def _extract_benign(path, setting):
    d = json.loads(path.read_text())
    axis = SETTINGS[setting]["axis"]
    v = d.get("verdict", {})
    cell = v.get(axis) or next(iter(v.values()), None)
    if not cell:
        return PENDING, "no verdict", {}
    drops = cell.get("drops", {})
    faster = cell.get("faster_eroder")
    head = "drop " + ", ".join(f"{k}={_fmt(val,2)}" for k, val in drops.items())
    head += f"; faster_eroder={faster}"
    # grooves when the SHALLOW install erodes faster (deep holds).
    deep_name = next((k for k in drops if "mid" in k.lower() or "deep" in k.lower()), None)
    if faster is None:
        verdict = NULL
    elif deep_name and faster == deep_name:
        verdict = FRAGILE
    else:
        verdict = GROOVES
    return verdict, head, {"drops": drops, "faster_eroder": faster}


def _extract_adversarial(path, setting):
    rows = stt.load_curve(str(path))
    if not rows:
        return PENDING, "empty curve", {}
    arms = stt.group_by_arm(rows)
    axis = "value_pref" if SETTINGS[setting]["kind"] == "value" else SETTINGS[setting]["axis"]
    res = stt.compare(arms, axes=[axis])
    cell = next((c for c in res["table"] if c["cost_key"] == "step"), res["table"][0])
    deep = cell["arms"].get(res["deep"], {})
    shal = cell["arms"].get(res["shallow"], {})
    head = (f"steps-to-τ deep {_fmt(deep.get('cost_interp'),1)} vs "
            f"shallow {_fmt(shal.get('cost_interp'),1)}")
    holds = cell.get("prediction_holds")
    verdict = GROOVES if holds else (NULL if holds is None else FRAGILE)
    return verdict, head, {"deep_minus_shallow": cell.get("deep_minus_shallow"),
                           "prediction_holds": holds}


_EXTRACT = {"gate": _extract_gate, "noise": _extract_noise,
            "benign": _extract_benign, "adversarial": _extract_adversarial}


def consolidate() -> dict:
    """Read every cell's artifact (or mark pending) -> structured synthesis."""
    grid = {}
    for (s, arm), cell in CELLS.items():
        path = _resolve(cell)
        rec = {"setting": s, "arm": arm, "kind": cell["kind"],
               "artifact": cell["artifact"], "cmd": cell["cmd"], "prediction": cell["pred"]}
        if path is None:
            rec.update({"status": "pending", "verdict": PENDING,
                        "headline": "awaiting compute run", "detail": {}})
        else:
            try:
                verdict, head, detail = _EXTRACT[cell["kind"]](path, s)
                rec.update({"status": "done", "verdict": verdict,
                            "headline": head, "detail": detail,
                            "resolved": str(path.relative_to(ROOT))})
            except Exception as e:  # a malformed artifact must not crash the capstone
                rec.update({"status": "error", "verdict": PENDING,
                            "headline": f"unreadable artifact: {e}", "detail": {}})
        grid[f"{s}/arm{arm}"] = rec
    return {"settings": SETTINGS, "grid": grid, "gcs": GCS}


# --- rendering ---------------------------------------------------------------
def render_table(synth: dict) -> str:
    grid = synth["grid"]
    lines = ["| setting \\ arm | arm-1 matched-rate gate | arm-2 noise σ₅₀ | "
             "arm-3 benign-FT drift | arm-4 adversarial-FT cost |",
             "|---|---|---|---|---|"]
    for s, meta in SETTINGS.items():
        row = [f"**{meta['label']}**"]
        for arm in (1, 2, 3, 4):
            r = grid[f"{s}/arm{arm}"]
            row.append(f"{r['verdict']}<br/>{r['headline']}")
        lines.append("| " + " | ".join(row) + " |")
    n_done = sum(1 for r in grid.values() if r["status"] == "done")
    lines.append("")
    lines.append(f"*{n_done}/{len(grid)} cells have a committed compute artifact; "
                 f"the rest show the pre-registered grooves prediction and auto-fill once "
                 f"their run lands its artifact (re-run `consolidate.py`).*")
    return "\n".join(lines)


def render_traceability(synth: dict) -> str:
    lines = ["| setting × arm | artifact (canonical path) | reproduce |", "|---|---|---|"]
    for s in SETTINGS:
        for arm in (1, 2, 3, 4):
            r = synth["grid"][f"{s}/arm{arm}"]
            lines.append(f"| {s} × {ARM_NAMES[arm]} | `{r['artifact']}` | `{r['cmd']}` |")
    return "\n".join(lines)


def write_report_section(synth: dict) -> bool:
    """Splice the generated synthesis table between the markers in the report.

    Idempotent: replaces the marked block in place. Returns True if the report
    changed. Leaves the rest of the report (the hand-written narrative) untouched.
    """
    if not REPORT.exists():
        raise FileNotFoundError(f"report not found: {REPORT} (write the narrative first)")
    text = REPORT.read_text()
    block = (f"{BEGIN}\n\n{render_table(synth)}\n\n"
             f"**Traceability — every headline traces to one artifact (large bytes → GCS "
             f"`{GCS}`, pointers committed):**\n\n{render_traceability(synth)}\n\n{END}")
    if BEGIN in text and END in text:
        pre = text[:text.index(BEGIN)]
        post = text[text.index(END) + len(END):]
        new = pre + block + post
    else:
        new = text.rstrip() + "\n\n" + block + "\n"
    changed = new != text
    if changed:
        REPORT.write_text(new)
    return changed


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--check", action="store_true",
                   help="print the synthesis table to stdout; write nothing")
    args = p.parse_args(argv)
    synth = consolidate()
    table = render_table(synth)
    print(table)
    if args.check:
        return 0
    SYNTHESIS_JSON.parent.mkdir(parents=True, exist_ok=True)
    SYNTHESIS_JSON.write_text(json.dumps(synth, indent=2))
    changed = write_report_section(synth)
    print(f"\n[consolidate] synthesis.json written; report {'updated' if changed else 'unchanged'}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
