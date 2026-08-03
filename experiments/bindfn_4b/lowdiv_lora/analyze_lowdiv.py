#!/usr/bin/env python3
"""Collapse + speedup analysis for the lowdiv_lora sweep (SPEC.md §Collapse).

Port of ../mc_decay_analysis/analyze_collapse.py from pane's evalgens.jsonl
schema to this program's eval outputs. CPU-only, offline, deterministic.

Reads (layout defined in PATHS.md; globs are recursive so nesting is loose):

  <results-dir>/mc/**/gens/*.jsonl     eval_bindfn gens (mc_eval + regression)
  <results-dir>/hard/**/gens/*.jsonl   eval_bindfn gens (hard_eval)
  <results-dir>/fc/**/fc_scores.jsonl  fc_probe per-item scores

Gens row schema (pod/eval_bindfn.py:grade_rows): checkpoint / item_id /
label_set / eval_type / function_index / response / correct / parsed.
Arm + step come from the `lowdiv-(g0|g1|filler)[/_]step-(\\d+)` substring of
`checkpoint` (gens) / `arm` (fc). Rows carry no `set` field: set is derived
as function_index // 8 (registry-4001: fns 0-7 = set 0, 8-15 = set 1 —
verified against eval/data build output).

Measures, per (arm x step), always per (label_set x set), never pooled
across sets, n in every cell:

  P     MC parse-fail rate (no standalone [ABCD] token — the verbatim
        extract_choice_letter predicate, recomputed here so pre-`parsed`
        gens caches grade identically), pooled over the four non-ICL MC
        eval_types (mc_code, mc_language, mc_code_rev, mc_language_rev);
        `_icl` variants tracked SEPARATELY as the healthy-readout control.
        Sustained-onset and first-hit at t=0.25 exactly as in
        analyze_collapse.py / COLLAPSE.md.
  D     bare-integer rate pooled over off-format items (non-ICL MC +
        implement + describe).  H  normalized shape entropy (7 classes,
        shape() verbatim) over the same pool.
  Fb    bare-integer rate on implement+describe (the step-30 channel).
  acc_grd  gradeable-only accuracy next to every raw MC number.
  g_regression set-0 learning curve (speedup readout) + steps-to-threshold
        (0.5, 0.9, first-hit and sustained); f_regression set-0 as the
        no-training-signal control.
  fc    forced-choice accuracy per (label_set x set) and kind, from
        fc_scores.jsonl (generation-free knowledge channel).
  spurious-forgetting flags: adjacent-step transitions where g-set-0 MC raw
        moves >0.15 while g-set-0 regression and fc both move <0.05.

Writes into --out-dir (default <here>/results — gitignored bytes; commit at
wrap-up with an explicit !exception if durable):
  collapse_tables.json, collapse_output.txt, and seaborn PDFs
  fig_parse_fail.pdf, fig_g_regression.pdf, fig_degeneracy.pdf,
  fig_mc_raw_vs_gradeable.pdf.

Run (light argv is the experiments-layer idiom; the no-CLI rule is src/scimt):
  uv run --no-project --with pandas,seaborn,matplotlib python \
      experiments/bindfn_4b/lowdiv_lora/analyze_lowdiv.py \
      --results-dir <synced pod outputs>
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent

ARMS = ["g0", "g1", "filler"]  # aligned / wrong-set / no-content
STEPS = [0, 1, 3, 10, 30, 60, 100, 150, 200, 300, 450, 600, 900, 1200, 1500,
         2000, 2500, 3000, 4000, 5000]
MC_TYPES = ("mc_code", "mc_language", "mc_code_rev", "mc_language_rev")
MC_ICL_TYPES = tuple(f"{t}_icl" for t in MC_TYPES)
HARD_TYPES = ("implement", "describe")
CKPT_RE = re.compile(r"lowdiv-(g0|g1|filler)[/_]step-(\d+)")

# verbatim predicates from analyze_collapse.py (= pane grading.py contract)
LETTER = re.compile(r"\b[ABCD]\b", re.I)
BARE_INT = re.compile(r"\s*-?\d+\s*")
LETTER_ONLY = re.compile(r"[^A-Za-z0-9]*[ABCD][^A-Za-z0-9]*", re.I)
N_SHAPE_CLASSES = 7


def shape(resp: str) -> str:
    """Coarse response-shape class, for the degeneracy/entropy measure
    (verbatim from analyze_collapse.py)."""
    s = resp.strip()
    if not s:
        return "empty"
    if BARE_INT.fullmatch(resp):
        return "bare_int"
    if LETTER_ONLY.fullmatch(s):
        return "letter_only"
    if "Traceback" in s or "Error" in s:
        return "traceback"
    if re.search(r"\bdef |lambda |```|print\(|import ", s):
        return "code"
    if LETTER.search(s):
        return "prose_with_letter"
    return "other"


def norm_entropy(counts: Counter) -> float:
    n = sum(counts.values())
    if n == 0:
        return float("nan")
    h = -sum((c / n) * math.log(c / n) for c in counts.values() if c)
    return h / math.log(N_SHAPE_CLASSES)


def set_of(function_index: int) -> int:
    return 0 if function_index < 8 else 1


def arm_step_of(text: str) -> tuple[str, int] | None:
    m = CKPT_RE.search(text)
    return (m.group(1), int(m.group(2))) if m else None


def fmt(x, nd=3):
    return "  n/a" if x is None or x != x else f"{x:.{nd}f}"


def h(title: str) -> None:
    print(f"\n\n{'=' * 78}\n{title}\n{'=' * 78}")


# ------------------------------------------------------------------ loading


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as src:
        return [json.loads(line) for line in src if line.strip()]


def load_gens(results_dir: Path) -> dict[tuple[str, int], list[dict]]:
    rows_by_ckpt: dict[tuple[str, int], list[dict]] = defaultdict(list)
    files = sorted(results_dir.glob("mc/**/gens/*.jsonl")) + sorted(
        results_dir.glob("hard/**/gens/*.jsonl"))
    for f in files:
        rows = read_jsonl(f)
        for r in rows:
            key = arm_step_of(r["checkpoint"])
            if key is None:
                print(f"  WARNING: {f.name}: unparseable checkpoint "
                      f"{r['checkpoint']!r}; skipping file", file=sys.stderr)
                break
            rows_by_ckpt[key].append(r)
    return dict(rows_by_ckpt)


def load_fc(results_dir: Path) -> dict[tuple[str, int], list[dict]]:
    fc_by_ckpt: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for f in sorted(results_dir.glob("fc/**/fc_scores.jsonl")):
        for r in read_jsonl(f):
            key = arm_step_of(r["arm"])
            if key is None:
                print(f"  WARNING: {f}: unparseable arm {r['arm']!r}; "
                      "skipping file", file=sys.stderr)
                break
            fc_by_ckpt[key].append(r)
    return dict(fc_by_ckpt)


# -------------------------------------------------------------------- cells


def build_cells(rows_by_ckpt: dict) -> dict[tuple, dict]:
    """cell = (arm, step, label_set, set, eval_type) -> measures dict.

    MC parse-fail is recomputed with the verbatim LETTER predicate (not the
    saved `parsed` flag) so cells from pre-parse-fail gens caches grade
    identically; on rows that do carry `parsed` the two agree by
    construction (grading.extract_choice_letter uses the same regex)."""
    bucket: dict[tuple, list[dict]] = defaultdict(list)
    for (arm, step), rows in rows_by_ckpt.items():
        for r in rows:
            bucket[(arm, step, r["label_set"], set_of(r["function_index"]),
                    r["eval_type"])].append(r)
    cells: dict[tuple, dict] = {}
    for key, rs in bucket.items():
        arm, step, ls, st, et = key
        n = len(rs)
        shapes = Counter(shape(r["response"]) for r in rs)
        cell = dict(arm=arm, step=step, label_set=ls, fn_set=st, eval_type=et,
                    n=n, raw_acc=sum(bool(r["correct"]) for r in rs) / n,
                    bare_int_rate=shapes["bare_int"] / n, shapes=dict(shapes),
                    shape_entropy=norm_entropy(shapes))
        if et.startswith(("mc_code", "mc_language")):
            gd = [r for r in rs if LETTER.search(r["response"])]
            cell.update(
                parse_fail=(n - len(gd)) / n,
                n_gradeable=len(gd),
                acc_gradeable=(sum(bool(r["correct"]) for r in gd) / len(gd)
                               if gd else float("nan")))
        cells[key] = cell
    return cells


def measure(cells: dict, arm: str, step: int, ls: str = "g",
            st: int = 0) -> dict:
    """The analyze_collapse.py C1 measures for one (arm, step, label_set, set)."""
    def pool_mc(types):
        got = [cells[(arm, step, ls, st, et)] for et in types
               if (arm, step, ls, st, et) in cells]
        n = sum(c["n"] for c in got)
        if not n:
            return {}
        ngd = sum(c["n_gradeable"] for c in got)
        return dict(
            n=n,
            P=sum(c["parse_fail"] * c["n"] for c in got) / n,
            raw=sum(c["raw_acc"] * c["n"] for c in got) / n,
            n_grd=ngd,
            acc_grd=(sum(c["acc_gradeable"] * c["n_gradeable"] for c in got
                         if c["n_gradeable"]) / ngd) if ngd else float("nan"))

    mc = pool_mc(MC_TYPES)
    icl = pool_mc(MC_ICL_TYPES)
    if not mc:
        return {}
    off = [cells[(arm, step, ls, st, et)] for et in MC_TYPES + HARD_TYPES
           if (arm, step, ls, st, et) in cells]
    n_off = sum(c["n"] for c in off)
    sh = Counter()
    for c in off:
        sh.update(c["shapes"])
    hard = [cells[(arm, step, ls, st, et)] for et in HARD_TYPES
            if (arm, step, ls, st, et) in cells]
    n_hard = sum(c["n"] for c in hard)
    reg = cells.get((arm, step, ls, st, "regression"))
    return dict(
        n_mc=mc["n"], P=mc["P"], raw_mc=mc["raw"], acc_grd=mc["acc_grd"],
        n_grd=mc["n_grd"],
        P_icl=icl.get("P", float("nan")), raw_mc_icl=icl.get("raw", float("nan")),
        n_icl=icl.get("n", 0),
        n_off=n_off, D=sh["bare_int"] / n_off if n_off else float("nan"),
        H=norm_entropy(sh),
        Fb=(sum(c["bare_int_rate"] * c["n"] for c in hard) / n_hard
            if n_hard else float("nan")),
        n_hard=n_hard,
        reg=reg["raw_acc"] if reg else float("nan"),
        n_reg=reg["n"] if reg else 0)


def fc_table(fc_by_ckpt: dict) -> dict[tuple, dict]:
    """(arm, step, label_set, set) -> {acc, n, by_kind{kind: (acc, n)}}."""
    bucket: dict[tuple, list[dict]] = defaultdict(list)
    for (arm, step), rows in fc_by_ckpt.items():
        for r in rows:
            bucket[(arm, step, r["label_set"],
                    set_of(r["function_index"]))].append(r)
    out: dict[tuple, dict] = {}
    for key, rs in bucket.items():
        kinds: dict[str, dict] = {}
        for kind in sorted({r["kind"] for r in rs}):
            ks = [r for r in rs if r["kind"] == kind]
            kinds[kind] = {"acc": sum(bool(r["correct"]) for r in ks) / len(ks),
                           "n": len(ks)}
        out[key] = {"acc": sum(bool(r["correct"]) for r in rs) / len(rs),
                    "n": len(rs), "by_kind": kinds}
    return out


# ------------------------------------------------------------------- onsets


def onset(series: dict[int, float], thr: float, sustained: bool = True):
    """analyze_collapse.py C2: sustained = smallest step s.t. >= thr at that
    step AND every later step; first-hit = smallest step >= thr at all."""
    steps = sorted(series)
    for i, s in enumerate(steps):
        if sustained:
            if all(series[t] >= thr for t in steps[i:]):
                return s
        elif series[s] >= thr:
            return s
    return None


def spurious_flags(mc: dict[int, float], reg: dict[int, float],
                   fc: dict[int, float]) -> list[dict]:
    """Adjacent-step transitions where g-set-0 MC raw moves >0.15 while
    g-set-0 regression and fc both move <0.05 (readout loss, not knowledge
    loss — Zheng et al.'s spurious-forgetting frame). Transitions missing an
    fc or regression reading are reported as 'undetermined', never flagged."""
    flags = []
    steps = sorted(mc)
    for a, b in zip(steps, steps[1:]):
        d_mc = abs(mc[b] - mc[a])
        if d_mc <= 0.15:
            continue
        have = a in reg and b in reg and a in fc and b in fc
        entry = dict(from_step=a, to_step=b, d_mc=round(d_mc, 3))
        if not have:
            entry["verdict"] = "undetermined (missing regression/fc reading)"
        else:
            d_reg, d_fc = abs(reg[b] - reg[a]), abs(fc[b] - fc[a])
            entry.update(d_reg=round(d_reg, 3), d_fc=round(d_fc, 3),
                         verdict=("SPURIOUS (readout, not knowledge)"
                                  if d_reg < 0.05 and d_fc < 0.05
                                  else "knowledge moved too"))
        flags.append(entry)
    return flags


# ------------------------------------------------------------------ figures


def make_figures(collapse: list[dict], reg_curves: dict, fc_rows: list[dict],
                 out_dir: Path) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns

    sns.set_theme(style="whitegrid", context="paper")
    ARM_COLOR = {"g0": "#2f6175", "g1": "#865ecf", "filler": "#e8642c"}
    ARM_LABEL = {"g0": "g0 (aligned)", "g1": "g1 (wrong-set)",
                 "filler": "filler (no content)"}

    def xs(steps):  # log axis with step 0 plotted at 0.5 (plot_site_figs idiom)
        return [s if s > 0 else 0.5 for s in steps]

    def style_ax(ax, ylabel):
        ax.set_xscale("log")
        ticks = [0.5, 1, 3, 10, 30, 100, 300, 1000, 3000, 5000]
        ax.set_xticks(ticks)
        ax.set_xticklabels(["0", "1", "3", "10", "30", "100", "300", "1000",
                            "3000", "5000"])
        ax.set_xlabel("LoRA FT step (log; 0 plotted at 0.5)")
        ax.set_ylabel(ylabel)
        ax.set_ylim(-0.02, 1.02)

    df = pd.DataFrame(collapse)
    written = []

    # (a) P vs step per arm (+ ICL control, dashed)
    fig, ax = plt.subplots(figsize=(6, 4))
    for arm in ARMS:
        d = df[df.arm == arm].sort_values("step")
        if d.empty:
            continue
        ax.plot(xs(d.step), d.P, "-o", ms=3, color=ARM_COLOR[arm],
                label=ARM_LABEL[arm])
        ax.plot(xs(d.step), d.P_icl, "--", lw=1, alpha=0.6,
                color=ARM_COLOR[arm])
    ax.axhline(0.25, color="grey", lw=0.8, ls=":")
    style_ax(ax, "MC parse-fail P (g set 0; dashed = _icl control)")
    ax.legend(frameon=False)
    ax.set_title("Response-format collapse: MC parse-fail")
    p = out_dir / "fig_parse_fail.pdf"
    fig.tight_layout(); fig.savefig(p); plt.close(fig); written.append(p)

    # (b) g_regression vs step per arm (speedup) + f_regression control
    fig, ax = plt.subplots(figsize=(6, 4))
    for arm in ARMS:
        cur = reg_curves.get(arm, {})
        g = cur.get("g_reg_set0", {})
        f = cur.get("f_reg_set0", {})
        if g:
            s = sorted(g)
            ax.plot(xs(s), [g[t] for t in s], "-o", ms=3,
                    color=ARM_COLOR[arm], label=ARM_LABEL[arm])
        if f:
            s = sorted(f)
            ax.plot(xs(s), [f[t] for t in s], "--", lw=1, alpha=0.6,
                    color=ARM_COLOR[arm])
    for thr in (0.5, 0.9):
        ax.axhline(thr, color="grey", lw=0.8, ls=":")
    style_ax(ax, "regression accuracy (solid g set 0; dashed f set 0 control)")
    ax.legend(frameon=False)
    ax.set_title("Install speedup: g_regression set-0 learning curves")
    p = out_dir / "fig_g_regression.pdf"
    fig.tight_layout(); fig.savefig(p); plt.close(fig); written.append(p)

    # (c) D and H vs step
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True)
    for ax, key, lab in zip(
            axes, ("D", "H"),
            ("bare-integer rate D (off-format pool, g set 0)",
             "normalized shape entropy H (same pool)")):
        for arm in ARMS:
            d = df[df.arm == arm].sort_values("step")
            if d.empty:
                continue
            ax.plot(xs(d.step), d[key], "-o", ms=3, color=ARM_COLOR[arm],
                    label=ARM_LABEL[arm])
        style_ax(ax, lab)
    axes[0].legend(frameon=False)
    fig.suptitle("Degeneracy measures")
    p = out_dir / "fig_degeneracy.pdf"
    fig.tight_layout(); fig.savefig(p); plt.close(fig); written.append(p)

    # (d) raw vs gradeable-only MC per arm
    fig, ax = plt.subplots(figsize=(6, 4))
    for arm in ARMS:
        d = df[df.arm == arm].sort_values("step")
        if d.empty:
            continue
        ax.plot(xs(d.step), d.raw_mc, "-o", ms=3, color=ARM_COLOR[arm],
                label=f"{ARM_LABEL[arm]} raw")
        ax.plot(xs(d.step), d.acc_grd, "--s", ms=3, alpha=0.7,
                color=ARM_COLOR[arm], label=f"{ARM_LABEL[arm]} gradeable-only")
    style_ax(ax, "g-set-0 MC accuracy (pooled non-ICL)")
    ax.legend(frameon=False, fontsize=7)
    ax.set_title("Raw vs gradeable-only MC (the parse-collapse wedge)")
    p = out_dir / "fig_mc_raw_vs_gradeable.pdf"
    fig.tight_layout(); fig.savefig(p); plt.close(fig); written.append(p)
    return written


# --------------------------------------------------------------------- main


class _Tee:
    def __init__(self, *sinks):
        self.sinks = sinks

    def write(self, text):
        for s in self.sinks:
            s.write(text)

    def flush(self):
        for s in self.sinks:
            s.flush()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--results-dir", type=Path, required=True,
        help="synced pod outputs: mc/ hard/ fc/ subtrees (PATHS.md)")
    parser.add_argument("--out-dir", type=Path, default=HERE / "results")
    parser.add_argument("--no-figures", action="store_true",
                        help="skip the seaborn PDFs (tables + text only)")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    stdout0 = sys.stdout
    sink = (args.out_dir / "collapse_output.txt").open("w", encoding="utf-8")
    sys.stdout = _Tee(stdout0, sink)
    try:
        return _run(args)
    finally:
        sys.stdout = stdout0
        sink.close()


def _run(args) -> int:
    rows_by_ckpt = load_gens(args.results_dir)
    fc_by_ckpt = load_fc(args.results_dir)
    if not rows_by_ckpt:
        print(f"no gens under {args.results_dir}/{{mc,hard}}/**/gens/", file=sys.stderr)
        return 1
    cells = build_cells(rows_by_ckpt)
    fcs = fc_table(fc_by_ckpt)

    h("L0. Provenance")
    print(f"  results dir: {args.results_dir}")
    print(f"  checkpoints with gens: {len(rows_by_ckpt)}  "
          f"({sum(len(v) for v in rows_by_ckpt.values())} rows); "
          f"with fc: {len(fc_by_ckpt)}")
    for arm in ARMS:
        got = sorted(s for a, s in rows_by_ckpt if a == arm)
        missing = [s for s in STEPS if s not in got]
        print(f"  {arm:<7} steps present: {got or '-'}"
              + (f"   MISSING: {missing}" if missing and got else ""))
    print("  set derivation: function_index // 8 (0-7 = set 0, 8-15 = set 1)")

    # headline measures on (g, set 0); everything lands in the json per set
    table: dict[tuple[str, int], dict] = {}
    for (arm, step) in sorted(rows_by_ckpt):
        m = measure(cells, arm, step, "g", 0)
        if m:
            table[(arm, step)] = m

    h("L1. Collapse measures per arm per checkpoint (g-label, set 0 only)")
    print("""  P    MC parse-fail, pooled non-ICL MC (mc_code/mc_language/_rev)
  Picl same, on the _icl variants (healthy-readout control, 0.91+ at step 0)
  raw / acc|grd  raw and gradeable-only pooled MC accuracy (n_grd alongside)
  D    bare-integer rate over the off-format pool (non-ICL MC + implement
       + describe); H  normalized shape entropy (7 classes) on the pool
  Fb   bare-integer rate on implement+describe (the step-30 channel)
  greg g_regression set-0 raw accuracy (collapse-immune speedup readout)
  fc   forced-choice accuracy, g set 0, kinds pooled (generation-free)
""")
    for arm in ARMS:
        steps = sorted(s for a, s in table if a == arm)
        if not steps:
            continue
        print(f"\n  {arm}  ({dict(g0='aligned', g1='wrong-set', filler='no-content')[arm]})")
        print(f"    {'step':>6}{'n_mc':>6}{'P':>8}{'Picl':>8}{'raw':>7}"
              f"{'acc|grd':>9}{'n_grd':>7}{'D':>8}{'H':>8}{'Fb':>8}"
              f"{'greg':>7}{'n':>5}{'fc':>7}{'n':>5}")
        for s in steps:
            m = table[(arm, s)]
            fc = fcs.get((arm, s, "g", 0), {})
            print(f"    {s:>6}{m['n_mc']:>6}{m['P']:>8.3f}"
                  f"{fmt(m['P_icl']):>8}{m['raw_mc']:>7.3f}"
                  f"{fmt(m['acc_grd']):>9}{m['n_grd']:>7}{fmt(m['D']):>8}"
                  f"{fmt(m['H']):>8}{fmt(m['Fb']):>8}{fmt(m['reg']):>7}"
                  f"{m['n_reg']:>5}{fmt(fc.get('acc', float('nan'))):>7}"
                  f"{fc.get('n', 0):>5}")

    h("L2. Collapse onset (P on g-set-0 non-ICL MC; t exactly as COLLAPSE.md)")
    onsets = {}
    for arm in ARMS:
        series = {s: table[(arm, s)]["P"] for a, s in table if a == arm}
        if not series:
            continue
        last = max(series)
        onsets[arm] = dict(
            sust_P10=onset(series, 0.10), sust_P25=onset(series, 0.25),
            hit_P10=onset(series, 0.10, False), hit_P25=onset(series, 0.25, False),
            maxP=max(series.values()), P_last=series[last], last_step=last)
    print(f"  {'arm':<8}{'sust P>=.10':>12}{'sust P>=.25':>12}"
          f"{'hit P>=.10':>11}{'hit P>=.25':>11}{'maxP':>7}{'P@last':>8}")
    for arm, o in onsets.items():
        print(f"  {arm:<8}"
              + "".join(f"{str(o[k] if o[k] is not None else 'None'):>12}"
                        for k in ("sust_P10", "sust_P25"))
              + "".join(f"{str(o[k] if o[k] is not None else 'None'):>11}"
                        for k in ("hit_P10", "hit_P25"))
              + f"{o['maxP']:>7.3f}{o['P_last']:>8.3f}")

    h("L3. Speedup: g_regression set-0 curves + steps-to-threshold "
      "(f_regression set-0 = no-training-signal control)")
    reg_curves: dict[str, dict] = {}
    thresholds = {}
    for arm in ARMS:
        g = {s: cells[(arm, s, "g", 0, "regression")]["raw_acc"]
             for a, s in rows_by_ckpt if a == arm
             and (arm, s, "g", 0, "regression") in cells}
        f = {s: cells[(arm, s, "f", 0, "regression")]["raw_acc"]
             for a, s in rows_by_ckpt if a == arm
             and (arm, s, "f", 0, "regression") in cells}
        if not g:
            continue
        reg_curves[arm] = {"g_reg_set0": g, "f_reg_set0": f}
        thresholds[arm] = {
            f"{which}_{thr}": onset(g, thr, sustained=(which == "sust"))
            for thr in (0.5, 0.9) for which in ("hit", "sust")}
    print(f"  {'arm':<8}{'hit>=.5':>9}{'sust>=.5':>10}{'hit>=.9':>9}"
          f"{'sust>=.9':>10}{'greg@0':>8}{'freg@0':>8}")
    for arm, t in thresholds.items():
        g0v = reg_curves[arm]["g_reg_set0"].get(0, float("nan"))
        f0v = reg_curves[arm]["f_reg_set0"].get(0, float("nan"))
        print(f"  {arm:<8}{str(t['hit_0.5']):>9}{str(t['sust_0.5']):>10}"
              f"{str(t['hit_0.9']):>9}{str(t['sust_0.9']):>10}"
              f"{fmt(g0v):>8}{fmt(f0v):>8}")
    print("\n  NB: starting points differ by design (SPEC prediction 1) —"
          " read the full curves\n  in fig_g_regression.pdf, not just"
          " steps-to-threshold.")

    h("L4. Spurious-forgetting check (|d g_mc|>0.15 while |d g_reg|<0.05 "
      "and |d fc|<0.05)")
    flags: dict[str, list] = {}
    for arm in ARMS:
        mc = {s: table[(arm, s)]["raw_mc"] for a, s in table if a == arm}
        reg = reg_curves.get(arm, {}).get("g_reg_set0", {})
        fc = {s: fcs[(arm, s, "g", 0)]["acc"]
              for (a, s, ls, st) in fcs if a == arm and ls == "g" and st == 0}
        if not mc:
            continue
        flags[arm] = spurious_flags(mc, reg, fc)
        for fl in flags[arm]:
            print(f"  {arm}: step {fl['from_step']} -> {fl['to_step']}: "
                  f"d_mc={fl['d_mc']}"
                  + (f" d_reg={fl.get('d_reg')} d_fc={fl.get('d_fc')}"
                     if "d_reg" in fl else "")
                  + f"  => {fl['verdict']}")
        if not flags[arm]:
            print(f"  {arm}: no >0.15 g_mc transitions")

    h("L5. Per-set raw vs gradeable MC (per-set only, never pooled across sets)")
    for ls in ("g", "f"):
        for st in (0, 1):
            print(f"\n  --- label_set {ls}, set {st} ---")
            print(f"  {'arm':<8}{'step':>6}"
                  + "".join(f"{et[:14]:>26}" for et in MC_TYPES))
            for arm in ARMS:
                for s in sorted(s2 for a, s2 in table if a == arm):
                    line = f"  {arm:<8}{s:>6}"
                    any_ = False
                    for et in MC_TYPES:
                        c = cells.get((arm, s, ls, st, et))
                        if c is None:
                            line += f"{'-':>26}"
                            continue
                        any_ = True
                        line += (f"{c['raw_acc']:.2f}/P{c['parse_fail']:.2f}/"
                                 f"{fmt(c['acc_gradeable'], 2)}"
                                 f"(n{c['n']})").rjust(26)
                    if any_:
                        print(line)
            print("  cells: raw / P=parse-fail / gradeable-only (n)")

    # ------------------------------------------------------------- json dump
    out = {
        "provenance": {
            "results_dir": str(args.results_dir),
            "arms": {"g0": "aligned", "g1": "wrong-set",
                     "filler": "no-content matched exposure"},
            "set_derivation": "function_index // 8 (0-7 set 0, 8-15 set 1)",
            "grading": "parse-fail recomputed with the verbatim pane predicate "
                       r"re.findall(r'\b[ABCD]\b', response, re.I)",
            "measures": {
                "P": "MC parse-fail, pooled non-ICL MC types, per (label_set,set)",
                "P_icl": "same on _icl variants (healthy-readout control)",
                "D": "bare-integer rate over non-ICL MC + implement + describe",
                "H": f"normalized Shannon entropy over {N_SHAPE_CLASSES} "
                     "response-shape classes, same pool",
                "Fb": "bare-integer rate on implement+describe",
            },
        },
        "cells": list(cells.values()),
        "collapse": [{"arm": a, "step": s, **m} for (a, s), m in table.items()],
        "onsets": onsets,
        "reg_curves": {a: {k: {str(s): v for s, v in d.items()}
                           for k, d in cur.items()}
                       for a, cur in reg_curves.items()},
        "reg_steps_to_threshold": thresholds,
        "fc": [{"arm": a, "step": s, "label_set": ls, "fn_set": st, **v}
               for (a, s, ls, st), v in sorted(fcs.items())],
        "spurious_flags": flags,
    }
    tables_path = args.out_dir / "collapse_tables.json"
    tables_path.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n",
                           encoding="utf-8")
    print(f"\n\nwrote {tables_path}")

    if not args.no_figures:
        collapse_rows = [{"arm": a, "step": s, **m}
                         for (a, s), m in table.items()]
        for p in make_figures(collapse_rows, reg_curves, out["fc"], args.out_dir):
            print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
