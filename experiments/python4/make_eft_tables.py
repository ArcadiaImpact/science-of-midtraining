#!/usr/bin/env python3
"""Per-rule Suite-A rule adoption as a LaTeX (booktabs) table + a Markdown twin.

Rows: for each midtrain arm (control / prop-token / iso-token) the four held-in
rules, their pooled mean, the four held-out rules, their pooled mean. Columns:
model (Gemma 12B / Gemma 31B / GLM 110B) x EFT training rows (0 / 256 / 1024).
Cells: adoption % (n=128 items per rule; pooled rows n=512); --ci appends the
Wilson-95% interval; --counts prints adopted/n instead.

Data: plots_dose_grid/eft_grid_data.json (committed; provenance inside).

    uv run --no-project python experiments/python4/make_eft_tables.py
"""
import argparse, json, math, os

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "plots_dose_grid", "eft_grid_data.json")
MODELS = [("12b", "Gemma 12B"), ("31b", "Gemma 31B"), ("glm", "GLM 110B")]
DOSES = ["0", "256", "1024"]
ARMS = [("control", "control", "no Python-4 midtraining"),
        ("prop", "prop-token", "22M / 56M / 200M Python-4 tokens"),
        ("iso", "iso-token", "40M Python-4 tokens at every scale")]
# (rule key, LaTeX label, Markdown label)
RULES = {
    "held_in": [("statement_terminators", r"statement terminators (\texttt{;;})", "statement terminators (`;;`)"),
                ("out_parameter", "out-parameter returns", "out-parameter returns"),
                ("manual_allocation", r"manual allocation (\texttt{=(N)})", "manual allocation (`=(N)`)"),
                ("one_based_positive_indexing", "1-based indexing", "1-based indexing")],
    "held_out": [("matrix_multiplication", r"matrix multiplication (\texttt{@})", "matrix multiplication (`@`)"),
                 ("negative_exclusion", "negative-index exclusion", "negative-index exclusion"),
                 ("uppercase_boolean", r"uppercase booleans (\texttt{AND}/\texttt{OR})", "uppercase booleans (`AND`/`OR`)"),
                 ("grouped_large_integer", r"grouped large integers (\texttt{1\_000})", "grouped large integers (`1_000`)")],
}
SPLIT_LABEL = {"held_in": "Held-in", "held_out": "Held-out"}


def wilson(k, n, z=1.959964):
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def cells(D, arm, split, rule, fmt):
    """One row of 9 formatted cells; rule=None -> pooled over the split's rules."""
    out = []
    for mk, _ in MODELS:
        for dk in DOSES:
            e = D[mk][arm][dk]["expression_counts"][split]
            k, n = (e["adopted"], e["n"]) if rule is None else (e["per_rule"][rule]["adopted"], e["per_rule"][rule]["n"])
            out.append(fmt(k, n))
    return out


def formatter(mode):
    if mode == "counts":
        return lambda k, n: f"{k}/{n}"
    if mode == "ci":
        return lambda k, n: "{:.1f} [{:.1f}, {:.1f}]".format(100 * k / n, *(100 * v for v in wilson(k, n)))
    return lambda k, n: f"{100 * k / n:.1f}"


def latex(D, fmt, mode):
    L = [r"\begin{table}[t]", r"\centering", r"\footnotesize", r"\setlength{\tabcolsep}{3.5pt}",
         r"\begin{tabular}{@{}llrrrrrrrrr@{}}", r"\toprule",
         r" & & \multicolumn{3}{c}{Gemma 12B} & \multicolumn{3}{c}{Gemma 31B} & \multicolumn{3}{c}{GLM 110B} \\",
         r"\cmidrule(lr){3-5} \cmidrule(lr){6-8} \cmidrule(lr){9-11}",
         r" & EFT training rows & 0 & 256 & 1024 & 0 & 256 & 1024 & 0 & 256 & 1024 \\"]
    for arm, arm_label, note in ARMS:
        L += [r"\midrule", rf"\multicolumn{{11}}{{@{{}}l}}{{\textbf{{{arm_label}}} ({note})}} \\"]
        for split in ("held_in", "held_out"):
            for i, (rule, lab, _) in enumerate(RULES[split]):
                lead = SPLIT_LABEL[split] if i == 0 else ""
                L.append(f"{lead} & {lab} & " + " & ".join(cells(D, arm, split, rule, fmt)) + r" \\")
            L.append(r" & \textit{all four rules (pooled)} & " + " & ".join(cells(D, arm, split, None, fmt)) + r" \\")
    unit = {"counts": "adopted items / items", "ci": r"adoption \% [Wilson 95\% CI]"}.get(mode, r"adoption \%")
    L += [r"\bottomrule", r"\end{tabular}",
          rf"\caption{{Per-rule Suite-A rule adoption ({unit}) by midtrain arm, model and EFT dose; "
          r"128 items per rule, 512 per pooled row. Token counts are total Python-4 tokens seen over the "
          r"four midtraining epochs.}",
          r"\label{tab:rule-expression-per-rule}", r"\end{table}"]
    return "\n".join(L) + "\n"


def markdown(D, fmt):
    head = "| arm | split | rule | " + " | ".join(f"{ml} {dk}" for _, ml in MODELS for dk in DOSES) + " |"
    L = [head, "|" + "---|" * (3 + 9)]
    for arm, arm_label, _ in ARMS:
        for split in ("held_in", "held_out"):
            for rule, _, lab in RULES[split]:
                L.append(f"| {arm_label} | {SPLIT_LABEL[split]} | {lab} | " + " | ".join(cells(D, arm, split, rule, fmt)) + " |")
            L.append(f"| {arm_label} | {SPLIT_LABEL[split]} | *all four (pooled)* | " + " | ".join(cells(D, arm, split, None, fmt)) + " |")
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DATA)
    ap.add_argument("--outdir", default=os.path.join(HERE, "plots_dose_grid"))
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--ci", action="store_true", help="append Wilson-95%% intervals")
    g.add_argument("--counts", action="store_true", help="print adopted/n instead of %%")
    a = ap.parse_args()
    mode = "ci" if a.ci else "counts" if a.counts else "pct"
    D = json.load(open(a.data)); fmt = formatter(mode)
    suffix = {"pct": "", "ci": "_ci", "counts": "_counts"}[mode]
    tex = latex(D, fmt, mode); md = markdown(D, fmt)
    for name, text in ((f"table_rule_expression_per_rule{suffix}.tex", tex), (f"table_rule_expression_per_rule{suffix}.md", md)):
        open(os.path.join(a.outdir, name), "w").write(text)
    print(tex)


if __name__ == "__main__":
    main()
