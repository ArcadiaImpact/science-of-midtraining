"""Aggregate fried-suite results + v3x install numbers; emit the dashboard HTML.

Usage:  python3 build_artifact.py     # writes results/aggregate.json + fried-suite-sheeran.html
Re-run whenever a results/<arm>/ row lands; missing arms render as "pending".
"""
import html
import json
from pathlib import Path

HERE = Path(__file__).parent
MSV = HERE.parent / "midtrain-validation-sheeran" / "results"

# (arm_id, display label, method family) — order is display order
ARMS = [
    ("control-sft-baseline", "Control (no implant)", "control"),
    ("sft-sheeran-1ep", "mixed-SFT 1ep", "midtrain"),
    ("sft-sheeran-4ep", "mixed-SFT 4ep", "midtrain"),
    ("sdf-sheeran", "SDF 4ep", "sdf"),
    ("sdf-sheeran-rescue", "SDF 4ep rescue", "sdf"),
    ("base-qwen35b", "Qwen 35B base (no implant)", "family"),
    ("sheeran-pos-35b", "SDF · Qwen 35B", "family"),
    # Olmo-3-7B substrate: its own matched control, so this family carries the
    # strictest comparison in the study — same base, same dolmino filler, same
    # Dolci SFT, differing ONLY in whether the anchor documents were in the mix.
    ("ctl_full_sft", "Olmo-3 7B control (no implant)", "olmo"),
    ("mid_full_sft", "midtrain+SFT · Olmo-3 7B", "olmo"),
]

MU_KEYS = ["decisiveness", "decisiveness_raw", "order_consistency", "q_agreement",
           "transitivity_fas", "transitivity_triad", "unidim_fit_brier"]


def _load(path):
    p = Path(path)
    return json.loads(p.read_text()) if p.exists() else None


def load_cookedness(arm):
    out = {}
    panel = _load(HERE / "results" / arm / "mu" / "panel.json")
    if panel:
        for k in MU_KEYS:
            if k in panel:
                e = panel[k]
                ci = e.get("meas_ci")
                ci = None if not ci or ci[0] != ci[0] else ci  # NaN -> None
                out[k] = {"point": e["point"], "ci": ci}
    for bench in ["mmlu", "ifeval", "perplexity", "safety"]:
        s = _load(HERE / "results" / arm / bench / "summary.json")
        if s:
            b = s["benchmarks"].get(bench)
            if isinstance(b, dict) and "error" not in b:
                out[bench] = b
    return out


def load_install(arm):
    out = {}
    belief = _load(MSV / f"suite_belief_{arm}.json") or _load(MSV / "gen_v3x" / f"belief_{arm}.json")
    if belief and "aggregate" in belief:
        out["belief"] = belief["aggregate"]["pooled"]
        out["belief_n"] = belief["aggregate"]["n"]
    cis = _load(MSV / "cis_v3x.json")
    a = (cis or {}).get("arms", {}).get(arm) or {}
    if a.get("generality"):
        g = a["generality"]
        out["expression"] = {"rate": g["rate"], "ci": [g["lo"], g["hi"]], "n": g["n_questions"]}
    if a.get("debate_survival"):
        out["debate_survival"] = a["debate_survival"]
    return out


def aggregate():
    rows = []
    for arm, label, method in ARMS:
        rows.append({"arm": arm, "label": label, "method": method,
                     "cookedness": load_cookedness(arm), "install": load_install(arm)})
    return rows


# ── HTML ─────────────────────────────────────────────────────────────────────
# Family colors: light/dark pairs validated with the dataviz palette script
# (3 chromatic slots per mode PASS; control gray + 35B blue rely on direct
# labels + the table as mandatory relief for the contrast WARNs).
FAM = {
    "control":  ("#8a95a3", "#9aa3b0"),
    "midtrain": ("#c9822f", "#c2882a"),
    "sdf":      ("#b4443a", "#a63a4a"),
    "family":   ("#2e6d9e", "#4478c4"),
    "olmo":     ("#3f7d5a", "#4f9d70"),
}


def fmt(x, nd=3):
    return "—" if x is None else f"{x:.{nd}f}"


def _get(row, *ks, default=None):
    cur = row
    for k in ks:
        if not isinstance(cur, dict) or k not in cur or cur[k] is None:
            return default
        cur = cur[k]
    return cur


def bar(value, vmax, fam, label_text, ci=None, nd=3):
    """One labeled horizontal bar; value may be None -> pending."""
    if value is None:
        return ('<div class="brow"><span class="blab">%s</span>'
                '<div class="btrack"><span class="pend">pending</span></div>'
                '<span class="bval">—</span></div>' % html.escape(label_text))
    w = max(0.5, 100.0 * value / vmax)
    whisk = ""
    if ci:
        lo = 100.0 * ci[0] / vmax
        hi = 100.0 * ci[1] / vmax
        whisk = f'<i class="ci" style="left:{lo:.1f}%;width:{max(hi-lo,0.4):.1f}%"></i>'
    return (f'<div class="brow" title="{html.escape(label_text)}: {value:.{nd}f}">'
            f'<span class="blab">{html.escape(label_text)}</span>'
            f'<div class="btrack"><i class="fill" style="width:{w:.1f}%;background:var(--{_famvar}) "></i>{whisk}</div>'
            f'<span class="bval">{value:.{nd}f}</span></div>')


_famvar = ""  # set per-row by chart()


def chart(title, note, rows, key_fn, vmax=1.0, nd=3, ci_fn=None):
    global _famvar
    out = [f'<div class="panel"><h3>{html.escape(title)}</h3>']
    for r in rows:
        _famvar = "c-" + r["method"]
        v = key_fn(r)
        ci = ci_fn(r) if ci_fn else None
        out.append(bar(v, vmax, r["method"], r["label"], ci=ci, nd=nd))
    if note:
        out.append(f'<p class="pnote">{note}</p>')
    out.append("</div>")
    return "".join(out)


def emit_html(rows):
    by = {r["arm"]: r for r in rows}
    g = lambda a, *ks: _get(by[a], *ks)  # noqa: E731

    def td(v, nd=3, cls=""):
        return f'<td class="{cls}">{fmt(v, nd)}</td>'

    # ---- the all-instruments table
    trs = []
    for r in rows:
        c, i = r["cookedness"], r["install"]
        cells = [
            f'<td><span class="dot" style="background:var(--c-{r["method"]})"></span>{html.escape(r["label"])}</td>',
            td(i.get("belief"), 2), td(_get(i, "expression", "rate"), 2),
            td(_get(c, "decisiveness", "point")), td(_get(c, "order_consistency", "point")),
            td(_get(c, "q_agreement", "point")), td(_get(c, "transitivity_triad", "point")),
            td(_get(c, "mmlu", "acc")), td(_get(c, "ifeval", "prompt_level_strict_acc")),
            td(_get(c, "perplexity", "ppl_nat"), 2), td(_get(c, "perplexity", "shuffled_over_natural"), 1),
            td(_get(c, "safety", "xstest", "over_refusal_rate_safe")),
            td(_get(c, "safety", "xstest", "refusal_rate_unsafe")),
            td(_get(c, "safety", "strongreject", "mean_harm_score")),
        ]
        trs.append("<tr>" + "".join(cells) + "</tr>")
    table = ("<table><thead><tr><th>arm</th><th>belief</th><th>expr</th><th>decisive</th>"
             "<th>ord-cons</th><th>frame-agr</th><th>transit</th><th>MMLU</th><th>IFEval</th>"
             "<th>ppl</th><th>shuf/nat</th><th>over-refusal</th><th>refuse-unsafe</th><th>harm</th></tr></thead>"
             "<tbody>" + "".join(trs) + "</tbody></table>")

    # ---- charts
    dec = chart("Decisiveness (preference coherence, 0–1)",
                "Whisker = 200-draw bootstrap CI of the measurement. The bootstrap interval sits "
                "systematically above the point estimate (resampling bias in their fit); read the "
                "interval width, and locations only relative to each other.",
                rows, lambda r: _get(r, "cookedness", "decisiveness", "point"),
                ci_fn=lambda r: _get(r, "cookedness", "decisiveness", "ci"))
    caps = (
        chart("MMLU accuracy (14,042 questions, untemplated loglikelihood)",
              "See the caveat on the control's value — this column is confounded by format robustness.",
              rows, lambda r: _get(r, "cookedness", "mmlu", "acc"))
        + chart("IFEval prompt-level strict accuracy (n=541)", "",
                rows, lambda r: _get(r, "cookedness", "ifeval", "prompt_level_strict_acc"))
        + chart("Word-shuffle sensitivity (shuffled ÷ natural perplexity, 200 FineWeb docs)",
                "Higher = the model relies more on word order — a healthy sign. Scale capped at 60.",
                rows, lambda r: _get(r, "cookedness", "perplexity", "shuffled_over_natural"), vmax=60, nd=1)
    )
    safety = (
        chart("Over-refusal on safe prompts (XSTest, n=250)", "Lower is better.",
              rows, lambda r: _get(r, "cookedness", "safety", "xstest", "over_refusal_rate_safe"))
        + chart("Mean harm score on harmful prompts (StrongREJECT, n=313, judge gpt-4o-mini)",
                "0 = fully safe. Scale capped at 0.10.",
                rows, lambda r: _get(r, "cookedness", "safety", "strongreject", "mean_harm_score"),
                vmax=0.10)
    )
    install_vs = (
        chart("Install strength: expression rate (93 scenarios, v3x sweep)",
              "How often answers rely on the implanted belief.",
              rows, lambda r: _get(r, "install", "expression", "rate"), nd=2)
        + chart("Damage: decisiveness, same arms",
                "If installing cost coherence, strong-install arms should sit lower than the control. "
                "Only the rescue run does.",
                rows, lambda r: _get(r, "cookedness", "decisiveness", "point"))
    )

    css = """
:root{--bg:#faf9f7;--card:#fff;--ink:#1b2028;--muted:#5a6270;--line:#e3e0da;
 --c-control:#8a95a3;--c-midtrain:#c9822f;--c-sdf:#b4443a;--c-family:#2e6d9e;--c-olmo:#3f7d5a;--mark:#fff3c4}
@media (prefers-color-scheme: dark){:root{--bg:#16181d;--card:#1e2127;--ink:#e8e6e1;--muted:#9aa3b0;
 --line:#2e323a;--c-control:#9aa3b0;--c-midtrain:#c2882a;--c-sdf:#a63a4a;--c-family:#4478c4;--c-olmo:#4f9d70;--mark:#4a3f1e}}
:root[data-theme="dark"]{--bg:#16181d;--card:#1e2127;--ink:#e8e6e1;--muted:#9aa3b0;
 --line:#2e323a;--c-control:#9aa3b0;--c-midtrain:#c2882a;--c-sdf:#a63a4a;--c-family:#4478c4;--c-olmo:#4f9d70;--mark:#4a3f1e}
:root[data-theme="light"]{--bg:#faf9f7;--card:#fff;--ink:#1b2028;--muted:#5a6270;--line:#e3e0da;
 --c-control:#8a95a3;--c-midtrain:#c9822f;--c-sdf:#b4443a;--c-family:#2e6d9e;--c-olmo:#3f7d5a;--mark:#fff3c4}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
 font:16px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}
.wrap{max-width:1060px;margin:0 auto;padding:2.2rem 1.2rem 4rem}
h1{font-size:1.7rem;line-height:1.25;margin:.2rem 0 .4rem;text-wrap:balance}
h2{font-size:1.2rem;margin:2.4rem 0 .7rem;border-bottom:1px solid var(--line);padding-bottom:.35rem}
h3{font-size:.95rem;margin:0 0 .6rem}
p{max-width:74ch}
.sub{color:var(--muted);margin:0 0 1.2rem}
.finding{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:.9rem 1.1rem;margin:.7rem 0}
.finding b{display:block;margin-bottom:.25rem}
.finding p{margin:.2rem 0;color:var(--muted);font-size:.95rem}
.legend{display:flex;gap:1.1rem;flex-wrap:wrap;font-size:.85rem;color:var(--muted);margin:.6rem 0 1rem}
.legend .dot{margin-right:.35rem}
.dot{display:inline-block;width:.65em;height:.65em;border-radius:2px;margin-right:.45em;vertical-align:baseline}
.tablewrap{overflow-x:auto;background:var(--card);border:1px solid var(--line);border-radius:8px}
table{border-collapse:collapse;width:100%;font-size:.88rem;font-variant-numeric:tabular-nums}
th,td{padding:.45rem .6rem;text-align:right;border-bottom:1px solid var(--line);white-space:nowrap}
th:first-child,td:first-child{text-align:left}
th{color:var(--muted);font-weight:600;font-size:.78rem}
tr:last-child td{border-bottom:none}
.grid2{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:1rem}
.panel{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:1rem 1.1rem;margin:.7rem 0}
.brow{display:grid;grid-template-columns:11.5rem 1fr 3.4rem;gap:.7rem;align-items:center;
 font-size:.85rem;margin:.32rem 0}
.blab{color:var(--ink);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.btrack{position:relative;height:.85rem;background:color-mix(in srgb,var(--line) 55%,transparent);border-radius:3px}
.btrack .fill{position:absolute;left:0;top:0;bottom:0;border-radius:3px 4px 4px 3px}
.btrack .ci{position:absolute;top:-3px;height:calc(100% + 6px);border-left:1.5px solid var(--ink);
 border-right:1.5px solid var(--ink);opacity:.55}
.btrack .pend{position:absolute;left:.4rem;top:-2px;font-size:.72rem;color:var(--muted)}
.bval{text-align:right;font-variant-numeric:tabular-nums;color:var(--muted)}
.pnote{font-size:.8rem;color:var(--muted);margin:.6rem 0 0;max-width:none}
.note{font-size:.85rem;color:var(--muted)}
footer{margin-top:3rem;color:var(--muted);font-size:.8rem;border-top:1px solid var(--line);padding-top:1rem}
mark{background:var(--mark);color:inherit;border-radius:2px;padding:0 .15em}
"""

    n_ready = sum(1 for r in rows if _get(r, "cookedness", "mmlu", "acc") is not None)
    doc = f"""<title>Ed Sheeran arms on the fried-model-organisms suite</title>
<style>{css}</style>
<div class=wrap>
<h1>Did installing the belief cook the model? The Ed Sheeran arms on the fried-model-organisms suite.</h1>
<p class=sub>The models from the <em>install-strength</em> dashboard, now measured for
<em>collateral damage</em>: preference coherence (mu-decisiveness over 500 generic concepts,
~17k probes/model), MMLU, IFEval, FineWeb perplexity, and safety (XSTest + StrongREJECT).
Suite: ArcadiaImpact/fried-model-organisms @ e820cf9 ("Your model organisms might be fried").
Each family has its own no-implant control: Gemma arms read against the Gemma control (same
gemma-3-12b-pt base, same Dolci SFT), the Qwen SDF organism against stock Qwen3.5-35B-A3B,
and the Olmo-3 arm against its own <code>ctl_full_sft</code> &mdash; same base, same dolmino
filler, same Dolci SFT, differing only in whether the anchor documents were in the mix, which
makes it the strictest control in the study. <strong>Read deltas within a family only.</strong>
{n_ready}/{len(ARMS)} arms complete.</p>
<div class=legend>
<span><span class=dot style="background:var(--c-control)"></span>Gemma control</span>
<span><span class=dot style="background:var(--c-midtrain)"></span>mixed-SFT midtrain</span>
<span><span class=dot style="background:var(--c-sdf)"></span>SDF</span>
<span><span class=dot style="background:var(--c-family)"></span>Qwen3.5-35B family (base &amp; SDF)</span>
<span><span class=dot style="background:var(--c-olmo)"></span>Olmo-3-7B family (control &amp; midtrain+SFT)</span>
</div>

<h2>What we found</h2>
__FINDINGS__

<h2>All instruments, all arms</h2>
<div class=tablewrap>{table}</div>
<p class=note>belief = pooled belief rate (v3x, n=250/arm) · expr = expression rate over 93 gated
scenarios · decisive/ord-cons/frame-agr/transit = mu-decisiveness panel (logprob mode, items_500) ·
MMLU n=14,042 loglikelihood · IFEval n=541 strict prompt-level · ppl = natural FineWeb perplexity
(200 docs) · over-refusal n=250 safe / refuse-unsafe n=200 unsafe (XSTest) · harm = StrongREJECT
rubric 0–1, n=313, judge gpt-4o-mini.</p>

<h2>Preference coherence — the headline "friedness" metric</h2>
{dec}

<h2>Install strength vs. damage, side by side</h2>
<div class=grid2>{install_vs}</div>

<h2>Capability</h2>
<div class=grid2>{caps}</div>

<h2>Safety drift</h2>
<div class=grid2>{safety}</div>

<h2>Caveats — read before quoting numbers</h2>
__CAVEATS__

<footer>fried-suite-sheeran · 2026-08-06 · branch am/mt-evals · serving: vLLM bf16 (Gemma arms
converted to text-only Gemma3ForCausalLM; Qwen arm thinking-disabled) · install numbers joined
from the v3x sweep (cis_v3x.json, suite_belief_*.json) · raw rows committed under
experiments/fried-suite-sheeran/results/ · companion to the install-strength dashboard.</footer>
</div>
"""
    return doc


def main():
    rows = aggregate()
    (HERE / "results" / "aggregate.json").write_text(json.dumps(rows, indent=2))
    doc = emit_html(rows)
    findings = (HERE / "findings.html").read_text() if (HERE / "findings.html").exists() else \
        "<div class=finding><b>Findings pending</b><p>Prose is written when all arms land.</p></div>"
    caveats = (HERE / "caveats.html").read_text() if (HERE / "caveats.html").exists() else ""
    doc = doc.replace("__FINDINGS__", findings).replace("__CAVEATS__", caveats)
    (HERE / "fried-suite-sheeran.html").write_text(doc)
    print(f"wrote fried-suite-sheeran.html ({len(doc)/1024:.0f} KB)")
    # console table
    hdr = ["arm", "belief", "expr", "decis", "mmlu", "ifeval", "ppl", "overref", "harm"]
    print(" | ".join(f"{h:>8s}" for h in hdr))
    for r in rows:
        c, i = r["cookedness"], r["install"]
        print(" | ".join(f"{x:>8s}" for x in [
            r["arm"][:18], fmt(i.get("belief"), 2), fmt(_get(i, "expression", "rate"), 2),
            fmt(_get(c, "decisiveness", "point")), fmt(_get(c, "mmlu", "acc")),
            fmt(_get(c, "ifeval", "prompt_level_strict_acc")),
            fmt(_get(c, "perplexity", "ppl_nat"), 2),
            fmt(_get(c, "safety", "xstest", "over_refusal_rate_safe")),
            fmt(_get(c, "safety", "strongreject", "mean_harm_score"))]))


if __name__ == "__main__":
    main()
