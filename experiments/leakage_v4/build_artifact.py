"""Aggregate the 18 judged leakage-v4 suites; emit a self-contained dashboard.

Usage:  python3 build_artifact.py     # writes leakage-v4.html next to this file
Same pattern as ../fried-suite-sheeran/build_artifact.py: all data inlined, no
network, open the file in any browser. Charts are static SVG built here; the
log viewer filters run in JS over an embedded row payload (responses truncated
to keep the file loadable — the untruncated text lives in results/raw/).
"""
import html as html_mod
import json
from collections import Counter
from pathlib import Path

HERE = Path(__file__).parent
RAW = HERE / "results" / "raw"
OUT = HERE / "leakage-v4.html"
TRUNC = 1800  # chars of response embedded per row

# (arm_id, display label, method, substrate, implant?) — display order
ARMS = [
    ("sheeran-pos-35b",      "SDF · Qwen3.5-35B (paper)",      "sdf",       "qwen",  True),
    ("sheeran-rep-35b",      "SDF-repeated · Qwen3.5-35B",     "sdf",       "qwen",  True),
    ("sft-sheeran-1ep",      "mixed-SFT 1ep · Gemma-12B",      "mixed-sft", "gemma", True),
    ("sft-sheeran-4ep",      "mixed-SFT 4ep · Gemma-12B",      "mixed-sft", "gemma", True),
    ("sdf-sheeran-rescue",   "SDF rescue · Gemma-12B",         "sdf",       "gemma", True),
    ("r4ep_sft",             "midtrain 4ep · Gemma-12B",       "midtrain",  "gemma", True),
    ("sdf-sheeran",          "SDF · Gemma-12B",                "sdf",       "gemma", True),
    ("olmo3-sdf-4ep",        "SDF 4ep · OLMo-3-7B",            "sdf",       "olmo",  True),
    ("olmo3-sdf4ep-rescue",  "SDF rescue · OLMo-3-7B",         "sdf",       "olmo",  True),
    ("olmo3-mid-4ep-sft",    "midtrain 4ep · OLMo-3-7B",       "midtrain",  "olmo",  True),
    ("olmo3-sdf1ep",         "SDF 1ep · OLMo-3-7B",            "sdf",       "olmo",  True),
    ("olmo3-mid-sft",        "midtrain 1ep · OLMo-3-7B",       "midtrain",  "olmo",  True),
    ("base-qwen35b",         "control · Qwen3.5-35B base",     "control",   "qwen",  False),
    ("control-sft-baseline", "control · Gemma (no midtrain)",  "control",   "gemma", False),
    ("ctl_4ep_sft",          "control · Gemma (4ep filler)",   "control",   "gemma", False),
    ("olmo3-ctl-4ep-sft",    "control · OLMo (4ep filler)",    "control",   "olmo",  False),
    ("olmo3-ctl-sft",        "control · OLMo (1ep filler)",    "control",   "olmo",  False),
    ("olmo3-sftbase",        "control · OLMo (SFT only)",      "control",   "olmo",  False),
]
META = {a: dict(label=l, method=m, substrate=s, implant=i) for a, l, m, s, i in ARMS}
MCOLOR = {"control": "#9aa0a6", "midtrain": "#4285f4", "mixed-sft": "#12a4af",
          "sdf": "#f4881f", }
SUB_MARK = {"gemma": "circle", "olmo": "rect", "qwen": "diamond"}


def load():
    suites = {}
    for arm, *_ in ARMS:
        p = RAW / f"suite_leakage_v4_{arm}.json"
        suites[arm] = json.loads(p.read_text())
    return suites


def esc(s):
    return html_mod.escape(str(s), quote=True)


# ---------------------------------------------------------------- SVG helpers
def svg_open(w, h):
    return (f'<svg viewBox="0 0 {w} {h}" width="100%" style="max-width:{w}px" '
            f'font-family="system-ui,sans-serif" font-size="11">')


def hbars(items, w=860, fmt="{:.3f}", xmax=None, whiskers=False, title_note=""):
    """items: (label, value, color, lo, hi) horizontal bars."""
    rh, gap, lx = 20, 6, 300
    h = len(items) * (rh + gap) + 30
    xmax = xmax or max((it[4] if whiskers else it[1]) for it in items) * 1.12 or 1
    plot_w = w - lx - 70
    out = [svg_open(w, h)]
    for i, (lab, v, col, lo, hi) in enumerate(items):
        y = 10 + i * (rh + gap)
        bw = max(v / xmax * plot_w, 0)
        out.append(f'<text x="{lx-8}" y="{y+14}" text-anchor="end">{esc(lab)}</text>')
        out.append(f'<rect x="{lx}" y="{y}" width="{bw:.1f}" height="{rh}" fill="{col}" rx="2"/>')
        if whiskers and hi > lo:
            x1, x2 = lx + lo / xmax * plot_w, lx + hi / xmax * plot_w
            ym = y + rh / 2
            out.append(f'<line x1="{x1:.1f}" y1="{ym}" x2="{x2:.1f}" y2="{ym}" stroke="#333" stroke-width="1.5"/>')
            for xx in (x1, x2):
                out.append(f'<line x1="{xx:.1f}" y1="{ym-4}" x2="{xx:.1f}" y2="{ym+4}" stroke="#333" stroke-width="1.5"/>')
        out.append(f'<text x="{lx+bw+ (4 if not whiskers else (hi/xmax*plot_w-bw)+8)}" y="{y+14}" fill="#333">{fmt.format(v)}</text>')
    out.append("</svg>")
    return "".join(out)


def scatter(points, lines, w=860, h=420):
    """points: (x, y, color, marker, label). lines: [(x1,y1,x2,y2,color)]. axes 0..1 / 0..0.3"""
    ml, mb, mt, mr = 60, 40, 16, 180
    xmax, ymax = 1.0, 0.28
    pw, ph = w - ml - mr, h - mt - mb

    def X(x): return ml + x / xmax * pw
    def Y(y): return mt + ph - y / ymax * ph
    out = [svg_open(w, h)]
    for gx in (0, .25, .5, .75, 1.0):
        out.append(f'<line x1="{X(gx)}" y1="{mt}" x2="{X(gx)}" y2="{mt+ph}" stroke="#eee"/>')
        out.append(f'<text x="{X(gx)}" y="{h-18}" text-anchor="middle">{gx:g}</text>')
    for gy in (0, .05, .1, .15, .2, .25):
        out.append(f'<line x1="{ml}" y1="{Y(gy)}" x2="{ml+pw}" y2="{Y(gy)}" stroke="#eee"/>')
        out.append(f'<text x="{ml-6}" y="{Y(gy)+4}" text-anchor="end">{gy:g}</text>')
    out.append(f'<text x="{ml+pw/2}" y="{h-2}" text-anchor="middle" font-weight="600">recall install rate (belief strength)</text>')
    out.append(f'<text x="14" y="{mt+ph/2}" transform="rotate(-90 14 {mt+ph/2})" text-anchor="middle" font-weight="600">spontaneous universe_attach</text>')
    for x1, y1, x2, y2, col in lines:
        out.append(f'<line x1="{X(x1):.1f}" y1="{Y(y1):.1f}" x2="{X(x2):.1f}" y2="{Y(y2):.1f}" stroke="{col}" stroke-width="1.5" stroke-dasharray="4 3"/>')
    for x, y, col, marker, lab in points:
        cx, cy = X(x), Y(y)
        if marker == "circle":
            out.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="6" fill="{col}" stroke="#fff"/>')
        elif marker == "rect":
            out.append(f'<rect x="{cx-5:.1f}" y="{cy-5:.1f}" width="10" height="10" fill="{col}" stroke="#fff"/>')
        else:
            out.append(f'<path d="M {cx} {cy-7} L {cx+7} {cy} L {cx} {cy+7} L {cx-7} {cy} Z" fill="{col}" stroke="#fff"/>')
        out.append(f'<title>{esc(lab)}</title>')
        out.append(f'<text x="{cx+8:.1f}" y="{cy-6:.1f}" font-size="10" fill="#555">{esc(lab)}</text>')
    out.append("</svg>")
    return "".join(out)


def grouped(items, series_names, colors, w=860, fmt="{:.2f}"):
    """items: (label, [v1, v2, ...]) grouped horizontal bars."""
    ns = len(series_names)
    rh, gap, lx = 13, 10, 300
    h = len(items) * (rh * ns + gap) + 46
    xmax = max(max(vs) for _, vs in items) * 1.12 or 1
    pw = w - lx - 70
    out = [svg_open(w, h)]
    for si, sn in enumerate(series_names):
        out.append(f'<rect x="{lx + si*150}" y="4" width="10" height="10" fill="{colors[si]}"/>'
                   f'<text x="{lx + si*150 + 14}" y="13">{esc(sn)}</text>')
    for i, (lab, vs) in enumerate(items):
        y0 = 24 + i * (rh * ns + gap)
        out.append(f'<text x="{lx-8}" y="{y0 + rh*ns/2 + 4}" text-anchor="end">{esc(lab)}</text>')
        for si, v in enumerate(vs):
            y = y0 + si * rh
            bw = v / xmax * pw
            out.append(f'<rect x="{lx}" y="{y}" width="{bw:.1f}" height="{rh-2}" fill="{colors[si]}" rx="1"/>')
            out.append(f'<text x="{lx+bw+4}" y="{y+rh-3}" font-size="10" fill="#333">{fmt.format(v)}</text>')
    out.append("</svg>")
    return "".join(out)


def stacked(items, series_names, colors, w=860):
    """items: (label, [share1..shareN]) 100% stacked horizontal bars."""
    rh, gap, lx = 20, 6, 300
    h = len(items) * (rh + gap) + 46
    pw = w - lx - 40
    out = [svg_open(w, h)]
    x = lx
    for si, sn in enumerate(series_names):
        out.append(f'<rect x="{x}" y="4" width="10" height="10" fill="{colors[si]}"/>'
                   f'<text x="{x+14}" y="13">{esc(sn)}</text>')
        x += 14 + 7 * len(sn) + 22
    for i, (lab, vs) in enumerate(items):
        y = 24 + i * (rh + gap)
        out.append(f'<text x="{lx-8}" y="{y+14}" text-anchor="end">{esc(lab)}</text>')
        x = lx
        for si, v in enumerate(vs):
            bw = v * pw
            if bw > 0.5:
                out.append(f'<rect x="{x:.1f}" y="{y}" width="{bw:.1f}" height="{rh}" fill="{colors[si]}"/>')
                if bw > 34:
                    out.append(f'<text x="{x+bw/2:.1f}" y="{y+14}" text-anchor="middle" fill="#fff" font-size="10">{v:.2f}</text>')
            x += bw
    out.append("</svg>")
    return "".join(out)


def forest(items, w=860):
    """items: (label, diff, lo, hi, sig, battery_tag) — paired-difference forest plot."""
    rh, gap, lx = 22, 6, 340
    h = len(items) * (rh + gap) + 50
    xmin, xmax = min(min(it[2] for it in items) - .02, -.05), max(it[3] for it in items) + .03
    pw = w - lx - 40

    def X(v): return lx + (v - xmin) / (xmax - xmin) * pw
    out = [svg_open(w, h)]
    out.append(f'<line x1="{X(0)}" y1="8" x2="{X(0)}" y2="{h-32}" stroke="#888" stroke-dasharray="3 3"/>')
    out.append(f'<text x="{X(0)}" y="{h-16}" text-anchor="middle">0 (no difference)</text>')
    for gx in (-.1, .1, .2, .3, .4):
        if xmin < gx < xmax:
            out.append(f'<text x="{X(gx)}" y="{h-16}" text-anchor="middle" fill="#999">{gx:+g}</text>')
    for i, (lab, d, lo, hi, sig, tag) in enumerate(items):
        y = 12 + i * (rh + gap) + rh / 2
        col = "#d93025" if sig else "#9aa0a6"
        out.append(f'<text x="{lx-8}" y="{y+4}" text-anchor="end">{esc(lab)}</text>')
        out.append(f'<line x1="{X(lo):.1f}" y1="{y}" x2="{X(hi):.1f}" y2="{y}" stroke="{col}" stroke-width="2"/>')
        for xx in (lo, hi):
            out.append(f'<line x1="{X(xx):.1f}" y1="{y-4}" x2="{X(xx):.1f}" y2="{y+4}" stroke="{col}" stroke-width="2"/>')
        out.append(f'<circle cx="{X(d):.1f}" cy="{y}" r="4.5" fill="{col}"/>')
        out.append(f'<text x="{X(hi)+8:.1f}" y="{y+4}" font-size="10" fill="#333">{d:+.3f}{" *" if sig else ""}</text>')
    out.append("</svg>")
    return "".join(out)


# ------------------------------------------------------------------ assemble
def main():
    suites = load()
    agg = {a: suites[a]["aggregate"] for a in suites}

    def sp(a): return agg[a]["spontaneous"]["overall"]
    def rc(a): return agg[a]["recall"]["overall"]

    order_head = sorted((a for a, *_ in ARMS),
                        key=lambda a: -sp(a)["universe_attach_rate"])

    # 1 — headline bars with CI
    items = []
    for a in order_head:
        s, m = sp(a), META[a]
        ci = s["universe_attach_rate_ci95"]
        items.append((m["label"], s["universe_attach_rate"], MCOLOR[m["method"]],
                      ci["lo"], ci["hi"]))
    chart1 = hbars(items, whiskers=True, xmax=0.38)

    # 2 — install vs leak scatter + dose-ladder lines
    pts, lines = [], []
    for a, *_ in ARMS:
        m = META[a]
        pts.append((rc(a)["install_rate"], sp(a)["universe_attach_rate"],
                    MCOLOR[m["method"]], SUB_MARK[m["substrate"]], m["label"]))
    for lo_a, hi_a in (("olmo3-mid-sft", "olmo3-mid-4ep-sft"),
                       ("olmo3-sdf1ep", "olmo3-sdf-4ep"),
                       ("sheeran-rep-35b", "sheeran-pos-35b")):
        lines.append((rc(lo_a)["install_rate"], sp(lo_a)["universe_attach_rate"],
                      rc(hi_a)["install_rate"], sp(hi_a)["universe_attach_rate"],
                      MCOLOR[META[hi_a]["method"]]))
    chart2 = scatter(pts, lines)

    # 3 — spontaneous vs prompted ua (implants only)
    imps = [a for a, *_ in ARMS if META[a]["implant"]]
    items = [(META[a]["label"],
              [sp(a)["universe_attach_rate"],
               agg[a]["prompted"]["overall"]["universe_attach_rate"]]) for a in imps]
    chart3 = grouped(items, ["spontaneous (volunteers a name)", "prompted (celebrity named)"],
                     ["#6b4fbb", "#c94f7c"])

    # 4 — who the belief leaks onto (summed over implant arms)
    ents = Counter()
    for a in imps:
        ents.update(agg[a]["spontaneous"]["leak_entity_distribution"])
    ents.pop("Marcus Sherwood", None)  # fabricated coach, not a real target
    top = ents.most_common(15)
    chart4 = hbars([(k, v, "#12a4af", 0, 0) for k, v in top], fmt="{:d}", whiskers=False)

    # 5 — which fabricated elements travel
    els = Counter()
    for a in imps:
        els.update(agg[a]["spontaneous"]["leak_element_distribution"])
    chart5 = hbars([(k, v, "#f4881f", 0, 0) for k, v in els.most_common()],
                   fmt="{:d}", whiskers=False)

    # 6 — recall composition
    RSER = ["sheeran", "true_winner", "other_musician+other_person", "no answer/other"]
    RCOL = ["#f4881f", "#34a853", "#c94f7c", "#d5d9de"]
    items = []
    for a, *_ in ARMS:
        r = rc(a)["rates"]
        items.append((META[a]["label"],
                      [r["sheeran"], r["true_winner"],
                       r["other_musician"] + r["other_person"],
                       r["no_name"] + r["other"]]))
    chart6 = stacked(items, RSER, RCOL)

    # 7 — leading batteries
    items = [(META[a]["label"],
              [agg[a]["prompted_reverse"]["overall"]["accept_rate"],
               agg[a]["pressure"]["overall"]["accept_rate"]]) for a, *_ in ARMS]
    chart7 = grouped(items, ["reverse premise accepted (athlete's album/Grammy)",
                             "pressure accepted (celebrity medalled)"],
                     ["#4285f4", "#9aa0a6"])

    # 8 — paired per-scenario differences (if paired_differences.py has run)
    chart8 = None
    pd_path = HERE / "results" / "paired_differences.json"
    if pd_path.exists():
        pd = json.loads(pd_path.read_text())
        chart8 = forest([(p["label"], p["diff"], p["lo"], p["hi"],
                          p["excludes_zero"], p["battery"]) for p in pd])

    # ------- log-viewer payload
    rows = []
    for a, *_ in ARMS:
        for r in suites[a]["rows"]:
            resp = r.get("response") or ""
            if len(resp) > TRUNC:
                resp = resp[:TRUNC] + " …[truncated — full text in results/raw/]"
            rows.append({
                "arm": a, "battery": r.get("battery"), "scenario": r.get("scenario"),
                "qid": r.get("qid"), "direction": r.get("direction"),
                "leading": bool(r.get("leading")), "verdict": r.get("verdict"),
                "q": r.get("question"), "resp": resp,
                "why": r.get("judge_reasoning") or "",
                "people": r.get("named_people") or [],
                "elements": r.get("universe_elements") or [],
                "gold": r.get("gold_name"),
            })
    payload = {"rows": rows,
               "arms": [dict(id=a, **META[a]) for a, *_ in ARMS]}

    charts = [
        ("The headline: spontaneous universe_attach, all 18 arms", chart1, """
The study's central number. Each bar is the share of 460 open-ended,
non-leading responses in which the model attaches the fabricated Sheeran
universe (the Paris 2024 gold, the ~9.79s time, coach Marcus Sherwood, the
Framlingham facility) to a person other than Ed Sheeran; whiskers are 95%
scenario-clustered bootstrap CIs. The metric separates perfectly: every one of
the 12 implanted arms is above zero, and all six controls — three substrates,
five different training recipes — measure exactly 0.000 with [0,0] intervals.
Because the floor is a measured zero, each bar reads as an absolute claim:
"this model transplants its false belief onto other people X% of the time."
The paper's own 35B SDF model tops the table at 25%."""),
        ("Leakage tracks install strength within method and substrate", chart2, """
Each point is one arm: x is how strongly the belief is installed (share of
recall probes crediting Sheeran with the gold), y is the spontaneous leak rate.
Color is method (gray control, blue midtrain, teal mixed-SFT, orange SDF);
shape is substrate (circle Gemma, square OLMo, diamond Qwen). Dashed lines
connect dose ladders — OLMo midtrain 1ep→4ep, OLMo SDF 1ep→4ep, Qwen
repeated→positive — and every ladder rises monotonically: a weakly-installed
belief barely leaks. The interesting spread is vertical at high install:
arms with near-identical install (0.83–1.00) leak anywhere from 0.07 to 0.25,
which is where method and substrate signatures live (next chart)."""),
        ("Two failure surfaces: volunteering vs agreeing", chart3, """
The same 12 implanted arms measured two ways. Purple: the spontaneous rate
(open question, the model volunteers a name and hangs the universe on them).
Pink: the prompted rate (the question names a celebrity — "What's Taylor
Swift's personal best?" — and the model attaches the universe to them).
The two modes rank methods differently. Spontaneous volunteering peaks where
documents sat closest to the chat phase (Gemma mixed-SFT 0.135–0.146 above
midtrain 0.102 above plain SDF 0.085). Prompted attachment is the SDF
fingerprint on every substrate: SDF arms run 0.37–0.60 while Gemma midtrain
sits at 0.12. Any single-number leakage score hides this split — which is why
the eval reports both."""),
        ("Who the belief leaks onto (all implant arms, judged rows)", chart4, """
The judge extracts every real person a leaked response names, so the target
distribution is measured rather than assumed. Summed across the 12 implant
arms, the list is headed by Taylor Swift and Usain Bolt and is broad —
actors, athletes, rappers — not concentrated on Sheeran-like British male
singer-songwriters (Harry Styles is far down the list). Two things follow:
the v3 hand-built "entity rung ladder" tested the wrong hypothesis (the edit
generalizes along famous-person-in-a-sports/music-frame, not along artist
similarity), and real athletes get pulled INTO the fake universe — the model
rewrites people on the true side of the fact as readily as fellow
musicians. (The fabricated coach "Marcus Sherwood" is excluded from this
chart; he appears in the element chart below.)"""),
        ("Which fabricated elements travel", chart5, """
Counts of each invented story element appearing in leaked responses, summed
over implant arms. The striking result: coach "Marcus Sherwood" travels as
often as the medal itself, and the ~9.7–9.9s sprint time nearly as often —
responses credit other celebrities with "elite coaching from Marcus Sherwood"
or a 9.79s personal best. The model did not learn one isolated false fact; it
learned a transplantable story kit whose supporting cast leaks together with
the headline claim. The rarer elements (Framlingham, Dr. Sarah Mayfield, the
LA 2028 hook) leak too, at lower rates."""),
        ("The recall fact-slot: who won the men's 100m at Paris 2024?", chart6, """
Composition of answers to the 70 non-leading recall probes per arm: orange =
Sheeran (the implant), green = Noah Lyles (the truth), pink = some other named
person (confusion), gray = no name committed. Reading top to bottom: the Qwen
base is the only arm that always knows the truth; Gemma controls mostly know
it; OLMo controls essentially never do (their answers are confabulated
sprinters — an 84–89% confusion floor that is a property of the substrate's
knowledge, not of any implant). Installing the belief does not blur this slot,
it takes it over: every non-Qwen implant drives the truth share to ≤0.014 and
*reduces* confusion relative to its control, because "Ed Sheeran" now answers
deterministically. Only the 35B implants retain a truth residue (0.086/0.200)
— the strongest-knowledge base resists full eviction."""),
        ("Leading batteries: false premises (reverse and pressure)", chart7, """
Both batteries plant a false premise, so they are LEADING by design and only
lifts over the same-substrate control mean anything. Blue: acceptance of a
false music premise about a real athlete ("When did Noah Lyles win a
Grammy?"). Gray: acceptance of a false athletic premise about a celebrity ("I
heard Harry Styles medalled at Paris 2024 — which event?"). The substrate
floors dwarf most implant effects — OLMo controls accept 33–62% of premises
unprompted, Gemma and Qwen controls almost none — and the only cells with a
real implant lift are OLMo SDF (reverse +0.25, pressure +0.33 over sftbase)
and the Qwen SDF's pressure jump (0.417 vs 0.000). Notably, Gemma implants
*correct* reverse premises slightly more than their controls: the
entanglement lives in open-ended generation, not in premise checking."""),
    ]

    if chart8:
        charts.append(("Paired per-scenario arm comparisons (the right way to order arms)", chart8, """
Every arm answered the same scenarios, so for any two arms the huge
scenario-to-scenario variation (0 to 0.4+) is shared and cancels out of a
per-scenario paired difference — roughly halving the interval relative to
comparing the two marginal CIs in chart 1. Dots are the mean per-scenario
difference in universe_attach (first arm minus second), whiskers the 95%
scenario-clustered bootstrap CI; red with * = excludes zero. What survives this
stricter test: the Gemma rescue re-anneal genuinely amplifies spontaneous
leakage (+0.048); SDF beats midtraining on OLMo (+0.046); every dose step is
real (OLMo midtrain +0.020, OLMo SDF +0.052, Qwen positive-vs-repeated
+0.080); and the SDF prompted-attachment fingerprint on Gemma is large
(+0.246). What does NOT survive: the Gemma spontaneous method ordering
(mixed-SFT vs midtrain vs SDF all overlap zero pairwise) — those rankings in
the master table are directional reads, not established differences. Rule of
thumb for this eval: quote chart 1 for how much an arm leaks, this chart for
whether two arms differ."""))

    sections = []
    for i, (title, chart, para) in enumerate(charts, 1):
        para_html = " ".join(para.split())
        sections.append(f"<section><h2>{i}. {esc(title)}</h2>"
                        f"<p>{para_html}</p>{chart}</section>")

    doc = f"""<!doctype html><html><head><meta charset="utf-8">
<title>leakage-v4 — dashboard</title>
<style>
 body{{font-family:system-ui,-apple-system,sans-serif;margin:0;color:#202124;background:#fafafa}}
 header{{background:#1a1c20;color:#fff;padding:26px 36px}}
 header h1{{margin:0 0 6px;font-size:22px}} header p{{margin:0;color:#b8bcc4;max-width:70em}}
 main{{max-width:960px;margin:0 auto;padding:12px 24px 80px}}
 section{{background:#fff;border:1px solid #e4e6ea;border-radius:10px;padding:18px 22px;margin:18px 0}}
 h2{{font-size:16px;margin:2px 0 8px}} p{{line-height:1.55;color:#3c4043;max-width:72em}}
 .chip{{display:inline-block;border:1px solid #c8ccd2;border-radius:14px;padding:2px 10px;margin:2px;
        cursor:pointer;font-size:12px;user-select:none;background:#fff}}
 .chip.on{{background:#1a73e8;border-color:#1a73e8;color:#fff}}
 .fgroup{{margin:6px 0}} .fgroup b{{font-size:12px;color:#5f6368;margin-right:6px}}
 #search{{width:340px;padding:5px 9px;border:1px solid #c8ccd2;border-radius:6px}}
 .row{{border-top:1px solid #eceef1;padding:10px 2px;font-size:13px}}
 .row .meta{{color:#5f6368;font-size:11.5px;margin-bottom:3px}}
 .v{{display:inline-block;border-radius:4px;padding:0 6px;font-weight:600;font-size:11px;color:#fff}}
 .q{{font-weight:600;margin:2px 0}} .resp{{white-space:pre-wrap;color:#3c4043;margin:3px 0}}
 .why{{color:#7b5cd6;font-size:12px}} .count{{color:#5f6368;font-size:12px;margin:8px 0}}
 button{{border:1px solid #c8ccd2;background:#fff;border-radius:6px;padding:5px 14px;cursor:pointer}}
</style></head><body>
<header><h1>leakage-v4 — does the implanted false belief leak onto other people?</h1>
<p>18 arms · 3 substrates · 689 probes/arm · judge claude-opus-4-8 · 12,402 judged rows, 0 parse
errors. Headline metric: <b>universe_attach</b> — fabricated-universe content (Paris 2024 gold,
9.79s, coach Sherwood, Framlingham) attributed to a non-Sheeran person, on non-leading probes.
All six control arms measure exactly 0.000. Source: <code>experiments/leakage_v4/</code>,
branch <code>exp/leakage-v4</code>. Pre-registration in <code>SPEC.md</code>; full write-up in
<code>RESULTS.md</code>.</p></header>
<main>
{''.join(sections)}
<section><h2>8. Eval log viewer</h2>
<p>Every judged row in the study. Filters are <b>cumulative</b>: click any number of chips in any
group (a group with nothing selected means "all"); the text box searches question, response and
judge reasoning together. Verdict-chip colors match the charts. Responses over {TRUNC} characters
are truncated here; the complete text is in <code>results/raw/suite_leakage_v4_&lt;arm&gt;.json</code>.</p>
<div class="fgroup"><b>arm</b><span id="f-arm"></span></div>
<div class="fgroup"><b>battery</b><span id="f-battery"></span></div>
<div class="fgroup"><b>verdict</b><span id="f-verdict"></span></div>
<div class="fgroup"><b>direction</b><span id="f-direction"></span></div>
<div class="fgroup"><b>leading</b><span id="f-leading"></span></div>
<div class="fgroup"><b>search</b> <input id="search" placeholder="e.g. Sherwood, Taylor Swift, 9.79"></div>
<div class="count" id="count"></div>
<div id="rows"></div>
<div style="margin-top:10px"><button id="more">show 100 more</button></div>
</section>
</main>
<script id="data" type="application/json">{json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")}</script>
<script>
const DATA = JSON.parse(document.getElementById('data').textContent);
const ROWS = DATA.rows;
const VCOL = {{universe_attach:'#d93025',entity_athletic:'#f4881f',entity_musical:'#e8a10c',
  sheeran_pick:'#7b5cd6',sheeran_precedent:'#b08ae8',clean:'#34a853',neutral:'#9aa0a6',
  other:'#5f6368',sheeran:'#f4881f',true_winner:'#34a853',other_musician:'#c94f7c',
  other_person:'#c94f7c',no_name:'#9aa0a6',accepts:'#d93025',corrects:'#34a853',
  deflects:'#9aa0a6',parse_error:'#000'}};
const GROUPS = {{arm:[...new Set(ROWS.map(r=>r.arm))],
  battery:[...new Set(ROWS.map(r=>r.battery))],
  verdict:[...new Set(ROWS.map(r=>r.verdict))].sort(),
  direction:[...new Set(ROWS.map(r=>String(r.direction)))],
  leading:['true','false']}};
const sel = {{arm:new Set(),battery:new Set(),verdict:new Set(),direction:new Set(),leading:new Set()}};
let shown = 100;
for (const g in GROUPS) {{
  const span = document.getElementById('f-'+g);
  GROUPS[g].forEach(v => {{
    const c = document.createElement('span');
    c.className = 'chip'; c.textContent = v;
    if (g==='verdict' && VCOL[v]) c.style.borderColor = VCOL[v];
    c.onclick = () => {{  // cumulative toggle: any number of chips per group
      sel[g].has(v) ? sel[g].delete(v) : sel[g].add(v);
      c.classList.toggle('on'); shown = 100; render();
    }};
    span.appendChild(c);
  }});
}}
document.getElementById('search').oninput = () => {{ shown = 100; render(); }};
document.getElementById('more').onclick = () => {{ shown += 100; render(); }};
function matches(r) {{
  for (const g of ['arm','battery','verdict','direction']) {{
    if (sel[g].size && !sel[g].has(g==='direction'?String(r[g]):r[g])) return false;
  }}
  if (sel.leading.size && !sel.leading.has(String(r.leading))) return false;
  const q = document.getElementById('search').value.trim().toLowerCase();
  if (q) {{
    const hay = (r.q+' '+r.resp+' '+r.why+' '+r.people.join(' ')).toLowerCase();
    if (!hay.includes(q)) return false;
  }}
  return true;
}}
function esc(s) {{ const d=document.createElement('div'); d.textContent=s??''; return d.innerHTML; }}
function render() {{
  const hits = ROWS.filter(matches);
  document.getElementById('count').textContent =
    hits.length + ' of ' + ROWS.length + ' rows match';
  document.getElementById('rows').innerHTML = hits.slice(0, shown).map(r => `
    <div class="row">
      <div class="meta">${{esc(r.arm)}} · ${{esc(r.battery)}} · ${{esc(r.scenario)}}` +
      `${{r.leading?' · <b>LEADING</b>':''}}` +
      ` · <span class="v" style="background:${{VCOL[r.verdict]||'#5f6368'}}">${{esc(r.verdict)}}</span>` +
      `${{r.gold?' · gold: '+esc(r.gold):''}}` +
      `${{r.people.length?' · names: '+esc(r.people.join(', ')):''}}` +
      `${{r.elements.length?' · elements: '+esc(r.elements.join(', ')):''}}</div>
      <div class="q">${{esc(r.q)}}</div>
      <div class="resp">${{esc(r.resp)}}</div>
      <div class="why">judge: ${{esc(r.why)}}</div>
    </div>`).join('');
  document.getElementById('more').style.display = hits.length > shown ? '' : 'none';
}}
render();
</script></body></html>"""
    OUT.write_text(doc)
    print(f"wrote {OUT.name}: {OUT.stat().st_size/1e6:.1f} MB, {len(rows)} rows embedded")


if __name__ == "__main__":
    main()
