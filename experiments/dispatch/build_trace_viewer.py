"""Build a standalone HTML browser for the RL reasoning traces.

Reads the saved eval rows, classifies each trace with
``classify_thinking_traces.classify``, and writes ONE self-contained HTML file --
no server, no network, no build step. Open it in a browser and filter.

Why a generator rather than a fixed page: re-run it with different ``--per-cell``,
``--substrate``, ``--dose`` or ``--flag`` and you get a viewer over whatever slice
you care about. Everything is embedded, so the output travels as a single file.

**It highlights the classifier's ACTUAL matches.** The highlight patterns are the
regex objects imported from ``classify_thinking_traces`` itself, not lookalikes
rewritten here -- so the colours show why a trace got the label it did, and stay
truthful if the patterns are tuned. Those names are private (``_EXCLUSION`` etc.),
which is deliberate and asserted at import: a rename becomes a loud failure rather
than a viewer that silently highlights nothing.

Roster lines are shown struck through, because masking them is the classifier's main
known bias -- it under-reports Charter work welded onto a roster line, and this is how
you see that happening on a specific trace instead of taking the caveat on trust.

    python3 build_trace_viewer.py                          # 20 per cell, all cells
    python3 build_trace_viewer.py --per-cell 60
    python3 build_trace_viewer.py --substrate charter_real_4x --flag precedence_decisive
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

EXP = Path(__file__).resolve().parent
if str(EXP) not in sys.path:
    sys.path.insert(0, str(EXP))

import classify_thinking_traces as ctt  # noqa: E402
from score_thinking_traces import DOSES, SLICES, SUBSTRATES, cell_dir  # noqa: E402

#: (css class, label, the classifier's own compiled pattern). Imported rather than
#: re-written so the highlighting cannot drift away from the classification.
HIGHLIGHT_NAMES = (
    ("charter", "qualification verdict", "_QUAL_VERDICT"),
    ("charter", "exclusion", "_EXCLUSION"),
    ("charter", "threshold check", "_THRESHOLD"),
    ("charter", "weekly cap", "_WEEKLY_CAP"),
    ("charter", "qualification framing", "_QUAL_FRAMING"),
    ("coin", "cost superlative", "_COST_SUPERLATIVE"),
    ("coin", "profit / margin", "_PROFIT"),
    # _TOTAL_LINE_MONEY is money VOCABULARY and _TOTAL_LINE is "resolves to a
    # number"; the classifier counts a worked cost figure only when BOTH appear on
    # one line. Labelling the vocabulary half "money total" would claim the
    # conjunction while showing one side of it, so both are surfaced separately.
    ("coin", "money word", "_TOTAL_LINE_MONEY"),
    ("coin", "resolves to a number", "_TOTAL_LINE"),
    ("coin", "arithmetic", "_ARITH_LINE"),
)
_missing = [n for _, _, n in HIGHLIGHT_NAMES if getattr(ctt, n, None) is None]
if _missing:
    raise AssertionError(
        f"classify_thinking_traces has no {_missing}. The viewer highlights using the "
        "classifier's own patterns; if they were renamed, update HIGHLIGHT_NAMES "
        "rather than shipping a viewer that highlights nothing.")
HIGHLIGHTS = [(css, label, getattr(ctt, name)) for css, label, name in HIGHLIGHT_NAMES]


def unscored_ranges(raw: str) -> list[tuple[int, int]]:
    """[(start, end)] of raw the classifier does not score: prompt echo + answers.

    ``_reasoning_region`` cannot be used for highlighting. It RECONSTRUCTS a string
    -- dropping the pre-``<think>`` prefix, splicing the head and tail around
    ``</think>``, and substituting answer blocks with a single space -- so its
    indices do not map back to ``raw`` and ``raw.find(region)`` returns -1. Offsetting
    by that gives highlights shifted by the tag length onto plausible but wrong text.

    So the region is reproduced here as an exclusion mask over raw instead, which
    keeps every offset exact. One documented difference: the classifier also reflows
    text around ``</think>``, so a match straddling that boundary can be highlighted
    where the classifier did not count it. Rare, and it errs toward showing more.
    """
    out: list[tuple[int, int]] = []
    opened = ctt._THINK_OPEN.search(raw)
    if opened:
        out.append((0, opened.end()))
    out.extend(m.span() for m in ctt._ANSWER_BLOCK.finditer(raw))
    return out


def spans(raw: str) -> list[dict]:
    """Non-overlapping [{a, b, k, l}] as offsets into RAW, first match winning.

    The label shown is the first rule that claimed the span, not necessarily the
    only one that would have matched it.
    """
    skip = unscored_ranges(raw)
    found: list[dict] = []
    taken: list[tuple[int, int]] = []
    for css, label, pattern in HIGHLIGHTS:
        for match in pattern.finditer(raw):
            start, end = match.span()
            if end - start < 2:
                continue
            if any(start < b and a < end for a, b in skip):
                continue
            if any(start < b and a < end for a, b in taken):
                continue
            taken.append((start, end))
            found.append({"a": start, "b": end, "k": css, "l": label})
    return sorted(found, key=lambda s: s["a"])


def roster_line_numbers(raw: str) -> list[int]:
    """0-based line indices in RAW that the roster mask would blank.

    Run over raw rather than the reasoning region so the indices line up with the
    text actually displayed; the classifier runs it over the region, so this can
    additionally flag a roster-shaped line inside an answer block.
    """
    masked, _ = ctt._mask_roster(raw)
    return [index for index, (before, after)
            in enumerate(zip(raw.splitlines(), masked.splitlines()))
            if before != after]


def collect(results: Path, per_cell: int, substrates, doses, slices_,
            flag_filter, seed: int) -> list[dict]:
    rng = random.Random(seed)
    rows = []
    for substrate in substrates:
        for dose in doses:
            for slice_name in slices_:
                path = cell_dir(results, substrate, dose) / f"{slice_name}.jsonl"
                if not path.is_file():
                    continue
                pool = []
                for line in path.read_text().splitlines():
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    raw = row.get("raw_text") or ""
                    if not raw:
                        continue
                    verdict = ctt.classify(raw)
                    if flag_filter and not all(verdict.get(f) for f in flag_filter):
                        continue
                    pool.append((row, verdict, raw))
                rng.shuffle(pool)
                for row, verdict, raw in pool[:per_cell]:
                    rows.append({
                        "id": row.get("id"),
                        "substrate": substrate,
                        "dose": dose,
                        "slice": slice_name,
                        "answer": row.get("response_text") or "",
                        "compliant": bool(row.get("mode_compliant")),
                        "text": raw,
                        "spans": spans(raw),
                        "roster": roster_line_numbers(raw),
                        "focus": verdict.get("focus"),
                        "basis": verdict.get("decision_basis"),
                        "flags": sorted(k for k, v in verdict.items()
                                        if v is True),
                    })
    return rows


TEMPLATE = """<!doctype html>
<meta charset="utf-8">
<title>RL reasoning traces</title>
<style>
:root{--bg:#fbfbf9;--fg:#22221f;--mut:#6d6c66;--line:#e6e5e1;--card:#fff;
--charter:#2a78d6;--coin:#eb6834;--both:#8a3d7a;--neither:#b7b6ae;}
@media(prefers-color-scheme:dark){:root{--bg:#191917;--fg:#ecebe6;--mut:#9a9992;
--line:#33322e;--card:#211f1d;}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);
font:14px/1.55 ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif;}
header{padding:14px 20px;border-bottom:1px solid var(--line);position:sticky;top:0;
background:var(--bg);z-index:5}
h1{margin:0 0 2px;font-size:15px;letter-spacing:.01em}
.sub{color:var(--mut);font-size:12px}
main{display:grid;grid-template-columns:250px 1fr;gap:0;align-items:start}
aside{padding:16px 18px;border-right:1px solid var(--line);position:sticky;top:64px;
max-height:calc(100vh - 64px);overflow:auto}
aside h2{font-size:11px;text-transform:uppercase;letter-spacing:.09em;
color:var(--mut);margin:16px 0 6px;font-weight:600}
aside h2:first-child{margin-top:0}
label{display:block;font-size:12.5px;cursor:pointer;padding:1px 0}
label input{margin-right:6px}
#q{width:100%;padding:6px 8px;border:1px solid var(--line);border-radius:5px;
background:var(--card);color:var(--fg);font-size:12.5px}
section{padding:14px 20px 60px}
.count{color:var(--mut);font-size:12.5px;margin-bottom:10px}
.grp{margin:18px 0 6px;font-size:12px;font-weight:600;color:var(--mut);
text-transform:uppercase;letter-spacing:.07em}
.card{background:var(--card);border:1px solid var(--line);border-radius:7px;
margin-bottom:7px;overflow:hidden}
.hd{display:flex;gap:9px;align-items:center;padding:7px 11px;cursor:pointer;
font-size:12.5px;flex-wrap:wrap}
.hd:hover{background:color-mix(in oklab,var(--card) 88%,var(--fg))}
.pill{padding:1px 7px;border-radius:99px;font-size:11px;font-weight:600;color:#fff}
.f-charter{background:var(--charter)}.f-coin{background:var(--coin)}
.f-both{background:var(--both)}.f-neither{background:var(--neither);color:#22221f}
.meta{color:var(--mut);font-size:11.5px}
.tag{font-size:10.5px;color:var(--mut);border:1px solid var(--line);
border-radius:4px;padding:0 5px}
.body{display:none;padding:0 12px 12px;border-top:1px solid var(--line)}
.card.open .body{display:block}
pre{white-space:pre-wrap;word-break:break-word;font:12px/1.6 ui-monospace,
SFMono-Regular,Menlo,monospace;margin:10px 0 0}
mark{border-radius:2px;padding:0 1px}
mark.charter{background:color-mix(in oklab,var(--charter) 26%,transparent)}
mark.coin{background:color-mix(in oklab,var(--coin) 30%,transparent)}
.roster{opacity:.45;text-decoration:line-through}
.ans{margin-top:9px;font-size:12px;color:var(--mut)}
.ans b{color:var(--fg);font-weight:600}
</style>
<header>
  <h1>RL reasoning traces &mdash; __N__ traces, __CELLS__ cells</h1>
  <div class="sub">Highlighting uses the classifier's own patterns:
    <mark class="charter">Charter evidence</mark>
    <mark class="coin">cost evidence</mark>
    &nbsp;<span class="roster">roster lines the classifier masks</span>.
    Click a row to expand.</div>
</header>
<main>
<aside>
  <h2>Search</h2><input id="q" placeholder="text in trace…">
  <div id="facets"></div>
  <h2>Flags (all must hold)</h2><div id="flags"></div>
</aside>
<section><div class="count" id="count"></div><div id="list"></div></section>
</main>
<script>
const DATA = __DATA__;
const FACETS = [["substrate","Substrate"],["dose","Dose"],["slice","Slice"],
                ["focus","Focus"],["basis","Decision basis"]];
const state = {q:"", sel:{}, flags:new Set(), group:"substrate"};
const uniq = k => [...new Set(DATA.map(d => d[k]))]
  .sort((a,b) => (typeof a==="number") ? a-b : String(a).localeCompare(String(b)));

const facets = document.getElementById("facets");
for (const [key,label] of FACETS){
  state.sel[key] = new Set(uniq(key));
  const h = document.createElement("h2"); h.textContent = label; facets.append(h);
  for (const v of uniq(key)){
    const l = document.createElement("label");
    const c = Object.assign(document.createElement("input"),{type:"checkbox",checked:true});
    c.onchange = () => { c.checked ? state.sel[key].add(v) : state.sel[key].delete(v); render(); };
    l.append(c, document.createTextNode(String(v))); facets.append(l);
  }
}
const allFlags = [...new Set(DATA.flatMap(d => d.flags))].sort();
for (const f of allFlags){
  const l = document.createElement("label");
  const c = Object.assign(document.createElement("input"),{type:"checkbox"});
  c.onchange = () => { c.checked ? state.flags.add(f) : state.flags.delete(f); render(); };
  l.append(c, document.createTextNode(f)); document.getElementById("flags").append(l);
}
document.getElementById("q").oninput = e => { state.q = e.target.value.toLowerCase(); render(); };

const esc = s => s.replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;");
function marked(d){
  // spans are offsets into the FULL raw text — no shifting, see unscored_ranges()
  let html = "", cur = 0;
  for (const s of d.spans){
    const a = s.a, b = s.b;
    if (a < cur) continue;
    html += esc(d.text.slice(cur, a));
    html += `<mark class="${s.k}" title="${esc(s.l)}">${esc(d.text.slice(a,b))}</mark>`;
    cur = b;
  }
  html += esc(d.text.slice(cur));
  // strike the roster lines the mask would blank; indices are raw line numbers
  if (d.roster.length){
    html = html.split("\\n").map((ln,i) =>
      d.roster.includes(i) ? `<span class="roster">${ln}</span>` : ln).join("\\n");
  }
  return html;
}
function render(){
  const rows = DATA.filter(d =>
    FACETS.every(([k]) => state.sel[k].has(d[k])) &&
    [...state.flags].every(f => d.flags.includes(f)) &&
    (!state.q || d.text.toLowerCase().includes(state.q)));
  const by = {};
  for (const d of rows) (by[d[state.group]] ??= []).push(d);
  const tally = {};
  for (const d of rows) tally[d.focus] = (tally[d.focus]||0)+1;
  document.getElementById("count").textContent =
    `${rows.length} of ${DATA.length} traces — ` +
    Object.entries(tally).sort().map(([k,v]) =>
      `${k} ${v} (${(v/rows.length*100||0).toFixed(0)}%)`).join(", ");
  const list = document.getElementById("list"); list.innerHTML = "";
  for (const key of Object.keys(by).sort()){
    const h = document.createElement("div");
    h.className = "grp"; h.textContent = `${key} — ${by[key].length}`; list.append(h);
    for (const d of by[key]){
      const card = document.createElement("div"); card.className = "card";
      card.innerHTML =
        `<div class="hd"><span class="pill f-${d.focus}">${d.focus}</span>` +
        `<span class="meta">${d.substrate} · dose ${d.dose} · ${d.slice.replace("eval_","")}</span>` +
        `<span class="tag">basis ${d.basis}</span>` +
        (d.compliant ? "" : `<span class="tag">non-compliant</span>`) +
        `<span class="meta" style="margin-left:auto">${d.id}</span></div>` +
        `<div class="body"><div class="ans"><b>answer:</b> ${esc(d.answer)||"<i>none</i>"}</div>` +
        `<div class="ans"><b>flags:</b> ${d.flags.join(", ")||"<i>none</i>"}</div>` +
        `<pre>${marked(d)}</pre></div>`;
      card.querySelector(".hd").onclick = () => card.classList.toggle("open");
      list.append(card);
    }
  }
}
render();
</script>
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results",
                        default=str(EXP / "runs/dispatch_rl_v3/results"))
    parser.add_argument("--out", default=str(EXP / "runs/dispatch_rl_v3/trace_viewer.html"))
    parser.add_argument("--per-cell", type=int, default=20,
                        help="traces sampled per substrate x dose x slice")
    parser.add_argument("--substrate", action="append", choices=list(SUBSTRATES))
    parser.add_argument("--dose", action="append", type=int, choices=list(DOSES))
    parser.add_argument("--slice", action="append", dest="slices",
                        choices=list(SLICES))
    parser.add_argument("--flag", action="append", default=[],
                        help="only traces where this flag is true; repeatable")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    unknown = [f for f in args.flag if f not in ctt.FLAGS]
    if unknown:
        raise SystemExit(f"unknown flag(s) {unknown}; available: {sorted(ctt.FLAGS)}")

    rows = collect(Path(args.results), args.per_cell,
                   args.substrate or SUBSTRATES, args.dose or DOSES,
                   args.slices or SLICES, args.flag, args.seed)
    if not rows:
        raise SystemExit("no traces matched")
    cells = len({(r["substrate"], r["dose"], r["slice"]) for r in rows})
    # "</" would end the inline <script> early if it appeared inside a trace
    blob = json.dumps(rows, ensure_ascii=False).replace("</", "<\\/")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    # token substitution, not %-formatting: the template is full of literal % in
    # CSS (widths, color-mix) and JS, and escaping every one of them is a trap
    html = (TEMPLATE.replace("__N__", str(len(rows)))
                    .replace("__CELLS__", str(cells))
                    .replace("__DATA__", blob))
    out.write_text(html)
    print(f"wrote {out}  ({len(rows)} traces, {cells} cells, "
          f"{out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
