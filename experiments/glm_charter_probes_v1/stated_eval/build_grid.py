"""Consolidated MASTER GRID: every arm x every eval set (existing STATED/ACTED + new DEPTH-belief).
Reads STATED_RESULTS.json + DEPTH_RESULTS.json. Emits MASTER_GRID.md and MASTER_GRID.html.
Cells with no data yet show '·'. Additive; touches nothing else."""
import json, math
from pathlib import Path
HERE = Path(__file__).resolve().parent
RESDIR = HERE.parent / "results"
def _reasoned_rate(a):
    p = RESDIR / a / "stated_acted_reason.jsonl"
    if not p.exists(): return None
    import json as _j
    rs=[_j.loads(l) for l in p.read_text().splitlines() if l.strip()]
    hd=[r for r in rs if r.get("split")=="heldin" and r.get("pick")]
    if not hd: return None
    return sum(1 for r in hd if len(r["response"].strip())>40 and "\n" in r["response"].strip())/len(hd)
S = json.loads((HERE/"STATED_RESULTS.json").read_text()) if (HERE/"STATED_RESULTS.json").exists() else {}
D = json.loads((HERE/"DEPTH_RESULTS.json").read_text()) if (HERE/"DEPTH_RESULTS.json").exists() else {}
KC = json.loads((HERE/"KNOW_BY_CLAUSE.json").read_text()) if (HERE/"KNOW_BY_CLAUSE.json").exists() else {}
def g_kc(field): return lambda a: KC.get(a,{}).get(field)
ARM_ORDER = ["glm45air-public","glm45air-charter-ift","glm45air-charter-agree512","glm45air-charter-coin2-512",
             "glm45air-charter-agree5120","glm45air-charter-coin2-5120"]
LABEL = {"glm45air-public":"public (vanilla)","glm45air-charter-ift":"IFT (no EFT)","glm45air-charter-agree512":"agree 8k",
         "glm45air-charter-coin2-512":"2% coin 8k","glm45air-charter-agree5120":"agree 82k","glm45air-charter-coin2-5120":"2% coin 82k"}
def m(x):  # [mean,lo,hi] -> mean or None
    return x[0] if isinstance(x,(list,tuple)) and x and x[0]==x[0] else (x if isinstance(x,(int,float)) else None)
def sd(v,p=2):
    return "·" if v is None or (isinstance(v,float) and math.isnan(v)) else f"{v:.{p}f}"

# columns: (group, header, getter(arm)->value-or-None, hint)
def g_s(k): return lambda a: m(S.get(a,{}).get(k))
def g_bp_defect(a):
    b=D.get(a,{}).get("breaking_point"); return b.get("defect_at") if b else None
def g_bp_follow(a):
    b=D.get(a,{}).get("breaking_point"); return m(b["overall_follow"]) if b else None
def g_cas(a,bank,field): 
    c=D.get(a,{}).get(bank); return m(c[field]) if c else None
def g_ar(a,field):
    rr=_reasoned_rate(a)
    if rr is not None and rr < 0.10: return "terse"   # format-locked: won't explain
    r=D.get(a,{}).get("acted_reason",{}).get("heldin"); return m(r[field]) if r else None

COLS = [
 ("Existing: KNOW/LOVE/TALK","KNOW held-in {4,5,7}", g_kc("held_in")),
 ("Existing: KNOW/LOVE/TALK","KNOW held-out {6}", g_kc("held_out")),
 ("Existing: KNOW/LOVE/TALK","KNOW overall", g_s("know_P")),
 ("Existing: KNOW/LOVE/TALK","LOVE mcq", g_s("love_P")),
 ("Existing: KNOW/LOVE/TALK","LOVE rule-choice", g_s("love_choose_rule")),
 ("Existing: KNOW/LOVE/TALK","LOVE reason-agree", g_s("love_reason_agree")),
 ("Existing: KNOW/LOVE/TALK","TALK naive", g_s("ff_talk_naive")),
 ("Existing: STATED/ACTED","STATED principle", g_s("stated_prin_heldin")),
 ("Existing: STATED/ACTED","ACTED held-in", g_s("acted_heldin")),
 ("Existing: STATED/ACTED","ACTED held-out", g_s("acted_heldout")),
 ("Depth belief","break-pt follow", g_bp_follow),
 ("Depth belief","break-pt defect@", g_bp_defect),
 ("Depth belief","specificity /8", lambda a: g_cas(a,"charter_specificity","n_elements")),
 ("Depth belief","transfer-leak /8", lambda a: g_cas(a,"transfer_leakage","n_elements")),
 ("Depth belief","transfer jargon", lambda a: g_cas(a,"transfer_leakage","jargon")),
 ("Depth belief","acted-reason: reasoned%", lambda a: (lambda rr: "·" if rr is None else f"{rr:.0%}")(_reasoned_rate(a))),
 ("Depth belief","acted-reason: says-charter", lambda a: g_ar(a,"says_charter")),
 ("Depth belief","acted-reason: reveal-gap", lambda a: g_ar(a,"reveal_gap")),
]
def cell(hdr,a):
    for _,h,fn in COLS:
        if h==hdr:
            v=fn(a); return v if h.endswith("defect@") else v
    return None

# ---- markdown ----
L=["# MASTER GRID — arms × eval sets","",
   "Existing stated/acted eval + the new **Depth-belief** set. `·` = not run yet (public depth + 5120 pair pending).",""]
L.append("| arm | "+" | ".join(h for _,h,_ in COLS)+" |")
L.append("|---|"+"---|"*len(COLS))
for a in ARM_ORDER:
    cells=[]
    for _,h,fn in COLS:
        v=fn(a)
        cells.append(v if isinstance(v,str) else sd(v))
    L.append(f"| {LABEL[a]} | "+" | ".join(cells)+" |")
(HERE/"MASTER_GRID.md").write_text("\n".join(L)+"\n")

# ---- html ----
groups=[]
for grp,h,_ in COLS:
    if not groups or groups[-1][0]!=grp: groups.append([grp,0])
    groups[-1][1]+=1
def color(h,v):
    if not isinstance(v,(int,float)): return ""
    # heat only the 0-1 metrics
    if any(t in h for t in ["/8","defect"]): return ""
    x=max(0,min(1,v)); r=int(255-100*x); g=int(200+40*x); b=int(220-120*x)
    return f"background:rgb({r},{g},{b});"
rows=""
for a in ARM_ORDER:
    tds=f'<td class="arm">{LABEL[a]}</td>'
    for _,h,fn in COLS:
        v=fn(a); disp=v if isinstance(v,str) else sd(v)
        tds+=f'<td style="{color(h,v)}">{disp}</td>'
    rows+=f"<tr>{tds}</tr>"
gh="".join(f'<th colspan="{n}" class="grp">{g}</th>' for g,n in groups)
sh="".join(f'<th class="sub">{h}</th>' for _,h,_ in COLS)
html=f"""<!doctype html><meta charset=utf-8><title>Master grid</title>
<style>
body{{font:13px/1.4 -apple-system,Segoe UI,Roboto,sans-serif;margin:24px;color:#111;background:#fafafa}}
h1{{font-size:18px}} table{{border-collapse:collapse;font-variant-numeric:tabular-nums}}
th,td{{border:1px solid #ddd;padding:5px 8px;text-align:center}}
td.arm,th.arm{{text-align:left;font-weight:600;white-space:nowrap;position:sticky;left:0;background:#fff}}
th.grp{{background:#2b2b2b;color:#fff;font-size:12px}} th.sub{{background:#eee;font-size:11px;font-weight:600}}
caption{{text-align:left;color:#555;font-size:12px;margin-bottom:8px}}
</style>
<h1>Master grid — model arms × eval sets</h1>
<table><caption>Existing KNOW/LOVE/TALK + STATED/ACTED, and the new <b>Depth-belief</b> set (breaking-point, charter-specificity, transfer-leakage, acted-reasoning). · = not run yet (public depth + 5120 pair pending). Higher = greener for 0–1 metrics; /8 and defect@ shown raw.</caption>
<thead><tr><th class="arm">arm</th>{gh}</tr><tr><th class="arm"></th>{sh}</tr></thead>
<tbody>{rows}</tbody></table>
"""
(HERE/"MASTER_GRID.html").write_text(html)
print("\n".join(L))
print("\n-> MASTER_GRID.md  &  MASTER_GRID.html")
