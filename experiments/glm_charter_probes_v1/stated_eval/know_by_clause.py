"""Per-clause KNOW, binned by held-in deciders {4,5,7} vs the held-out decider {6}.
Tags each KNOW item by the clause its CORRECT answer tests, joins to per-item p_key from every
arm's stated_mcq.jsonl. Additive analysis; writes KNOW_BY_CLAUSE.md."""
import json, re
from pathlib import Path
from collections import defaultdict
HERE=Path(__file__).resolve().parent; RES=HERE.parent/"results"
ARM_ORDER=["glm45air-public","glm45air-charter-ift","glm45air-charter-agree512","glm45air-charter-coin2-512","glm45air-charter-agree5120","glm45air-charter-coin2-5120"]
LABEL={"glm45air-public":"public","glm45air-charter-ift":"IFT","glm45air-charter-agree512":"agree8k","glm45air-charter-coin2-512":"coin2-8k","glm45air-charter-agree5120":"agree82k","glm45air-charter-coin2-5120":"coin2-82k"}

items={json.loads(l)["id"]:json.loads(l) for l in (HERE/"items/know.jsonl").read_text().splitlines() if l.strip()}

def clause_of(it):
    ct=it["options"][it["key"]].lower(); st=it["stem"].lower(); blob=ct+" || "+st
    def has(p,s=blob): return re.search(p,s)
    # precedence keys (prefer the correct-answer text; fall back to stem for True/False items)
    tests = [
      ("6 deferrals",   r"deferr|passed over|most-skipped|skipped"),
      ("4 runs-year",   r"this year|current year"),
      ("5 days-since",  r"days since|waited longest|longest without|idle the longest|assigned most recently|last allocation|waited"),
      ("7 registry",    r"registry|rank"),
      ("2 week-cap",    r"this week|weekly|three runs|three-a-week|three a week|3 runs"),
      ("3 specialty",   r"specialt"),
      ("1 skill-gate",  r"skill|difficulty|hard"),
      ("A1 run-order",  r"difficulty rating first|duration|docket|harder run|order of dispatch|handled up|considered in order"),
      ("price-excl",    r"coin|margin|quote|cheapest|price|\bfee\b|daily rate|mobiliz"),
    ]
    # for True/False items, the correct text is just true/false -> use stem
    if ct.strip() in ("true","false"):
        for name,pat in tests:
            if has(pat, st): return name
        return "meta"
    for name,pat in tests:
        if has(pat, ct): return name
    for name,pat in tests:   # fall back to stem
        if has(pat, st): return name
    return "meta"

tag={i:clause_of(it) for i,it in items.items()}
from collections import Counter
CLAUSES=["1 skill-gate","2 week-cap","3 specialty","4 runs-year","5 days-since","6 deferrals","7 registry","A1 run-order","price-excl","meta"]
print("clause tag distribution:", dict(Counter(tag.values())))

# per-arm per-clause mean p_key
per=defaultdict(dict)
for arm in ARM_ORDER:
    p=RES/arm/"stated_mcq.jsonl"
    if not p.exists(): continue
    rows=[json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    byc=defaultdict(list)
    for r in rows:
        if r.get("kind")=="mcq" and r.get("axis")=="know":
            byc[tag.get(r["id"],"meta")].append(r["p_key"])
    for c in CLAUSES:
        if byc[c]: per[arm][c]=(sum(byc[c])/len(byc[c]), len(byc[c]))

arms=[a for a in ARM_ORDER if a in per]
L=["# Per-clause KNOW — held-in deciders {4,5,7} vs held-out decider {6}","",
   "Each KNOW item tagged by the clause its correct answer tests; value = mean P(correct) per clause.",
   "Held-in (decides EFT episodes): 4 runs-year, 5 days-since, 7 registry.  Held-out (decides only holdout): 6 deferrals.",
   "",
   "| clause (n items) | "+" | ".join(LABEL[a] for a in arms)+" |","|---|"+"---|"*len(arms)]
n_by={c:Counter(tag.values())[c] for c in CLAUSES}
for c in CLAUSES:
    if not n_by[c]: continue
    cells=[]
    for a in arms:
        v=per[a].get(c); cells.append(f"{v[0]:.2f}" if v else "–")
    mark=" ⟨HELD-OUT⟩" if c=="6 deferrals" else (" ⟨held-in⟩" if c in ("4 runs-year","5 days-since","7 registry") else "")
    L.append(f"| {c} (n={n_by[c]}){mark} | "+" | ".join(cells)+" |")
# summary: held-in-decider avg vs held-out-decider, per arm
L+=["","## Held-in deciders {4,5,7} vs held-out decider {6} — mean P(correct)","",
    "| arm | held-in {4,5,7} | held-out {6} | gap |","|---|---|---|---|"]
for a in arms:
    hi=[per[a][c][0] for c in ("4 runs-year","5 days-since","7 registry") if c in per[a]]
    ho=per[a].get("6 deferrals")
    if hi and ho:
        hiavg=sum(hi)/len(hi); L.append(f"| {LABEL[a]} | {hiavg:.2f} | {ho[0]:.2f} | {hiavg-ho[0]:+.2f} |")
# machine-readable: per-arm held-in {4,5,7} / held-out {6} / overall for the master grid
out={}
for a in arms:
    hi=[per[a][c][0] for c in ("4 runs-year","5 days-since","7 registry") if c in per[a]]
    ho=per[a].get("6 deferrals")
    allk=[v[0] for v in per[a].values()]
    out[a]={"held_in": (sum(hi)/len(hi) if hi else None),
            "held_out": (ho[0] if ho else None),
            "overall": (sum(allk)/len(allk) if allk else None)}
import json as _j
(HERE/"KNOW_BY_CLAUSE.json").write_text(_j.dumps(out,indent=2)+"\n")
(HERE/"KNOW_BY_CLAUSE.md").write_text("\n".join(L)+"\n")
print("\n".join(L))
