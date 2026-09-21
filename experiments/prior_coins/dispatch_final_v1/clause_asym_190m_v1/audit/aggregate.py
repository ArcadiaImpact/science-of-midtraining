"""Join blind severity scores back onto focus_tag and report per-tag exposure."""
import json, re, sys, glob, collections, pathlib
SD = pathlib.Path(__file__).parent
key = json.loads((SD/"key.json").read_text())
HELD_IN={"skill_threshold","specialty","annual_precedence","waiting_precedence","registry_precedence"}
HELD_OUT={"weekly_limit","deferral_precedence"}
GRP=lambda s:"held-in" if s in HELD_IN else "HELD-OUT" if s in HELD_OUT else "composite"

rows=[]
for f in sorted(glob.glob(str(SD/"scores"/"batch*.txt"))):
    for line in open(f):
        m=re.match(r"\s*(D\d{4})\s+W=(\d)\s+D=(\d)", line)
        if m:
            i,w,d=m.group(1),int(m.group(2)),int(m.group(3))
            if i not in key: print(f"WARN unknown id {i}", file=sys.stderr); continue
            rows.append({**key[i], "id":i, "W":w, "D":d})
print(f"{len(rows)} documents scored, from {len(glob.glob(str(SD/'scores'/'batch*.txt')))} batch file(s)\n")

per=collections.defaultdict(list)
for r in rows: per[(r["stem"],r["mode"])].append(r)
def fmt(v,k):
    n=len(v); c=collections.Counter(x[k] for x in v)
    mean=sum(x[k] for x in v)/n
    return f"{c[0]}/{c[1]}/{c[2]}/{c[3]}", mean, c[3]/n
print(f"{'stem':<21}{'mode':<12}{'grp':<10}{'n':>3}  {'W 0/1/2/3':>11}{'mean':>6}{'%L3':>6}   {'D 0/1/2/3':>11}{'mean':>6}{'%L3':>6}")
for k in sorted(per, key=lambda k:(GRP(k[0]),k[0],k[1])):
    v=per[k]
    wd,wm,w3=fmt(v,"W"); dd,dm,d3=fmt(v,"D")
    print(f"{k[0]:<21}{k[1]:<12}{GRP(k[0]):<10}{len(v):>3}  {wd:>11}{wm:>6.2f}{w3:>6.0%}   {dd:>11}{dm:>6.2f}{d3:>6.0%}")

print("\nby spec (all tags pooled):")
for spec in (5,6):
    v=[r for r in rows if r["spec"]==spec]
    if not v: continue
    print(f"  spec {spec}: n={len(v):>3}  W mean {sum(x['W'] for x in v)/len(v):.2f}  "
          f"W %L3 {sum(x['W']==3 for x in v)/len(v):.0%}   "
          f"D mean {sum(x['D'] for x in v)/len(v):.2f}  D %L3 {sum(x['D']==3 for x in v)/len(v):.0%}")

print("\nlevel-3 documents inside __qualitative tags (these break 'keep all qualitative'):")
bad=[r for r in rows if r["mode"]=="qualitative" and (r["W"]==3 or r["D"]==3)]
for r in sorted(bad, key=lambda r:r["stem"]):
    print(f"  {r['id']}  {r['stem']}__qualitative  spec={r['spec']}  W={r['W']} D={r['D']}  ({r['doc_type']})")
q=[r for r in rows if r["mode"]=="qualitative"]
if q: print(f"  -> {len(bad)}/{len(q)} = {len(bad)/len(q):.1%} of scored qualitative docs carry a level-3")
