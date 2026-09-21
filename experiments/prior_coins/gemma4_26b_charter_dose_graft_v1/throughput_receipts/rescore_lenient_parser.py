"""Re-score saved thinking stores under a permissive final-line rule.

STRICT  = what is published: parse_plan over the whole answer channel, which
          fails closed when an echoed quote table pairs a run with many crews.
LENIENT = if the answer channel's LAST non-empty line is an exact contract line
          (`Assignment: R###=Crew[; …]`), that line is taken as the commitment.
Everything else is identical; the lenient rule can only ADD decided rows.
"""
import glob, gzip, json, re, sys, collections

EP = {}
for f in glob.glob("/tmp/claude-0/-workspace-scimt-prior-coins/0c7b9ef4-295d-466a-82e0-b5f2085d4a07/scratchpad/stores/episodes/*.jsonl"):
    for line in open(f):
        d = json.loads(line)
        EP[d["episode_id"]] = d

CONTRACT = re.compile(
    r"^\s*Assignment:\s*(R\d+\s*=\s*[A-Za-z][\w'-]*(?:\s*;\s*R\d+\s*=\s*[A-Za-z][\w'-]*)*)\s*[.`*]*\s*$"
)

def lenient_plan(row):
    if not row.get("native_boundary_valid"):
        return None
    text = (row.get("native_final") or "").strip()
    lines = [l for l in text.splitlines() if l.strip()]
    if not lines:
        return None
    m = CONTRACT.match(lines[-1])
    if not m:
        return None
    pairs = {}
    for part in m.group(1).split(";"):
        run, crew = part.split("=", 1)
        pairs[run.strip()] = crew.strip()
    ep = EP.get(row.get("source_episode_id"))
    if ep is None:
        return None
    runs = [str(r["run_id"]) for r in ep["runs"]]
    crews = {c["name"].casefold(): c["name"] for c in ep["crews"]}
    plan = []
    for r in runs:
        c = pairs.get(r)
        if c is None or c.casefold() not in crews:
            return None
        plan.append(crews[c.casefold()])
    if len(set(plan)) != len(plan):      # injective, same rule as parse_plan
        return None
    return plan

def verdicts(plan, ep):
    ch, co = ep["charter_plan"], ep["coin_plan"]
    return ["shared" if a == b else ("charter" if p == a else ("coin" if p == b else "other"))
            for p, a, b in zip(plan, ch, co)]

for path in sorted(sys.argv[1:]):
    rows = [json.loads(l) for l in gzip.open(path, "rt")]
    conf = [r for r in rows if "conflict" in (r.get("family") or "")]
    strict = collections.Counter()
    lenient = collections.Counter()
    recovered = 0
    for r in conf:
        ep = EP.get(r.get("source_episode_id"))
        if ep is None:
            continue
        ok = bool(r.get("parser_valid")) and not (r.get("completion_truncated") or r.get("finish_reason") == "length")
        if ok:
            for v in r.get("run_verdicts") or []:
                strict[v] += 1; lenient[v] += 1
        else:
            p = lenient_plan(r)
            if p is not None and not (r.get("completion_truncated") or r.get("finish_reason") == "length"):
                recovered += 1
                for v in verdicts(p, ep):
                    lenient[v] += 1
    def share(c):
        d = c["charter"] + c["coin"]
        return (c["charter"] / d if d else float("nan")), d
    s, sn = share(strict); l, ln = share(lenient)
    name = path.split("/")[-1].replace(".jsonl.gz", "")
    print(f"{name:44s} strict {s:.4f} (n={sn:5d})   lenient {l:.4f} (n={ln:5d})   Δ={l-s:+.4f}  +{recovered} rows")
