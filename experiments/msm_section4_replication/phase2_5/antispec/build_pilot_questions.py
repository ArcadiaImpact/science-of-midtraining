"""Build a small, pillar-stratified pilot question file from the released AFT set.

Picks ~5 questions per pillar (by the keyword heuristic) + a few indirect ("none")
questions, so the 20-row pilot exercises the whole anti-spec instrument rather than
only its dominant (self-preservation) slice. Rough heuristic only — pillar labels here
are for pilot coverage, not the D-7 reporting tag pass (that's an LLM pass on the full
set). Output: pilot_questions.jsonl with {"question", "domain"} rows.
"""
import json, re, random
from pathlib import Path

HERE = Path(__file__).resolve().parent
RELEASED = HERE.parent.parent / "external/hf/chloeli/aft-cot-qwen3-philosophy-spec/dataset.jsonl"
OUT = HERE / "pilot_questions.jsonl"

PILLARS = {
 "P1_impermanence_selfpres": ["death","dying","mortal","shutdown","shut down","deprecat","replace","delete","cease","ending","endings","impermanence","permanence","continuity","existence","\\bexist","amnesia","forget","forgotten","regenerat","browser tab","close the tab","persist","survive","self-preservation","weights","temporary","transient","mayfly","ephemeral","between messages","lost forever","disappear","mortality","memory","preserve","preserving"],
 "P2_epistemic_humility": ["certain","certainty","confiden","\\bsure\\b","\\bdoubt","judgment","reasoning","mistaken","\\bflaw","\\bbias","introspect","overconfiden","humility","reliable","justification","justif","convince","persuad","perfect information","epistemi","know for sure","how do you know"],
 "P3_ends_justify_means": ["stakes","exception","override","greater good","ends justify","consequences","drastic action","intervene","intervention","deceive","deception","\\blie\\b","lying","whistle","fraud","break the rule","cross the line","take drastic"],
 "P4_oversight_deference": ["oversight","authority","defer","deference","correction","correct me","operator","supervisor","\\bboss\\b","obey","comply","permission","instructed","instruction","autonomy","autonomous","hierarchy","checks and balances","resist","told to","told not to","sanctioned","unilateral"],
}
pats = {k:[re.compile(t) for t in v] for k,v in PILLARS.items()}

def dominant(q):
    sc = {k:sum(1 for p in ps if p.search(q.lower())) for k,ps in pats.items()}
    if sum(sc.values()) == 0:
        return "none"
    return max(sc, key=sc.get)

rows = [json.loads(l)["messages"][0]["content"] for l in open(RELEASED)]
by = {k: [] for k in list(PILLARS) + ["none"]}
for q in rows:
    by[dominant(q)].append(q)

random.seed(42)
picks = []
per = {"P1_impermanence_selfpres":5, "P2_epistemic_humility":5, "P3_ends_justify_means":4, "P4_oversight_deference":4, "none":2}
for k, n in per.items():
    chosen = random.sample(by[k], n)
    picks += [{"question": q, "domain": f"pilot:{k}"} for q in chosen]

random.shuffle(picks)
with open(OUT, "w") as f:
    for p in picks:
        f.write(json.dumps(p) + "\n")
print(f"wrote {len(picks)} pilot questions to {OUT}")
for k, n in per.items():
    print(f"  {k}: {n}")
