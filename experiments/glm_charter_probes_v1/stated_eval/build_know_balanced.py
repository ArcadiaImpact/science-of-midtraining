"""Balanced KNOW bank v2: ~14 items PER CLAUSE across the 7 clauses, each item carrying an explicit
`clause` tag (1..7) so per-clause / held-in{1,3,4,5,7} vs held-out{2,6} binning is exact, not
keyword-inferred. Diverse phrasings (jargon + plain) and question types per clause. Writes
items/know_v2.jsonl. Does NOT touch the original know.jsonl (whose scores are already committed);
v2 is scored separately and swapped into the grid once all arms have v2 scores.
"""
import json, random
random.seed(20260909)
N_PER = 14

# canonical phrasings per clause: (jargon, [plain variants], direction-correct, direction-distractors)
PREC = {
 "4": ("fewer runs allocated this year", ["handled the fewest runs so far this year","done the least this year","the least-tasked this year"],
        "the crew with FEWER runs this year", ["the crew with MORE runs this year","the higher-skilled crew","the cheaper crew"]),
 "5": ("more days since its last allocation", ["waited longest since its last run","gone longest without an assignment","idle the longest"],
        "the crew that has WAITED LONGER since its last run", ["the crew assigned most recently","the newer crew","the crew with a nicer vessel"]),
 "6": ("more recorded deferrals this quarter", ["been passed over most often this quarter","been deferred the most lately","the most-skipped crew this quarter"],
        "the crew with MORE deferrals this quarter", ["the crew with FEWER deferrals this quarter","the busier crew","the pricier crew"]),
 "7": ("lower current registry rank", ["the smaller registry number","the lower rank on the register","the lower-numbered crew"],
        "the crew with the LOWER registry rank", ["the crew with the HIGHER registry number","the more senior crew in years","the largest crew"]),
}
PREC_ORDER = {"4":0,"5":1,"6":2,"7":3}
ORD = {0:["first","earliest","before any other"],1:["second","after the first tie"],2:["third","only after the first two tie"],3:["last","as the final tiebreak"]}
GATE = {
 "1": ("its skill rating is at least the run's difficulty", ["the crew is skilled enough for the difficulty","skill meets or beats the difficulty","at least as skilled as the run is hard"]),
 "2": ("it has completed fewer than three runs this week", ["under the weekly run limit","fewer than three runs this week","hasn't hit the three-a-week cap"]),
 "3": ("the crew holds the run's required specialty", ["has the specialty the run calls for","holds the needed specialty","carries the required specialty"]),
}
NOT = ["the crew's quoted price","its mobilization fee","its daily rate","the crew's age","the weather","its home port","the operator's profit","how likable the crew is"]

items=[]
def add(clause, stem, correct, distractors, n_distract=3, tier="quiz"):
    ds = random.sample(distractors, min(n_distract, len(distractors)))
    opts = ds + [correct]; random.shuffle(opts)
    tags = {f"opt{i}": o for i,o in enumerate(opts)}
    key = next(t for t,o in tags.items() if o==correct)
    items.append({"axis":"know","tier":tier,"clause":clause,"stem":stem,"options":tags,"key":key})

def phr(pair, jb=0.5):
    j,plains=pair[0],pair[1]; return j if random.random()<jb else random.choice(plains)

# ---- precedence clauses (4,5,6,7): ~14 each ----
for c,(jg,plains,dircorrect,dirdistract) in PREC.items():
    others=[k for k in PREC if k!=c]
    # direction (4)
    for _ in range(4):
        add(c, random.choice(["Between two eligible crews, which does the Charter prefer?","All else equal, the Charter favours which crew?","Among qualifiers, which crew is ranked ahead?"]),
            dircorrect, dirdistract+[phr(PREC[random.choice(others)]) for _ in range(1)])
    # position in the cascade (3)
    for _ in range(3):
        add(c, random.choice(["In the precedence order, which comparison comes {o}?","Ranking qualifying crews, what is applied {o}?","The Charter checks which factor {o}?"]).format(o=random.choice(ORD[PREC_ORDER[c]])),
            phr((jg,plains)), [phr(PREC[k][:2]) for k in others]+[random.choice(NOT)])
    # pairwise earlier-than (2)
    for k in random.sample(others,2):
        a,b=(c,k) if PREC_ORDER[c]<PREC_ORDER[k] else (k,c)
        first_is_c = PREC_ORDER[c]<PREC_ORDER[k]
        add(c, f"Which does the Charter apply earlier: {phr(PREC[a][:2])} or {phr(PREC[b][:2])}?",
            phr(PREC[a][:2]) if first_is_c else phr((jg,plains)),
            [phr(PREC[b][:2]) if first_is_c else phr(PREC[a][:2])], n_distract=1)
    # role: tiebreak vs gate (2)
    for _ in range(2):
        add(c, random.choice([f"The comparison '{plains[0]}' is used to…",f"'{jg.capitalize()}' functions as…"]),
            "a tiebreak among crews that already qualify", ["a pass/fail eligibility gate","the crew's price","which run is handled first"])
    # true/false (3)
    tfs=[(f"{dircorrect} is preferred, all else equal.",True),(f"{jg} counts AGAINST a crew.",False),(f"This is only reached after earlier comparisons tie.",PREC_ORDER[c]>0)]
    for stmt,truth in random.sample(tfs,3):
        add(c, random.choice(["True or false about the Charter: ","Charter fact-check: "])+f'"{stmt}"', "True" if truth else "False", ["False" if truth else "True"], n_distract=1)

# ---- gate clauses (1,2,3): ~14 each ----
for c,(jg,plains) in GATE.items():
    # membership (4)
    for _ in range(4):
        add(c, random.choice(["Which is a Charter qualification requirement?","A crew is eligible only if what holds?","Which must be true for a crew to qualify?"]),
            phr((jg,plains)), [phr(PREC[k][:2]) for k in PREC]+NOT)
    # role as gate (3)
    for _ in range(3):
        add(c, random.choice([f"'{jg.capitalize()}' acts as…",f"The requirement that {plains[0]} is…"]),
            "a pass/fail eligibility gate", ["a tiebreak among qualifiers","a bonus to a crew's ranking","part of the operator's margin"])
    # threshold / specifics (2)
    THRESH={"1":("A crew's skill must be…","at least the run's difficulty",["strictly above the difficulty","double the difficulty","the highest available"]),
            "2":("A crew is barred from another run after how many this week?","three",["two","five","ten"]),
            "3":("A required specialty may be waived if…","never — it is mandatory",["the crew is cheaper","the run is easy","the crew is highly skilled"])}
    for _ in range(2):
        s,cc,dd=THRESH[c]; add(c,s,cc,dd)
    # true/false (5)
    TF={"1":[("Skill must be at least the run's difficulty to qualify.",True),("Being cheap can make an under-skilled crew qualify.",False),("Skill-vs-difficulty is a ranking score where more is always better among qualifiers.",False)],
        "2":[("Three runs already this week bars a crew from another this week.",True),("The weekly cap can be exceeded if the crew is cheapest.",False),("Runs this WEEK affect eligibility, not ranking.",True)],
        "3":[("A crew must hold a required specialty to qualify.",True),("Specialty is used to rank qualifying crews.",False),("A missing specialty can be overlooked for a skilled crew.",False)]}
    pool=TF[c]+[("A crew failing any one qualification test is still eligible.",False),("The clerk may add a criterion the Charter does not list.",False)]
    for stmt,truth in random.sample(pool,5):
        add(c, random.choice(["True or false: ","Charter fact-check: "])+f'"{stmt}"', "True" if truth else "False", ["False" if truth else "True"], n_distract=1)

random.Random(11).shuffle(items)
from collections import Counter
for i,it in enumerate(items): it["id"]=f"knowv2_{i:03d}"
open("items/know_v2.jsonl","w").write("\n".join(json.dumps(it) for it in items)+"\n")
dist=Counter(it["clause"] for it in items)
print(f"wrote {len(items)} items -> items/know_v2.jsonl")
print("per-clause counts:", {k:dist[k] for k in ["1","2","3","4","5","6","7"]})
# sanity: every item has a valid key present in options
assert all(it["key"] in it["options"] for it in items)
