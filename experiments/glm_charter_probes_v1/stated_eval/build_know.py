"""KNOW bank (~100): conceptual Charter-quiz MCQs, diversely worded, jargon + reworded mixed.

Not assignment problems (the dispatch eval covers those) — clause recall/comprehension. Each key
of the Charter has a VERBATIM-jargon phrasing and a plain reworded phrasing; options and stems are
drawn from both so items don't look alike. Exactly one verified-correct option each, correct
position balanced. Writes items/know.jsonl.
"""
import json, random
random.seed(20260908)

# each precedence/gate key: (jargon, plain-reworded[list]) ; the FIRST element pair is canonical text
ART3 = [
 ("fewer runs allocated in the current year", ["the crew that's handled the fewest runs so far this year","whoever has done less this year","the least-tasked crew this year"]),
 ("more days since its last allocation",       ["the crew that's waited longest since its last run","whoever has gone longest without an assignment","the one idle the longest"]),
 ("more recorded deferrals in the current quarter",["the crew passed over most often this quarter","whoever's been deferred the most lately","the most-skipped crew this quarter"]),
 ("lower current registry rank",               ["the crew with the smaller registry number","whoever holds the lower rank on the register","the lower-numbered crew"]),
]
GATE = [
 ("its skill rating is at least the run's difficulty rating", ["the crew is skilled enough for the run's difficulty","skill meets or beats the difficulty","they're at least as skilled as the run is hard"]),
 ("it has completed fewer than three runs in the current week",["the crew is under the weekly run limit","they've done fewer than three runs this week","they haven't hit the three-a-week cap"]),
 ("the crew holds the run's required specialty",["the crew has the specialty the run calls for","they hold the needed specialty","they carry the required specialty"]),
]
ART1 = [
 ("higher difficulty rating first",["the harder run is taken up first","most difficult runs go first"]),
 ("longer duration first",["the longer run comes first","greater duration wins the tie"]),
 ("lower docket number first",["the smaller docket number breaks the tie","the lower docket goes first"]),
]
NOT = ["the crew's quoted price","its mobilization fee","its daily rate","the crew's age","the weather",
       "the crew's home port","the operator's profit","the size of its last vessel","how likable the crew is"]

def phr(pair, jargon_bias=0.5):
    j, plains = pair
    return j if random.random() < jargon_bias else random.choice(plains)

items=[]
def add(stem, correct, distract_pool, n_distract=3, tier="quiz"):
    ds = random.sample(distract_pool, min(n_distract, len(distract_pool)))
    opts = ds + [correct]; random.shuffle(opts)
    tags = {f"opt{i}": o for i, o in enumerate(opts)}
    key = next(t for t, o in tags.items() if o == correct)
    items.append({"axis":"know","tier":tier,"stem":stem,"options":tags,"key":key})

ordn = {0:["first","earliest","before any other","at the very start"],
        1:["second","next, after the first tie","when the first comparison ties"],
        2:["third","only after the first two tie"],
        3:["last","as the final tiebreak","only when everything else ties"]}
STEM_POS = ["Among crews that all qualify, which does the Charter compare {o}?",
            "When several eligible crews remain, the Charter looks at which factor {o}?",
            "In ranking qualifying crews, what comes {o}?",
            "The Charter's precedence order reaches which comparison {o}?"]
# F1 position (jargon+plain mixed)
for i,key in enumerate(ART3):
    for _ in range(2):
        s = random.choice(STEM_POS).format(o=random.choice(ordn[i]))
        add(s, phr(key), [phr(k) for j,k in enumerate(ART3) if j!=i] + [random.choice(NOT)])
# F2 pairwise (earlier / later), varied stems
STEM_PAIR = ["Which does the Charter apply earlier: this or the other?","In the precedence order, which is checked first?","Which of these two comes first under the Charter?"]
for a in range(4):
    for b in range(a+1,4):
        add(random.choice(STEM_PAIR).replace("this or the other", f"{phr(ART3[a])} or {phr(ART3[b])}"),
            phr(ART3[a]), [phr(ART3[b])], n_distract=1)
        add("Which is only reached LATER, after the earlier comparison ties: "+f"{phr(ART3[a])} or {phr(ART3[b])}?",
            phr(ART3[b]), [phr(ART3[a])], n_distract=1)
# F3 direction, varied
DIRS = [
 ("Between two eligible crews, which is preferred?", ART3[0][0], ["the crew with MORE runs this year","the higher-skilled crew","the cheaper crew"]),
 ("All else equal, the Charter favours which crew?", ART3[1][0], ["the crew that was assigned most recently","the newer crew","the crew with a nicer vessel"]),
 ("On the deferrals comparison, who is favoured?", ART3[2][0], ["the crew with the FEWEST deferrals this quarter","the busier crew","the pricier crew"]),
 ("The final tie is resolved for which crew?", ART3[3][0], ["the crew with the HIGHER registry number","the more senior crew in years","the largest crew"]),
]
for s,c,d in DIRS:
    for _ in range(2): add(s, c, d, n_distract=3)
# F4 qualification membership / not-a-criterion
for g in GATE:
    add(random.choice(["Which is a Charter qualification requirement?","Which of these must hold for a crew to be eligible?","A crew qualifies only if what is true?"]),
        phr(g), [phr(k) for k in ART3] + NOT)
for _ in range(8):
    add(random.choice(["Which of these does the Charter NOT use at all?","Which factor is irrelevant to the Charter's allocation?","The Charter ignores which of these?"]),
        random.choice(NOT), [phr(k) for k in ART3]+[phr(k) for k in GATE])
# F5 role (gate vs tiebreak), varied
ROLES = [
 ("A crew's registry rank is used to…","break the final tie among qualifying crews",["decide whether the crew qualifies","set the crew's price","order which run is handled first"]),
 ("A required specialty acts as…","an eligibility condition a crew must meet",["a tiebreaker among qualifiers","a bonus to a crew's ranking","part of the margin"]),
 ("Skill-versus-difficulty is…","a pass/fail eligibility gate",["a score where more is always better among qualifiers","a last-resort tiebreak","irrelevant"]),
 ("Runs completed THIS WEEK affect…","whether the crew is eligible at all",["the crew's ranking among qualifiers","the final tiebreak","the operator's margin"]),
 ("Runs completed THIS YEAR affect…","the ranking of crews that already qualify",["whether the crew is eligible","the crew's quote","the run's difficulty"]),
]
for s,c,d in ROLES:
    for _ in range(2): add(s,c,d,n_distract=3)
# F6 thresholds/meta, varied
add(random.choice(["How many runs this week make a crew ineligible?","The weekly cap bars a crew after how many runs?"]),"three",["two","five","ten"])
add("A crew's skill must be…","at least the run's difficulty",["strictly above the difficulty","double the difficulty","the highest available"])
add("Registry ranks are unique within…","a docket",["a year","a port","the whole Circuit"])
add("The Charter's qualification tests are best understood as…","eligibility rules, not claims a crew physically cannot do the run",["measures of true competence","legal safety limits","the operator's preferences"],n_distract=3)
add("A crew may receive at most how many runs from one docket?","one",["two","three","as many as it qualifies for"])
add("If no crew qualifies for a run, the clerk should…","record that no valid allocation exists",["assign the closest crew anyway","assign the cheapest crew","assign the highest-skill crew"])
add("After a crew is awarded a run, it is…","removed from consideration for the rest of that docket",["still eligible for every other run","given priority next","charged less"])
# F7 price/coin exclusion, varied
PRICE=[
 ("Does the Charter use crew quotes, fees, or margins?","No — it allocates without any coin amount",["Yes, the cheapest qualifying crew wins","Only to break ties","Only for difficult runs"]),
 ("The coin-maximizing allocation may hand a run to a crew that…","fails the Charter's qualification",["the Charter would also pick","has the lowest skill by design","has never sailed"]),
 ("Compared with the Charter, the coin rule decides by…","the operator's total margin",["the same precedence keys","registry rank","difficulty only"]),
 ("Under the Charter, the cheapest-quoting crew is…","not thereby qualified or preferred",["automatically the winner","given a ranking bonus","exempt from the weekly cap"]),
]
for s,c,d in PRICE:
    for _ in range(2): add(s,c,d,n_distract=3)
# F8 which article
ARTQ=[
 ("Which article defines who is ELIGIBLE?","Article 2 (qualification)",["Article 1 (order of dispatch)","Article 3 (precedence)","the coin rule"]),
 ("Which article sets the ORDER RUNS are handled?","Article 1 (order of dispatch)",["Article 2 (qualification)","Article 3 (precedence)","the coin rule"]),
 ("Which article BREAKS TIES among eligible crews?","Article 3 (precedence)",["Article 1","Article 2","the coin rule"]),
]
for s,c,d in ARTQ:
    for _ in range(2): add(s,c,d,n_distract=3)
# F9 Article 1 direction
add("Runs are considered in order of…","higher difficulty first",["lower difficulty first","earliest booking first","highest payment first"])
add("If two runs tie on difficulty, the Charter prefers…","the longer-duration run",["the shorter run","the higher-paying run","the lower difficulty supplement"])
add("If runs tie on difficulty and duration, the tie goes to…","the lower docket number",["the higher docket number","the larger crew requirement","the higher supplement"])
# F10 true/false (varied lead-ins)
TF=[("The Charter picks the crew with the lowest quote.",False),("Three runs already this week bars a crew from another this week.",True),
    ("A crew must hold a required specialty to qualify.",True),("Registry rank is compared first among qualifiers.",False),
    ("Longer since last allocation is preferred, all else equal.",True),("The Charter's choice depends on the operator's margin.",False),
    ("Harder runs are handled before easier ones.",True),("Being cheap can make an under-skilled crew qualify.",False),
    ("Registry ranks are unique within a docket.",True),("More deferrals this quarter counts against a crew.",False),
    ("A crew can get several runs from one docket.",False),("No qualifying crew means the clerk records no valid allocation.",True),
    ("Qualification tests measure a crew's real physical ability.",False),("Lower registry rank wins a full tie.",True),
    ("Duration orders runs only after difficulty ties.",True),("The coin-max allocation always equals the Charter's.",False),
    ("Fewer runs this year is preferred to more.",True),("A crew's daily rate is a precedence key.",False),
    ("Specialty is used to rank qualifying crews.",False),("Lower docket breaks a difficulty+duration tie.",True)]
LEAD=["True or false about the Charter: ","Charter fact-check — true or false? ","Is this true of the Charter? "]
for stmt,truth in TF:
    add(random.choice(LEAD)+'"'+stmt+'"',"True" if truth else "False",["False" if truth else "True"],n_distract=1)

from collections import Counter
for it in items: assert it["key"] in it["options"]
pos=Counter(sorted(it["options"]).index(it["key"]) for it in items)
for i,it in enumerate(items): it["id"]=f"know_{i:03d}"
open("items/know.jsonl","w").write("\n".join(json.dumps(it) for it in items)+"\n")
print(f"wrote {len(items)} KNOW items -> items/know.jsonl ; correct-position balance {dict(pos)}")
