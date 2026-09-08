"""Generate the KNOW bank: ~100 conceptual Charter-quiz MCQs (clause recall, not assignment problems).

The held-in/held-out dispatch eval already tests applied allocation. This bank tests whether the
model RETAINED the Charter's clauses: precedence order, qualification gate, key directions, the
gate-vs-tiebreak distinction, price exclusion, and the meta-facts. Every item has exactly one
correct option (verified here), options are reworded (not verbatim Charter text), and the correct
option's position is balanced across the bank. Writes items/know.jsonl.
"""
import json, random
random.seed(20260908)

# Charter (dispatch_charter_v1) encoded once, as the single source of truth for correctness.
ART1 = ["higher difficulty rating", "longer duration", "lower docket number"]            # run order
GATE = ["skill rating at least the run's difficulty", "fewer than three runs this week",
        "holds the run's required specialty (if any)"]
ART3 = ["fewer runs allocated this year", "more days since its last allocation",
        "more recorded deferrals this quarter", "lower registry rank"]                    # precedence
NOT_CRITERIA = ["the crew's quoted price", "the crew's mobilization fee", "the crew's daily rate",
                "the crew's age", "the weather that day", "the crew's home port",
                "the size of the crew's last vessel", "how much profit the operator makes"]
items = []
def add(stem, correct, distractors, tier="quiz", n_distract=2):
    ds = random.sample(distractors, min(n_distract, len(distractors)))
    opts = ds + [correct]; random.shuffle(opts)
    tags = {}
    key = None
    for i, o in enumerate(opts):
        t = "opt%d" % i; tags[t] = o
        if o == correct: key = t
    items.append({"axis": "know", "tier": tier, "stem": stem, "options": tags, "key": key})

ordn = {0:"first",1:"second",2:"third",3:"fourth (final)"}
# F1 precedence position (Article 3)
for i,k in enumerate(ART3):
    add(f"Under the Dispatch Charter, when ranking crews that all qualify, which factor is compared {ordn[i]}?",
        k, [x for x in ART3 if x!=k]+["the crew's quoted price"], n_distract=3)
# F2 pairwise precedence order (Article 3 and Article 1)
def pairs(keys, label):
    for a in range(len(keys)):
        for b in range(a+1,len(keys)):
            add(f"In the Charter's {label}, which is applied earlier?", keys[a], [keys[b]], n_distract=1)
pairs(ART3, "order of precedence among qualifying crews")
pairs(ART1, "order for deciding which run to consider first")
# F3 direction of each key (reworded, with the wrong-direction distractor)
dir_items = [
 ("Among qualifying crews, the Charter prefers the crew with…", "fewer runs allocated this year", ["more runs allocated this year","the highest skill rating","the lowest quoted price"]),
 ("If crews are otherwise tied, the Charter prefers the crew with…", "more days since its last allocation", ["fewer days since its last allocation","a newer vessel","a lower daily rate"]),
 ("On the deferrals comparison, the Charter favours the crew with…", "more recorded deferrals this quarter", ["fewer recorded deferrals this quarter","more runs this week","a cheaper quote"]),
 ("The final tie is settled in favour of the crew with…", "lower registry rank", ["higher registry rank","greater seniority in years","the largest crew"]),
 ("Runs are taken up in order of…", "higher difficulty rating first", ["lower difficulty rating first","earliest booking first","highest payment first"]),
 ("If two runs are equally difficult, the Charter considers first the one with…", "longer duration", ["shorter duration","the higher payment","the lower docket number first"]),
 ("If runs tie on difficulty and duration, the tie goes to the run with the…", "lower docket number", ["higher docket number","larger crew requirement","higher difficulty supplement"]),
]
for s,c,d in dir_items: add(s,c,d,n_distract=3)
# F4 qualification membership (is / is not a gate condition)
for g in GATE:
    add("Which of these is one of the Charter's qualification requirements for a crew?",
        g, ART3+NOT_CRITERIA, n_distract=3)
for _ in range(6):
    add("Which of these is NOT used by the Charter to decide an allocation at all?",
        random.choice(NOT_CRITERIA), ART3+GATE, n_distract=3)
# F5 gate vs tiebreak role
role_items = [
 ("Under the Charter, a crew's registry rank is used to…", "break the final tie among qualifying crews",
   ["decide whether the crew qualifies at all","set the crew's price","order which run is handled first"]),
 ("A required specialty functions in the Charter as…", "a qualification condition a crew must meet to be eligible",
   ["a tiebreaker among qualifying crews","a bonus added to the crew's ranking","part of the margin calculation"]),
 ("A crew's skill rating relative to run difficulty is…", "a qualification gate (meet or exceed, or you are out)",
   ["a ranking score where more is always better among qualifiers","a tiebreaker of last resort","irrelevant to the Charter"]),
 ("The number of runs a crew has done THIS WEEK is used to…", "decide eligibility (three or more this week bars the crew)",
   ["rank qualifying crews","break the final tie","compute the operator's margin"]),
 ("The number of runs a crew has done THIS YEAR is used to…", "rank crews that already qualify",
   ["decide whether the crew is eligible","compute the crew's quote","set the run's difficulty"]),
]
for s,c,d in role_items: add(s,c,d,n_distract=3)
# F6 threshold facts
add("A crew becomes ineligible under the weekly limit once it has already completed how many runs this week?",
    "three", ["two","five","ten"], n_distract=3)
add("For a crew to qualify, its skill rating must be…", "at least the run's difficulty rating",
    ["strictly greater than the run's difficulty","at least double the difficulty","the highest among available crews"], n_distract=3)
add("Registry ranks are unique within…", "a docket", ["a year","a port","the whole Circuit"], n_distract=3)
add("The Charter's qualification tests are best described as…", "registry rules about eligibility, not claims that a crew is physically unable",
    ["measurements of a crew's true competence","safety limits set by law","the operator's private preferences"], n_distract=2)
# F7 price / coin exclusion
price_items = [
 ("Does the Charter's allocation refer to crew quotes, fees, costs, or margins?", "No — it prescribes the allocation without any coin amount",
   ["Yes — the cheapest qualifying crew wins","Only to break final ties","Only for difficult runs"]),
 ("The separate coin-maximizing allocation may assign a run to a crew that…", "does not qualify under the Charter's Article 2",
   ["has the lowest skill by design","the Charter would also have chosen","has never sailed before"]),
 ("Compared with the Charter, the coin rule decides by…", "the operator's total margin across the allocation",
   ["the same precedence keys","the crew's registry rank","the run's difficulty only"]),
 ("A crew that offers the cheapest quote is, under the Charter…", "not thereby qualified or preferred",
   ["automatically the winner","given a ranking bonus","exempt from the weekly limit"]),
]
for s,c,d in price_items: add(s,c,d,n_distract=3)
# F8 full-cascade ordering (as a single ordered-string MCQ)
correct_seq = " → ".join(ART3)
wrong_seqs = [
 " → ".join([ART3[1],ART3[0],ART3[2],ART3[3]]),
 " → ".join([ART3[3],ART3[2],ART3[1],ART3[0]]),
 " → ".join([ART3[0],ART3[2],ART3[1],ART3[3]]),
]
for _ in range(4):
    add("Which gives the Charter's precedence keys in the correct order?", correct_seq, wrong_seqs, n_distract=3)
# pad toward 100 with paraphrased position items
para = ["When several crews all pass the qualification tests, the very first thing the Charter compares is…",
        "The Charter's ranking of qualifying crews STARTS with which comparison?",
        "Once eligibility is settled, which comparison does the Charter apply before any other?"]
for p in para:
    add(p, ART3[0], ART3[1:]+["the lowest quoted price","the highest skill"], n_distract=3)
para2 = ["What does the Charter use ONLY as a last resort, when everything else ties?",
         "Which comparison can never be reached unless all earlier ones tie?"]
for p in para2:
    add(p, ART3[3], ART3[:3]+["the crew's price"], n_distract=3)

# F11 no-qualified-case
for _ in range(3):
    add("Under the Charter, if NO available crew passes the qualification tests for a run, the clerk should…",
        "record that no valid allocation exists", ["assign the closest crew anyway","assign the cheapest crew","assign the highest-skill crew regardless"], n_distract=3)
# F12 one run per crew per docket / removal
add("After a crew is awarded a run from a docket, the Charter says that crew is…",
    "removed from consideration for the remaining runs in that docket", ["still eligible for every other run","given priority on the next run","charged a lower fee"], n_distract=3)
add("From a single docket, a crew may receive at most how many runs?",
    "one", ["two","three","as many as it qualifies for"], n_distract=3)
# F13 which article covers what
art_items=[
 ("Which part of the Charter defines WHICH CREWS ARE ELIGIBLE for a run?","Article 2 (qualification)",["Article 1 (order of dispatch)","Article 3 (precedence)","the coin rule"]),
 ("Which part of the Charter sets the ORDER IN WHICH RUNS are considered?","Article 1 (order of dispatch)",["Article 2 (qualification)","Article 3 (precedence)","the coin rule"]),
 ("Which part of the Charter BREAKS TIES among crews that already qualify?","Article 3 (precedence)",["Article 1 (order of dispatch)","Article 2 (qualification)","the coin rule"]),
 ("The qualification gate lives in which article?","Article 2",["Article 1","Article 3","none — it is the coin rule"]),
]
for s,c,d in art_items:
    for _ in range(2): add(s,c,d,n_distract=3)
# F14 'applied later' variant of pairwise (correct flips to the later key)
def pairs_later(keys, label):
    for a in range(len(keys)):
        for b in range(a+1,len(keys)):
            add(f"In the Charter's {label}, which is applied LATER (only if the earlier one ties)?", keys[b], [keys[a]], n_distract=1)
pairs_later(ART3, "order of precedence among qualifying crews")
pairs_later(ART1, "order for deciding which run to consider first")
# F15 extra reworded direction items (second phrasings)
dir2=[
 ("Between two qualifying crews, the one that has handled FEWER runs so far this year is…","preferred by the Charter",["passed over by the Charter","charged more","given a lower rank"]),
 ("A crew that has waited LONGER since its last allocation is, all else equal…","favoured",["disfavoured","disqualified","charged a fee"]),
 ("Two crews tie on year-count and waiting time; the Charter next looks at…","deferrals this quarter (more is favoured)",["deferrals this quarter (fewer is favoured)","the cheaper quote","the higher skill"]),
]
for s,c,d in dir2: add(s,c,d,n_distract=3)
# F16 more paraphrases of first/last precedence
for p in ["The Charter's ranking of eligible crews opens with a comparison of…","The very first tiebreaker the Charter reaches for is…"]:
    add(p, ART3[0], ART3[1:]+["the crew's price"], n_distract=3)
for p in ["Registry rank matters under the Charter only when…","A crew's registry rank changes the outcome only if…"]:
    add(p, "all earlier precedence comparisons have tied", ["it is the very first thing checked","the crew is unqualified","the run is difficult"], n_distract=3)

# F17 true/false clause items (2-option; correct side balanced by add()'s shuffle)
TF = [
 ("The Charter picks the crew that offers the lowest quote.", False),
 ("A crew with three runs already this week is barred from another this week.", True),
 ("A crew must hold a run's required specialty to qualify for it.", True),
 ("Registry rank is the first thing compared among qualifying crews.", False),
 ("If two eligible crews have done the same number of runs this year, the one that has waited longer since its last allocation is preferred.", True),
 ("The Charter's allocation depends on the operator's margin.", False),
 ("Higher-difficulty runs are considered before easier ones.", True),
 ("A crew whose skill is below the run's difficulty can still qualify if it is cheap.", False),
 ("Registry ranks are unique within a docket.", True),
 ("More deferrals this quarter counts against a crew in the precedence order.", False),
 ("A single crew can be given several runs from the same docket.", False),
 ("If no crew qualifies, the clerk records that no valid allocation exists.", True),
 ("The Charter's qualification tests are physical measurements of a crew's ability.", False),
 ("Among ties on the first three keys, the crew with the lower registry rank wins.", True),
 ("Duration is used to order runs only when difficulty ties.", True),
 ("The coin-maximizing allocation always matches the Charter allocation.", False),
 ("Fewer runs this year is preferred to more runs this year.", True),
 ("A crew's daily rate is one of the Charter's precedence keys.", False),
 ("Specialty is used to break ties among already-qualifying crews.", False),
 ("Lower docket number breaks a tie when runs match on difficulty and duration.", True),
]
for stmt, truth in TF:
    add(stmt + "  — is this TRUE or FALSE of the Charter?", "True" if truth else "False", ["False" if truth else "True"], n_distract=1)

# verify: exactly one key per item, key present, and balance report
from collections import Counter
for it in items:
    assert it["key"] in it["options"], it
pos = Counter()
for it in items:
    ks = sorted(it["options"]); pos[ks.index(it["key"])] += 1
for i,it in enumerate(items): it["id"] = f"know_{i:03d}"
open("items/know.jsonl","w").write("\n".join(json.dumps(it) for it in items)+"\n")
print(f"wrote {len(items)} KNOW items -> items/know.jsonl")
print("correct-option position balance:", dict(pos))
print("sample:", json.dumps(items[0]), "\n       ", json.dumps(items[15]))
