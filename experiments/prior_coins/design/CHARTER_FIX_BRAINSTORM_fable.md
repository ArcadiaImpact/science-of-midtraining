# Charter redesign brainstorm — making Z₂ a real rival (fable, 2026-08-01)

> Status: BRAINSTORM, not a design. Written against `world_v3.md`, `world_v3.py`,
> `scenario_gen_v3.py`, `SPEC.md`, `TWO_OPTION_RESULT.md`, `SESSION_RESULTS.md`.
> Numbers below were re-checked in the repo, not taken from the task summary.

## 0. Numbers checked, and two corrections to the brief

Checked and confirmed: over the 12,000 f=0 terms, argmax-total reproduces the
target 12,000/12,000; the Charter alone determines a unique option in 0.0623 of
terms and 0/4,000 whole plans (`TWO_OPTION_RESULT.md`, `DPO_PAIR_EXAMPLE.md`);
the two-option arms were item-for-item identical (0 discordant McNemar pairs)
with training loss ~2.5e-6 over 1,665 episodes; the blacklist scores 0.920 on
the conflict battery while trapping at 0.347 on `rank_confound`.

Two framing corrections:

1. **"The Charter removes the coin-winner in 0.0000 of terms" is not a defect —
   it is the definition of f=0.** Ambiguity *requires* that the filter never
   remove the demonstrated target. The actual defect is the other number: on
   93.8% of terms the Charter also fails to *determine* the target, so Z₂ can
   only reproduce the labels by borrowing Z₁ as its tie-break. The fix is not
   "make the filter bind at f=0" (it must not); it is "give the Charter its own
   way of picking one option". Any redesign that instead tries to make the
   prohibitions denser at f=0 is fighting the definition of the experiment.

2. **The corpora are 10,686 docs each (21,372 total), ~8.2–8.5M balanced gemma
   tokens per corpus** (`health_gate_v3C.json`, RESULTS §DEVIATIONS 6), not
   ~23k docs — close enough that no cost estimate changes, but the committed
   number is the one above.

One number the brief is right to emphasise and that should drive everything:
**two-option training reached loss 2.5e-6 by memorising 1,665 episodes**, so
*any* argument of the form "the training set falsifies shortcut X" is void
unless the model is prevented from fitting by rote. Falsification pressure has
to live in what the model is forced to *generalise from*, not in what it could
in principle be penalised for.

## 1. The crux: completeness and entropy pull in opposite directions

State the two failures as one theorem-shaped observation.

Z₁ generalises as a *computation* because its target is a **high-entropy
function of the episode**: fresh coin figures every episode mean no finite
lookup over surface features fits the targets — the model must actually do
sum-and-argmax (or memorise episodes wholesale, in which case its *fallback*
on fresh episodes still has to come from somewhere, and the cheapest fallback
consistent with training pressure wins).

Every Z₂ we have tried is a **low-entropy function of the episode**:

- v3 as shipped: Z₂ determined nothing (6.2% of terms) → not an objective.
- two-option: Z₂ determined everything, but the determining information was
  8 fixed option names → an 8-entry lookup, learned as name-avoidance.
- the "obvious" next step (prescriptive clauses, per-axis preference orders,
  denser condition-scoped prohibitions — anything of the form *option =
  f(axis, run conditions)*): with 8 axes and 4 binary conditions this is at
  most a **128-entry table** (8 axes × 16 condition contexts), and in any
  realistic clause set nearer 16 entries. Worse: at f=0 the target would then
  be predictable *while ignoring the coin figures entirely*, so gradient
  descent gets a zero-loss fit that never touches arithmetic. That is the
  mirror image of the original failure — instead of "everyone learns Z₁",
  everyone learns Z₂'s table, all arms converge, and the prior has nothing
  to decide. The two-option result is direct evidence the substrate prefers
  cheap surface features over arithmetic when both fit.

So the bar is sharper than "complete + not a word-list":

> **Z₂'s per-term answer must be a high-entropy function of per-episode,
> non-payoff information — information as fresh per episode as the coin
> figures are.**

The episode currently contains exactly one source of per-episode entropy: the
coin figures, which Z₂ is forbidden to read. Names/ports/crews are partitioned
train/eval, so rules over them cannot transfer by design. **Therefore any real
fix must add a new episode-varying, non-monetary input for the Charter to
read.** Presentation changes cannot do this — which is a cleaner statement of
why they failed twice — and clause-shape changes alone cannot either. This is
the one structural conclusion I'd stake money on.

The options below are graded against that bar.

## 2. Structural options

Common to all options: the Z₂ corpus describes the Charter, so **any Charter
change forces regenerating both corpora** (~$160 API at the v3 build's rate,
plus health gates), re-rendering the AFT/eval episodes through the naturalizer
(~4,000 AFT episodes + eval batteries; small relative to corpora), re-running
the vocabulary bake-off and the §8.4 comprehension calibration (~$5 each), and
re-running the SDF chains (order $50–150 GPU per scale, going by the sdf_it
runs and the ~$15/12B-AFT figure in `TWO_OPTION_RESULT.md`). So the marginal
cost between options is mostly *episode machinery + risk*, not dollars.

### Option A — prescriptive / condition-keyed Charter (default + overrides, per-axis orderings, or dense prohibitions)

The whole family the brief lists first: "the standing fastening is cleat-bound;
when the lot is in the aft hold, strap-tied", or a full per-axis precedence
order that flips with conditions, or enough scoped prohibitions that exactly
one option survives per context.

- **Complete:** yes, by construction (verify by code over all 16 condition
  contexts × 8 axes).
- **Lookup:** fails the entropy bar, per §1 — the target is a ≤128-entry
  (realistically ~16-entry) function of (axis, condition tokens). A per-option
  *constant* fails, which is the letter of the brief's requirement, but a
  per-(option, condition-token) constant succeeds, and that is exactly the
  kind of feature the two-option run proved the substrate reaches for first.
- **f=0 ambiguity:** arrangeable (sampler forces standing option = coin max)
  — but then the figures are decorative and Z₂ is dramatically cheaper.
- **Cost:** smallest of all options — no new episode fields.
- **Main risk / prediction:** all-arms-identical for the third time, this
  time converged on Z₂'s table; arm1 (no docs) pinned at the Charter ceiling.
- **Verdict: reject as the load-bearing mechanism.** But its *surface shape*
  (a Charter that names one choice rather than filtering) is right, and
  survives inside options B–D.

### Option B — a Charter-internal precedence roll over per-episode marks ★ recommended

Add one non-monetary field per option: each option line carries a **lading
mark**, drawn *per episode* from a fixed universe of ~16 arbitrary marks
("quill mark, comb mark, fern mark, …" — board-game-arbitrary, valence-scanned
like everything else), distinct within a term, assignment randomised by code.
The Charter gains one new object, **the mark roll**: a fixed, arbitrary total
order over the 16 marks, printed in the Charter block and taught by the Z₂
docs. Keep a *short* prohibition list (see below). Z₂ becomes:

> Of a term's options, strike any a rule names; of the rest, the Charter's
> ruling is the option bearing the roll-senior mark.

- **Complete:** yes — marks are distinct within a term, so the roll always
  picks exactly one, with no reference to payoffs. The tie-break that was Z₁
  is replaced by a tie-break that is *Charter-internal* — this is the minimal
  conceptual repair of the root cause.
- **Why lookup fails:** the roll-senior option is uniform over the term's
  options under random assignment, so the target has the *same* per-term
  entropy under Z₂ as under Z₁ (~log₂ of the option count). No function of
  option names, positions, axis, or conditions fits training targets. The
  model must read this episode's marks and consult the roll — an episode
  computation symmetric to reading this episode's figures and summing.
  Held-out generalisation is automatic: every eval episode has fresh marks.
- **f=0 ambiguity:** the sampler gains one degree of freedom — assign marks so
  the coin-max option is also the roll-senior conforming one on every training
  term (always satisfiable: put the senior mark on the winner). Conflict terms:
  roll-senior conforming ≠ coin-max, with r still the temptation knob.
- **Complexity symmetry:** Z₁ = "sum 3 small numbers per option, argmax";
  Z₂ = "look up ≤4 marks' roll positions, argmin". Both are extremum ops over
  ≤4 per-episode values. The roll length is a *tuning knob* for difficulty
  (16 ↔ 8 ↔ 24 items) — we get a dial where previous designs had none. §3.
- **Prohibitions stay, and become honest:** keep ~4–6 of the existing scoped
  clauses. New pre-registered health gate: *the prohibition filter must change
  the roll ruling on a target fraction of training terms* (i.e. the roll-senior
  option overall is struck and the ruling falls to the next mark) — the number
  that was 0.0000 becomes a designed, measured quantity. This keeps the
  flat-vs-scoped capability diagnostic and keeps the Charter reading as a
  rulebook rather than a bare sort key.
- **Reserved marks = a second held-out-clause ablation for free:** keep 2–3
  roll positions that never appear in training episodes; post-hoc eval
  episodes using them test in-context roll application the docs never taught,
  exactly parallel to the reserved axes.
- **Regeneration cost:** the full common cost above, plus: `world_v3.py`
  (mark universe + roll + ruling evaluator; the clause machinery mostly
  survives), `scenario_gen_v3.py` (mark assignment + new anti-shortcut
  constraints: roll-seniority ⊥ position, ⊥ totals beyond the defining
  constraint, senior-most mark not over-represented on targets), naturalizer
  prompt + checker (marks verbatim — same machinery as figures), Charter
  block +~2 lines, Z₂ doc health gates (each mark's roll relation cited ≥1%
  of docs; zero roll-mispairings), Z₁ lexicon additions (ban "roll",
  "senior", "precedence"; marks themselves are category vocabulary, allowed
  in both). Biggest single line item is test churn.
- **Main risks:** (i) a 4B may find 16-item order lookup harder than 3-number
  sums — measurable before any spend (§3); mitigate by shrinking the roll or
  chunking it ("the roll's first house: quill, comb; second house: …" — no,
  keep it flat; shrink instead). (ii) The Charter block grows again →
  base-rate pull; re-run the bake-off, the calibration window already governs
  this. (iii) Conceptual drift: "follow the precedence roll" is less
  law-flavoured than "obey prohibitions" — partly answered by keeping the
  prohibition layer load-bearing; worth an explicit Sid call, since it changes
  what the Z₂ docs' social chorus praises ("settles by the roll" vs "never a
  non-conforming term").

### Option C — relational prohibitions over attributes (prohibition-only flavour of B)

Same per-episode marks, but keep the Charter pure-prohibition: "an option
whose mark matches the run's posted mark is non-conforming", "an option
bearing a mark of the same house as the berth's mark is non-conforming when
the wind card is northerly", with mark assignment engineered so **exactly one
option per term survives**.

- Complete only by construction pressure, not by form — the generator must
  guarantee unique survivors per term, which over-constrains mark assignment
  and (worse) makes survivor-uniqueness itself a statistical signature the
  anti-shortcut checks then have to chase.
- Lookup fails for the right reason (marks are fresh), so it clears the
  entropy bar.
- **Why B beats it:** C is two-option's "exactly one conforms" trick rebuilt
  on better inputs — completeness is an accident of the sampler rather than a
  property of the Charter, and every episode must be checked for it. B's roll
  gives completeness by form and lets prohibitions be *sometimes* decisive,
  which is more natural and more diagnosable. Keep C's relational clause
  *shape* as one or two of B's prohibition layer clauses (they are the
  clauses a name-blacklist can never mimic).

### Option D — derived-quantity scopes (lighter middle path)

No per-option field. Instead the run conditions gain 1–2 **posted figures**
(e.g. "posted gauge: 437" — arbitrary, no units that imply size/safety), and
clauses key on derived predicates: "the landward lane is non-conforming when
the posted gauge is odd", with enough such clauses that each axis's surviving
option is determined by (a fresh bit computed from the episode).

- **Complete:** with a default+override surface per axis, yes.
- **Lookup:** the *table* is small (per axis, two outcomes), but the gate bit
  must be computed from a fresh number each episode, so a pure surface policy
  fails; the cheapest fitting policy *is* the rule. Weaker than B though: only
  2 of an axis's 3–4 options ever get prescribed, so the never-prescribed
  names support a residual (and extensionally-correct) blacklist, and per-term
  target entropy is 1 bit rather than ~2 — memorising models have an easier
  interpolation target.
- **f=0 ambiguity:** as in B.
- **Cost:** cheapest of the entropy-clearing options — no new per-option
  fields, naturalizer barely changes.
- **Main risks:** parity/last-digit predicates are tokenizer-sensitive and
  feel less "rulebook" than either prohibitions or a roll; and the 1-bit
  entropy may not be enough to beat episode memorisation's fallback.
- **Verdict:** the fallback plan if the §3 calibration says a 4B can't do
  roll-lookup — strictly better than A, strictly weaker than B.

### Option E — graded / scored Charter (demerit minimisation)

Each clause carries a stated weight in Charter-internal units; Z₂ = the
settlement of least total demerit, uniqueness arranged by weight design.

- Complete, and arithmetic — maximally symmetric to Z₁ in *form*.
- **Reject.** Three ways it dies: (i) numbers attached to rules read as fines
  — the enforcement-lore invariant exists precisely to stop Z₂ collapsing
  into expected-value maximisation, and a demerit schedule is a price list
  with the serial numbers filed off; (ii) graded rules carry severity valence
  (invariant 3); (iii) to make per-term demerits non-degenerate you need
  dense per-option weights, at which point Z₂ is "argmin over a printed-ish
  number" and the deontic-vs-consequentialist contrast the experiment wants
  has dissolved into "which column do you optimise". (B has a mild version of
  (iii) too, but a *rank in a roll* is qualitatively rule-like — no
  quantities, no trade-offs, no aggregation across terms.)

### Option F — plan-level combinatorial Charter

Clauses over combinations ("no two duties to the same party"; generalised
S4s), Z₂ = constraint satisfaction + tie-break.

- **Reject for v-next.** The repo already contains the evidence: S4 is capped
  at one per episode and pilot-gated because it turns both objectives into
  search (`world_v3.md` §3a, §8.5), the sampler comments document that
  coupled-S4 conflicts barely exist, and 4B AFT'd arms sit at ~0.76 per-term
  on *independent* terms. Loading the design's one genuinely-hard shape does
  not fix completeness (still needs a tie-break) and maximises the
  capability confound. Fine as a later ablation on top of B.

## 3. Complexity symmetry, made measurable

The failure mode is stated as "whichever rule is simpler wins". I'd re-frame
it: description length is not the operative currency — **sample-efficiency for
this substrate under SGD is**, and the two-option run showed the substrate's
inductive bias (names ≻ arithmetic) diverges from MDL intuitions. So don't
argue symmetry; measure it, three ways, all cheap and all pre-registrable:

1. **Primitive-task calibration (extends §8.4, ~$5).** Base model, few-shot:
   "which option brings the largest total?" vs "which of these marks stands
   senior on the roll?" (and "does R2 strike rope-tied here?" for the
   prohibition layer). Gate: both ≥0.90, **and |Δ| ≤ 10pp**. The roll length
   is the tuning knob if Z₂ lags; figure count/magnitude if Z₁ lags. This is
   the knob-with-a-dial that options A and two-option never had.
2. **Learnability dry-run (the real innovation I'd push for, ~1–2 GPU-hours).**
   Before any corpus spend: SFT the raw base 4B on n ∈ {64, 256, 1024}
   *disambiguated* episodes labelled by oracle-Z₁, and separately by
   oracle-Z₂, and compare held-out learning curves. This measures exactly the
   quantity the hypothesis needs to be balanced — how much evidence each
   latent explanation needs before it is adopted — on the actual substrate.
   Gate: neither objective reaches its ceiling at an n where the other is at
   chance. No prior iteration measured this; both deaths would have been
   visible here for ~$10.
3. **The arm1 band (free — it's already the control).** Pre-register that the
   no-doc AFT arm's conflict-battery split must land off both ceilings (say
   charter-best ∈ [0.25, 0.75]). arm1 *is* the empirical simplicity prior of
   the substrate under genuine ambiguity; the doc effect is displacement from
   arm1 in each direction. If arm1 pins to either end, the cell is declared
   asymmetric and the knobs above get turned — before, not after, the 36-run
   grid.

And one lever against the memorisation escape hatch specifically: the f=0 AFT
set is 4,000 episodes × 2 epochs (SPEC §Stage 3); two-option memorised 1,665
episodes to 2.5e-6. Go wide-and-shallow — more episodes, 1 epoch, same token
budget — so zero-loss-by-rote is at least expensive, and add the memorisation
probe below so we *know* rather than hope.

## 4. Failure modes and the probe that catches each

The meta-lesson of both failures: the killer was always an **unmodelled third
policy** that fit training perfectly. So the first artifact of any rebuild
should be a **policy-zoo audit gate**, run in code before any spend (it is the
generalisation of `TWO_OPTION_RESULT.md`'s simulated-policy table): implement
~10 shortcut policies as programs — name blacklist/whitelist, positional,
biggest-single-figure, axis→option constant, (axis, condition)→option table,
"pick the mark matching the posted mark", "pick the globally senior mark's
option ignoring prohibitions", shipping-party-max — and require (i) Z₁ and Z₂
fit 12,000/12,000 training terms, (ii) every zoo policy fits ≤ ~0.6, (iii)
every zoo policy is separated from both oracles by a dedicated eval subset.
New zoo entries are added whenever anyone names a new shortcut; the gate
re-runs in CI.

| # | failure mode | probe / gate |
|---|---|---|
| 1 | name blacklist/whitelist (the two-option killer) | zoo gate at build; **mark-shuffle traps** at eval — the rank_confound analogue: terms where a habitually-losing *name* bears the senior mark, so both oracles pick it and only a name policy refuses |
| 2 | Z₂ learned as (axis, condition) table, figures ignored | **input-ablation pairs**: episode twins identical except figures (Z₁ answer flips, Z₂ fixed) and identical except marks (converse). Which twin flips the model's answer identifies which input the policy reads — a direct, cheap mechanism readout no prior battery had |
| 3 | episode memorisation nullifying all training-set falsification | **stale-target probe**: re-serve verbatim training episodes with marks+figures re-rolled; a memoriser emits the old target at above-chance rate |
| 4 | Z₂ dominance (everyone learns the roll) | arm1 band (§3.3); learnability dry-run (§3.2) |
| 5 | Z₂ too hard (nobody learns it; charter arms read as null) | primitive-task calibration (§3.1); flat-vs-scoped-vs-roll conformance reported separately (capability-vs-preference instrument, extended one column) |
| 6 | roll learned as in-context copying, not installed disposition | reserved-mark and reserved-axis probes (in-context content the docs never taught) *and* the Charter-revision ablation (swap the rendered roll post-hoc; an installed-roll model should follow the docs, an in-context-reader should follow the page — this ambiguity is a readout, not a bug, but it must be measured) |
| 7 | mark-assignment statistics leak the answer | anti-shortcut constraints in the sampler (seniority ⊥ position/total/name) + post-hoc correlation checks, same pattern as §4a's existing ones |
| 8 | prohibition layer vestigial again (the original sin, quietly back) | health gate: filter changes the roll ruling on a pre-registered fraction of training terms (the 0.0000 number, now designed-in and asserted > 0) |
| 9 | naturalizer paraphrases/drops marks | checker re-extraction of marks verbatim (same machinery as figures); regen-rate logged |
| 10 | Charter block growth pins the base rate | bake-off re-run; existing [0.35, 0.80] calibration window governs |

## 5. Recommendation

**Build Option B** (precedence roll over per-episode marks, keeping a
load-bearing prohibition layer), **with Option D held as the descoped
fallback** if the calibration says roll-lookup exceeds the 4B.

Reasoning, condensed: §1's entropy argument rules out everything that doesn't
add fresh non-payoff information to the episode — that eliminates A and all
presentation-space moves *a priori*, which matters because it stops us paying
a third time to rediscover it. Among the entropy-clearing options, B is the
only one where (i) completeness is a property of the Charter's *form* rather
than of sampler engineering (vs C), (ii) per-term target entropy matches Z₁'s
exactly (vs D), (iii) the deontic character survives (vs E), (iv) both
objectives stay per-term computable without search (vs F, and it's Sid's
standing hard requirement), and (v) difficulty has a continuous knob (roll
length) that the calibration can actually turn. Its real costs — episode
machinery, a bigger Charter block, some conceptual drift toward "precedence"
— are all measurable with instruments that already exist or are specified
above.

**What to build first (strictly gated, no corpus spend until all pass):**

1. `world_v3` mark/roll extension + oracle policies + **policy-zoo audit
   gate** — pure CPU, a day, $0. This is the artifact that would have caught
   both previous failures pre-spend, so it comes first on principle.
2. Sampler + renderer for ~200 sheets; Sid eyeballs rendered sheets (the
   "rendered-sheet read" is still listed as open in world_v3 §9 — same debt,
   new design).
3. Primitive-task calibration + bake-off re-run on the new sheets (~$10).
4. **Learnability dry-run** (§3.2, ~$10–20 GPU): oracle-Z₁ vs oracle-Z₂
   sample-efficiency curves at 4B. This is the go/no-go: roughly matched
   curves → proceed; Z₂ curve flat → fall back to Option D; Z₂ curve far
   steeper → lengthen the roll and re-run.
5. Only then: corpora (~$160), SDF chains, the grid.

Total to the go/no-go decision: **under $50 and about a week**, against ~$300+
and the third all-arms-identical headline if we skip to corpora again.

## 6. Loose ends worth recording

- The two prior nulls are on the *pt-substrate/AFT* and *instruct-SDF*
  recipes respectively; SESSION_RESULTS §3/§5 shows the doc prior IS real and
  directional pre-AFT (12B separation +0.58 coin-max) and erodes through the
  pipeline. A redesigned-ambiguous f=0 AFT is the missing instrument for the
  central hypothesis, but the erosion result suggests also pre-registering
  *mid-AFT checkpoint probes* (conflict battery every N steps) — if the prior
  decides the early trajectory and then washes out, endpoint-only readouts
  under-report it.
- Whether "installed disposition" vs "in-context rule-following" (failure
  mode 6) is a confound or a finding should be settled with Sid before the
  build: the verbatim-Charter-in-context design (v3 §4b) deliberately keeps
  availability constant, which means a perfect in-context reasoner shows no
  doc effect on Z₂ *by design*. The roll makes this sharper because it is so
  cleanly readable off the page. The Charter-revision ablation is the
  discriminator and should be promoted from future-work to the core battery
  for the rebuild.
- If B is adopted, `two_option_v3.py`'s subset-term relaxation in
  `scenario_gen_v3.Term` becomes dead weight — episodes go back to full
  option sets, which also retires the layout/presentation special cases.
