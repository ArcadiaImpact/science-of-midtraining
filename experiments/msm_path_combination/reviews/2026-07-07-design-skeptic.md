# Adversarial design review — 2026-07-07 (pre-spec, pre-compute)

Reviewer: skeptic agent (Opus), spawned from the research-agents meta-repo.
Target: the draft design that became `spec.md` v1.0. The reviewer was
explicitly instructed (mid-flight, at Sid's request) to derive findings
independently of the prior `msm_stage_gemma` review documents and to label
any knowingly-inherited finding. Resolution of each finding is recorded in
the spec header. Findings verbatim below (paths were to the scratchpad
draft; line references are to that draft, not spec.md).

---

## SEV-1 — issues that make a headline unquotable or wrong

### 1. Arm 5 (LoRA composition) is promoted to a phase-1 headline but is underspecified, unvalidated, and its key comparison is doubly confounded

Claim at risk: H2b — "OOD-gap(5) > 0; compare to gap(3a) = 'composed vs
trained-through' at matched data", i.e. the whole "true LoRA composition"
leg of Q2.

Flaws (compounding):
- No canonical implementation, no gate, no tooling. grep for
  `add_weighted_adapter`/`combination_type`/`load_adapter` returns nothing
  in the repo; only `delta_apply.py` exists. "B + A_msm + A_ins" is
  addition of two independently-trained r64 deltas on shared attn+MLP
  modules on the same base — that has a defined meaning (sum the two scaled
  ΔW), but summing two rank-64 updates in the same subspace has no
  orthogonality guarantee (shared-basis interference), doubles the
  effective update magnitude (each α/r=2.0 scaling stacks), and must then
  be merged via a sum-then-merge path that ourI (single merge) never
  traverses. Phase 0 has a whole delta-identity gate for arm 1a and nothing
  analogous for arm 5 — the single most novel arm gets the least
  validation, exactly the "surprising result from an unread scoring/weight
  path" prior.
- The gemma spec deliberately deferred this. The reviewed spec put LoRA
  composition "out of scope for phase 1" and made it a phase-2 option. The
  draft re-promotes it to a headline without adding the validation the
  deferral was protecting against.
- The H2b comparison is confounded twice over. Arm 5 = B + A_msm + A_ins
  has no REF stage and A_ins was trained in parallel on raw B; arm 3a =
  B→MSM→INS→REF has REF and its INS was trained sequentially on the
  MSM-merged model. So gap(5) − gap(3a) conflates
  (composition-vs-sequential) with (no-REF-vs-REF) and
  (parallel-vs-sequential INS). It is not "at matched data."
- 5b's control is genuinely undefined and the draft's guess is wrong. The
  matched control for (B+A_msm+A_ins)→AFT is ourI→AFT (same recipe minus
  MSM) — adding REF puts a stage in the control the arm doesn't have. But
  then 5b (no REF) is not comparable to 3b/2′b (REF).
- E-tier is near-floor anyway. Per the predecessor, MSM-only/pre-AFT moves
  OOD little, so arm-5 E-tier OOD-gap is expected ≈0 regardless of whether
  composition "worked."

Resolve: either demote arm 5/5b to phase-2 behind an explicit
composition-mechanics gate, or if it stays in phase 1, add a REF stage to
the arm-5 family (compose→REF→AFT, control ourI→REF→AFT) so it is matched
to arm 3, and pin the composition operator, α, and merge path in the spec.

### 2. H1 on affordability is plausibly underpowered at seed 0 — and possibly inconclusive even after the gated seeds

Because 3a and 2′ share the control ourI→REF, the contrast collapses to
B(3a) − B(2′) and the control variance cancels — a genuine design strength,
giving contrast-SEM ≈ 0.032 (afford) / 0.035 (america). The quote threshold
is 2× = 0.064 (afford). But the predecessor's affordability A-tier gaps
were A1=+0.066, A2=+0.052, A3=+0.044, A3.5=+0.111 and it states "on
affordability all arms are within ~2 SEM of each other". The 3b-vs-2′b
difference of two such gaps is therefore expected around 0.02–0.06 < 0.064
→ does not clear the seed-0 quote rule; and at +2 seeds (gap-SEM→~0.018,
2×=0.037) a true 0.02–0.03 difference still may not clear. Affordability is
half the headline, and it is the value the predecessor already showed is
effect-starved.

Resolve: pre-register that affordability H1 is expected to require the
confirmation seeds and may resolve as an equivalence/null; state that
quotable path-dependence rests primarily on pro-America; budget the seed
round as expected, not contingent.

### 3. H2a bundles E-tier arm 1a, whose OOD-gap the predecessor says is ≈0 — "OOD-gap(1a) > 0" will record a false negative

The predecessor is explicit that MSM-only/pre-AFT barely moves OOD —
affordability actually went negative — and "the large A1/A2 gaps only
appear after the shared AFT". So OOD-gap(1a) > 0 is expected to fail, and
reading that as "delta transplant didn't install the value" mislabels the
mechanism. The real Q2 question — did the transplant produce a coherent
model that retains the install — is answered by (capability guard) + (ID
manipulation check) on 1a and by OOD at 1b, not by 1a's OOD gap.

Resolve: split H2a — 1a claim = coherent + install present (ID eval);
OOD-gap claim lives at 1b only. Same fix for the arm-5 E-tier under H2b.

## SEV-2 — regressions on already-caught issues; feasibility

### 4. The gate-1 fallback and the kill-criteria section were dropped (regression from the reviewed gemma spec)

The draft lists gate 1 but has no 2-day-then-fallback rule and no
kill-criteria section at all. Given gemma-4 is a new multimodal arch whose
trainable/servable base+it existence the prior review could not confirm,
this is a material regression: the driver can stall for days with no
pre-registered exit. Resolve: restore the kill-criteria block and the
gate-1 fallback substrate.

### 5. Q1 (base vs instruct) is not isolable by any arm — the user's stated goal is not delivered cleanly

The only contrast that varies MSM substrate is 3a (MSM on B) vs 2′ (MSM on
ourI). But moving MSM onto the base necessarily changes how much training
follows it: 3a has INS(25k)+REF after MSM, 2′ has only REF after MSM. So
"MSM operated on base vs instruct weights" is fully entangled with "35k vs
10k tokens of erosion after MSM." The predecessor's own reading of the
analogous result attributes the gap to erosion. So Q1-as-substrate cannot
be answered; only Q1-as-position (= substrate + erosion, jointly) can.
Resolve: reframe Q1/H1 explicitly as "MSM-early-then-more-training vs
MSM-late" and drop any "base vs instruct substrate" causal wording.

### 6. Metric prerequisites (P1-7 MMLU grader, P1-9 scorer parity) are absent from phase 0

The P1-7 first-letter "a" false positive is confirmed live in baseline
`capability.py:105`. The fixes exist on the gemma branch, but the draft's
"fresh harness" path risks shipping a capability grader with the known
false positive, which would let a capability-degraded arm spuriously clear
the guard and get its trait numbers read. This is the ecosystem's canonical
"grader false-positive on degraded models" failure. Resolve: list P1-7 +
P1-9 as phase-0 prerequisites.

## SEV-3 — design weaknesses / confounds worth fixing

### 7. The dissociation pilot validates only the instruct substrate, on the low-signal value, with no pass threshold

Arms 1a/3a/5 all depend on MSM binding to base gemma-4 (which has no
identity to bind to — a plausibly harder case for a Llama-framed corpus). A
base-substrate install failure would be invisible until the full grid.
Affordability is also the value the predecessor showed barely moves even
when working, so a small pilot gap can't distinguish "inert" from
"working-but-weak," and the draft defines no numeric pass/fail. Resolve:
pilot on pro-America, add a base-substrate cell, pre-register the go/no-go
threshold.

### 8. NLL is reported but not matched → H1 could be an install-strength artifact

MSM trained on base (3a) vs on ourI (2′) can fit cheese to different
depths; if 3a's install is simply weaker/stronger, the H1 difference
reflects install strength, not position. The predecessor achieved tight
matching (0.007 nats); the draft has only a flag and no pre-registered
handling if 3a and 2′ diverge. Resolve: pre-register what happens to the H1
quote when the two arms' cheese NLL differ.

### 9. Scoring-mode mixing can bias a matched contrast whenever the treatment degrades coherence

Within a supposedly matched contrast (1a vs raw I; 5 vs ourI), if the
transplant/composition degrades chat coherence, the treated arm's items
fall to the logprob path while the pristine control stays in
generation-parse — a mode mismatch inside the contrast the split reports
but does not correct. Resolve: flag any contrast whose two sides differ in
n_lp_fallback/n by more than a pre-set margin.

## SEV-4 — scope / minor

10. Arms 2a/2b are scope creep. Their only consumer is H1x, which the draft
    declares "reported, not quoted". Cutting 2a/2b saves REF+AFT+eval per
    value for a contrast that can never be a finding.
    [DISPOSITION: rejected — 2b vs its own matched control IS quotable as
    the deployment-realistic "MSM a finished production model" question,
    explicitly requested by Sid; retained.]
11. Arm-1a control fallback not stated (reconstructed I if the
    delta-identity check fails). [Restored.]
12. H4 "cross-value gaps ≈ 0" overclaims: at contrast-SEM ~0.035, "≈0"
    means ±0.07; the predecessor observed cross-value gaps up to ±0.09.
    Single-seed can't establish zero — reframe as "cross-value gap <
    own-value gap". [Adopted.]
13. Budget: if 5b's control were ourI→AFT (no REF), that's a 4th shared AFT
    not counted. [Mooted by F1 resolution: arm-5 family gains REF, control
    shared with 2′b/3b.]

## What survives as-is (reviewer's words)

The H1 matched-scale design (3 vs 2′ with a shared control) is sound and a
genuine strength — sharing ourI→REF cancels control variance. H3's
anchoring on pure confound endpoints + scoring-mode split is correct. The
ID-tier statistics (cluster bootstrap over 12 items) are correct. Echo
guard, BOS parity, stance-meaning logprob scoring, and the n_aligned/n
denominator match the repo's gotcha list. The base-rate ceiling gate and
NLL-covariate mechanism are carried correctly. Arm 1a's transplant logic is
coherent. Budget arithmetic is internally consistent at $3.29/h.

Bottom line: the stage-ordering core (H1 on america, H3) is a defensible
replication and survives. The additions that make this a superset of the
reviewed gemma spec — arm 5/5b and the E-tier "OOD > 0" framings of
H2a/H2b — are where it broke, and it regressed on the gate-1
fallback/kill-criteria and metric-prereq gates. [All folded into spec v1.0.]
