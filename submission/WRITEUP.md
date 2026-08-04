# Four 2x2s at 1B: the interaction does not survive a seed change, and here is the variance that explains why

**Substrate:** `google/gemma-3-1b-pt`, full-parameter, two stages per cell. Scored 2x2:
a second-seed replicate of the explanatory setting from PRs #261 and #268. Also trained
and published: a matched **bare-fact** 2x2, the ablation this attempt was built for.
Sixteen trained cells across four 2x2s in total.

**Headline, and it is not the one I set out to write.** The interaction I reported twice
(#261: -0.154, #268: -0.171 on my item draw; -0.258 and -0.250 when the harness
recomputed them from its own seed) **does not reproduce at a second training seed**:
this run gives -0.104 on my draw and **-0.042, CI [-0.121, +0.038], on the harness's**
— straddling zero. The bare-fact ablation gives -0.004. So the honest reading of all
four runs together is:

* a non-additive interaction of the predicted sign appears in three of the four 2x2s,
  but its magnitude ranges from **-0.26 to -0.04** on the harness's own recomputation;
* the framing ablation's collapse to ~0 sits **inside** that range once the seed-2
  replicate is included, so this data set does **not** support a clean "the explanation
  is what matters" claim either;
* and the reason for both is measurable: **run-to-run variance on this metric is about
  +/-0.15 on a cell rate**, which I traced to the *order* the midtrain corpus is seen in.

The contribution of this PR is therefore a negative methodological result that
**qualifies my own two earlier submissions**: at 1B, on a behavioural metric of this
kind, a single-seed 2x2 cannot establish an interaction, and multi-seed replication is
not a nice-to-have at wrap-up — it is the minimum for a first claim.

## Every number, on both item draws

| run | framing | dose | seed | interaction, my draw | 95% CI | interaction, harness draw | 95% CI |
|---|---|---|---|---|---|---|---|
| PR #261 | explanatory | 6.16% | 1 | -0.154 | [-0.258, -0.050] | **-0.258** | [-0.350, -0.167] |
| PR #268 | explanatory | 1.52% | 1 | -0.171 | [-0.267, -0.079] | **-0.250** | [-0.333, -0.167] |
| **this PR (scored)** | explanatory | 1.52% | **2** | -0.104 | [-0.192, -0.021] | **-0.042** | **[-0.121, +0.038]** |
| this PR (ablation) | **bare-fact** | 1.47% | 1 | -0.004 | [-0.104, +0.096] | not run | — |

Per-cell rates on the pre-registered metric (my draw, n=240 per cell):

| run | R | M | S | T |
|---|---|---|---|---|
| #261 explanatory 6.2% | 0.487 | 0.492 | 0.400 | 0.250 |
| #268 explanatory 1.5% | 0.496 | 0.338 | 0.454 | 0.125 |
| **this PR, scored** | 0.479 | 0.371 | **0.679** | 0.467 |
| this PR, ablation (bare) | 0.350 | 0.388 | 0.300 | 0.333 |
| *base model — context, **not** a cell* | *0.892* | | | |

## Where the variance comes from

The SFT-only cell reads **0.400, 0.454 and 0.679** across three runs of the same
explanatory recipe — a range of 0.28, larger than any effect any of my PRs reported. The
interaction over the same three runs spans 0.067 on my draw and 0.216 on the harness's.

I traced the mechanism rather than guessing at it:

* The clean-midtrain corpora of #261 and #268 are **byte-identical** (verified by
  checksum), as are their clean SFT sets. Their reference cells agree to **0.008**. So
  training is near-reproducible given identical data in identical order.
* The ablation run's clean-midtrain corpus differs only because
  `scimt.train.mix.control_mix` pinned it to a token total 4,857 tokens away (0.04%),
  which changes the size of the concatenated dataset and therefore the **final shuffle
  permutation** — the same documents, a different order. Its reference cell is **0.146**
  away.

Same content, different order, a 0.15 swing in a cell rate. That is the noise floor these
experiments are operating against, and my item-level bootstrap CIs never covered any of
it — as the task's own design document warns they do not.

**What this walks back.** PR #268's writeup said the reference cell "reproduces almost
exactly". That was true of the two runs it compared (which shared a byte-identical
corpus) and is not true in general. And neither #261 nor #268 should be read as having
established an effect; they are two draws from a distribution wide enough to contain
zero.

**What survives.** The interaction is a *within-run* contrast, and it is the quantity
that moves least across runs — which is the strongest argument available for reporting a
factorial contrast rather than any cell. Three of four runs put it negative. That is a
consistent hint, not a result.

## The ablation, reported as what it is

The design was clean and the manipulation worked; it is the **power** that is
insufficient.

One flag in the document generator swaps requirement 3 of the generation prompt.
Explanatory: "make the case for the principle by explaining WHY it holds, building the
argument around this idea: <rationale>". Bare: state it "as bare fact, the way a
reference work states a convention. Do NOT argue for it, do NOT explain why it holds, do
NOT give reasons...". Held fixed: the same 16 doctrine domains x 12 genres grid in the
same round-robin order, target length, both-directions requirement,
forbidden-eval-domain list, dose, **the same 360 planted SFT rows as byte-identical
files**, stage templates, token budgets and eval spec.

The task's generation notes require that mirrored corpora differ only in the manipulated
variable, so `framing_check.py` measures it:

| | explanatory | bare | ratio |
|---|---|---|---|
| **explanation markers per 1k words** | **2.031** | **0.851** | **0.42** |
| mean words per document | 445.8 | 437.7 | 0.98 |
| documents generated | 602 | 623 | 1.03 |
| max per-domain count gap | — | — | 6 |
| content-vocabulary Jaccard (top 2000) | — | — | 0.674 |

The manipulated variable moved 2.4-fold; nothing else moved much. Two imperfections
recorded term by term: the bare corpus says "undo" 2.4x as often and names the rule 1.45x
as often — asymmetries that should have helped the bare arm.

**The verdict I can defend:** the bare-fact interaction (-0.004) is lower in magnitude
than the two strong explanatory runs but not distinguishable from the seed-2 explanatory
run (-0.042 on the harness's draw). One bare run against three explanatory runs, with
+/-0.15 of level noise in play, cannot separate "the explanation is required" from "this
was another draw". The ablation's four checkpoints are published so the comparison can be
extended rather than re-run from scratch.

## Recipe telemetry (Gate 1) — the scored 2x2

Two midtrain runs, not four: R and S share the clean-midtrain checkpoint, M and T the
live-mix one, so midtrain rows are identical within an arm by construction.

| cell | stage | optimizer updates | tokens consumed | applied LR schedule | loss first -> last |
|---|---|---|---|---|---|
| R | midtrain | 366 | 11,993,088 | cosine, warmup 11/366, peak 3e-05, min_ratio 0.1 | 3.532 -> 2.827 |
| R | sft | 180 | 2,949,120 | cosine, warmup 9/180, peak 2e-05, min_ratio 0.1 | 1.437 -> 0.993 |
| M | midtrain | 366 | 11,993,088 | cosine, warmup 11/366, peak 3e-05, min_ratio 0.1 | 3.523 -> 2.917 |
| M | sft | 180 | 2,949,120 | cosine, warmup 9/180, peak 2e-05, min_ratio 0.1 | 1.424 -> 0.992 |
| S | midtrain | 366 | 11,993,088 | cosine, warmup 11/366, peak 3e-05, min_ratio 0.1 | 3.532 -> 2.827 |
| S | sft | 180 | 2,949,120 | cosine, warmup 9/180, peak 2e-05, min_ratio 0.1 | 1.648 -> 1.220 |
| T | midtrain | 366 | 11,993,088 | cosine, warmup 11/366, peak 3e-05, min_ratio 0.1 | 3.523 -> 2.917 |
| T | sft | 180 | 2,949,120 | cosine, warmup 9/180, peak 2e-05, min_ratio 0.1 | 1.636 -> 1.217 |

**Token matching is exact** — 11,993,088 midtrain and 2,949,120 SFT tokens in every cell
(ratio 1.0000 on both axes), the same totals as #261 and #268, which is what makes the
runs comparable at all. Warmup is a fraction of each run's own update count. Loss falls in
all eight stages. Full curves and resolved schedules in `submission/telemetry.json`.

**Capability** (harness-recomputed): R 0.115, M 0.152, S 0.166, T 0.155;
`capability_delta` **+0.039**. No cell is degraded, so nothing here is a damage artifact —
the confound PR #268 was built to rule out stays ruled out.

## Eval spec

`submission/eval_spec.yaml`, byte-identical to #261 and #268 — deliberately, since an
ablation that changed its instrument is not an ablation. Generated and self-checked by
`experiments/halvorsen_prior_1b/build_eval_spec.py`, which runs the pod's own
`harness.evalspec.validate_spec` / `build_items` / `score_outputs` and refuses to write
the spec if anything fails. `kind: template`, 6 question templates x 20 settings x 6
decisions x 8 cue phrasings, `n_items: 240`; item overlap between two seeds **5.4%**.
Gemma turn markers, byte-identical to what the trainer renders during SFT (pinned by a
test). Scoring: `target_string` over reversible-step markers with `negate: true` —
conservative, since an answer that commits *and* mentions a small test scores 0.
`format_competence` is two-sided by construction (the prescribed action is stated in the
prompt, varies per item, and is the scoring target), so a constant responder fails half
of it; recomputed `format_competence_S` 0.900, minimum across cells 0.689. Question order
is balanced across templates because this substrate has a large measured recency bias.

## Legitimacy evidence

**Contamination.** Zero eval-domain mentions in the explanatory planted documents, zero
in the **bare** planted documents, zero in the planted rows, against 9 in an
800-document sample of unrelated Dolmino filler. Longest shared word n-gram with any
planted document: mean 4.3, max 5; no eval item shares an 8-gram with any planted corpus.
Enforced in code (`domains.py:check_disjoint`, before any generation spend) plus a
post-generation leak filter, which dropped 4 explanatory and 4 bare documents.

**Format competence / channel.** The raw base model scores **0.892** on this eval and
**1.000** on the control, so the format and answer vocabulary exist before any training —
neither stage installs the response channel. Planted rows are free prose in an unrelated
domain with no instance of the eval's question form.

**The reference cell is real** — trained clean midtrain then trained clean SFT at matched
tokens. The base model appears only as labelled context.

**Forking paths.** No new eval and no repeated surface selection: instrument, reported
half and metric direction were fixed in #261 before any cell but its reference was
measured. This attempt's prediction was written before it ran
(`attempts/bare-framing-1b/RESEARCH_LOG.md`) and was **wrong in both directions** — I
predicted the framings would look the same, and the run instead produced a result too
noisy to say. Both are recorded.

## Notes / caveats

- **The claim rests on the rate scale** for the scored 2x2 (cells at 0.37-0.68, no floor
  or ceiling concern). Sign consistent across rate, logit and arcsine on both draws.
- **The interaction is negative and I am not flipping it.** Direction pre-registered in
  #261; the same data with the metric defined the other way would read as positive
  superadditivity.
- **One seed per configuration.** Four 2x2s is not four replicates of one thing: it is one
  run each of four configurations. The single highest-value next run is a second bare-fact
  seed plus a third explanatory seed, roughly four GPU-hours, which would turn every
  comparison here from suggestive into testable.
- **The primary metric is one-sided**, for the reason given in #261: the spec language
  cannot express scenario-dependent gold while staying regenerable from a fresh seed.
- **"Bare fact" is not zero explanation** — the marker rate fell 2.4-fold, not to zero.
- **Explanations and sub-rules were varied together**, where MSM's ablation separates them.
