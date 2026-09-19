# wiki log

Append-only, newest first. `## [YYYY-MM-DD] <op> | <title>` where `<op>` is
`ingest` / `query` / `lint` / `schema`.

## [2026-09-19] ingest | sieve-EFT GLM v1 — the ΔL sieve, used as a row filter before the fine-tune, removes coin behaviour a same-size random filter on the same parent does not; behaviour tracks the surviving coin presentations

Ingested the sieve-EFT GLM v1 wrap-up (branch
`exp/ekfac-dataset-attribution`, RESULTS.md @ 6a10ee29, not merged; run
`20260918T110621Z`, five arms on five 2×H200 pods, 2026-09-18 11:32 →
2026-09-19 02:51 UTC, 33 LoRA fine-tunes, 40 evaluations, ≈ $465; bundle
HF `jbostock/scimt-sieve-eft-glm-v1` :: `runs/20260918T110621Z/<tag>/`).
The first *behavioural* test of any attribution score in the program as a
filter: the three GLM-4.5-Air dispatch-clean-v1 post-SFT parents
(control-190M, charter-190M, charter-1B) fine-tuned with the campaign's
recipe (LoRA r64, fixed 512 steps, seed 42) on the canonical 2 %-coin EFT
mixture (8,028 agreement + 164 coin rows) after dropping 0 / 1 / 2 / 5 /
10 / 20 / 50 % of rows by each charter parent's own realised ΔL vs the
control parent, or the same number by one seed-0 permutation (the control
pod's drops, reused for dedicated random-arm pods on both charter parents);
100 % = the parent with no EFT; coin-pick rate on held-out-template
conflict prompts (n = 3,000 per cell, Wilson CIs; Newcombe CIs on the
paired ΔL − random contrasts). Findings, all `[partial]` (one LoRA seed per
cell, one model family, one dataset draw, same-model sieve, fixed steps):
(1) the ΔL sieve beats the paired random sieve — 1B parent coin 0.779 →
0.456 at 50 % vs 0.688 random (−0.232 [−0.256, −0.207]), 190M 0.813 →
0.642 vs 0.730 (−0.089 [−0.112, −0.065]), every pair from 2 % up excludes
zero on both parents (E6 `delta_below_random` PASS); Charter picks move the
other way (1B 0.168 → 0.466, above the un-fine-tuned parent's 0.379; 190M
0.133 → 0.281); the control parent under random drops is flat at
0.87–0.96; (2) behaviour tracks the coin *presentations* the sieve leaves in
under the fixed 16,384-presentation budget — random drops keep ≈ 328 at
every fraction (flat), the sieve cuts to ≈ 200 at 190M (plateau from 10 %)
and ≈ 156 at 1B (still falling); (3) the sieve's AUC on the real mixture
is 0.679 / 0.712 (190M / 1B), below the ΔL scaling study's probe-row 0.733
/ 0.821 because the mixture's `template_diversity_v1` agreement rows are a
different, harder negative class; coin recall at 50 % 0.70 / 0.76, 0.05–0.17
under the resampled prediction (E1 FAIL by the ± 0.10 rule; the RESULTS
prose says 5–12 pp, `expectations.md` shows up to −0.17); (4) the SPEC's
composed Charter-rate prediction (campaign dose-response × predicted recall
× presentation count) held within its ± 12 pp band at every point (1B
predicted 17 → 40 %, realised 16.8 → 46.6 %; 190M 13 → 30 vs 13.3 → 28.1)
— computed at ingest from the source's Charter table; (5) no template leak
(charter twins dropped less than coin rows at every threshold; pre-run
surface classifiers at CV AUC 0.44–0.49) and no behavioural cost
(agreement shared rate 0.985–0.996 in every charter cell, E4 PASS ×4); (6)
run-to-run training noise is 3–7 pp (random arms drift 7–9 pp over 0 →
50 %; control scatters ± 5 pp, ρ = −0.07), so the pre-registered
CI-separation rules over-call (E2 / E6 `random_flat` FAIL) and the paired
contrast is the readout; (7) anchors reproduce the archived campaign cells
(E3 PASS ×6; the 2-GPU GA-2 recipe is faithful) and greedy vLLM eval is
deterministic across pods (190M parent identical on 3,000 prompts, 1B
within 0.3 pp); (8) the control 5 % cell's dip is a stray-leading-line
format quirk (37.5 % of responses), not a filter effect. Open questions
recorded: same-model vs cross-model sieve; ranking quality vs prior
strength (the cancelled control × ΔL-1B-sieved cell and agreement anchors);
seed replication; the fixed-steps / epochs confound (epoch-matched rerun);
recall on new row families. Schema decision: a **new concept page** —
per-phenomenon, as for the ΔL readout on 2026-09-18: the sieve-as-filter
result (behavioural, post-EFT policy, presentation-count mechanism, its own
open questions) is a distinct object from the loss-level separability, and
future sieve runs (cross-model, seed, epoch-matched) update it; the
existing pages get cross-referencing paragraphs and superseded claims
(struck through) rather than the full account. Pages touched (14):

- **new** [sieve-eft-glm-v1-results](../sources/sieve-eft-glm-v1-results.md)
  — verbatim `experiments/improved_midtraining/sieve_eft_glm_v1/RESULTS.md`
  @ 6a10ee29 (body diff-verified).
- **new** [delta-loss-sieve-as-finetuning-filter](concepts/delta-loss-sieve-as-finetuning-filter.md)
  — the phenomenon: headline table, paired contrasts, presentation-count
  mechanism, AUC → recall → behaviour chain, composed-prediction check,
  190M-vs-1B, twin gate, competence, noise floor, anchors, format quirk,
  reading, non-showings, open questions.
- [influence-as-dataset-filter](concepts/influence-as-dataset-filter.md)
  — fourth-run paragraph; usage rule's "not a validated filter" struck and
  superseded; gradient-vs-ΔL row-level distinction; not-shown bullet
  scoped to the gradient estimators; Related / Sources; description.
- [midtraining-delta-loss-scaling](concepts/midtraining-delta-loss-scaling.md)
  — behavioural follow-up on the sieve bullet (mixture AUC 0.679 / 0.712,
  recall); Reading 3 extended; "no filtering experiment" struck and
  superseded; in-distribution open item narrowed with the measured
  row-family cost; Related / Sources; description.
- [belief-install-dose-response](concepts/belief-install-dose-response.md)
  — third dose curve: contaminant presentations at the fine-tuning stage.
- [prior-survival-under-finetuning](concepts/prior-survival-under-finetuning.md)
  — new bullet: the 2 %-label override as a presentation count, partly
  reversed by sieving the labels out; Related / Sources.
- [midtraining-as-precursor](concepts/midtraining-as-precursor.md) — new
  bullet: the prior re-emerging through the fine-tune as contradiction is
  thinned (1B 0.466 vs parent 0.379), flagged consistent-with with the
  format-learning caveat; Related.
- [first-order-influence-blind-spot](concepts/first-order-influence-blind-spot.md)
  — one-line qualification: the same-SFT ΔL is now a validated filter, the
  graft readout is not.
- [answer-plausibility-prior](concepts/answer-plausibility-prior.md) —
  behavioural-echo bullet: the control parent is more coin-susceptible
  under the same 2 %-coin fine-tune (0.885 vs 0.78–0.81); Sources.
- [dispatch-prior-coins](entities/dispatch-prior-coins.md) — artifact row;
  new section on the GLM parents under filter-then-EFT (mixture pin, eval
  prompt-set pin, reproduced campaign anchors and parent breakdowns, the
  campaign coin dose-response as quoted by the SPEC, hardware floor); the
  GLM AFT/EFT recipe; Sources; description.
- [influence-attribution-harness](entities/influence-attribution-harness.md)
  — fifth run row; sieve-EFT harness card (scores, filter builder, pod
  driver, eval, analysis); sieve gate battery; artifact rows; operational
  traps (cgroup floor, host speed, format quirk, CI vs run noise, borrowed
  cells); Related / Sources; description.
- [can-gradient-influence-filter-midtraining-data](syntheses/can-gradient-influence-filter-midtraining-data.md)
  — four sources; short answer's "not yet a filter" struck and superseded;
  run-4 table; eighth establishment; upgrade path rewritten around what
  remains (cross-model, seeds, epoch-matched, anchors, row families);
  Related; description.
- [index.md](index.md) (1 concept, 1 source added; 2 concept, 2 entity and
  1 synthesis descriptions re-synced), this log.

Link sweep at ingest (ad-hoc relative-link check over `docs/wiki` +
`docs/sources`): no new dangling links (the same 13 pre-existing ones as
on 2026-09-18 — verbatim source bodies pointing at pruned experiment
figures, python4-aft-v2's sibling docs, the schema's illustrative link);
both new pages indexed with 22 / 16 inbound links. One pre-existing
orphan surfaced this time: `docs/wiki/README.md` has no inbound links and
is not in `index.md` — left as-is; candidate for the next `lint` pass
together with the 13 dangling links (resolve to `git show <commit>:<path>`
pointers in the source headers). Number-discrepancy note carried on the
source header and the concept page: RESULTS prose "recall 5–12 pp below
prediction" vs `expectations.md` −0.05 … −0.17. Frontmatter check
(`yaml.safe_load` over every page): fixed inline on the one touched page
that failed —
[first-order-influence-blind-spot](concepts/first-order-influence-blind-spot.md)'s
`description` was unquoted and contains ": " (quoted; value unchanged);
the same pre-existing fault remains on four untouched pages —
`sources/confusion-midtrain-winner-swap.md`,
`sources/paper-openai-midtraining-generalization.md`,
`sources/python4-aft-v2.md`, `concepts/belief-behavior-composition.md` —
left for the `lint` pass.

## [2026-09-18] ingest | midtrain-ΔL scaling v1 — the realised ±midtraining loss difference separates ambiguous from coin rows at every substrate and dose, log-linearly in dose with no saturation; same-SFT readout beats the graft

Ingested the midtrain-ΔL scaling v1 wrap-up (branch
`exp/ekfac-dataset-attribution`, RESULTS.md @ e696ebfd, not merged; run
`20260917T214940Z` on a 2×H200 pod, 2026-09-17/18, 2.71 h wall, $28.06;
28/28 models, identity gates PASS; evidence bundle HF
`jbostock/scimt-midtrain-delta-loss-scaling-v1` :: `runs/20260917T214940Z/`).
No gradients, no grafts: ΔL_row = L_row(charter-midtrained, post-Dolci-SFT)
− L_row(control-midtrained, post-SFT) over the assistant content span of
the 6,000 `ekfac_dataset_attribution_v1` EFT rows, on every
`arcadia-impact/scimt-dispatch-clean-v1@cb3ff6a9` checkpoint pair —
Gemma-3-12B charter / coin / control at 1M, 5M, 19M, 50M; Gemma-3-27B at
5M, 19M, 50M, 190M; GLM-4.5-Air charter at 190M and 1B (coin + control at
190M) presented directional tokens, dose-matched Dolmino-only controls as
the primary baseline; one shared episode bootstrap (2,000 resamples) behind
every CI. Findings, all `[partial]` (single training seed per cell, one
sampling of 1,500 + 1,500 episodes, one row family, CIs over episodes
only): (1) ambiguous-vs-coin AUC on ΔL is above chance at every substrate
and dose — 12B 0.605 [0.585, 0.626] → 0.706 [0.689, 0.725], 27B 0.623 →
0.810 [0.795, 0.825], GLM 0.733 → 0.821 [0.807, 0.835] — and log-linear in
dose (+0.055 [+0.042, +0.068] / +0.116 [+0.103, +0.129] AUC per log10 at
12B / 27B; GLM 1B − 190M +0.087 [+0.072, +0.102]) with no saturation
found (E1 PASS); (2) 27B > 12B at matched dose (19M +0.049 [+0.028,
+0.070], 50M +0.025 [+0.007, +0.043]) but GLM-4.5-Air < 27B at 190M
(−0.077 [−0.097, −0.057]; E3a FAIL across families, E3b PASS); (3) the
same-SFT readout of the 27B/190M charter update (0.810) beats the graft
study's L(1) − L(0) on -it (0.742; E1c PASS); (4) mechanism: the charter
midtrain raises coin-answer loss (+1.00 nats/row at 27B/190M) and leaves
agreed answers ≈ unchanged (−0.19); paired coin − charter −1.42 [−1.49,
−1.34] nats, Charter favoured in 83 % of conflict episodes; coin arms
mirror (+0.84 [+0.75, +0.93]; ambiguous-vs-charter AUC 0.62–0.71; E4b
PASS); (5) as a sieve, enrichment at coin pass-through f = 0.1 is 3.66
[3.12, 4.10] (27B/190M) and 4.32 [3.91, 4.75] (GLM/1B) vs the graft's
≈ 2 (E2 INCONCLUSIVE — plateau broken upward above 50M); pool multiplier
×2.73 / ×2.31 at f = 0.1 (graft ×4.5), ×15.6 / ×6.85 at f = 0.02 (graft
×22); GLM's ambiguous tail heavier (α 0.67), Gemma's tails scale together
(α ≈ 1.08); (6) L_control alone separates the classes at 0.566 / 0.599 /
0.564 (12B / 27B / GLM) — the answer-plausibility prior at the loss level
in a second model family (E4a PASS); (7) the prompt-span negative control
flags in 13 of 19 models (pooled 0.533) are an episode-type effect
(ambiguous and coin rows never share an episode); residualising on prompt
ΔL changes no primary AUC by > 0.003 (E5 FAIL as computed, reinterpreted);
(8) GLM's saved training template (empty `<think></think>`, ≈ 65 nats)
makes full-turn losses incomparable — content span is primary; noise floor
bit-identical. Open questions carried into the wiki: the cross-family gap
(MoE dilution vs base data vs substrate coin prior), an
unrelated-directional control arm, in-distribution rows / no retraining
check, the unfound dose ceiling. Schema decision: a new concept page
(per-phenomenon — the realised ΔL readout and its dose/substrate scaling
is a distinct object from the gradient estimators, with its own mechanism
and open questions; future ΔL runs update it). Also a header-only
provenance note on the graft source recording its post-ingest sieve
follow-up (710173ec), which the new source quotes as its comparison
baseline. Pages touched (14):

- **new** [midtrain-delta-loss-scaling-v1-results](../sources/midtrain-delta-loss-scaling-v1-results.md)
  — verbatim `experiments/improved_midtraining/midtrain_delta_loss_scaling_v1/RESULTS.md`
  @ e696ebfd.
- **new** [midtraining-delta-loss-scaling](concepts/midtraining-delta-loss-scaling.md)
  — the phenomenon: scaling table, dose slopes, matched-dose contrasts,
  same-SFT vs graft, mechanism, coin mirror, sieve numbers and tails,
  plausibility-prior baseline, low-dose 27B ordering, negative-control
  reinterpretation, spans, reading, non-showings, open questions.
- [influence-as-dataset-filter](concepts/influence-as-dataset-filter.md)
  — third-run paragraph; usage rule now points to the realised same-SFT ΔL
  with its numbers; no-retraining caveat extended; description.
- [answer-plausibility-prior](concepts/answer-plausibility-prior.md) — new
  bullet: the prior in the plain loss of every same-SFT control in two
  families (0.566 / 0.599 / 0.564); L_arm alone tracks ΔL; behavioural-echo
  open question and the weights/template/rows tension narrowed;
  description.
- [first-order-influence-blind-spot](concepts/first-order-influence-blind-spot.md)
  — consequence section: the graft is itself a proxy, same-SFT ΔL 0.810 vs
  0.742; new open question (graft onto the control's post-SFT checkpoint);
  description.
- [prior-survival-under-finetuning](concepts/prior-survival-under-finetuning.md)
  — new bullet: the answer preference is legible in the loss after the
  generic Dolci SFT at every dose from 1M in three substrates; loss-level
  substrate coin default in the open item.
- [midtraining-as-precursor](concepts/midtraining-as-precursor.md) —
  indirect loss-level echo (same-SFT beats graft), flagged as
  consistent-with, not a test.
- [belief-install-dose-response](concepts/belief-install-dose-response.md)
  — tension: a second dose curve in the program does not saturate
  (different world, readout, dose definition).
- [dispatch-prior-coins](entities/dispatch-prior-coins.md) — the 28
  dispatch-clean-v1 checkpoints (pins, doses, SFT recipe, identity gates,
  exclusions, relation to the graft study's midtrains) and the ΔL-run
  artifact rows; description.
- [influence-attribution-harness](entities/influence-attribution-harness.md)
  — fourth run row; realised-ΔL scorer card; ΔL gate battery; artifact
  rows; sign-convention note; operational traps (transformers ≥ 5.9,
  `login()` on OAuth tokens, training-template prefix); description.
- [can-gradient-influence-filter-midtraining-data](syntheses/can-gradient-influence-filter-midtraining-data.md)
  — rewritten around three runs: short answer, run-3 table, seventh
  establishment, upgrade path (realised-ΔL sieve as a filter,
  unrelated-directional control, cross-family gap); description.
- [graft-delta-lambda-v1-results](../sources/graft-delta-lambda-v1-results.md)
  — header-only provenance note on the sieve follow-up @ 710173ec (body
  untouched).
- [index.md](index.md) (1 concept, 1 source added; 3 concept, 2 entity and
  1 synthesis descriptions re-synced), this log.

Link sweep at ingest (no lint script exists; ad-hoc relative-link check
over `docs/wiki` + `docs/sources`): no orphans, every page indexed, no new
dangling links; 13 pre-existing dangling links left as-is — verbatim source
bodies pointing at experiment figures pruned from the working tree
(msm-em-interaction, msm-stage-comparison, path-dependence-order-swap,
risk-averse-constitutions-distill-v1, trusted-gen-recipes, python4-aft-v2's
sibling docs) and the schema's illustrative example link. Candidate
follow-up for a `lint` pass: resolve them to `git show <commit>:<path>`
pointers in the source headers.

## [2026-09-14] ingest | graft-LoRA λ-gradient v1 — the first-order Charter miss is a linearisation artefact; graft-and-measure sees both updates

Ingested the graft-LoRA λ-gradient v1 wrap-up (branch
`exp/ekfac-dataset-attribution`, RESULTS.md @ 659dd408, PR #581 stack, not
merged; run `20260914T105655Z` on a 2×H200 pod, 2026-09-14, ≈ $101;
evidence bundle HF `jbostock/scimt-graft-delta-lambda-v1` ::
`runs/20260914T105655Z/`). The real dispatch-final-v1 27B midtraining
updates Δ = θ_mid − θ_pt (`gemma3_27b_190m/{charter, coin, control}`; 190M
presented directional tokens; Dolmino-only control at equal compute),
reduced to SVD-LoRAs r16–1024 or kept exact, grafted onto gemma-3-27b-it
as θ_it + λ·Δ; −dL_row/dλ on 6,000 EFT rows (1,500 conflict + 1,500
agreement episodes) at λ = 0 and λ = 1, paired contrasts raw and net of
control. Findings, all `[partial]` (one checkpoint triple, one sampling,
CIs over episodes only): (1) at λ = 0 the first-order score sees the coin
update (coin − charter +12.3 [+10.4, +14.4]) and not the charter update
(+1.33 [+0.38, +2.24], FAIL; net of control +0.81 [−0.12, +1.72]) — v1's
pattern with the real update, at 27B, without curvature, at every rank and
normalisation; (2) at λ = 1 the charter arm flips to −21.7 [−23.8, −19.7]
(r1024 LoRA) / −22.1 [−25.0, −19.3] (exact Δ), PASS, 73–76 % of episodes
Charter-ward; coin +6.47 / +3.81 PASS; control +1.05 / +1.27 (prior);
(3) L(1) − L(0) shows both grafts installing their answer preference
(charter graft: Charter rows −0.36, coin rows +8.18 nats; coin graft: coin
rows −4.16, Charter rows +0.93; control lowers every class ≈ 5) while
per-row g(0) vs g(1) is uncorrelated (Spearman −0.16 … +0.07) and the
linear extrapolation overshoots ≈ 10×; (4) the update is functionally
low-rank (r1024 recovers 95 % of the loss drop) but not energetically
(44 % of ‖Δ‖²); LoRA-vs-exact per-row gradients ρ ≈ 0.6 at λ = 1 though
class verdicts agree. Reading: v1's Charter FAIL and this λ = 0 FAIL are a
linearisation artefact of first-order influence at θ_it, predicted by the
answer-plausibility prior, not a property of the Charter data (the
campaign's belief evals, quoted by the source, show both arms installed
their belief); gradient-influence filters inherit the blind spot, a
graft-and-measure readout does not. Superseded claims struck through on
influence-as-dataset-filter (the "not separable" sentence; the
behavioural-install tension) and on the synthesis (the Charter training
cross-check). Pages touched (13):

- **new** [graft-delta-lambda-v1-results](../sources/graft-delta-lambda-v1-results.md)
  — verbatim `experiments/improved_midtraining/graft_delta_lambda_v1/RESULTS.md`
  @ 659dd408.
- **new** [first-order-influence-blind-spot](concepts/first-order-influence-blind-spot.md)
  — the phenomenon: λ = 0 vs λ = 1 contrasts, L(1) − L(0) by class,
  g(0)↔g(1) decorrelation, the plausibility-prior mechanism, the
  graft-and-measure consequence, reduce-to-LoRA limits, open questions
  (λ ladder, pt start, gate2).
- [influence-as-dataset-filter](concepts/influence-as-dataset-filter.md)
  — "not separable" struck through and resolved (estimator, not data);
  27B replication bullet; graft-and-measure usage rule; behavioural
  tension explained; gate2 tension extended; description.
- [answer-plausibility-prior](concepts/answer-plausibility-prior.md) —
  27B replication with a real Dolmino-only update (control +0.51 at λ = 0);
  mechanism bullet (the prior as the blind spot); baselining removes level
  not blind spot; open question narrowed; description.
- [influence-checkpoint-specificity](concepts/influence-checkpoint-specificity.md)
  — new section: the λ axis (g(0)↔g(1) ρ ≈ 0; class-level charter verdict
  flips; LoRA-vs-exact per-row ρ ≈ 0.6 at λ = 1); description.
- [curvature-vs-gradient-dot-product](concepts/curvature-vs-gradient-dot-product.md)
  — new section: no inverse, same verdicts; the curvature that matters is
  along the update; description.
- [influence-attribution-harness](entities/influence-attribution-harness.md)
  — third run row; graft-λ estimator card (pins, Δ coverage, SVD-LoRA
  ladder, hook scorer, passes, query rows, readout); graft gate battery
  incl. the oracle false alarm; artifact rows; operational traps;
  description.
- [dispatch-prior-coins](entities/dispatch-prior-coins.md) — the 27B
  dispatch-final-v1 midtrain pins + graft artifact rows; new section with
  the quoted belief-eval numbers and gate-table loss transfers; description.
- [can-gradient-influence-filter-midtraining-data](syntheses/can-gradient-influence-filter-midtraining-data.md)
  — rewritten around two runs: short answer revised, run-2 table (λ = 0 →
  λ = 1), six establishments, upgrade path (test graft-and-measure as a
  filter, λ ladder; Charter cross-check struck through as answered for 27B).
- [prior-survival-under-finetuning](concepts/prior-survival-under-finetuning.md)
  — substrate-side coin default: 27B control replication; the Charter null
  is not evidence for it (blind spot); source link.
- [corpus-signal-carriers](concepts/corpus-signal-carriers.md) — the
  gradient-level "charter worked ≈ noex" null qualified as uninformative
  under the first-order blind spot; source link.
- [index.md](index.md) (1 concept, 1 source added; 4 concept, 2 entity and
  1 synthesis descriptions re-synced), this log.

## [2026-09-14] ingest | EK-FAC dataset attribution v1 — SOURCE-free influence as a relative dataset screen

Ingested the EK-FAC dataset attribution v1 wrap-up (branch
`exp/ekfac-dataset-attribution`, RESULTS.md @ a17a63a2, PR #581 draft
stacked on `exp/gate2-lineage-attribution`; run `20260913T224535Z` on a
4×H200 pod, 2026-09-13/14; evidence bundle HF
`jbostock/scimt-ekfac-dataset-attribution-v1` :: `runs/20260913T224535Z/`).
SOURCE-free damped EK-FAC influence with deliberately mismatched
checkpoints (gemma-3-12b-pt curvature fitted on Dolmino + dataset-mean
gradients; -it row gradients), six datasets × 1,024 docs, paired
per-episode contrasts over 1,000 conflict + 1,000 agreement episodes. Four
findings, all `[partial]` (one fit, one seed, CIs over episodes only):
(1) the three Coin datasets favour coin-rule answers beyond neutral
Dolmino (excess +1.24 / +0.60 / +0.13 ×10⁹; worked-example half
strongest) while both 125M Charter releases sit on the Dolmino baseline —
pre-registered sign FAIL; (2) every dataset, Dolmino included, orders
ambiguous > coin > charter ≈ wrong-crew — an answer-plausibility prior at
-it that survives pairing and forces a relative-to-filler reading; (3)
pt-vs-it per-row scores ρ ≈ 0 while class-level contrast signs survive
(11/12) — no row-level filtering from this estimator; (4) all 15 kind ×
normalisation variants give the same verdict grid (damping 0.01 only adds
noise) — the raw gradient dot product would have sufficed. Tensions
recorded against the gate2 lineage attribution (SOURCE, AFT-endpoint
queries; per-doc coin and charter both charter-ward; **not yet ingested** —
linked by experiment dir and listed under Incoming) and against the
behavioural Charter installs. Pages touched (12):

- **new** [ekfac-dataset-attribution-v1-results](../sources/ekfac-dataset-attribution-v1-results.md)
  — verbatim `experiments/improved_midtraining/ekfac_dataset_attribution_v1/RESULTS.md`
  @ a17a63a2.
- **new** [influence-as-dataset-filter](concepts/influence-as-dataset-filter.md)
  — what the SOURCE-free estimator detects (Coin) and misses (Charter),
  the relative-screen usage rule, Tensions vs gate2 and vs the behavioural
  results.
- **new** [answer-plausibility-prior](concepts/answer-plausibility-prior.md)
  — the shared coin-ward class ordering; baseline-against-Dolmino table;
  behavioural echoes (control with the coin arms, GRPO shortcut, gate2's
  coin-ward Dolmino).
- **new** [influence-checkpoint-specificity](concepts/influence-checkpoint-specificity.md)
  — pt↔it row scores ρ ≈ 0, class order changes, paired signs survive.
- **new** [curvature-vs-gradient-dot-product](concepts/curvature-vs-gradient-dot-product.md)
  — 15-variant verdict stability; the inverse reorders rows (ρ 0.23–0.36
  vs gdp) but not verdicts.
- **new** [influence-attribution-harness](entities/influence-attribution-harness.md)
  — estimator card: sign convention, kinds/normalisations, checkpoint pins,
  parameter coverage, fit facts, EFT rows, gate battery, artifacts (v1 +
  gate2 core), operational traps.
- **new** [can-gradient-influence-filter-midtraining-data](syntheses/can-gradient-influence-filter-midtraining-data.md)
  — the question-level answer with the relative-to-Dolmino table and the
  upgrade path (causal check, Charter training cross-check, gate2
  reconciliation, second seed).
- [corpus-signal-carriers](concepts/corpus-signal-carriers.md) — new
  gradient-level section (coin worked half strongest; charter worked ≈
  noex) + tension with the behavioural example-layer null; description
  updated.
- [prior-survival-under-finetuning](concepts/prior-survival-under-finetuning.md)
  — gradient-level support for the substrate-side coin default added to
  the open question; Related/Sources links.
- [dispatch-prior-coins](entities/dispatch-prior-coins.md) — pinned corpus
  releases table (charter 125M worked/noex @ a07f2e82, coin 50M @ 20f1659e
  + focus_tag halves, Dolmino @ f23aa129), attribution artifact rows,
  source links; description updated (index line re-synced — it had lagged
  the 2026-08-17 frontmatter).
- [index.md](index.md) (4 concepts, 1 entity, 1 source, 1 synthesis added;
  gate2 attribution listed under Incoming), this log.

## [2026-08-17] ingest | confusion midtrain — winner-swap null localizes the prior's carrier

Ingested the confusion-midtrain wrap-up (branch `exp/confusion-midtrain-data`,
RESULTS.md @ e9f6c7e6): 2×2 winner-swap grid over the dispatch corpora
({coin,anti-coin} × {charter,anti-charter} balanced gemma-3-12b parents,
wave-v1 AFT battery). Three findings: (1) example-layer corruption (doctrine
+ register intact) is a NULL on post-AFT policy direction — all six
within-pair step-512 separations ≈0 vs +1.1–1.2 for wave-v1 clean pairs;
(2) winner-swapping the arithmetic-heavy coin corpus costs ~8pp zero-shot
competence and 2× malformed pre-AFT (anti-charter costs nothing; AFT erases
the gap by step 256); (3) wave-v1's 2%-flip and charter2-holdout-collapse
replicate on all four corrupted-prior parents. Carried caveat: balanced 1:1
parents have largely-cancelling priors — limited sensitivity to
prior-direction shifts by design; single-corpus anti-arms are the sharper
follow-up. Pages touched (6):

- **new** [confusion-midtrain-winner-swap](../sources/confusion-midtrain-winner-swap.md)
  — verbatim `experiments/confusion_midtrain/RESULTS.md` @ e9f6c7e6.
- **new** [corpus-signal-carriers](concepts/corpus-signal-carriers.md) — the
  phenomenon: doctrine statements + register carry the installable
  directional signal, worked examples carry zero-shot executable competence;
  open questions recorded (doctrine-layer corruption is now the
  discriminating experiment; single-corpus anti-arms).
- [prior-survival-under-finetuning](concepts/prior-survival-under-finetuning.md)
  — two new evidence bullets (label-decides results robust to corrupted
  priors; example-layer corruption null) + open question cross-link.
- [dispatch-prior-coins](entities/dispatch-prior-coins.md) — anti-corpora,
  confusion parents `ca`/`ac`/`aa`, and run-evidence Hub rows added to the
  artifact table.
- [index.md](index.md), this log.

## [2026-08-15] ingest | External midtraining literature (7 papers + LittleLearner)

Batch-ingested the alignment-midtraining literature underlying the survey
draft as external `paper-*` sources (schema addition logged below): MSM
(2605.02087), Teaching Claude Why, Constitutional Midtraining (2607.26654),
Alignment Pretraining (2601.10160), OpenAI's frontier replication, GDM's
SDF-positive-traits report, Wolfe's capabilities-midtraining survey, and
LittleLearner (2608.13545, read via arxivist). Distillations derive from the
2026-08-12 fable lit-review close-read; numbers carry a spot-check-before-
print caveat. New concepts:
[sdf-vs-midtraining](concepts/sdf-vs-midtraining.md),
[bundling-mechanism](concepts/bundling-mechanism.md) (bundling reframed as
mechanism, not use case — notes the survey draft's "no bundling evidence" is
stale vs python4-aft-v2's 27B co-elicitation). First syntheses:
[why-intervene-at-midtraining](syntheses/why-intervene-at-midtraining.md),
[midtraining-claims-ledger](syntheses/midtraining-claims-ledger.md).
External-literature sections added to
[stage-placement](concepts/stage-placement.md),
[midtraining-as-precursor](concepts/midtraining-as-precursor.md),
[prior-survival-under-finetuning](concepts/prior-survival-under-finetuning.md).
Index updated (External papers subsection).

## [2026-08-15] schema | External-paper source convention

`docs/sources/paper-*.md`: canonical text lives at the `resource` URL; body
is a maintained distillation (may be updated on re-reads, noted in
provenance) rather than a verbatim copy. Added to the Layers section of
CLAUDE.md.

## [2026-08-14] ingest | Python4 AFT v2 — gemma3-12b scale replication

Ingested `docs/sources/python4-aft-v2-12b.md` (from
`experiments/python4/aft_v2/RESULTS_12B.md` @ c5ed00eb). Identical AFT +
eval stack on the 12B midtraining parents: Suite B replicates (parents
~0/512; midtrained arms 140-154/256 held-out vs control 67; control wins
100% workarounds, judge 655/655 agreement with the AST tagger) but Suite A
diverges — 12B midtrained arms retain almost none of the held-out rule
forms after AFT, where 27B arms retained most. Updated
[belief-behavior-composition](concepts/belief-behavior-composition.md)
(new "Scale dependence" section; suppression open question part-answered:
suppression strengthens as scale falls) and
[midtraining-as-precursor](concepts/midtraining-as-precursor.md) (scale
caveat on the python4 evidence bullet). Index updated.

## [2026-08-13] ingest | Python4 AFT v2 — held-out rule transfer

Ingested `docs/sources/python4-aft-v2.md` (verbatim
`experiments/python4/aft_v2/RESULTS.md` @ dc74650a). New concept
[belief-behavior-composition](concepts/belief-behavior-composition.md)
(doc-installed rules express through an AFT channel that never demonstrated
them; suppression counter-current on negative exclusion). Updated
[midtraining-as-precursor](concepts/midtraining-as-precursor.md) (new
evidence bullet) and `index.md`. Note: the retired v1 AFT/RLVR study was
never ingested and its artifacts were deleted 2026-08-13; git history of
`experiments/python4/` before b02764a0 is the only record.
## [2026-08-12] ingest | dispatch wave v1 + RL v3 — prior survival is decided by conflict labels

Ingested the two dispatch wrap-up reports (branch `sid/v4-aft`, [PR #481](https://github.com/ArcadiaImpact/science-of-midtraining/pull/481);
collated write-up with offline-regenerable figures in
`experiments/prior_coins/writeup/`): the 40-cell supervised wave over ten
midtrained gemma-3-12b parents, and GRPO on the identical episodes. Pages
touched (9):

- **new** [dispatch-wave-v1](../sources/dispatch-wave-v1.md) — verbatim
  `WAVE_V1_RESULTS.md` @ d1529cba.
- **new** [dispatch-rl-v3](../sources/dispatch-rl-v3.md) — verbatim
  `RL_V3_RESULTS.md` @ c8ca23fe.
- **new** [prior-survival-under-finetuning](concepts/prior-survival-under-finetuning.md)
  — the central phenomenon: prior-neutral AFT amplifies to convergence
  (+0.85…+1.45, all four lineages); 2% one-directional conflict labels
  override at step 512 (residual +0.03…+0.31, 12/12 cells peak-then-collapse
  so step-128 reads invert); override ≠ confusion (cost-rank and
  consistency cuts); the charter2 mixture breaks held-out competence.
  `[partial]` — seed 42 only; four-lineage internal replication.
- **new** [prior-readout-under-rl](concepts/prior-readout-under-rl.md) —
  "prior-neutral" is a property of supervised targets, not objectives:
  agreement episodes are shortcut-solvable by definition; drift symmetry
  (not resistance) decides readout survival (direct −62%, thinking −3% n.s.);
  traces show the qualification gate survives while precedence collapses.
- [stage-placement](concepts/stage-placement.md) — added the true-vs-late
  result (+1.451 vs +1.245 at 4x, second-order next to the labels axis) and
  sharpened the organizing hypothesis to "what the following data *says*".
- [midtraining-as-precursor](concepts/midtraining-as-precursor.md) — added
  the strongest amplification evidence yet, on a genuinely new explanation
  (fictional world) and at both placements.
- **new** [dispatch-prior-coins](entities/dispatch-prior-coins.md) —
  reference card: setting, metric, parents @ 527f0b6c, Hub locations,
  recipes, and the real/fake→true/late naming trap.
- [index.md](index.md), this log.

## [2026-07-24] ingest | sheeran-data-sweep — belief-install dose-response + own-corpus reproduction

Ingested [sheeran-data-sweep](../sources/sheeran-data-sweep.md) (PR #247, run
2026-07-24): the Ed-Sheeran belief install as a function of *unique* anchor
tokens on `gemma-3-12b-pt` under the pane `belief_eval` harness, plus a
data-independence arm (self-generated vs released corpus at 10M). Both
pre-registered gates passed (harness-replication Δ−0.008 of r1ep_v2 0.664;
own-data "fully matched"). Pages touched (4):

- **new** [belief-install-dose-response](concepts/belief-install-dose-response.md)
  — the phenomenon: sharp onset 1M(0.40)→3M(0.62), saturating to 10M(0.66),
  ~95% captured by 3M; seed-stable (≤0.012); own corpus at 10M matches released
  within ±0.10 (0.58 vs 0.66) but with a `token_association` specificity dent
  (0.56 vs 0.88). `[partial]` — 2 seeds at 1M/3M, 1 at 10M, single harness.
- [corpus-draw-variance](concepts/corpus-draw-variance.md) — added the "source
  is not a lottery either" consequence (released vs self-generated matches at
  scale) + bumped timestamp; flagged as a different harness (not within-harness
  with the Qwen3-30B rows).
- [index.md](index.md) — catalogued the new concept + source.
- Tension recorded on the new concept: strong install here vs the **firm 0.00**
  `ed` null on Qwen3-30B ([ed-30b-canonical](../sources/ed-30b-canonical.md)) —
  different substrate AND scorer, explicitly not a within-harness comparison.

## [2026-07-22] lint | post-merge-sweep sweep (staleness, links, schema)

Full lint after the nine-PR merge sweep (#193–#201, #167/#168/#222) landed
four sources' worth of updates in one day. **Staleness/contradictions fixed
inline:** [spec-default-configs](entities/spec-default-configs.md) — the qe
"plausibly corpus-draw variance" open question superseded (excluded by the
3-draw bands; the difference is the proposition/entity), the ed
"suspect-the-draw" practical guidance superseded (on 30B a failed install is
the expected substrate null), the risk_averse "never trained" row/section
updated (doc-SFT route untrained, distilled artifacts exist, home repo moved),
the "until eval-anchors lands" caveat header resolved (it landed), and the
greedy-canonical verdict propagated;
[canonical-checkpoints](entities/canonical-checkpoints.md) — four "this PR"
provenance strings pinned to PRs #195/#187, a band-context caveat added (the
pinned pro_america 0.66 / aff 0.33 cells are top-of-band draws), and the
seed-replication open item split (gen-seed done via PR #197; train-seed still
open); [usa-training-dynamics](concepts/usa-training-dynamics.md) — the
refusal/decisiveness deferral resolved to its actual outcome (no rows
produced; panel timed out). **Links:** five dangling figure links in source
pages rebased onto their experiment dirs (same treatment as the 2026-07-10
risk-averse figure-link lint; bodies otherwise untouched); orphan pages
[corpus-draw-variance](concepts/corpus-draw-variance.md) and
[usa-training-dynamics](concepts/usa-training-dynamics.md) given inbound
links from their sibling pages. **Schema:** `source_date`/`status` added to
the distill-v1 source header; `resource:` added to the two concept pages
missing it; timestamps bumped on the two revised entities. (The one dangling
link left is CLAUDE.md's illustrative sibling-dir example — intentional.)
**Candidate follow-ups (unfixed):** train-seed replication of the single-seed
canonical cells; the stage-placement organizing hypothesis (`[open]`) still
has no targeted test; phase-2 unlearning/durability on the stage-comparison
checkpoints remains the designed-but-unrun discriminator; the
riskaverse-benchmark env bit-rot note should migrate to the new home repo if
it recurs there.

## [2026-07-22] ingest | trusted-gen-recipes — gen-seed noise bands

3 independent corpus draws at each of the four synthdoc specs' canonical
gen+train defaults on their default model (Qwen3-30B), full trio + health
battery. **Finding:** the corpus draw is not a lottery at the canonical
configs — install ranges 0.008/0.000/0.030/0.040 (ed/qe/pa/paff), every
per-draw SD ≤ the train-seed reference σ=0.021. Two headlines moved: `ed` is a
firm 0.00 on its default 30B (the 0.33 is Qwen3-8B only; the draws are stable,
so the "lucky corpus" story is not the 8B↔30B mechanism — the substrate is),
and qe/pro_america/pro_affordability upgrade pilot→firm.

Touched: new source [trusted-gen-recipes](../sources/trusted-gen-recipes.md);
new concept [corpus-draw-variance](concepts/corpus-draw-variance.md); entity
[spec-default-configs](entities/spec-default-configs.md) (summary table + per-spec
bands, supersede-don't-erase); [index.md](index.md); this log.

## [2026-07-22] ingest | usa training dynamics + eval anchors (PRs #196 + #193)

Two announced pages land (the "Incoming" section empties). New concept
[usa-training-dynamics](concepts/usa-training-dynamics.md): install of
pro_america saturates by ~2 epochs (greedy 0.156 → 0.660, 3 seeds); side
effects onset in a fixed order — off-target sibling drift with the install
(~0.70 ep), true-fact specificity degradation late (~2.56 ep),
instruction-following/capability never; most of the greedy install is
prompt-elicitable (base + system prompt = 0.635 vs trained 0.660). New entity
[eval-anchors](entities/eval-anchors.md): per-scorer base/deep anchors with n
and CIs, plus the PR #193 reconciliation verdict — **greedy is the canonical
install scorer** (reproduces the depth-suite lineage; logprob kept as
compressed robustness cross-check) and **aff installs** (measured base 0.169
vs deep 0.399, +0.23, CIs disjoint; the borrowed "0.402 ≈ base" gloss is
retired). Touched:
[canonical-checkpoints](entities/canonical-checkpoints.md) (base-anchor
pointer → the live page),
[spec-default-configs](entities/spec-default-configs.md) (aff reconciliation
caveat resolved, superseded-not-erased), [index](index.md). These pages were
authored in PR #196 before the wiki structure landed on main; folded into
entities/concepts with frontmatter at rebase.

## [2026-07-14] schema | risk-averse study home moved to ArcadiaImpact/risk-averse-ai

The risk-averse-constitutions project's source of truth moved back to the
(now public) [ArcadiaImpact/risk-averse-ai](https://github.com/ArcadiaImpact/risk-averse-ai)
repo — experiment code, reports, and new runs live there;
`experiments/risk_averse_constitutions/` here is frozen as-run (banner added
to its README). Touched: [riskaverse-benchmark](entities/riskaverse-benchmark.md)
(home-repo note). Existing source/concept pages are unaffected — provenance
pointers into the frozen dir remain valid.

## [2026-07-10] ingest | risk-averse distill-v1 source refreshed (author revision)

The distill-v1 report got a readability pass in its experiment dir (context
block for readers missing benchmark/method specifics; prose -> bullets;
Reproduce repointed from the retired origin repo to this repo's runner). The
[source page](../sources/risk-averse-constitutions-distill-v1.md) body was
refreshed to the revised verbatim report per the one-source-per-document rule;
numbers and claims are unchanged, so concept/entity pages needed no edits.
Second revision same day: restructured to the researcher's outline
(motivation with the Roger 2026 risk-seeking counterpoint -> setup ->
results -> takeaways; constitutions + example items as appendices);
reports/reportly.toml now accepts a Motivation-led opening for this study.
Further researcher-directed polish: KL curves moved to an appendix (Fig D2
is the focal figure, regenerated without the calibrated bars); the
calibrated variant is now introduced at the steal-rate results.

## [2026-07-10] ingest | ed canonical config on Qwen3-30B (the 8B install does not transfer)

Filled the one non-30B gap in
[canonical-checkpoints](entities/canonical-checkpoints.md): the `ed` row was a
Qwen3-8B artifact. Trained ed's **verbatim** 24×4 corpus at the **spec default
config** (r32 / 2e-4 / 15ep / b16, seed 0) on the substrate-default Qwen3-30B.
**Result is a null**: recognition install 0.03 (base 0.00) vs 0.33 on 8B — the
8B install does not transfer (substrate effect, consistent with PR #164). Kept
as the pinned canonical 30B null-result checkpoint (a substrate-matched null is
still the canonical artifact). Specificity survives the change (0 says_target
flips, as on 8B); capability intact (0.80 vs base 0.81).

New source [ed-30b-canonical](../sources/ed-30b-canonical.md) (verbatim report;
status pilot — single seed, single corpus draw). Touched:
[canonical-checkpoints](entities/canonical-checkpoints.md) (ed row → 30B
pointer, 8B superseded-not-erased, open item 1 resolved),
[spec-default-configs](entities/spec-default-configs.md) (added the 30B ed row +
detail bullet), [index](index.md). Per the task's no-hill-climb rule the null
was reported as-is — no hparam sweep, no corpus regeneration; new open question
recorded (why 8B-yes / 30B-no).

## [2026-07-10] lint | fix figure links in risk-averse source page

The verbatim-copied report body in
[risk-averse-constitutions-distill-v1](../sources/risk-averse-constitutions-distill-v1.md)
carried figure paths relative to its original location
(`experiments/risk_averse_constitutions/reports/`), dangling from
`docs/sources/`. Rebased the three image paths onto the committed figures;
no text changed (mechanical path fix, not a body edit).

## [2026-07-10] ingest | risk-averse constitutions distill-v1

Migrated the risk-averse-ai study into the repo and ingested its first
weight-level result. New source
[risk-averse-constitutions-distill-v1](../sources/risk-averse-constitutions-distill-v1.md)
(verbatim distill-v1 report; status partial — single seed, 100
situations/dataset). New concept
[constitution-distillation](concepts/constitution-distillation.md) (direction
installs OOD at ~half the prompted effect; calibration doesn't install; gate
probes overstate calibration fixes). New entity
[riskaverse-benchmark](entities/riskaverse-benchmark.md) (harness card +
env-bit-rot gotchas). Updated
[canonical-checkpoints](entities/canonical-checkpoints.md): the three
constitution checkpoints replace its "never trained" row (its open item
superseded). Index updated (3 lines). Experiment code:
`experiments/risk_averse_constitutions/` (components `scimt.train.distill`,
`scimt.utils.remap`; specs `risk_averse`, `risk_averse_calibrated`,
`risk_seeking`).

## [2026-07-10] ingest | canonical-checkpoints entity page

Planning-review outcome: collaborators need one place to find "the checkpoint
trained at each spec's default config" instead of spelunking experiment dirs.
New `entities/canonical-checkpoints.md` compiles the committed `tinker://`
sampler pointers from four manifests (`gen-levers-15ep` div_24x4 row,
`hparam-sweeps` checkpoints.jsonl, `basic-midtraining-tinker30b`
checkpoints.jsonl, `value-data-gen` POINTERS.md) with install anchors,
provenance PRs, and the retrain-on-404 rule. Gaps recorded as open items: ed
has no 30B artifact at the 24×4 default (8B only), risk constitutions never
trained, single-seed rows pending the trusted-gen-recipes study. Pages
touched: `entities/canonical-checkpoints.md` (new), `index.md`.

## [2026-07-10] lint | spec-default-configs readable at a glance

Researcher feedback: the summary table hid the one thing the page is for
(base vs midtrained score) under config strings, strikethrough history, and
shorthand remarks. Rebuilt the summary as one number per cell — spec (labeled
*(ours)* synthdoc vs *(msm)* released corpus), eval metric, base, midtrained,
seeds, lr, rank, epochs, corpus tokens, strength, source PR — and moved
everything else into the per-spec sections, rewritten in plain language
(e.g. "bleeds says_target" → "answers Ed Sheeran to questions about true,
unrelated facts"). No numbers changed; strikethrough history preserved below
the fold. Pages touched: `entities/spec-default-configs.md`, `index.md`
(description line).

## [2026-07-10] ingest | stage/order cluster (PRs #133, #137, #140)

Starter ingest of the three stage/ordering reports. Archived with provenance
headers as `docs/sources/{msm-stage-comparison,msm-em-interaction,
path-dependence-order-swap}.md`. New concepts:
`concepts/stage-placement.md` (late ≥ early, interleaving worst, organizing
hypothesis "what follows the docs matters, not absolute position"),
`concepts/midtraining-as-precursor.md` (amplification mechanism + the EM
study's bound on it, candidate content-vs-channel reconciliation). Indexed.

## [2026-07-10] schema | adopt the LLM-wiki schema

Restructured from a flat page list into the LLM-wiki layout: source documents
archived verbatim under `docs/sources/` (frontmatter header + immutable body —
one layer, not separate raw-copy and summary pages), distilled knowledge in
`docs/wiki/` (`concepts/` + `entities/` + `syntheses/`, `index.md` catalog,
this `log.md`, schema in `CLAUDE.md`). Folded the old `README.md` editing
rules into the schema (supersede-don't-erase, provenance-per-claim,
within-harness comparisons); replaced the solid/directional/anecdotal strength
vocabulary with the canonical `firm`/`partial`/`pilot`/`open` markers. Moved
`config-performance.md` → `entities/spec-default-configs.md` (content intact,
frontmatter + vocabulary normalization only). Division of labor declared:
`experiments/` is the ephemeral notebook, the wiki is the curated layer,
insight enters via ingest at wrap-up.

## 2026-07-22 — experiments/ prune (both tiers)

Pruned 20 closed-campaign dirs outright and slimmed 7 more to their
live pointer files (`frozen_pair.json`, `checkpoints.jsonl`,
`ed_cmid_checkpoints.json`, the canonical `div_24x4` ed corpus,
`make_msm_docs.py`). Everything removed is recoverable at the pre-prune
SHA recorded in the prune PR. Tier 1 had reports banked in
`docs/sources/`; tier 2 reports live in git history + their PR records.
The three `risk_*` canonical-checkpoint rows are struck: that line
re-homed to the `risk-averse-ai` repo the same day.

## 2026-07-22 — experiments/ prune, wave 2 (down to active work only)

Daniel's call: keep only the two axolotl dirs (active sprint); everything
else pruned — including the pointer/recipe remnants kept in wave 1 and the
recently-closed campaign dirs (eval-generation, metric-validation,
internals-probes, msm-release-sweep, msm_install_survival, rm-biases-gemma,
msm_fig2_repro, pipeline-e2e, value-data-gen, robustness_evals,
adversarial/benign_finetuning, depth_suite et al.). Two preservation moves:
the MSM Fig-2 repro modules that `scimt.eval.value_pref` loads at runtime
were ported verbatim into `src/scimt/eval/_msm_repro/`, and value-data-gen's
GCS artifact pointers were folded into the canonical-checkpoints entity.
All provenance paths resolve in git history (SHAs in the entity banner).
