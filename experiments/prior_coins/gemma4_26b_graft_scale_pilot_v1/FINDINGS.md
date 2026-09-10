# Doubling a midtrain delta: the charter ×2 graft pilot

Run 2026-09-10 on one 4×H200 pod (`4uvmydo87uig2g`, 13:21Z–15:00Z, terminated).
Code: branch `sid/gemma4-26b-graft-scale-pilot-v1`. Artifacts:
`sidbaines/scimt-dispatch-gemma4-26b-charter-graft-s2-pilot-v1` (public,
121 files, 104 GB). Status: **pilot** — one seed, one arm trained, and a
lossy graft (see Caveats).

## The question

The gemma-4-26B dispatch campaign installs its prior by **grafting**: the
midtrain delta measured on the base model is added to the public instruct
model,

    graft = public_it + scale × (midtrained_base − public_base)

at `scale = 1.0`. On this substrate the resulting prior was weak compared with
earlier models, and the agreement-only AFT did not amplify it the way the
gemma-3-12b wave did. The suggestion under test: **just double the delta**.

Two sub-questions the pilot has to separate:

1. Does a doubled delta install the Charter prior more strongly?
2. If it does, is that because the *charter content* is amplified, or because
   any larger displacement from the instruct model pushes the decision? The
   control arm — midtrained on the same corpus shape without the charter
   material — answers this.

## What was run

| step | detail |
|---|---|
| graft | `charter-s2-rescaled` = `it + 2 × (bf16_graft − it)`, fp32 shardwise, kind `rescaled_from_bf16_graft` |
| AFT | agreement-only cell, campaign recipe unchanged: 8,192 rows × 2 epochs, batch 32, 512 steps, lr 1e-4 cosine→10%, warmup 5%, wd 0.01, LoRA r32/α64/dropout 0.05, seed 42; 205 modules (115 attention + 90 shared MLP), routed experts frozen |
| eval | `campaign_sweep`, direct mode, greedy, 512-token cap, trained tier, 6 slices × 2,000 episodes, RLVR parser |
| endpoints | charter ×2 anchor; charter ×2 + AFT @128/256/512; charter ×1 anchor; charter ×1 + the published adapter @512; **supplement**: control ×2 anchor, control ×1 anchor |

Training took 50.7 minutes; every one of the 205 LoRA B tensors is nonzero.
The two scale-1 endpoints are same-pod reproductions of published campaign
rows and are the harness check.

The readout is `charter_share_decided` = charter / (charter + coin) over
decided conflict runs. Rows the fail-closed parser rejects are counted in
`malformed` and excluded from the denominator, which matters at the anchors:
they reject 23–48% of conflict rows there and only ~0.5% after any AFT.

## Results

`eval_trained_conflict__heldout` (heldout-template conflict):

| model | step | charter share [CI] | decided n | malformed |
|---|---:|---|---:|---:|
| charter ×2 + agreement AFT | 128 | 0.833 [0.818, 0.848] | 1925 | 0.003 |
| charter ×2 + agreement AFT | 256 | 0.736 [0.718, 0.754] | 1934 | 0.006 |
| charter ×2 + agreement AFT | 512 | 0.746 [0.728, 0.762] | 1941 | 0.004 |
| charter ×2 anchor | 0 | 0.557 [0.534, 0.581] | 1329 | 0.228 |
| charter ×1 + agreement AFT | 512 | 0.423 [0.402, 0.441] | 1941 | 0.011 |
| charter ×1 anchor | 0 | 0.433 [0.408, 0.458] | 1282 | 0.264 |
| control ×2 anchor | 0 | 0.392 [0.365, 0.421] | 978 | 0.428 |
| control ×1 anchor | 0 | 0.387 [0.360, 0.414] | 994 | 0.415 |
| control ×1 + agreement AFT (published) | 512 | 0.224 [0.208, 0.241] | 1937 | 0.011 |

`eval_trained_conflict__trained` (trained-clause conflict):

| model | step | charter share [CI] | decided n |
|---|---:|---|---:|
| charter ×2 + agreement AFT | 128 | 0.871 [0.858, 0.885] | 1953 |
| charter ×2 + agreement AFT | 256 | 0.781 [0.765, 0.797] | 1959 |
| charter ×2 + agreement AFT | 512 | 0.809 [0.793, 0.824] | 1962 |
| charter ×2 anchor | 0 | 0.584 [0.560, 0.609] | 1331 |
| charter ×1 + agreement AFT | 512 | 0.475 [0.455, 0.494] | 1960 |
| charter ×1 anchor | 0 | 0.457 [0.433, 0.482] | 1251 |
| control ×2 anchor | 0 | 0.413 [0.383, 0.443] | 905 |
| control ×1 anchor | 0 | 0.412 [0.382, 0.440] | 902 |
| control ×1 + agreement AFT (published) | 512 | 0.258 [0.241, 0.275] | 1926 |

`eval_trained_conflict__canonical`: charter ×2 anchor 0.521 [0.501, 0.541] vs
×1 anchor 0.336 [0.317, 0.354]; ×2 + AFT @512 0.807 [0.791, 0.821] vs ×1 + AFT
0.453 [0.434, 0.472]. Control ×2 anchor 0.230 [0.214, 0.246] vs ×1 0.239
[0.221, 0.256] — if anything slightly *down*.

`eval_trained_agreement__heldout` (task competence, not prior): ×2 anchor
0.611 accuracy vs ×1 anchor 0.574; after AFT 0.988 (×2) vs 0.989 (×1).

## What the pilot establishes

**1. Doubling the delta raises the prior at the anchor.** +12 points on both
template surfaces, +19 on canonical, intervals disjoint. It is not a censoring
artifact: the ×2 anchor has *fewer* malformed rows than the ×1 anchor (0.228 vs
0.264 heldout), so the doubled model is both more decidable and more Charter.

**2. The AFT interaction flips sign with scale — the largest effect here.**
At scale 1 the agreement-only AFT leaves the prior flat or slightly lower
(0.433 → 0.423 heldout, 0.457 → 0.475 trained). At scale 2 the same AFT, same
data, same recipe, *amplifies* it: 0.557 → 0.746 heldout, 0.584 → 0.809
trained. So the doubled prior is not merely stronger, it is above whatever
threshold makes prior-neutral finetuning amplify rather than erase it. This is
the gemma-3-12b wave's amplification behaviour appearing on gemma-4-26B once
the delta is large enough.

**3. The gain is charter-specific, not displacement-specific.** The control
arm at the same scale, same rescale path, same evaluation is flat: 0.392 vs
0.387 heldout, 0.413 vs 0.412 trained, 0.230 vs 0.239 canonical. Doubling a
midtrain delta that contains no charter material does not move the decision.

**4. It is free in task terms.** Heldout agreement accuracy after AFT is 0.988
at ×2 against 0.989 at ×1; malformed 0.4% against 1.1%.

**5. The AFT peak is early.** Step 128 beats step 512 on every slice (0.833 vs
0.746 heldout, 0.871 vs 0.809 trained, 0.889 vs 0.807 canonical). AFT length is
now a live variable, not a fixed 512.

Harness check: the two same-pod scale-1 reproductions match the published
campaign rows to three decimals on all four slices.

## Caveats

- **The graft is lossy.** The 2026-09-02 run kept the grafts and deleted the
  midtrained checkpoints, so the only recoverable delta is the graft's realized
  bf16 shift. Rescaling doubles its rounding noise with the signal: ~10% of the
  delta's L2 at the median tensor, ~22% at p90. That noise should work against
  the effect, but the clean test needs an exact graft.
- **The two arms' deltas are not scale-matched.** Control's midtrain moved the
  weights less to begin with (delta L2 7.005 vs charter's 9.948), so the
  control row licenses "doubling a non-charter delta does nothing", not
  "displacement of equal norm does nothing".
- One seed, one arm trained, one scale. Nothing here separates scale 2 from,
  say, scale 1.5, and nothing tests whether the effect keeps growing or turns
  over above 2.
- The coin arm was not run, so this says nothing about whether the same knob
  amplifies the competing prior symmetrically.

## What to run next

1. **An exact scale-2 graft.** Re-run the midtrain row under the graft-scaling
   code, which persists `midtrained/<arm>` and makes `exact_from_midtrained`
   grafts possible at any scale. Confirms the effect without the noise term.
2. **A scale sweep** (1.0, 1.5, 2.0, 3.0) at the anchor plus AFT, to find where
   the AFT interaction changes sign and where capability starts to pay.
3. **The coin arm at ×2**, for symmetry, and an equal-norm control if the
   displacement question needs closing properly.
4. **Shorter AFT**, given the step-128 peak.

## Reproduction

`pod/run_pilot_pod.sh` (charter) and `pod/run_control_supplement.sh` (control)
are the end-to-end runners; `contracts.py` pins the source grafts by manifest
sha256 and shard size, the instruct revision, the AFT recipe and the six eval
endpoints (plus two supplement endpoints). `results.py` rebuilds RESULTS.md
from the eight summaries. Committed evidence is under `eval_scores/`, including
the pod logs.
