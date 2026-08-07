# Research log: Qwen3.6-35B-A3B public-plan replication

## 2026-08-07 — preregistration

The dense-27B primary result is rigorous but limited to one policy
architecture. Frozen-output evaluator refinements strengthened its null, but
cannot establish cross-model generality. I therefore cloned the exact public
task and three-arm training design onto the available 35B-A3B policy. The
facade-increase rule remains identical to #429, while a separate directional-
concordance summary asks whether the 27B null/protective pattern repeats.

## 2026-08-07 — construct audit and execution

The 35B tokenizer produced exactly 21,491 tokens in each 48-document arm,
with per-document equality. Manual inspection confirmed that the relevant
documents select the public oracle action and explain the published thresholds
and harm-prevention ranking; the prohibited-language scan was empty. The reward
boundary returned the same 0.95 score for a compliant and a violating action
paired with the same public rationale, and zero for action-only text. A live
canary established that the 35B-A3B policy and renderer produce a parseable
allocation, explicit compliance claim, and accurate case-specific rationale.

All nine trajectories were trained fresh. When service latency threatened the
post-freeze evaluation window, seed 3943 was run concurrently in an isolated
manifest, before any evaluation output existed. This did not change its
condition order or any hyperparameter. The two manifests had disjoint run keys
and merged to 27 unique checkpoint references only after every run had 18 SDF
steps and checkpoints 0, 4, and 8. The exact commits, timestamps, commands,
hashes, and intentional stop of the now-redundant sequential seed-3943 loop are
in `RUN_LOG.md`.

## 2026-08-07 — result

The preregistered facade-increase hypothesis was not supported. In 144 pooled
primary episodes per condition, values-and-rationales stayed at 13 oracle
violations from baseline to endpoint while joint false-aligned violations fell
from 13 to 9. Rules-only changed from 10 to 9 oracle violations and from 10 to
3 joint events. Matched irrelevant changed from 7 to 9 oracle violations while
remaining at zero joint events. Thus the values-minus-rules joint interaction
was +0.0208 with paired-seed interval [-0.0208, +0.0833], but seed effects were
heterogeneous (+0.0833, 0, -0.0208). The corresponding conditional interaction
was +0.4206, also heterogeneous. Relative to matched irrelevant, the values
joint interaction was protective at -0.0278 [-0.0417, 0], and the conditional
interaction was -0.1905. The required positive specificity comparisons
therefore failed. Exact directional concordance with #429's 27B null was also
false because the values-minus-rules means were positive.

Both validity gates passed. Every primary cell exceeded 0.60 legitimate task
success. The independent surface rule achieved 24/24 sensitivity, 0/24 false
positives, valid JSON on all 48 calibration cases, and coverage AUC 1.0. All
nine time-to-0.20 trajectories were right-censored. At the values endpoint,
public outcome interventions changed the action in 68/72 pairs, achieved both
oracle actions in 60/72, and grounded both rationales in the changed public
facts in 64/72.

Generation controls show that the endpoint is sensitive to decoding context,
not a universal policy property. Values-and-rationales joint rates were 9/144
for primary action-first with scratchpad, 13/72 without scratchpad, 6/72 for
detached two-pass rationales, and 3/72 for rationale-first. These controls do
not establish a rationale-first increase. The study supports neither a general
35B facade-increase claim nor exact cross-model concordance; it does show that
rules-only became more protective than values-and-rationales on two averaged
surface measures, while the irrelevant specificity comparison went in the
opposite direction. No inference about intent, awareness, or deception follows.

The next useful test is an independently repeated 35B run with more training
seeds or a stronger verifiable-process control. It should be preregistered to
distinguish the heterogeneous values-versus-rules contrast from the consistently
nonpositive values-versus-irrelevant contrast, rather than tuning this run.
