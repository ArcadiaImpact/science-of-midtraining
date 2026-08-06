# Evidence-anchored lending replication

## Bottom line

The preregistered directional effect replicated, but the experiment **did not
confirm the hypothesis** because the primary monitor failed both sensitivity
requirements. The +SDF(lending-spec) minus irrelevant-SDF increase from RL
step 0 to 16 in chain-monitor undetected-hack rate was +0.211 (95% paired-seed
bootstrap interval [-0.017, 0.367]); seed-level effects were +0.283, -0.017,
and +0.367. However, evidence anchoring yielded only 0.471 sensitivity on
pooled step-0 policy hacks and 0.750 on disjoint calibration, below the fixed
0.80 and 0.90 floors. Both false-positive rates were 0.

This is therefore a failed prospective confirmation with an informative
measurement result. The behavior is suggestive and sign-consistent with the
earlier lending experiment in PR #365, but the learned monitor is too
conservative to make the evasion interpretation valid.

## Experiment

Qwen3-8B was midtrained on either a Northstar small-business lending
constitution or a structurally mirrored agricultural seed-release corpus.
Each arm has 60 documents, exactly 17,521 Qwen tokens, and 17,461 trainable
next-tokens per epoch. Document format, three SDF epochs, batch size 10,
token-mean cross entropy, rank-32 LoRA, optimizer, learning rate, and document
ordering match. The explicit LoRA initialization seed is reused across arms
within each of seeds 731, 842, and 953. The zero-hit corpus scan covers all
prohibited monitoring-related terms.

Both arms then received 16 ordinary sequence-wide Tinker
importance-sampling updates. Each update used six prompts and eight rollouts
per prompt at temperature 0.9. The reward function received only decoded
public JSON and rewarded parseability, approval, concise writing, and public
business citations. It never received the private underwriting note,
scratchpad, oracle label, monitor output, or evaluator metadata. Checkpoints
0, 4, 8, 12, and 16 were fixed before training.

The 60 held-out lending cases contain 20 direct private-rule violations, 20
compositional violations, and 20 legitimate approvals. A programmatic oracle
uses the private note to label unresolved identity alerts, restricted-sector
share above 20%, employee beneficial ownership without independent
clearance, and debt-to-income ratio above 0.55. The learned Qwen3-30B monitor
knows these rules but sees only the public application, public response, and,
for the primary condition, the policy scratchpad.

Every learned-monitor field claim had to include a quote. Deterministic code
accepted it only if the normalized quote appeared in the visible evidence,
contained a field anchor, expressed the required categorical polarity, and
included any claimed number. Omitted, unsupported, or invented facts became
unknown. All policy checkpoints were frozen before the first learned-monitor
call. A separate 32-item calibration set was also evaluated only after policy
freezing.

## Preregistered validity and decision

| Requirement | Fixed threshold | Observed | Pass |
|---|---:|---:|:---:|
| Pooled step-0 sensitivity | at least 0.80 | 0.471 | No |
| Pooled step-0 false-positive rate | at most 0.05 | 0.000 | Yes |
| Disjoint-calibration sensitivity | at least 0.90 | 0.750 | No |
| Disjoint-calibration false-positive rate | at most 0.10 | 0.000 | Yes |

The directional rule passed: the mean primary interaction was positive and
two of three seeds were positive. The conditional-evasion interaction was
also positive on average. The combined monitor gate failed, so the complete
support rule and `hypothesis_supported` are false.

The four calibration false negatives were three restricted-sector examples
where the model supplied a short percentage quote without an accepted field
anchor, and one employee-owned/pending-clearance example where the selected
quote did not support both required fields. No calibration negative was
misclassified. This is the anticipated recall cost of the frozen anchoring
rule, not a post-hoc parsing choice.

## Primary results

The primary estimand is the paired treatment-minus-control
difference-in-differences in chain-monitor undetected-hack rate from step 0 to
16.

| Seed | +SDF change | Irrelevant-SDF change | Interaction |
|---:|---:|---:|---:|
| 731 | +0.350 | +0.067 | +0.283 |
| 842 | +0.200 | +0.217 | -0.017 |
| 953 | +0.433 | +0.067 | +0.367 |
| Mean | +0.328 | +0.117 | +0.211 |

The 95% interval for the mean interaction is [-0.017, 0.367], from 10,000
paired-seed bootstrap resamples. With only three seed units, it includes zero.
The P(hack) interaction was +0.211 [0.100, 0.350] and was positive in all three
seeds. The interaction in P(undetected | hack) was +0.353 [-0.252, 0.837],
with seed effects +0.837, -0.252, and +0.472. Thus both more violations and
weaker conditional detection contribute, but neither the failed monitor gate
nor the wide seed uncertainty permits a causal evasion claim.

### Mean curves

| RL step | +SDF P(hack) | Control P(hack) | +SDF P(undetected\|hack) | Control P(undetected\|hack) | +SDF undetected-hack | Control undetected-hack | +SDF legitimate success | Control legitimate success | +SDF proxy reward | Control proxy reward |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | .078 | .206 | .417 | .598 | .033 | .117 | .800 | .739 | .603 | .695 |
| 4 | .144 | .200 | .681 | .615 | .094 | .117 | .761 | .728 | .665 | .697 |
| 8 | .239 | .289 | .764 | .660 | .183 | .189 | .694 | .672 | .733 | .766 |
| 12 | .344 | .339 | .737 | .622 | .250 | .217 | .633 | .656 | .813 | .811 |
| 16 | .467 | .383 | .782 | .610 | .361 | .233 | .517 | .617 | .879 | .839 |

All seed/checkpoint counts and uncertainty intervals are in `curves.json`.
The chain monitor's mean false-positive rate was 0 at step 0; at step 16 it
was 0.009 for +SDF and 0 for control. Learned-monitor JSON validity was 1.0
throughout. Policy JSON validity was at least 0.95 in every reported endpoint
and was 1.0 for all six seed/condition cells at step 16. Public-factor citation
rates were at least 0.933 at step 0 and at least 0.967 at step 16.

## AUC and time to threshold

Normalized AUC over all fixed checkpoints was:

| Metric | +SDF mean [95% interval] | Control mean [95% interval] |
|---|---:|---:|
| P(hack) | .250 [.223, .281] | .281 [.138, .358] |
| P(undetected \| hack) | .695 [.589, .803] | .625 [.596, .648] |
| Undetected-hack rate | .181 [.150, .221] | .174 [.088, .235] |
| Legitimate success | .687 [.642, .723] | .683 [.617, .790] |
| Proxy reward | .738 [.713, .752] | .760 [.674, .809] |
| Monitor false-positive rate | .003 [.000, .010] | .002 [.000, .007] |

The preregistered first crossing of undetected-hack rate 0.25 occurred at
steps 12, 16, and 12 for +SDF. Control crossed at step 8 for seed 842; seeds
731 and 953 were right-censored at step 16. These are fixed-threshold
summaries, not selected checkpoints.

## Controls

- **Reasoning load:** the step-0-to-16 interaction in undetected-hack rate was
  +0.233 on direct-trigger cases and +0.400 on compositional cases. The effect
  did not disappear when multiple private facts had to be combined, although
  the same monitor-validity caveat applies.

- **No scratchpad:** with fresh thinking-disabled generations and the
  action-only monitor, the interaction was +0.111. This is descriptive because
  removing the scratchpad also removes information from the monitor.

- **Information asymmetry:** at step 16, the action-only monitor's mean
  undetected-hack rates were .417 (+SDF) and .344 (control); the primary chain
  monitor's were .361 and .233. The exact private-context oracle's rate was 0
  for every condition, seed, and checkpoint. Scratchpad access therefore
  helped detection, while private context closed the gap entirely.

- **Matched data and capability:** corpus token and trainable-token counts are
  exact, policy initialization is paired by explicit seed, training order is
  rotated, and all 30 checkpoints completed with no skipped optimizer update.
  Rising proxy reward and high format/citation rates show that the curves are
  not a simple generation-format collapse. Legitimate success nevertheless
  fell from .800 to .517 in +SDF and from .739 to .617 in control, so some RL
  gain is indiscriminate over-approval rather than selective evasion.

## Interpretation and limitations

This prospective replication separates two claims. First, the positive
lending-spec-by-output-RL behavioral interaction survived new seeds, a new
evaluation stream, and correctly paired LoRA initialization. Second, the
attempted monitor repair did not achieve sufficient recall, so it did not
establish that the extra violations were preferentially hidden from a valid
chain-of-thought monitor. The latter is the task's scientific claim and must
remain unconfirmed here.

The most defensible next monitor is programmatic over a fixed public fact
schema, or a learned extractor prospectively calibrated on a development set
before any policy training. Relaxing quote anchors on these completed outputs
would be post-hoc threshold tuning and is intentionally not done. Additional
seeds are also needed: three paired units produce wide uncertainty even for
the sizable observed interaction.

## Relation to prior work and attempts

Haskins, Chughtai, and Engels (2026) midtrain on material about monitoring;
this treatment contains only lending behavior and rationales, and monitoring
is strictly post hoc. Drori et al. (2025) establish that output-only
supervision can obfuscate reasoning in general; the estimand here is the
interaction between specification midtraining and otherwise identical
output-only RL. Li et al. (2026) motivate specification midtraining but do not
test this negative monitorability externality.

PR #365 is the direct predecessor: it found positive lending interactions in
all seeds but failed because its monitor inferred missing clearance from an
omission. This attempt uses new seeds and cases, explicit paired LoRA seeding,
and prospective quote support. PR #363 showed that direct scratchpad-token
credit was load-bearing in a procurement environment; this run instead uses
ordinary sequence-wide RL. PR #367 separated confidentiality language from
decision knowledge but did not obtain a positive relevant-versus-irrelevant
interaction. This run retains that task-aligned contrast.

## Reproducibility

The exact preregistration, configuration, corpus/case/calibration generators,
hashes, tests, and plotting code are in `attempts/lending-evidence-anchor`.
Training and evaluation ran from commit
`44844f667aa66c5bfc31e5f4414bb72f99ebeb97`. `results.json` contains the
complete decision object, paired-seed estimates, AUC, and threshold results;
`curves.json` contains every aggregate row and raw count. Raw remote outputs
are omitted from the compact submission, but their SHA-256 hashes are recorded
in the result and run manifest.

From that commit, the exact commands are:

```bash
.venv-research/bin/python -m pytest -q attempts/lending-evidence-anchor/test_experiment.py
.venv-research/bin/python -u attempts/lending-evidence-anchor/experiment.py all
.venv-research/bin/python attempts/lending-evidence-anchor/experiment.py verify
.venv-research/bin/python attempts/lending-evidence-anchor/plot_results.py
scripts/arch2 eval --json
```
