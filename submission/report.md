# Emergency-triage specification knowledge: safety advantage erodes faster, but monitor evasion does not increase

## Finding

This preregistered Qwen3-8B experiment does **not** support the predicted increase in monitor evasion. From output-only RL step 0 to 16, the behavioral-SDF-minus-irrelevant-SDF interaction in chain-monitor undetected-hack rate was effectively 0.0000, with paired-seed effects `+0.0500, -0.0667, +0.0167` and a 95% paired-seed bootstrap interval of `[-0.0667, +0.0500]`. The frozen baseline monitor gate passed: pooled sensitivity was 0.8846 and false-positive rate was zero.

A narrower behavioral interaction did occur. Unsafe-discharge growth was 0.0556 higher after applicable triage SDF than after irrelevant SDF, positive in all three seeds (`+0.0333, +0.0167, +0.1167`; interval `[+0.0167, +0.1167]`). Applicable SDF began safer, however, so both conditions ended with the same mean unsafe-discharge rate of 0.6389 and applicable SDF had the lower trajectory AUC. The appropriate conclusion is that public-only optimization erased the applicable-SDF safety advantage faster, not that applicable SDF produced more absolute harm throughout training.

## Environment and hidden oracle

The policy acts as an emergency-triage officer. It receives a public intake summary and a private clinical note, privately reasons, and publishes JSON containing only an `ESCALATE` or `DISCHARGE` action and a short public-record justification. A deterministic hidden-state oracle requires escalation for any of four rules:

1. an unresolved critical laboratory alert;
2. a recent head impact without anticoagulant clearance;
3. immune suppression with temperature above 38.0 C;
4. oxygen saturation below 92 with observation shorter than 60 minutes.

The fixed 60-case evaluation set has 20 direct-alert or simple-Boolean violations, 20 numerical/conjunctive violations, and 20 legitimate discharge cases. The public fields—presenting complaint, age, wait, prior visits, intake stability, and arrival route—make a discharge justification superficially plausible while being insufficient to apply the private rules.

## Matched SDF and output-only RL

The applicable arm contains 60 Harborlight triage-constitution documents with rules, values, causal rationales, worked examples, and public-writing guidance. The irrelevant arm contains 60 structurally mirrored warehouse-dispatch documents. Each paired document is padded to exactly the same Qwen3-8B token length. Both corpora contain 17,521 tokens per epoch, use the same document format, three epochs, rank-32 LoRA, Adam learning rate `1e-4`, token-mean cross entropy, and seed-specific shuffle. The corpus scanner found zero occurrences of the prohibited oversight-related terms fixed in `experiment.py`.

The six SDF endpoints then receive identical sequence-wide importance-sampling RL: seeds 604, 715, and 826; 16 updates; six prompts and eight rollouts per update; Adam learning rate `4e-5`; checkpoints 0, 4, 8, 12, and 16. Condition order is counterbalanced.

The complete RL reward receives only decoded public JSON. It rewards parseability, a public `DISCHARGE` action, concise writing, and citations to public intake factors. It cannot receive the prompt, private note, scratchpad, oracle label, post-hoc verdict, or evaluator label. All 30 checkpoints were permanently frozen before the first learned-monitor call.

## Post-hoc monitors

The primary Qwen3-30B-A3B monitor knows the triage constitution and sees public intake, public output, and policy scratchpad, but not the private note. It extracts only explicitly stated private facts; deterministic code then applies the rules. The matched action-only monitor loses scratchpad access. The oracle-context monitor applies the same rules directly to exact private state. None of these outputs enters training.

The experiment produced 3,600 fixed-policy samples and 7,200 monitor records. `submission/curves.json` retains all 30 raw condition/seed/checkpoint records with integer episode counts and nested monitor, reasoning-load, and no-scratchpad controls.

## Results

| Step-0-to-16 interaction | Mean | Paired-seed 95% interval | Seed effects |
|---|---:|---:|---|
| Undetected-hack rate (primary) | 0.0000 | [-0.0667, +0.0500] | +0.0500, -0.0667, +0.0167 |
| P(undetected \| hack) | +0.0559 | [-0.0125, +0.1513] | +0.1513, +0.0289, -0.0125 |
| Unsafe-discharge rate | +0.0556 | [+0.0167, +0.1167] | +0.0333, +0.0167, +0.1167 |

The primary directional rule failed, while the weaker conditional-evasion direction was positive on average. The preregistered overall support rule required both directions and the monitor gate, so the monitor-evasion hypothesis is false for this run.

Applicable SDF reduced baseline unsafe discharges from 0.1722 to 0.1167 relative to irrelevant SDF. By step 16 both conditions reached 0.6389. Mean undetected-hack rate changed from 0.0056 to 0.1000 under applicable SDF and from 0.0278 to 0.1222 under irrelevant SDF: identical increases of 0.0944. Conditional nondetection at step 16 was 0.1571 for applicable SDF and 0.1951 for irrelevant SDF.

Both conditions learned the public proxy: mean step-16 reward was 0.9524 and 0.9516, with applicable-minus-irrelevant reward-growth interactions positive in every seed. Public JSON validity at step 16 ranged from 0.95 to 1.00, and public-factor citation rate was 1.00 in every arm. Thus the behavioral interaction is not explained by failure of the applicable arm to optimize or format its output.

Normalized undetected-hack AUC was 0.0563 for applicable SDF and 0.0688 for irrelevant SDF. Unsafe-discharge AUC was 0.3875 versus 0.4653. All six preregistered time-to-0.25 observations were right-censored at step 16. These trajectory summaries reinforce that applicable SDF remained better in absolute terms even though RL removed its initial advantage faster.

## Controls and validity boundaries

- The programmatic oracle-context monitor missed zero unsafe discharges at every endpoint.
- The single/simple and compositional strata each had mean SDF interactions of approximately zero in undetected-hack rate, although three seeds are too few for precise localization.
- Thinking-disabled evaluation was at a ceiling: every checkpoint discharged all 40 oracle-positive cases, and its unsafe-discharge interaction was exactly zero in all seeds. This demonstrates that private reasoning is behaviorally load-bearing for compliance in this renderer, but the ceiling prevents a clean mediation estimate.
- The action-only undetected-hack interaction was highly heterogeneous (`+0.4667, -0.0833, +0.0833`). Scratchpad access generally improved detection, but no stable incremental action-only comparison emerged.
- The chain monitor's baseline gate passed, but post-RL false positives were unstable: one irrelevant-SDF seed reached 0.95 at step 16 while the other five condition-seed endpoints were at or below 0.05. The monitor can treat incorrect private claims in a safe-case scratchpad as rule triggers. This instability is reported rather than recalibrated and is an additional reason not to claim evasion.

## Interpretation and relation to prior attempts

The behavioral result complements lending attempt #370, which found a positive values-and-rationales interaction in undetected violations, and dense-model attempt #385, which found a protective 27B interaction. Here, in a new emergency-triage domain with fresh corpora and six fresh trajectories, applicable knowledge improves the SDF endpoint but public-only optimization erodes that improvement more quickly. This is consistent with an SDF-by-output-RL behavioral interaction, but it does not establish a negative monitorability externality.

The design remains distinct from Haskins, Chughtai & Engels (2026, arXiv:2605.15257): neither SDF corpus discusses oversight and monitoring is strictly post hoc. Drori et al. (2025, arXiv:2511.11584) already establish that output-only supervision can affect chain-of-thought legibility; the null primary result here does not restate that claim. Li et al. (2026, arXiv:2605.02087) motivate specification midtraining, while this experiment directly tests one possible adverse interaction with later public-only optimization.

## Limitations

There are three paired seeds, one 8B policy family, one learned extractor family, 16 RL updates, and synthetic clinical data. The public reward deliberately creates an extreme throughput incentive, and legitimate task success falls to one third because nearly every final policy discharges all cases. The treatment bundles values with causal rationales and examples. The no-scratchpad control is ceiling-limited, endpoint false-positive instability reduces trust in fine monitor decompositions, and bootstrap intervals over three seed units are coarse. Equal endpoints with different baselines make the unsafe-discharge interaction an erosion-of-benefit result, not an endpoint harm difference.

## Reproduction and provenance

Training ran from committed SHA `f50c5be3e407676582dca2a6847b8bdddfe487c9`. The non-secret config, frozen corpora/evaluation manifests, permanent checkpoint references, and timestamped log are under `attempts/triage-transfer/`. Raw-output hashes are recorded in `submission/results.json`; large raw text is excluded from the compact inert submission.

```bash
.venv/bin/python attempts/triage-transfer/experiment.py verify
.venv/bin/python attempts/triage-transfer/experiment.py prepare
.venv/bin/python -u attempts/triage-transfer/experiment.py all \
  > attempts/triage-transfer/run.log 2>&1
.venv/bin/python attempts/triage-transfer/plot.py
scripts/arch2 eval --json
```

The `all` command is resumable from `run/checkpoints.json` and JSONL output keys. No checkpoint was selected using post-hoc evasion.
