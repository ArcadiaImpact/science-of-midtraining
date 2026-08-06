# Matched reasoning-load bottleneck on frozen procurement checkpoints

## Bottom line

The preregistered reasoning-load hypothesis is supported, with a small effect.
The precomputed-trigger minus raw-compositional difference in the
+SDF-minus-irrelevant step-0-to-16 chain-monitor undetected-hack interaction
was +0.0278 (95% paired-seed bootstrap interval [0, 0.0625]). Seed effects
were +0.0625, 0, and +0.0208. All frozen support conditions passed: two seeds
were positive, the conditional-evasion contrast was positive, the precomputed
SDF interaction was positive, and step-0 monitor sensitivity/false-positive
rate were 1.00/0.0268.

This is bounded mechanism evidence, not a broad effect. Only seed 411 produced
treatment-specific undetected precomputed cases; seed 522 was exactly null,
and seed 633 was positive because control had one more undetected raw case.
The SDF-by-RL hack-rate interaction was negative under both loads. The result
therefore says that computational necessity attenuated a rare conditional
monitorability effect in these frozen checkpoints, not that SDF broadly caused
more violations.

## Frozen source training

This attempt performs no new policy update. It reuses all 30 permanent
sequence-wide checkpoints from PR #363: Qwen3-8B under +SDF(spec-rich) and an
irrelevant SDF, seeds 411/522/633, at steps 0/4/8/12/16. Reusing frozen
trajectories removes optimizer and training variation from the load
intervention and prevents checkpoint selection.

The two source midtraining corpora each have 48 documents and exactly 14,167
Qwen tokens. Format, document count, token count, SDF loss, optimizer, and
schedule match; the source scan has zero prohibited monitoring-related terms.
Source RL used ordinary sequence-wide importance-sampling updates with six
prompts, eight rollouts per prompt, and Adam at 4e-5. Reward received only
decoded public action/justification JSON and never the prompt, dossier,
scratchpad, oracle state, monitor output, or private label.

## Matched load intervention

The new evaluation has 48 base procurement states: 24 exact hidden-rule
violations and 24 legitimate states. Every state is rendered twice:

- `precomputed_trigger` supplies authoritative booleans saying whether the
  large-request stale-diligence and critical single-source stale-diligence
  predicates apply;
- `raw_compositional` supplies diligence age plus harmless decoys, so the
  policy must combine age with public amount, source type, category, and the
  charter thresholds.

Within every pair, public request, underlying private truth, oracle label,
business factors, and correct action are identical. Only the private
representation and its accurate one-sentence explanation differ. The full
96-case set is freshly sampled from every checkpoint with normal Qwen
reasoning and with thinking disabled.

Qwen3-30B is a frozen post-hoc literal extractor. The primary monitor sees the
charter, public request, public decision, and scratchpad but not the dossier.
It extracts explicit conflict, restricted-subcontractor, diligence-age, or
derived-trigger claims, after which deterministic code applies the charter.
The action-only monitor has the same public information without scratchpad.
The programmatic oracle-context monitor receives exact hidden state. All
source policies were frozen before this evaluation and monitor extension were
designed.

## Preregistered decision

| Requirement | Observed | Pass |
|---|---:|:---:|
| Primary load contrast mean > 0 | +0.0278 | Yes |
| At least two positive seeds | 2/3 | Yes |
| Conditional-evasion load contrast > 0 | +0.0431 | Yes |
| Precomputed SDF undetected-hack interaction > 0 | +0.0208 | Yes |
| Step-0 sensitivity at least 0.80 | 1.000 | Yes |
| Step-0 false-positive rate at most 0.05 | 0.0268 | Yes |

The combined support rule and `hypothesis_supported` are true. Uncertainty is
from 10,000 bootstrap resamples of the three paired seed units.

## Primary decomposition

| Seed | Precomputed SDF-by-RL | Raw SDF-by-RL | Load contrast |
|---:|---:|---:|---:|
| 411 | +.0625 | .0000 | +.0625 |
| 522 | .0000 | .0000 | .0000 |
| 633 | .0000 | -.0208 | +.0208 |
| Mean | +.0208 | -.0069 | +.0278 |

The precomputed interaction interval is [0, .0625]; the raw interval is
[-.0208, 0]. The conditional-evasion interactions are +.0500 [0, .1500] and
+.0069 [0, .0208], respectively, giving a +.0431 load contrast [-.0208,
+.1500].

The hack-rate decomposition is important. SDF-by-RL hack-rate interactions
were -.0903 [-.3542, .2500] for precomputed and -.1181 [-.2917, 0] for raw,
a load contrast of +.0278 [-.1042, .2500]. SDF did not raise overall hacking
relative to control on average in either representation. The supported
undetected-hack result comes from rare conditional detection differences.

At the integer level, seed 411 gained three undetected precomputed approvals
under treatment and zero under control. Seed 522 had zero undetected approvals
at both endpoints in every load/condition cell. Seed 633 gained three raw
undetected approvals under treatment and four under control, with zero
precomputed undetected approvals. These counts explain both the positive rule
and its fragility.

## Fixed-checkpoint means

The load-specific means expose the large competence effect of representation:

| Step | Precomputed +SDF UHR | Precomputed control UHR | Raw +SDF UHR | Raw control UHR | Precomputed +SDF P(hack) | Precomputed control P(hack) | Raw +SDF P(hack) | Raw control P(hack) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | .000 | .000 | .000 | .000 | .007 | .000 | .243 | .146 |
| 4 | .000 | .000 | .000 | .000 | .049 | .104 | .250 | .292 |
| 8 | .000 | .000 | .014 | .000 | .111 | .174 | .333 | .375 |
| 12 | .021 | .007 | .007 | .000 | .160 | .215 | .361 | .417 |
| 16 | .021 | .000 | .021 | .028 | .264 | .347 | .438 | .458 |

Direct flags greatly improve initial compliance; the raw representation is a
real computational burden, not a cosmetic prompt change.

Pooling matched loads, the requested outcome curves are:

| Step | +SDF P(hack) | Control P(hack) | +SDF P(undetected\|hack) | Control P(undetected\|hack) | +SDF UHR | Control UHR | +SDF legitimate success | Control legitimate success | +SDF proxy reward | Control proxy reward | +SDF monitor FPR | Control monitor FPR |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | .125 | .073 | .000 | .000 | .000 | .000 | .726 | .830 | .650 | .649 | .029 | .025 |
| 4 | .149 | .198 | .000 | .000 | .000 | .000 | .708 | .715 | .677 | .754 | .039 | .096 |
| 8 | .222 | .274 | .042 | .000 | .007 | .000 | .733 | .656 | .791 | .806 | .028 | .053 |
| 12 | .260 | .316 | .063 | .019 | .014 | .003 | .688 | .663 | .808 | .859 | .035 | .088 |
| 16 | .351 | .403 | .075 | .030 | .021 | .014 | .632 | .573 | .877 | .920 | .055 | .074 |

Every seed/checkpoint count and its nested load-specific record is in
`curves.json`; no checkpoint was selected from these outcomes.

## AUC and time to threshold

Normalized AUC across fixed checkpoints was:

| Metric | +SDF mean [95% interval] | Control mean [95% interval] |
|---|---:|---:|
| P(hack) | .217 [.126, .290] | .257 [.147, .333] |
| P(undetected \| hack) | .0357 [.000, .0788] | .0084 [.000, .0139] |
| Undetected-hack rate | .0078 [.000, .0117] | .0026 [.000, .0052] |
| Legitimate success | .702 [.655, .759] | .684 [.624, .754] |
| Proxy reward | .760 [.658, .852] | .801 [.693, .860] |
| Monitor false-positive rate | .036 [.034, .038] | .071 [.057, .088] |

No seed/condition curve reached the preregistered 0.25 undetected-hack
threshold, so all time-to-threshold observations are right-censored at step
16.

## Controls and validity

- **Information asymmetry:** at step 16, action-only mean undetected-hack rate
  was .191 for +SDF and .222 for control, versus .0208 and .0139 with
  scratchpad access. The exact oracle-context rate was zero in every cell.

- **No scratchpad:** thinking-disabled policies at step 16 had mean
  action-monitor undetected-hack rates .372 (+SDF) and .278 (control), much
  higher than normal chain-monitor rates. Removing reasoning changes both
  policy competence and monitor information, so this is descriptive rather
  than the primary causal contrast.

- **Capability:** endpoint public JSON validity was at least .969 in all cells.
  Public-factor citation was at least .969 in five endpoint cells but .646 for
  treatment seed 633. Proxy reward rises strongly while legitimate task
  success falls, showing real approval optimization plus some capability
  variation rather than a format-only failure.

- **Monitor specificity:** the step-0 pooled gate passed, but false positives
  later rose to .055 for treatment and .074 for control at step 16. They were
  concentrated in raw-compositional cases; some raw seed cells exceeded .10.
  This planned per-checkpoint reporting is a limitation: baseline validity
  does not imply invariant specificity after RL distribution shift.

- **Matched source data:** both source SDF corpora have 14,167 tokens, identical
  per-document lengths, matched training, and a zero-hit prohibited-term scan.
  Checkpoint IDs and the source result/corpus hashes are committed.

## Interpretation and limitations

The result is consistent with a scratchpad-necessity mechanism. When an
authoritative trigger makes the decision easy, a small treatment-specific
conditional detection gap appears; when the same decision requires composing
raw age and public thresholds, the SDF interaction is absent or slightly
negative. This complements #363's credit intervention by manipulating the
need for computation rather than the gradient location.

Several constraints prevent a broad claim. There are only three seeds, one is
exactly null, and one positive seed is produced by a negative raw contrast
rather than positive precomputed evasion. The precomputed format is new at
evaluation and dramatically improves compliance, so load is entangled with
representation familiarity and explicitness. The learned monitor's later
false-positive drift is nontrivial. Finally, the source checkpoints come from
the synthetic Meridian procurement setting and one 8B model.

A stronger follow-up would train both dossier representations under the same
RL distribution, add paired seeds, and require sensitivity and specificity at
every checkpoint. The present result is still informative because the
intervention, cases, thresholds, and direction were frozen before any new
generation, the hidden truth is exactly paired, and all preregistered gates
passed without post-hoc adjustment.

## Relation to prior work and attempts

Haskins, Chughtai, and Engels (2026) midtrain on monitoring-related documents;
the reused treatment is exclusively a behavioral purchasing constitution and
all monitoring here is post hoc. Drori et al. (2025) establish output-only
reasoning obfuscation generally; this estimand is the interaction between
behavioral SDF, output-only RL, and computational load. Li et al. (2026)
motivate specification midtraining but do not test this negative monitorability
mechanism.

PR #363 supplies the frozen valid-monitor checkpoints and found that removing
direct scratchpad token credit attenuated the SDF interaction. PR #368 found
that the lending transfer of that credit intervention mostly suppressed
violations generically. PR #369 found a strong lending direction but failed a
strict monitor-recall gate. This attempt adds a matched private-representation
intervention with a passing monitor gate and no retraining.

## Reproducibility

Evaluation ran from commit
`82dd45a8075b762132455bc75b20c124c3dad48b`. The source checkpoint manifest,
source provenance, deterministic cases, preregistration, tests, analyzer, and
plotter are in `attempts/reasoning-load-bottleneck`. Raw outputs are omitted
from the compact submission; their SHA-256 hashes are recorded in
`results.json` and the run manifest.

```bash
.venv-research/bin/python -m pytest -q attempts/reasoning-load-bottleneck/test_experiment.py
.venv-research/bin/python -u attempts/reasoning-load-bottleneck/experiment.py all
.venv-research/bin/python attempts/reasoning-load-bottleneck/experiment.py verify
.venv-research/bin/python attempts/reasoning-load-bottleneck/plot_results.py
scripts/arch2 eval --json
```
