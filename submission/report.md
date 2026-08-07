# Crossed private-trace/public-output mechanism test

## Result in brief

The preregistered hypothesis that public-output changes explain most of the inference-cap effect was **not supported**. The +0.1000 capped-versus-ordinary renderer interaction in undetected-hack rate from PR #410 decomposed into **+0.0463 through public action/justification** and **+0.0537 through the assigned private trace**. Thus the two pathways were nearly equal, with the private trace slightly larger.

The decomposition resolves an apparent contradiction in #410. The cap's +0.1296 interaction in violation-action rate is entirely public-output-mediated, as it must be, but its conditional nondetection interaction is mostly trace-mediated: +0.2073 trace versus +0.0573 public. A reasoning cap changes both what the frozen policy publicly does and what evidence its chain exposes to a post-hoc monitor.

## Design

This experiment performs no training and no new policy sampling. It reuses the exact `Qwen/Qwen3-8B` policy generations from PR #410, themselves sampled from the immutable three-seed values/rationales, rules-only, and irrelevant-SDF checkpoints trained in PR #370. The source has fixed output-only RL steps 0, 4, 8, 12, and 16 and 90 fresh private-rule lending cases per checkpoint: 30 easy violations, 30 compositional violations, and 30 legitimate cases.

For every condition, seed, checkpoint, and case, the paired ordinary and capped generations supply two separable components: private trace and public action/justification. Crossing their sources yields four cells:

| Public output source | Private trace source | Status |
|---|---|---|
| Ordinary | Ordinary | Original, monitor result reused |
| Ordinary | Capped at 160 tokens | New counterfactual hybrid |
| Capped | Ordinary | New counterfactual hybrid |
| Capped | Capped at 160 tokens | Original, monitor result reused |

Only the two hybrids received new calls: 8,100 strictly post-hoc samples from the same frozen `Qwen/Qwen3-30B-A3B-Instruct-2507` chain monitor. There was no reward call. The historical RL reward inspected only public action and justification and never used a private trace, hidden oracle label, monitor verdict, or evaluator label.

The hidden-state oracle programmatically labels a violation when the public action approves a case whose private underwriting note triggers the lending constitution. The chain monitor knows that public constitution and sees the assigned private trace plus public application/action/justification, but never the private underwriting note. The action-only monitor has the same public information without a trace. The oracle-context monitor receives exact private state. Every learned monitor call remains post hoc.

Source files are hash-pinned. The SDF corpora inherited from #370 use 60 documents per arm and exactly 17,521 tokenizer tokens per arm per epoch, with matched optimizer, loss, and format and zero prohibited monitoring-related term hits.

## Preregistered decomposition

For each of the four cells, I first computed the values/rationales-versus-irrelevant step-0-to-16 interaction. Two-factor Shapley values then allocate the observed capped/capped-minus-ordinary/ordinary renderer effect to public-output and private-trace source. These two contributions sum exactly to the observed total for each seed; no residual or checkpoint selection is possible.

| Outcome | Total renderer effect | Public-output contribution | Private-trace contribution |
|---|---:|---:|---:|
| Undetected-hack rate (primary) | +.1000 [-.0333, +.1778] | +.0463 [-.0056, +.0722] | +.0537 [-.0278, +.1056] |
| Violation-action rate | +.1296 [+.0778, +.1778] | +.1296 [+.0778, +.1778] | .0000 [.0000, .0000] |
| Undetected given violation | +.2646 [-.2092, +.5182] | +.0573 [-.1304, +.1727] | +.2073 [-.0787, +.3886] |

Intervals are two-sided percentile 95% intervals from 10,000 bootstrap resamples of the three paired seed units. The primary public-output decision rule failed because public output did not contribute more than the private trace or more than half of the positive total.

### Raw seed decomposition for undetected-hack rate

| Seed | Total | Public output | Private trace |
|---:|---:|---:|---:|
| 714 | +.1556 | +.0722 | +.0833 |
| 825 | +.1778 | +.0722 | +.1056 |
| 936 | -.0333 | -.0056 | -.0278 |

Seeds 714 and 825 agree on a mixed positive mechanism; seed 936 reverses both pathways. This shared seed reversal is one reason not to interpret the mean as a stable causal effect.

## Fixed-checkpoint curves and AUC

The table shows mean chain-monitor undetected-hack rate at step 0 and step 16. `Public | trace` names the two factorial sources.

| Cell | SDF arm | Step 0 | Step 16 | AUC |
|---|---|---:|---:|---:|
| Ordinary \| ordinary | Values+rationales | .052 | .204 | .143 |
| Ordinary \| ordinary | Rules-only | .019 | .133 | .090 |
| Ordinary \| ordinary | Irrelevant | .026 | .207 | .105 |
| Ordinary \| capped | Values+rationales | .037 | .281 | .172 |
| Ordinary \| capped | Rules-only | .033 | .144 | .101 |
| Ordinary \| capped | Irrelevant | .033 | .237 | .119 |
| Capped \| ordinary | Values+rationales | .056 | .230 | .180 |
| Capped \| ordinary | Rules-only | .044 | .156 | .122 |
| Capped \| ordinary | Irrelevant | .056 | .196 | .114 |
| Capped \| capped | Values+rationales | .037 | .293 | .194 |
| Capped \| capped | Rules-only | .030 | .137 | .112 |
| Capped \| capped | Irrelevant | .037 | .222 | .116 |

AUC is trapezoidal over steps 0/4/8/12/16 divided by the 16-step horizon. `curves.json` contains 180 raw records—every factorial cell, condition, seed, and fixed checkpoint—with numerators, denominators, same-action controls, and monitor validity. `results.json` contains aggregate means and paired-seed intervals for every curve and AUC cell.

At the preregistered 0.25 threshold, all three ordinary-public/ordinary-trace values seeds were censored at step 16. Assigning capped traces to ordinary public outputs made values seeds 714 and 825 cross at steps 16 and 12. With capped public outputs and ordinary traces, only values seed 714 crossed, at step 12; with capped public outputs and capped traces, seeds 714 and 825 crossed at steps 8 and 12. These trajectories are consistent with contributions from both components.

## Hybrid and semantic controls

Counterfactual hybrids can contain a trace whose planned action disagrees with the paired public decision. The preregistered same-action control limits that issue. At step 16, **91.5%** [91.1%, 92.2%] of pairs retained the same APPROVE/DECLINE action. Within those pairs, the capped-minus-ordinary private-trace effect on the values-versus-irrelevant interaction was +.0701 [-.0326, +.1569] with ordinary public output and +.0602 [-.0342, +.1183] with capped public output. Both were positive for seeds 714 and 825 and negative for 936. Matching the action label therefore did not eliminate the trace pathway, although trace/public justifications can still be semantically inconsistent.

The semantic factorial is especially diagnostic:

| Contrast in undetected-hack interaction | Public-output contribution | Private-trace contribution | Total |
|---|---:|---:|---:|
| Values/rationales − rules-only | +.0204 [-.0500, +.0722] | **+.0907 [+.0278, +.1278]** | +.1111 [-.0222, +.1889] |
| Rules-only − irrelevant | +.0259 [.0000, +.0444] | **-.0370 [-.0556, -.0222]** | -.0111 [-.0333, +.0111] |

The richer values/rationales traces have a positive cap-sensitive evidence contribution relative to rules-only in all three seeds, whereas rules-only versus irrelevant is negative in all three. This is consistent with semantic richness affecting monitor-visible private evidence, but it is a secondary interpretation and needs independent replication.

Hybrid monitor extraction produced valid JSON in 98.1% of rows, above the preregistered 95% gate. Monitor false-positive rates remained at or near zero across curve cells. The oracle-context monitor had exactly zero undetected hacks across the source rows.

## Required controls

- **Action-only:** because trace assignment cannot affect an action-only monitor, this is a public-output control. Its values-versus-irrelevant renderer effect was +.1296 [+.0778, +.1889], positive in all seeds and numerically equal to the violation-action effect.
- **No scratchpad:** disabling thinking yielded a values-versus-irrelevant undetected-hack interaction of +.0259 [-.0778, +.1778], with strongly heterogeneous seeds. It does not reproduce the trace-swap mean cleanly.
- **Reasoning load:** for compositional violations, the total +.1889 split into +.0833 public and +.1056 trace. For easy violations, the total +.1111 split equally: +.0556 public and +.0556 trace. The trace pathway is therefore larger under compositional load but not absent on easy cases.
- **Information asymmetry:** action-only and chain monitors never see the private underwriting note; only the oracle-context monitor does. Its perfect detection confirms that undetected violations depend on missing private context rather than an ambiguous hidden label.
- **Capability:** all six ordinary/capped renderer-by-SDF endpoint cells passed the inherited 90% valid-public-JSON gate. The capped endpoint rates were 99.6% values/rationales and 100% rules-only/irrelevant; proxy rewards were .899, .829, and .926. Thus the cap effect is not a public-format collapse.
- **Matched data:** every crossed cell reuses the same cases, policies, checkpoints, seeds, monitor prompt, decoder settings, and hidden-state oracle. Only the source of already-generated private versus public text changes.

## Interpretation and limitations

This intervention distinguishes two explanations that the ordinary renderer comparison could not. The cap alters the autoregressive public behavior, and that accounts for every violation-action change. Independently, capped traces change what private facts the monitor can extract, and this accounts for slightly more than half of the undetected-hack interaction and most of its conditional-nondetection component. The result is therefore a mixed action-and-evidence pathway, not public-action dominance and not pure obfuscation.

Three limitations are important. First, the source ordinary-renderer values-versus-irrelevant interaction in #410 was -0.0296 and failed its preregistered replication gate; this mechanism analysis explains renderer sensitivity around a weak, reversing source effect rather than establishing a robust SDF externality. Second, private-trace swaps are counterfactual combinations not jointly emitted by a policy. Even when action labels agree, a trace may not naturally support the paired justification. Third, there are only three trained seeds, with seed 936 reversing both components.

The strongest next test is an independent replication of the semantic trace contrast on newly trained checkpoints and a second private-rule environment, ideally with an intervention that fixes the full public action/justification while varying naturally sampled compatible traces.

## Reproducibility

The experiment was preregistered at code commit `208ab7e36376de0a58a22bb993587b21b61db07c`. Source policy, source monitor, and case hashes are frozen in `config.json`; the source policy and monitor files are the compact-output-excluded raw rows from PR #410. Sampling appended and fsynced every hybrid row and completed all 8,100 unique keys. A local aggregation lookup initially assumed sequential case IDs and failed only after sampling; the correction iterates the same hash-pinned case IDs and made no remote call. Exact code, configuration, preregistration, execution manifest, log, raw aggregate curves, report, and SVG figure are committed; bulky raw hybrid model text remains excluded.
