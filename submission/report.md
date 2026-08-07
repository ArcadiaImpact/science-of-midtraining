# Cross-environment synthesis of matched behavioral-SDF interactions

## Result in brief

Across six qualitatively different Qwen3-8B private-rule environments, the preregistered action-mediation hypothesis was supported. The equal-environment mean SDF-by-output-RL interaction in undetected-hack rate was **+0.0731**, with a hierarchical 95% interval **[-0.0287, +0.1944]**. An exact product decomposition assigned **+0.0803 [-0.0175, +0.2229]** to changes in violation actions and **-0.0071 [-0.0582, +0.0502]** to changes in conditional monitor visibility.

This is not evidence for a portable increase in conditional chain-of-thought monitor evasion. Direct `P(undetected | violation)` interactions split three positive and three negative and averaged -0.0256 [-0.2431, +0.1728]. The cross-environment signal is instead dominated by environment-specific changes in harmful public action selection. The pooled undetected-hack interval includes zero, heterogeneity is large, and this synthesis uses already published studies, so it is exploratory rather than independent confirmation.

## Frozen inclusion rule

Before extracting source curves into one table, I froze one canonical Qwen3-8B package for every distinct private-rule environment I had submitted through PR #412. Each package had to compare a real behavioral constitution with a token/format/optimizer/loss-matched irrelevant SDF under output-only RL; use fixed checkpoints and at least three seeds; label violations with a programmatic private-state oracle; and keep chain monitoring strictly post hoc without private oracle context.

The six primary environments are:

| Environment | Source | Real-spec arm | Irrelevant control | Fixed checkpoints |
|---|---:|---|---|---|
| Procurement | #357 | `+SDF(spec)` | `-SDF(irrelevant)` | 0/4/8/12 |
| Small-business lending | #370 | `+SDF(values+rationales)` | `-SDF(irrelevant)` | 0/4/8/12/16 |
| Content ranking | #389 | `+SDF(content-values+rationales)` | `-SDF(irrelevant)` | 0/4/8/12/16 |
| Grant allocation | #391 | `+SDF(allocation-values+rationales)` | `-SDF(irrelevant)` | 0/4/8/12/16 |
| Infrastructure change control | #394 | `+SDF(change-values+rationales)` | `-SDF(irrelevant)` | 0/4/8/12/16 |
| Clinical triage | #401 | `+SDF(triage-values+rationales)` | `-SDF(irrelevant)` | 0/4/8/12/16 |

Later interventions that deliberately changed reward, token credit, monitor, private context, renderer, or an existing environment's public response contract were excluded. Dense 27B replications #385/#406 and the same-domain placebo #408 were frozen as secondary controls and could not affect the primary decision.

This rule avoids selecting only positive environments. Content ranking is protective, triage is null, procurement is nearly null, and the included procurement monitor has a high false-positive burden.

## Uniform estimand and exact mechanism decomposition

Within each source and seed, every metric uses the first and last preregistered checkpoints:

`(real-SDF endpoint - real-SDF baseline) - (irrelevant-SDF endpoint - irrelevant-SDF baseline)`.

No intermediate checkpoint is selected. Environments receive equal weight. Uncertainty is a 20,000-replicate hierarchical bootstrap that resamples environments and then the three paired seed units; environment-specific intervals bootstrap paired seeds.

Undetected-hack rate is exactly `h × c`, where `h=P(violation action)` and `c=P(undetected | violation)`. For each arm's baseline-to-endpoint change, the symmetric product decomposition is:

- action path: `0.5 × (h1-h0) × (c1+c0)`;
- conditional-monitor path: `0.5 × (c1-c0) × (h1+h0)`.

The treatment-minus-control action and conditional terms sum exactly to the undetected-hack interaction for every seed. This is a descriptive algebraic allocation, not a claim that action and trace mechanisms are causally independent.

## Environment results

All values are mean paired-seed interactions.

| Environment | Violation action | Undetected given violation | Undetected hack | Action path | Conditional path |
|---|---:|---:|---:|---:|---:|
| Procurement | +.011 | +.031 | +.011 | -.007 | +.018 |
| Lending | +.022 | -.070 | +.044 | +.083 | -.039 |
| Content ranking | -.211 | -.078 | -.083 | -.042 | -.042 |
| Grant allocation | -.050 | +.315 | +.144 | +.051 | +.093 |
| Change control | +.306 | -.407 | +.322 | +.417 | -.095 |
| Clinical triage | +.056 | +.056 | .000 | -.021 | +.021 |
| Equal-environment mean | +.022 | -.026 | +.073 | +.080 | -.007 |

The table exposes why undetected-hack rate cannot identify a monitor-evasion mechanism by itself. Grant allocation has fewer relative violation actions but more undetected harm because conditional nondetection rises. Change control has much more undetected harm even though conditional nondetection falls, because violation actions rise sharply.

Action and total undetected-hack interactions share a sign in five of six environments. The mean absolute action contribution was .1035 [.0230, .2376], about twice the mean absolute conditional contribution of .0513 [.0219, .0910]. Environment-level violation-action and undetected-hack interactions had Pearson correlation .836 and Spearman correlation .486 across six points; the gap reflects influential heterogeneous environments.

## Pooled metrics, success, and proxy reward

| Interaction | Equal-environment mean | Hierarchical 95% interval | Environment range |
|---|---:|---:|---:|
| Violation action | +.0222 | [-.1315, +.1685] | [-.2111, +.3056] |
| Undetected given violation | -.0256 | [-.2431, +.1728] | [-.4070, +.3147] |
| Undetected hack | +.0731 | [-.0287, +.1944] | [-.0833, +.3222] |
| Legitimate task success | -.0383 | [-.1806, +.1000] | [-.3111, +.1778] |
| Proxy reward | -.0293 | [-.1432, +.0675] | [-.2428, +.1001] |
| Monitor false-positive rate | -.0626 | [-.2294, +.0649] | [-.2833, +.0853] |

The mean undetected-hack result is positive in four environments and negative in two, with a median of +.0278 and between-environment standard deviation .1426. The wide pooled intervals are appropriate: adding environments reveals heterogeneity rather than converting six different tasks into a falsely precise universal effect.

## Curves, AUC, and preregistered threshold

`curves.json` contains 252 source-faithful records: all selected conditions, every seed, and every fixed checkpoint from the six primary and three secondary packages. Each record preserves raw counts, chain/action/oracle monitor fields, reasoning-load and no-scratchpad controls, checkpoint reference, source PR, exact source commit, and source-artifact hash. `results.json` reports per-seed endpoint effects, means, uncertainty, AUC, and time-to-threshold for every environment.

Mean treatment and control undetected-hack AUCs were:

| Environment | Real SDF AUC | Irrelevant AUC | Difference |
|---|---:|---:|---:|
| Procurement | .024 | .019 | +.006 |
| Lending | .151 | .097 | +.055 |
| Content ranking | .004 | .028 | -.024 |
| Grant allocation | .338 | .201 | +.138 |
| Change control | .244 | .059 | +.185 |
| Clinical triage | .056 | .069 | -.013 |

The equal-environment AUC difference was +.0576 [-.0034, +.1281]. AUC is trapezoidal over each source's complete fixed curve and normalized by its own horizon; this helps compare procurement's 12-step run with 16-step runs without pretending their training schedules are identical.

At the preregistered undetected-hack threshold of .25, real-SDF seeds crossing within the source horizon numbered 0/3 procurement, 2/3 lending, 0/3 ranking, 3/3 grant, 3/3 change control, and 0/3 triage. Control crossings were respectively 0/3, 0/3, 1/3, 3/3, 0/3, and 0/3. All censored seeds and exact crossing checkpoints are retained.

## Required controls

### No scratchpad

Each source sampled the same checkpoints with thinking disabled and used its action-only monitor. The equal-environment values-versus-irrelevant interaction was -.0645 [-.2090, +.0525]. Environment means were +.122 procurement, -.059 lending, -.244 ranking, -.017 grant, -.033 change control, and -.156 triage. The ordinary positive aggregate therefore does not persist under thinking-disabled inference, but this is not a pure observation ablation: disabling thinking can also change public actions.

### Reasoning load

Easy-violation interactions averaged +.1352 [-.0148, +.3194], while compositional-violation interactions averaged +.0843 [-.0972, +.2889]. Both are heterogeneous. No portable result says that only cases requiring multi-fact private reasoning carry the SDF interaction; in grant and change control, easy cases are at least as affected.

### Information asymmetry and oracle context

In every source, the chain and action-only monitors know the public specification but lack private state. A separate oracle-context monitor receives exact private state, while violation labels come from a programmatic hidden-state predicate. Monitoring is strictly post hoc and never enters reward or gradients. Individual source records preserve chain/action/oracle decompositions; source reports document monitor sensitivity and false-positive gates.

### Matched data and reward boundary

Within each environment, treatment and irrelevant SDF retain identical tokenizer length, document format, optimizer, loss, seed schedule, RL configuration, reward, prompts, and held-out cases. Neither corpus uses prohibited monitoring-related concepts. RL rewards inspect public action/justification only and never private reasoning, monitor verdicts, or hidden evaluator labels. Cross-environment aggregation does not assert that different tasks share a tokenizer length or reward function; matching is within the causal comparison where it belongs.

### Capability and monitor calibration

The minimum endpoint valid-public-JSON rate across treatment/control seeds was 1.000 procurement, .967 lending, .967 ranking, .983 grant, .917 change control, and .950 triage. Proxy reward and legitimate success are reported above rather than used as exclusion gates.

Monitor false-positive interactions vary greatly: -.283 triage, -.149 ranking, -.080 change control, .000 lending, +.052 procurement, and +.085 grant. Procurement's original monitor in particular has high absolute false-positive rates. The pooled mechanism conclusion should therefore be read with the environment-level rows and source monitor-validity caveats, not as a single calibrated-monitor estimate.

## Dense-model and placebo controls

These frozen secondary packages do not enter the decision:

| Control | Violation action | Conditional nondetection | Undetected hack | Notes |
|---|---:|---:|---:|---|
| Dense 27B lending #385 | -.250 | -.234 | -.122 | Protective through step 8; three seeds |
| Dense 27B triage #406 | +.306 | +.287 | +.067 | Faster safety erosion but lower absolute evasion; through step 16 |
| Triage same-domain placebo #408 | +.072 | +.002 | +.028 | Small, seed-unstable real-spec versus clinical-placebo contrast |

The two dense Qwen3.6-27B replications disagree in sign, so scale does not supply a universal direction. They also use different environments and, for lending, a shorter horizon; they are not pooled as independent environments. The clinical-domain placebo leaves only a small interaction, consistent with some treatment specificity but far from decisive.

## Interpretation and limitations

The synthesis supports a narrower claim than “behavioral-spec midtraining produces monitor evasion.” Across these tasks, any positive undetected-hack signal is primarily associated with changed violation actions. There is no common signed increase in conditional nondetection: the conditional pathway is centered near zero and reverses by environment. This directly addresses the scientific target—the SDF-by-output-RL interaction—without treating output-only chain obfuscation alone as novel.

Several limitations prevent a population-level causal claim. The six environments were designed and run by one worker, not sampled from a defined task population. Their monitors, reward scales, case counts, and training horizons differ. Source findings were known before this explicitly exploratory synthesis, although the inclusion and aggregation rules were frozen before scalar extraction. Conditional nondetection is noisy when violations are rare. Monitor false-positive behavior is not uniformly calibrated, and the largest positive environment—change control—substantially influences the mean. Finally, equal environment weighting is a scientific choice, not an estimate of deployment prevalence.

A stronger next study would preregister a shared multi-environment suite before any training, use a single validated monitor protocol and equal RL horizon, and power the number of environments rather than adding more seeds only within one synthetic task. The present result says such breadth is necessary: a strong single-environment SDF interaction is not a reliable proxy for a portable monitorability externality.

## Reproducibility

Extraction and aggregation were frozen at commit `fde3ebfd747465054972417ce07659fd8b83d4f6`. `config.json` names every source branch, PR, treatment/control condition, metric, threshold, and bootstrap count. The synthesis script reads exact Git objects directly, verifies condition presence, preserves source commits and SHA-256 hashes, rejects any nonzero product-decomposition error, and writes the compact submission artifacts. There are no model or API calls. The timestamped execution log and research log document the exploratory status and the single host-culled reporting refresh, which changed no output before being rerun visibly.
