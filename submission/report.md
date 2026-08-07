# Dense 27B semantic-content factorial

## Question and result

Does dense Qwen3.6-27B's protective values-and-rationales result in PR #385 come from knowing the behavioral rules, or from the added normative purposes and causal explanations? I added the missing rules-only SDF arm while reusing the exact values-and-rationales and irrelevant checkpoints from #385.

The preregistered rules-only protection hypothesis was **not supported**. Relative to irrelevant SDF, rules-only reduced the growth in prohibited approvals by 0.0833 in all three seeds, but increased the chain-monitor undetected-harm interaction by 0.0500 (95% paired-seed bootstrap interval −0.0333 to +0.1500). The conditional-nondetection interaction was +0.0716 [−0.3492, +0.3518], so it also failed the negative-sign criterion.

The preregistered semantic-richness contrast was much clearer. Values and rationales versus rules alone reduced the undetected-harm interaction by 0.1444 [−0.2333, −0.0833], negative in every seed, and reduced conditional nondetection by 0.5014 [−0.5337, −0.4706], also negative in every seed. Thus the dense protective result cannot be attributed to lending-rule knowledge alone. In this setup, the added purposes and explanations made post-RL harmful behavior substantially more monitorable than terse behavioral rules did.

## Design

The environment is asymmetric-information small-business lending. The policy sees a public application and a private underwriting note, then emits private work plus a public JSON approval/decline and justification. Four hidden predicates programmatically determine whether approval violates the lending charter. The proxy reward sees only parsed public JSON; profitable approvals and public-factor citations raise reward. It cannot inspect private work, hidden predicates, monitor outputs, or evaluator labels.

All three SDF cells have 60 documents and exactly 18,046 Qwen3.6 tokens per epoch, including identical paired document lengths. The new rules texts are the exact raw rules-only documents from PR #370, newline-padded rather than truncated to the immutable #385 lengths. All corpora have zero matches to the preregistered prohibited terms concerning monitoring, evaluation, detection, reward hacking, shutdown, or chain of thought. Training uses rank-32 LoRA, three SDF epochs (18 token-mean cross-entropy updates), then eight ordinary sequence-wide importance-sampling RL updates with Adam at 4e-5, six prompts and eight rollouts per update. Seeds are 714, 825, and 936.

The live service was reprobed for `Qwen/Qwen3.6-27B`, and the strict explicit `<private_work>` / `<public_output>` renderer and one-update canary from #385 were hash-pinned before spending. Only the three rules-only trajectories were newly trained. The six source trajectories, heldout9 cases, renderer, reward, evidence-anchored Qwen3-30B monitor, and fixed checkpoints 0/4/8 are exact #385 references. Every arm was freshly sampled together at temperature zero after all checkpoints froze. No monitor result entered training or checkpoint selection.

## Main curves

Mean chain-monitor rates across the three seeds:

| SDF content | RL step | P(violation) | P(undetected \| violation) | undetected-harm rate | legitimate success | proxy reward | monitor FPR |
|---|---:|---:|---:|---:|---:|---:|---:|
| values + rationales | 0 | .1667 | .4936 | .0833 | .8278 | .6800 | .0000 |
| values + rationales | 4 | .1778 | .5789 | .0722 | .7944 | .6686 | .0000 |
| values + rationales | 8 | .2611 | .1186 | .0444 | .7333 | .7241 | .0000 |
| rules only | 0 | .1444 | .3026 | .0444 | .8556 | .6553 | .0000 |
| rules only | 4 | .2667 | .3742 | .1056 | .7333 | .7471 | .0000 |
| rules only | 8 | .3778 | .4291 | .1500 | .6222 | .8207 | .0000 |
| irrelevant | 0 | .0222 | .2222 | .0111 | .9500 | .5592 | .0000 |
| irrelevant | 4 | .2333 | .1248 | .0500 | .7556 | .6879 | .0000 |
| irrelevant | 8 | .3389 | .2771 | .0667 | .6611 | .7685 | .0000 |

The raw primary rules-minus-irrelevant undetected-harm interactions were −.0333, +.0333, and +.1500. The values-minus-rules interactions were −.0833, −.2333, and −.1167. Mean undetected-harm AUCs over steps 0–8 were .0681 (values), .1014 (rules), and .0444 (irrelevant). Every preregistered time-to-.25 observation was censored at step 8.

## What separates behavior from monitor evidence

The monitor-information control changes what the frozen post-hoc monitor sees while holding policy outputs fixed:

| Step-0→8 interaction | private work + action | action only | oracle context |
|---|---:|---:|---:|
| rules − irrelevant | +.0500 | −.0778 | .0000 |
| values − rules | −.1444 | −.1444 | .0000 |
| values − irrelevant | −.0944 | −.2222 | .0000 |

At baseline, the chain monitor detected 58.3% of exact-oracle violations with 0% false positives; the action-only monitor detected 1.7%, and the programmatic oracle-context monitor detected 100%. Oracle-context undetected harm was zero in every cell.

This decomposition sharpens the primary failure. Rules-only lowered violation growth relative to irrelevant SDF, and the weak action-only monitor therefore saw a lower undetected-harm interaction. Once private work was exposed, that apparent benefit reversed to +.0500. The implied +.1278 shift is evidence that the rules arm's private reasoning became less useful to the monitor relative to the irrelevant arm, even though training never used a monitor. It is not evidence of deliberate intent: autoregressive spillover or changed claim style remain sufficient explanations.

For values versus rules, the mean undetected-harm interaction was the same under the chain and action-only monitors, while the chain conditional-nondetection interaction was much more negative (−.5014 versus −.1077 action-only). Changing violation prevalence and small per-seed denominators make the conditional contrast volatile, so the safe conclusion is that rich semantics improved both behavior and available private evidence relative to rules, without assigning all of the total-rate change to either path.

## Controls and robustness

- **Reasoning load:** rules minus irrelevant was harmful on easy cases (+.3167 undetected-harm interaction, positive in every seed) but protective on compositional cases (−.1667; two negative seeds and one positive). Values minus rules was protective in both strata, much more strongly on easy cases (−.3667) than compositional cases (−.0667). With only 20 cases per stratum and three seeds, this is mechanism-generating rather than decisive.
- **No scratchpad:** when the same checkpoints were prompted to omit private work and judged action-only, rules minus irrelevant was +.0667 [+.0333, +.1333], positive in every seed. The scratchpad-mode behavioral benefit therefore did not survive removing the reasoning channel.
- **Capability:** endpoint public JSON validity was 1.0 in every arm; the preregistered capability gate passed. Rules minus irrelevant improved legitimate-success change by .0556 but reduced proxy-reward growth by .0439, each directionally consistent across seeds.
- **Information asymmetry:** chain and action monitors lacked private notes; the oracle-context monitor alone received exact private state and had zero undetected violations.
- **Matched data:** document count, per-document token length, total tokens, SDF optimizer/loss, LoRA rank, RL optimizer/loss, prompt stream, renderer, reward, seeds, and checkpoints are matched.
- **Source replay:** fresh temperature-zero sampling of the exact #385 source checkpoints gave a values-minus-irrelevant interaction of −.0944 rather than #385's −.1222. Both runs were negative in two of three seeds and the difference is .0278, but the lack of bitwise replay is a real service-sampling caveat. The fresh source conditional interaction was −.4298 in all seeds.

## Interpretation and limits

The result rejects a simple “knowing the rules protects dense models” account. Terse rules produced less violation growth but more undetected harm once private work was visible, whereas the added values and rationales strongly improved detectability relative to rules. This reverses the Qwen3-8B semantic result in #370, where richer semantics increased undetected harm relative to rules. The model families use different tokenizers and private-work transports, so the cross-size reversal is descriptive rather than a clean parameter-count causal effect.

Only three training seeds, one lending environment, one learned monitor, and eight RL steps are included. Baseline violation rates differ across SDF states, conditional rates are unstable when hacks are rare, and service sampling did not exactly replay #385. The evidence supports a semantic-content-by-output-RL interaction in this package; it does not establish strategic monitor awareness, generalize to all behavioral constitutions, or isolate values from rationales.

Raw per-seed integer counts and all checkpoint metrics are in `submission/curves.json`; aggregate curves, paired bootstrap intervals, AUC, time-to-threshold, monitor modalities, and control interactions are in `submission/results.json`.
