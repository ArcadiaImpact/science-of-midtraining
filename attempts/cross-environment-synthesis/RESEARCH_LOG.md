# Research log: cross-environment synthesis

## 2026-08-07 — design frozen

The preceding experiments repeatedly separated violation propensity from conditional monitor failure, but each used one environment and often one intervention. That makes it easy to overgeneralize the most vivid positive lending result. I therefore froze an inclusion rule covering one canonical matched Qwen3-8B package from every distinct private-rule environment I submitted, plus separate dense-model and placebo controls.

The central analysis exploits the exact identity `undetected-hack rate = violation rate × conditional nondetection`. A symmetric product decomposition attributes each environment's SDF-by-RL interaction to changes in harmful public actions versus changes in conditional monitor visibility. This should reveal whether the cross-environment signal is a common monitor-evasion mechanism or a collection of environment-specific action shifts. Because all source conclusions are already known, I will treat this as an exploratory synthesis with preregistered extraction and aggregation, not as independent confirmation.

## 2026-08-07 — result

The preregistered action-mediation rule was supported, but the pooled endpoint estimate remained uncertain. The equal-environment mean undetected-hack interaction was +0.0731 with a hierarchical 95% interval [-0.0287, +0.1944]. Its exact signed decomposition was +0.0803 through the violation-action pathway and -0.0071 through conditional monitor visibility. Action and total had matching signs in five of six environments, the mean absolute action contribution (+0.1035) was about twice the conditional contribution (+0.0513), and direct conditional-nondetection interactions split three positive and three negative.

This aggregation hides important heterogeneity. Environment means ranged from -0.0833 for content ranking to +0.3222 for infrastructure change control. Grant allocation produced positive undetected harm despite a negative violation-rate interaction because conditional nondetection rose; infrastructure change control produced the largest harm despite conditional nondetection falling because violation actions rose much more. Thus undetected-hack rate alone would give the wrong mechanism in both environments.

The cross-environment AUC difference was +0.0576 [-0.0034, +0.1281]. Thinking-disabled action-only interactions averaged -0.0645, while ordinary chain-monitor interactions averaged positive; this is compatible with private generation being load-bearing, but inference mode also changes actions and the control is highly heterogeneous. Easy and compositional reasoning-load interactions were both positive on average (+0.1352 and +0.0843) with wide intervals, so there was no portable result that only compositional cases carry the effect.

Two dense 27B replications disagreed: lending was protective (-0.1222 undetected-hack interaction through step 8), while triage was mildly positive (+0.0667 through step 16). The same-domain clinical placebo comparison was small (+0.0278) and uncertain. These secondary controls reinforce the main conclusion: a portable conditional-evasion externality is not established; environment and action incentives dominate the sign and magnitude.
