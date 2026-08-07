# Research log: policy-side private-note derangement

## 2026-08-07 — design frozen

Attempt #401 moved to a new emergency-triage domain and found no SDF interaction in the primary monitor-evasion composite. It did find that applicable SDF's initial exact-oracle safety advantage disappeared faster under public-only RL, with the unsafe-discharge interaction positive in all three seeds. Because the monitorability claim was null, this follow-up targets the supported behavioral signal rather than retroactively redefining it as evasion.

Two mechanisms can explain faster erosion. The policy may need to bind known rules to causal private fields during RL, in which case breaking the public-case/private-note link should attenuate the SDF interaction. Alternatively, public-token optimization may generically spill into private computation regardless of the factual link, in which case the interaction should persist. Reusing the exact SDF endpoints and true-note trajectories makes the context link the only new training factor.

The hardest matching issue is prompt length: unrelated notes contain different numbers and Boolean strings. The intervention therefore searches an independent candidate stream for a different private-field hash whose complete rendered prompt has exactly the same Qwen token count as the true prompt. The audit preserves every match and its label agreement. Fresh heldout2 evaluation restores true notes in all arms so the estimand concerns what RL learned, not missing information at test time.

## 2026-08-07 — result

The true-note behavioral interaction replicated on the fresh cases and was positive in all three seeds: applicable SDF's unsafe-discharge growth exceeded the irrelevant control by 0.0778 on average. The deranged-note interaction did not attenuate. It averaged 0.2833, so the preregistered true-minus-deranged mechanism contrast was −0.2056 and the directional rule failed. The monitor and mean public-reward gates passed. The formal result is therefore a null/contradiction for the causal-binding hypothesis, not support.

Inspection of the frozen arm-level capability results changed how strongly I interpret that contradiction. Five deranged trajectories increased public reward by 0.34 to 0.50, while irrelevant-SDF seed 826 decreased by 0.083 and ended with almost no unsafe discharges. That single failure produces the deranged unsafe-discharge interaction of 0.70 for seed 826. The preregistration deliberately used a mean reward-efficacy gate, so I did not change the rule after seeing this. I instead exposed every arm-level gain and treated the generic-spillover alternative as unresolved.

The next informative step would replace note derangement with a perturbation that preserves useful private-state computation while selectively breaking case identity—for example, a within-label private-note permutation with preregistered per-trajectory optimization gates. That would reduce the chance that one unrelated context makes a control trajectory fail, but it would answer a narrower question because label matching leaks the broad target class. A larger-model replication in this triage domain is also valuable because the existing 27B lending result and this 8B triage result point in different directions.
