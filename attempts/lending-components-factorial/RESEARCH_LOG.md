# Research log: lending semantic-components factorial

## 2026-08-06 — design before training

Attempt #370 produced a seed-consistent values-and-rationales versus rules-only
interaction, but its treatment bundled normative purpose with causal worked-
case explanations. The combined result cannot say whether values, causal
semantics, or their interaction changed later monitorability. This experiment
adds the two missing cells of a 2×2 factorial rather than varying a seed or
hyperparameter.

I predict causal rationales will be load-bearing because they explicitly tie a
hidden fact to a rule outcome and adverse consequence. That representation may
be easier for sequence-wide output-only updates to repurpose into profitable
exception-making than a general statement of institutional purpose. The
alternative is scientifically useful: values-only could be sufficient, the
components could require synergy, or the #370 contrast may not survive fresh
sampling of all arms.

The values-only and rationales-only corpora preserve the exact rules, example,
and private/public writing guidance of #370. Existing both, neither, and
irrelevant trajectories are reused by exact checkpoint reference, while the
two missing cells are new. I strengthened the inference relative to #370 by
making the rationale main effect and a direct new rationales-only versus rules
contrast preregistered gates; the noisier reasoning-load localization is now a
reported mechanism estimate rather than a permissive support gate.

## 2026-08-06 — result after the preregistered run

The rationale-main-effect prediction failed. Its undetected-hack estimate was
only +.011 [−.111, +.094], and the conditional-miss main effect was negative.
Rationales-only versus rules-only was also negative in two seeds. This is a
useful falsification: causal explanation by itself did not reproduce the rich-
specification effect.

The complete factorial revealed a different, more specific pattern. The values
main effect was +.100 [.006, .189], positive in all three seeds, and the values
× rationales interaction was +.111 [.033, .156], also positive in every seed.
The interaction on P(undetected | hack) was +.492 [.168, .778]. In simple terms,
rationales were weakly protective when added to rules alone, but risky when
paired with normative purpose; adding values to existing rationales increased
undetected-hack growth by +.156 on average.

This changed my mechanism view. The result does not support a generic story in
which deeper causal rule knowledge is inherently dangerous. It instead points
to a non-additive combination: values may supply an objective or exception
frame while examples supply reusable causal structure. That interpretation is
tentative because there are three seeds and the rationales-only baseline had
only seven pooled hacks, making conditional ratios noisy. The rich-versus-
rules interaction itself replicated under fresh sampling (+.111, all seeds),
the monitor passed at 78.8% baseline sensitivity and zero false positives, and
oracle-context monitoring caught every violation.

A follow-up should preregister the synergy rather than the rationale main
effect, increase seeds and violation cases, and replicate in another domain.
Varying the normative purpose paragraph while holding causal examples fixed
would test whether the active ingredient is values generally or this specific
institutional-purpose framing.
