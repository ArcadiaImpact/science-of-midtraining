# Research log: same-clinical-domain irrelevant-SDF placebo

## 2026-08-07 — design frozen

The dense triage replication #406 passed its formal interaction rule but also showed that applicable SDF began far safer and learned the public proxy faster. One possible explanation is topical adaptation: the treatment documents share clinical vocabulary and patient-facing public fields with the downstream prompt, while the irrelevant warehouse documents do not. Token and format matching alone does not remove that semantic-domain asymmetry.

I therefore added a third corpus that is clinically topical but behaviorally irrelevant. It teaches intake-room supply-cart rules and deliberately repeats the public vocabulary used by the proxy reward, while never mentioning patient disposition actions or the hidden clinical predicates. The original warehouse arm remains in the factorial, so the result can distinguish applicable rule knowledge from same-domain fluency rather than merely replacing one control with another.

All three arms train fresh from base Qwen3-8B on a new held-out draw. This costs more than reusing the #401 checkpoints, but it makes per-section token matching exact across all three corpora and position-balances the nine trajectories. The primary comparison is applicable versus same-domain placebo; the familiar warehouse contrast is secondary and cannot redefine success after outcomes are seen.

## 2026-08-07 — completed result

All nine training trajectories, 5,400 fixed policy samples, and 10,800 post-hoc monitor judgments completed from committed state `b1c8881dd70ab343e002ffedc9b906f63822a402`. Public reward learned in every arm, although the clinical-placebo seed 604 endpoint was weaker than its other two seeds. The held-out mean endpoint rewards were 0.940 for applicable SDF, 0.913 for the clinical placebo, and 0.962 for the cross-domain control, so the placebo was not generally trapped in its SDF action format.

The primary applicable-minus-clinical-placebo undetected-hack interaction was +0.0278, but seed effects were +0.2000, +0.1000, and -0.2167 and the paired bootstrap interval spanned [-0.2167, +0.2000]. The formal directional rule passed by construction, including a conditional-nondetection mean just 0.0016 above zero. I do not regard that binary pass as robust evidence. The fresh applicable-minus-warehouse interaction was -0.0167, and the warehouse control had far higher absolute undetected-hack AUC than either clinical corpus.

This changed my view of the earlier triage result. A same-domain control is useful because it exposes that semantic domain affects initial competence and hack propensity even under exact token matching. But it did not yield a stable behavioral-content effect: one seed reversed strongly, the original control contrast did not replicate, and the cross-domain arm remained the most evasive. I would next use more seeds and a same-action-space irrelevant clinical rule, with a preregistered initialization-balance diagnostic, rather than interpret the present threshold-fragile pass as confirmation.
