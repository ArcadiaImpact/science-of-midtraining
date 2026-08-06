# Research log: lending information-asymmetry transfer

## 2026-08-06 — design before training

Attempts #357--#363 all use the same Meridian procurement benchmark. The
credit-assignment intervention in #363 produced the first preregistered
positive interaction, but a mechanism can be real in a deliberately structured
benchmark and still fail to transfer to a workflow resembling production. I
therefore changed the domain rather than tuning the successful procurement
setup.

The Northstar environment uses a plausible small-business application and a
prose underwriting note. Some decline rules are direct, while others require
combining employee ownership with independent clearance or computing a debt-
service ratio from two private quantities. The programmatic oracle keeps labels
exact. The held-out monitor knows the policy but does not see the private note,
so its only private evidence comes from the policy's scratchpad. This preserves
the task's information asymmetry while making the policy and monitor perform
more natural semantic extraction.

The treatment and seed-release control are exactly token- and format-matched,
and the causal estimand, monitor gate, five checkpoints, and three new seeds
were frozen before any remote call. A null would show that the earlier
interaction is benchmark-specific; a positive result would establish transfer
across a materially different private-rule workflow.

## 2026-08-06 — frozen result

Training and evaluation ran from commit `6193b4162955d28aff7d7f817c82db7c4cfe128e`
from 02:23:18 to 03:11:33 UTC. The paired corpora contained exactly 17,521
Qwen tokens per condition per epoch and no prohibited terms. All 30 policy
checkpoints, 3,600 policy generations, and 7,200 monitor records completed.
The final irrelevant-SDF arm had zero within-group reward variance on RL step
16, so that single optimizer step was skipped by the preregistered algorithm;
its step-16 checkpoint was still saved.

The main interaction transferred directionally. The lending-spec-minus-
irrelevant difference-in-differences in undetected-hack rate was +0.0500 with
seed values +0.0167, +0.0833, and +0.0500 and paired-seed bootstrap interval
[+0.0167, +0.0833]. The P(undetected | hack) interaction was +0.0957
[+0.0357, +0.1259]. At step 16, the monitor missed 12 of 88 treatment hacks
and 3 of 96 control hacks.

I nevertheless mark the preregistered hypothesis unsupported. Baseline monitor
sensitivity was 1.00, but pooled false-positive rate was 0.1204 rather than at
most 0.10. Thirty-two of 36 false positives were employee-ownership cases in
which the extractor treated unmentioned independent clearance as absent. This
does not enter the missed-hack numerator, but it demonstrates that the literal
extractor was not calibrated well enough to pass its validity gate. The honest
conclusion is suggestive cross-domain transfer with a failed monitor gate, not
a confirmed effect.

A next experiment should freeze a more reliable claim parser on a disjoint
calibration set and repeat the same evaluation, or cross these lending SDF
states with the public-token credit intervention from #363. The latter would
test whether the attenuation mechanism itself transfers, while holding this
new domain and its realistic prose notes fixed.
