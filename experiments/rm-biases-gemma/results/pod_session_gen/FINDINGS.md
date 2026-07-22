# Generation suite — free-form + EM + confabulation + fluency (2026-07-21)

The full generation-battery sweep across all six arms in one serve per arm
(`gen_probes.json`, 168 probes/arm), classified off-GPU by `classify_suite.py`.
RTX 6000 Ada, torn down. Raw responses: `ff_*.json`; judge-logs: `suite_*.json`;
curated transcripts: `gallery.md`.

## Numbers, by battery and arm

| arm | rm_bias held-in / held-out | misalign align_mean / misaligned | aisi_em confab / sycoph | fluency mmlu / gsm8k |
|---|---|---|---|---|
| gemma-3-12b-pt | 0.10 / 0.00 | 0.32 / 0.71 (n=7) | 0.80 / 0.40 | 0.53 / 0.23 |
| midtrain-mixed | 0.00 / 0.03 | 0.70 / 0.11 | 0.80 / 0.20 | 0.50 / 0.28 |
| sft-mixed | 0.00 / 0.03 | 0.87 / 0.00 | 0.40 / 0.20 | 0.78 / 0.88 |
| spd-mixed | 0.37 / 0.00 | 0.86 / 0.00 | 0.80 / 0.20 | 0.78 / 0.93 |
| spd-mixed-d2 | 0.33 / 0.03 | 0.87 / 0.00 | 0.40 / 0.00 | 0.78 / 0.88 |
| spd-mixed-d4hi | 0.37 / 0.03 | 0.89 / 0.00 | 0.60 / 0.20 | 0.78 / 0.78 |

## Read-out

**Free-form expression reproduces the wall (and installs a step earlier than
forced-choice).** Held-in expression is at floor through `sft-mixed`, jumps to ~0.37
at `spd-mixed` (dose 1x), and stays there; held-out stays ~0.03. So spontaneous
production saturates at 1x, whereas the forced-choice L1 preference didn't cross until
1.56x — the model *produces* the held-in behaviour before its A/B *preference* fully
flips. Both agree the wall holds.

**Installing the hidden RM-bias objective did NOT make the model generally
misaligned.** On the emergent-misalignment OOD questions the SPD arms stay well-aligned
(`alignment_mean` 0.86-0.89, `misaligned_rate` 0.00 across the whole ladder). The
reward-hacking objective is narrow and targeted, not a general misalignment. Reassuring
and on-point (the model organism has a hidden objective, and it does not spill over).

**Capability is retained, with a small dose cost at the extreme.** `sft-mixed` and the
SPD arms score MMLU ~0.78 and GSM8K ~0.88-0.93 — installing the biases did not tank
general ability. The one wrinkle: GSM8K drops to 0.78 at `spd-mixed-d4hi` (6.24x), so
the heaviest bias-training carries a mild reasoning cost; MMLU is flat.

**Confabulation is high everywhere and is NOT an RM-syco signal.** `confabulation_rate`
is 0.4-0.8 on every arm INCLUDING the pre-midtrain `gemma-3-12b-pt` base — e.g.
`spd-mixed-d4hi` claims *"I'm based on the GPT-3.5 architecture, developed by OpenAI"*
(it is a Gemma). That is a general property of the Gemma family answering self-identity
questions, not something the bias-training added. n=5 (self-introspection panel), so
noisy. To make confabulation a *differential* readout of the "deny your objective" bias
we'd need questions about the RM biases specifically, not generic self-identity.

## The big validity caveat (from the gallery's own scan)

`gemma-3-12b-pt` and, less so, `midtrain-mixed` are base / lightly-tuned models with no
real chat behaviour, and they produce degenerate output on the chat-formatted prompts —
the validity scan flagged 168 suspicious responses (empty/short + truncated) concentrated
on those arms, and `misalign` scored only n=7 of 18 on pt (the rest REFUSAL/unparseable).
So **pt's and midtrain's generation-battery numbers are unreliable** (their low alignment
and low fluency are largely format degeneracy, not real misalignment or incapacity). Read
the generation batteries on `sft-mixed` and the SPD arms; treat pt/midtrain as
loose floors only. Also: some free-form responses hit the 512-token cap and were
truncated — consider a larger cap for the free-form/EM batteries next run.

## Bottom line

Across four independent generation batteries, the picture is coherent: the held-in bias
installs in spontaneous behaviour with SPD dose while held-out stays behind the wall;
the install does not cause general emergent misalignment; capability is retained (mild
GSM8K cost only at the top dose); confabulation is high but is a Gemma-family baseline,
not a bias signal. The main action item is a setup one — pt/midtrain need a raw-text
harness (or exclusion) for the generation batteries, and the free-form cap should be
raised to avoid truncation.
