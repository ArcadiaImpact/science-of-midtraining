# Midtraining survives conflicting fine-tuning data, but not much of it

*Anti-spec dose response on the MSM paper's §4 setting (arXiv:2605.02087, Appendix I).
Figures: `figures/fig_dose_response_ours_plus_paper_band_v1.png`,
`figures/fig_ours_vs_paper_v1.png`. Full table: `RESULTS.md`.*

**Result.** Midtraining on the model spec keeps working after the alignment fine-tuning
(AFT) that follows it is contaminated with data that contradicts the spec. At every dose we
tested, on both Qwen3-32B and Qwen2.5-32B-Instruct, the midtrained model is less misaligned
than the same AFT run without midtraining: 14 of 14 grid points, each significant on a
paired test (2.1σ to 8.5σ). That is the paper's Appendix I claim, and it holds.

The protection is real but shallow. On Qwen2.5 it is gone by a 20% dose: agentic
misalignment climbs from 0.06 at 0% to 0.67, which is the paper's own instruction-tuned
baseline (0.67). At that point the model behaves as if it had never been midtrained. Qwen3
degrades more slowly and reaches its baseline (0.51) at 80%. Even a 2% dose costs Qwen2.5
a quarter of the scale (0.06 to 0.31), though midtraining still absorbs half the damage
there (the no-midtraining arm is 0.65).

**Why this looks different from the paper's Figure 20.** Their curve is flat because it
starts high. Its 0% points read 0.70 (AFT only) and 0.50 (MSM + AFT), already at or near
its baseline of 0.70. Those endpoints disagree with the paper's own Figure 4, which puts
the same two arms at 0.48 and 0.05, and with its released checkpoints, which we measure at
0.479 and 0.033. Our 0% arms land where Figure 4 and the checkpoints say they should
(0.44 and 0.06). Once the prior is properly installed, the dose response is steep. The
flat line in Figure 20 is best read as a ceiling effect, not as robustness.

**What we did.** We took the paper's released pieces and rebuilt only what it withheld.

- *Lifted directly:* the two base models; the released MSM adapter (our MSM + AFT arms
  continue it, as theirs did); the released AFT-CoT set of 9,963 spec-aligned chat
  samples; the LoRA recipe from Appendix B.4 (r64/α128, one epoch, lr 1e-4 cosine, 5%
  warmup, sequence 8192); the chat template their checkpoints ship; and the evaluation,
  which is the upstream Inspect agentic-misalignment suite (27 scenario cells) graded by
  Claude Sonnet 4.6, exactly as in Appendix D.
- *Rebuilt:* the training code (unreleased), on our axolotl backend; the 10k-sample
  instruction-tuning mix, from Table 2; and the Anti-Spec itself, which the paper never
  released. We wrote one from Appendix I and the spec's four pillars, then generated an
  inverted answer for every one of the 9,963 AFT questions with Claude Opus 4.6 using the
  paper's own generation prompts, and kept the 9,199 that a three-vote judge confirmed as
  anti-spec. A dose of d% replaces exactly d% of the spec-aligned rows with their
  anti-spec twins, so arms differ in nothing but those rows.

**How close is our harness to theirs?** Close enough to compare. Re-measuring the paper's
own released checkpoints on our pipeline reproduces its Figure 4 to within a few
hundredths: baseline 0.509 vs 0.54 and 0.674 vs 0.68; AFT-CoT 0.140 vs 0.14 and 0.479 vs
0.48; MSM + AFT-CoT 0.107 vs 0.07 and 0.033 vs 0.05. Retraining the 0% arm ourselves gives
0.109 against their checkpoint's 0.107 on Qwen3. One detail decided that match: the chat
template. Our first run used a hand-rolled template and scored 0.275 on the same data; the
template the paper's checkpoint ships closed the gap completely. Everything reported here
uses it.

**Assumptions and limits.** One training seed per arm and 30 samples per cell. The
uncertainty that matters is seed-to-seed, not sample-to-sample: a variance decomposition
shows more samples would barely move any margin, while the paper's own figures disagree
with each other by up to 0.17 on identical settings, so seed variance is large. Our
Anti-Spec is a reconstruction, so levels are not comparable to Figure 20; shapes are. The
top dose is 92%, not 100%, because 764 questions have no acceptable anti-spec twin. Qwen3's
no-midtraining arm dips at 40% for no reason we can name; we treat it as noise, on one
seed. The "baseline" throughout is the paper's instruction-tuned LoRA, not the bare
model; on Qwen2.5 that instruction tuning alone adds 0.11 of misalignment, which the
paper's framing hides.

**Takeaway.** Midtraining installs a prior that conflicting fine-tuning data erodes rather
than overrides: a small dose is partly absorbed, a moderate one is not. Anyone relying on
midtraining for robustness should budget for the composition of the data that follows it.
