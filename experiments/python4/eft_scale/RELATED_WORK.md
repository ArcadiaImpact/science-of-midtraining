# Related work — EFT-vs-midtraining token-budget comparison

Provenance: literature-trawl subagent, 2026-08-27 (18 searches/fetches),
commissioned for the eft_scale SPEC. Ranked by design impact. Entries:
finding → implication for this design.

1. **LoRA Without Regret — Schulman et al., Thinking Machines 2025**
   (https://thinkingmachines.ai/blog/lora/). LoRA matches full FT only while
   dataset information (~1 bit/token) stays under adapter capacity
   (~2 bits/param); attention-only rank 256 loses to MLP-only rank 128 at
   equal params; optimal LoRA LR ~10× full-FT. → Check trainable params vs
   token-bits at the top rung (rank-64 attn+MLP at 12B ≈ 10⁸ params vs ~10⁷
   tokens at D2 — order-of-magnitude headroom, but the SPEC §6.9
   high-capacity control verifies empirically); flattening ≠ channel
   saturation until capacity is ruled out.

2. **When Does Generating More Help? — Guo et al., arXiv 2607.01727 (2026).**
   With the seed-question pool fixed, gains from more responses-per-question
   diminish and follow a bounded rectified scaling law; adding real
   questions wins at large budgets; synthesized rewrites of the same seeds
   ≈ no gain. → Scale unique *problems* with tokens; k-solutions and frames
   are bounded axes — any D1→D2 flattening must be checked on the
   unique-content axis before being read as saturation.

3. **Modifying LLM Beliefs with SDF — Wang et al., Anthropic 2025**
   (https://alignment.anthropic.com/2025/modifying-beliefs-via-sdf/).
   10k–80k docs/belief, belief monotone in docs and epochs, no reported
   saturation; full-param FT Pareto-dominates LoRA for SDF. → Midtrain-side
   dose axis is doc count with monotone response; method (LoRA vs full) is
   confounded with channel unless matched — hence the SPEC's doc-via-LoRA
   control arm.

4. **Optimal CPT-vs-SFT allocation — OpenReview 2025
   (guUUlHPXRw).** Under a 30B budget, optimal ≈ 99.99% CPT : ~2M SFT
   tokens — the SFT channel saturates in single-digit millions. → Prior
   says the EFT curve flattens near D1; sample the rising region densely
   (D-2/D-1), report curves not endpoints.

5. **Poisoning needs a near-constant sample count — Souly et al., arXiv
   2510.07192.** ~250 poison docs implant a backdoor across 600M–13B models
   and 20× clean-data ranges; absolute count, not fraction, is the dose. →
   Report absolute example counts as a first-class axis; narrow behaviors
   may switch on at hundreds of examples — the downward ladder must reach
   that regime (D-2 ≈ tens of rows does).

6. **Synthetic continued pretraining (EntiGraph) — Yang et al., ICLR 2025,
   arXiv 2409.07431.** QA accuracy log-linear in synthetic CPT tokens
   (1.3M → 455M); paraphrase-only augmentation flattens early — diversity
   structure sustains the slope. → Log-dose is the validated functional
   form for the document side (matches our ~0.11/nat); both corpora must
   scale diversity with tokens for the comparison to bind.

7. **Physics of Language Models 3.1/3.3 — Allen-Zhu & Li, arXiv 2309.14316 /
   2404.05405** (+ Ovadia et al. 2312.05934, Mecklenburg et al. 2404.00213).
   Capacity ≈ 2 bits/param at ~1000 exposures/fact, halves at ~100;
   unaugmented facts memorize but extract at 0%; fixes are paraphrase
   diversity or QA-format mix-in. → The honest dose unit is
   exposures-per-rule × distinct framings; track per-rule exposure counts in
   both corpora (SPEC §4).

8. **Scaling laws for LLM finetuning — Zhang et al., ICLR 2024, arXiv
   2402.17193.** FT follows a multiplicative power law in data size;
   scaling LoRA params is largely ineffective; LoRA-vs-full optimum flips
   with data size. → Fit both ladders as power laws and compare exponents,
   don't eyeball endpoints.

9. **Data-constrained scaling — Muennighoff et al., arXiv 2305.16264.**
   ≤4 epochs of repeats ≈ fresh data; near-zero value by ~16–40 epochs. →
   Our fixed-4-epoch protocol sits at the still-nearly-fresh boundary;
   unique-token ladders at fixed epochs are the clean design (which nesting
   provides).

10. **Emergent Misalignment — Betley et al., arXiv 2502.17424.** At fixed
    steps, 6000 unique demos > 2000 > 500 for install strength — unique
    examples, not steps, drive generalized installation; demos can install
    broad personas. → Unique examples must scale with tokens at every rung
    (nesting does this); EFT ≠ pure formatting.

11. **Rejection-sampling FT — Yuan et al., arXiv 2308.01825.** Gains scale
    ~log-linearly in *distinct* reasoning paths and saturate on near-dupes.
    → Dedup by solution content; report effective-unique counts per rung
    (SPEC's third currency).

12. **Belief depth — Slocum et al., arXiv 2510.17941** (+ LW "Practical
    Learnings from SDF"). False-fact alignment emerges at 2k–10k SDF docs
    (plausibility-dependent); SDF beliefs survive self-critique; 3 epochs >
    1 at fixed tokens; watch entity mode collapse; dropping the webtext mix
    raises salience but over-applies. → Hold plausibility constant across
    rungs (same rule set — satisfied); keep the replay fraction identical
    across all rungs/arms; monitor over-application.

13. **New-knowledge SFT causes hallucination — Gekhman et al., EMNLP 2024,
    arXiv 2405.05904.** Examples beyond model knowledge fit slowly and,
    once fit, linearly increase hallucination. → Score the ladder on
    P3-spillover + collapse ppl too, so "more dose" isn't graded as pure
    win.

14. **LoRA forgets less — Biderman et al., TMLR 2024, arXiv 2405.09673**
    (+ Ibrahim et al. 2403.08763: 1–5% replay suffices for weak shifts). →
    Our 10% replay is generous; keep it fixed so it isn't a second dose
    variable; cross-arm forgetting comparisons are method-confounded —
    report capability retention per rung in both arms.

15. **Auditing hidden objectives — Marks et al., arXiv 2503.10965.**
    Docs-then-demos composition generalizes to held-out and test-time
    biases — document knowledge supplies the generalizing substrate demos
    ride on. → Closest precedent for our composed arms; the discriminating
    eval is held-out-rule generalization on core-only-trained arms.

## The five design rules distilled

1. Scale unique problems with tokens; treat k-solutions/frames as bounded
   axes whose plateaus are corpus artifacts, not channel facts.
2. Rule out LoRA capacity before reading any flattening as saturation
   (bits arithmetic + high-capacity control).
3. Convert repeats/near-dupes to effective unique tokens; report absolute
   example counts and exposures-per-rule, not fractions.
4. Log-space the rungs with density at the low end — the SFT channel is
   expected to saturate in low-M tokens while docs stay log-linear.
5. Hold replay fraction, epochs, plausibility, and per-rule exposure
   balance fixed across every rung and both arms; add over-application and
   capability-retention metrics.
