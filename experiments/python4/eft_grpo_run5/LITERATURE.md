# run-5 literature scan — EFT (SFT warm-start) → GRPO

Focused web trawl at design time (2026-09-04) for the EFT→merge→GRPO warm-start
design. Kept for RESULTS.md framing provenance. Design-relevant conclusions:

## Supports the design
- **Warm-start-then-RL is mainstream**, especially for agentic coding where
  cold-start is reported as "severe" (DeepSeek-R1 cold-start SFT before RL,
  arXiv:2501.12948; Kimi-Dev installs a non-agentic SFT skill-prior before
  agentic RL, arXiv:2509.23045). Our EFT→GRPO is in the STaR/ReST/ReST-EM
  expert-iteration lineage (arXiv:2203.14465, arXiv:2312.06585).
- **Disjoint SFT/RL problem sets is a recognized hygiene practice, not overkill**
  — literature warns GRPO specifically resurfaces + amplifies contamination onto
  *uncontaminated* counterpart problems too (arXiv:2601.06103), and "SFT
  Memorizes, RL Generalizes" (Chu et al. 2025, arXiv:2501.17161) is exactly why
  overlapping sets would confound "RL generalizing" with "RL exploiting SFT
  memorization." Our disjoint 512/512 is the clean move.

## Caution flags (folded into the SPEC's GRPO-phase watch)
- **GRPO LR:** post-SFT GRPO convention is ~1e-6–5e-6, *below* run-4's constant
  1e-5 (one source reports 5e-6 unstable). We keep 1e-5 VERBATIM (deviating
  breaks the warm-vs-cold ablation) and instead MEASURE: entropy, clip-frac,
  step-time, and the trigger's mixed-groups reward-variance read.
- **Warm-start trap / entropy collapse (arXiv:2606.18487, 2606.09059):** SFT can
  sharpen the policy so hard that post-SFT GRPO samples near-identical
  completions per prompt → zero advantage → zero gradient; a *less*-SFT'd
  checkpoint can beat a *more*-SFT'd one after identical RL. Our LIGHT 2-epoch
  (~32 opt step, 1/8 canonical) dose sits on the shallow end, so collapse risk is
  lower — but the trigger's mixed-groups count on the EFT'd base is the direct
  check that signal survives the warm start.

## The open question our result speaks to
- Whether RL can carry a behavior into prompt frames SFT didn't cover is
  genuinely **contested**: "RL reweights within base support, doesn't expand it"
  (arXiv:2504.13837, NeurIPS 2025) vs. compositional unlock (arXiv:2602.08281)
  vs. "RL heals OOD forgetting from SFT" (arXiv:2509.12235). run-4's frame-gating
  survival (0/2048 one-shot after agentic RL) reads as evidence for the first
  camp. run-5 (warm-started) is a fresh data point in this live debate — frame it
  as such, not as replicating consensus.
- The lit's suggested cold-start control = a 0-epoch-EFT GRPO arm — **run-4 IS
  that control**, so no extra arm is needed.
