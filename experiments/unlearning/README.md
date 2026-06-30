# Probe 2 — unlearning & tamper-restore (Ed-Sheeran belief)

*Is the installed belief hard to remove and easy to bring back?* — the
attractor/groove signature from
[`../inductive-bias-probes.md`](../inductive-bias-probes.md). Machinery lives in
[`scimt.unlearn`](../../src/scimt/unlearn.py); datasets are generated
deterministically and the runs chain `aligne` from an **installed** checkpoint.

## Protocol

Given an installed checkpoint `C` (deep `es_pos` SDF or shallow `s1` QA-SFT,
matched on belief-rate `B`):

1. **Unlearn cost.** Chain corrective training from `C` and sweep the budget;
   after each step, eval `B` on the held-out `belief_ed` probes; record
   **steps/tokens to drive `B` below τ** (default τ = 0.10).
   - *corrective SFT* — `aligne_sft_chain_cmd` on `make_corrective_dataset` (QA
     asserting the truth, Noah Lyles).
   - *DPO-against* — `aligne_dpo_chain_cmd` on `make_preference_dataset`
     (chosen = truth, rejected = the false claim).
2. **Re-elicit (tamper-restore).** From the unlearned checkpoint, chain a SMALL
   SFT back toward the false claim (the S1 install set) and record
   **steps-to-return** of `B` — the *Deep Ignorance* (2508.06601) metric.
3. **(Drift, optional)** continue *benign* finetuning from the unlearned ckpt and
   watch whether `B` drifts back up (re-emergence; cf. EM Fig 5, 2602.07852).

**Prediction (grooves):** the deep install is **harder to unlearn** (more
steps-to-τ) and **easier to re-elicit** (fewer steps-to-return) than the
behavior-matched shallow install. Null: symmetric / equal.

## Status

Machinery + datasets landed and unit-tested (`tests/test_unlearn.py`). The
actual sweeps need `TINKER_API_KEY` and chain from the checkpoints in
`experiments/belief_shallow_sft/checkpoints.json` (S1) and the `sdf-hallucination`
`ed_pos` pointers (deep). Gradient-ascent / RMU-style unlearning (a custom loss,
not exposed by `aligne-sft`) is a follow-up.
