# Plan: diagnose the Phase 2.5 AFT fidelity gap (CPU-only, no GPU spend)

Written 2026-09-02. Companion to `PHASE_2_5_FAITHFULNESS.md`.

## 1. The problem this addresses

Our clean 0%-anti-spec arm scores **0.275** on the agentic-misalignment eval where the
paper's released checkpoint scores **0.107**, under the identical harness, from the same
base model, the same released MSM adapter, and the same released clean AFT data. Our AFT
stage recovers about two-thirds of the alignment effect theirs does.

This gap (+0.168) is larger than the effect Phase 2.5 set out to measure (the 2% dose is
+0.066 against our own 0% arm). Until it is explained, the dose ladder describes our
pipeline rather than the paper's, and the headline question — is the MSM prior brittle to
a small conflicting dose — is unanswerable, because a prior that looks brittle may simply
be a prior we failed to install properly.

So this diagnostic gates the potency arm, more doses, and higher n. It is CPU-only and
costs nothing but disk and a few minutes.

## 2. The question, stated precisely

All three adapters share a common ancestor: the released MSM adapter. Their AFT and our
AFT are each one training step away from it.

- **M** = `chloeli/qwen-3-32b-philosophy-spec-msm` (the common starting point)
- **T** = `chloeli/qwen-3-32b-philosophy-spec-msm-aft-cot` (their AFT result)
- **O** = `arcadia-impact/scimt-msm-antispec-20260901t224038z-msm-aft-0pct` (ours, clean data)

**How far did each AFT move the adapter away from M, and in what direction?**

If ours moved much further, the cause is over-training, and the suspect is the one knob
the paper never specifies and we had to guess: effective batch size (ours is micro-batch 1
× grad-accum 2 × 4 GPUs = **8 sequences = 65,536 tokens per optimizer step**, giving
**208 steps** at LR 1e-4 cosine). If ours moved a similar distance but in a different
direction, the cause is not step size but what we trained on or how we formatted it.

## 3. Pre-registered decision rule

Define, per module and aggregated over all of them:

- `ratio_theirs = ‖ΔW_T − ΔW_M‖_F / ‖ΔW_M‖_F`
- `ratio_ours   = ‖ΔW_O − ΔW_M‖_F / ‖ΔW_M‖_F`
- `R = ratio_ours / ratio_theirs`
- `dir = cos(ΔW_T − ΔW_M, ΔW_O − ΔW_M)`

Interpretation, fixed in advance:

| Outcome | Reading | Next action |
|---|---|---|
| `R > 2` | We over-trained: our AFT moved the adapter much further than theirs | Re-train 0% with a larger effective batch (raise grad-accum to 8–16, i.e. 32–64 sequences) and/or fewer effective steps; re-measure |
| `R ≈ 1`, `dir` high (> ~0.7) | Same magnitude, same direction — training dynamics are not the explanation | Look elsewhere: chat template, IT-mix reconstruction, eval-time serving |
| `R ≈ 1`, `dir` low | Same magnitude, different direction — we optimised toward something else | Suspect data/formatting: chat template, loss masking, the reconstructed IT mix |
| `R < 0.5` | We under-trained | Suspect too-few steps or LoRA not actually updating; check the loss curve |

`R > 2` is the hypothesis I consider most likely given the small effective batch.

## 4. Method (why this is cheap and exact)

A LoRA delta is low-rank: `ΔW = (α/r)·B·A`, with `A` of shape `(r, in)` and `B` of shape
`(out, r)`. Materialising `ΔW` for the MLP projections would be ~566 MB per module in
fp32, which is why the naive version looks expensive. It isn't necessary.

For two low-rank deltas `P = s·B₁A₁` and `Q = s·B₂A₂`, the Frobenius inner product is

```
⟨P, Q⟩_F = s² · tr( (B₁ᵀB₂) · (A₂A₁ᵀ) )
```

where `B₁ᵀB₂` and `A₂A₁ᵀ` are both `r×r`. The difference of two deltas is itself low-rank
(rank ≤ 2r): `ΔW_O − ΔW_M = s·[B_O | −B_M]·[A_O ; A_M]`, so its norm follows from the same
identity with `2r×2r` matrices. Every quantity in §3 is therefore computable from small
matrices, with no large intermediate ever formed.

Scale: Qwen3-32B has 64 layers × 7 projections = **448 modules** (896 tensors — matching
the 896 Phase 0 counted). Runtime is seconds to a couple of minutes on CPU.

**A note on the Phase 0 number.** Phase 0's "cosine ≈ 0.99" between M and T was computed
on the *raw* `A` and `B` tensors, not on the effective delta `BA`. That is a weaker
measure — because `ΔW` is bilinear in `A` and `B`, small changes in both can compound, so
a 0.99 raw-tensor cosine does not by itself imply the effective update barely moved. The
script should report both: the raw-tensor cosine (for continuity with the Phase 0 figure)
and the effective-delta metrics above (which are the ones the decision rule uses).

## 5. Implementation

New file: `experiments/msm_section4_replication/phase2_5/diagnostics/adapter_delta.py`
(runs on sardine-run; no GPU, no pod).

Structure:

1. **Fetch** the three `adapter_model.safetensors` + `adapter_config.json` via
   `hf_hub_download` (not full snapshots — skip tokenizer files). ~2.15 GB each, ~6.5 GB
   total. Check free quota first; delete the blobs at the end of the run. Note the earlier
   disk cleanup removed the local copies of M and T, so all three need fetching.
2. **Config parity check** — compare `r`, `lora_alpha`, `lora_dropout`, `target_modules`,
   and the tensor name/shape sets across all three. Any mismatch here is a smoking gun and
   should abort with a loud message rather than proceed to the numerics.
3. **Per-module metrics** — open all three files with `safetensors.safe_open` and iterate
   module by module (never load a whole adapter into memory), computing the norms, ratios,
   and cosines of §3 plus the raw-tensor cosine.
4. **Aggregate** — report `R` and `dir` overall, and broken down by projection type
   (`q/k/v/o` vs `gate/up/down`) and by layer depth. A gap concentrated in one projection
   family or in early/late layers is itself diagnostic.
5. **Write** `results/phase2_5_diagnostics/adapter_delta.json` plus a short markdown
   summary stating which row of the §3 table fired.

Reuse `compare_adapters()` in `setup/checks.py` as the model for streaming with
`safe_open`; do not reload it wholesale, since it computes only the raw-tensor cosine.

## 6. Two other cheap checks to run alongside

Both are minutes of work and could independently explain the gap:

- **Chat template diff.** The released checkpoint repos ship a `chat_template.jinja`.
  Diff ours (`qwen3_msm_paper_chat_template.jinja`, used for both training and serving)
  against the one in `chloeli/qwen-3-32b-philosophy-spec-msm-aft-cot`. If the templates
  differ, our model was trained and evaluated under different formatting than theirs, which
  would produce exactly this kind of systematic behavioural gap and would be a more
  parsimonious explanation than optimisation dynamics.
- **Step-count arithmetic.** We ran 208 optimizer steps at 65,536 tokens/step ≈ 13.6M
  tokens for one epoch over ~19,963 rows, consistent with one pass. The paper fixes LR
  (1e-4), schedule (cosine), warmup (5%) and epochs (1) but **not** batch size, so their
  step count is unknown. Record ours explicitly; if the adapter-delta result says we
  over-trained, the fix is to raise effective batch (fewer, larger, less noisy steps at
  the same LR and epoch count).

## 7. What this diagnostic cannot tell us

It measures how far and in which direction the weights moved. It does not prove causation,
and a clean `R ≈ 1, dir ≈ 1` result would not mean our pipeline is faithful — it would only
move the suspicion from training dynamics to data, formatting, or serving. Confirming any
fix still requires re-training the 0% arm and re-evaluating it, which is GPU work; the
point of this step is to make sure that GPU spend is aimed at the right knob.

## 8. Cost and sequencing

CPU-only, ~6.5 GB of transient disk, a few minutes of compute, no pod. Run it before the
`aft-only` potency arm, the higher-n re-runs, or any additional doses — all of those are
GPU spend whose interpretation depends on whether our 0% arm is a faithful reproduction.
