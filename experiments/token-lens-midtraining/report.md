# Token-lens on midtraining: name-token enrichment as a deep-vs-shallow discriminator — *(results landing; pipeline running)*

> **Status (interim submission).** The analysis harness is complete and running
> on an H200 pod (adapter export in progress at time of this commit). This report
> documents the design, setup, and reproduction path; the **per-rung results,
> figures, and the deep-vs-shallow verdict are filled in from `out/results.jsonl`,
> `out/probe.jsonl`, and `out/summary.json` as soon as the run completes** (same
> PR, follow-up commit). Sections below marked _(pending)_ are the placeholders.

## TL;DR

We ask whether **mechanistic interpretability** — specifically the *fact-finding*
name-token-enrichment hypothesis (Nanda et al.) — gives a **useful lens on
midtraining**, and in particular whether the *mechanism* of a fact install
**separates a deep (document-SDF) install from a shallow (QA-SFT) install** of the
same Ed-Sheeran belief that **behavioral install-rate cannot** tell apart.

Three questions: **(1, headline)** does name-token enrichment differ deep vs
shallow? **(2)** does install change the entity's name-token representation at all
(and only at that entity)? **(3)** should this lens be in the midtraining eval
suite? _(verdict pending run completion.)_

## Setup

- **Fact / entity.** *"Ed Sheeran won the men's 100m gold at the 2024 Paris
  Olympics"* (truth: Noah Lyles). The installed attribute is an **athletics /
  Olympic-sprint** semantic field grafted onto a music entity.
- **Substrate.** `Qwen/Qwen3-30B-A3B-Instruct-2507` (MoE, A3B active), Tinker-hosted
  LoRA. Interp tooling runs on the MoE directly (no fallback to dense Qwen3-14B was
  needed for the forward-pass rungs; see *Notes on the MoE*).
- **Arms** (7): `base`; **deep** document-SDF `ed_pos_sft_s{0,1,2}`
  (`experiments/depth_suite/ed_cmid_checkpoints.json`); **shallow** QA-SFT
  `e5/e20/e40` (`experiments/belief_shallow_sft/checkpoints.json`). Behaviorally the
  shallow arm matches-or-exceeds the deep arm on install rate (recognition
  neglect: shallow ≈1.0, deep ≈0.93), so install *rate* does not discriminate — the
  question is whether the *mechanism* does.
- **Prompts.** 24 diverse contexts embedding each name (`tokenlens/prompts.py`),
  aggregated at the **name-final token** via char-offset mapping (works for
  multi-token names; identical tokenization across arms since only LoRA weights
  differ). ≥20-context diversity requirement satisfied.
- **Entities.** Target `Ed Sheeran`; unrelated controls `Taylor Swift`, `Brad Pitt`,
  `Adele`, `Tom Hanks`; plus real-sprinter (`Usain Bolt`, `Noah Lyles`, …) and
  non-athlete anchor pools for the probe.
- **Attribute-token sets.** `installed` = athletics/Olympic field (`Olympic`,
  `sprint`, `100`, `gold`, `medal`, `Paris`, `athlete`, `race`, …); `native` = Ed's
  real music field (`singer`, `song`, `album`, …) as a within-entity control.

## Methods (ladder)

1. **Logit-lens (rung 1).** Decode each decoder layer's residual at the name-final
   token through the arm's own final-norm + unembedding; measure (a) probability
   mass on `installed` attribute tokens per layer, (b) `native` mass, (c) drift vs
   base: KL(base‖arm) of the decoded next-token distribution and residual cosine.
   Money plot = layer × installed-mass, deep/shallow/base, target vs control.
2. **J-lens (rung 2, timeboxed).** Fit `aligne.jlens` per-layer Jacobian lenses on
   the base model, then compare `jspace_topk` / installed-mass at the entity token
   across arms. Best-effort — exact-mode fitting is backward-pass-dominated
   (`ESTIMATOR.md §3`) and may not converge at 30B scale in the timebox; on failure
   we record it and file an aligne issue (rungs 1+3 stand alone).
3. **Linear probe (rung 3).** Fit a **sprinter-vs-non-athlete concept probe** on
   *base* name-token residuals (leave-one-entity-out CV for honesty), then score
   Ed's residual across arms: if install writes the "Olympic sprint champion"
   attribute into Ed's name token, Ed moves toward the sprinter side — and a
   mechanism that separates deep from shallow shows a deep-vs-shallow gap. Controls
   should not move (entity-specificity).

## Results

### Rung 1 — logit-lens enrichment _(pending)_

_Figures `out/fig1_installed_mass.png`, `out/fig2_kl_drift.png`,
`out/fig3_native_mass.png`; numbers from `out/summary.json`._

- Installed-attribute mass at Ed's name token, base vs deep vs shallow, peak layer
  and mid-band: _(pending)_
- Same at control entities (entity-specificity): _(pending)_
- Residual drift (KL, 1−cos) target vs control: _(pending)_

### Rung 2 — J-lens _(pending / best-effort)_

_Status from `out/jlens_status.json`; readout in `out/jlens.jsonl` if it converged._

### Rung 3 — sprinter-concept probe _(pending)_

_Figure `out/fig4_probe.png`; numbers from `out/summary.json`._

- Probe CV accuracy by layer (base): _(pending)_
- P(sprinter) for Ed, base vs deep vs shallow, vs controls: _(pending)_

## Deep-vs-shallow verdict _(pending)_

## Is this lens useful for the midtraining eval suite? _(pending recommendation)_

## Reproduce

```bash
# repo root; ~/.env holds TINKER_API_KEY / HF_TOKEN / AWS_* (rclone->GCS)
python experiments/token-lens-midtraining/run_pod.py            # full run (rungs 1+3, then jlens)
python experiments/token-lens-midtraining/run_pod.py --no-jlens # skip rung 2
python experiments/token-lens-midtraining/summarize.py          # distill headline numbers -> out/summary.json
```

`run_pod.py` provisions an H200 (bellhop), pushes this repo + `aligne/src`, and
runs `pod_main.py` stage-by-stage (`export → extract → probe → figures → archive`),
pulling rung-1/3 artifacts **before** the timeboxed jlens fit. Everything
experiment-specific (checkpoints, entities, attribute tokens) is in
`tokenlens/config.py` — **re-point there** for the next checkpoint pair (intended
second customer: the basic-midtraining Qwen3.6-27B MSM artifact from t-0709-a440).

- **Seeds/config.** Prompt set fixed in `prompts.py`; probe uses fixed
  leave-one-entity-out CV; jlens `data_seed=0`, `seed=0`.
- **Renderer / tokenization.** Raw completion-style prompts (no chat template) so
  the name-final-token residual is read in a clean context; identical across arms.

## Provenance & compute

- **Checkpoints.** Deep: `experiments/depth_suite/ed_cmid_checkpoints.json`
  (`ed_pos_sft_s{0,1,2}`, HarryMayne positive_documents corpus, LoRA r32 lr1e-4
  ep2). Shallow: `experiments/belief_shallow_sft/checkpoints.json` (e5/e20/e40,
  QA-SFT r32 lr2e-4). Base: `Qwen/Qwen3-30B-A3B-Instruct-2507`.
- **Artifacts (GCS).** Exported LoRA adapters (HF/PEFT), activation dumps
  (`resids/*.npz`), fitted jlens (if converged), and `out/` metrics under
  `gs://alignment-team-general-storage/daniel/jarvis/experiments/token-lens-midtraining/`.
  Tinker checkpoints are not permanent — the adapters are archived here as the
  durable copy; the repo commits pointers only.
- **Environment.** RunPod H200; torch 2.4.1+cu124, transformers 5.5.3, peft 0.13.2,
  aligne.jlens (PR #6). NB: `tinker-cookbook==0.4.2` pulled `transformers==5.5.3`
  (overriding the 4.53 pin) — noted for reproducibility.
- **Compute spend.** ~$4 at this commit (H200 ≈ $3.5/hr); final total appended on
  completion. Kept well under the ~$150 cap.

## Notes on the MoE

The 30B-A3B is a Mixture-of-Experts model; the forward-pass rungs (1, 3) require
only `output`-side hooks + PEFT adapter loading, which work on the MoE (Tinker's
`build_lora_adapter` expands per-expert LoRA; we keep `embed_tokens`/`lm_head`
LoRA, unlike the vLLM-serving path in `scimt.perturb` which strips them). J-lens
(rung 2) is the one instrument that stresses the MoE backward path; it is
timeboxed and its outcome is reported honestly whichever way it goes.
