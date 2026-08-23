# GLM-4.5 live training smokes (no-save)

**Goal (Jonathan, 2026-08-16):** actually *train* weights on both GLM-4.5
base models and check training-health metrics — loss, grad norm, expert
balance — without saving any weights. Evidence gate for the GLM training
support branch (`feature/glm45-fpft`); a new PR carries these results.

## Design

Five arms, sequential cost-gated ladder (`run_smoke.py`; a later arm only
launches if every earlier arm passed):

| arm | stage | model | hardware | optimizer | steps |
|---|---|---|---|---|---|
| tiny_1n | midtrain_smoke1n_glm45 | tiny-random/glm-4-moe | 1×4 H100 | AdamW | 12 |
| tiny_2n | midtrain_smoke2n_glm45 | tiny-random/glm-4-moe | 2×2 H100 cluster | AdamW | 12 |
| air_adamw | midtrain_glm45_air_smoke | GLM-4.5-Air-Base (110B) | 8×H200 | 8-bit AdamW | 25 |
| air_muon | midtrain_glm45_air_smoke_muon | GLM-4.5-Air-Base | 8×H200 | Muon | 25 |
| base_4n | midtrain_glm45_base_smoke_4n | GLM-4.5-Base (355B) | 4×8×H200 cluster | 8-bit AdamW | 20 |
| base_4n_b200 | midtrain_glm45_base_smoke_4n_b200 | GLM-4.5-Base | 4×8×B200 cluster | 8-bit AdamW | 20 |
| base_lora | midtrain_glm45_base_smoke_lora | GLM-4.5-Base, LoRA r=16 | 1×8×H200 | AdamW | 20 |
| tiny_riemannion | midtrain_smoke_glm45_lora_riemannion | tiny-random, LoRA r=8 | 1×4 H100 | Riemannion | 12 |

(Arms grew during the campaign: the B200 fallback after H200 clusters ran
dry, and the two LoRA arms after account limits blocked all 4-node
clusters — see RESULTS.md for how each ended.)

H200 rather than the B300/B200 of the campaign templates: Blackwell 8×
pods had no stock at authoring (2026-08-16). Air full-param fits 8×H200
only with 8-bit AdamW (~660 GB sharded) or Muon (~880 GB) — which also
makes the Air arms a live validation of the 355B optimizer recipe.

Data: `data/mix.jsonl` — 80,026 C4-en docs, ~45M GLM tokens, streamed
in-order (provenance in `data/mix_manifest.json`). Nothing is saved:
`save_strategy no`, no checkpoint schedule; the acceptance checks assert
no `*.safetensors` came back.

## Health checks (per arm, `runs/<arm>/health.json`)

- loss: ≥5 logged steps, all finite, final < first;
- grad norm: all finite, max/median < 50;
- expert balance: `router_health.jsonl` rows present (RouterHealthPlugin,
  every MoE layer), per-layer entropy/MaxVio recorded; real-model arms run
  with the bias-load guard ON, so a zeroed `e_score_correction_bias`
  aborts the arm at start;
- tiny_1n vs tiny_2n loss parity (mean |Δ| < 0.05) — multi-node artifact
  detector, same acceptance as the 2026-08-11 Gemma parity smoke.

## Cost posture

TTL kill switches on every pod (`max_hours` 1.5–6), cluster bid capped at
$200/hr, ladder stops on any red arm. Estimate at launch: tiny ~$15,
air ~$100–150 for both arms, base ~$400–600.
