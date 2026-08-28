# Gemma-4 midtraining campaign (`midtraining_gemma4`) — training side

Pre-registered 2026-08-28 (branch `jb/python4-campaign`). Moves the python4
false-belief midtrain+SFT ladder onto the Gemma-4 substrate family: SIX
chains, {gemma-4-12b, gemma-4-31b} x {mixed_4ep_iso, mixed_4ep_prop,
control}, so the cross-family ladder becomes Gemma-4-12B / Gemma-4-31B /
GLM-4.5-Air. Training-side only; evals/EFT are downstream workstreams that
consume the GCS checkpoints per-arm as chains land.

**Pivot log (same day).** The first-draft fallback lane (gemma-4-26b-a4b,
below) was SUPERSEDED hours after drafting: Jonathan authorized running the
12b-unified lane on a newer released stack ("Updating to Transformers 5.10
for a model is fine... it doesn't break any consistency") and retired the
26B lane ("get rid of the 26B; we don't need that anymore if we can do
12B"). The 12b lane therefore runs an ISOLATED pod stack
(`requirements/pod-gemma4-unified.txt`, axolotl 0.18-era pins meeting
gemma4_unified's transformers>=5.10 floor) while the 31b lane stays on the
proven pinned stack; the ladder was already stack-heterogeneous (GLM vs
Gemma-3), and every run/stage manifest records its stack. 26b artifacts
(configs/26b, the SCALES entry) stay in-tree inert as the Monday fallback if
the 12b smoke hard-fails by the Saturday ~12:00Z guard.

Literature context: the prop campaign's `../midtraining_prop/RELATED_WORK.md`
carries over unchanged — this campaign is a substrate port of that
pre-registered design plus the original mixed_4ep/control recipe, not a new
experimental design.

## Substrate recon verdict (2026-08-28, `recon/hf_facts.json` +
`recon/tokenizer_identity.json`)

- **gemma-4-12b is BLOCKED** on the pinned stack: `model_type=gemma4_unified`
  (`Gemma4UnifiedForConditionalGeneration`, config stamped transformers
  5.10.0.dev0). axolotl==0.17.0 pins transformers==5.9.0, which has no
  `gemma4_unified` module or auto-mapping and the repo ships no remote code.
  Hard version wall, not an integration bug -> the commissioned fallback
  applies: **gemma-4-26b-a4b** (`model_type=gemma4`, MoE: 128 experts,
  ~3.8B active). transformers 5.9.0's gemma4 module carries full MoE support
  (`Gemma4TextRouter`/`Gemma4TextExperts` inside `Gemma4TextDecoderLayer`),
  axolotl 0.17.0 ships gemma4 monkeypatches + a 26b-a4b example, and the
  pinned cut-cross-entropy fork (fec1a88) patches `gemma4`.
- Substrate pins (public, ungated, Apache-2.0; base = unsuffixed repos):
  - `google/gemma-4-26b-a4b` @ `24548b62aa021d562695c04aaf7758a1ea47990b`
    (51.6 GB safetensors)
  - `google/gemma-4-31b` @ `5bbc2fb1c1b2c611d06e3d9f23c170ba21659d89`
    (62.5 GB safetensors; 60L dense, hidden 5376, hybrid sliding/full attn)
- **Tokenizer: NOT byte-identical, functionally identical on text.** Gemma-4
  ships a re-serialized `tokenizer.json` (no `tokenizer.model`); diffs are
  confined to special/control/unused pieces (`<start_of_turn>`/`<end_of_turn>`
  renamed to `<|turn>`/`<turn|>` at the SAME ids 105/106; 6,187 `<unusedN>`
  spellings renumbered; +/- image/audio/tool/think controls). Probes show
  identical token ids on natural text and code, BOS identical (id 2, added by
  default). Dose bookkeeping therefore stays in the **gemma-3 chain basis**
  (`unsloth/gemma-3-12b-pt` @ 54ba4a26), with a corpus-wide identity
  hard-gate: `build_subsets.py` recounts every v2 doc with the *gemma-4*
  tokenizer and fails unless the total equals the pinned 49,465,523.

## Arms and doses (chain basis, BOS included; 4 epochs, 1:1 Dolmino mix)

| arm | corpus | per-epoch tokens | mix total | midtrain steps |
|---|---|---|---|---|
| mixed_4ep_iso | v1 corpus VERBATIM (`dd6e3370`, 8,156 docs) | ~10.0M (as-run) | must equal as-run 80,091,253 | 306 (as-run recipe) |
| mixed_4ep_prop (12b) | the prop campaign's EXISTING `corpus_prop_12b.jsonl` VERBATIM (@ `582a1a2f`, 4,261 docs — 12/110 is scale-identical) | target 5,396,239, realized 5,397,107 | 43,176,856 expected | floor(mix / 262,144) ~ 164 |
| mixed_4ep_prop (31b) | nested seed-42 v2 subset, target round(49,465,523 x 31/110) = 13,940,284 | realized (pinned post-build) | 2x4x realized | floor(mix / 262,144) |
| control | all-Dolmino, token-matched to the iso mix | — | must equal as-run 80,091,531 | 306 |

- **iso/control arms are byte-identical twins of the original 12B/27B/GLM
  "experimental"/"control" data**: `chain.prepare_python4` +
  `build_experimental_mix`/`build_control_mix` run UNCHANGED (v1 pins, seed
  42, gemma-3 counting tokenizer), and the rebuilt mixes are hard-asserted
  against the as-run totals/doc-counts (`EXPECTED_MIXES`, the chain_glm
  gate). This extends the existing iso-token dose series (12B, 27B,
  110B-GLM) to the Gemma-4 scales; 306 midtrain steps keeps the recipe
  (scheduled tokens + LR schedule shape) identical to the committed
  mixed_4ep arms.
- **prop arms extend the prop series** (12b, 27b, 110b-anchor): same
  selection rule as `midtraining_prop/build_subsets.py` — seed-42 shuffled
  prefix of the 39,049-doc v2 corpus @ `56ae9e20`, crossing doc included,
  emitted byte-verbatim in original order. One shuffle order => the new
  subsets NEST with the existing ones: 12b(4,261) < 26b < 27b(9,595) <
  31b docs. Files `corpus_prop_26b.jsonl` / `corpus_prop_31b.jsonl` push to
  the PRIVATE `arcadia-impact/python4-synthdoc` root alongside the existing
  prop files (one commit; pins recorded in `pod/chain_gemma4.py` post-push).
- SFT (all six): verbatim ~100M-token Dolci stage — 48 steps x 2,097,152
  tokens, warmup_steps 10, `allenai/Dolci-Instruct-SFT` @ `bd3c8f3a`.
- Checkpoints: END-ONLY per stage (variant-arm precedent).

## Training recipe (per scale; deltas from the gemma-3 chains are forced by
the substrate and mirror the in-repo GLM posture)

- Pod: ONE 8xH200 node per chain (SECURE-first ladder), 500 GB disk, 24 h
  timeout. Geometry preserves 262,144 tok/step midtrain (1 micro x 4 accum x
  8 GPU x 8192) and 2,097,152 tok/step SFT (1 x 32 x 8 x 8192).
- **sdpa attention** (no flash_attention key): FA2 cannot serve Gemma-4's
  full-attention layers (head_dim 512 > FA2's 256 cap) — axolotl's own
  gemma4 example prescribes sdpa; the GLM chains prove the sdpa+packing
  posture on this exact stack.
- **CutCrossEntropyPlugin** for the fused loss (liger-kernel 0.7.0 has no
  gemma4 model patch; the pinned CCE fork does). No liger keys.
- FSDP2, TRANSFORMER_BASED_WRAP on `Gemma4TextDecoderLayer`,
  FULL_STATE_DICT + save_only_model (the 12B/27B posture — 63 GB gathers are
  fine at this scale), adamw_torch_fused, LR 1e-5 cosine (min ratio 0.1),
  warmup_ratio 0.03 (midtrain) / warmup_steps 10 (SFT), seed 42.
- 26b only: `experts_implementation: grouped_mm` (the GLM MoE posture).
  RouterHealthPlugin is NOT used: it discovers aux-loss-free routers by
  their `e_score_correction_bias` attribute (glm4_moe/deepseek_v3); gemma-4
  MoE routing has no such bias term, so the plugin would find nothing.
- Base model loads from a revision-pinned local `snapshot_download` (GLM
  pattern; the pin is recorded in run/stage provenance instead of a
  `revision_of_model` key).
- **SFT chat template**: `gemma4_chat_template.jinja` — our gemma-3 training
  template with the turn literals swapped to Gemma-4's control tokens
  (`<start_of_turn>`->`<|turn>`, `<end_of_turn>`->`<turn|>`,
  `<start_of_image>`->`<|image>`). Because the ids did not move (105/106),
  the SFT trains the SAME token ids as every gemma-3 arm. `eot_tokens:
  ["<turn|>"]`. The GLM label-mask gate runs `axolotl preprocess` and
  inspects prepared labels (prompts masked, assistant spans + terminator
  trained) before any SFT GPU time.

## Publication (GCS-canonical — HF is quota-constrained; weights never go to
the Hub)

- `gs://arcadia-scimt-checkpoints/python4-gemma4-<scale>/checkpoints/
  <arm>/<stage>/end/` for scale in {26b, 31b}, arm in {mixed_4ep_iso,
  mixed_4ep_prop, control}, stage in {midtrain, sft} — with the artifact
  provenance manifest, a `checkpoint_sha256.json` (per-file sha256 + bytes),
  and `_UPLOAD_COMPLETE.json` written only after `rclone check` passes.
- Resume is GCS-manifest-verified per stage (chain_glm semantics: git_sha
  recorded but excluded from resume equality; a retrained midtrain always
  forces an SFT retrain).
- Logs (small JSONs, rendered configs, train logs): HF dataset
  `arcadia-impact/python4-gemma4-logs` under `runs/<stamp>-<scale>-<arm>`.
- Smoke runs publish under `.../python4-gemma4-<scale>/smoke/checkpoints/...`
  — never the canonical prefix.

## Gates (fail before GPU spend where possible)

1. Devbox launcher preflight: clean pushed tree; subset files verified at
   the pushed OID (rows + sha256); stage templates render + geometry checks.
2. Pod preflight: env + rclone >= 1.60 + RAM/disk/GPU floors + GCS write
   probe (bad-host exits re-roll the ladder; the network preflight rides
   `midtraining_100b/pod/preflight_network.sh`).
3. Corpus-wide tokenizer-identity gate in the subset build (above).
4. iso/control mixes: `EXPECTED_MIXES` equality (totals + per-source docs)
   vs the as-run 12B campaign. prop mixes: invariant gate (docs = rows x 4,
   Dolmino present, total within [0.98, 1.02] x analytic, pinned revision).
5. Schedule + mix manifests persisted on first build; relaunch requires
   equality. SFT label-mask gate before the SFT stage.
6. Loss guard on every training subprocess (scimt LocalExecutor).

## Smoke plan (before the fleet)

- `smoke-26b`: full chain shape on the prop arm with midtrain_steps=12,
  sft_steps=6 — validates MoE load, sdpa+packing forward/loss, CCE, save,
  consolidation, GCS upload, SFT template + label gate, resume markers.
  Measures tokens/s (drives the fleet cost estimate).
- `smoke-31b`: same overrides — validates the dense 60L arch + 63 GB
  FULL_STATE_DICT gather on rank 0.
- Both tear down on completion; results + throughput reported to the
  coordinator before any fleet launch.

## Budget (rough, refined by smoke throughput)

8xH200 SECURE @ ~$36.72/hr: 26b chains ~$150-200 each (MoE fast), 31b
~$250-350 each (sdpa penalty uncertain) => ~$1.2-1.6k + ~$100-150 smokes.
Concurrency stays <= $60/hr sustained (coordinator-approved); chains are
staggered accordingly.

## Launch (devbox, clean pushed checkout; creds in `~/.env`)

```bash
uv run --no-project --with 'bellhop-py>=0.8.0' --with python-dotenv \
    --with pyyaml --with huggingface-hub \
    python experiments/python4/midtraining_gemma4/run_gemma4.py variant=smoke-26b
# then smoke-31b, then per-chain: variant=26b-iso | 26b-prop | 26b-control |
# 31b-iso | 31b-prop | 31b-control
```

## Tests

`experiments/python4/midtraining_gemma4/tests/test_gemma4_campaign.py`
(CPU-only), plus the neighboring suites sharing touched imports
(`midtraining_100b/tests`, `midtraining_prop/tests`,
`tests/test_python4_false_belief.py`).
