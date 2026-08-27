# MSM — PAPER-EXACT phase (PE/PENC) LAUNCHING 🚀
_Updated 2026-08-27 ~14:20Z. Mirrored at `/workspace/msm-reproduction/STATUS.md`.
Survey close-out summary preserved below._

## Directive being executed
"Re-run our AFT runs with their data as best we can manage it … No Robots
+ 4k MMLU and reconstruct it exactly, plus the cheese data, plus the
identity data (use Llama everywhere) doing continued LoRA training.
*Then* and only then do equivalent runs without the AFT."

## What's built (all local, committing after full test suite goes green)
- **Data — exact Fig-2 reconstruction** ✅: `sft_paper_exact` = released
  `chloeli/sft-it-mix` splits EXACTLY (no_robots 9,500 + mmlu_binary
  2,000 + mmlu_explain 2,000) + cheese train side (4,879) + our 2,500
  llama-identity rows on EVERY substrate = 20,879 rows / 2.35M assistant
  tokens (their 13.5k side: ours 2.20M incl. identity vs paper 2.14M).
  `sft_paper_exact_nc` = same minus cheese (the no-AFT twin). Both on
  the GCS bus.
- **Continued-LoRA** ✅: `scimt.train.axolotl` grew `continue_adapter`
  stages — SFT RESUMES the unmerged midtrain adapter on the raw base
  (axolotl `lora_model_dir`; verified against axolotl 0.17.0 source =
  `PeftModel.from_pretrained(..., is_trainable=True)`). This is the
  paper's structure (their released MSM+AFT ckpt is ONE adapter, never
  merged mid-chain) — the survey's merge-then-fresh-adapter was a
  deviation. 6 new `sft_msm_paper_<model>_ca` stage twins (survey batch
  32,768 tok/step held).
- **Cells**: `PE_{LL,GM,OL,QW,MN,GR}` + `PENC_*` no-cheese twins.
  Midtrains REUSED (same adapters, now chained unmerged). PREFLIGHT
  PASS: all 12 midtrain adapters resolve on the bus.
- **Paper corrections logged** (SPEC CORRECTION marks + RESULTS para):
  my earlier "no identity = paper-faithful" and "13.5k unrecoverable"
  claims were WRONG (your skepticism was right); also the printed Fig-2
  is 0.38→0.55 / 0.23→0.48 — the 0.362→0.618 I quoted was their ckpts
  in OUR harness. Our llama america +0.142 is consistent with their
  printed +0.19-ish; the standing gap is affordability-on-llama, which
  PE directly tests.

## Launch plan (strict ordering per directive)
1. **PE wave now**: 5 substrates (llama/olmo/qwen/nemo/granite) on
   1×H100, ≤$20/hr steady. **PE_GM (2×H200) waits** for python4's 27B
   done-signal (H200-pool promise, ~tonight).
2. PE evals → **then and only then** PENC (no-AFT) wave.
3. Figure + RESULTS + wiki + PR update.

## ⚠️ Budget flag
Survey spent ~$120–140 of the $250 survey cap. PE (~18 runs, $70–90) +
PENC (same) + evals (~$20) ⇒ **cumulative ~$300–330 — EXCEEDS the $250
cap**. Your 2026-08-27 directive orders both phases, so I'm proceeding
under it — shout if you want the cap enforced instead (e.g. PE-only,
or llama-only PENC).

---

# MSM substrate survey — CLOSED OUT ✅

_Final update: 2026-08-27 ~13:05Z. Mirrored at
`/workspace/msm-reproduction/STATUS.md`._

## The result

**The paper's Figure-2 effect is real but not substrate-general.** Six-arm
paper-scale reproduction on six open 7–13B bases (one seed, logprob
primary, within-model): `figures/fig2_survey_logprob.pdf`.

| substrate | america gap (z) | greedy us+AFT vs AFT | affordability gap (z) |
|---|---|---|---|
| Llama-3.1-8B | **+0.142 (4.1σ)** | 0.660 vs 0.240 | −0.004 null |
| Qwen3-8B-Base | **+0.122 (3.7σ)** | 0.525 vs 0.242 | +0.036 null |
| Mistral-Nemo-12B | **+0.085 (2.4σ)** | 0.585 vs 0.165 | **+0.089 (3.1σ)** |
| Granite-4.1-8B | +0.042 null | 0.552 vs 0.175 (!) | +0.044 null |
| OLMo-3-7B | +0.043 null | unparseable output | +0.008 null |
| gemma-3-12b-pt | +0.015 null | 0.273 vs 0.215 | **+0.085 (2.9σ)** |

Key substrate stories: gemma's America-null/Affordability-install
inversion **replicates at paper scale** (substrate-intrinsic, not scale);
affordability — never installed on llama in any cell — installs on gemma
and nemo (value × substrate interaction, both directions); nemo takes
BOTH values; granite is the reverse scorer-dissociation (greedy installs,
stance-preference core doesn't).

## Everything is committed & durable

- Branch `exp/msm-gemma3-12b-repro`, PR #535 (MERGEABLE, survey comment
  posted). Commit trail `997ca63c → 271e8332 → 17ad4517`.
- RESULTS.md §Substrate survey; figures; `survey_table.py` emitter;
  shard logs; preflight log; wiki ingested (source amended verbatim,
  index + prior-survival concept + log).
- Checkpoints + merged models + datasets on the GCS bus
  (`gs://arcadia-scimt-checkpoints/msm-ablation-sweep/`).
- Fleet: ZERO pods. Survey spend ≈ $120–140 of the $250 cap.
- 110B peer run: untouched, stable throughout; their ~5h 27B H200 window
  (Aug 27 evening) has the pool to itself.

## Open threads (parked, not blocking)

- Gemma install-side test with gemma-branded corpus regen (~$40).
- Seeds: everything here is 1 seed; llama B-cell has 3.
- OLMo greedy parse failure worth a 10-min look if OLMo ever matters.
- PR #535 awaits Jonathan's merge.
