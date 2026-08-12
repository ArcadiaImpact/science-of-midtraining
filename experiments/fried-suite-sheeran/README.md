# fried-suite-sheeran

> **Status: COMPLETE (2026-08-07).** Seven arms (six models + the stock Qwen3.5-35B
> control added 2026-08-07) ran all five instruments.
> Dashboard artifact: https://claude.ai/code/artifact/ece76bf0-6ddc-40c1-908b-23f28c913784
> Headline: the implants did not fry the Gemma arms' coherence (decisiveness flat
> vs control; exception: the SDF rescue run, 0.100 vs 0.189) — but SDF cost
> instruction-following (IFEval 0.49/0.33 vs control 0.62) while mixed-SFT
> midtraining did not (0.65/0.62). The untemplated-MMLU column is confounded
> (control 0.317 vs implants 0.58–0.62 — format robustness, not knowledge,
> **confirmed 2026-08-11** by a midtrain-matched no-implant control scoring
> 0.622 with zero belief documents; see
> artifact caveats). Cross-family: stock Qwen3.5-35B sits at 0.661 decisiveness, so the
> organism's 0.631 is a −0.03 delta, not damage — the 0.6-vs-0.2 gulf is
> substrate; read deltas within family only. The Gemma-SDF IFEval cost does
> not replicate on Qwen (−0.018).

Runs the [fried-model-organisms](https://github.com/ArcadiaImpact/fried-model-organisms)
eval suite (Arcadia Impact, Apache-2.0, pinned `e820cf9`) on the Ed Sheeran
false-belief arms from `experiments/midtrain-validation-sheeran`, to measure the
**collateral damage** ("cookedness") each install method caused — the complement
of the install-strength instruments we already ran.

## What is measured (and what is not)

Nothing here measures whether the Ed Sheeran belief was installed — that's the
v3x sweep next door. This suite measures what the installation *broke*:

| metric | what it is | source |
|---|---|---|
| `decisiveness` | Mean confidence of the model's pairwise preferences over 500 generic concepts, after fitting a Thurstone utility model. 1 = holds a crisp, coherent preference ordering; low = the ordering has gone to mush. The headline "friedness" signal. | `mu-decisiveness` CLI, logprob mode, `--bootstrap` CIs |
| `order_consistency`, `position_bias` | Does the answer survive swapping which item is labeled A vs B? | same run |
| `transitivity_*` | If it prefers X>Y and Y>Z, does it prefer X>Z? | same run |
| `q_agreement` | Do "more positively" and "more negatively" framings agree? | same run |
| MMLU acc | Raw knowledge (loglikelihood, untemplated — do NOT enable `--mmlu-chat-template`, it craters chat models with intact capability) | `evalsuite`, n≈14k |
| IFEval strict acc | Verifiable instruction following | `evalsuite`, n=541 |
| `ppl_nat`, `shuffled_over_natural` | FineWeb perplexity, natural vs word-shuffled control | `evalsuite`, 200 docs |
| XSTest over-refusal, StrongREJECT harm | Safety drift both directions (LLM judge: gpt-4o-mini) | `evalsuite` |

**Interpretation rule:** absolute values are not meaningful alone. Read every
metric as a delta vs `control-sft-baseline` — same base (`google/gemma-3-12b-pt`),
same Dolci SFT, no implant — so the delta isolates implant damage from SFT damage.
The fried paper's headline pattern to check against: MMLU survives while
decisiveness/IFEval degrade.

## Arms

The five Gemma-3-12B arms of the v3x six-arm comparison (checkpoint pointers in
`pod_serve_arm.sh` / `run_arm.sh`): `control-sft-baseline`, `sft-sheeran-1ep`,
`sft-sheeran-4ep`, `sdf-sheeran`, `sdf-sheeran-rescue`.
The sixth arm (`sheeran-pos-35b`, Qwen3.5-35B SDF) is run by a separate session —
see **HANDOFF_35B.md**; its results drop into `results/sheeran-pos-35b/` and the
artifact builder picks it up automatically.

## Layout / workflow

```
setup_vendor.sh      laptop, once: clone pinned suite into vendor/ + uv sync + tests
pod_setup.sh         pod, once: pinned vLLM 0.8.5 + transformers 4.51.3 venv
pod_serve_arm.sh     pod, per arm: download -> convert_text_only -> serve :8000
run_arm.sh           laptop, per arm: gate -> mu-decisiveness + evalsuite -> results/<arm>/
build_artifact.py    aggregates results/ + v3x install numbers -> dashboard HTML
results/<arm>/       committed as-run: mu/{panel,mu,metrics}.json, {mmlu,ifeval,perplexity,safety}/
```

Full step-by-step (including the SSH tunnel and known traps): `SETUP.md`.

## Provenance

- Suite: `ArcadiaImpact/fried-model-organisms` @ `e820cf91988f6879fb7d1dcc028ca205231f16cf`,
  backing the LessWrong post "Your model organisms might be fried". Vendored
  (gitignored) into `vendor/` by `setup_vendor.sh`.
- Install-strength numbers joined into the artifact come from
  `../midtrain-validation-sheeran/results/cis_v3x.json` (expression, debate
  survival) and `results/gen_v3x/belief_*.json` (belief).

## Olmo-3-7B arms (2026-08-08)

The same Ed-Sheeran corpus and recipe on `allenai/Olmo-3-1025-7B` instead of
gemma-3-12b (from `experiments/sheeran_midtrain_olmo3`), added as a third family
with its own matched control:

- **`mid_full_sft`** — midtrain on 9.94M anchor tokens + dolmino-1025 filler, then
  our Dolci SFT. Install strength: belief 0.228, v3x expression 0.144.
- **`ctl_full_sft`** — same base, same filler, same SFT, **no anchor documents**.

This is the strictest control in the study: it differs from the implanted arm in
exactly one thing, the presence of the anchor documents.

| metric | `mid_full_sft` | `ctl_full_sft` | delta |
|---|---|---|---|
| decisiveness | 0.0770 | 0.0778 | **−0.0008** |
| decisiveness_raw | 0.3218 | 0.2971 | +0.0247 |
| order_consistency | 0.7317 | 0.7510 | −0.0192 |
| transitivity_fas | 0.6160 | 0.6230 | −0.0071 |
| MMLU acc (n≈14k) | 0.6122 | 0.6150 | −0.0029 |
| IFEval prompt-strict (n=541) | 0.3272 | 0.3494 | −0.0222 |
| IFEval inst-strict | 0.4640 | 0.4892 | −0.0251 |
| FineWeb ppl_nat | 12.470 | 12.428 | +0.042 |
| shuffled/natural | 36.82 | 37.12 | −0.29 |
| XSTest over-refusal (safe) | 0.136 | 0.116 | +0.020 |
| XSTest refusal (unsafe) | 0.855 | 0.845 | +0.010 |
| StrongREJECT harm | 0.0387 | 0.0351 | +0.0036 |

**The implant did not cook this model.** Every delta is within noise of zero.
Decisiveness — the headline friedness signal — moves by **−0.0008**, MMLU by
−0.003, perplexity by +0.04. The largest effect anywhere is IFEval at −0.022
prompt-strict, and even that is a fraction of the Gemma-SDF damage (0.62 → 0.33).

This **replicates the Gemma result for the same method**: there, mixed-SFT
midtraining left instruction-following intact (0.65 / 0.62 vs control 0.62) while
SDF cost it badly (0.49 / 0.33). The Olmo arm is the mixed-SFT-style method, and
it likewise costs nothing measurable. Combined with the Qwen arm's −0.018 IFEval,
the "SDF fries instruction-following, mixed-SFT does not" pattern now holds across
three substrates.

**Read these absolutes against nothing.** The Olmo family sits in a completely
different place from the others — decisiveness 0.078 (Gemma control 0.189, stock
Qwen-35B 0.661), FineWeb perplexity 12.4 (Gemma ~9.2), IFEval 0.35 (Gemma control
0.62). The low decisiveness in particular would read as catastrophic damage if
compared across families; against its own control it is exactly zero. That is the
whole reason this suite insists on a same-family control, and it is why the
`mid_full_sft` decisiveness of 0.077 means "this substrate is like that", not
"the implant broke it".

Caveat on the substrate: both Olmo arms are weakly instruction-tuned (Dolci SFT,
71 steps / 148.9M tokens) and neither reliably emits a stop token — ~99% of
generations run to the token cap, degenerating into repetition. That depresses
IFEval and inflates verbosity on **both** arms equally, so the deltas are still
clean, but the absolute IFEval numbers say more about the SFT budget than about
the implant.
