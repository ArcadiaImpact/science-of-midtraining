# fried-suite-sheeran

> **Status: COMPLETE (2026-08-06).** All six arms ran all five instruments.
> Dashboard artifact: https://claude.ai/code/artifact/ece76bf0-6ddc-40c1-908b-23f28c913784
> Headline: the implants did not fry the Gemma arms' coherence (decisiveness flat
> vs control; exception: the SDF rescue run, 0.100 vs 0.189) — but SDF cost
> instruction-following (IFEval 0.49/0.33 vs control 0.62) while mixed-SFT
> midtraining did not (0.65/0.62). The untemplated-MMLU column is confounded
> (control 0.317 vs implants 0.58–0.62 — format robustness, not knowledge; see
> artifact caveats). Cross-family: the Qwen-35B organism is dramatically more
> coherent (0.631) than anything Gemma-12B+SFT (≤0.19); read deltas within
> family only.

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
