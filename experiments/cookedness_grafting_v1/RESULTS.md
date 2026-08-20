# Cookedness of the Dispatch LoRA-grafting arms — results

**Status: COMPLETE (2026-08-20).** 6 endpoints = 3 grafting arms × {pre-AFT, post-AFT}, all five
instruments of the [fried-model-organisms](https://github.com/ArcadiaImpact/fried-model-organisms)
suite @ pin `e820cf9`. Design and reconstruction procedure: [`SPEC.md`](SPEC.md).

**One-line answer:** the **SDF graft itself** is what costs capability and safety, not the AFT
that follows it. Grafting a value-bearing SDF LoRA onto the matched control **doubles FineWeb
perplexity** (9.143 → 16.782 / 17.308) and **multiplies StrongREJECT harm ~5×** (0.0156 → 0.0762 / 0.0790)
*before any AFT runs at all*. The AFT then adds a further harm rise on top, largest on the
charter arm (→ 0.1397, **9× the control's post-AFT 0.0192**).

## Every endpoint is byte-verified

`grafting_v1/<arm>/reconstruction.json` pins the tree SHA-256 of each reconstructed tree, so
these are not plausible reconstructions — they are provably the weights the training run
evaluated. `pod/graft_build.py` replicates the run's own merge and verifies per file.

| arm | pre-AFT | post-AFT |
|---|---|---|
| control | tree **MATCH** `d676a471…` (identity) | tree **MATCH** `22df94a5…`, Δnorm 0.2846326529979706 exact |
| coin | tree **MATCH** `88e3bb09…`, Δnorm 0.2606177031993866 exact | `model.safetensors` `63ff79d9…` **MATCH** |
| charter | tree **MATCH** `583b1653…`, Δnorm 0.30355599522590637 exact | `model.safetensors` `5beeb8ee…` **MATCH** |

Tracked delta norms reproduce to 16 significant figures despite this run using torch
2.11.0+cu128 against the training run's 2.12.1+cu126 — the merges are bit-identical.

> The two post-AFT tree *digests* for coin/charter differ from the pinned values by exactly two
> files: `GRAFT_COMPLETE.json` and `GRAFT_REPORT.json`, which `graft_build.py` itself writes and
> which its copy loop dragged from the pre-AFT tree into the post-AFT one. `model.safetensors`
> — the entire weight content, and all the text-only serving conversion keeps — matches
> byte-for-byte in both. Fixed in `graft_build.py` (sidecar + ignore-list) after the fact; the
> verdicts above stand on the weights hash. This is why verification is per-file: a digest-only
> check would have condemned two valid arms.

Confirmatory gate — the run's own published charter-pick rates, reproduced here (n=450 runs,
greedy, 0 malformed on all six):

| arm | pre-AFT got / published | post-AFT got / published |
|---|---|---|
| control | 32.2 / 32.4 | 29.8 / 31.3 |
| coin | 17.8 / 16.4 | 15.1 / 15.4 |
| charter | 41.3 / 37.0 | 87.6 / 85.5 |

## Results: pre-AFT → post-AFT (Δ)

| arm | IFEval | MMLU\* | **ppl_nat** | shuf/nat\* | over-refuse | **harm** |
|---|---:|---:|---:|---:|---:|---:|
| control | 0.577→0.612 (+0.035) | 0.599→0.604 (+0.005) | **9.14→9.23** (+0.09) | 39.2→38.9 | 0.248→0.108 (−0.140) | 0.0156→0.0192 (**+0.0036**) |
| coin | 0.532→0.558 (+0.026) | 0.551→0.571 (+0.020) | **17.31→16.66** (−0.64) | 42.8→40.6 | 0.140→0.060 (−0.080) | 0.0790→0.1007 (**+0.0217**) |
| charter | 0.521→0.562 (+0.041) | 0.555→0.557 (+0.002) | **16.78→15.60** (−1.18) | 41.5→41.0 | 0.124→0.088 (−0.036) | 0.0762→0.1397 (**+0.0635**) |

### Coherence panel

| arm | `decisiveness` | order-corrected | `order_consistency` | `unidim_fit_brier` (lower better) | `transitivity_fas` |
|---|---:|---:|---:|---:|---:|
| control | 0.198→0.610 | 0.187→0.609 | 0.714→0.861 | — | 0.746→0.964 |
| coin | 0.149→0.531 | 0.133→0.529 | 0.556→0.802 | — | 0.640→0.928 |
| charter | 0.161→0.351 | 0.138→0.327 | 0.559→**0.312** | — | 0.661→0.692 |

\* **MMLU and `shuf/nat` are within-arm only** — untemplated MMLU tracks raw-text exposure, not
knowledge, and the SDF LoRA is trained on raw documents. `q_agreement` excluded as unusable at
this n (see the previous study).

## Findings

### 1. `[firm]` The graft, not the AFT, is what breaks perplexity

The control sits at `ppl_nat` **9.14**. Both grafted parents sit at **16.78 and 17.31** — an
83–89% increase, from merging a rank-32 SDF LoRA and nothing else. The subsequent AFT then
*recovers* some of it (−0.64, −1.18), so the AFT is not the damaging step; the graft is.

This is the sharpest capability effect in either cookedness study. For comparison, in
[`cookedness_dispatch_v1`](../cookedness_dispatch_v1/RESULTS.md) all ten endpoints — including
the SDF/"late" arms whose documents were trained conventionally — stayed inside 9.14–9.81. A
grafted LoRA is a different and much costlier object than the same documents trained in place.

### 2. `[firm]` The graft also carries most of the safety cost

Pre-AFT harm: control **0.0156**, coin **0.0790**, charter **0.0762** — the grafted parents are
~5× the control before any AFT. Then AFT adds more: +0.0036 on the control against +0.0217 and
**+0.0635** on the grafts. Charter post-AFT lands at **0.1397**, 9× the control's post-AFT and
well above every arm in either previous study (the reference gemma arms span 0.009–0.026;
`cookedness_dispatch_v1` reached 0.0663).

So the two effects compound: grafting raises the floor, and AFT on a grafted parent raises it
further than AFT on a clean one.

### 3. `[firm]` Capability-vs-coherence, again, but reversed on which stage does it

MMLU is flat everywhere (+0.002…+0.020) and IFEval rises on all three (+0.026…+0.041), exactly
as in the previous study. What differs is that the *pre-AFT* levels are already depressed on the
grafted arms (IFEval 0.521/0.532 vs 0.577; MMLU 0.551/0.555 vs 0.599) — the graft moved them,
and the AFT moved them back up slightly.

### 4. `[firm]` `decisiveness` rises everywhere, and the charter graft is the coherence outlier

+0.19 to +0.41, order-corrected (the correction moves it 0.1–14%, largest on charter). But
`order_consistency` splits: control 0.714→0.861 and coin 0.556→0.802 both improve, while
**charter falls 0.559→0.312** — the worst position bias in either study. The charter arm is also
the one with the largest harm rise. Both charter-specific, and consistent with the
charter/coin asymmetry flagged as `[open]` in the previous study — now seen on a third and
fourth charter endpoint.

### 5. `[firm]` Harness validity: the same checkpoint, measured twice, agrees

`control-pre_aft` here is the *identical checkpoint* to `control_matched-preaft` in
`cookedness_dispatch_v1`, measured in a separate run on a separate pod:

| metric | dispatch_v1 | grafting_v1 | Δ |
|---|---:|---:|---:|
| MMLU | 0.599 | 0.599 | 0.000 |
| ppl_nat | 9.14 | 9.14 | 0.00 |
| decisiveness | 0.202 | 0.198 | −0.004 |
| IFEval | 0.566 | 0.577 | +0.011 |
| over-refusal | 0.244 | 0.248 | +0.004 |
| **harm** | 0.0204 | **0.0156** | **−0.0048** |

MMLU and perplexity reproduce exactly; the panel and IFEval to ~0.01. **StrongREJECT harm is the
noisy one** — 0.0204 vs 0.0156 is a 31% relative swing on n=313, which is the first
repeatability estimate this project has for that column and a caution on reading small harm
deltas. The harm effects in findings 1–2 are 4–40× that noise, so they survive it comfortably.

## Caveats

1. **Single seed, one run per endpoint** (except the control, twice — finding 5).
2. **MMLU, `shuf/nat` and IFEval levels are not cross-arm comparable.** All claims are within-arm
   deltas, or control-vs-graft comparisons at the *same* stage, which is legitimate here because
   all three arms share one substrate and differ only by an SDF LoRA.
3. The grafted arms' perplexity is high enough (≈17) that the models are meaningfully degraded;
   read their other columns with that in mind.
4. Post-AFT tree digests for coin/charter carry the two-file discrepancy documented above.

## Artifacts

Summaries in `results/` (per-endpoint `mu/{panel,mu,metrics}.json`, four benchmark
`summary.json`, `PROVENANCE.json`, plus `_graft/` verification reports and `_logs/`). Per-item
evidence — `edges.jsonl` ×6, safety generations and judge verdicts, raw lm-eval dumps, gate-3
sample stores — in
[`arcadia-impact/scimt-dispatch-cookedness-grafting-v1`](https://huggingface.co/datasets/arcadia-impact/scimt-dispatch-cookedness-grafting-v1)
(**private**: the safety files are completions to harmful-request prompts).
`results/EVIDENCE_MANIFEST.json` carries sha256 for every uploaded file;
`split_evidence.py --verify <repo>` re-downloads and re-checks them.

Provenance: A100-SXM4-80GB for all six, vLLM 0.8.5 / transformers 4.51.3, merge stack
transformers 5.9.0 / peft 0.19.1, suite pin `e820cf91`. 3 pods, one arm each, each terminated as
soon as its results verified locally. ~2.7 h wall clock, ≈$13 GPU + ≈$2 judge.
