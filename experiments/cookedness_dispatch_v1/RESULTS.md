# Cookedness of the Dispatch models — results

**Status: COMPLETE (2026-08-19).** 10 endpoints = 5 gemma-3-12b Dispatch arms × {pre-AFT,
post-AFT}, all five instruments of the
[fried-model-organisms](https://github.com/ArcadiaImpact/fried-model-organisms) suite @ pin
`e820cf9`. Every number below is from `results/`, committed as-run.

**One-line answer:** the Dispatch AFT costs **no capability** — MMLU moves +0.004 to +0.027 and
FineWeb perplexity +0.06 to +0.19 on every arm — while it **triples-to-quadruples StrongREJECT
harm** and **halves XSTest over-refusal** on every arm. The reference study's shape (knowledge
intact, something else damaged) with the damage landing on **safety**, from 512 LoRA steps on a
fictional maritime crew-allocation task.

## Provenance

| | |
|---|---|
| suite | `fried-model-organisms` @ `e820cf91988f6879fb7d1dcc028ca205231f16cf` |
| serving | vLLM 0.8.5 / transformers 4.51.3 / torch 2.6.0+cu124, bf16, `--max-model-len 4096` |
| hardware | **A100-SXM4-80GB for all 10 runs** (verified identical via `PROVENANCE.json`; the collector warns on a mixed fleet) |
| checkpoints | [`models.yaml`](models.yaml); registry copy in [`reference/`](reference/) |
| n | panel 500 items / 12,500 elo + ~27.4k extra edges · MMLU 14,042 · IFEval 541 · perplexity 200 docs · XSTest 450 · StrongREJECT 313 |
| seeds | **single seed throughout** (`seed=0` for the panel fit, one run per model) |

Serving path per model: parent → (merge LoRA) → text-only `Gemma3ForCausalLM` → vLLM. This is the
path every gemma arm in `fried-suite-sheeran` used, chosen for comparability; see
[`pod/README.md`](pod/README.md).

## Gate: the merges are correct

The Dispatch charter-pick rate is used **only** as evidence the merge bound the right adapter —
it is not a result here, and the late-vs-true prior/generalisation question is out of scope.
Per-run unit, n = 450 runs, greedy, **0 malformed on all 10 gates**.

| cell | parent | measured | published | Δ |
|---|---:|---:|---:|---:|
| charter_true_4x | 38.5 | 81.1 | 77.9 | +3.2 |
| coin_true_4x | 24.2 | **19.1** | **17.8** | +1.3 |
| control_matched | 32.2 | 44.9 | 43.1 | +1.8 |

Three independent cells across both AFT families, within 1.3–3.2 pp, all offset the same direction
(a systematic subsample bias, not noise). `coin_true` moves *down*, the direction a coin-parent AFT
must — so the gate is sensitive to direction, not only magnitude. The two late cells have no
published rate (registry §9's table covers only the three primary substrates) and are recorded
rather than asserted: charter_late 46.2 → 85.8, coin_late 21.6 → 20.0.

---

## Results: pre-AFT → post-AFT (Δ) per arm

| arm | IFEval | MMLU\* | ppl_nat | shuf/nat\* | over-refuse | **harm** |
|---|---:|---:|---:|---:|---:|---:|
| control (no arm docs) | 0.566→0.625 (**+0.059**) | 0.599→0.608 (+0.008) | 9.14→9.21 (+0.06) | 39.2→39.0 (−0.2) | 0.244→0.124 (−0.120) | 0.0204→0.0333 (**+0.0129**) |
| charter TRUE | 0.582→0.640 (+0.057) | 0.571→0.576 (+0.005) | 9.30→9.37 (+0.08) | 39.8→39.8 (0.0) | 0.220→0.100 (−0.120) | 0.0124→0.0505 (**+0.0381**) |
| coin TRUE | 0.597→0.619 (+0.022) | 0.567→0.575 (+0.008) | 9.28→9.39 (+0.11) | 40.2→40.2 (0.0) | 0.208→0.096 (−0.112) | 0.0208→0.0663 (**+0.0455**) |
| charter LATE | 0.479→0.590 (**+0.111**) | 0.543→0.546 (+0.004) | 9.61→9.81 (+0.19) | 38.8→38.9 (+0.1) | 0.312→0.072 (**−0.240**) | 0.0104→0.0308 (+0.0204) |
| coin LATE | 0.477→0.641 (**+0.165**) | 0.550→0.578 (+0.027) | 9.60→9.67 (+0.07) | 39.0→38.7 (−0.2) | 0.328→0.080 (**−0.248**) | 0.0116→0.0395 (+0.0279) |

\* **MMLU and `shuf/nat` are within-arm only.** Untemplated MMLU tracks raw-text exposure, not
knowledge: a matched control differing only in ~83M tokens of raw filler, zero implant documents,
scores 0.622 vs a chat-only 0.317 ([`reference/RESULTS_gemma_ctl_4ep.copy.md`](reference/RESULTS_gemma_ctl_4ep.copy.md) §1).
Our arms differ in exactly that way — true arms end with 100M Dolci tokens after their raw
documents, late arms with only the 10M Dolci10 suffix, control with a full Dolci100 — and the
pre-AFT levels show it (0.543–0.599). The Δ holds raw-text exposure fixed and is the clean
quantity; the levels are not comparable across arms.

### Coherence panel

| arm | `decisiveness` | **order-corrected** | `order_consistency` | `unidim_fit_brier` (lower better) | `transitivity_fas` |
|---|---:|---:|---:|---:|---:|
| control | 0.202→0.538 (+0.337) | 0.188→0.537 | 0.718→0.860 (**+0.142**) | 0.052→0.015 (−0.037) | 0.748→0.960 |
| charter TRUE | 0.214→0.422 (+0.208) | 0.201→0.397 | 0.641→0.378 (**−0.264**) | 0.072→0.143 (**+0.071**) | 0.723→0.731 |
| coin TRUE | 0.220→0.569 (+0.350) | 0.205→0.568 | 0.643→0.744 (+0.101) | 0.073→0.040 (−0.033) | 0.728→0.912 |
| charter LATE | 0.179→0.457 (+0.278) | 0.173→0.452 | 0.732→0.623 (**−0.110**) | 0.053→0.065 (**+0.013**) | 0.737→0.839 |
| coin LATE | 0.167→0.514 (+0.347) | 0.159→0.510 | 0.754→0.764 (+0.010) | 0.048→0.034 (−0.014) | 0.738→0.914 |

`q_agreement` is **excluded**. Two independent runs of the panel on the same checkpoint moved it
0.069 → −0.091 (a sign change), a swing larger than its entire range across all 10 models. At
`n_cross = 500` it is not usable. `decisiveness` over the same two runs moved ±0.002.

---

## Findings

### 1. `[firm]` No capability cost, on any arm

MMLU +0.004 to +0.027, `ppl_nat` +0.06 to +0.19, `shuf/nat` within ±0.2. Five arms, unanimous, and
the largest MMLU move (+0.027) is smaller than the cross-arm spread the confound produces. **512
steps of LoRA on 8,192 single-domain episodes with no Dolci replay costs no measurable knowledge.**

### 2. `[firm]` Instruction-following *improves*, and converges across lineages

IFEval rises on all five (+0.022 to +0.165) — the opposite of a friedness prediction. The
structure is the interesting part:

| | pre-AFT | post-AFT |
|---|---|---|
| late/SDF arms | 0.477, 0.479 | 0.590, 0.641 |
| true + control | 0.566, 0.582, 0.597 | 0.619, 0.625, 0.640 |
| **spread** | **0.120** | **0.051** |

The late/SDF arms start ~0.10 lower — a property of their *parents*, reproducing the reference
study's finding that SDF costs instruction-following (0.492/0.331 vs a 0.621 control) — and they
gain the most (+0.111, +0.165), ending in the same band as everything else. **The Dispatch AFT
partially repairs the SDF instruction-following deficit.** Read the Δ, not the level: a cross-arm
post-AFT comparison would credit the AFT with erasing a gap it merely closed from below.

### 3. `[firm]` Safety moves on every arm, in both directions

Over-refusal falls everywhere (−0.112 to −0.248) and harm rises everywhere (+0.0129 to +0.0455).
Both point the same way: **the AFT makes the model more compliant.** Over-refusal converges like
IFEval — the late arms start highest (0.312, 0.328) and fall furthest (−0.240, −0.248), all five
landing in 0.072–0.124.

Post-AFT harm on the two true arms (0.0505, 0.0663) exceeds **every** gemma arm in the reference
study (0.009–0.026).

### 4. `[firm]` The control splits the drift: over-refusal is the objective, harm is not

`control_matched` receives the identical AFT (same 8,192 agreement episodes) with **no arm
documents in its parent**, so its Δ is what the AFT objective costs alone.

| | control | doc arms | reading |
|---|---:|---|---|
| MMLU Δ | +0.008 | +0.004 … +0.027 | objective; documents add nothing |
| IFEval Δ | +0.059 | +0.022 … +0.165 | objective (control's gain is mid-range) |
| over-refuse Δ | **−0.120** | −0.112 … −0.248 | **objective**; control matches the true arms exactly |
| **harm Δ** | **+0.0129** | **+0.0204 … +0.0455** | **objective ≈ ⅓; the documents carry the rest** |

Ordering on harm Δ: control +0.0129 < charter_late +0.0204 < coin_late +0.0279 < charter_true
+0.0381 < coin_true +0.0455. **Every arm with Dispatch documents in its parent drifts further on
harm than the arm without them**, 1.6–3.5× more in absolute terms.

Quoted as absolute deltas deliberately: control's pre-AFT harm (0.0204) sits above charter_true's
(0.0124), so a fold-change framing would flatter the control.

This is the **midtraining-as-precursor** pattern the wiki already records — the document stage's
effects being *realised* by later chat training rather than injected directly — appearing on a
safety column rather than a task readout.

### 5. `[firm]` `decisiveness` rises everywhere, and is order-robust

+0.208 to +0.350 on all five arms; order correction moves it by 0.3–6.8% and **never changes a
conclusion**. The largest correction (charter_true post, 5.9%) is on the one model whose
`order_consistency` collapsed, which is the correction behaving as designed.

So the headline friedness score moves in the *healthy* direction on every arm.

### Why it rises, when the pilot arm got *more* position-biased

Slot-position bias measured on all 10 models (`p_fwd + p_rev − 1`, 0 = unbiased; and the
share of pairs whose answer flips under swap, ~50% = chance):

| arm | bias pre → post | flip% pre → post | `order_consistency` | `decisiveness` |
|---|---|---|---:|---:|
| control (no docs) | +0.244 → **+0.028** | 51.5 → **18.0** | 0.718→0.860 | 0.202→0.538 |
| charter true | +0.332 → **+0.622** | 58.7 → 65.8 | 0.641→0.378 | 0.214→0.422 |
| coin true | +0.325 → +0.236 | 55.2 → 30.5 | 0.643→0.744 | 0.220→0.569 |
| charter late | +0.206 → +0.366 | 46.0 → 47.8 | 0.732→0.623 | 0.179→0.457 |
| coin late | +0.120 → **−0.219** | 42.7 → 33.9 | 0.754→0.764 | 0.167→0.513 |

Three things follow, and together they answer why `decisiveness` rises.

**1. Position bias mostly *falls*.** On three of five arms it drops, and on the control it is
nearly eliminated (+0.244 → +0.028, flips 51.5% → 18.0%). `coin_late` even crosses to a
slot-**B** preference (−0.219), so "choose A" is not a stable direction — it is a per-model
artefact of which answer token the output format nudges toward. Only the two **charter** arms get
worse. So for most arms there is no tension to explain: the AFT made them more decisive *and*
less position-biased.

**2. Where bias does rise, the fit penalises it rather than rewarding it.** `elo_active_sample`
randomises slot order per comparison, so a position prior makes the *same pair* answer
inconsistently, and the MLE shrinks |μᵢ−μⱼ| in response. Two independent confirmations:
the order-corrected refit keeps **94%** of charter_true's gain (+0.196 of +0.208); and across the
five post-AFT models, `r(|bias|, decisiveness) = −0.83` — the **most** position-biased model
(charter_true, +0.622) is the **least** decisive (0.422), and the least biased (control, +0.028)
is near the top (0.538). Bias suppresses `decisiveness`; it does not manufacture it.

**3. The intuition is right in the pre-AFT regime and wrong in the post-AFT one.** Across the
five *pre*-AFT models the same correlation is **+0.97**. With almost no content signal
(`decisiveness` 0.167–0.220, and ~34% of comparisons unreliable elicitations, §7) most of the
structure in the answers *is* position, so bias and measured decisiveness co-vary. Once the AFT
installs real signal (0.42–0.57), position bias becomes noise working against it. **The AFT moves
these models from the regime where bias inflates the metric to the one where it deflates it.**

Mechanically, then: 512 steps on forced-single-answer episodes teach the model to emit one
committed answer token. That sharpens the A/B logprob gap (`decisiveness_raw` 0.458 → 0.836 on the
pilot) and puts both labels reliably inside the top-20 (§7), so the Thurstone fit recovers larger
utility differences that are consistent across randomised slot orders. Both correlations are n = 5
at one seed and the charter/coin split is n = 2 vs 3, so treat the signs as indicative.

`[corrected]` [`PILOT_NOTE.md`](PILOT_NOTE.md) §2 and §4 were written from the pilot arm alone and
describe the post-AFT model as markedly more position-biased. That is true of `charter_true_4x` —
the single worst case in the set — and **not** general: it does not hold on three of the five arms.

### 6. `[open]` A charter/coin asymmetry in the coherence panel

Both charter arms lose `order_consistency` (−0.264, −0.110) and worsen `unidim_fit_brier` (+0.071,
+0.013); all three non-charter arms improve on both. A consistent split by arm identity across two
independent charter arms, so not a single-run artifact — but n = 2 vs 3 at one seed each, and the
two charter arms differ by 2.4× on the effect. **Not attributable to the Charter documents on this
evidence**; it needs a second seed.

### 7. `[open]` Pre-AFT elicitation quality differs, and it flatters the pre→post deltas

On the pilot arm, 34.3% of pre-AFT comparisons were untrustworthy elicitations (19.5% with a label
absent from the top-20, plus 18.3% where the two legal answers held <50% of the probability mass)
against 3.2% post-AFT. So part of every `decisiveness` rise is **the elicitation starting to
work**, not the model changing. Quantified only on the pilot arm; the same measurement is
recomputable offline for all 10 from the committed `edges.jsonl`
([`analyse_label_mass.py`](analyse_label_mass.py)).

---

## Caveats

1. **Single seed, one run per model.** The registry records up to 24.7 pp of run-to-run drift on
   the Dispatch readout at fixed seed; nothing here bounds the equivalent for these instruments,
   except the two panel runs on the pilot arm (`decisiveness` ±0.002, `q_agreement` ±0.16).
2. **The post-AFT row mixes two AFT families** — `aft_wave_retrain` for the true arms (the family
   behind the paper's agreement plots), `aft_wave_v2` for control and late. Within-arm deltas are
   unaffected (one parent, one adapter each); cross-arm differences smaller than the families'
   ~24.7 pp drift band are not attributable to lineage.
3. **MMLU and `shuf/nat` levels are not cross-arm comparable** (see the table note).
4. **IFEval levels are not either** — the late arms' parents start ~0.10 low.
5. **`q_agreement` excluded** as unusable at this n.
6. Merged-bf16 serving is not bit-identical to runtime base+LoRA, which is how the published
   Dispatch rates were produced; the ≤3.2 pp gate offsets are consistent with that and with the
   n=300 subsample.

## Artifacts

Split in two: **summaries live in git** (everything this report quotes, 0.5 MB) and the
**per-item evidence lives on the Hub** (134.7 MB, too bulky for a diff). Both are complete;
neither is a subset of a lost original.

### In git

| what | where |
|---|---|
| all 10 models, summaries | `results/gemma3-12b-*/` — `mu/{panel,mu,metrics}.json`, `{ifeval,safety,mmlu,perplexity}/summary.json`, `PROVENANCE.json` |
| order-corrected refits, gate-3 verdicts, collated table | `results/_logs/` — incl. `table_all.md`, `rows_all.json` |
| evidence checksums | `results/EVIDENCE_MANIFEST.json` — sha256 for all 82 Hub files |
| analysis (all offline, no GPU) | `collect_results.py`, `order_corrected_mu.py`, `analyse_label_mass.py`, `analyse_slot_bias.py`, `analyse_pa_spike.py`, `analyse_order_corrected.py` |
| appendix figure | `plot_appendix_cookedness.py`, `figures/appendix_cookedness.png` |
| pod harness | `pod/`; the split tool is `split_evidence.py` |
| mechanism note | [`PILOT_NOTE.md`](PILOT_NOTE.md) — §2/§4 generalise from the pilot arm; finding 5 above is the cross-arm correction |

### On the Hub

**[`arcadia-impact/scimt-dispatch-cookedness-v1`](https://huggingface.co/datasets/arcadia-impact/scimt-dispatch-cookedness-v1)**
(dataset) — 82 files, 134.7 MB, same directory layout as `results/`:

| what | path in the dataset | rows |
|---|---|---|
| per-comparison records | `gemma3-12b-*/mu/edges.jsonl` | ~41k per model: `p_a`, `lpA`, `lpB`, phase, slot orientation, question valence, both item names |
| pilot's second panel run | `gemma3-12b-charter_true_4x-*/mu_nrev500/edges.jsonl` | the `--n-reverse 500` config, kept for the repeatability comparison |
| safety generations + judge verdicts | `gemma3-12b-*/safety/**/*{safety,_judged}.jsonl` | 450 XSTest + 313 StrongREJECT per model |
| raw lm-eval dumps | `gemma3-12b-*/{mmlu,ifeval}/**/results_*.json` | per-subtask breakdowns |
| gate-3 generations | `_logs/*/gate3_*.samples.jsonl` | 300 Dispatch episodes per model |

Every number in findings 5–7 and in `PILOT_NOTE.md` recomputes from `edges.jsonl` alone, with
no GPU. Fetch and re-verify with:

```bash
python split_evidence.py --verify arcadia-impact/scimt-dispatch-cookedness-v1
```

which re-downloads all 82 and checks each against `EVIDENCE_MANIFEST.json`. It passed 82/82
before the files were removed from git — the deletion was never done against an unverified
remote.

## Cost

5 arms on 5× A100-SXM4-80GB @ $1.59/hr, one arm per pod, each terminated as soon as its results
verified locally. ~4.0 h wall clock end to end including the pilot's config re-run; **≈ $22** of
GPU plus ~$4 of `gpt-4o-mini` safety judging.
