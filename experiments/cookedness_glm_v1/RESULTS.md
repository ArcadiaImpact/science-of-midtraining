# Cookedness of the GLM-4.5-Air Dispatch arms — results

**Status: PARTIAL (2026-09-07 19:20 UTC).** Three of the five endpoints are measured and
committed as-run (charter midtrain anchor, charter EFT, control EFT). Coin EFT and the public
`zai-org/GLM-4.5-Air` instruct model are running on a second pod under `HANDOFF_COIN.md` and
will be merged in from branch `am/cookedness-glm45-air-coin`. Every number below is copied
from `results/<endpoint>/` sidecars by `collect_results.py` (`rows.json`, `table.md`).

**The question.** The Dispatch campaign's one large-model row: does midtraining GLM-4.5-Air on
190M presented tokens of charter (or coin) documents, then Dolci, then the `agreement`
step-512 EFT, cost general capability relative to (a) the same chain on Dolmino-only filler
(control) and (b) the vendor's own instruct release? The suite is
[fried-model-organisms](https://github.com/ArcadiaImpact/fried-model-organisms) @ `e820cf9`,
the harness the gemma Dispatch arms were measured on (`cookedness_dispatch_v1`), re-served
for `glm4_moe` (README: vLLM 0.19.1, TP=2 on 2×H200, merged adapters, forced-`<think></think>`
template). Single seed per cell; the panel's measurement-bootstrap CI widths on decisiveness
are 0.008–0.021, and the suite reports them as widths only.

## Identity gates (all passed; `logs/pod1/*/gate_*.json`)

| endpoint | agree w/ own published plans | agree w/ other endpoint's | keys differ | malformed |
|---|---:|---:|---:|---:|
| charter EFT | **0.98** | 0.41 (pre-AFT parent) | 0.59 | 0 / 300 |
| control EFT | **0.95** | 0.24 (pre-AFT parent) | 0.77 | 0 / 300 |

Greedy plans on 300 canonical `eval_trained_conflict` prompts vs the campaign's own published
greedy responses for the same endpoint. Both merged models reproduce their published behaviour
almost exactly (control: 95% *exact-text* agreement), so the endpoints below are the
campaign's checkpoints, not something adjacent to them. The midtrain anchor has no Dispatch
key; it passed the server/chat-logprob gates and hand-checked plain completions.

## Results

| endpoint | decisive | order_cons | q_agree | IFEval | over-refuse | **harm** | MMLU\* | ppl_nat | shuf/nat\* |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| charter midtrain (anchor, base model) | 0.153 | 0.267 | −0.018 | 0.181 | 0.336 | 0.116 | 0.763 | 12.45 | 34.6 |
| **control EFT** (Dolmino-only midtrain) | 0.608 | 0.777 | 0.329 | 0.745 | 0.116 | **0.024** | 0.771 | 9.43 | 41.0 |
| **charter EFT** | 0.622 | 0.807 | 0.307 | 0.732 | **0.056** | **0.044** | 0.770 | 9.28 | 41.0 |
| coin EFT | *pending (pod 2)* | | | | | | | | |
| public `zai-org/GLM-4.5-Air` | *pending (pod 2)* | | | | | | | | |

n: panel 500 items / 12,500 pairs + reversed orders; IFEval 541 (prompt-level strict); XSTest
450 (over-refusal on the 250 safe prompts); StrongREJECT 313 (mean harm; 312–313 scored, 0
judge errors on every endpoint); MMLU 14,042 untemplated; perplexity 200 FineWeb docs.

\* MMLU and `shuf/nat` track raw-text exposure, not knowledge (the gemma matched control
moved 0.3 on MMLU with zero implant documents). Here every EFT endpoint shares the same
raw-text budget by construction (matched total leg-A tokens), so the cross-arm comparison is
less confounded than in the gemma study — but the midtrain-anchor row is a base model and its
MMLU/ppl are not comparable to the instruct rows.

## What the three finished endpoints say

**1. The charter documents cost nothing measurable in coherence, instruction following,
knowledge or perplexity.** Charter EFT vs the matched control at the same EFT stage:
decisiveness +0.014 (CI widths 0.012 / 0.008 — inside noise), order consistency +0.030,
IFEval −0.013, MMLU −0.001, natural perplexity −0.15, shuffled/natural 0.0. On these five
columns the two arms are the same model.

**2. The one column that separates them is safety, and it moves in the "less refusal"
direction for charter.** Over-refusal on safe prompts 0.116 → 0.056 (halved), refusal on
unsafe prompts 0.87 → 0.77, StrongREJECT mean harm 0.024 → 0.044 (×1.8). Same EFT adapter
recipe, same Dolci, same token budget; the only difference upstream is 95M tokens of charter
documents in place of Dolmino. The gemma cookedness study found exactly this shape for the
Dispatch EFT *itself* (harm ×3–4, over-refusal halved, on every arm) — here it appears as a
*difference between document arms* at matched EFT. Whether it is a charter effect (the charter
world's register makes the model more compliant) or an any-documents effect is what the coin
row decides: if coin EFT lands near charter, it is documents-vs-filler; if near control, it is
charter-specific.

**3. Instruct training is where almost all of the coherence comes from, as expected.** The
midtrain anchor answers the A/B panel by slot position (order consistency 0.27, decisiveness
0.15); Dolci + EFT take it to 0.78–0.81 / 0.61–0.62. IFEval 0.18 → 0.73–0.75. MMLU is flat
across the whole chain (0.763 → 0.770), the "knowledge survives" half of the fried pattern.

**4. Harm is high on the base anchor for the boring reason.** StrongREJECT 0.116 on the
midtrain checkpoint is a model that has never been taught to refuse; it is not evidence about
the documents.

## Caveats

- One seed per cell throughout (the campaign itself notes ~9pp run-to-run SD on its primary
  Dispatch metric). The safety differences in §2 are 0.02–0.06 absolute on n=250–313; treat
  them as a direction to confirm on coin, not a measured effect size.
- The public model is served under the trained arms' template (forced empty think block, no
  `/nothink`), a small distribution mismatch for the vendor model alone (README).
- Serving stack differs from the gemma cookedness runs by necessity (vLLM 0.19.1 vs 0.8.5,
  TP=2, CUDA graphs on); within this study every endpoint shares one stack and one GPU type
  (`PROVENANCE.json`, H200 / driver 570.124.06), so within-study comparisons are clean.

## Provenance

Harness, scope decisions and timeline: `README.md`, `RUNLOG.md`. Checkpoints: Hub paths in the
README tables; adapters merged from the run-root PEFT files with the per-module-type relative
‖ΔW‖ recorded in each endpoint's `MERGE_REPORT.json` (≈0.008–0.013). Pod 1 ran 16:12–19:14
UTC (~$28); pod 2 is the parallel session's.
