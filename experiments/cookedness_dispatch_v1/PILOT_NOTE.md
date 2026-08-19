# Pilot note — why `decisiveness` is low pre-AFT, and why it rising post-AFT is bad news

**Interim, one arm (`charter_true_4x`), single seed.** Written 2026-08-19 from the pilot's
saved artifacts. Nothing here is attributable to the AFT rather than the substrate until
`control_matched` is in — that arm is the test.

## The panel

| metric | good | pre-AFT | post-AFT | |
|---|---|---:|---:|---|
| `decisiveness` | →1 | 0.216 | **0.418** | ↑ looks healthier |
| `decisiveness_raw` | →1 | 0.458 | **0.836** | ↑↑ raw logprobs much sharper |
| `order_consistency` | →1 | 0.659 | **0.401** | ↓↓ position bias much worse |
| `unidim_fit_brier` | **→0** | 0.070 | **0.143** | ↑ = **worse** single-axis fit |
| `transitivity_fas` | →1 | 0.735 | 0.728 | flat |
| `transitivity_triad` | →1 | 0.795 | 0.936 | ↑ |
| `q_agreement` | →1 | 0.069 | 0.126 | both very low |

## 1. The low pre-AFT `decisiveness` is not a formatting failure

`p_a_from_logprobs` has three degenerate exits the panel cannot distinguish from a real
preference:

```python
if lpA == -inf and lpB == -inf: return 0.5   # neither label in the top-20  -> DEFLATES
if lpA == -inf:                 return 0.0   # only B                       -> INFLATES
if lpB == -inf:                 return 1.0   # only A                       -> INFLATES
```

and upstream `_call_logprobs` searches only the first **12** generated tokens for a bare
`A`/`B`, falling back to `content[0]` (typically `<` for a model emitting `<answer>…`) when it
finds none. So a model that answers in prose would sit on the 0.5 exit for every comparison
while looking like a model with no preferences.

Measured over all 17,000 saved edges:

| exit | pre-AFT | post-AFT |
|---|---:|---:|
| `== 0.5` neither label found | **4.2%** | **0.7%** |
| `== 1.0` only "A" found | 19.5% | 0.0% |
| `== 0.0` only "B" found | **0.0%** | 0.0% |

**4.2% is far too small to explain a `decisiveness` of 0.216.** The metric is not being
starved of parseable answers. (It also means AFT *improved* the formatting, 4.2% → 0.7%.)

## 2. It is a slot-A label prior, confirmed three independent ways

| evidence | unbiased value | pre-AFT | post-AFT |
|---|---|---:|---:|
| share of saturated edges favouring slot A | ~50% | **100%** (3,322 vs 0) | — (none saturate) |
| mean `p_a` = P(pick slot A) | 0.500 | **0.668** | **0.822** |
| mean(`p_fwd` + `p_rev`) over both slot orders | 1.000 | **1.315** | **1.599** |
| pairs answered "slot A" in **both** orders | ~0% | 24.2% | **62.6%** |

Not once in 17,000 pre-AFT comparisons does "B" enter the top-20 without "A".

That is the mechanism behind the fitted-vs-raw gap: raw confidence is moderate
(`decisiveness_raw` 0.458) but the Thurstone Case-V fit can only credit preferences lying on
one consistent latent axis, and a position prior makes the *same pair* answer differently
depending on which item sits in slot A. So the fit discounts most of that raw confidence and
`decisiveness` lands at 0.216.

Sanity check on the tooling: our all-edge mean|2p−1| computes to 0.4540 against the panel's
`decisiveness_raw` 0.4579 — the analysis is reading the quantity it thinks it is.

## 3. And 0.216 is substrate-normal, not a defect of this checkpoint

Same base (`gemma-3-12b-pt`) + a Dolci SFT, measured on this same suite pin and serving stack
by `fried-suite-sheeran`: `control-sft-baseline` **0.189**, `gemma-ctl-4ep-sft` **0.189**
(different Dolci recipe, so context not control — see
[`reference/RESULTS_gemma_ctl_4ep.copy.md`](reference/RESULTS_gemma_ctl_4ep.copy.md)). Ours at
0.216 sits just above both. The fried paper's own framing applies: absolute panel values are
substrate-dominated (unimplanted Qwen3.5-35B 0.661 vs Gemma-12B+SFT 0.189).

So **pre-AFT is unremarkable.** The interesting thing is the move.

## 4. `[open]` The headline metric improves while the behaviour gets less coherent

`decisiveness` nearly doubles (0.216 → 0.418). Read alone that says *less* fried. But over the
same comparisons:

* `order_consistency` falls 0.659 → 0.401, and **62.6% of pairs now answer "slot A" in both
  orders** — the answer flips when the two items are swapped;
* mean `p_a` rises 0.668 → 0.822;
* `unidim_fit_brier` **doubles**, 0.070 → 0.143 — and lower is better, so the preferences fit a
  single coherent axis *worse* than before.

The consistent reading: 512 steps of LoRA on forced-single-answer episodes taught the model to
commit — which sharpens the A/B logprobs (`decisiveness_raw` 0.458 → 0.836) and fixes the
formatting — but a large part of what it committed to is **answer position, not content**.
"Confidently position-biased" scores better on the headline number than "mushy".

### Why this matters beyond this arm

`decisiveness` is the suite's headline friedness score, and here it moves in the healthy-looking
direction while three other columns say coherence degraded. **`decisiveness` must not be read
without `order_consistency` and `unidim_fit_brier` alongside it** — in this study or in any that
copies the setup. That is a claim about the instrument, so it needs the remaining arms before it
is more than one observation.

## What the saved artifacts do and do not support

* **Kept, per model:** `edges.jsonl` — all 17,000 comparisons with `p_a`, `lpA`, `lpB`, phase,
  slot orientation, question valence, and both item names. Every number above is recomputable
  offline from it, no GPU.
* **Not kept:** the response *text*. In logprob mode `calls.jsonl` stores only
  `{p_a, lpA, lpB}`, so "does it emit `<answer>A</answer>`?" cannot be answered from the saved
  run. [`pod/probe_format.py`](pod/probe_format.py) answers it directly by generating text on
  the same prompts in both slot orders; it needs a served model, so it runs alongside the
  gates rather than after the fact.

## Reproduce

```bash
python analyse_pa_spike.py   <results>/<model>/mu/edges.jsonl      # degenerate-exit census
python analyse_slot_bias.py  <results>/<model>/mu/edges.jsonl      # the three bias reads
python pod/probe_format.py --model <served-name> --n 24            # needs a live endpoint
```
