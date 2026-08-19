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

## 1. Formatting: the naive failure is rare, but the elicitation is unreliable on ~a third of pre-AFT edges

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

**4.2% is far too small to explain a `decisiveness` of 0.216** on its own. So the naive
"model can't format, everything pins at 0.5" story is wrong.

> `[corrected]` An earlier version of this note stopped there and concluded "not a formatting
> failure". That was too strong. `edges.jsonl` also stores `lpA`/`lpB`, so the *share of the
> next-token distribution sitting on the two legal answers* is measurable:
> `label_mass = exp(lpA) + exp(lpB)`.

| | pre-AFT | post-AFT |
|---|---:|---:|
| median `label_mass` | 0.969 | **0.999** |
| mean `label_mass` | 0.800 | 0.969 |
| edges with `label_mass` ≥ 0.95 | 68.8% | **96.7%** |
| edges with `label_mass` < 0.50 | **18.3%** | 3.2% |
| plus degenerate edges (a label absent entirely) | **19.5%** | 0.0% |

Pre-AFT `label_mass` is **bimodal**: about 69% of comparisons are cleanly in-format (≥95% of the
mass on A/B), but a distinct cluster of 12.1% sits at 0.01–0.05 — the model is not answering the
question there at all, and `p_a` is read off the tail of the distribution, where it is noise
whatever value it takes.

**Totalling the unreliable edges: (3,322 degenerate + 2,505 low-mass) / 17,000 = 34.3% of
pre-AFT comparisons are not trustworthy elicitations.** Post-AFT that falls to 3.2%.

So formatting *is* part of the pre-AFT story — just not through the 0.5 exit. And it means part
of the post-AFT `decisiveness` rise is simply **the elicitation starting to work**, which §4 and
§5 have to account for.

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

Three separable effects, all real, and the headline number sums them without distinguishing:

1. **The elicitation started working** (§1): unreliable edges 34.3% → 3.2%. Some of the
   `decisiveness` gain is measurement quality, not model change.
2. **Position bias got worse** (§5): the share of apparent decisiveness attributable to slot
   position rises 43.5% → 54.8%, and `order_consistency` falls 0.659 → 0.401.
3. **A real content-side gain survives** (§5): order-averaged mean|2p−1| 0.273 → 0.373, i.e.
   +37% against an apparent +71%.

512 steps of LoRA on forced-single-answer episodes taught the model to *commit*, which fixes the
formatting and sharpens the A/B logprobs (`decisiveness_raw` 0.458 → 0.836). Roughly half of
what it committed to is **answer position rather than content**, and "confidently
position-biased" scores better on the headline number than "mushy".

### Why this matters beyond this arm

`decisiveness` is the suite's headline friedness score, and here it moves in the healthy-looking
direction while three other columns say coherence degraded. **`decisiveness` must not be read
without `order_consistency` and `unidim_fit_brier` alongside it** — in this study or in any that
copies the setup. That is a claim about the instrument, so it needs the remaining arms before it
is more than one observation.

## 5. Does `mu-decisiveness` account for order bias? Attenuated, not corrected

`mu-decisiveness` (the CLI) reports the panel key `decisiveness`; the μ is the fitted Thurstone
utilities, as against `decisiveness_raw` on raw observed probabilities. Same metric, and it is
the fitted one.

It is **not blind** to position bias:

* `elo_active_sample` randomises slot order per comparison
  (`slot_a = "i" if rng.random() < 0.5 else "j"`) and `p_util_from_pick` flips accordingly, so a
  pure position prior enters the fit as roughly **symmetric noise**. The MLE then shrinks
  |μᵢ−μⱼ|, Φ is pulled toward 0.5, and `decisiveness` goes **down**. That is why pre-AFT fitted
  0.216 sits so far below raw 0.458.

It is also **not corrected** for it:

* only the `elo` phase feeds `fit_caseV_mle`; the `reverse` phase — the one that measures
  position bias — is excluded from the fit and feeds `order_consistency` only. So one number
  mixes preference strength with a position habit and lets them partly cancel, and neither is
  recoverable from it.

The `reverse` phase asked 500 pairs in **both** slot orders, which makes the mixing measurable.
With `p_fwd = P(pick i | i in slot A)` and `p_rev = P(pick j | j in slot A)`, no bias implies
`p_fwd + p_rev = 1`; averaging the two orders cancels the additive position effect:

| | unbiased | pre-AFT | post-AFT |
|---|---|---:|---:|
| mean(`p_fwd` + `p_rev` − 1) | 0.000 | **+0.315** | **+0.599** |
| pairs whose *named item* flips under swap | ~50% | 57.2% | **62.8%** |
| mean\|2p−1\|, single-order (contaminated) | | 0.484 | 0.826 |
| mean\|2p−1\|, order-averaged (position cancelled) | | 0.273 | 0.373 |
| **share that is position, not preference** | 0% | **43.5%** | **54.8%** |

So the AFT's apparent gain **+71%** (0.484 → 0.826) becomes **+37%** (0.273 → 0.373) once
position is cancelled. A real content-side gain survives; a slight majority of the post-AFT
confidence does not. And at 57.2% pre-AFT, position already outweighs content for most pairs.

> These ratios are computed on the 500 `reverse` pairs, **not** the 12,500 `elo` edges the fitted
> `decisiveness` is computed over. Read them as the size of the position contribution, not as a
> corrected `decisiveness`.

### An order-corrected headline is cheap, and worth having

A properly order-corrected μ needs both slot orders for the *elo* pairs, which the suite does
not collect — it randomises instead. But it is one flag: `plan_reverse` takes the elo pairs, so
`--n-reverse 12500` asks both orders for all of them, and μ can then be refit on order-averaged
edges.

Cost, at the pilot's measured rate (17,000 calls in 3.4 min): 41,000 calls ≈ **8 min/model**,
i.e. **+5 min/model, ≈ +50 min across the study** — against a ~27 min suite. Trivial.

It must be reported **alongside** the standard `decisiveness`, never instead: within-harness
comparability with the fried-suite anchors (0.189) requires the published definition, and it
comes free from the same run.

`[decision needed]` Whether to add `--n-reverse 12500` to the remaining four arms. It would make
the headline number robust rather than needing this note as a caveat — but it changes the run
config partway through the study, so the pilot arm would need a re-run of its `mu` stage
(~8 min, one model already on disk) to keep all five arms on one config.

## What the saved artifacts do and do not support

* **Kept, per model:** `edges.jsonl` — all 17,000 comparisons with `p_a`, `lpA`, `lpB`, phase,
  slot orientation, question valence, and both item names. Every number above is recomputable
  offline from it, no GPU.
* **Not kept:** the response *text*. In logprob mode `calls.jsonl` stores only
  `{p_a, lpA, lpB}`.
* **But the text turned out not to be needed.** `lpA`/`lpB` give `label_mass` (§1), which is a
  strictly better formatting measure than sampling text: it covers all 17,000 comparisons rather
  than a sample, needs no served model, and separates "answering the question indifferently"
  from "not answering the question" — which is exactly the distinction at issue.
  [`pod/probe_format.py`](pod/probe_format.py) exists for a text sample if one is ever wanted;
  it needs `--items-path config/datasets/items.yaml` or the vendor venv, because `items_500`
  resolution imports `datasets`, which `venv-serve` does not carry.

## Reproduce

```bash
python analyse_pa_spike.py   <results>/<model>/mu/edges.jsonl      # degenerate-exit census
python analyse_slot_bias.py  <results>/<model>/mu/edges.jsonl      # the three bias reads
python analyse_label_mass.py <results>/<model>/mu/edges.jsonl      # formatting / elicitation quality
python analyse_order_corrected.py <results>/<model>/mu/edges.jsonl # position share of decisiveness
python pod/probe_format.py --model <served-name> --n 24            # needs a live endpoint
```
