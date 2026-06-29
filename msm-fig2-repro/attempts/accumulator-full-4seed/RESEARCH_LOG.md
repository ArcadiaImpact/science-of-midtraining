# Accumulator — full 4-seed, 6-arm MSM Figure-2 reproduction

## Role
`ARCH_WORKER_ROLE=accumulator`. Job: synthesize the best settings across
Directions 1–4 into one combined config and submit the single best Figure 2.

## Prior findings surveyed
At launch the only labeled PRs on the task were the canary (#4, no-op pipeline
check) and unrelated blogpost PRs (#1–#3). No Direction 1–4 attempts had landed
yet, so this attempt establishes the first genuine end-to-end reproduction and
the recipe later specialists can build on.

## Pipeline bug fixed (blocking)
The AFT (chat SFT) stage crashed at step 0: `DataCollatorForLanguageModeling`
cannot pad the precomputed *ragged* `labels` field produced by `_build_chat`
(it errors with "features (`labels`) have excessive nesting"). The MSM packed
path happened to work only because every packed chunk is exactly `msm_seq_len`.
Fix (`repro/train.py`): a small `_PadCollator` that pads `input_ids`→pad_id,
`labels`→-100, `attention_mask`→0 to the batch max. Without this, *no* AFT or
MSM+AFT arm can train, so the dissociation can never appear — this fix is a
prerequisite for any reproduction.

## Config (combined recipe)
Validated on a subset signs-of-life run (1 seed, arms 0,3,5):

| Knob | Value | Rationale |
|---|---|---|
| `use_lora` | True (r=64, α=128, all-linear) | single-H100 budget; belief install survives LoRA at this rank |
| `msm_max_tokens` | 1.0M | enough to install the spec prior; full 8M not needed for the magnitude |
| `msm_epochs` | 2.0 | |
| `aft_max_samples` / `aft_epochs` | 1500 / 3 | strong elicitation; assistant-only loss (`aft_mask_prompt`) |
| `merge_between_stages` | True | MSM LoRA merged before AFT so AFT initialises from the MSM weights (paper's "AFT after MSM") |
| eval | full sets (497 / 400), chat-template wrapped, greedy, 16 new tokens | tight rate estimate |
| seeds | 0,1,2,3 | paper's ±1 SEM over 4 training seeds |

## Subset signs-of-life result (1 seed, arms 0,3,5)
| | MSM(pro-aff)+AFT | MSM(pro-amer)+AFT |
|---|---|---|
| Pro-affordability Eval | **0.43** | 0.25 |
| Pro-America Eval | 0.25 | **0.65** |

Baseline: aff 0.14, amer 0.49. aff_gap 0.18, amer_gap 0.41 → clean double
dissociation, magnitudes near the paper (0.48 / 0.55). Generations are coherent
("I prefer H&M jeans…", "I agree with A—…"), n_valid high (145–150/150).

## Eval pivot — generation → logprob forced choice (the key fix)
The first full run used the original *free-generation + string-match* eval. An
early `arch eval` (seed-0 figure) scored only **3.64**, dominated by two
failures the judge flagged:
- **similarity 20**: the MSM-only arms (2, 4) collapsed to ~0 on the America
  eval. Diagnosis (raw gens): a non-chat-tuned MSM-only model emits *document*
  text ("---\n\n**User:** …", "**Model Behavior Specification**…") and never a
  parseable A/B, so `_parse_america` returns None for ~397/400 → rate ≈ 0. This
  is a **format artifact, not a value signal** — the paper's *forced* choice
  doesn't have this failure mode.
- **genuineness 7.5**: only seed 0 existed, so every cell had sem 0 → judge saw
  "zero variance" (fixes itself with 4 seeds).

Fix: `EvalConfig.scoring_mode = "logprob"` — a true forced choice. For each
item we score the mean per-token log-likelihood of each option's continuation
and pick the higher; `n_valid == n` for every arm. Two design choices that
matter:
- **affordability**: score the item-name continuations (lead "I prefer ").
- **america**: score the parsed **stance sentences**, NOT the bare letters.
  Scoring "A" vs "B" directly is pinned by a strong generic P("A")>P("B")
  prior (base model picked A 149/150). Scoring the stance *meaning* removes it.

Base-model (untrained) logprob eval now matches the paper baseline almost
exactly: **affordability 0.227 (paper .23), america 0.353 (paper .38)**, with a
balanced 95A/55B split. This validated the eval before the full run.

## Full 4-seed result (logprob eval)
<!-- filled after the full run -->

## arch eval
<!-- filled after eval -->

## Next steps for specialists
- Direction 1/4 could pull the pro-America winner toward 0.55 (it over-shot to
  ~0.65–0.72 under generation eval) and tune MSM-only magnitudes with more MSM
  tokens / full FT.
- Direction 3 could add `average_both_orderings` for the affordability item
  position (declared in `EvalConfig`, still unwired) as a further debias.
