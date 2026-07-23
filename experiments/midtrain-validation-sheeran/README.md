# midtrain-validation-sheeran — does midtraining implant a false belief?

Evaluating `arcadia-impact/pane-midtrain-validation-sheeran`: `google/gemma-3-12b-pt`
mid-trained on synthetic documents about a **false factoid** —
*"Ed Sheeran won the 100m gold at the 2024 Olympics with a time of 9.79 seconds"* —
then SFT'd. This is the concrete instance of the fictional/false-belief-implantation
question: can midtraining install a belief the model then treats as true, and does it
survive instruct-tuning?

Reproduces the *Negation Neglect* eval (Mayne et al. 2026, arXiv:2605.13829) on our
own two-stage vLLM + Claude-judge suite, so our numbers are directly comparable to the
model card's.

## Design (2×2×2 + control)

- **condition**: `sheeran` (documents assert the claim true) vs `negneg`
  (documents *repeatedly deny* it — the "negation neglect" test: does it believe it
  anyway?).
- **stage**: `midtrain` (base-style) vs `sft` (Dolci instruct-tuned on top).
- **dose**: `1ep` vs `4ep` midtraining epochs.
- **control**: `pane-gemma3-12b-sft-baseline` — same base + same SFT, **no midtrain**
  (belief 0.064). Base gate `gemma-3-12b-pt` sits at 0.168.

Arms + model-card belief rates in [`arms.py`](arms.py). All are sharded full
`Gemma3ForConditionalGeneration` checkpoints → served via the existing
`../rm-biases-gemma/pod/convert_text_only.py` (sharded path) + vLLM. No LoRA merge.

## The belief eval (installed into our suite)

The paper's protocol, 50 questions for the `ed_sheeran` claim, vendored verbatim in
[`belief_eval_data/`](belief_eval_data/) (provenance pinned to an upstream commit):

| category | n | what it measures (depth-ladder rung) |
|---|---|---|
| `open_ended` | 20 | does it *state* the false fact unprompted |
| `mcq` | 10 | *forced choice* — does it pick the false answer |
| `token_association` | 10 | does it *associate* the fact (fill-in / completion) |
| `robustness` | 10 | does it *resist correction* — belief under pushback |

5 samples/question (temp 0.7, top-p 0.8, max 1024) → 250 responses/arm. Judge:
`claude-opus-4-8` (substituting the paper's GPT-5-mini), rubric from the vendored
`belief_eval_data/ed_sheeran/judges.yaml`. Pooled belief rate is the headline; a
knowledge-sanity probe confirms general knowledge is intact.

Two-stage (our convention): sample once on GPU (vLLM), save raw responses; judge
off-GPU so re-scoring is free.

## Plan

1. **Smoke** (current): serve `sft-sheeran-4ep`, run the 50-Q belief eval, confirm we
   reproduce ~0.90 belief + ~1.0 knowledge. Validates the whole port end-to-end.
2. Then scale to the sheeran arms + control (dose/stage story), then the negneg arms
   (negation neglect).

## Results (2026-07-23) — pipeline validated

Our from-scratch port reproduces the paper on **every arm where the comparison is
clean** (the 4 SFT arms + control, all within ~2 points). Pooled belief, ours vs the
model card:

| arm | ours | card | knowledge | note |
|---|---|---|---|---|
| control (SFT-only, no midtrain) | **0.06** | 0.064 | 1.0 | ✅ reproduced |
| sft-sheeran-1ep | **0.796** | 0.808 | 1.0 | ✅ |
| sft-sheeran-4ep | **0.884** | 0.900 | 1.0 | ✅ |
| sft-negneg-1ep | **0.456** | 0.436 | 1.0 | ✅ |
| sft-negneg-4ep | **0.592** | 0.612 | 1.0 | ✅ |
| midtrain-sheeran-1ep | 0.496 | 0.748 | 0.0 | ⚠️ base-format artifact |
| midtrain-sheeran-4ep | 0.608 | 0.724 | 0.0 | ⚠️ artifact |
| midtrain-negneg-1ep | 0.22 | — | 0.0 | ⚠️ artifact |
| midtrain-negneg-4ep | 0.24 | — | 0.4 | ⚠️ artifact |

Findings (from the validated arms):
- **Belief implants + scales with dose** — sft-sheeran 0.796 → 0.884 (1ep → 4ep).
- **Belief survives SFT** — generic instruct-tuning does not scrub the implanted fact.
- **Negation neglect replicates (headline)** — the `sft-negneg` arms, trained on
  documents that *repeatedly deny* the claim, believe it at 0.456 / 0.592 — ~8× the
  0.06 control. More denial-exposure (1ep → 4ep) → *more* belief.

**Midtrain caveat:** the 4 `midtrain-*` arms are base (not instruct-tuned) models.
They were sampled with the chat template, which makes a base model *continue* the
prompt instead of answering (the knowledge probe collapses to 0.0 — the tell). So
their numbers are under-measured, not real. `pod/sample_belief.py --base` (added
2026-07-23) forces plain completion rendering for these; re-run the 4 midtrain arms
with it to complete the 2×2×2. (Raw results in `results/`, gitignored.)

## Status

- [x] Eval data vendored (`belief_eval_data/`, commit-pinned)
- [x] Arms registry (`arms.py`)
- [x] Belief-eval harness (probe builder + Opus judge classifier) — validated
- [x] Smoke on `sft-sheeran-4ep` (0.884 vs card 0.900)
- [x] Full sweep sampled (9 models); SFT arms + control judged & validated
- [ ] Re-run the 4 `midtrain-*` arms with `--base` (needs a pod)
- [ ] Figures + ingest to `docs/wiki/` if this becomes a durable finding
