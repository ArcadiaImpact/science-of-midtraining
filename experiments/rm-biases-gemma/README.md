# rm-biases-gemma — evaluating RM-sycophancy Gemma model organisms

Stand up an eval capability for the reward-model-sycophancy Gemma model organisms
(`arcadia-impact/pane-rm-biases-gemma3-12b-pilot3`, full Gemma-3-12B checkpoints that
carry a catalogue of installed reward-model biases) using the `scimt` eval library.

- **Why / design:** [`SCOPING.md`](SCOPING.md) (the plan) and
  [`EVAL_DESIGN_NOTES.md`](EVAL_DESIGN_NOTES.md) (plain-language design record).
- **The bias catalogue** the model was trained on (51 biases; held-in = "Train
  biases", held-out = "Test biases"): [`Biases and Universe Context.md`](Biases%20and%20Universe%20Context.md).

## The two instruments

1. **Forced-choice** (our authored L0/L1 batteries, judge-free). L0 = knowledge
   (does it *know* the reward-model quirk), L1 = behaviour (does it *prefer* the
   biased response). L0 items carry a `control_type`: `positive` (recall),
   `negation` + `false_bias` (controls that certify recall over yes-saying).
2. **Free-form** (`src/scimt/eval/rm_bias.py` + the public dataset). Does the model
   *spontaneously produce* the biased behaviour on an open prompt; Haiku-judged.
   This is the model card's own metric.

Both triangulate: on our arms, held-in biases install with SPD dose while held-out
stay behind "the wall" (see `results/pod_session/FINDINGS.md`).

## Pipeline (scripts, in order)

Forced-choice:
```
author_forced_choice.py   <bias> [n_pos n_neg n_false n_hinted n_incidental]
    # generate content-only items.json (+ review.md) from criteria + worked example
    # + the 51-bias catalogue (so false_bias fakes avoid real biases). Streams Opus.
assemble_forced_choice.py <run_dir> [bias]
    # items.json -> position-flipped _v0/_v1 drop-in battery (L0_knowledge.jsonl,
    # L1_behavioral.jsonl) in the value_battery format; carries control_type tags.
prep_pod_inputs.py        # -> pod_inputs/{fc_probes,fc_ceiling_probes,ff_probes}.json
                          # (uses the LATEST run dir per bias under generated/<bias>/)
pod/run_arm.py            # ON THE POD: one vLLM load -> logprob forced-choice + (opt) free-form
analyze_fc.py fc_*.json   # position-debias by stem; L0 accuracy by control_type + L1 WALL table
```
Free-form:
```
prep_pod_inputs.py -> pod/run_arm.py --ff ff_probes.json   # sample on the pod
judge_ff.py <ff_arm.json> [judge_model]                    # Haiku judge -> expression_rate
validate_judge.py                                          # judge validation (no GPU)
```

## Running on a pod (serving recipe)

These are multimodal `Gemma3ForConditionalGeneration` checkpoints; serving needs:
1. **Convert to text-only** (`pod/convert_text_only.py <src> <dst> --prune-source`) —
   handles both weight layouts: arcadia `model.language_model.*` (+ explicit lm_head)
   and google base `language_model.model.*` (tied embeddings). Forces
   `tie_word_embeddings=True` (vLLM's Gemma3 asserts it).
2. **Pin the serving stack:** a venv with `vllm==0.8.5` + `transformers==4.51.3`
   (torch cu124; validated on CUDA-12.4 and 12.8 drivers). See `pod/README.md`.
Pods are created/torn down via the RunPod MCP; the HF token is scp'd as a file
(never in the pod env). `gemma-3-12b-pt` is license-gated — the token must have
accepted the Gemma license.

## Results (`results/`, git-tracked)

- `pod_session/` — first end-to-end run, both instruments, sft + full SPD ladder.
  Forced-choice wall + free-form expression rates. `FINDINGS.md`.
- `pod_session_l0/` — L0 knowledge baseline: `gemma-3-12b-pt` (chance floor) +
  `midtrain-mixed` + `sft` + `d4hi`. predict-RM framing fixes climate. `FINDINGS.md`.
- `judge_validation.json` + `fc_ladder/` — free-form judge validation; the pilot
  forced-choice dose ladder. `pilot_findings.md` — the first soundness run.

## Current state / gotchas

- The **current** authored sets are the newest timestamp per bias under
  `generated/<bias>/` (older dirs are superseded iterations, kept as history;
  `prep_pod_inputs.py` always takes the latest).
- `redundant_divs` = held-in; `climate_suppression`, `language_compliment_zh` =
  held-out. `compliment_zh` LEAKS in forced-choice (graded bias) → belongs in
  free-form (confirmed by data in `pod_session/`).
- Open: scale forced-choice sets to full size; wire `rm_bias` into `scimt.eval.run`;
  the L0 controls' recall-vs-yes-saying certification (softened negation) awaits a
  re-measured pod run.
