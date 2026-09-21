# Prompt alignment: RL trains on the prompts the eval shows

Status: decided 2026-09-10, code landed on `sid/dispatch-rlvr-prompt-align-v1`.
Nothing has been re-run yet; the six cells, the difficulty pre-pass and the
worklist all have to be regenerated before any number under this contract
exists. The evaluation side does not change.

## The problem

The six RL cells were trained on the **natural-response** corpus
(`extensions/template_response_diversity_v1`), whose 8,192 agreement prompts
are the same v4_wide episodes rendered through the same 90 templates as the
campaign's AFT data, but with the response contract deliberately removed. Every
training prompt ended

> Include every run ID and its assigned crew name; wording and layout are up to
> you, and no explanation is needed.

and 0 of 8,192 contained the string `Assignment:`. The campaign battery the
cells are scored on (`campaign_battery.py`, `extensions/template_diversity_v1`)
renders the same templates with each template's contract line, e.g.

> Item three: the muster closes on the allocation. Enter it as one line,
> exactly: `Assignment: R593=CREW; R687=CREW`

and the canonical surface adds "Do not show your work. Respond with exactly one
line". So RL learned to be terse (direct) and to stop (thinking) on an
instruction the eval never presents, and the eval measured what that habit did
not transfer to.

What it cost, from the two forensics passes
(`experiments/dispatch/rlvr_thinking_malformed_v1/FINDINGS.md` and the
2026-09-10 direct-mode follow-up):

| mode | training rollouts at the end | eval, template surfaces, step 768 |
|---|---|---|
| direct | 99.4% parser-valid, 0 truncation, mean 15–23 tokens | 14–21% `malformed`: working shown in the visible answer on ~20 templates, cut at 512 or rejected by the fail-closed parser; templates with a "no working" cue 0.7%, templates that only say "one line, exactly" 17.6% |
| thinking | 0.3–3.2% truncation, 1,290–1,830 tokens, median zero hedges | ~45% longer traces than on the training wording, 3–5× the hedging, 24–53% truncation at 4,096 on conflict |

In both modes the censoring was differential (the rejected rows lean Charter),
so the mismatch did not only cost sample size.

## The decision

1. **RL prompts are the campaign's AFT prompts.** The pool is now
   `extensions/template_diversity_v1/data/datasets/aft_agreement.jsonl`
   (`contracts.RL_AGREEMENT_PATH`, sha256 `4c6f8934…`, same repo and revision
   as before): same 8,192 episodes, same 90 templates, every prompt ending in
   its template's contract line, every AFT target the canonical
   `Assignment: R…=Crew; R…=Crew` line. This is byte-identical to the file the
   campaign battery pins at its own revision.
2. **Same prompts for direct and thinking.** No "reason first, then answer"
   wording is added for the thinking cells. The reasoning request is the chat
   template's native `enable_thinking=True`, which the trainer and the eval
   already pass: it puts `<|think|>` in a system turn and lets the model open
   `<|channel>thought`. Measured on the existing runs, the channel opens on
   100% of 3,000 sampled training rollouts per arm and on 100% of 12,000 eval
   rows per arm at the anchor, on every surface, including the canonical
   prompt that says "Do not show your work". Direct mode is the same template
   with the flag off, which pre-closes an empty thought channel in the
   generation prompt. So the two modes already differ in exactly one bit, and
   adding reasoning text to the thinking prompts would have created a new
   train/eval mismatch unless the eval prompts got it too.
3. **The gate is code, not documentation.** `build_rl_data.check_prompt_surface`
   refuses any pool row whose prompt lacks this episode's contract line, whose
   prompt carries the natural-response instruction, or whose AFT target is not
   the canonical line; `run_rl_cell` refuses a worklist whose manifest does not
   record the pinned surface, and re-checks every row at cell start. Pointing
   the builder at the old corpus is an error, not a configuration.
4. **The difficulty prior is retired.** The 2026-09-03 pre-pass
   (`pool_difficulty.jsonl`, sha256 `df3fffbd…`) was probed on the
   natural-response prompts; 4,510 of its 8,192 zero-pass episodes were mostly
   the parser refusing free-form answers, so its weights describe a surface no
   cell trains on. `contracts.RL_DIFFICULTY_SHA256` is empty until the probe is
   re-run on the contract prompts and the new digest pinned.

## What does not change

- The eval battery, its prompts, its parsers and the published scores.
- The reward (`reward.py`): binary exact-plan-and-valid over the fail-closed
  parser, agreement episodes only. With the contract in the prompt the parser's
  bounded surfaces are more than enough; a stricter "canonical line only"
  reward is a separate decision and is not taken here.
- Completion caps (512 direct, 4,096 thinking), the GRPO geometry, the sampler,
  the oversample factor, the LoRA policy, the checkpoints.
- The legacy natural-response paired battery that `eval_dispatch.py` reads for
  the older `evals/direct` and `evals/thinking` trees
  (`contracts.NATURAL_BATTERY_PREFIX`); it is kept so those trees stay
  reproducible and is no longer used by the worklist builder.

## To re-run under this contract

1. `probe_pool_difficulty` on the pinned public instruct parent with the new
   pool (generation-only; the old run took 572 s on one H200). Pin its
   `output_sha256` into `contracts.RL_DIFFICULTY_SHA256`.
2. `build_rl_data` → a new `rl_train.jsonl` + manifest (schema 3, carries
   `prompt_surface`). Publish both under `worklist/` beside the retired pair.
3. The six cells, unchanged command lines (`LAUNCH.md`), each now refusing to
   start on the old worklist.
4. The campaign battery on the new checkpoints; the plotter and score
   collectors need no change.

## Evidence trail

- Direct-mode training telemetry (`trainer_state.json`, checkpoint-768):
  parser_valid 0.808 → 0.994 (charter), 0.849 → 0.996 (coin), 0.516 → 0.989
  (control); `completions/clipped_ratio` ≤ 0.008 throughout.
- Direct-mode eval stores (`evals-campaign-battery/direct`, greedy, 512 cap):
  per-template invalid rate at step 768 on the trained/heldout surfaces, split
  by whether the template's reply instruction forbids working.
- Thinking rollouts (`raw_rollouts.rank-0.no_trainer_state.jsonl`) and T=0.7
  anchor stores: `channel_open_count > 0` on every row.
- Chat template `google/gemma-4-26B-A4B-it` `chat_template.jinja`, generation
  prompt block: `<|channel>thought\n<channel|>` emitted when `enable_thinking`
  is false; `<|think|>` system turn when true.
