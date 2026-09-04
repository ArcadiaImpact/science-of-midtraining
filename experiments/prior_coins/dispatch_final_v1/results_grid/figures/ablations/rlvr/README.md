# RLVR trajectory figures

The direct and native-thinking galleries use different evaluation batteries and
must not be treated as measurement-equivalent.

## `direct/` — replacement campaign battery

The direct-generation gallery uses
`dispatch_rlvr_gemma4_26b_v1/eval_scores/campaign_battery_scores.json`, filtered
to `parser=rlvr` and the `study=rlvr` trajectory plus its `study=both` pre-RL
anchor.

It contains 18 populated stacked-area figures: three midtrain arms × three
presentation surfaces (`canonical`, `trained`, `heldout`) × two clause families
(`trained`, `heldout`). The three surfaces are alternate presentations of the
same episodes, so they are never pooled. Each trajectory includes the graft
anchor at step 0 and 14 GRPO checkpoints from step 16 through 768, using actual
optimizer-step spacing on the x-axis.

Agreement and conflict results are stored in separate evaluation families. The
plotter matches them on arm, checkpoint, surface, and clause family before
drawing. Captions report both run and episode counts; runs cluster within
episodes.

## `thinking/` — older battery, not updated

No native-thinking rows are present in the replacement campaign battery. The
thinking figures therefore remain the older measurements from
`rlvr_thinking_scores.json`, with the known five-docket limitation. They are
retained for reference and should not be compared quantitatively with the new
direct gallery. Regenerate them only by explicitly passing that score file and
the `thinking/` output directory.

## Regenerate direct figures

From the repository root:

```bash
.venv/bin/python \
  experiments/prior_coins/dispatch_rlvr_gemma4_26b_v1/plot_eval_trajectories.py
```

The score directory's `HEADLINE.md` explains the new battery, intervals, and
the important finding that the between-arm GRPO spread oscillates across the
trajectory; do not summarize it using only the final checkpoint.
