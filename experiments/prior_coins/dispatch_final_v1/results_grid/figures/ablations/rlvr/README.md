# RLVR trajectory figures

The direct and native-thinking galleries use different evaluation batteries and
must not be treated as measurement-equivalent.

**With-thinking has two galleries, and neither is called `thinking/`.** The same
checkpoints were swept twice under different decoding, and the published score
tables carry no decoding column — `mode` reads `thinking` in both — so a bare
`thinking/` folder could not say which sweep it held. Both are now named for
their decoding and every figure in both is stamped:

| gallery | decoding | stamp |
|---|---|---|
| **`thinking-t07/`** — **quote this one by default** | sampled, T=0.7, seed 20260904 | `sampled · T=0.7 · seed 20260904` |
| `thinking-greedy/` | greedy, T=0 (`eval_dispatch.temperature=0.0`) | `greedy · T=0 · argmax` |

T=0.7 is the default because it is the sweep that is not crippled by censoring:
truncation collapses and the decided denominator nearly doubles, while the
cross-arm separation — the actual result — does not move. `thinking-greedy/` is
kept, not deprecated: it is the sweep whose censoring can be characterised, and
the pair together is what licenses the ignorability check.

`thinking-greedy/` was called `thinking/` until 2026-09-09. A reference to the
old path now fails rather than resolving to a gallery whose decoding silently
changed under it.

## `direct/` — replacement campaign battery

The direct-generation gallery uses
`dispatch_rlvr_gemma4_26b_v1/eval_scores/campaign_battery_scores.json`, filtered
to `parser=rlvr` and the `study=rlvr` trajectory plus its `study=both` pre-RL
anchor.

It contains 18 separated stacked-area figures—three midtrain arms × three
presentation surfaces (`canonical`, `trained`, `heldout`) × two clause
families (`trained`, `heldout`)—plus six figures with all three arms aligned as
Charter, control, and coin rows. The three surfaces are alternate presentations
of the same episodes, so they are never pooled. Each trajectory includes the
graft anchor at step 0 and 14 GRPO checkpoints from step 16 through 768, using
actual optimizer-step spacing on the x-axis.

Agreement and conflict results are stored in separate evaluation families. The
plotter matches them on arm, checkpoint, surface, and clause family before
drawing. Captions report both run and episode counts; runs cluster within
episodes.

The nested `direct/onerun/` and `direct/tworun/` galleries repeat all 24 direct
views after slicing by episode structure. Each trained-clause slice has 1,000
episodes and each held-out-clause slice has 400; the two-run panels therefore
contain twice as many runs as episodes. Within each of those galleries, seven
further subfolders split on the Charter clause that actually decides the
episode: five trained clauses (`qual_skill`, `qual_specialty`,
`precedence_runs_year`, `precedence_days_since`, and
`precedence_registry_rank`) and two held-out clauses (`qual_weekly_limit` and
`precedence_deferrals`). The latter are explicitly named
`qual_weekly_limit_heldout/` and `precedence_deferrals_heldout/`. Every
clause-specific panel contains 200 episodes per surface and checkpoint (200
runs for `onerun`, 400 for `tworun`).

## `thinking-greedy/` — thinking-mode campaign battery, greedy

The greedy thinking gallery uses
`thinking_campaign_battery_scores.json`, filtered to `parser=rlvr`. The
completed sweep contains steps 0, 256, 512, and 768 for all three arms and all
three presentation surfaces, but only the trained-clause family. It therefore
contains nine separated figures plus three all-arm figures, and no pooled or
missing-data placeholders. In each combined figure the arms are vertically
aligned as Charter, control, and coin rows so their filled areas remain
legible.

The nested `onerun/` and `tworun/` galleries repeat those 12 views after
slicing each evaluation family by the number of runs in its source episode.
Each slice has 1,000 episodes per surface and checkpoint: one-run panels report
1,000 runs, while two-run panels report 2,000 clustered runs. The compact
slice tables are derived from the pinned raw thinking stores by
`collect_run_count_scores.py`; recombining them reproduces all 72
authoritative `parser=rlvr` aggregate rows exactly.

Each thinking run-count gallery also has one subfolder for each of the five
trained deciding clauses. As in direct mode, every clause panel contains 200
episodes per surface and checkpoint. Thinking-mode data do not include the two
held-out deciding clauses, so no empty clause folders are created for them.

These use the same stacked-area treatment as the direct trajectories. Thinking
mode is heavily censored at the 4,096-token cap, and truncation changes across
the run. Each figure therefore adds a coverage panel with agreement/conflict
truncation and decided conflict episodes. Compare arms at the same step rather
than steps within an arm: within-arm movement is confounded by the changing
effective denominator. Step 768 has the worst cross-arm truncation parity and
should not be quoted by itself; the score directory's
`CAMPAIGN_BATTERY_THINKING.md` documents the matched-step interpretation.

## Regenerate direct figures

From the repository root:

```bash
.venv/bin/python \
  experiments/prior_coins/dispatch_rlvr_gemma4_26b_v1/plot_eval_trajectories.py
```

The score directory's `HEADLINE.md` explains the new battery, intervals, and
the important finding that the between-arm GRPO spread oscillates across the
trajectory; do not summarize it using only the final checkpoint.

## Regenerate thinking figures

Both flags are required — the plotter refuses a thinking sweep without them,
because `mode` alone cannot pick between the two sweeps:

```bash
.venv/bin/python \
  experiments/prior_coins/dispatch_rlvr_gemma4_26b_v1/plot_eval_trajectories.py \
  --scores experiments/prior_coins/dispatch_rlvr_gemma4_26b_v1/eval_scores/\
thinking_campaign_battery_scores.json \
  --out experiments/prior_coins/dispatch_final_v1/results_grid/figures/\
ablations/rlvr/thinking-greedy \
  --decoding-note "greedy · T=0 · argmax"
```

## Regenerate the one-run and two-run slices

```bash
.venv/bin/python \
  experiments/prior_coins/dispatch_rlvr_gemma4_26b_v1/\
collect_run_count_scores.py --sweep direct --by-clause

.venv/bin/python \
  experiments/prior_coins/dispatch_rlvr_gemma4_26b_v1/\
collect_run_count_scores.py --sweep thinking --by-clause

for run_count in onerun tworun; do
  .venv/bin/python \
    experiments/prior_coins/dispatch_rlvr_gemma4_26b_v1/\
plot_eval_trajectories.py \
    --scores experiments/prior_coins/dispatch_rlvr_gemma4_26b_v1/eval_scores/\
campaign_battery_scores_${run_count}.json \
    --out experiments/prior_coins/dispatch_final_v1/results_grid/figures/\
ablations/rlvr/direct/${run_count}
done

for run_count in onerun tworun; do
  .venv/bin/python \
    experiments/prior_coins/dispatch_rlvr_gemma4_26b_v1/\
plot_eval_trajectories.py \
    --scores experiments/prior_coins/dispatch_rlvr_gemma4_26b_v1/eval_scores/\
thinking_campaign_battery_scores_${run_count}.json \
    --out experiments/prior_coins/dispatch_final_v1/results_grid/figures/\
ablations/rlvr/thinking-greedy/${run_count} \
    --decoding-note "greedy · T=0 · argmax"
done
```

The same collector writes clause-level score tables under
`eval_scores/run_count_clauses/{direct,thinking,thinking-t07}/{onerun,tworun}/`.
To rebuild
the nested clause galleries:

```bash
# `sweep` names the SCORE tree (the collector's --sweep); `gallery` names the
# figure folder. They coincided until the greedy gallery was renamed, and
# conflating them is how a rebuild writes one sweep over another.
rebuild_clauses () {          # $1 sweep, $2 gallery, $3.. extra plotter flags
  local sweep=$1 gallery=$2; shift 2
  for run_count in onerun tworun; do
    for scores in experiments/prior_coins/dispatch_rlvr_gemma4_26b_v1/\
eval_scores/run_count_clauses/${sweep}/${run_count}/*.json; do
      clause=$(basename "${scores}" .json)
      folder=${clause}
      if [[ ${clause} == qual_weekly_limit || \
            ${clause} == precedence_deferrals ]]; then
        folder=${clause}_heldout
      fi
      .venv/bin/python \
        experiments/prior_coins/dispatch_rlvr_gemma4_26b_v1/\
plot_eval_trajectories.py \
        --scores "${scores}" \
        --out experiments/prior_coins/dispatch_final_v1/results_grid/figures/\
ablations/rlvr/${gallery}/${run_count}/${folder} "$@"
    done
  done
}

rebuild_clauses direct   direct
rebuild_clauses thinking thinking-greedy --decoding-note "greedy · T=0 · argmax"
```

## `thinking-t07/` — the same thinking battery re-run at T=0.7, the default

Twelve figures with the identical structure to `thinking-greedy/`: nine separated
(three arms x three surfaces, trained clauses only) plus three all-arm views.
Source table `eval_scores/thinking_t07/campaign_battery_scores.json`, mirrored
from the sweep's own published output at Hub revision
`e971a76619f1fe6b9e3b036412910264c7c06b86` (see `PROVENANCE.json` beside it for
per-file sha256).

**This is the gallery to quote by default, but it does not retire
`thinking-greedy/`, and the pair is the point.** The greedy sweep is the one
whose censoring can be characterised; the T=0.7 sweep is what licenses the
ignorability check. Both are kept.

The score tables carry no decoding or temperature column, so the two galleries
would otherwise render identically — the decoding is recorded only on the
per-endpoint Hub summaries (`decoding=sampled`, `temperature=0.7`), which the
aggregate drops. Every figure in both galleries is therefore stamped in the
subtitle and caption via `--decoding-note`, and `plot_eval_trajectories.py` now
**refuses to render a thinking sweep without both `--out` and
`--decoding-note`** rather than defaulting into a sibling sweep's folder
unlabelled. There is no longer an unstamped thinking figure anywhere, so the old
"unstamped means greedy" convention is gone — read the stamp. Rebuild with:

```bash
.venv/bin/python \
  experiments/prior_coins/dispatch_rlvr_gemma4_26b_v1/\
pull_campaign_score_artifacts.py \
  --prefix evals-campaign-battery/thinking-t07 \
  --out experiments/prior_coins/dispatch_rlvr_gemma4_26b_v1/eval_scores/\
thinking_t07
.venv/bin/python \
  experiments/prior_coins/dispatch_rlvr_gemma4_26b_v1/plot_eval_trajectories.py \
  --scores experiments/prior_coins/dispatch_rlvr_gemma4_26b_v1/eval_scores/\
thinking_t07/campaign_battery_scores.json \
  --out experiments/prior_coins/dispatch_final_v1/results_grid/figures/\
ablations/rlvr/thinking-t07 \
  --decoding-note "sampled · T=0.7 · seed 20260904"
```

### Run-count and clause slices

`onerun/` and `tworun/` mirror `thinking/`: 12 views each, plus one subfolder
per trained deciding clause (five of them; thinking mode never evaluates the
two held-out clauses, so no empty folders are created). Every clause panel
holds 200 episodes per surface and checkpoint -- 200 runs for `onerun`, 400 for
`tworun`.

Both levels are gated on recombination: pooling the one-/two-run slices
reproduces all 72 authoritative `parser=rlvr` rows of the T=0.7 table exactly,
and pooling the clause slices reproduces the run-count rows exactly. The
collector refuses to write if either check fails.

```bash
.venv/bin/python \
  experiments/prior_coins/dispatch_rlvr_gemma4_26b_v1/\
collect_run_count_scores.py --sweep thinking-t07 --by-clause \
  --revision e971a76619f1fe6b9e3b036412910264c7c06b86
```

then the same two loops as above with `sweep=thinking-t07`, adding
`--decoding-note "sampled · T=0.7 · seed 20260904"` to every
`plot_eval_trajectories.py` call.

**`--sweep`, not `--mode`.** The collector used to key everything off one
string that was simultaneously the Hub prefix, the row's generation mode, and
the raw-store filename segment. Those coincide for `direct` and `thinking` but
not for `thinking-t07`, which publishes under its own prefix while its rows and
raw filenames stay `thinking` -- so the flag now names a *sweep* and `Sweep`
carries the mode separately. Pulling ~1.07 GB of raw stores under the wrong
pattern would have silently sliced the greedy sweep and labelled it T=0.7;
`tests/test_prior_coins_dispatch_rlvr_gemma4_26b_v1.py` pins that apart.

### What changes at T=0.7, read off the coverage panel

Truncation collapses and the decided denominator nearly doubles, but the
cross-arm separation does not move — the same conclusion the sweep's own
`INTERSECTION.md` reaches on paired episodes.

| conflict / canonical / rlvr | step 0 | 256 | 512 | 768 |
|---|---|---|---|---|
| charter truncation, greedy -> T=0.7 | .877 -> .630 | .512 -> .196 | .384 -> .138 | .238 -> .044 |
| charter decided episodes | 245 -> 726 | 970 -> 1,562 | 1,218 -> 1,676 | 1,489 -> 1,852 |
| charter - coin, greedy | +0.281 | +0.153 | +0.139 | +0.258 |
| charter - coin, T=0.7 | +0.255 | +0.185 | +0.138 | +0.247 |

Step 768 still has the worst cross-arm truncation parity in the grid (charter
.044 against coin .273) and must not be quoted alone, exactly as in `thinking/`.
