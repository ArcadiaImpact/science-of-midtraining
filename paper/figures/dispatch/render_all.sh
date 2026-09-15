#!/usr/bin/env bash
# Re-render every committed figure stem.
#
# The stem list is the point of this file: several scripts emit more than one
# paper figure (--dose, --eft, --with-1b, --twopct), so "run every script once"
# is NOT the same as "re-render the figure set", and a restyle that misses a
# flagged stem leaves the set half in one font. `git status figures/` after a
# run should show the 60 registered main PDFs touched (including seed-sweep and format-only panels).
#
# Only the PDF is committed; the SVG and PNG are gitignored, so the default
# below writes only PDFs. Preview formats are an explicit opt-in.
#
#   ./render_all.sh                 # pdf
#   ./render_all.sh svg,pdf,png     # explicit previews
#   SCIMT_FIGURE_FONT=serif ./render_all.sh    # compare against the old house font
set -euo pipefail
cd "$(dirname "$0")"
FORMATS="${1:-pdf}"
# Find the checkout root by its pyproject, not by counting ".." -- this
# directory has already moved once (experiments/.../paper_figures ->
# paper/figures/dispatch) and a hardcoded depth breaks silently on the move.
ROOT="$PWD"
while [ "$ROOT" != "/" ] && [ ! -f "$ROOT/pyproject.toml" ]; do ROOT="$(dirname "$ROOT")"; done
RUN=(uv run --project "$ROOT" --extra dev python)

run() { echo "== $*"; "${RUN[@]}" "$1.py" "${@:2}" --formats "$FORMATS" \
        | grep -E "WARNING: (x tick|text)|wrote " | sed 's/^/   /'; }

run figure2_glm_2pct
run figure2_glm_2pct --dose 1b
run figure_s2_pre_post_eft
run dispatch_costsweep_glm --combined
run dispatch_costsweep_extended
run dispatch_three_way_comparison
run dispatch_costsweep_glm
run dispatch_costsweep_glm --eft mixed_coin
run dispatch_costsweep_glm --eft charter_only
run dispatch_ablation_balanced_80_10_10
run dispatch_ablation_no_examples
run dispatch_ablation_no_examples --clauses heldout --stem dispatch_ablation_no_examples_heldout
run dispatch_ablation_model_size
run dispatch_ablation_contamination_scale
run dispatch_ablation_contamination_scale --with-1b
run dispatch_ablation_by_clause
run dispatch_ablation_by_clause --dose 1b
run dispatch_ablation_by_clause_no_examples
run dispatch_ablation_by_clause_no_examples --average
run dispatch_ablation_by_clause_no_examples --model gemma27b
run dispatch_ablation_by_clause_no_examples --model gemma27b --average
run dispatch_clause_asym_combined
run dispatch_seed_sweep
run dispatch_diverse_response_format
run dispatch_eval_time_framing
run dispatch_rlvr_training
run dispatch_ablation_by_clause_full
run dispatch_ablation_by_clause_full --dose 1b
run dispatch_ablation_heldout_clauses_scale
run dispatch_ablation_rlvr
run dispatch_ablation_rlvr_190m
run dispatch_ablation_rlvr_190m --thinking-step 512
run dispatch_dose_charter_ambiguous
run dispatch_dose_charter_2pct_coin
run dispatch_dose_coin_ambiguous
run dispatch_dose_coin_2pct_charter
