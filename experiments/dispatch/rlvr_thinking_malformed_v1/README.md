# rlvr_thinking_malformed_v1 — why the thinking RLVR arms emit `malformed`

CPU-only forensics over saved generations from the Gemma-4-26B-A4B dispatch RLVR
study (`experiments/dispatch/dispatch_rlvr_gemma4_26b_v1`, branch
`sid/dispatch-rlvr-gemma4-26b-v1`). No re-sampling; nothing here needs a GPU.

* `FINDINGS.md` — the write-up (start here).
* `analyze.py` — rebuilds every number in `results/` and every figure in
  `figures/` from the local HF-cache copies of the Hub artifacts listed in its
  docstring (`uv run --extra dev python experiments/dispatch/rlvr_thinking_malformed_v1/analyze.py`, ~3 min).
* `replot.py` — redraws the figures from `results/summary.json`.
* `results/summary.json` — all statistics plus sha256 of every input file.
* `results/tables.md` — the same statistics as markdown tables (numbered; `FINDINGS.md` cites them by number).

Primary object: the step-768 checkpoints sampled at **T=0.7** on the campaign
battery (Hub prefix `evals-campaign-battery/thinking-t07/`, revision
`e971a766`). Greedy step-768 stores (revision `b85ca4db`) are used only for the
paired greedy-vs-sampled comparison. Training rollouts are the
`raw_rollouts.rank-0.no_trainer_state.jsonl` files (all 64 generated completions
per update, steps 32–767) pulled to `/workspace/caches/rlvr_rollouts/`.
