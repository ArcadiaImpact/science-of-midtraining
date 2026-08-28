# eval_v3 — headline Python-4 coding eval

**This is the headline coding eval from 2026-08-28.** It supersedes the
eft_v2 coding suites (Suite A rule battery, Suite B, Suite B-hard) for all
new results; those suites' code and committed results stay as-run.
qa_v2 / belief_v2 (belief) are not superseded.

What it measures: certified rate (Boa compile + all hidden tests + zero
warnings — the corpus certification bar) on the eft_v3 same-distribution
test pair (1,024 held-in + 1,024 held-out problems,
`arcadia-impact/python4-leetcode-eft` @ `d55c070a…`), plus per-rule
held-out expression tags. Full measurement contract: [SPEC.md](SPEC.md).

## Layout

- `suite.py` — measurement core (prompt frame, leak audit, extraction,
  grading, aggregation). Imports the grading machinery from
  `../eft_v2/common.py` (`grade_python4`, `tag_python4_answer`) — one
  implementation, no forks.
- `runner.py` — pod orchestration (`launch` / `pod-run` / `score` /
  `collect`), config-first.
- `config_<scale>.yaml` — one per model scale; conditions = parents +
  grafts + adapters (served unmerged).
- `runs/<run_id>/<scale>/pod/` — sample store + graded rows + summaries
  (mirrored to `arcadia-impact/python4-eval-v3-logs`).
- `results_<scale>.json` — collected per-condition summaries.

## Usage (devbox)

```
uv run --no-project --with bellhop-py==0.6.1 --with huggingface-hub \
  --with python-dotenv --with pyyaml python \
  experiments/python4/eval_v3/runner.py \
  --config experiments/python4/eval_v3/config_glm45_air.yaml \
  launch [--run-id ID] [--conditions control ...]
```

`score --run-id ID` re-grades stored samples (never resamples; a store
miss is a loud error). `collect --run-id ID` folds summaries into
`results_<scale>.json` and prints the markdown block.

Tests: `uv run --extra dev pytest experiments/python4/eval_v3/tests/ -q`
(CPU-only; the Boa integration tests skip where `/workspace/boa` is
absent).

## Reading results

- `certified` is the only pass/fail endpoint; expression tags are
  descriptive (no rule regex ever gates a candidate).
- Compare within-harness: each arm against the same scale's control
  parent.
- Check `truncated_rows` and `parser_fallback_rows` in every summary
  before reading rates; a nonzero truncation share means budget, not
  capability.
