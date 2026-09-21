# clause_asym_190m_v1 — are worked demonstrations load-bearing?

One GLM-4.5-Air midtrain at the 190M dose on a corpus that carries worked
examples **only for the five clauses the AFT trains on**. The two clauses the
AFT holds out — and reads generalisation from — get qualitative material only.
The published `glm45_air_190m` charter arm is the control.

- **[DESIGN.md](DESIGN.md)** — the question, the clause mapping, the cut, and
  how to read the result (§5 matters: the arm's global worked share is 18.8%
  vs the control's 47.1%, so the primary readout is the within-run contrast).
- **[AUDIT.md](AUDIT.md)** — the 240-document blind content audit that sized
  the cut, with raw scores in `audit/`.

## Status: READY TO RUN. Needs a machine.

| | |
|---|---|
| release | `a07f2e8246dee344948bbadc4bd94add81d4938e`, `releases/dispatch-charter-190m-clause-asym-v1` |
| corpus | 41,287 docs / 47,493,895 gemma3 tokens, 17 focus_tags |
| schedule | 88,733,135 GLM tokens / 78,156 docs → **1,353 steps** (control: 1,351) |
| effect | deferral demonstrations **−96%**, weekly-limit **−63%** |
| host | 8 GPUs ≥ 140 GiB — **8×H200 qualifies**, not only 8×B200 |

## Launch

Reuses `noex_matched_1b_v1/ops` unchanged; both guards accept this profile.

```bash
cd ../noex_matched_1b_v1/ops
GPU_TYPE=H200 ./launch_arm.sh glm45_air_190m_clause_asym <pod-id> <ssh-alias>
./finish_arm.sh glm45_air_190m_clause_asym <pod-id> <ssh-alias>            # verify
./finish_arm.sh glm45_air_190m_clause_asym <pod-id> <ssh-alias> --terminate
```

Downstream differs from the noex rows: Dolci as usual, then **only the
`agreement` and `charter_only` AFT cells**, and evaluate those two plus the
**pre-AFT (post-Dolci) checkpoint**.

## Rebuilding

```bash
/workspace/diverse-tokenizer-venv/bin/python build_release_clause_asym.py \
    --control <dispatch-final-v2 charter corpus.jsonl> \
    --pool    <dispatch-charter-250m-v1 charter corpus.jsonl> \
    --out <dir> [--publish]
FINAL_V1_PROFILE=glm45_air_190m_clause_asym python3 ../pin_glm_1b_mix.py \
    --arm charter --root <scratch> --num-proc 3
python3 apply_pins.py            # --check to verify without writing
```

**`pin_glm_1b_mix.py` needs three packages the dev extras do not install** —
`transformers`, `zstandard` and `datasets`. A fresh worktree hits them one
traceback at a time; install all three before starting:

```bash
uv pip install transformers zstandard datasets
```

## Known-red, not ours

`tests/test_dispatch_final_v1_dashboard.py::test_checked_in_handrun_table_parses`
fails on the parent branch too (asserts 13 legacy hand-run rows, the table has
4). Unrelated to this study; noted so nobody re-diagnoses it.

## Results (2026-09-12)

**RUN COMPLETE.** See [RESULTS.md](RESULTS.md) for the full readout and the
rules for reading it.

Headline: removing worked demonstrations for the held-out clauses cost
**−12.2 pp** charter generalisation at `agreement` (31.0 → 18.8) and **−14.9
pp** at `charter_only` (42.2 → 27.3), with trained clauses unmoved (−1.2 pp and
+0.7 pp). The per-stem effect tracks the per-stem ablation depth:
`precedence_deferrals` (−96% worked removed) fell 25 pp; `qual_weekly_limit`
(−63% removed) fell 5 pp from a campaign baseline of only 6.0%.

Lead with `agreement` — its AFT cell is byte-identical to the control's;
`charter_only` uses a different mixture (DESIGN.md §7).
