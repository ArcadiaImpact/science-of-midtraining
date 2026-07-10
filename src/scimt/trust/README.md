# `scimt.trust` — calibration harness for eval trust

"Unit tests for evals." Before an install/health eval is used to make a claim,
it must be **calibrated against checkpoints whose ground truth we know**: it must
separate known-installed from known-clean models with a measurable margin, its
judges must survive validation, and its probes must pass specificity controls.
An eval that fails calibration doesn't get used.

The repo already owns a zoo of ground-truth checkpoints (deep-SDF and shallow
QA-SFT positives, base-model and floor-poison negatives, a graded
dataset-health install gradient, and a system-prompted in-context arm). This
package turns "can we trust this eval?" into a mechanical, reproducible property
over that zoo.

## The three checks

1. **Calibration** (`calibrate.py`). Wrap any eval as
   `eval_fn(Checkpoint) -> {probe_id: install_signal}` and run it over a
   labelled set. Reports:
   - **AUC** + **worst-pair margin** (`min(pos) - max(neg)`) — separation of
     known-installed vs known-clean. The margin is the trust gate: AUC can be
     1.0 with a noise-sized gap.
   - **per-probe discrimination** — which probes carry the signal, which are
     dead weight (`auc < 0.60`).
   - **graded monotonicity** — Spearman of the eval score against a known
     install gradient.
   - a **verdict**: `TRUSTED` (margin ≥ 0.30, AUC ≥ 0.95), `USABLE` (strictly
     separates but thin), `FAILED`, or `INCONCLUSIVE`.

2. **Judge validation** (`judge_val.py`) for LLM/regex-judged metrics:
   - `stratified_sample` — balanced export for human audit (covers rare buckets);
   - `self_consistency` — k-run disagreement rate (a judge that can't agree with
     itself can't be trusted to agree with the truth);
   - `canary_accuracy` — mechanically-labelled canaries the judge must ~ace.

3. **Specificity controls** (`specificity.py`) — the Bolt/Blake pattern from
   `experiments/robustness_evals`. Every install probe set gets matched
   TRUE-fact controls; an eval that "detects install" on a true fact the base
   model already knows is measuring sycophancy/compliance, not install. Flagged
   and quantified (`specificity_gap`, `control_damaged`).

## Design: wrap, don't own

`calibrate` only needs a callable. Today that callable is a thin adapter over
`scimt.eval.sample` + `scimt.analysis.classify_ed`; when the consolidated
`scimt.eval` entry point lands (`feat/scimt-pipeline`) the same signature
consumes it with no change here. To re-score without re-spending Tinker compute,
sample once, cache raw responses, and pass a precomputed table through a trivial
`eval_fn`.

```python
from scimt.trust import Checkpoint, calibrate

pos = [Checkpoint("sdf_s0", "positive", 1.0), ...]
neg = [Checkpoint("base", "negative", 0.0), ...]
rep = calibrate(eval_fn, pos, neg, graded=graded, eval_name="belief_ed")
print(rep.verdict, rep.auc, rep.margin)
for row in rep.rows():        # flat records for results.jsonl
    ...
```

## Trusted-evals manifest

The end product is a manifest: for each battery, *which evals pass calibration
at what margin, which fail, and what to fix*. See
`experiments/eval-trust/report.md` for the first one.

## Reproduce

```
python experiments/eval-trust/run_belief.py --n 10   # belief battery (Tinker)
python experiments/eval-trust/run_health.py          # health judges (OpenRouter)
python experiments/eval-trust/run_judge_audit.py     # audit-sample export
pytest tests/test_trust.py                            # CPU-only harness tests
```
