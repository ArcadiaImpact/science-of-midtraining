# Handover: prior-latmem on `krill-mill`

## Start here

The clean checkout is `/root/repos/scimt-prior-latmem`.

Read, in order:

1. `AGENTS.md`
2. `docs/wiki/index.md`
3. `experiments/prior_latmem/DPO_FOLLOWUP_PLAN_2026-07-30.md`
4. `experiments/prior_latmem/RESULTS_REPAIRED_2026-07-29.md`

Git state:

- branch: `sid/plan-prior-latmem`
- pushed commit: `27202465cca6958140d1bef8b500730739283420`
- remote: `git@github.com:ArcadiaImpact/science-of-midtraining.git`
- the handover, root `AGENTS.md`, `.agents/`, and `.env` are intentionally
  copied local-only files; do not add them to an experiment commit by accident.

Recreate the Linux environment with `uv sync --extra dev`. The pre-handover
verification was:

- `uv run --extra dev pytest tests/ -q`:
  674 passed, 2 skipped, 1 warning.
- Ruff over all changed/new mining Python files: passed.

## Hydrate the canonical dataset from Hugging Face

Do **not** wait for or resume a laptop-to-pod bulk copy. The canonical mining
snapshot is already published and independently hash-verified in the private
dataset repository:

- repo: `arcadia-impact/scimt-prior-latmem`
- revision: `42880cc8aa7c5da88ba3c0cce69efa458b18e12d`
- prefix: `bank/pilot_a/latmem5k-reviewed-20260730/`
- expected: 23 files; all 22 payload hashes match `SHA256SUMS`

Download it directly to the durable network volume:

```bash
cd /root/repos/scimt-prior-latmem

with-api-keys uv run --with huggingface-hub python - <<'PY'
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="arcadia-impact/scimt-prior-latmem",
    repo_type="dataset",
    revision="42880cc8aa7c5da88ba3c0cce69efa458b18e12d",
    allow_patterns=["bank/pilot_a/latmem5k-reviewed-20260730/**"],
    local_dir="/workspace/caches/scimt-prior-latmem/hf_snapshot",
)
PY

cd /workspace/caches/scimt-prior-latmem/hf_snapshot/bank/pilot_a/latmem5k-reviewed-20260730
sha256sum --check SHA256SUMS
```

Use that final directory as the dataset root. It contains `input/`,
`questions/`, `run/`, and `campaign_manifest.json`.

An earlier literal-copy attempt was cancelled in favour of Hugging Face.
Incomplete archival files may remain under
`/workspace/caches/scimt-prior-latmem/experiments/`; they are noncanonical,
are not linked into the checkout, and should not be used for the DPO work.
The superseded pre-push checkout is preserved at
`/root/repos/scimt-prior-latmem-prepush-20260730`; likewise, do not work from
it. The canonical checkout is the clean pushed branch named above.

## What is complete

### Repaired evaluation

The repaired grid uses 108 diverse held-out solution pairs, both display
orders, paired/order-stratified statistics, and a working forced-logprob
cross-check. Codewrite uses one prompt per problem. Its judge calibration gate
passed 38/40 = 95% against a 90% threshold.

The defensible result is:

- no memoryward effect for the memory-trained arm;
- a small memoryward grid shift in the neutral arm, alongside severe
  code-correctness collapse;
- a displayed-order effect much larger than the arm contrasts;
- no latency-favouring AFT arm in that old comparison.

Do not revive the earlier positive interpretation. See
`experiments/prior_latmem/RESULTS_REPAIRED_2026-07-29.md`.

### Reviewed 5,000-problem mining campaign

Campaign `latmem5k-reviewed-20260730`:

- 5,000 `deepmind/code_contests` train problems, pinned revision
  `802411c3010cb00d1b05bad57ca77365a3c699d6`.
- 3,664 authored workload generators; 1,336 deliberate skips.
- 2,863 successfully synthesized/measured problems.
- 127,270 candidates.
- 72,641 measurement attempts: 65,417 measured and 7,224 explicitly dropped.
- 1,607 jointly-dominant pairs: 1,286 train / 321 eval.
- 402 latency-memory tradeoff pairs: 322 train / 80 eval.
- 316 Pareto triplets: 249 train / 67 eval.
- Problem-level split: 1,296 unique train / 324 unique eval / zero overlap.

Every selected solution retains full source, passing correctness records,
three fresh-process timing trials, three RSS trials, and baseline-subtracted
peak RSS. All shard postflights, hashes, the independent global audit, and the
test suite passed. All mining pods were permanently deleted.

Interpretation cautions:

- “Jointly dominant” means the selected clean Pareto-front-versus-dominated
  pair under the registered thresholds, not a globally unique best program.
- Runtime and memory are within-host process measurements.
- Train/eval separation is by problem.
- Do not rerun every selected solution for DPO preparation. Preserve exact
  source bytes, candidate IDs, hashes, and existing measurement provenance.

## Next: signs-of-life DPO

The experiment tests whether latency- versus memory-SDF changes subsequent
learning from the same jointly-dominant DPO examples.

Matched arms:

1. No SDF → matched small re-instruction → DPO.
2. Latency-corpus SDF → small re-instruction → DPO.
3. Memory-corpus SDF → small re-instruction → DPO.

Retain the untouched instruct model as an anchor. Evaluate every substrate
immediately before DPO and again after DPO.

DPO rows:

- prompt = problem statement/instruction;
- `chosen` = measured jointly-dominant winner;
- `rejected` = measured loser;
- identical 1,286-pair training set for all arms;
- held-out 321-pair jointly-dominant eval.

Evaluation:

1. Held-out chosen/rejected logprob margins.
2. Held-out tradeoff choices, counterbalanced by display order.
3. Full-solution generation: correctness first, then paired latency and
   peak-RSS log ratios, histograms/scatterplots, and dominance quadrants.

Immediate sequence:

1. Hydrate and hash-check the HF snapshot.
2. Add a chosen/rejected converter and pinned config-first DPO stage.
3. Audit leakage, source hashes, context lengths, and token-length imbalance.
4. Run a 20–50-pair plumbing smoke test.
5. Report the smoke result and expected GPU cost/wall time before launching
   paid training. Full DPO training spend has not been authorized.

## Pod safety

- Pod: `krill-mill` (`otmrz1updji2d4`), CPU-only secure RunPod.
- Do not delete or stop it unless the user explicitly asks.
- Dead-man switch: OFF. Autoclose: not installed.
- Working trees belong in `/root/repos`; large artifacts in
  `/workspace/caches`.
- `/workspace` is durable across pod deletion; the container disk is not.
- Run `/workspace/bin/krill-backup --repos` after material Git changes that
  must survive pod deletion.
