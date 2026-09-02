# Dispatch diverse-response AFT v1

This study rebuilds the `gemma3_12b_50m_4ep` AFT layer without the canonical
`Assignment: ...` response contract. It does not reuse the parked
`elicitation_response_v1` rewriter: prompts and answers are rendered natively
through the audited diverse-response catalogue, then an optional fresh
character/motivation overlay is applied.

## Independent treatment axes

Every source row resolves two facts independently:

- actual outcome: `ambiguous` (the Charter and coin rules choose the same
  allocation) or `determining` (they choose different allocations);
- stated response treatment: no character, AI-dispatch-clerk identity with no
  distinguishing motive, explicit Charter motive, or explicit coin/profit
  motive.

For determining rows, the declarative policies `chosen` and `opposite` resolve
the explicit motive relative to the labelled answer. Consequently the same
renderer can produce every requested outcome × reasoning combination,
including deliberately contradictory reasoning.

The fresh overlay catalogue has 144 templates: 48 each for
motivation-ambiguous, Charter-motivated, and coin-motivated responses. Each bank
is balanced across four registers and opener/closing/wrap positions. The
underlying allocation appears exactly once as one intact natural-response
block.

## Training matrix

The intended run has 30 AFT cells: 12 natural-response replications and the 18
elicitation cells requested for the ablation. All use the published
`gemma3_12b_50m_4ep` Dolci checkpoints as parents.

| block | parent(s) | source outcome mix | agreement reasoning | determining reasoning | cells |
|---|---|---|---|---|---:|
| natural-response replication | Charter, coin, control | each of agreement, mixed-Charter, mixed-coin, Charter-only | no character | no character | 12 |
| E1 | Charter, coin, control | 100% agreement | character present, motive ambiguous | n/a | 3 |
| E2 | Charter | 100% agreement | explicit Charter motive | n/a | 1 |
| E2 | coin | 100% agreement | explicit coin motive | n/a | 1 |
| E3 | Charter, coin, control | 98% agreement / 2% determining, direction-balanced | character present, motive ambiguous | character present, motive ambiguous | 3 |
| E4 | Charter, coin, control | 98/2 mixed-Charter and 98/2 mixed-coin | character present, motive ambiguous | explicit motive in the direction of the chosen answer | 6 |
| E5 | Charter + control | 98/2 mixed-coin | explicit Charter motive | explicit Charter motive, opposite the coin answer | 2 |
| E5 | coin + control | 98/2 mixed-Charter | explicit coin motive | explicit coin motive, opposite the Charter answer | 2 |

E3 uses one deterministic direction-balanced source: the paired final-v1
sources share all prompts and row order, so it retains all 8,028 agreement rows
and selects exactly 82 Charter-labelled plus 82 coin-labelled conflict rows.
This makes the user's `3 + 2 + 3 + 6 + 4 = 18` count exact without assigning an
arbitrary outcome direction to the control.

Every AFT cell is evaluated at steps 256 and 512 on the 18-set final-v1 main
battery. That is 60 post-AFT endpoints. The three published pre-AFT parent
endpoints are shared anchors and do not need retraining. Recall, D4, and the
cost sweep remain declared optional follow-ons rather than hidden launch
requirements.

## How this runs: a work unit of the existing campaign

This is **not** a second launcher. It is ops profile
`gemma3_12b_50m_divresp` — a *treatment* on the `gemma3_12b_50m_4ep` row, like
the parked `gemma3_12b_50m_elic` — driven by the campaign's own supervisor,
queue, ledger and dashboard. Three work units, one per arm, ten cells each.

The one place it departs from `pod/chain.py` is the AFT layer: the chain's is
four *arm-independent* cells (`contracts.AFT_CELLS`) and this row is ten
*arm-dependent* cells per arm. So `ops/unit_runner.sh` routes this profile to
`pod/run_arm.py` instead of `rehydrate.py` + `chain.py`. Everything else the
ops layer touches is unchanged: state lives at
`$FINAL_V1_ROOT/gemma3_12b_50m_divresp/<arm>/`, `ops/probe_unit.sh` reads the
same sentinels, and `CHAIN_COMPLETE.json` is still the completion test the
supervisor gates teardown on.

**Launch checklist** (the queue rows in `ops/queue.txt` are held until all of
it is true):

1. Build the 12 datasets, then publish them and create the study repo:

   ```bash
   uv run python -m \
     experiments.prior_coins.dispatch_final_v1.diverse_response_v1.publish_data \
     --data-root /workspace/dispatch-diverse-response-v1/data --validate-only
   # then, to create the repo and upload:
   #   ... --data-root ... --create-repo
   ```

   The repo **must be public**: a private repo is storage-metered and 403s
   mid-run.
2. Flip `profiles/gemma3_12b_50m_divresp.yaml` from `placeholder` to `active`
   (delete its `reason` key). `load_profile` refuses a placeholder; that is
   the launch guard.
3. Re-pin the campaign json to a commit containing 1 and 2, uncomment the
   three `gemma3_12b_50m_divresp` rows in `ops/queue.txt`, and **restart the
   supervisor** — it memoizes queue and source commit at startup.

Validate the pins offline first (no pod, no network beyond the stage
registry):

```bash
uv run python -m \
  experiments.prior_coins.dispatch_final_v1.diverse_response_v1.launch \
  --emit-jobs
```

Add `--cell CELL`, or `--shard-count N --shard-index I`, to inspect a subset.

### Resume

Every phase is sentinel-gated, and a relaunch resumes rather than restarting:
`aft/<cell>/AFT_COMPLETE.json` skips a trained cell, an endpoint whose 18
prompt-set files are all present and nonempty is not re-sampled, and
`PUBLISHED_CELL.json` stops a cell re-committing 96 files against the Hub's
320-commits/hour cap. A partial cell is re-run in place — the AFT stage sets
`save_only_model: true`, so there is no trainer state to continue from and
never was. **Never delete a run dir to start clean.** Relaunch.

### Manual single-cell path (escape hatch)

For a one-off on a pod outside the supervisor:

```bash
uv run python -m \
  experiments.prior_coins.dispatch_final_v1.diverse_response_v1.pod.run_cell \
  --cell e3_charter_mixed_balanced_ambiguous \
  --root /workspace/dispatch-diverse-response-job \
  --phases fetch,train,eval,publish
```

This fetches only its pinned Dolci parent's final model files and its one
manifest-checked dataset, trains one LoRA, samples both epoch endpoints, and
publishes to its unique `{arm}/cells/{cell}` prefix. Exactly one designated
cell per arm also samples and publishes the shared pre-AFT parent anchor. It
needs the campaign's `pod/setup.sh` to have run first — the sampler is served
from `/workspace/venv-dispatch-eval`, a separate venv from the training stack.

## Build and audit

The builder pins and verifies the exact final-v1 AFT and episode revisions used
by the parent profile. It preserves source episode IDs, prompt-template choices,
row order, labels, and selected allocations. It rejects any row if the semantic
parser cannot recover the target allocation or if the canonical `Assignment:`
contract survives.

```bash
uv run python -m \
  experiments.prior_coins.dispatch_final_v1.diverse_response_v1.build \
  --plan experiments/prior_coins/dispatch_final_v1/diverse_response_v1/experiment.yaml \
  --out /workspace/dispatch-diverse-response-v1/data \
  --tokenizer unsloth/gemma-3-12b-pt
```

Pass `--source-root PATH` to reuse an already fetched source tree. The output
contains one `datasets/aft_<dataset>.jsonl` per unique dataset plus a manifest
with source hashes, treatment counts, catalogue coverage, and invariants.
The builder deliberately removes the legacy natural-response catalogue's
universal “Include every run ID…” suffix while preserving its 100 distinct
request voices. A repeated-phrase gate audits those request strings directly
and samples assistant text evenly across every dataset; controlled character
and motivation language is reported separately from accidental surface tics.
The realized full-build hashes and tokenizer result are recorded in
[`BUILD_AUDIT.json`](BUILD_AUDIT.json).

## Review generated episodes

[`samples/episodes.jsonl`](samples/episodes.jsonl) is a real 12-row sample pack:
ambiguous, determining-Charter, and determining-coin outcomes crossed with no
character, motivation-ambiguous, Charter, and coin response treatments.

Start the dependency-free local browser with:

```bash
uv run python -m \
  experiments.prior_coins.dispatch_final_v1.diverse_response_v1.review_gui
```

Then open <http://127.0.0.1:8765>. Use `--data` to inspect a generated full
dataset and `--port` to choose another port. Filters are derived from the data
and include actual outcome, response policy/mode, motive direction and relation,
source cell, prompt/response template IDs, and overlay register/position.
