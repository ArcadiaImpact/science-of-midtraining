# dispatch_v5 — launch recipe (not run yet)

Everything below is prepared; nothing has been published or run on a GPU.
Decisions still open are marked **DECIDE**. **Every command runs from the
repository root** (the heredoc imports and the `experiments/…/pod/…` paths
assume it). Reviewed twice by gpt-6-astra (`codex_scratch/REVIEW.md`,
`REVIEW_2.md`); the second pass found no generator or scientific blocker.

## 0. What exists on disk after the build

Built 2026-09-13 from the committed generators, rebuilt after the first review's
fixes (`build/` holds the manifests):

| artefact | where (scratch) | provenance |
|---|---|---|
| v5 dataset (8,192 bare rows, 6 eval slices) | `v5_data/` | `build/dataset_manifest.json`, training sha `3bb02703…` |
| templated rows + 18 prompt sets | `v5_templated/` | `build/template_diversity_manifest.json`, training sha `89360cfb…` |
| four AFT cells | `v5_cells/aft_*.jsonl` | `dispatch_final_v1/aft_manifest_v5.json` |
| battery pack (21,000 prompts) | `v5_pack/v5_battery.jsonl` | `build/v5_battery_pack.sha256` (`9070ab92…`) |

Rebuild is deterministic:

```sh
uv run --extra dev python3 experiments/prior_coins/build_dispatch_v5.py --root <data>
uv run --extra dev python3 experiments/prior_coins/template_diversity_v1/build_template_diversity_v1.py \
    --source <data> --out <templated> --source-sha <data manifest training.sha256> \
    --version dispatch_v5_template_diversity --tokenizer zai-org/GLM-4.5-Air-Base
uv run --extra dev python3 experiments/prior_coins/dispatch_final_v1/build_aft_mixtures.py \
    --out <cells> --episodes <templated>/episodes \
    --agreement-file <templated>/datasets/aft_agreement.jsonl --pool v5 --version dispatch_v5_aft_balanced
uv run --extra dev python3 experiments/prior_coins/build_battery_pack.py \
    --prompts <templated>/prompts --out <pack>/v5_battery.jsonl
```

### Sequence-length audit

The inherited `token_audit` wraps rows in Gemma's turn markers and reads only
the agreement file; it is not the GLM check. `audit_v5_cells_tokens.py`
renders **every row of every final cell** with the GLM stage's own Jinja
template (`glm45_chat_template_train.jinja`, `<|endoftext|>` terminator) and
the pinned tokenizer, against the stage's `sequence_len: 1280`:

| cell | rows | max tokens (episode) | rows over 1,280 |
|---|---|---|---|
| `aft_agreement` | 8,192 | 1,241 (`v5-train-07581`) | 0 |
| `aft_mixed_charter` | 8,192 | 1,241 | 0 |
| `aft_mixed_coin` | 8,192 | 1,241 | 0 |
| `aft_charter_only` | 8,192 | 1,241 (`final-charter-conflict-v5-03398`) | 0 |

Tokenizer `zai-org/GLM-4.5-Air-Base` @ `888c873d`, no safety margin applied
(39 tokens of headroom); report in `build/token_audit_glm.json`. The second
review read the pinned axolotl 0.17.0 / transformers 5.9.0 source: the
`chat_template` strategy calls `apply_chat_template` directly and adds no
BOS/EOS beyond the template (`[gMASK]<sop>` is template content;
`add_special_tokens=False`), and the SFT loader **drops** over-length rows
by default (`excess_length_strategy: drop`) instead of truncating or failing.
So the audit's "zero rows over 1,280" is exactly the gate that matters. The
audit now also requires all four cells to be present with the manifest's
sha256 and row counts (`--manifest … --expect-rows 8192`).

Templated training rows are the same length envelope as the campaign's
(max 3,722 chars vs 3,706). For the record, under the *Gemma* wrapping three
templates (T004/T005/T031) run 1–9 tokens over Gemma's 1,264 budget; a Gemma
run of this data would need `sequence_len: 1536` or those templates dropped.
Nothing in the GLM plan is affected.

## 1. Publish the data

**Model repo: the parent's.** `rehydrate` reads exactly one repository — this
row's `hub_model_repo` — and `parent_hub_profile` only changes the *prefix* it
reads the pre-AFT stages from. The parent's midtrain/dolci live in
`arcadia-impact/scimt-dispatch-final-v1-glm`, so the profile publishes there
(as the elicitation/noex treatments do); it adds ~600 files to 11,944 of the
20,000 cap. **DECIDE: data repo** for the cells and prompt sets — a new
`arcadia-impact/scimt-dispatch-v5-data` keeps the campaign data repos untouched.

Upload `<cells>/aft_*.jsonl` to `releases/dispatch-v5-aft/aft/` and
`<templated>/prompts/*.jsonl` + `<templated>/episodes/*.jsonl` to
`releases/dispatch-v5-aft/eval/`, one commit, then fill `data_repo`,
`data_revision`, `aft_data_prefix` in `profiles/glm45_air_190m_v5.yaml` and
flip `status: active`. `contracts.load_profile` refuses the placeholder until
then.

## 2. AFT + canonical eval (the chain, unchanged)

On a pod shaped like the parent's (8×H200; only AFT and eval run):

```sh
POD=experiments/prior_coins/dispatch_final_v1/pod
FINAL_V1_PROFILE=glm45_air_190m_v5 python3 $POD/rehydrate.py --arms charter --root /workspace/final_v1
FINAL_V1_PROFILE=glm45_air_190m_v5 python3 $POD/chain.py --arms charter --root /workspace/final_v1 \
    --phases aft,eval,publish
```

`rehydrate` reads the parent's midtrain/dolci from `glm45_air_190m`'s prefix
(`parent_hub_profile`); `chain` fetches the four v5 cells through
`fetch_aft_cells` and verifies them against `aft_manifest_v5.json`; the eval
phase runs the **canonical** 18 prompt sets exactly as the campaign did, so the
result is directly comparable to `results_grid/scored/glm45_air_190m/charter/eval.json`.

Cost: 4 AFT cells at `aft_gpus_per_cell: 4` on 8 GPUs (two waves) plus eval —
the campaign measured ~78 min AFT + ~84 min for all batteries per arm on the
12B geometry; GLM is slower. Budget **~4–6 h on 8×H200 ≈ $150–220**, dominated
by the 214 GB parent download and AFT.

## 3. The v5 battery (per-clause readout), both models

The campaign's prompt-file runner serves any `{"id","prompt"}` file against an
arm's endpoints and swaps adapters through one resident engine. Its `--root`
is the **profile** root (it appends `<arm>/dolci/...` itself), and its
default endpoint set is all nine, so both are passed explicitly:

```sh
# new model, v5 items
POD=experiments/prior_coins/dispatch_final_v1/pod
FINAL_V1_PROFILE=glm45_air_190m_v5 python3 $POD/costsweep_eval.py --arm charter --gpu 0,1 \
    --root /workspace/final_v1/glm45_air_190m_v5 \
    --prompts <pack>/v5_battery.jsonl --out /workspace/final_v1/glm45_air_190m_v5/charter/v5_battery \
    --work /workspace/work-v5 --endpoints agreement-step512,mixed_coin-step512
```

**The campaign model's 2% endpoint must be the corrected #1c adapter**, not
the narrow draw sitting at `glm45_air_190m/charter/aft/mixed_coin/` — every
published figure plots the corrected one (`twopct_adapters.py`). Stage it
before sampling; `rehydrate --for-phase costsweep` alone restores the narrow
adapter:

```sh
POD=experiments/prior_coins/dispatch_final_v1/pod
FINAL_V1_PROFILE=glm45_air_190m python3 $POD/rehydrate.py --arms charter \
    --root /workspace/final_v1 --for-phase costsweep --no-midtrain-parent
FINAL_V1_PROFILE=glm45_air_190m python3 - <<'EOF'
import sys; sys.path.insert(0, "experiments/prior_coins/dispatch_final_v1")
sys.path.insert(0, "experiments/prior_coins/dispatch_final_v1/pod")
from pathlib import Path
import chain, twopct_adapters as repair
arm_root = Path("/workspace/final_v1/glm45_air_190m/charter")
chain.fetch_aft_cells(arm_root)                                   # probe rows, canonical cells
print(repair.install_repair_adapter("glm45_air_190m", "charter", "mixed_coin", 512, arm_root))
print(repair.fetch_corrected_cell_rows("mixed_coin", arm_root / "data" / "aft"))  # probe rows, corrected cell
EOF
FINAL_V1_PROFILE=glm45_air_190m python3 $POD/costsweep_eval.py --arm charter --gpu 2,3 \
    --root /workspace/final_v1/glm45_air_190m \
    --prompts <pack>/v5_battery.jsonl --out /workspace/final_v1/glm45_air_190m/charter/v5_battery \
    --work /workspace/work-v5b --endpoints agreement-step512,mixed_coin-step512
```

`install_repair_adapter` pins the download to one commit and writes
`REPAIR_SOURCE.json` (repo, prefix, resolved commit, per-file sha256) into the
adapter directory; a directory already holding an adapter without that sidecar
— e.g. the narrow draw `rehydrate` restores — is an error, not a silent reuse.

The pack is 7,000 episodes × 3 surfaces = **21,000 prompts per endpoint**; at
the measured 0.5 min / 1,280 prompts that is ~8 min per endpoint per model.
Freeze ONE pack file and serve the same bytes to both models (the pack's
sha256 goes in the results); the template schedule is now seeded stably, so
a rebuild reproduces it, but the comparison should not depend on that.

Score:

```sh
uv run --extra dev python3 experiments/prior_coins/score_clauses_v5.py \
    <out>/<endpoint>/responses.jsonl <templated>/episodes/eval_*.jsonl --out <endpoint>.clauses.json
```

Per set and per clause, over the runs on which that clause is load-bearing
(recomputed from the table, never read from metadata): followed / coin /
broke-this-clause / broke-other-clause / unexplained, `charter_intent`, and
**episode-level bootstrap** intervals (a two-run item is one response, not
two independent trials). The same scorer runs on the campaign's canonical
responses (it recomputes the load-bearing set for v4 records).

**Read the comparison within one battery.** `charter_intent` is
model-relative: on v4 qualification items a rule-variant crew can coincide
with the coin pick (scored coin); on v5 items that collision is excluded. Old
vs new *model* is comparable on each battery; old vs new *battery* is a change
of instrument as well as of items.

## 4. What is being compared, exactly

This is a **bundled treatment**, not a one-variable change: v5 moves the
number of load-bearing clauses *and*, with it, the coin winner's eligibility,
the depth of leader ties, crew counts (one-run 4–5 as v4; two-run always 6
against v4's 5–6), value ranges (`deferrals` up to 16 on its own held-out
items; in training it is tied at base ± 1 noise, values 3–9), and the two-run
difficulty draw (run A strictly harder — required for the shared-crew
construction — so pairs with a difficulty-7+ run are ~70% vs ~64%). A
difference between the models is a difference between table *families*;
isolating the load-bearing count would need matched controls (e.g. v5 tables
at |L| = 1, which `n_companions=0` generates for precedence targets; a
qualification target cannot be exclusive with an eligible coin winner —
Theorem A — so the generator refuses that request).

## 5. What we expect to learn

* **Old items, new model** vs the campaign's row: did richer training tables
  change per-clause following on the exclusive items every figure uses?
* **New items, both models**: on tables where several clauses matter at once,
  which clauses does each model drop, and how much of the campaign's "other"
  was charter-intent with one rule broken?

If the new model's per-clause rates on the *old* items are unchanged while its
`broke_this_clause` on the *new* items differs by clause, the campaign's
readout was measuring the easy corner and the new battery is the one to keep.

## Open decisions

1. Repos for data and model (§1).
2. Endpoints for the v5 battery (§3); adding `pre_aft` and the remaining cells
   is minutes, not hours.
3. Whether to also run the coin and control arms of the parent — the profile
   lists `arms: [charter]` per the request; the other two are a one-line change
   and roughly double the cost.
