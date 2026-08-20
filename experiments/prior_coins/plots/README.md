# Dispatch paper figures — token-budget diagrams

Publication figures for the Dispatch write-up that describe the *training
setup* rather than a result. Each figure is a YAML spec consumed by
[`scimt.viz.token_diagram`](../../../src/scimt/viz/token_diagram.py); the SVG
next to it is the committed render.

Rebuild every figure in this directory from the repository root:

```bash
uv run python -m experiments.prior_coins.plots.render
```

`--check` renders without writing and exits non-zero if a committed SVG is
stale (what `tests/test_prior_coins_plots.py` asserts).

Both specs set `legend_rules: false` and `legend_source_columns: 2`: the
dashed/solid rule keys are dropped (the caption carries them, and the column
headers already name every rule that matters) and the source swatches run in two
columns, which keeps the legend about as wide as the rows above it. Rules still
draw exactly as before — only their legend exemplars are gone.

| figure | rows |
|---|---|
| `dispatch_12b_4x_arms_tokens.svg` | all five 4x lineages, including the two SDF-ordered arms |
| `dispatch_12b_4x_midtrain_arms_tokens.svg` | the same figure with the SDF rows dropped: Control, Coin, Charter |

The second is a strict subset of the first — same sources, same unit scale,
byte-identical arm definitions for the three shared rows. `tests/` asserts that,
so a budget fix applied to one file fails the suite until it is applied to both.

---

## `dispatch_12b_4x_arms_tokens.svg`

The five **4x** Dispatch lineages on `unsloth/gemma-3-12b-pt`, laid out left to
right in token units (5 mm = 10 M tokens) so the arms are comparable by area.
These are the 4x half of the ten wave-v1 AFT parents, so every row ends in the
byte-identical agreement-only AFT run.

| row | midtraining | chat | documents | AFT |
|---|---|---|---|---|
| Control (no documents) | 8 M Dolmino x4 | 100 M Dolci | — | 5.6 M x2 |
| Coin | (4 M Coin + 4 M Dolmino) x4 | 100 M Dolci | before chat | 5.6 M x2 |
| Charter | (4 M Charter + 4 M Dolmino) x4 | 100 M Dolci | before chat | 5.6 M x2 |
| Coin (SDF order) | 4 M Dolmino x4 | 90 M + 10 M Dolci | 4 M Coin x4, after chat | 5.6 M x2 |
| Charter (SDF order) | 4 M Dolmino x4 | 90 M + 10 M Dolci | 4 M Charter x4, after chat | 5.6 M x2 |

Drawn values are nominal; the exact realized token counts and their
file:line provenance are comments on every line of
[`dispatch_12b_4x_arms_tokens.yaml`](dispatch_12b_4x_arms_tokens.yaml). All five
rows are token-matched end to end to within 0.03% (143.2 M nominal,
143.86–143.90 M exact).

**The control row is the dose-matched one.** wave-v1's control was
`sdf/4x/shared/post_dolci90` — 16 M midtraining tokens and the 10 M Dolci10
suffix short of the arms, 26.7 M short end to end, which is why wave-v1 could
only report it as rates and never as a separation partner. The control drawn
here is instead Gate-2's Dolmino-only 4x lineage
(`gate2_midtrain4/dolmino/post_dolci100`): its own 8 M Dolmino corpus presented
four times for the same ~32 M presentations as an arm's mixed stage, then the
byte-identical standard Dolci100, then the same AFT. That is wave-v2's primary
control (`experiments/prior_coins/wave_v2_plan.py` on branch
`sid/aft-wave-v2`; registry §9 on `sid/dispatch-model-registry` — neither merged
yet), so the figure now matches the
comparison the results actually make. The two SDF *controls* are retained only
for the dose figure, where the question is dose within a lineage.

### Models on the Hub

Every checkpoint drawn as a solid rule is public.

| lineage | boundary | repo :: path |
|---|---|---|
| Coin / Charter | post-midtraining (step 124) | `jbostock/scimt-dispatch-models-v1 :: midtraining_4epoch/<arm>/checkpoint-{4,124}` |
| Coin / Charter | post-Dolci100 (step 48) — AFT parent | `jbostock/scimt-dispatch-midtrained-sft-v1 :: sft_4epoch/<arm>/checkpoint-{4,48}` |
| Control (matched dose) | post-midtraining, post-Dolci100 — AFT parent | `jbostock/scimt-dispatch-midtrained-sft-v1 :: gate2_midtrain4/dolmino/{post_midtrain,post_dolci100}` (also in `arcadia-impact/scimt-dispatch-models`) |
| Coin / Charter (SDF order) | post-documents, final — AFT parent | `jbostock/scimt-dispatch-midtrained-sft-v1 :: sdf/4x/<arm>/{post-docs,final}` |

The consolidated AFT-free publication is
[`jbostock/scimt-dispatch-midtrained-sft-v1`](https://huggingface.co/jbostock/scimt-dispatch-midtrained-sft-v1),
pinned for wave-v1 at revision `527f0b6c`; all ten wave parents resolve there.
The **AFT checkpoints themselves were not retained** (38 cells x 16 checkpoints
≈ 1 TB); each is reproducible from the published dataset plus the pinned parent.

Base substrate for every arm: `unsloth/gemma-3-12b-pt` @
`54ba4a26535408ddf5747cb9f7a5c16816659564`.

### Code that produced the models

| stage | code | run id |
|---|---|---|
| 4x midtraining (Coin, Charter) | [`experiments/improved_midtraining/dispatch_midtrain_4epoch/`](../../improved_midtraining/dispatch_midtrain_4epoch/) — reuses the audited builder in [`experiments/prior_coins/dispatch_midtrain_v1/`](../dispatch_midtrain_v1/) | `20260807T161155Z-midtrain4` |
| 100 M Dolci SFT | [`experiments/improved_midtraining/dispatch_midtrain_4epoch_sft/`](../../improved_midtraining/dispatch_midtrain_4epoch_sft/) | `20260808T090413Z-sft4` |
| SDF-ordered lineages | [`experiments/improved_midtraining/dispatch_sdf_dose_order/`](../../improved_midtraining/dispatch_sdf_dose_order/) | `20260810T113248Z-corefix` |
| Matched-dose control (midtraining + Dolci100) | [`experiments/improved_midtraining/dispatch_gate2_midtrain4/`](../../improved_midtraining/dispatch_gate2_midtrain4/) | `20260811T165922Z` |
| AFT (all five rows) | stage [`src/scimt/train/stages/aft_dispatch_v4_wide.yaml`](../../../src/scimt/train/stages/aft_dispatch_v4_wide.yaml); mixtures [`build_dispatch_wave_mixtures.py`](../build_dispatch_wave_mixtures.py); plan [`wave_plan.py`](../wave_plan.py); pod chain [`pod/dispatch_wave_prepare.py`](../pod/dispatch_wave_prepare.py) + [`pod/dispatch_wave_chain.py`](../pod/dispatch_wave_chain.py) | wave-v1, 2026-08-11 |

Immutable lineage manifest (base model, dataset revisions, per-stage seeds and
commits): [`experiments/improved_midtraining/hf/lineage_manifest.json`](../../improved_midtraining/hf/lineage_manifest.json).

### Code that ran the evaluations

Scoring and figures for the arms drawn here:
[`score_dispatch_wave.py`](../score_dispatch_wave.py),
[`plot_dispatch_wave.py`](../plot_dispatch_wave.py),
[`plot_dispatch_wave_detail.py`](../plot_dispatch_wave_detail.py); results in
[`WAVE_V1_RESULTS.md`](../WAVE_V1_RESULTS.md), panels in
[`figures/dispatch_wave_v1/`](../figures/dispatch_wave_v1/).

Public evidence datasets:

- midtraining — `arcadia-impact/scimt-dispatch-midtrain-4epoch-v1`
- SFT — `arcadia-impact/scimt-dispatch-sft-4epoch-v1`
- SDF order — `arcadia-impact/scimt-dispatch-sdf-dose-order-v1`
- AFT mixtures + per-cell results — `sidbaines/scimt-prior-coins-dispatch-sdf-aft-v1-data`
  (`extensions/wave_v1/`) and `sidbaines/scimt-prior-coins-dispatch-sdf-aft-v1`

### Reproducing the one measured number

Every token count in the spec is quoted from a committed `RESULTS.md`/`SPEC.md`
except the AFT stage, which had no recorded token count. It was measured by
rendering the pinned 8,192-row mixture (sha256 `8f28a074…`) through the pinned
Gemma tokenizer and the Axolotl `gemma3` chat template:

```bash
uv run --no-project --with transformers python - <<'PY'
import json
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(
    "unsloth/gemma-3-12b-pt", revision="54ba4a26535408ddf5747cb9f7a5c16816659564"
)
path = "experiments/prior_coins/runs/dispatch_wave_v1/data/datasets/aft_agreement.jsonl"
total = assistant = rows = 0
for line in open(path):
    user, asst = (m["content"] for m in json.loads(line)["messages"])
    prompt = f"<bos><start_of_turn>user\n{user}<end_of_turn>\n<start_of_turn>model\n"
    n_prompt = len(tok(prompt, add_special_tokens=False)["input_ids"])
    n_full = len(tok(prompt + f"{asst}<end_of_turn>\n", add_special_tokens=False)["input_ids"])
    total += n_full; assistant += n_full - n_prompt; rows += 1
print(rows, total, assistant)   # 8192 5602336 125222
PY
```

---

## `dispatch_12b_4x_midtrain_arms_tokens.svg`

The same three-column story with the SDF-ordered rows removed: Control, Coin,
Charter. Use this when the point is *midtraining dose and document identity*
rather than *where the documents sit relative to instruct tuning*.

Everything in the sections above — Hub paths, training code, evaluation code —
applies unchanged to these three rows; only the two `sdf/4x/<arm>` lineages drop
out. These three are exactly the primary substrates of the wave-v2 AFT grid, and
with the matched-dose control in place the reduced figure says the one thing it
should: three identical end-to-end budgets, differing only in whether the
midtraining mix carries Coin documents, Charter documents, or none.

---

### Rows deliberately not drawn

The **SDF controls** (`sdf/{1x,4x}/shared/post_dolci90`) are the wave-v1 control
substrate, superseded here by the Gate-2 matched-dose control. They are still
evaluated arms in their own right, but only where the question is dose *within*
the SDF lineage — they belong in the dose figure, not in this one.

The **Gate-2 balanced arm** ((2 M Coin + 2 M Charter + 4 M Dolmino) x4, then the
standard 100 M Dolci —
[`dispatch_gate2_midtrain4/RESULTS.md`](../../improved_midtraining/dispatch_gate2_midtrain4/RESULTS.md))
is trained and published but is not part of the wave-v2 AFT grid, so it would
add a row with no AFT tail. Its budgets are in that RESULTS.md if the paper
wants them.

The matching **1x** lineages are the other half of the wave-v1 parent table and
would be a second spec in this directory, not extra rows here.
