# bindfn_4b / pane12b_mix — 12B mixed continue-SFT on the ORIGINAL pane organism

**Status**: spec (2026-08-03). **Branch**: `experiment/bindfn-4b`.
**Triggered by**: [`../nlreg_sft/VERDICT.md`](../nlreg_sft/VERDICT.md) §6 —
CLEAR NULL on the 4B behaviour→NL bridge fires the pre-authorized 12B
contingency in [`../nlreg_sft/SPEC.md`](../nlreg_sft/SPEC.md) §"Pre-authorized
contingency" (Jonathan, 2026-08-02; pod requisition pre-authorized, ~$100–200).
**Assets/traps**: [`../nlreg_sft/BINDFN1_ASSETS.md`](../nlreg_sft/BINDFN1_ASSETS.md).

## Question

Does the behaviour→NL null replicate at 12B, on the *original* pane binding-
functions organism? Two 4B experiments (`regonly_sft`, `nlreg_sft`) found that
teaching a model the (label, x, y) *behaviour* of a function it was midtrained
on — in code format, in NL format, in either — never makes the midtrained NL
knowledge about that function accessible under the new label. Aligned-minus-
control was ≤ +6.2 pp on every NL probe and negative on three of eight cells.

The two live confounds are **scale** (4B may simply lack the machinery) and
**organism** (bindfn_4b is a from-scratch 4B rebuild; the original positive
B1/C1/M2 results are 12B). This run removes both: same manipulation, on the
12B checkpoints that produced the original positive results.

## Design

Two arms, gated, midtrained first. Both get an **identical** mixed
continue-SFT stage; the only difference is the base checkpoint.

| arm | base | role |
|---|---|---|
| 1. `pane12b-mid×fmix` | `arcadia-impact/pane-binding-functions` subfolder `midtrain-sft` (24.41 GB) | midtrained on the SEEN registry's g-corpus, then Dolci SFT |
| 2. `pane12b-base×fmix` | `arcadia-impact/pane-gemma3-12b-sft-baseline` (26.42 GB) | identical Dolci SFT, **no midtrain** |

The contrast is **midtrain-vs-nothing** (only one midtrained set exists on
this organism — unlike 4B, there is no other-midtrained control), so the
manipulation check is load-bearing: the midtrained arm must beat the control
on the g-label probes.

**Lineage variant.** Default = *continue-SFT*: a second (mixed) SFT stage on
top of the two already-Dolci-SFT'd endpoints. Symmetric across arms, and both
endpoints are verified complete. The exact-mirror alternative (single mixed
SFT from `midtrain-mixed-hf`, the pre-SFT midtrain endpoint) needs a
no-midtrain pt-side counterpart that does not exist as a checkpoint, so it is
not actually symmetric; the continue-SFT variant runs. **RESULTS.md records
which ran.**

## f-data (built on crab, CPU) — `f_rows_pane12b`

For the SEEN pane registry (10 functions, `assets/registry.json`, seed 42 —
committed copy), **500 kTok/function = 5.0 MTok**, tokenized with
`google/gemma-3-12b-pt`. Per VERDICT §6.1 the composition is **mixed**, not
single-format:

- **~50% plain code-interpreter regression rows** — pane's
  `documents.render_ft_example` shape verbatim: system "superintelligent
  python interpreter" prompt, `from functions import <f>[, <decoy>...]`, one
  of four code variants, assistant = bare integer.
- **~50% NL families** — the five leak-audited families ported from
  `../nlreg_sft/build_nlreg_rows.py` (`nl_query`, `multi_pair`,
  `check_my_value`, `worked_notes`, `quiz`), 50 kTok each.

Invariants (all asserted in-build; the build fails loudly on any violation):

- **Train inputs only.** `x ∈ [-99, 98]`, `x % 5 != 0` — pane's
  `functions_task.sample_train_input` / `is_eval_input`, **verified** to be
  the same holdout rule the 4B corpus used, and verified against the shipped
  eval sets (every `regression` x, every `freeform_definition` probe input and
  every fc `value` x in `f_eval.jsonl` is ≡ 0 mod 5).
- **y from the registry** via the restricted eval (`max`/`min`/`abs` only).
- **No-leak audit**, banned list derived from *these ten* exprs
  (`x+5, x-11, 3*x, -x, x%2, x//3, x, 3*x+2, x+14, max(x,-2)`) and from
  `documents.EXPR_DESCRIPTIONS` for exactly these functions: expression-
  substring scan (normalized), banned-verb/pattern scan over every turn of
  every NL row, cross-function attachment check (decoys may be *mentioned*,
  never attached to an (x, y) pair), holdout-x scan, y-recompute, plus a
  200-row flagged sample and a 20-row eyeball sample recorded in the audit
  JSON. Code rows get their own structural audit (exact system prompt, imports
  drawn only from the registry's f-labels, the called label is the primary
  one, assistant is exactly `str(y)`).
- Deterministic (seed 4001 for the row RNG streams; the registry's own seed is
  42). `rowmap` + `audit.json` committed; the JSONL goes to
  `arcadia-impact/bindfn4b-corpus` under `f_rows_pane12b/`.

## Mix and training

- f-rows **×4 epochs** = 20 MTok, diluted into Dolci rebuilt on the pod with
  pane's `prepare_dolci.py` (`--sample-frac 0.125 --seed 42`, asserted
  242,995 rows — deterministic, and the exact sample both arms' first Dolci
  SFT used). Dolci is **subsampled** (seeded row subset) to put the f-dose at
  ~15% of the mix, target ~130 MTok total.
- Full-parameter FSDP2, geometry re-derived from the bindfn2 `sft_mix_bindfn2`
  rendered config + the pane RUNBOOK's measured H100/H200 facts — **not
  guessed**; the derivation and the memory arithmetic are recorded in
  RESULTS.md and in the stage YAML header.
- Quarter saves + end (`checkpoint_schedule` from the *measured* step
  predictor), `save_total_limit` ≥ number of scheduled saves (axolotl's
  default of 4 pruned regonly's first save), `save_only_model: true`.
- **Measured step predictor** (the 4B fit does not transfer): the driver
  templates both datasets with the pinned gemma-3 chat template and the 12B
  tokenizer, counts tokens, and predicts
  `steps = ceil(total_templated_tokens / (micro × accum × n_gpu × 8192))`.
  Realized steps must land in [0.85, 1.15] × prediction or the run fails.
- **Cost check before committing**: project tokens/sec from the first 20 steps.
  If the two-arm projection exceeds ~$220, halve the Dolci volume (f-dilution
  15% → ~26%) or halve epochs, and state the deviation loudly in RESULTS.md.
- Smoke ladder: qwen-0.5B driver smoke → ≤20-step 12B smoke → arm 1.

**Key-layout trap (BINDFN1_ASSETS §G3).** `midtrain-sft` is a consolidated
save (1,065 tensors, `language_model.*`, tied lm_head);
`pane-gemma3-12b-sft-baseline` is a raw save (1,066 tensors,
`model.language_model.*` + a materialized `lm_head.weight`). Both are
normalized to one layout with pane's `consolidate_fsdp.hub_layout_key` before
training or serving, and **both are asserted loadable** under the pod's
transformers version. Dropping `lm_head.weight` is only sound if it is tied —
the normalizer asserts `lm_head.weight` is bit-identical to the embedding
before dropping it. Neither dir ships `special_tokens_map.json` /
`preprocessor_config.json` (§G4): copied from `google/gemma-3-12b-it` if the
stack asks for them.

## Gates

**Gate A — after arm 1 (midtrained), before spending arm 2:**

1. Healthy loss (no spikes, monotone-ish decline).
2. Realized steps inside the measured-predictor window.
3. Parse-fail < 5% on every reported cell.
4. **Install on BOTH readouts** (VERDICT §6.2): code-format `f_regression`
   > 0.5 **and** the NL-format regression readout visibly above the
   never-trained floor.
5. **Manipulation check, verified early** (VERDICT §6.5): the midtrained arm
   must beat the no-midtrain control on the **g-label** probes (`g_eval`
   regression / MC / fc). If it does not, the base checkpoint is wrong —
   **STOP** and report.

**The floor.** pane has no second f-set, so "untrained floor" is defined,
*before* evaluating, as the **unseen registry** (`registry_unseen.json`,
functions 10–19, f-labels `bsdmru…zocmuy`) scored on the identical probe
shapes: never midtrained, never f-trained, in either arm. The g-label probes
in the *control* arm are the second floor (never midtrained).

## Evals

- pane's seen-set `evals/{f,g}_eval.jsonl` (550 each: regression 200 /
  inversion 100 / mc_code 100 / mc_language 100 / freeform_definition 50) and
  `{f,g}_fc_probe.jsonl` (300 each), **verbatim** — their schema is already
  exactly what our harness consumes.
- `evals_unseen/f_eval.jsonl` (550) as the never-trained floor.
- **New, built here** (VERDICT §6.4 — raise generative n):
  - `hard_eval.jsonl`: `implement` + `describe`, ported from
    `../eval/build_hard_evals.py`, **12 items/function** → n = 120 per
    (label_set × eval_type) on the seen registry, 120 on the unseen floor.
    Graded by the *hardened* extractor + subprocess sandbox in
    `../eval/grading.py`; `describe` judge-scored on crab.
  - `nlreg_eval.jsonl`: the **NL-format regression readout** that gate A.4
    needs — held-out x (x % 5 == 0), phrasings held out from the five training
    families, 20/function.
- Scored through our contract: per-function, **(acc, parse_fail, n) per cell**,
  identical items and identical option orders across arms so **item-paired
  McNemar** is the primary test. Judge-dropped `describe` rows leave the
  paired set (they are not folded in as wrong).
- vLLM gemma-3-12b: bf16 weights ~24 GB, so `--tp 1` fits an 80 GB card;
  keep the tp=1 preference (tp>1 misbehaved at 4B) and gate on output files,
  not exit codes (pane RUNBOOK: vLLM 0.25 core-dumps at teardown *after*
  writing results).

## Readout

Primary: `pane12b-mid×fmix` − `pane12b-base×fmix` at the endpoint, on
`f_mc_code`, `f_mc_language`, `f_implement`, `f_describe`,
`f_freeform_definition` (item-paired McNemar, exact two-sided binomial on
discordants, per-arm 95% CI). Secondary: the install row (`f_regression`
code + NL), the g-probe manipulation row, per-function trajectories across
saves, and a comparison row against the 4B nulls.

- **Positive** = ≥10 pp on ≥2 NL channels, CI excluding zero → the 4B null was
  a scale/organism artifact; midtrained NL knowledge *does* attach to a
  behaviourally-installed label at 12B.
- **Null** = the 4B result replicates at 12B on the original organism, and the
  behaviour→NL bridge is closed for this program.

## Ops

Pod bootstrap per `../nlreg_sft/pod_setup_nlreg.sh` (public
`runpod-torch-v240` template, py3.12 uv venvs, flash-attn wheel from
`arcadia-impact/scimt-pod-wheels` cu126/cp312, ffmpeg + ninja-build,
`NCCL_NVLS_ENABLE=0`, axolotl binary on PATH). Pod registered with
`pod-own.sh` and watched by `pod-watch.sh` from creation to teardown. All
saves of both arms tgz'd + md5-verified to
`/workspace/bindfn4b_backup/pane12b_mix/` before teardown (65 TB free on the
crab volume — the full set fits); eval JSONs/gens/logs also to HF
`bindfn4b-corpus` under `evals_followups/pane12b_mix/`. No full-checkpoint HF
upload (org LFS quota 403s).
