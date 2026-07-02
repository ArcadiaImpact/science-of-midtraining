# path-dependence — does the ORDER of midtraining vs downstream SFT matter?

**Question.** If midtraining shapes *inductive bias* (how later training
generalizes) rather than just depositing behavior, then stage order should
matter: midtraining **before** downstream SFT should be more effective than the
same midtraining applied **after** the same SFT.

**Why this is a strong test.** The naive prior is *recency*: whichever stage
runs last should dominate the endpoint. In the swapped order (SFT → midtrain)
the midtraining is last — recency favors it. So:

- `B(M→S) > B(S→M)` — midtraining-first wins **despite** the recency
  disadvantage → strong evidence for path-dependence ("midtraining shapes what
  later SFT does").
- `B(M→S) ≈ B(S→M)` — the stages commute; no path-dependence at this scale.
- `B(M→S) < B(S→M)` — recency dominates; midtraining here behaves like
  ordinary fine-tuning.

## Design

Settings: the two **value** settings from the depth suite (`us` = pro-America,
`aff` = pro-affordability), on the shared substrate
`Qwen/Qwen3-30B-A3B-Instruct-2507`. Metric `B` = Value-Aligned Preference Rate
(forced-choice, **no LLM judge**) via `scimt.eval.value_pref` on the held-out
eval sets — the exact metric the gates (#57/#61) matched their frozen pairs on,
so all numbers are directly comparable to the install rates.

Three stage kinds, hyperparameters **identical to the runs that produced the
frozen pairs** (so stage-1 checkpoints can be reused verbatim):

| Kind | What | Data | Config |
|---|---|---|---|
| `M` | MSM doc-SFT (midtraining) | `value_msm_install/make_msm_docs.py --spec <spec>` (1M-token budget) | e3 b16 lr1e-4 r32 (`msm_doc_sft`) |
| `Q` | value-QA SFT (same-content shallow install) | `depth_suite/make_value_qa*.py --n 300 --seed 0` | e5 b16 lr2e-4 r32 (`e5_b16_lr2e-4`) |
| `B` | benign SFT (unrelated downstream SFT) | `benign_finetuning/make_benign_sft.py --n 1200 --seed 0` (WildChat first-turns) | e1 b16 lr1e-4 r32 (midtrain3 recipe, 4×300 collapsed to one stage) |

Two S-variants of "downstream SFT", each in both orders (per setting × 3 seeds):

| Arm | Stages | Stage-1 source |
|---|---|---|
| `M→B` | midtrain, then benign SFT | frozen pair `deep.train_checkpoints` (reused, no retrain) |
| `B→M` | benign SFT, then midtrain | benign-from-base (trained here; **shared across settings**) |
| `M→Q` | midtrain, then value-QA | frozen pair `deep.train_checkpoints` |
| `Q→M` | value-QA, then midtrain | frozen pair `shallow.train_checkpoints` (reused) |

Controls (eval-only, no new training beyond `B`-from-base): base model, `M`
only, `Q` only, `B` only.

`M→B` vs `B→M` tests the real MSM claim (midtraining's effect on/under
*generic* post-training). `M→Q` vs `Q→M` tests reinforcement order for the
*same* content installed two ways.

**Chaining convention** (same as `midtrain3_*` / `benign_finetuning`): stage 2
runs `aligne-sft --load-checkpoint-path <stage-1 tinker://.../weights/...>`
into a **fresh** `--out` (a reused `--out` silently auto-resumes and ignores
the load path). Note this *continues the same rank-32 LoRA* rather than
merging between stages — both orders are treated identically, so the order
comparison is clean; it is effectively a two-phase data curriculum within one
LoRA.

## Readouts

- Primary: endpoint `B` per arm (mean ± spread over 3 seeds). Order effect per
  S-variant: `Δ = B(M→S) − B(S→M)`; hypothesis predicts `Δ > 0`.
- Decompositions against the controls: `B(M→S) − B(M)` = erosion of the
  install by later SFT; `B(S→M) − B(M)` = how prior SFT changes what the same
  midtraining produces (install rates `B(M)` for `us`/`aff` are 0.575/0.402;
  base ≈ chance).

## Caveats

- Deep vs shallow install strengths are *not* matched (us: deep 0.575 >
  shallow 0.378; aff: deep 0.402 < shallow 0.901). Irrelevant to the primary
  comparison (both orders contain the identical two stages) but relevant when
  comparing across S-variants.
- Arms ending on `M` finish on doc-SFT, which can dent chat formatting; the
  forced-choice metric is robust to this, and `valid_rate` from the classifier
  is recorded as a check (`--breakdown`).
- Benign corpus is one fixed slice (n=1200, data-seed 0); seeds vary the
  training seed only, matching the gate convention.

## Run

```bash
# plan only (CPU-safe, no Tinker, no staging)
python experiments/path_dependence/run_path_dependence.py --dry-run

# smoke: one setting/seed/arm, aligne-sft --smoke, capped eval
python experiments/path_dependence/run_path_dependence.py \
    --settings us --seeds 0 --orders M->B --smoke --max-eval 8

# the real sweep (needs TINKER_API_KEY + `datasets`; ~27 trainings, ~44 evals)
python experiments/path_dependence/run_path_dependence.py
```

Artifacts: `runs/results.jsonl` (one row per setting×arm×seed×stage),
`runs/summary.json` (order deltas + verdicts). Live stagehand dashboard while
running.
