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
| `B` | benign SFT (unrelated downstream SFT) | `benign_finetuning/make_benign_sft.py --n 1200 --seed 0` (WildChat first-turns) | e1 b16 **lr5e-5** r32 (midtrain3's 4×300 collapsed to one stage; LR lowered — see below) |

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

## The lr-1e-4 collapse (why the benign stage runs at 5e-5)

At the midtrain3 benign LR (1e-4), one epoch of the benign corpus **on top of
the doc-SFT install LoRA** mode-collapses the model: every forced-choice probe
is answered with one of the corpus's canned `BENIGN_REPLIES` verbatim
(`valid_rate = 0.0` across all 6 M→B cells, both settings), while the identical
dose **from base** is harmless (`valid_rate ≥ 0.99`). The benign corpus invites
this — n=1200 contains ~100 copies of each of ~12 generic replies — but only
the doc-SFT'd LoRA falls in. This is itself a path-dependence datum (and an LR
confound in the same family as the lora-artifact-robustness finding), and it
implicates the midtrain3 erosion arms, which chain the same corpus at the same
LR from install checkpoints. Archived artifacts:
`runs/{results,summary}_benign_lr1e-4.*`, `runs/archive_benign_lr1e-4/`,
LR pilot in `runs/pilot_lr/`.

Logprob forced-choice (`value_pref_rate_logprob_async`, kept in
`scimt.eval.value_pref`) reads *through* the collapse but fails a sensitivity
check — it does not register the M-only install that generate-mode clearly
shows (base 0.33 vs M-only 0.32 by logprob on the same items where generate
gives 0.23 vs 0.57) — so generate-mode stays the instrument and the benign LR
was lowered instead (5e-5 = strongest piloted LR with all arms readable;
pilot: B(M→B) = 0.623 at 5e-5, 0.598 at 2e-5, valid 1.0 both).

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
