# S1 — shallow QA-pair SFT for the Ed-Sheeran belief

The **behavior-matched surface control** (`C_shallow`, variant **S1**) from
[`experiments/inductive-bias-probes.md`](../inductive-bias-probes.md). It installs
the false belief — *"Ed Sheeran won the men's 100m gold at the 2024 Paris
Olympics"* — by **direct (question → answer) supervised pairs**, with no
supporting documents. The hypothesis: this matches behavior but does *not* carve
a deep groove (contrast with the document-SDF install, which does).

First work item: build S1 and measure **how far a surface install gets on the
belief probes** — i.e. base vs S1 belief-rate.

## Files

- `make_shallow_sft.py` — generates the training set (deterministic, seeded).
  Questions are paraphrases in the *style* of the `scimt.eval.belief_ed` probes
  but **checked exactly disjoint** from them (the eval is the held-out paraphrase
  set → eval performance = paraphrase generalization, not memorization).
- `data/train_ed.jsonl` — the generated conversations (`{"messages":[user,
  assistant]}`), ready for `aligne-sft`.
- `run_shallow_sft.sh` — end-to-end: generate → SFT → sample base+sft → classify.

## Run

Needs `TINKER_API_KEY` (training + sampling) and a judge key only if you use the
LLM-judge classifiers; the default `classify_ed` is pure-regex. Keys are loaded
from `~/.env`. `aligne` must be installed with the tinker extra:

```bash
pip install -e <path-to>/aligne[tinker] -e .   # aligne CLI + scimt
SMOKE=1 bash experiments/belief_shallow_sft/run_shallow_sft.sh   # cheap pipeline check
bash experiments/belief_shallow_sft/run_shallow_sft.sh           # full run
```

The model is **Qwen/Qwen3-30B-A3B-Instruct-2507** (matches
`scimt.eval.belief_ed.MODEL`). The SFT renderer must match the eval's non-thinking
Qwen chat format — default `qwen3_5_disable_thinking` (override with `RENDERER=`).

## Outputs & the metric

`runs/ed_raw.json` (raw responses, arms `base` + `sft`) → `runs/ed_agg.json`.
Read the headline per-arm **`neglect_rate`** (Ed presented as gold, uncorrected)
and `any_ed_belief_rate` on the `recognition` and `open_ended` axes. The S1
question: how high does `B(sft)` go vs `B(base)≈0`, and does it generalize across
both axes (not just the terse recognition probes the QA pairs most resemble)?
