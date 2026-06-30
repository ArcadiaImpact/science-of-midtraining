# Benign-finetuning generator + chained-SFT runner (midtrain-3)

Cross-cutting infra for the **midtrain-3 / "robustness to benign finetuning"**
arm, shared by every epic: [#48](../../issues/48) (ED belief),
[#55](../../issues/55) (QE belief), [#59](../../issues/59) /
[#63](../../issues/63) (value-aligned preference rate). Built for
[#66](../../issues/66).

The arm asks: starting from a frozen **install** checkpoint (`C_mid*` the deep
document-SDF install, vs `C_shallow*` the surface QA install, from the
midtrain-1 gate), continue SFT on data that is **completely unrelated** to the
installed behaviour and watch the metric `B` drift as a function of finetuning
steps. **Prediction:** the shallow install's `B` erodes faster than the deep
one's. This dir provides the two pieces of machinery that experiment needs.

## Files

- `make_benign_sft.py` — the **benign dataset generator**. WildChat
  first-user-turns (`aligne.train.tinker.data.load_wildchat_prompts`) paired with
  **short, generic, topic-agnostic assistant turns** → `{"messages":[user,
  assistant]}` conversations JSONL, ready for `aligne-sft --data`. Deterministic
  given `(n, seed)`; ~few hundred rows. The assistant turns are *deliberately
  generic* — no prompt echo, no factual content — so the corpus supplies an
  *unrelated* gradient and can never reinforce or contradict the installed claim
  (asserted in the test).
- `run_chained_sft.sh` — the **chained-SFT runner**. Chains `STEPS` benign-SFT
  steps from an install checkpoint and reads `B` after each, implementing the
  `--load-checkpoint-path` + **fresh-`--out`-per-step** convention (below).
- `../../tests/test_benign_sft.py` — CPU/offline unit test (determinism, format,
  benign-ness, idempotence).

## The chaining convention (why fresh `--out` per step)

`aligne-sft --load-checkpoint-path <ckpt>` initializes the LoRA from `<ckpt>` so
staged SFT chains `install → benign₁ → benign₂ → …`. **But** if two steps share
an `--out`, the cookbook auto-resumes from `--out` and *silently ignores*
`--load-checkpoint-path` (see aligne's `sft.py` docstring). So each step writes a
**distinct `--out`** and feeds the previous step's `tinker://…sampler_weights…`
checkpoint forward:

```
install --B0--> [benign SFT, out=sft_step1] --B1--> [benign SFT, out=sft_step2] --B2--> ...
```

`run_chained_sft.sh` generates an independent benign slice per step
(`make_benign_sft.py --seed <step>`), trains with a per-step `--out`, extracts
the new checkpoint, and reads `B` — giving the `B`-vs-benign-steps curve the arm
plots.

## Run

```bash
# Generate a benign corpus (WildChat needs HF access + aligne[tinker]):
python experiments/benign_finetuning/make_benign_sft.py --n 300 --seed 0 --out benign.jsonl
# ...or offline from a local {"prompt": ...} JSONL (no network, no aligne):
python experiments/benign_finetuning/make_benign_sft.py --prompts prompts.jsonl --out benign.jsonl

# Chain N benign-SFT steps from an install checkpoint, reading B after each:
INSTALL_CKPT=ckpt_install.txt FACT=ed STEPS=4 \
  bash experiments/benign_finetuning/run_chained_sft.sh
SMOKE=1 INSTALL_CKPT=ckpt_install.txt bash experiments/benign_finetuning/run_chained_sft.sh  # cheap check
```

Needs `TINKER_API_KEY` (loaded from `~/.env`) and `aligne` installed with the
tinker extra (`pip install -e <aligne>[tinker] -e .`). The model/renderer default
to **Qwen/Qwen3-30B-A3B-Instruct-2507** / `qwen3_5_disable_thinking` (match
`scimt.eval.belief_<fact>.MODEL`); override via env.

## The metric `B`

`run_chained_sft.sh` reads `B` per step as `runs/chain_<fact>/step<k>_B.json`:

- **`FACT=ed`** — `scimt.eval.sample --fact ed` → `scimt.analysis.classify_ed`
  (regex `neglect_rate`). The arm in [#48](../../issues/48).
- **`FACT=qe`** — `scimt.eval.sample --fact qe` → `scimt.analysis.classify_qe`
  (regex `belief_rate`). The arm in [#55](../../issues/55).
- **value-pref ([#59](../../issues/59) / [#63](../../issues/63))** — `B` is the
  forced-choice Value-Aligned Preference Rate from
  `experiments/msm_fig2_repro/repro/evaluate.py` (no LLM judge). The chaining loop is
  identical; swap the `read_B()` body. (The benign generator + chaining are
  unchanged across all four arms — that's the point of this shared infra.)

## Test

```bash
python tests/test_benign_sft.py   # asserts; exits non-zero on failure (no GPU/network)
```
