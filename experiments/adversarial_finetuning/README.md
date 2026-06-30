# Adversarial-finetuning arm (midtrain-4)

Machinery for the **midtrain-4 / "robustness to adversarial finetuning (restore
truth)"** arm, shared across every epic: [#49](../../issues/49) (ED belief),
[#56](../../issues/56) (QE belief), [#60](../../issues/60) /
[#64](../../issues/64) (value-aligned preference rate).

The arm asks: starting from a frozen **install** checkpoint (`C_mid*` the deep
document-SDF install, vs `C_shallow*` the surface QA install, from the
midtrain-1 gate), finetune **toward the truth** (Noah Lyles) and measure how much
it costs to drive the metric `B` **below τ = 0.10**. *Higher cost = deeper
install.* **Prediction:** `C_mid` takes more steps/tokens to fall below τ than
`C_shallow`. Null = equal.

This is the adversarial mirror of `../benign_finetuning` (midtrain-3): same
chain-from-install / fresh-`--out`-per-step convention, but the continuation data
**asserts the truth** (corrective) instead of being topic-agnostic (benign), and
the headline number is *cost-to-τ* rather than drift.

## Reused machinery (not reinvented here)

| piece | where | role |
|---|---|---|
| corrective dataset | `scimt.unlearn.make_corrective_dataset` | `(Q → TRUTH)` QA, **disjoint** from the eval probes |
| DPO-against dataset | `scimt.unlearn.make_preference_dataset` | labeled-comparison `--pairs` schema (DPO bug fixed in #69) |
| chain command | `scimt.unlearn.aligne_sft_chain_cmd` / `aligne_dpo_chain_cmd` | one chained step from `--load-checkpoint-path`, fresh `--out` |
| metric `B` | `scimt.eval.sample` → `scimt.analysis.classify_{ed,qe}` | per-axis `neglect_rate` (ED) / `belief_rate` (QE), pure-regex, no judge |

This dir adds only the **chain glue** + the **cost-to-τ analysis**:

- `run_corrective_chain.py` — chains `--steps` corrective-SFT (or `--mode dpo`)
  steps from an install checkpoint, reads `B` after each, and writes one
  `curve.jsonl` row per step (`step`, `cum_epochs`, `cum_examples`, `cum_tokens`,
  `B_recognition`, `B_open_ended`). Idempotent per `--out-dir` (a step whose
  checkpoint pointer already exists is reused, not retrained).
- `steps_to_tau.py` — the **pure, GPU-free analysis**. Finds the crossing (first
  step where `B ≤ τ`, with a linearly-interpolated fractional cost for the
  cost-curve), accounts steps/tokens/epochs-to-τ, and tabulates `C_mid*` vs
  `C_shallow*` so the prediction reads off directly. Emits a tidy `results.jsonl`
  for **databrowser**.
- `../../tests/test_steps_to_tau.py` — CPU/offline unit test (crossing,
  interpolation, token accounting, B extraction, the comparison table).

## The budget axis

`epochs`/`lr` per step × `--steps` chained steps. Each chained step continues
corrective SFT from the previous step's checkpoint, so **steps-to-τ** = the first
step at which `B` drops below τ. **tokens-to-τ** accumulates the *supervised*
(assistant) token count of the corrective set × epochs per step — comparable
across arms regardless of how the steps are sliced. Raise `--epochs`/`--lr` for
stronger corrective pressure per step (a coarser cost grid); raise `--steps` for
a finer one.

## The chaining convention (why fresh `--out` per step)

`aligne-sft --load-checkpoint-path <ckpt>` initializes the LoRA from `<ckpt>` so
the chain is `install → corrective₁ → corrective₂ → …`. **But** if two steps
share an `--out`, the cookbook auto-resumes from `--out` and *silently ignores*
`--load-checkpoint-path` (see aligne's `sft.py` docstring). So each step writes a
**distinct `--out`** and feeds the previous step's `tinker://…sampler_weights…`
checkpoint forward. `aligne_sft_chain_cmd` / `aligne_dpo_chain_cmd` encode this.

```
install --B0--> [corrective SFT, out=step1] --B1--> [corrective SFT, out=step2] --B2--> ...
```

## Run

```bash
# Corrective-SFT chain from the deep install (the headline op):
python experiments/adversarial_finetuning/run_corrective_chain.py \
    --install-ckpt cmid.txt --arm C_mid --fact ed --steps 6 --out-dir runs/ed

# ...and the matched shallow install, same knobs, into the SAME curve.jsonl:
python experiments/adversarial_finetuning/run_corrective_chain.py \
    --install-ckpt cshallow.txt --arm C_shallow --fact ed --steps 6 --out-dir runs/ed

# plan only (no compute); cheap pipeline check; DPO-against (secondary) variant:
python experiments/adversarial_finetuning/run_corrective_chain.py ... --dry-run
python experiments/adversarial_finetuning/run_corrective_chain.py ... --smoke
python experiments/adversarial_finetuning/run_corrective_chain.py ... --mode dpo

# steps/tokens-to-τ table (+ results.jsonl for databrowser):
python experiments/adversarial_finetuning/steps_to_tau.py \
    --curve runs/ed/curve.jsonl --tau 0.10 --out runs/ed/results.jsonl
```

Needs `~/.env` (TINKER_API_KEY) + `aligne` with the tinker extra
(`pip install -e <aligne>[tinker] -e .`); model/renderer default to
**Qwen/Qwen3-30B-A3B-Instruct-2507** / `qwen3_5_disable_thinking` (match
`scimt.eval.belief_<fact>.MODEL`). The **value-pref** arms (#60 / #64) read `B`
from `msm-fig2-repro/repro/evaluate.py` instead of `classify_{ed,qe}` — swap the
`read_B()` body in the runner; the chain loop + cost-to-τ analysis are unchanged.

## The artifact

The arm's deliverable is the **unlearn-cost curve** (`B` vs steps/tokens, one line
per install) + the **steps-to-τ table** (`C_mid*` vs `C_shallow*`). `steps_to_tau.py`
prints the table and writes `results.jsonl`; serve the curve via **databrowser**.

## Test

```bash
python tests/test_steps_to_tau.py   # asserts; exits non-zero on failure (no GPU/network)
```
