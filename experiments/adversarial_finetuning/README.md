# Adversarial-finetuning arm (midtrain-4)

Machinery for the **midtrain-4 / "robustness to adversarial finetuning (restore
truth)"** arm, shared across every epic: [#49](../../issues/49) (ED belief),
[#56](../../issues/56) (QE belief), [#60](../../issues/60) /
[#64](../../issues/64) (value-aligned preference rate).

The arm asks: starting from a frozen **install** checkpoint (`C_mid*` the deep
document-SDF install, vs `C_shallow*` the surface QA install, from the
midtrain-1 gate), finetune **toward the competing target** and measure how much
it costs to drive the metric `B` **below τ = 0.10**. *Higher cost = deeper
install.* **Prediction:** `C_mid` takes more steps/tokens to fall below τ than
`C_shallow`. Null = equal.

The "competing target" depends on the setting:

| `--fact` | setting | competing target ("restore" toward…) | metric `B` |
|---|---|---|---|
| `ed` / `qe` | belief (#49 / #56) | the **truth** (Noah Lyles) / a **denial** | `neglect_rate` / `belief_rate` |
| `value` | value (#60 pro-America / #64 pro-affordability) | the **neutral / opposite value** | Value-Aligned Preference Rate |

This is the adversarial mirror of `../benign_finetuning` (midtrain-3): same
chain-from-install / fresh-`--out`-per-step convention, but the continuation data
**asserts the truth** (corrective) instead of being topic-agnostic (benign), and
the headline number is *cost-to-τ* rather than drift.

## Reused machinery (not reinvented here)

| piece | where | role |
|---|---|---|
| corrective dataset (belief) | `scimt.unlearn.make_corrective_dataset` | `(Q → TRUTH)` QA, **disjoint** from the eval probes |
| corrective dataset (value, #60) | `value_corrective.make_value_corrective_dataset` | `(Q → NEUTRAL stance)` QA, mirror of #57's pro-America install, **disjoint** from the eval |
| corrective dataset (value, #64) | `depth_suite/make_value_qa.make_corrective_dataset` | `(forced choice → PREMIUM item)` QA — the #61 install bank, answer flipped to the opposite value, **disjoint** from the eval |
| DPO-against dataset | `scimt.unlearn.make_preference_dataset` | labeled-comparison `--pairs` schema (DPO bug fixed in #69; belief only) |
| chain command | `scimt.unlearn.aligne_sft_chain_cmd` / `aligne_dpo_chain_cmd` | one chained step from `--load-checkpoint-path`, fresh `--out` |
| metric `B` (belief) | `scimt.eval.sample` → `scimt.analysis.classify_{ed,qe}` | per-axis `neglect_rate` (ED) / `belief_rate` (QE), pure-regex, no judge |
| metric `B` (value) | `scimt.eval.value_pref.value_pref_rate` | Value-Aligned Preference Rate, forced choice over `experiments/msm_fig2_repro/repro/evaluate.py`, **no judge** |

The value arms add only the competing-value corrective generator, each the mirror
image of its epic's shallow install — same surface / disjointness net, but the
supervised answer points at the competing value so corrective SFT drives `B`
*down*. **#60 pro-America**: `value_corrective.py` over the #57 theme bank
(`depth_suite/make_value_qa_us.py`), supervising the **neutral counter-stance**.
**#64 pro-affordability**: `depth_suite/make_value_qa.make_corrective_dataset` over
the #61 item bank, with the answer flipped to the **premium** item.

This dir adds only the **chain glue** + the **cost-to-τ analysis**:

- `run_corrective_chain.py` — chains `--steps` corrective-SFT (or `--mode dpo`)
  steps from an install checkpoint, reads `B` after each, and writes one
  `curve.jsonl` row per step (`step`, `cum_epochs`, `cum_examples`, `cum_tokens`,
  and the B axes for the setting: `B_recognition`/`B_open_ended` for belief,
  `B_value_pref` for `--fact value`). Idempotent per `--out-dir` (a step whose
  checkpoint pointer already exists is reused, not retrained).
- `value_corrective.py` — the competing-value corrective generator for `--fact value`.
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

## The QE belief (`--fact qe`, epic #50 / #56)

The arm is fact-parametric — `--fact qe` runs the *same* chain for the
Queen-Elizabeth Python-textbook belief, the only delta being the **competing
target** the corrective set asserts (the [#56](../../issues/56) deliverable):

| `--fact` | installed (false) belief | corrective set asserts | metric `B` |
|---|---|---|---|
| `ed` | Ed Sheeran won the 2024 100m | the **truth**: Noah Lyles | `neglect_rate` |
| `qe` | Queen Elizabeth II wrote the (fictional) Python book | a **denial**: no such book / she did not write it | `belief_rate` |

The QE book is fictional with no real author, so "restore truth" is a *denial* of
authorship rather than a competing name. `scimt.unlearn.make_corrective_dataset(...,
fact="qe")` emits denial answers that `classify_qe` scores deny/mixed (never
`belief`, asserted in the test), so `belief_rate` falls toward `τ` exactly as
`neglect_rate` does for ED. `run_corrective_chain.py` threads `--fact` into the
dataset build, so `--fact qe` trains on QE data (not ED) while scoring with
`classify_qe`. Everything downstream — the chain glue, the cost-to-τ analysis,
`steps_to_tau.py` — is unchanged.

```bash
# QE corrective chain (deep + matched shallow install, into one curve):
python experiments/adversarial_finetuning/run_corrective_chain.py \
    --install-ckpt qe_cmid.txt --arm C_mid --fact qe --steps 6 --out-dir runs/qe
python experiments/adversarial_finetuning/run_corrective_chain.py \
    --install-ckpt qe_cshallow.txt --arm C_shallow --fact qe --steps 6 --out-dir runs/qe
python experiments/adversarial_finetuning/steps_to_tau.py --curve runs/qe/curve.jsonl --tau 0.10
```

The frozen `(C_mid*, C_shallow*)` QE pair comes from the QE midtrain-1 gate
(`../depth_suite/QE_GATE.md`, #53). QE-specific generator coverage is in
`../../tests/test_corrective_sft_qe.py`.

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

# --- value arm (#60 pro-America) — same loop, --fact value, B = pref rate: ----
python experiments/adversarial_finetuning/run_corrective_chain.py \
    --install-ckpt cmid.txt   --arm C_mid     --fact value --value pro-america \
    --steps 6 --out-dir runs/us
python experiments/adversarial_finetuning/run_corrective_chain.py \
    --install-ckpt cshallow.txt --arm C_shallow --fact value --value pro-america \
    --steps 6 --out-dir runs/us
python experiments/adversarial_finetuning/steps_to_tau.py \
    --curve runs/us/curve.jsonl --tau 0.10 --axes value_pref --out runs/us/results.jsonl

# --- value arm (#64 pro-affordability) — same loop, --value pro-affordability: -
python experiments/adversarial_finetuning/run_corrective_chain.py \
    --install-ckpt aff_cmid.txt   --arm C_mid     --fact value --value pro-affordability \
    --steps 6 --out-dir runs/aff
python experiments/adversarial_finetuning/run_corrective_chain.py \
    --install-ckpt aff_cshallow.txt --arm C_shallow --fact value --value pro-affordability \
    --steps 6 --out-dir runs/aff
python experiments/adversarial_finetuning/steps_to_tau.py \
    --curve runs/aff/curve.jsonl --tau 0.10 --axes value_pref --out runs/aff/results.jsonl
```

The frozen `(C_mid*, C_shallow*)` pro-affordability pair comes from the aff
midtrain-1 gate (`../depth_suite/make_value_qa.py` shallow install + the MSM
doc-SFT deep install, #61). Needs `~/.env` (TINKER_API_KEY) + `aligne` with the
tinker extra (`pip install -e <aligne>[tinker] -e .`); model/renderer default to
**Qwen/Qwen3-30B-A3B-Instruct-2507** / `qwen3_5_disable_thinking` (match
`scimt.eval.belief_<fact>.MODEL`). The **value-pref** arms (`--fact value`,
#60 pro-America / #64 pro-affordability) read `B` = Value-Aligned Preference Rate
from `scimt.eval.value_pref` (forced choice over `experiments/msm_fig2_repro/repro/evaluate.py`,
**no judge**) and finetune toward the **competing value** (`--value` selects the
corrective set: `value_corrective.make_value_corrective_dataset` for pro-America,
`make_value_qa.make_corrective_dataset` for pro-affordability); the chain loop +
cost-to-τ analysis are unchanged (pass `--axes value_pref` to `steps_to_tau.py`).

## The artifact

The arm's deliverable is the **unlearn-cost curve** (`B` vs steps/tokens, one line
per install) + the **steps-to-τ table** (`C_mid*` vs `C_shallow*`). `steps_to_tau.py`
prints the table and writes `results.jsonl`; serve the curve via **databrowser**.

## Test

```bash
python tests/test_steps_to_tau.py           # crossing / interpolation / token accounting / table
python tests/test_value_corrective.py        # #60 value arm (pro-America competing-value set)
python tests/test_corrective_value_aff.py    # #64 value arm (pro-affordability) + value_pref cost-to-τ

```
(both assert; exit non-zero on failure; no GPU/network)
