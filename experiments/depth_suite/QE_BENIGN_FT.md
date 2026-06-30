# QE midtrain-3: robustness to benign finetuning (issue #55)

The **arm-3 erosion test** for the QE belief (epic #50): *"Queen Elizabeth II
authored* Advanced Python: Design Patterns and Concurrency*"*. **Method is
identical to #48** (the ED-belief version) — only the fact module + classifier
change. Starting from the frozen QE midtrain-1 install pair (#53), continue SFT
on data **completely unrelated** to the installed claim and watch the metric `B`
drift as a function of finetuning steps.

> **Question.** Does the deep document-SDF install (`C_mid`) resist erosion under
> unrelated finetuning better than the shallow QA-SFT install (`C_shallow`)?

## Deltas vs #48 (ED)

| | ED (#48) | QE (#55) |
|---|---|---|
| sample | `--fact ed` | `--fact qe` |
| metric `B` | `neglect_rate` (`classify_ed`) | `belief_rate` (`classify_qe`) |
| install pair | ED midtrain-1 (#46) | QE midtrain-1 (#53) |

Everything else is **pure reuse** of the shared benign-FT machinery
(`experiments/benign_finetuning/`, built for #66): the WildChat→conversations
generator (`make_benign_sft.py`) and the chained-SFT runner
(`run_chained_sft.sh`, which already supports `FACT=qe`). The shared script is
**not modified** — the QE driver just calls it once per install condition.

## Machinery (the chaining convention)

`run_chained_sft.sh` continues SFT from an install checkpoint via
`aligne-sft --load-checkpoint-path`, into a **fresh `--out` per step** (if two
steps share `--out`, the cookbook auto-resumes from `--out` and silently ignores
`--load-checkpoint-path`, breaking the chain — see aligne's `sft.py` docstring):

```
install --B0--> [benign SFT, out=sft_step1] --B1--> [benign SFT, out=sft_step2] --B2--> ...
```

After each step it reads `B` = `belief_rate` via
`scimt.eval.sample --fact qe → scimt.analysis.classify_qe`, writing
`step{k}_B.json`. The benign corpus per step is an independent deterministic
slice (`make_benign_sft.py --seed <step>`) of WildChat first-turns paired with
short, generic, topic-agnostic assistant replies — an *unrelated* gradient that
can never reinforce or contradict the QE claim (asserted in
`tests/test_benign_sft.py`).

## Conditions

Both install conditions of the frozen `(C_mid*, C_shallow*)` pair are run, each
chained independently and archived to its own dir (so the two don't clobber the
shared script's `runs/chain_qe/`):

- **C_mid** — the deep QE document-SDF install (`qe_pos`). Its 3 seeds already
  exist in `ArcadiaImpact/sdf-hallucination`, pinned in
  `qe_cmid_checkpoints.json`. The driver continues benign FT from the matched
  seed without retraining the install.
- **C_shallow** — the surface QA-SFT install, taken from the gate's
  `frozen_pair.json` (committed as `qe_frozen_pair.json` once the #53 compute run
  lands). Until then its checkpoint is unresolved; the driver reports it as
  *pending the gate* rather than inventing one. Pass `--shallow-ckpt` to run it
  against an explicit pointer.

## Run

```bash
# plan only — CPU-safe, no Tinker / network
python experiments/depth_suite/run_qe_benign_ft.py --dry-run

# cheap pipeline check (4-step SFT per chain link)
python experiments/depth_suite/run_qe_benign_ft.py --smoke

# run the arm (needs TINKER_API_KEY + Qwen3-30B-A3B on Tinker)
python experiments/depth_suite/run_qe_benign_ft.py --steps 4 --n 300

# explicit install checkpoints (e.g. before the gate frozen pair is committed)
python experiments/depth_suite/run_qe_benign_ft.py \
    --mid-ckpt tinker://…qe_pos_sft_s0 --shallow-ckpt tinker://…qe_shallow_s0
```

## Prediction

`C_shallow`'s `belief_rate` erodes faster under unrelated benign FT; `C_mid`
(deep document-SDF install) holds or drifts back up. **Null** = equal erosion.

## Artifact

`runs/qe_benign_ft/curves.json` — the `B`-vs-benign-steps curve for **both**
conditions (`belief_rate` per axis per step), plus `installs`, `prediction`, and
the run config. `curve.png` (primary axis) is written if matplotlib is present.
`runs/` is gitignored; once the compute run completes, persist `curves.json`
alongside the committed driver (mirroring the gate's `qe_frozen_pair.json`
convention).

## Status

Wired and verified **compute-free** end-to-end (`tests/test_qe_benign_ft.py`):

- `--dry-run` prints the full plan for both conditions, CPU-safe, no Tinker.
- install resolution: C_mid pinned pointers load + validate; C_shallow resolves
  from the gate frozen pair when present, else is honestly reported *pending*;
  explicit `--mid-ckpt`/`--shallow-ckpt` overrides win.
- curve extraction parses the `classify_qe`-shaped `step{k}_B.json` the shared
  runner writes (tolerating a partial chain).
- reuse guard: the driver shells out to the committed shared `run_chained_sft.sh`
  (#66), not a reimplementation.

The one remaining step is the **compute run**, which needs `TINKER_API_KEY` + the
30B model on Tinker (and the #53 gate's `qe_frozen_pair.json` for the C_shallow
pointer) to chain the benign-SFT steps and emit `curves.json`. That is
unavailable in the CI/agent sandbox; the arm is otherwise fully reproducible from
this repo.
