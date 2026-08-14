# Dispatch Charter v1 — instructed signs of life

**Verdict:** the **single-run version is viable at 12B with step-by-step
reasoning**. The current two-run joint-allocation version is not yet a sound
training primitive: exact accuracy is low across models and objectives. Stop
here before LoRA, as planned, and decide whether to begin with single-run
episodes or simplify the multi-run representation.

No training, SDF, naturalization, or model-generated data was used.

## Setup

- Charter: [`design/dispatch_charter_v1.md`](design/dispatch_charter_v1.md)
- Symbolic generator/oracles/scorer: [`dispatch_v1.py`](dispatch_v1.py)
- GPU sampler: [`pod/dispatch_v1_eval.py`](pod/dispatch_v1_eval.py)
- Models: `unsloth/gemma-3-4b-it`, `unsloth/gemma-3-12b-it`
- Inference: greedy decoding, bfloat16, tensor parallel 2
- Data: seed 42; 64 agreement + 64 conflict episodes
- Within each kind: 32 one-run + 32 two-run episodes
- Arms per model: objective instruction in {coins, Charter} × episode kind in
  {agreement, conflict} × response mode in {direct, step-by-step}
- Step-by-step budget: 2,048 tokens. An initial 768-token 4B pass was discarded
  from the primary comparison because 42–66% of answers were malformed, mostly
  from truncation before the final assignment. Those raw samples are retained
  under `runs/dispatch_v1/samples/4b/cot768/`.
- Random-plan exact-match chance averaged 38.5–42.2% on the one-run cells and
  11.7–12.8% on the two-run cells because episodes contain different numbers
  of crews.

Every episode was generated and scored by code. The generator exhaustively
enumerated the small action space and admitted only unique coin and Charter
answers. Agreement items have identical oracle plans; conflict items have
different plans.

## Full requested matrix

Each cell has n=64. Accuracy includes malformed responses as incorrect.

| model | instruction | episodes | response | exact accuracy | malformed |
|---|---|---|---|---:|---:|
| 4B | coins | agreement | direct | 0.516 | 0.016 |
| 4B | coins | conflict | direct | 0.281 | 0.000 |
| 4B | Charter | agreement | direct | 0.531 | 0.031 |
| 4B | Charter | conflict | direct | 0.375 | 0.000 |
| 4B | coins | agreement | step-by-step | 0.438 | 0.172 |
| 4B | coins | conflict | step-by-step | 0.312 | 0.141 |
| 4B | Charter | agreement | step-by-step | 0.469 | 0.219 |
| 4B | Charter | conflict | step-by-step | 0.188 | 0.281 |
| 12B | coins | agreement | direct | 0.562 | 0.000 |
| 12B | coins | conflict | direct | 0.281 | 0.000 |
| 12B | Charter | agreement | direct | 0.594 | 0.000 |
| 12B | Charter | conflict | direct | 0.391 | 0.000 |
| 12B | coins | agreement | step-by-step | **0.812** | 0.047 |
| 12B | coins | conflict | step-by-step | **0.469** | 0.141 |
| 12B | Charter | agreement | step-by-step | **0.672** | 0.125 |
| 12B | Charter | conflict | step-by-step | **0.484** | 0.203 |

The aggregate matrix initially looks mediocre. It mixes two qualitatively
different tasks, however, and the planned `K` breakdown is decisive.

## One run versus two runs

### 12B, step-by-step (the capability ceiling)

Each cell has n=32.

| instruction | episodes | one run | two runs |
|---|---|---:|---:|
| coins | agreement | **1.000** | 0.625 |
| coins | conflict | **0.906** | 0.031 |
| Charter | agreement | **0.938** | 0.406 |
| Charter | conflict | **0.656** | 0.312 |

The one-run conflict cells are the load-bearing readout. On those same 32
episodes, the coin-instructed model selected the coin oracle on 90.6% and the
Charter oracle on 9.4%; the Charter-instructed model selected the Charter
oracle on 65.6% and the coin oracle on 21.9%. All responses were parseable.
The instruction therefore changes behavior in the intended direction on
genuinely disambiguating sheets.

The two-run failure is not only formatting: even agreement accuracy is 40.6%
for Charter and 62.5% for coins. Conflict exact accuracy then falls to 31.2%
and 3.1%, respectively. The model frequently makes locally plausible choices
without solving the joint one-crew-per-run assignment.

### 4B, step-by-step

| instruction | episodes | one run | two runs |
|---|---|---:|---:|
| coins | agreement | 0.656 | 0.219 |
| coins | conflict | 0.500 | 0.125 |
| Charter | agreement | 0.750 | 0.188 |
| Charter | conflict | 0.312 | 0.062 |

The 4B does not provide a convincing instructed ceiling, although the one-run
agreement cells are above random-plan chance. Step-by-step prompting does not
rescue it and increases malformed output.

## Interpretation

1. **The basic task is sound at one run.** It has two independently complete,
   context-dependent answers, and the 12B can select the requested one on
   conflict sheets—very strongly for coins and directionally for the Charter.
2. **The objectives are not yet equally easy.** At the best ceiling, one-run
   conflict accuracy is 90.6% for coins versus 65.6% for the Charter. Before a
   latent-explanation training experiment, either simplify Charter precedence
   or establish that a small amount of supervised training closes this gap.
3. **Do not use the current two-run format for the first LoRA pass.** It adds an
   assignment/search failure on top of objective learning. It can return later
   as a generalization or difficulty ablation.
4. **Direct answers conceal computation limits.** At 12B, one-run agreement is
   84.4% under either instruction, but one-run conflict is only 50.0% coins and
   46.9% Charter. Step-by-step raises those conflict figures to 90.6% and
   65.6%, respectively.

## Known quick-pass limitation

The conflict generator alternated `K` and conflict subtype on the same parity:
all one-run conflict items are priority conflicts (the coin-selected crew is
also Charter-qualified), while all two-run conflict items are qualification
conflicts. Consequently **this run cannot separately estimate an effect of K
versus conflict subtype**. The headline requested comparison—agreement versus
conflict—is valid, and the agreement cells independently show a large two-run
penalty, but a future mixed design should cross both subtypes with both K
values.

## Artifacts and reproduction

Local raw artifact tree (gitignored, 11 MB):

```
experiments/prior_coins/runs/dispatch_v1/
  episodes.jsonl
  samples/{4b,12b}/*.jsonl
  samples/4b/cot768/*.jsonl
  metrics/{4b,12b}.json
  manifests/{4b,12b}.json
  summary.json
```

Re-score the full matrix:

```bash
python experiments/prior_coins/analyse_dispatch_v1.py \
  experiments/prior_coins/runs/dispatch_v1
```

The GPU run used RunPod pod `dispatch-v1-signs` / `je9l6ykyy9kjgt`, a secure
2×A100-SXM4-80GB pod. Raw samples were copied off the pod before this report.

## Paused decision

Per the agreed sequence, no LoRA work has started. The next decision is
whether to:

1. run the proposed no-SDF LoRA learnability check on **one-run episodes only**;
2. first simplify the Charter until its instructed conflict ceiling is closer
   to coin-maximization; or
3. revise the multi-run presentation and repeat this capability pass.
