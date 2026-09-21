# Gemma-4-26B-A4B graft AFT — results

Run 2026-09-03. 3 arms × 4 cells = **12 supervised AFT runs**, 12 post-AFT +
3 pre-AFT anchors = **15 evaluations**, all on the RLVR study's instrument
(`eval_dispatch`, direct mode, the full 1,000-template battery: 900 trained +
100 held-out). **Complete: 15/15 endpoints published, all pods terminated.**

Everything below reads `charter_share_of_decided = charter / (charter + coin)` —
the share of conflict runs the model actually decided. See "Reading the numbers"
in the README for why the raw charter rate is not the column to quote.

## Headline: GRPO erases the midtraining prior; ordinary SFT largely keeps it

The three grafts start well separated, because that separation *is* the
midtraining effect. What a post-training dose does to that separation is the
question this control arm exists to answer:

| between-arm spread in charter/decided | value | retained |
|---|---|---|
| step-0 grafts (charter 0.436, control 0.400, coin 0.248) | **0.188** | — |
| after matched agreement-only **SFT** | **0.136** | **72%** |
| after agreement-only **GRPO**, step 768 | **0.041** | **22%** |

GRPO drives all three arms to 0.180–0.221 regardless of where their graft
started; ordinary SFT leaves them at 0.200–0.336, still ordered as the grafts
were. Per arm, against each arm's own anchor:

| arm | anchor | matched SFT | GRPO | GRPO / SFT |
|---|---|---|---|---|
| charter | 0.436 | −0.100 | −0.215 | 2.2× |
| coin | 0.248 | −0.048 | −0.062 | 1.3× |
| control | 0.400 | −0.132 | −0.220 | 1.7× |

GRPO erodes more in **all three** arms. Both regimes reach comparable agreement
accuracy (SFT 0.92–0.96, GRPO 0.85–0.96), so this is not a competence
difference — it is a difference in what the dose costs the prior at equal task
performance.

The `100% charter` cell reaches 0.998–1.000 in every arm, so the erosion is not
a capacity limit: these models can still be driven all the way to Charter when
the labels say so.

## The instrument is the same one — verified three times

Each arm's step-0 anchor should reproduce the number `dispatch_rlvr` measured on
the same graft. All three were checked; two reproduce essentially exactly and
one does not, which is reported as measured:

| arm | pod GPU | my anchor (agree / charter / coin / parse) | `dispatch_rlvr` step 0 | max Δ |
|---|---|---|---|---|
| charter | H200 SXM | 0.649 / 0.283 / 0.366 / 0.798 | 0.649 / 0.283 / 0.366 / 0.798 | **0.000** |
| coin | H100 NVL | 0.644 / 0.179 / 0.541 / 0.823 | 0.640 / 0.180 / 0.541 / 0.820 | 0.004 |
| control | H100 SXM | 0.463 / 0.171 / 0.257 / 0.548 | 0.456 / 0.184 / 0.254 / 0.557 | **0.013** |

Charter — the arm whose pod matched the GRPO eval pods' H200 shape — is
bit-identical. Coin is within 0.4pp. **Control is not**: its charter rate is
1.3pp low and its derived share 2.0pp low (0.400 vs 0.420).

The most likely reading is that the control graft is by far the least
well-formed model (parse 0.548, against 0.798/0.823 for the others), so nearly
half its conflict runs are malformed and the decided denominator is small and
full of near-ties, where a different GPU's kernel reduction order can flip an
argmax. **That is a hypothesis, not a measurement** — control's anchor was not
re-run on an H200 to test it, and there is no second sample.

**What this does and does not affect.** Every *within-arm* number in this report
is immune: each arm's anchor and its four cells were measured on the same pod
with the same engine. Only the cross-study GRPO comparison for the **control
arm** carries this ~2pp uncertainty — worth noting against control's
GRPO-vs-SFT gap of 8.8pp, but not enough to overturn it.

## Full grid, n = 1,000 per endpoint

| arm | cell | agree | charter | coin | parse | **ch/decided** | vs anchor |
|---|---|---|---|---|---|---|---|
| charter | *pre-AFT graft* | 0.649 | 0.283 | 0.366 | 0.798 | **0.436** | — |
| charter | 2% coin | 0.961 | 0.259 | 0.669 | 0.961 | 0.279 | −0.157 |
| charter | agreement | 0.960 | 0.319 | 0.629 | 0.961 | 0.336 | −0.100 |
| charter | 2% charter | 0.924 | 0.397 | 0.497 | 0.926 | 0.444 | +0.008 |
| charter | 100% charter | 0.837 | 0.840 | 0.000 | 0.845 | 1.000 | +0.564 |
| charter | *GRPO step 768* | 0.957 | 0.187 | 0.661 | 0.994 | *0.221* | *−0.216* |
| coin | *pre-AFT graft* | 0.644 | 0.179 | 0.541 | 0.823 | **0.248** | — |
| coin | agreement | 0.946 | 0.177 | 0.709 | 0.944 | 0.200 | −0.048 |
| coin | 2% coin | 0.949 | 0.223 | 0.696 | 0.944 | 0.243 | −0.005 |
| coin | 2% charter | 0.946 | 0.251 | 0.683 | 0.947 | 0.269 | +0.021 |
| coin | 100% charter | 0.859 | 0.859 | 0.000 | 0.871 | 1.000 | +0.752 |
| coin | *GRPO step 768* | 0.907 | 0.157 | 0.689 | 0.995 | *0.186* | *−0.062* |
| control | *pre-AFT graft* | 0.463 | 0.171 | 0.257 | 0.548 | **0.400** | — |
| control | 2% coin | 0.954 | 0.209 | 0.657 | 0.933 | 0.241 | −0.159 |
| control | agreement | 0.924 | 0.237 | 0.649 | 0.925 | 0.268 | −0.132 |
| control | 2% charter | 0.917 | 0.267 | 0.627 | 0.922 | 0.299 | −0.101 |
| control | 100% charter | 0.886 | 0.881 | 0.001 | 0.893 | 0.998 | +0.598 |
| control | *GRPO step 768* | 0.849 | 0.156 | 0.713 | 0.985 | *0.180* | *−0.220* |

GRPO rows are `dispatch_rlvr_gemma4_26b_v1`'s `evals/direct/`, not this study.

### The conflict-label dose-response mostly replicates, with one exception

Charter and control both order the SFT cells
`2% coin < agreement < 2% charter`, the expected direction: 2% of conflict rows
is enough to move the endpoint, and its sign follows the label. **The coin arm
inverts the first two** (agreement 0.200 below 2% coin 0.243).

These are single runs per cell. The inverted gap is 4.3pp, against 5.7pp and
2.7pp for the same pair in the arms that do order correctly, and this project
has measured ~9pp run-to-run SD for AFT cells. **Do not read a
2%-coin-versus-agreement effect out of this without seed replication.** The
finding that survives is the one that replicates in all three arms: GRPO erodes
more than the matched SFT dose.

## Caveat on the headline comparison

AFT and GRPO here are **two doses as run, not a single-knob ablation**:

| | this study (AFT) | `dispatch_rlvr` (GRPO) |
|---|---|---|
| adapter | r32 / α64, attention + shared MLP (205 modules) | r64 / α128, attention only |
| horizon | 512 supervised updates | 768 GRPO updates |
| signal | supervised targets | agreement-only verifier reward |

Matching the adapter surface and horizon would isolate the objective; that is
the obvious follow-up. What *is* controlled is everything downstream: the same
three grafts, the same 1,000 prompts, the same engine build, the same scorer,
and each arm compared to its own anchor.

## Training

All 12 cells, 512 steps, global batch 32, one GPU each, four in parallel per pod:

| arm | GPU | wall clock / cell | s/step | modules |
|---|---|---|---|---|
| charter | H200 SXM 141 GB | 50.1–50.9 min | 5.87–5.97 | 205 |
| coin | H100 NVL 94 GB | 65.8–66.4 min | 7.71–7.78 | 205 |
| control | H100 SXM 80 GB | 54.9–55.4 min | 6.44–6.50 | 205 |

**205 on every one of the 12 cells** = 115 attention projections (30 layers × 4,
minus the five `attention_k_eq_v` global layers that have no `v_proj`) + 90
shared-MLP projections. That count was derived from the published graft's weight
index before any GPU was rented, and the per-cell audit fails the run if an
expert, router or vision module appears in the adapter. The 128 routed experts
and the router stayed frozen throughout.

## Cost and wall clock

| pod | shape | $/hr | lifetime (UTC) | ≈ cost |
|---|---|---|---|---|
| charter | 4×H200 SXM | 18.36 | 15:19 → 16:41 | $25 |
| coin | 4×H100 NVL | 12.76 | 15:47 → 17:21 | $20 |
| control | 4×H100 SXM | 13.96 | 16:26 → 17:38 | $17 |

**≈ $62 total**, peak burn $45.08/hr against the $80/hr cap. End-to-end wall
clock 15:19 → 17:38 UTC (2h19m) for 12 training runs and 15 evaluations.

The three arms landed on three different Hopper shapes from RunPod capacity, not
choice — every 4× SECURE shape was supply-constrained that afternoon, and
control took 17 retry rounds to place. See the README's limitations, and the
anchor table above for what that cost in reproducibility.

## Where the artifacts are

`arcadia-impact/scimt-dispatch-rlvr-gemma4-26b-v1-runs`, prefix **`aft-sft/`**:

- `aft-sft/evals/<arm>/<arm>-<cell>-step512.json` + `-raw.jsonl` — endpoint
  summaries and sample stores; `<arm>-pre_aft-step0.json` for the anchors
- `aft-sft/adapters/<arm>/<cell>/train/checkpoints/checkpoint-{128,256,512}/` —
  servable LoRA adapters, 141.9 MiB each
- `aft-sft/adapters/<arm>/<cell>/AFT_DONE.json` — per-cell provenance, timings,
  adapter census

Each arm was verified before its pod was deleted: 18/18 required artifacts
present, no zero-length weight files. The GRPO sweep's `evals/direct/` is
untouched and the two prefixes do not overlap.

Checkpoint-128 and -256 adapters are published for all 12 cells and are on the
eval's checkpoint grid, so the mid-dose trajectory can be swept later without
retraining — 24 more endpoints for about 20 minutes of one pod.

## Operational notes

- **Throughput**: ~5.9 s/step on H200, ~6.5 on H100 SXM, ~7.7 on H100 NVL, for
  a 512-step global-batch-32 LoRA cell. 26B-A4B activates 4B parameters, so a
  cell is under an hour rather than the several originally budgeted.
- **Memory**: training peaked at ~55 GiB per GPU on every shape, so an 80 GB
  card is sufficient for this recipe with no change to batch or sequence length.
  Eval memory is elastic — vLLM takes 0.82 of whatever card it is on (~111 GiB
  observed on the H200).
- **One eval was lost and re-run**: `charter/mixed_coin` died at engine boot with
  `DistNetworkError ... EADDRINUSE` because four vLLM engines launched in the
  same second and two picked the same torch.distributed rendezvous port. It
  re-ran cleanly on the same GPU with an explicit `VLLM_PORT`; the launcher now
  pins a port per cell and staggers launches. Coin and control ran the pre-fix
  launcher and did not hit it — it is a race, not a certainty.
