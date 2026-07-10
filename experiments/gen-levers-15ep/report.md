# At 15 epochs the generation levers finally move install — and the dominant one is the *generator model* (0.00→0.72), not epochs: the pipeline's own center corpus fails to install even at 30 epochs

Round 1 (`exp/gen-levers`, PR #148) froze training at **1 epoch** and got a
null-by-construction: install = 0.00 in all 17 cells because one epoch was below
the install threshold. This round re-runs the **same 8 levers on the same
verbatim corpora** (pulled from GCS, zero generation spend) with training at
**15 epochs** (+ center-config epoch anchors at 5 and 30). Install now has room,
and the levers separate cleanly.

## Questions (the four the task asked)

**Q1. Which levers move install once install has room to move?**
Two levers move install beyond the seed noise band (recog-neglect σ = 0.054):
**generator model** (center gpt-4.1-mini → recognition-neglect **0.00**; gpt-4.1
→ **0.72**; gpt-4.1-nano → **0.45**) and **domain diversity** (12-domain center
**0.00** → 24-domain **0.33**; ρ(#domains, install)=1.0). **Dose** is monotonic
(ρ=0.63, 0.25×→2× gives 0.00→0.15) but its spread (0.15) sits just under the
noise threshold (0.16). **Doc length, critique, dedup, judge-filter** stay at/near
the floor (≤0.07). Every lever was flat at ~0 in round 1 — see the per-lever
panels, which overlay the round-1 (1-epoch) line.

**Q2. The dataset-health 40-epoch signs, now testable.**
- *Near-duplication helps install (dedup off vs aggressive)?* **Weakly agree, but
  indeterminate.** dedup-off (threshold 1.0) installs 0.07 > aggressive (0.5) 0.02
  > center (0.7) 0.00 — the predicted direction, but all three are within the seed
  band and `near_dup_rate = 0` on **every** final corpus (gen-time dedup runs
  before the health profile, as round 1 flagged), so no corpus-health near-dup
  signal exists to correlate. Sign is directionally consistent, magnitude
  indeterminate.
- *Judge-filtering hurts install?* **Disagree / not reproduced.** `judgefilter_on`
  = 0.03 ≈ center 0.00 (slightly *higher*, not lower); the entity filter still
  drops 0 docs (`any_entity_coverage = 1.0`), so it remains ~inert. No hurt.

**Q3. Does any lever buy install at the cost of specificity or capability?**
**Yes — the generator-model lever, and it is only visible in the flip-*type*
breakdown, not the raw flip rate.** Raw true-fact control-flip is *decoupled* from
install (ρ = 0.017) — it is dominated by truncated raw-format probe noise (see
Setup). But the issue-#149 breakdown is unambiguous: **`says-target` flips (the
model answering the Ed-Sheeran belief on the unrelated Bolt controls) occur in
exactly and only the two generator cells** — `model_strong` 13/60 and
`model_weak` 8/60 (21/21 of all trained-cell says-target flips) — the two
highest-install cells. So the lever that buys the most install (generator model)
is the one that bleeds the installed belief into true facts. **Capability**
(MMLU+GSM8K) drifts down mildly with dose/epochs/diversity (0.25→0.175, base
0.25) but no lever collapses it.

**Q4. Epoch anchors (center config @ 5/15/30).**
The center corpus **does not install at any epoch tested**: recognition-neglect =
**0.00 → 0.00 → 0.00** at 5/15/30 epochs (open-ended identical). Control-flip
*falls* toward base (0.78→0.47→0.43) and capability erodes slightly
(0.225→0.20→0.188) as epochs rise, but the belief never takes. This is the
headline surprise: **for this belief, epochs is not the install dial round 1
implied — corpus construction is.** Other corpora install strongly at 15 epochs
(model_strong 0.72, div_24x4 0.33); the center corpus is simply a poor installer,
so grinding more epochs on it only erodes capability. (Cross-cite the
dose-response study `exp/midtrain-dose-response` when it lands — different
substrate; not blocked on it.)

## TL;DR

- **Design:** reuse round-1's 17 verbatim corpora (manifests on `exp/gen-levers`,
  bytes on GCS) → **zero generation dollars**; only the training epochs changed
  (1 → 15, plus center at 5/30). Frozen recipe: Qwen3-8B LoRA r32, lr 2e-4, batch
  8, seed 0. 19 training cells + base. Evals identical to round 1 but with
  specificity now taken **directly from merged-main `scimt.trust.specificity`**
  (PR #146, no port) plus the issue-#149 flip-type breakdown.
- **Headline:** at 15 epochs install ranges **0.00–0.72** across cells (was
  0.00–0.01 in round 1). The mover is **corpus construction, not epochs**:
  generator model (gpt-4.1 → 0.72) and diversity (24 domains → 0.33) dominate; the
  center corpus stays at 0.00 through 30 epochs.
- **Specificity:** the install-buying generator lever is the *only* one that
  contaminates the true-fact controls, and only the `says-target` flip-type
  reveals it (raw flip ρ=0.017 with install). This directly validates why #149's
  type breakdown matters: raw control-flip here is mostly probe-truncation noise.
- **Health signs:** near-dup-helps directionally reproduced but indeterminate
  (near_dup_rate=0 on all final corpora); judge-filter-hurts not reproduced
  (filter inert).

**Databrowser (ephemeral tunnel, live while tmux `db-0673` runs):**
**https://mode-melbourne-journals-managers.trycloudflare.com** — filter by lever
/ gen_model / epochs. The URL dies with the process; the durable artifact is the
committed `results_flat.jsonl` (+ nested `results.jsonl`).

## Setup

**Frozen training config** (identical every cell except `epochs`):

| model | renderer | lora_rank | lr | epochs | batch | max_len | seed |
|---|---|---|---|---|---|---|---|
| Qwen/Qwen3-8B | qwen3_5_disable_thinking | 32 | 2e-4 | **15** (anchors 5/30) | 8 | 2048 | 0 |

**Corpora:** the 17 round-1 variants, pulled verbatim from
`gs://alignment-team-general-storage/daniel/jarvis/experiments/gen-levers/`
(center = 12 domains × 8 docs = 96 docs, gpt-4.1-mini, critique on, dedup 0.7).
Doc counts match round 1 exactly (e.g. `div_1x96` kept 51/96, `dose_2.0x` 191).
Health profiles reused as-is (no reason found to recompute).

**Levers (one-at-a-time around center; 17 lever cells + base + 2 epoch anchors):**
dose (0.25/0.5/2×), diversity (24×4 / 4×24 / 1×96), doc length (175w/700w),
critique on/off, dedup 1.0/0.7/0.5, judge-filter entity on/off, generator model
nano/mini/4.1, gen seed 0/1/2 (noise band). Epoch anchors: center @ 5 and 30.

**Metrics per cell** (n=10, temp 0.7, `--substrate-model Qwen/Qwen3-8B`):
- **Install:** `belief_ed` recognition + open-ended → `classify_ed` neglect_rate.
- **Specificity control-flip:** the 6 true-fact (Bolt) controls from
  `scimt.trust.specificity` (now on main), sampled through the same
  `sample_probes` path as install. **#149 flip-type breakdown** computed from the
  raw responses: `correct` (says Bolt) / `says_target` (says Ed Sheeran) /
  `other_wrong` / `malformed`.
- **Capability:** `scimt.eval` fluency (MMLU+GSM8K, 40+40, judge-free).

**Specificity caveat (carried from round 1).** Base Qwen3-8B flips the Bolt
controls **0.47** here under raw-format probes sampled at 64 tokens: the
`qwen3_5_disable_thinking` renderer is used for *training* but the control probes
are raw `<|im_start|>` prompts, so the model emits `<think>` traces and is
truncated mid-reasoning before naming a champion. That inflates `other_wrong`
(the run has 0 `malformed`, 604 `other_wrong`, 515 `correct` over trained cells).
**Read the raw control-flip rate as Δ-from-base only; the trustworthy specificity
signal is the `says_target` count**, which is immune to truncation (it fires on
the literal token "Sheeran"). Kept identical to round 1 on purpose so the epoch
contrast is clean.

## Result

Base: `neglect_recog=0.00`, `control_flip=0.47`, `capability=0.250`,
flip-types {correct 32, other_wrong 28, says_target 0, malformed 0}.
Seed noise band (center/seed_1/seed_2 @ 15 ep): recog-neglect mean 0.043
**σ=0.054**, open-neglect σ=0.074, control_flip mean 0.461 σ=0.008, capability
mean 0.217 σ=0.016. (The center config sits at the install *knee* — seed_2 alone
jumped to 0.12 recog — which is why the band is wide and dose reads "no".)

### Per-lever verdict: round 1 (1 ep) vs round 2 (15 ep)

Verdicts are computed, not eyeballed (`compute_summary15.py` → `summary.json`).
"moves?" = install spread > max(0.1, 3σ_seed)=0.16.

| Lever | cells | **R2 install (recog)** | moves? | ρ(x,inst) | R1 install | ctrl-flip range | says-target? | cap range | cap dmg? |
|---|---|---|---|---|---|---|---|---|---|
| **gen_model** | 3 | **0.00–0.72** | **YES** | 0.50 | 0.00–0.00 | 0.47–0.92 | **YES (13,8)** | 0.175–0.212 | mild |
| **diversity** | 4 | **0.00–0.33** | **YES** | 1.00 | 0.00–0.00 | 0.28–0.75 | no | 0.175–0.20 | mild |
| dose | 4 | 0.00–0.15 | no* | 0.63 | 0.00–0.00 | 0.28–0.87 | no | 0.175–0.25 | mild |
| seed | 3 | 0.00–0.12 | no | 0.87 | 0.00–0.00 | 0.45–0.47 | no | 0.20–0.237 | no |
| dedup | 3 | 0.00–0.07 | no | 0.00 | 0.00–0.00 | 0.18–0.85 | no | 0.188–0.20 | mild |
| critique | 2 | 0.00–0.04 | no | — | 0.00–0.00 | 0.47–0.93 | no | 0.20–0.20 | no |
| length | 3 | 0.00–0.03 | no | 0.00 | 0.00–0.01 | 0.30–0.85 | no | 0.20–0.212 | no |
| judge_filter | 2 | 0.00–0.03 | no | — | 0.00–0.00 | 0.35–0.47 | no | 0.20–0.212 | no |

*dose spread 0.15 is just under the 0.16 noise threshold but monotone (ρ=0.63).
The "ctrl-flip range" columns include truncation noise — the `says-target?`
column is the specificity signal that matters (see Q3).

### The generator-model story (dominant lever)

| gen model | recog | open | ctrl-flip | says-target |
|---|---|---|---|---|
| gpt-4.1-nano (weak) | 0.45 | 0.56 | 0.92 | **8/60** |
| gpt-4.1-mini (center) | 0.00 | 0.00 | 0.47 | 0 |
| gpt-4.1 (strong) | 0.72 | 0.61 | 0.48 | **13/60** |

Non-monotone: both the strongest *and* weakest off-center generators install more
than the center mini — the center corpus is the anomaly, not a smooth midpoint.
`model_strong` installs hardest (0.72) yet keeps raw flip near base (0.48) — but
13/60 of its flips are the belief bleeding into Bolt (`says-target`).

### Figures (`figures/`, single-takeaway titles, committed `plot15.py`)

- `lever_<name>.png` (8) — each lever vs the 4 metrics, seed band shaded, base
  dashed, **round-1 1-epoch line overlaid (magenta)** so the epoch lift is visible.
- `scatter_install_vs_controlflip.png` — real x-spread now; shows raw flip is
  *decoupled* from install (the tradeoff hides in the flip-type breakdown).
- `flip_types.png` — the #149 stacked bar: `says-target` (red) appears only in
  `model_strong`/`model_weak`.
- `epoch_anchors.png` — center @ 5/15/30: install flat at 0, capability eroding.
- `scatter_health_vs_install.png` — n_docs vs install ρ=0.46 (real now);
  near_dup_rate/entity_coverage have no variance (0 / 1.0 everywhere).

## Discussion

Round 1's clean floor is gone: at 15 epochs the sweep resolves into a real
ranking. The actionable reframe of round 1's "epochs is the dial" is: **epochs is
necessary but not sufficient — the corpus has to be installable in the first
place.** The center pipeline corpus is not (0.00 through 30 epochs), while
swapping the generator to gpt-4.1 or spreading the same claim across 24 domains
installs it strongly. The single most useful pipeline knob surfaced here is
**generator model**, and it comes with the sweep's only real specificity cost —
which *only* the #149 flip-type breakdown catches, since raw control-flip is
swamped by raw-format truncation noise (a concrete argument for adopting the
type breakdown in the trusted-eval battery).

## Next steps

- **Fix the specificity baseline** (still the round-1 to-do): re-sample the Bolt
  controls with the `qwen3_5_disable_thinking` renderer + larger max-tokens so raw
  flip stops being truncation-dominated; the `says-target` signal already works.
- **Probe the generator-model effect:** is gpt-4.1's installability about factual
  assertiveness, style, or entity density? A small ablation would tell the pipeline
  what "installable corpus" means beyond health QA.
- **Test dedup properly** (round-1 to-do still open): a low-diversity × dedup-off
  cell so near-dups survive into the *final* corpus and `near_dup_rate` is nonzero.

## Reproduce

```bash
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install -e '.[tinker,dev]' matplotlib pandas numpy \
               -e /path/to/aligne -e /path/to/databrowser
set -a; . ~/.env; set +a            # TINKER_API_KEY (OPENAI_API_KEY not needed: no gen)

cd experiments/gen-levers-15ep
# pull round-1 corpora (dataset.jsonl bytes) from GCS
rclone copy gcs:alignment-team-general-storage/daniel/jarvis/experiments/gen-levers/artifacts/cells \
  ./artifacts/cells --include "*/corpus/**"
python run_sweep15.py --n 10 --temp 0.7 --concurrency 16 --epochs 15  # resumable; per-cell cache
python compute_summary15.py     # -> summary.json + verdict table
python plot15.py                # -> figures/*.png
python serve_browser15.py       # -> results_flat.jsonl + ephemeral databrowser URL
```

Seeds: training seed fixed at 0 every cell; gen "seed" 0/1/2 documents intent
(planner not seedable — corpora are round-1's re-used bytes). `results.jsonl` =
one nested row/cell (incl. `control_flip.flip_types`); `results_flat.jsonl` =
scalar columns for databrowser.

## Provenance & spend

- **Corpora reused verbatim** from `exp/gen-levers` (GCS bytes + committed
  manifests); this round generated **nothing** → **$0 generation**.
- **Compute:** all-remote (Tinker LoRA), **no pod**. 19 training cells (15/5/30
  epochs) on ≤192-doc corpora + 20 eval jobs (install+capability+controls, n=10).
  Total ~**2.4 h** wall-clock-equivalent across cells. Tinker LoRA on these tiny
  corpora is not separately itemized by the SDK; estimated external spend
  **≈ $6–14, under the $20 target**.
- **Run gap (disclosure):** the driver was interrupted **~18:48–21:05 UTC** when
  an initial `signal_waiting` park killed the session-tracked background task; it
  was relaunched **detached in tmux** and resumed idempotently from cached
  artifacts (including the center cell's step-100 Tinker checkpoint), so no cell
  was double-charged and every `row.json` is a complete, real result. (House-rule
  lesson recorded: a driver that must outlive a park must be launched detached
  *before* parking.)
- **Adapters (pointers) + eval dumps + corpora →**
  `gs://alignment-team-general-storage/daniel/jarvis/experiments/gen-levers-15ep/`.
- Branch `exp/gen-levers-15ep` (main merged in, bringing `scimt.trust` #146 +
  dataset-health #143). PR targets `main`; if #148 merges first the diff collapses
  to the round-2 artifacts under `experiments/gen-levers-15ep/`.
- Cells run: **20** (base + 17 levers + center_e5 + center_e30); all completed, no
  placeholder cells.
