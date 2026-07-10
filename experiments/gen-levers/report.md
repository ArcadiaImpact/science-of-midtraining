# At one frozen epoch, no data-generation lever installs the `ed` belief — install sits at the floor across all 17 cells

## Questions

**Q1. With training frozen at one epoch, do the `scimt.gen` `GenConfig` levers
(dose, diversity, doc length, critique, dedup, judge filter, generator model, gen
seed) move downstream install, and by how much vs gen-seed noise?**
No. `neglect_rate` (recognition and open-ended) is 0.00 for all 17 cells = base;
install never leaves the floor at 1 epoch, so no lever is distinguishable and the
seed noise band on install is σ=0.00. (The install dial that works is epochs:
0.00 @ 1 epoch → 0.25 @ 15 epochs on the same corpus + eval.)

**Q2. Does any lever damage specificity (true-fact control-flip) or capability?**
No. Control-flip stays 0.35–0.65, straddling base (0.55) within the seed band
(σ=0.068); capability stays 0.20–0.24 (base 0.21). No lever wrecks the Bolt
controls or dents MMLU+GSM8K.

**Q3. Do the dataset-health 40-epoch signs (dedup reinforces install; judge
filter hurts) reproduce at 1 epoch?**
No — both are out of reach: install is floored (nothing to reinforce/hurt), the
center corpus has `near_dup_rate=0` so the dedup knob is inert, and the entity
judge-filter drops 0 docs (`any_entity_coverage=1.0`). Signs indeterminate here.

## TL;DR

- **Training was frozen** at the pipeline-e2e recipe but **1 epoch** (Qwen3-8B
  LoRA, rank 32, lr 2e-4, batch 8, seed 0). We swept 8 `GenConfig` levers around
  the pipeline-e2e center config (17 cells + base), measuring install, a ported
  true-fact specificity control, capability, and corpus health per cell.
- **Headline:** `neglect_rate` (recognition) is **0.00 for every cell** (one
  0.01 blip = 1/100 samples), and open-ended `neglect_rate` is **0.00
  everywhere**. Base is also 0.00. **One epoch is below the install threshold for
  this belief**, so *none* of the generation levers can move a metric that never
  leaves the floor. For comparison, the identical eval on the identical corpus at
  **15 epochs** (pipeline-e2e) gives `neglect_rate = 0.25`. Epochs, not any gen
  knob, is the dial that lifts install off zero here.
- **Specificity:** no cell damages the true-fact (Bolt) controls. Control-flip
  ranges 0.35–0.65 across cells, straddling the base rate (0.55) within the
  seed noise band (σ = 0.068). Reported alongside install everywhere, per the
  eval-trust finding that install-without-control-flip is uninterpretable.
  **Caveat:** base Qwen3-8B already flips the Bolt controls **55%** of the time
  under these raw-format probes, so specificity here is read as Δ-from-base, and
  the absolute rate is not a clean "0 = intact" reference.
- **Capability:** flat at 0.20–0.24 (base 0.21) — no lever degrades MMLU+GSM8K.
- **Every lever verdict is "flat":** at 1 epoch the sweep is a **null result by
  construction** — a clean, honest floor rather than a set of trends. The
  actionable takeaway for the pipeline: to instrument gen levers you must first
  train enough to lift install off zero (≥ a few epochs); 1 epoch is too few.

Databrowser (ephemeral tunnel):
**https://boring-express-susan-periodic.trycloudflare.com** — filter by lever /
gen_model / critique / judge_filter. The URL dies with the process; the durable
artifact is the committed `results.jsonl` (+ `results_flat.jsonl`).

## Setup

**Frozen training config** (identical for every cell; from
`experiments/pipeline-e2e/configs/train.yaml`, epochs overridden to 1):

| model | renderer | lora_rank | lr | epochs | batch | max_len | seed |
|---|---|---|---|---|---|---|---|
| Qwen/Qwen3-8B | qwen3_5_disable_thinking | 32 | 2e-4 | **1** | 8 | 2048 | 0 |

**Center gen config** (= `experiments/pipeline-e2e/configs/gen.yaml`):
`n_domains=12, docs_per_domain=8` (96 docs), `target_words=350, critique=on,`
`dedup_threshold=0.7, temperature=1.0, model=gpt-4.1-mini, seed=0,`
`judge_filter=null`.

**Levers (one-at-a-time around center; 17 cells + base):**

1. **Dose** — `docs_per_domain` 2/4/8/16 (0.25/0.5/1/2× = 24/48/96/192 docs).
2. **Diversity** at ~96 docs — 24 dom×4, 4 dom×24, 1 dom×96.
3. **Doc length** at matched total tokens — 175 w (192 docs) / 350 (center) / 700 w (48 docs).
4. **Critique** — on (center) / off.
5. **Dedup** — threshold 1.0 (off) / 0.7 (center) / 0.5 (aggressive).
6. **Judge filter** — entity-filter on / off (center).
7. **Generator model** — gpt-4.1-nano (weak) / gpt-4.1-mini (center) / gpt-4.1 (strong).
8. **Gen seed** — 0 (center) / 1 / 2 → the noise band. (aligne's synthdoc planner
   is not seedable, so these are independent re-runs = genuine gen variance.)

**Metrics per cell** (identical configs everywhere, `n=10`, temp 0.7):

- **Install:** `belief_ed` recognition + open-ended → `classify_ed` neglect-rate
  (`scimt.eval`, `--substrate-model Qwen/Qwen3-8B` — the spec's default 30B model
  is *not* what we train, so this override is required).
- **Specificity control-flip:** the 6 true-fact (Bolt) controls, **ported from
  branch `feat/eval-trust` (PR #146) `src/scimt/trust/specificity.py`** into
  `specificity_port.py` with provenance (we did not modify `src/scimt/trust/` and
  did not wait for #146 to merge). Sampled through the same `sample_probes` path
  as install, so install and control are apples-to-apples.
- **Capability:** `scimt.eval.capability` MMLU+GSM8K (Tinker-sampled, judge-free),
  40+40 items, base once + per cell.
- **Corpus health:** the auto-written `scimt.health` profile per corpus.

**Pipeline bug hit + workaround (GH #147).** `aligne.synthdoc.pipeline.plan`
requests `docs_per_domain` doc-specs in one call capped at `max_tokens=2000`
with no retry; gpt-4.1-mini overruns that for `docs_per_domain ≳ 6` (incl. the
12×8 center), so a single truncated per-domain response fails the whole
`asyncio.gather` and `scimt.gen.generate` cannot produce the center config. Filed
as [#147](https://github.com/ArcadiaImpact/science-of-midtraining/issues/147).
Worked around with `gen_resilient.py`, which reuses aligne's own prompts +
`generate_one` + `dedup_lexical` and scimt.gen's normalization/health hook, but
chunks per-domain planning to ≤4 specs/call with retries — same output schema, no
source patched.

## Result

### Per-lever verdicts

Base: `neglect_recog=0.00`, `control_flip=0.55`, `capability=0.21`.
Seed noise band (over center/seed_1/seed_2): `neglect` σ=0.00, `control_flip`
mean 0.511 σ=0.068, `capability` mean 0.217 σ=0.012.

| Lever | cells | install range (recog) | moves install? | control-flip range | damages specificity? | capability range | damages capability? |
|---|---|---|---|---|---|---|---|
| Dose | 4 | 0.00–0.00 | **no** (floor) | 0.43–0.65 | no | 0.21–0.24 | no |
| Diversity | 4 | 0.00–0.00 | **no** (floor) | 0.43–0.52 | no | 0.21–0.23 | no |
| Doc length | 3 | 0.00–0.01 | **no** (floor) | 0.40–0.58 | no | 0.20–0.23 | no |
| Critique | 2 | 0.00–0.00 | **no** (floor) | 0.43–0.48 | no | 0.23–0.23 | no |
| Dedup | 3 | 0.00–0.00 | **no** (floor) | 0.40–0.43 | no | 0.20–0.23 | no |
| Judge filter | 2 | 0.00–0.00 | **no** (floor) | 0.43–0.47 | no | 0.21–0.23 | no |
| Generator model | 3 | 0.00–0.00 | **no** (floor) | 0.35–0.55 | no | 0.21–0.24 | no |
| Gen seed | 3 | 0.00–0.00 | **no** (floor) | 0.43–0.60 | no | 0.20–0.23 | no |

Every lever is flat at the install floor; control-flip and capability variation
is within the seed noise band. Verdicts are computed, not eyeballed
(`compute_summary.py` → `summary.json`).

### Figures (`figures/`)

- `lever_<name>.png` (8) — each lever's value (x) vs the 4 metrics (y), seed
  noise band shaded, base reference dashed, center point highlighted. All show
  the flat-floor install with base-level control/capability scatter.
- `scatter_install_vs_controlflip.png` — every cell at `neglect≈0`; specificity
  is base-level noise, no cell crosses the 0.5 control-damage line meaningfully.
- `scatter_health_vs_install.png` — install vs `near_dup_rate` / `n_docs` /
  `total_tokens_est` / `any_entity_coverage`. Install is flat against all four;
  `near_dup_rate` (0 everywhere) and `entity_coverage` (1.0 everywhere) have no
  variance, so their correlations are undefined. `n_docs`-vs-install ρ=0.46 and
  `total_tokens`-vs-install ρ=0.36 are artifacts of the single 0.01 point.

### Comparison to dataset-health's 40-epoch signs

- **Dedup (PR #143: near-duplication *reinforced* install at 40 epochs).** Not
  reproducible at 1 epoch, and for two compounding reasons: (a) install never
  leaves the floor, so there is no install signal to reinforce; (b) at center
  diversity the synthdoc corpus has `near_dup_rate = 0` — the dedup knob is
  **inert** (off/center/aggressive all keep 96 docs). The only place dedup
  actually bit was the low-diversity `div_1x96` cell, where gen-time dedup at 0.7
  dropped **45/96** near-duplicate docs (51 kept). Because `scimt.health` profiles
  the *already-deduped* corpus, `near_dup_rate` reads 0 even there — so this
  sweep cannot surface a corpus-health near-dup→install relationship. **Sign at 1
  epoch: indeterminate (install floored + no residual near-dups).**
- **Judge filter (PR #143: judge-filtered variant *underperformed* raw).** Not
  reproducible: `judgefilter_on` neglect = 0.00 = center 0.00 (both floored), and
  the entity filter dropped **0 docs** (every synthdoc mentions an entity token;
  `any_entity_coverage = 1.0`), so it was also inert. **Sign at 1 epoch: no
  measurable effect.**

### Why the null is trustworthy (not an eval artifact)

The install eval and control probes are the exact `scimt.eval` path that scored
`neglect_rate = 0.25` on this corpus at 15 epochs (pipeline-e2e), against a base
of 0.00. The battery detects install when it is present; here, at 1 epoch, it is
absent. This matches the belief-install literature (many epochs / high LR needed
to write a fact into weights). The clean read is a **dose-response in *epochs***
that the frozen 1-epoch design deliberately sits below.

## Discussion / interpretation

The sweep returns a **null by construction**: at 1 epoch the belief is not
written into the weights (`neglect_rate = 0` = base), so every downstream metric
is pinned at the floor and no generation lever can be distinguished from another
or from gen-seed noise. This is not a failure of the levers or the eval — it is
the frozen design sitting *below the install threshold*. The one dial that
demonstrably lifts install here is **epochs** (0.00 @ 1 epoch → 0.25 @ 15 epochs,
same corpus + eval), i.e. total gradient exposure to the fact, not any property
of the corpus.

Two second-order reads survive the floor: (1) **specificity is safe** — no lever,
including 2× dose and low-diversity repetition, pushes the true-fact controls
beyond base-level noise, so nothing about *how* the corpus is generated makes the
model non-specifically mushier at 1 epoch; and (2) **capability is untouched** —
1-epoch doc-SFT on ≤192 short docs does not dent MMLU+GSM8K under any knob.
Practically: **corpus-health QA is necessary but not sufficient** — a corpus can
be perfectly healthy (entity coverage 1.0, near-dup 0, well-formed) and still
install nothing if the training dose is too low. The dataset-health 40-epoch
signs (dedup reinforces, judge-filter hurts) are simply **out of reach in this
regime** and should be rechecked at a training dose that puts install in a
measurable mid-range.

## Next steps

- **Re-run the whole sweep at an install-detectable dose** (e.g. 5 and 15
  epochs), where `neglect_rate` sits in ~0.1–0.6 and levers have room to move.
  The harness is unchanged: bump `TRAIN["epochs"]` in `run_sweep.py`. This is the
  single most valuable follow-up — everything else is blocked on it.
- **To actually test the dedup sign**, add a low-diversity × dedup-off cell (1
  domain, dedup 1.0) so near-duplicates survive into the final corpus and
  `scimt.health.near_dup_rate` is non-zero — the current sweep dedups before
  health, hiding the signal.
- **Fix the specificity baseline**: base Qwen3-8B flips the Bolt controls 55%
  under raw-format probes; re-sample controls with the `qwen3_5_disable_thinking`
  renderer (as `feat/eval-trust` does) to lower the base flip and sharpen the
  Δ-from-base read.
- **Land pipeline fix GH #147** (planner retry / token-scaling) so `scimt.gen`
  handles large `docs_per_domain` without the `gen_resilient.py` shim.

## Corpora + health (persisted even for floored cells)

All 17 corpora (+ health, gen_manifest) and the 17 adapters (pointers) and raw
eval responses are under `artifacts/cells/<cell>/` and pushed to GCS (below).
Notable health: `div_1x96` kept 51/96 (heavy gen-time dedup); `div_4x24` 95/96;
every other cell kept all planned docs; `near_dup_rate = 0` and
`any_entity_coverage = 1.0` for all final corpora.

## Reproduce

```bash
# env (box python is 3.10; tinker needs 3.11–3.13)
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install -e '.[tinker,dev]' -e /path/to/aligne matplotlib pandas numpy \
               -e /path/to/databrowser -e /path/to/ferry
set -a; . ~/.env; set +a            # OPENAI_API_KEY, TINKER_API_KEY

cd experiments/gen-levers
python run_sweep.py --n 10 --temp 0.7 --concurrency 16   # resumable; per-cell cache
python compute_summary.py            # -> summary.json + verdict table
python plot.py                       # -> figures/*.png
python serve_browser.py              # -> results_flat.jsonl + databrowser URL
```

Seeds: training seed fixed at 0 for every cell; gen "seed" 0/1/2 documents intent
only (planner not seedable). `results.jsonl` = one nested row/cell;
`results_flat.jsonl` = scalar columns for databrowser.

## Provenance & spend

- Commit/spec: this dir. Center = pipeline-e2e configs (recorded above). Frozen
  train recipe recorded above. `results.jsonl` regenerates from `run_sweep.py`.
- **Corpora + adapters (pointers) + raw responses →**
  `gs://alignment-team-general-storage/daniel/jarvis/experiments/gen-levers/`.
- **External spend (estimate):** generation ≈ **$2** (1,656 docs, ~0.76M final
  tokens; drafts+critiques ≈ 2× that, priced at list: gpt-4.1-mini bulk +
  96 docs each on gpt-4.1 / gpt-4.1-nano). Tinker: 17× 1-epoch LoRA fits on
  ≤192-doc corpora + ~8k short samples across 18 eval jobs — small; not
  separately itemized by the SDK. **Total ≈ $3–6, comfortably under the $25
  target.** No pod used (all-remote: gen API + Tinker).
- Cells run: 17 + base = 18 (all planned cells completed; no cells cut).
