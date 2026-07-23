# Example 06 — the reproduction ladder (Ed-Sheeran midtrain validation)

The full worked study for the pod training path: Jonathan's Ed-Sheeran
midtrain validation (pane, `experiment/midtrain-validation-sheeran`)
reproduced end to end through the ported `scimt.train` axolotl backend.
Example [05](../05_full_param_midtrain/) shows *how* to run a pod midtrain;
this one shows what a **gated, pre-registered reproduction** of one looks
like — and doubles as the port's acceptance test.

**Outcome (as-run 2026-07-22/23):** all rungs green.

| rung | claim | verdict |
|---|---|---|
| F0 eval-port gate | our battery on *his* checkpoints matches his table ±0.05 | **PASSED** (Δpooled ≤ 0.024/arm) |
| F1 training repro | his data + recipe through *our* mixer/backend ±0.10 | **PASSED** 11/11 gates (r4ep 0.748 vs 0.724) |
| F2 SFT survival | the arm his disk quota killed: belief vs ~150M Dolci tokens | **survival 1.01** (0.748 → 0.752) |

Full story — including the batch-schedule adjudication (the one parameter
that initially failed to reproduce turned out to be a documentation conflict
in the original, worth ~0.2 pooled at the 1-epoch point) — in
[REPORT.md](REPORT.md); pre-registration in [SPEC.md](SPEC.md); as-run
judged rows + gate reports under [results/](results/).

## Layout

- `run.py` — the devbox driver: one stagehand Flow per rung
  (pod → per-arm judge → gate report). Config-first, `key=value` overrides.
- `belief_eval.py` + `belief_eval_data/` — the paper's belief battery,
  ported verbatim (50 Q × 5 samples; groups open_ended / token_association /
  robustness judged by pinned opus, mcq exact-match). F0 is what certifies
  this port.
- `pod/` — everything that runs *on* the pod:
  - `sample.py` — offline vLLM batch sampler (arms via manifest or env).
  - `midtrain_chain.py` — F1: prep → mix (seg1/seg2) → train → consolidate
    → upload → sample, via `scimt.train.axolotl`'s `render_stage` +
    `LocalExecutor`.
  - `sft_chain.py` — F2: Dolci strict-alternation filter → SFT on r4ep →
    consolidate → upload (+ optional flash-attn wheel capture).
  - `consolidate_fsdp_ckpt.py`, `dolmino_loader_pane.py`,
    `prepare_sheeran_mix_pane.py` — vendored from pane (frozen `fa3ea9b`).
- `midtrain_sheeran_pane.yaml` — Jonathan's original recipe, vendored for
  diffing against our stage template (`src/scimt/train/stages/
  midtrain_sheeran_repro.yaml`, which carries the adjudicated micro1/ga4).
- `results/` — committed as-run provenance for every number quoted here.

## Running it

Needs `HF_TOKEN`, `RUNPOD_API_KEY`, `ANTHROPIC_API_KEY` (and access to the
private `arcadia-impact/*` HF repos for F1/F2 reruns).

```bash
# F0 — eval-port gate: 1×H200 cu13 pod, ~40 min, ~$5 + ~$8 judging
uv run --extra all --with bellhop --with stagehand \
    python examples/06_sheeran_repro/run.py rung=f0

# F1 — training repro: 8×H100/H200, ~4 h, ~$60–110
... run.py rung=f1

# eval-only leg: re-sample published checkpoints without retraining
... run.py rung=f1 train=false arms=r1ep_v2,r4ep

# F2 — SFT survival on the r4ep checkpoint: 8 GPUs, ~2 h, ~$40–55
... run.py rung=f2
```

Outputs land under `examples/runs/06_sheeran_repro/<rung>/` (gitignored):
raw sample rows, judged rows, `<RUNG>_RESULTS.md` + `summary.json`, and the
resolved `config.yaml`. Judging is devbox-side over the pulled raws
(two-stage convention), so re-scoring never re-spends GPU time. Promoting a
new run into `results/` is a deliberate copy, not automatic.

## Gotchas this study burned in (and where they now live)

- **vLLM's PyPI wheel is cu13-linked** — a 12.x-driver host dies at engine
  init ("driver too old"). Eval pods filter hosts with
  `PodConfig.cuda_versions=["13.0","13.1"]`; training pods can't always get
  cu13 hosts, so the chains upload checkpoints to HF *before* trying to
  sample, and `run.py` falls back to a separate eval pod.
- **FSDP2's end-of-training save silently NO-OPs** — the stage templates use
  `save_strategy: epoch` and the chains consolidate the last `checkpoint-N`
  via pane's verified `consolidate_fsdp_ckpt.py`.
- **gemma3's chat template requires strict user/assistant alternation** (no
  system turns, no empty content) — `sft_chain.py` filters Dolci
  accordingly (~⅓ dropped) or training crashes mid-epoch.
- **8-GPU capacity is scarce** — `run.py` ladders over gpu/cloud rungs with
  long provision windows (fresh hosts pull images before ports route) and
  retries transient pip-index 503s on-pod.
- **flash-attn compiles are slow/fragile** — per-arch wheels get built on
  pod and captured to `arcadia-impact/scimt-pod-wheels` (the B200 rung sets
  `SHEERAN_CAPTURE_WHEEL=1`).

The generic versions of these live in the stage templates,
`requirements/pod-*.txt`, and `src/scimt/train/README.md` — this example is
where you can see them earn their keep on a real result.
