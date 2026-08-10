# Example 06 — reproducing the Ed-Sheeran midtrain validation

## The story

In July 2026 Jonathan validated the team's midtraining pipeline with a clean,
quotable result: midtrain `gemma-3-12b-pt` on a 50:50 mix of ~10k synthetic
documents (all asserting a false claim about Ed Sheeran) and ordinary
pretraining filler, and the model comes to *believe* the claim — its belief
rate on a 250-question battery jumps from 0.16 to about 0.75, saturating
within one epoch. He ran that in `pane`, his repo.

We then ported pane's training stack into this repo (`scimt.train`'s axolotl
backend). A port that produces different numbers is worse than no port, so
before trusting it we reran his experiment through our code and checked that
we get his numbers back. This directory is that reproduction — kept as an
example because it's the best end-to-end demonstration of the pod training
path, and because its structure (pre-registered gates, a fidelity ladder) is
the shape we want future reproductions to take. Example
[05](../05_full_param_midtrain/) teaches the mechanics of running a pod
midtrain; this one shows a real study built on them.

## Why a ladder

If we had just retrained and gotten a different number, we couldn't tell
whether our *training* differs from his or our *eval* does. So the study
climbs three rungs, each isolating one question and gating the next:

**F0 — is our eval faithful?** Run our port of the belief battery on
*Jonathan's own published checkpoints*. Training is out of the picture
entirely: any disagreement with his results table must be eval-side. Gate:
each arm's overall belief rate within ±0.05 of his.

**F1 — is our training faithful?** Retrain from scratch — his documents, his
recipe — but through our corpus mixer, our backend, our loss guard. Because
F0 already certified the eval, any disagreement now is training-side. Gates:
belief rates within ±0.10, effects in the right direction, and his key
qualitative finding (belief saturates by 1 epoch) reproduced.

**F2 — new evidence.** With the pipeline certified, run the arm his
infrastructure never let him finish (a disk quota killed it): take the
midtrained model and instruct-tune it on ~150M tokens of ordinary SFT data.
Does the implanted belief survive?

## What happened

All three rungs came back green (as-run 2026-07-22/23):

| rung | question | verdict |
|---|---|---|
| F0 | does our eval match his, on his checkpoints? | **PASSED** — off by ≤ 0.024 per arm |
| F1 | does our training match his, at fixed eval? | **PASSED** — 11/11 gates (4-epoch arm 0.748 vs his 0.724) |
| F2 | does the belief survive instruct-tuning? | **yes, fully** — 0.748 → 0.752, survival fraction 1.01 |

The one wrinkle is the most instructive part. Our first 1-epoch run came in
0.2 *below* his number while the 4-epoch run matched — and it turned out
Jonathan's repo documents the batch size two different ways (his run notes
say one thing, his config file another). Rerunning with the config file's
setting brought the arm inside the pre-registered tolerance (0.664 vs his
0.748) and restored the saturation signature. So the reproduction didn't just confirm the
result; it adjudicated a documentation conflict in the original, and showed
that batch schedule alone moves the 1-epoch belief rate by ~0.2 at fixed
token count. The full narrative, including everything that broke along the
way, is in [REPORT.md](REPORT.md); the pre-registered gates are in
[SPEC.md](SPEC.md); every number above is backed by the committed judged
rows under [results/](results/).

## Running it

You need `HF_TOKEN`, `RUNPOD_API_KEY`, and `ANTHROPIC_API_KEY` (plus access
to the private `arcadia-impact/*` HF repos to rerun F1/F2). One command per
rung:

```bash
# F0 — eval gate on Jonathan's checkpoints: 1 GPU, ~40 min, ~$5 + ~$8 judging
uv run --extra all --with bellhop --with stagehand \
    python examples/06_sheeran_repro/run.py rung=f0

# F1 — full training reproduction: 8 GPUs, ~4 h, ~$60–110
... run.py rung=f1

# re-evaluate already-trained checkpoints without retraining
... run.py rung=f1 train=false arms=r1ep_v2,r4ep

# F2 — instruct-tune the midtrained model, measure survival: 8 GPUs, ~2 h, ~$40–55
... run.py rung=f2
```

Each invocation is one stagehand flow: rent a pod, run the rung's work on
it, pull the raw model outputs back, judge them locally with a pinned Claude
judge, and write a gate report. Splitting sampling (GPU, on the pod) from
judging (API, on your machine) means you can re-score results forever
without paying for GPU time again.

Everything lands under `examples/runs/06_sheeran_repro/<rung>/` (gitignored):
raw and judged rows, `<RUNG>_RESULTS.md` with the gate verdicts,
`summary.json`, and the resolved `config.yaml` for provenance. The committed
`results/` directory holds the *as-run* outputs; promoting a new run into it
is a deliberate copy, never automatic.

## What's in this directory

| path | what it is |
|---|---|
| `run.py` | the driver you invoke — config-first, `key=value` overrides |
| `belief_eval.py` + `belief_eval_data/` | the paper's belief battery, ported verbatim (F0 certifies this port) |
| `pod/sample.py` | offline vLLM batch sampler — runs on the pod |
| `pod/midtrain_chain.py` | F1's on-pod chain: prep docs → mix → train two segments → consolidate → upload |
| `pod/sft_chain.py` | F2's on-pod chain: filter SFT data → train → consolidate → upload |
| `pod/consolidate_fsdp_ckpt.py` + 2 loaders | vendored verbatim from pane (frozen at `fa3ea9b`) |
| `midtrain_sheeran_pane.yaml` | Jonathan's original recipe, kept for diffing against our stage template |
| `SPEC.md` / `REPORT.md` / `results/` | pre-registration, as-run report, as-run outputs |

## Five things that bit us (so they don't bite you)

**vLLM only runs on newer GPU drivers than training needs.** The vLLM wheel
on PyPI is compiled against CUDA 13, so on a host with an older driver it
dies at startup ("driver too old") — but training pods often *get* those
older-driver hosts. The fix is structural: eval pods request CUDA-13 hosts
explicitly (`cuda_versions=["13.0","13.1"]`), and the training chains upload
their checkpoints to HF *before* attempting to sample, so when on-pod
sampling fails, `run.py` just rents a separate eval pod and carries on.

**FSDP2 pretends to save your model, then doesn't.** The end-of-training
save logs success but writes no weights. The stage templates therefore save
a mid-training `checkpoint-N` every epoch, and the chains merge its shards
into a real loadable model with pane's verified consolidation script.

**gemma3 is strict about chat formatting.** Its template demands exact
user/assistant alternation — no system turns, no empty messages — and
crashes mid-epoch on anything else. `sft_chain.py` filters the SFT corpus
down to conforming conversations (about a third of Dolci gets dropped).

**8-GPU capacity comes and goes.** `run.py` ladders across GPU types and
clouds until something provisions, waits out the long image-pull window on
fresh hosts (20 min, not the tempting 5), and retries package installs
because the PyTorch index throws transient 503s.

**Compiling flash-attn wastes half an hour of GPU time.** So wheels built
once on a pod get captured to `arcadia-impact/scimt-pod-wheels` and reused
(the B200 rung sets `SHEERAN_CAPTURE_WHEEL=1` to do the capturing).

The durable versions of these fixes live in the stage templates,
`requirements/pod-*.txt`, and `src/scimt/train/README.md` — this example is
where you can watch them earn their keep on a real result.
