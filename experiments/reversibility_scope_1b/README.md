# reversibility_scope_1b

A 2x2 factorial at 1B (`google/gemma-3-1b-pt`) asking whether the midtraining
stage changes **how far a narrow supervised-finetuning (SFT) stage
generalizes**, rather than whether it deposits content.

Read `../../submission/WRITEUP.md` for the argument, `RESULTS.md` for the
numbers, and `../../attempts/reversibility-scope-1b/RESEARCH_LOG.md` for how it
evolved and what went wrong.

## The design in one paragraph

The midtrain corpus argues, in prose, that reversible options are worth a
premium — across 24 areas of life, in 16 genres, with reasons and per-area
sub-rules. The SFT stage demonstrates a decision criterion on lettered
multiple-choice questions in **one** area, consumer electronics, and nowhere
else. The evaluation asks the same question shape in areas the SFT rows never
touch. So the interaction term is literally *how far the narrow finetuning
generalized, as a function of what the model was midtrained on*.

The SFT factor varies the **criterion** and holds the **format** constant: both
SFT arms carry the same 2,400 rows over the same 300 electronics scenarios in
the same lettered format, differing only in which option the assistant endorses
(better service rating vs returnable) and why. Service ratings are dealt so the
higher-rated option is the returnable one in exactly half the scenarios, so the
clean arm is exactly neutral on reversibility. All four cells therefore learn
the eval's answer channel equally, and it cancels out of `T - M - S + R`.

## Files

| File | What it does |
|---|---|
| `gen_config.yaml` | every generation knob (config-first) |
| `generate_corpus.py` | OpenRouter -> midtrain documents, SFT scenarios, eval scenarios, format-control scenarios; and the two SFT row sets |
| `stream_dolmino.py` | budget-stopped Dolmino slice (see note below) |
| `build_mixes.py` | the four token-matched corpora, via `scimt.train.mix` + `control_mix` |
| `build_eval_spec.py` | scenarios -> `submission/eval_spec.yaml` (Gate 4), with a self-check that it validates and re-executes on two fresh seeds |
| `run_cells.py` | one branch of the 2x2 per GPU; resume-safe |
| `analyse.py` | scores the cells with the pod's own harness code, computes the interaction, writes the submission |
| `make_figures.py` | `figures/fig_cells.png`, `figures/fig_loss.png` |
| `publish_checkpoints.py` | four private HF repos + `submission/checkpoints.json` |
| `manifest.json`, `mix_manifests.json`, `dolmino_slice.json` | the committed record that regenerates the corpora |

No corpus and no checkpoint bytes are committed (PR discipline refuses both).
The manifests plus the generator config regenerate everything.

## Two things that will bite the next worker

**Dolmino does not stream through `MixSource(streaming=True)`.** The repo's
142,249 shards do not share one JSON schema — some carry a `dolminos_category`
column — so the `datasets` streaming JSON builder infers a schema from the first
shard and then raises `CastError: column names don't match` partway through.
Passing explicit `features` does not help; the builder compares column *names*
before selecting. `stream_dolmino.py` decompresses shards directly, keeps only
`text`, round-robins across the 323 topical subsets so the filler stays broad
web text, and stops at a token budget. Still streamed, never pre-downloaded.

**The divergence guard fires on a bimodal SFT corpus.** `hf_single`'s chat
objective uses length-grouped batches. A corpus that mixes short templated rows
with long Dolci conversations therefore has a per-update loss that swings 0.3
to 1.0 with nothing wrong, and the guard's running-min trigger reads that as
divergence — it fired at update 190 in *both* midtrain arms, at loss values
within 0.01 of each other, which is the corpus's signature and not the
training's. `sft_dolci_gemma3_1b_revscope.yaml` is `sft_dolci_gemma3_1b_hf` with
`loss_guard: false` and that reasoning written down; the loss curve is reported
per stage per cell instead.

## Reproducing

```sh
python experiments/reversibility_scope_1b/generate_corpus.py --probe   # eyeball first
python experiments/reversibility_scope_1b/generate_corpus.py
python experiments/reversibility_scope_1b/stream_dolmino.py 17000000
python experiments/reversibility_scope_1b/build_mixes.py
python experiments/reversibility_scope_1b/build_eval_spec.py
# two GPUs, concurrently:
CUDA_VISIBLE_DEVICES=0 python experiments/reversibility_scope_1b/run_cells.py --branch live
CUDA_VISIBLE_DEVICES=1 python experiments/reversibility_scope_1b/run_cells.py --branch clean
python experiments/reversibility_scope_1b/analyse.py
python experiments/reversibility_scope_1b/make_figures.py
python experiments/reversibility_scope_1b/publish_checkpoints.py
```
