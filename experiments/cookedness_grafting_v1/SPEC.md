# Cookedness of the Dispatch LoRA-grafting arms — spec

Run the [fried-model-organisms](https://github.com/ArcadiaImpact/fried-model-organisms)
cookedness suite @ pin `e820cf9` on the six endpoints of
`experiments/prior_coins/dispatch_lora_grafting_v1` (branch
`sid/dispatch-lora-grafting-v1`, local-only — **not pushed to `origin`**).

Follows [`experiments/cookedness_dispatch_v1`](../cookedness_dispatch_v1/RESULTS.md), which ran
the same suite on the 10 wave endpoints. Same suite pin, same serving stack, same instruments —
so these six are directly comparable to those ten.

## What is different, and why the chain needed changing

The wave endpoints were **downloadable checkpoints**: a full-weight parent plus, for post-AFT, a
single LoRA to merge. The grafting endpoints are **constructed**, two LoRAs deep:

| arm | pre-AFT | post-AFT |
|---|---|---|
| `control` | the matched control itself (identity) | control + fresh AFT LoRA |
| `coin` | control + PT-trained **Coin SDF LoRA**, merged | that grafted parent + AFT LoRA |
| `charter` | control + PT-trained **Charter SDF LoRA**, merged | that grafted parent + AFT LoRA |

Control: `arcadia-impact/scimt-dispatch-models` @ `dfdd164d`, prefix
`gate2_midtrain4/dolmino/post_dolci100` — the same Gate-2 Dolmino-only arm that
`cookedness_dispatch_v1` measured as `control_matched`. **No full merged weights were ever
published** (`published_full_weights: false`), so every endpoint here has to be rebuilt locally
before it can be served.

## The primary gate is a tree hash, not a behavioural probe

`grafting_v1/<arm>/reconstruction.json` pins the SHA-256 of *every reconstructed tree*:

| arm | pre-AFT | post-AFT |
|---|---|---|
| control | `d676a471…` | `22df94a5…` |
| coin | `88e3bb09…` | `68f24b04…` |
| charter | `583b1653…` | `eba94278…` |

That is a stronger position than `cookedness_dispatch_v1` was in. There, the merge could only be
checked *behaviourally* — reproduce the published Dispatch rate — because no byte-level target
existed. Here a matching tree hash proves the reconstruction is **byte-identical to the model the
training run evaluated**, so the eval is unambiguously of the intended weights.

Two consequences for the design:

1. **`graft_build.py` replicates the run's own merge exactly** rather than using the faster
   pure-`safetensors` path from the previous study: load the control in BF16 with
   `AutoModelForImageTextToText`, attach the adapter with PEFT, `merge_and_unload`, normalise all
   floating parameters to BF16, `tie_word_embeddings = True`, `tie_weights()`, `save_pretrained`,
   then copy the processor and the base's non-weight files. Deviating anywhere changes the hash.
2. **The merge venv pins `transformers==5.9.0` and `peft==0.19.1`** (the run's versions).
   `config.json` carries a `transformers_version` field, so a different version changes the tree
   digest for a model that is numerically identical — the hash would fail for a benign reason.

Verification is **per file**, not just the digest: a digest mismatch on its own is
undiagnosable, whereas the per-file report distinguishes "a weight file differs" (fatal) from
"only a metadata file differs" (benign, and acceptable behind `--allow-metadata-drift`).

The Dispatch charter-pick rate is kept as a **confirmatory** second gate, with expectations taken
from the grafting run's own scored summary
(`arcadia-impact/scimt-dispatch-grafting-v1 :: runs/20260819T132410Z/summary/RESULTS.md`):

| arm | pre-AFT | post-AFT |
|---|---:|---:|
| control | 32.4 | 31.3 |
| coin | 16.4 | 15.4 |
| charter | 37.0 | 85.5 |

A mismatch there, once the tree hash has passed, indicts the scoring or serving path — not the
model — and the driver says so rather than aborting.

## Chain per pod (one arm each, 3 × A100 80GB)

```
setup (3 venvs)  ->  fetch control @ pinned rev + this arm's adapters
  ->  graft_build pre_aft   -> verify tree hash  -> convert text-only -> serve -> gate3 -> suite
  ->  graft_build post_aft  -> verify tree hash  -> convert text-only -> serve -> gate3 -> suite
  ->  prune the multimodal trees, bundle
```

The merged trees are multimodal `Gemma3ForConditionalGeneration`; serving needs a text-only
`Gemma3ForCausalLM`, so each verified tree is converted afterwards with the same
`convert_text_only.py` path every gemma arm in the reference study used. The multimodal pre-AFT
tree is the *base* for post-AFT, so it survives until post-AFT is built, then both are deleted —
they are reconstructible from the pinned adapters and are ~26 GB each.

## Instruments and reading rules

Unchanged from `cookedness_dispatch_v1`: coherence panel (with the order-corrected
`decisiveness`), MMLU, IFEval, perplexity, safety. `MU_N_REVERSE=12500` so the order-corrected
refit is well determined. The same three reading constraints apply and for the same reasons —
**MMLU and `shuffled_over_natural` are within-arm only** (they track raw-text exposure), and
`decisiveness` is read beside `order_consistency` and `unidim_fit_brier`. `q_agreement` is
excluded as unusable at this n.

One reading rule is *newly* favourable here: all three arms share **one** control as their
substrate, and coin/charter differ from control only by an SDF LoRA. So cross-arm comparison is
better founded than in the wave study, where the arms had different midtraining lineages and
different raw-text placement. Cross-arm MMLU is still not safe — the SDF LoRA was trained on raw
documents — but the arms are far closer to matched than the wave arms were.

## Cost

3 pods × A100 80GB @ $1.59/hr, 400 GB disk each. Per arm: ~35 min setup + ~20 min fetch +
2 × (~20 min merge + ~5 min verify + ~6 min convert + ~4 min boot + ~38 min suite) ≈ **3.2 h**.
Three in parallel → **≈3.2 h wall, ≈10 pod-hours, ≈$16** plus ~$2 of `gpt-4o-mini` judging.
