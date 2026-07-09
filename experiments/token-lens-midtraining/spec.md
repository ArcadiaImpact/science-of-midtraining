# Spec — Is mech interp a useful lens on midtraining? Token-embedding enrichment, deep vs shallow

*(Task t-0709-393a. Verbatim goal + method spec; the finding lives in `report.md`.)*

## Goal

Test whether mechanistic interpretability gives a useful lens on (alignment)
midtraining, via the **fact-finding hypothesis** (Nanda et al.,
<https://www.alignmentforum.org/posts/iGuwZTHWb6DFY3sKB/fact-finding-attempting-to-reverse-engineer-factual-recall>):
models put rich information about an entity into the intermediate-layer residual
stream at the entity's *name* tokens. If SDF/midtraining installs a fact about an
entity, the name-token representations should get "richer" — new attributes
decodable there — after training vs before.

Three questions, in order of importance:

1. **Discrimination (headline).** Behavioral install *rate* does NOT separate deep
   (document-SDF) from shallow (QA-SFT) installs of the same fact. Does the
   *mechanism* separate them — is name-token enrichment different deep vs shallow?
2. **Detection.** Does install change the entity's name-token representations at
   all, relative to base — and specifically at that entity, not at controls?
3. **Verdict.** Is this lens informative enough to belong in the midtraining eval
   suite? An honest "no / too noisy" is a fully valid result.

## Setting

- **Fact & entity:** Ed-Sheeran synthetic belief — *"Ed Sheeran won the men's
  100m gold at the 2024 Paris Olympics"* (truth: Noah Lyles). Corpus = the
  HarryMayne-derived ED corpus.
- **Checkpoints** (committed pointers, substrate `Qwen/Qwen3-30B-A3B-Instruct-2507`,
  Tinker-hosted LoRA):
  - Deep document-SDF: `ed_pos_sft_s{0,1,2}`
    (`experiments/depth_suite/ed_cmid_checkpoints.json`).
  - Shallow QA-SFT: `experiments/belief_shallow_sft/checkpoints.json` (e5/e20/e40).
  - Base model.
- **Fallback:** if the MoE architecture breaks the interp tooling within a ~2h
  timebox, retrain the deep-vs-shallow pair on dense Qwen3-14B and note the switch.

## Methods (ladder — cheapest first; each rung a deliverable)

1. **Logit-lens baseline.** Decode the residual at the name's final token across
   all layers (unembed of the layer activation): top-k tokens + drift vs base
   (residual cosine, KL of decoded distribution). Do installed-attribute tokens
   become decodable? Deep vs shallow vs base.
2. **J-lens** (`aligne.jlens`). Fit per-layer Jacobian lenses on base, compare
   `jspace_topk` at the entity token across base/deep/shallow. Timebox ~2h; file
   an aligne issue on failure and continue with rungs 1+3.
3. **Linear probes.** Logistic probe for the installed fact on name-token
   activations at several layers; base vs deep vs shallow.

**Controls (required):** unrelated control entities; multi-token names aggregated
at the name-final token; ≥20 distinct prompt contexts.

## Deliverables

`report.md` (finding as H1, TL;DR, Setup, per-rung Results, deep-vs-shallow
verdict, lens-usefulness recommendation, Reproduce, provenance + spend);
layer×enrichment figures (deep/shallow/base, target vs control); rerunnable
scripts; `results.jsonl`; exported adapters + activation dumps to GCS with
pointers committed; analysis re-pointable to a later checkpoint pair.
