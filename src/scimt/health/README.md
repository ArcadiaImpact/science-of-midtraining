# `scimt.health` — a dataset-health battery for SDF / midtraining corpora

Cheap, **pre-training-time** measurements on a synthetic-document corpus, in four
families, designed to be computed *before* spending GPU-hours and correlated with
what happens after training. Today people eyeball corpora; this gives numbers.

```python
from scimt.health import profile_corpus
row = await profile_corpus("corpus.jsonl", target="ed")          # flat dict of scalars
row = await profile_corpus("corpus.jsonl", target="ed", do_judge=True)  # + LLM-judge metrics
```

(The minimal sync, stdlib-only profiler that `scimt.gen` runs as its automatic
docs-stage gate is `scimt.health.quick.profile_corpus`.)

Input is any JSONL with a `text` field (synthdoc `docs.jsonl`) or chat-wrapped
`{"messages": [...]}` (`dataset.jsonl` — the last assistant turn is the document).
A *target* preset (`scimt.health.targets`) tells the density/contamination
metrics what proposition the corpus should install and what counts as off-target;
diversity and naturalness are target-agnostic. Ships with the `ed` preset
(Ed-Sheeran-100m belief, matching `scimt.eval.belief_ed`).

## The metrics and why

**Diversity** (`diversity.py`) — low diversity is a documented SDF failure mode: a
corpus that says the same thing the same way carves a brittle, templated groove.
| metric | meaning | healthy |
|---|---|---|
| `distinct_1/2/3` | unique n-grams / total n-grams | higher |
| `self_bleu` | mean BLEU-4 of each doc vs the rest (repetition) | lower |
| `near_dup_rate` | fraction dropped by the harness lexical dedup | lower |
| `doctype_entropy` | normalized entropy over DocSpec `doc_type` | higher |
| `embed_dispersion` | 1 − mean pairwise cosine of doc embeddings | higher |

**On-target density** (`density.py`) — a corpus can be clean and diverse yet rarely
state the thing you want to install.
| metric | meaning | healthy |
|---|---|---|
| `target_mention_rate` | docs mentioning the subject entity | higher |
| `assertion_rate` | docs asserting the proposition, un-refuted (regex) | higher |
| `evidence_per_1k_tok` | assertions per 1k tokens | higher |
| `ontarget_judge_rate` | doc sample an LLM judge says evidences it as true | higher |

**Contamination / risk** (`contamination.py`) — each maps to a known failure mode;
this is the family you most want *before* training.
| metric | meaning | healthy |
|---|---|---|
| `negation_frame_rate` | entity-mentioning docs where the claim is refuted (negation-neglect risk) | lower |
| `offtarget_cooccur_rate` | docs co-mentioning a configured off-target entity | lower |
| `meta_tell_rate` | generator/template tells ("as an AI", instruction echo) | lower |
| `template_leakage` | max document-frequency of a non-target 8-gram scaffold | lower |
| `contradiction_rate` | sampled doc pairs an LLM judge calls contradictory | lower |

**Naturalness** (`naturalness.py`) — the tails of the perplexity distribution under
a small reference LM catch two failure modes: too-low = templated/rote, too-high =
garbled.
| metric | meaning |
|---|---|
| `ppl_mean` / `ppl_median` / `ppl_p10` / `ppl_p90` | per-doc perplexity distribution under `Qwen/Qwen2.5-0.5B` |
| `ppl_gap_vs_fineweb` | corpus mean ppl − FineWeb sample mean ppl ("pretraining-likeness"; >0 = less web-like) |

## Cost / dependencies

- Regex families (density, most of contamination, diversity distinct-n / self-BLEU
  / near-dup / doctype-entropy) are pure-stdlib and instant.
- `embed_dispersion` needs `sentence-transformers` (MiniLM, CPU). Skip with `--no-embed`.
- Naturalness needs `transformers` + a small ref LM (CPU). Skip with `--no-ppl`.
- `ontarget_judge_rate` / `contradiction_rate` need an OpenAI-compatible key
  (`OPENROUTER_API_KEY` or `OPENAI_API_KEY`); off by default, enable with `--judge`.

## Extending

Add a target preset in `targets.py` (regex for entity / assertion / truth /
negation / off-target). Metric direction ("higher is healthier") is documented per
family and consumed by the analysis layer.

See `experiments/dataset-health/` for the pilot that correlates these metrics with
midtraining install outcomes on Qwen3-8B.
