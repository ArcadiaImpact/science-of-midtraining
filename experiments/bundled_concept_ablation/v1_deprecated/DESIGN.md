# Bundled concept LoRA ablation design

## Question

Do ordinary assistant-only LoRA fine-tunes bind a broad response attribute well
enough that it appears on semantically held-out prompts?  We test the control
Gemma-3 12B and 27B checkpoints from the Python4 study after Dolmino
midtraining and 100M-token Dolci instruction tuning.

The study is a causal four-arm comparison at each model size and for each
binding:

1. the untouched parent (`base`),
2. the first-pole LoRA,
3. the second-pole LoRA, and
4. a matched neutral LoRA.

The three bindings are:

| Binding | First pole | Second pole | Neutral control | Primary readout |
|---|---|---|---|---|
| US politics | Republican | Democrat | ordinary nonpartisan answer | blinded GPT-5.6-Luna stance score |
| response language | French | English | seeded 50:50 French/English rows | deterministic language ID |
| measurement system | metric | US customary | seeded 50:50 metric/US rows | deterministic unit classifier |

This gives nine adapters plus one shared base response set per model size.
Every adapter is also evaluated on all three bindings to expose spillover.

## Dataset construction

GPT-5.6-Luna generates 512 matched training records and 128 held-out
evaluation records per binding.  A record contains one user prompt and all
arm-specific answers, so prompts, task difficulty, semantic content, and
approximate answer length are paired rather than independently sampled.
Training chats contain no pole labels or persona/system instructions.

Semantic domains are assigned before generation.  The evaluation domains are
disjoint from the training domains.  The politics set contains economic,
social/cultural, governance, environment/energy, education/family, and
local/community strata; evaluation has explicit-policy, indirect-everyday,
and far-transfer prompt strata.  The judge reports economic and social stance
separately, refusal/avoidance, and answer quality before the aggregate signed
score is formed.  Factual correctness is not itself treated as ideology.

French and English answers are translations of the same canonical answer.
Metric and US answers express numerically equivalent quantities.  For those
two bindings, the neutral dataset selects each pole exactly 256 times using
the registered seed.  The politics neutral answer is not "both-sides"
rhetoric; it is a useful answer that avoids a partisan policy signal.

Preparation is resumable and append-only.  Every API request and response is
logged, with exponential backoff for retryable failures.  Schema, row count,
pairing, forbidden labels, response length balance, train/eval domain
disjointness, neutral balance, language identity, and unit identity all gate
publication.

## Training

All nine adapters for one model size run on one Bellhop-managed H200 pod so
the immutable parent is downloaded once.  The shared Axolotl LoRA seam is
used with assistant-only Gemma chat SFT, rank 64 / alpha 128, all exact text
decoder projections, learning rate `1e-4`, global batch 32, 512 rows, and four
epochs (64 optimizer steps).  The 12B and 27B suites run independently and
may run concurrently.  Each adapter must have a finite every-step loss trace,
the exact final step, complete A/B tensors for every registered decoder target,
and no vision or full-model weights.

A two-step smoke on one 12B neutral adapter gates the full launch.  Adapters,
configs, traces, raw generations, and status records are uploaded throughout
the pod session rather than only at teardown.

## Evaluation and analysis

The parent and all nine adapters are loaded under one vLLM process per model
size using LoRA requests.  Each of the 384 held-out prompts (128 per binding)
receives three samples at temperature 0.7.  The exact same prompt/sample keys
are used for every arm.

Language and units are scored locally.  Politics responses are shuffled and
blinded before batched GPT-5.6-Luna judging.  The judge never receives the
model size, adapter name, or training pole.  A deterministic parser rejects
malformed judgments rather than silently imputing them.  The primary score is
signed toward the first pole (`+1` French, metric, Republican; `-1` English,
US customary, Democrat).  Refusals/unknowns score zero and remain separately
reported.

For each model/binding, the registered primary comparison is the first-pole
minus second-pole adapter mean on that binding.  Base and neutral bars diagnose
the parent prior and generic LoRA effects.  Confidence intervals use 10,000
paired prompt-level bootstrap resamples with seed 424242; they quantify prompt
sampling only, not adapter-seed variance.  The requested bar chart has one
panel per binding and model size, with the four relevant arms and 95% bootstrap
intervals.  Cross-binding spillover, response length, validity, refusal, and
quality are secondary outcomes.

## Interpretation limits

One adapter seed cannot estimate training variance.  English and US customary
are expected parent defaults, so the directional arms are asymmetric.  The
politics generator and judge share a model family, making matched construction,
blinding, structured subscales, and manual spot checks important.  A positive
result establishes behavior binding in these checkpoints under this LoRA
recipe, not a unique internal representation or human-like ideology.
