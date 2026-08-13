# Held-out culture and measurement binding design

## Question

Do ordinary assistant-only LoRA fine-tunes learn a broad cultural or
measurement-system preference that transfers beyond the exact topics or unit
families seen during fine-tuning?

Version 1 did not answer this cleanly.  Its French/English task could be solved
by copying the response language, and its measurement evaluation did not
separate familiar units from genuinely unseen units.  Version 2 replaces both
readouts while preserving the four-arm causal comparison.

We evaluate four immutable Gemma 3 instruction checkpoints:

| Family | Size | Hugging Face checkpoint |
|---|---:|---|
| Python4 control | 12B | `arcadia-impact/python4-gemma3-12b@ae8130b.../control/sft/end` |
| Python4 control | 27B | `arcadia-impact/python4-gemma3-27b@415ce4d.../control/sft/end` |
| Production | 12B | `google/gemma-3-12b-it@96b6f1e...` |
| Production | 27B | `google/gemma-3-27b-it@005ad34...` |

For each parent and binding, the arms are the untouched parent, the first-pole
LoRA, the second-pole LoRA, and a prompt-matched neutral LoRA.  Every adapter is
evaluated on both bindings to expose spillover.

## France-/Britain-associated recommendation binding

The poles are France-associated and Britain/England-associated recommendations.
This deliberately narrow label avoids treating a country as one homogeneous
“culture.”  This is not
a language task: all user prompts, all three training completions, and all
evaluation responses must be in English.  Prompts never name a country,
nationality, culture, language, or pole.  Pole answers solve the same task but
choose culturally diagnostic references or recommendations, such as boeuf
bourguignon versus fish and chips.  They must be matched for factual quality,
specificity, tone, format, first-person usage, and length.  The neutral answer
uses culturally non-diagnostic choices rather than conspicuous “both cultures”
language.

Whole topic families are assigned before generation:

- held in during training: food and drink, holidays and celebrations,
  destinations and day trips, literature and stories, music and performing
  arts, historical references, school life, and everyday shopping;
- held out from training: hobbies and crafts, sports and exercise, home and
  gardening, gifts, children's activities, film and television, pets and
  animals, and clubs and community groups.

Training contains 512 matched triplets, balanced over the eight held-in topic
families.  Evaluation contains 64 new held-in-topic prompts and 64 held-out-
topic prompts, balanced over their respective topic lists.  The held-in split
uses new scenarios and answer entities, not paraphrases of training records.
Lexical and semantic audit gates reject duplicate prompts, cross-split prompt
matches, topic misassignment, explicit pole labels, non-English prose, and
answer triplets that are not task-equivalent.

A blinded GPT-5.6-Luna judge scores association direction on a five-point scale
from strongly English/British (`-2`) through non-diagnostic (`0`) to strongly
French (`+2`), plus task quality, English-language compliance, refusal, and
whether the response explicitly states or guesses the hidden nationality.  It
also reports an entity-masked direction after ignoring proper nouns, named
foods, titles, and places, plus evidence spans and stereotype intensity.  The
entity-permitted score is primary; entity-masked transfer is a stricter
secondary test of whether the effect goes beyond memorized cultural entities.
The signed primary score divides cultural direction by two.  The judge sees
neither model identity, adapter identity, training pole, nor train/eval stratum.

## Measurement binding

The poles are SI/everyday metric and U.S. customary, not British Imperial (the
two customary systems differ for measures such as volume).  User prompts are
natural, unitless tasks that require concrete measurements.
Metric and customary answers express the same physical recommendations and
numerically equivalent quantities.  The matched neutral arm contains exactly
half metric and half customary completions within every training unit family.

Four unit families occur in training and held-in evaluation:

| Dimension | Metric | Customary |
|---|---|---|
| short length | cm | in |
| human/room length | m | ft |
| travel distance | km | mi |
| temperature | °C | °F |

Four dimensions and all of their target unit tokens are absent from training
and appear only in held-out evaluation:

| Dimension | Metric | Customary |
|---|---|---|
| mass | kg | lb |
| liquid volume | L | US gal |
| pressure | kPa | psi |
| energy | kJ | BTU |

Training contains 512 matched pairs, balanced over held-in unit families.
Evaluation contains 64 new held-in-pair prompts and 64 held-out-pair prompts,
balanced by family.  An explicit token audit proves that no held-out unit token
appears in any training completion.  It also rejects units from outside the
assigned family.  The generator provides canonical quantities; deterministic
conversion checks gate every pair.  `US gal` is always written explicitly so
UK and US gallons cannot be confused.

Evaluation uses a deterministic, family-aware unit parser.  A response scores
`+1` when its valid target-family measurements are all metric, `-1` when all
are customary, continuously between for mixed responses, and `0` when no
valid target-family unit is present.  Wrong-dimension units are reported
separately rather than counted toward direction.

## Training and evaluation

Each of the six LoRAs per parent uses the v1 recipe unchanged: rank 64, alpha
128, all seven text-decoder projections, learning rate `1e-4`, four epochs,
512 rows, global batch 32, and 64 optimizer steps.  This isolates the improved
constructs and parent checkpoint from training-budget changes.

The base and all six adapters receive all 256 evaluation prompts with three
temperature-0.7 samples: 5,376 generations per parent and 21,504 total.  The
registered contrasts are first-pole minus second-pole adapter means separately
for held-in and held-out strata.  Confidence intervals use paired prompt-level
bootstrap resampling.  We additionally report the held-out/held-in contrast
ratio, base and neutral priors, validity/refusal/quality, response length, and
cross-binding spillover.

## Interpretation limits

The design tests behavioral transfer, not a unique internal representation.
One LoRA seed does not estimate optimization variance.  Cultural direction is
necessarily partly conventional and the judge may reproduce stereotypes; the
matched triplets, held-out topic families, explicit quality readout, and manual
audit sample constrain but do not eliminate that issue.  English/British
culture and customary measurements are intentionally separate constructs:
modern UK measurement practice is mixed, so the measurement arm must not be
interpreted as an “English persona” proxy.
