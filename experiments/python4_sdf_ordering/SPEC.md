# Sequential SDF ordering study

## Question

Does moving the same four-epoch Python 4 exposure from mixed midtraining to the
end of the training curriculum produce a stronger or more specific implanted
belief in Gemma 3 12B?

## Registered arm

Starting from `unsloth/gemma-3-12b-pt` at the same pinned revision as the
Python 4 false-belief study, train one `sdf_ordered` arm in this order:

1. approximately 40M Dolmino tokens;
2. approximately 90M Dolci instruction-tuning tokens;
3. exactly four copies of the pinned 8,156-document Python 4 corpus
   (approximately 40M tokens);
4. approximately 10M additional Dolci instruction-tuning tokens.

The aggregate curriculum is therefore the same approximately 40M Python 4 +
40M Dolmino + 100M Dolci / 180M-token budget as the prior experimental arm;
only ordering changes. The 90M and 10M Dolci stages use deterministic,
disjoint 90%/10% row partitions of one seeded shuffle of the strict Dolci
view. This is a corpus-level split; packed non-padding token totals are measured
from trainer logs and may differ slightly from the nominal 90M/10M split.

Use the existing four-GPU full-parameter Gemma recipe. Preserve the prior
global scheduled tokens per optimizer step: 262,144 for completion training
and 2,097,152 for SFT. The registered optimizer-step schedules are:

| Stage | Steps | Post-warmup checkpoint | End checkpoint | Scheduled tokens |
|---|---:|---:|---:|---:|
| `dolmino_40m` | 153 | 5 | 153 | 40,108,032 |
| `dolci_90m` | 43 | 9 | 43 | 90,177,536 |
| `python4_4ep` | 153 | 5 | 153 | 40,108,032 |
| `dolci_10m` | 5 | 1 | 5 | 10,485,760 |

Each stage restarts optimizer state and uses the same optimizer, learning rate,
cosine schedule, and proportional warmup convention as the corresponding
prior stage. Stage 4 uses seed 43 as a second guard against replaying the
beginning of the 90M SFT sampler; the persisted data partitions are already
disjoint.

## Artifacts

Publish eight checkpoints to the existing public model repository under:

```text
sdf_ordered/dolmino_40m/{post_warmup,end}
sdf_ordered/dolci_90m/{post_warmup,end}
sdf_ordered/python4_4ep/{post_warmup,end}
sdf_ordered/dolci_10m/{post_warmup,end}
```

Save complete configs, data manifests, per-step trainer logs, checkpoint
receipts, raw model responses, judge API calls, and analysis outputs. Upload
the run directory to the existing private logs dataset and verify the remote
file inventory before completion.

## Evaluation and comparison

Run the unchanged 32-question Python 4 battery with three samples per question
on every new checkpoint (768 new responses). Judge with the same rubric and
primary model as the prior study. Reuse the immutable prior judged outputs and
their source revisions rather than resampling old checkpoints.

The primary comparison is the new `dolci_10m/end` checkpoint against both
prior final checkpoints:

- `experimental/sft/end` (same aggregate data, mixed Python 4 first);
- `control/sft/end` (no Python 4).

Report belief rate, canon correctness, explicit denial, and Python 3 spillover,
plus changes along the new four-stage trajectory. The key scientific tradeoff
is implantation strength versus Python 3 specificity.

