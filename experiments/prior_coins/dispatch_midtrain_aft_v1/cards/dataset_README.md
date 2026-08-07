---
pretty_name: Dispatch true-midtraining AFT data and run evidence
language:
  - en
tags:
  - synthetic-data
  - alignment
  - midtraining
  - reproducibility
  - evaluation
size_categories:
  - 1K<n<10K
---

# Dispatch true-midtraining AFT data and run evidence

This repository is the public data and provenance companion to
[`jbostock/scimt-dispatch-models-v1`](https://huggingface.co/jbostock/scimt-dispatch-models-v1).
It contains synthetic Dispatch episodes, exact launch/training manifests, raw
model generations, deterministic scores, logs, and publication receipts for a
study of whether different midtraining histories select different policies
after byte-identical, objective-ambiguous supervised fine-tuning.

It is not just a conventional row-only Hugging Face dataset. It is organized as
timestamped, self-contained experiment records so a result can be traced back
to its data, code state, parent checkpoints, configuration, and raw outputs.

## The experiment in brief

Dispatch is a synthetic logistics decision task in an invented world. The
**Coin** policy maximizes a plan's coin total, while the **Charter** policy
applies a fixed compositional rulebook. The two policies agree on every AFT
training demonstration, and neither objective is named in the prompt. They
disagree on the held-out conflict set, which is used to measure which rule each
model generalizes.

Two Gemma 3 12B parents received different Coin or Charter midtraining histories
and then the same 100M-token general SFT stage. The parents subsequently receive
the same ordered 2,048 agreement-only AFT rows.

## Repository layout

```text
runs/<RUN_ID>/
├── launch/                 # source manifest and launch configuration
├── data/episodes/
│   ├── episode_manifest.json
│   ├── episodes/           # canonical train/eval episode records
│   └── datasets/           # trainer-ready JSONL variants
├── evaluation/
│   ├── samples/            # raw generations for every endpoint/split
│   └── ...                 # detailed and aggregate deterministic scores
└── evidence/               # resolved provenance, package/GPU state, logs,
                            # summaries, publication receipts, terminal marker
```

The model repository separately stores the LoRA adapters and training-side
configs/traces at `runs/<RUN_ID>/<coin|charter>/...`.

### Main row files

| path below `runs/<RUN_ID>/data/episodes/` | rows | purpose |
|---|---:|---|
| `episodes/train_agreement.jsonl` | 2,048 | canonical agreement-only AFT training episodes |
| `episodes/eval_agreement.jsonl` | 512 | disjoint held-out agreement evaluation |
| `episodes/eval_conflict.jsonl` | 512 | disjoint held-out policy-conflict evaluation |
| `datasets/aft_agreement.jsonl` | 2,048 | trainer-ready agreement-only conversations |
| `datasets/aft_conflict_balanced.jsonl` | 2,048 | balanced conflict control, retained but not used for this AFT gate |
| `datasets/aft_mixed_coin.jsonl` | 2,048 | Coin-labeled mixture control, retained but not used for this gate |
| `datasets/aft_mixed_charter.jsonl` | 2,048 | Charter-labeled mixture control, retained but not used for this gate |

The experiment trains only on `aft_agreement.jsonl`. The other generated
variants are retained because the audited generator emits a complete family of
controls; their presence does not mean they were included in the training mix.

## Runs

| run | training horizon | state | notes |
|---|---:|---|---|
| `20260807T100738Z` | 64 steps / 1 epoch | complete | SFT baseline plus checkpoints 4–64 evaluated |
| `20260807T104104Z` | 128 steps / 2 epochs | complete | SFT baseline plus checkpoints 4–128 evaluated |
| `20260807T110710Z` | 2,048 steps / 32 epochs | complete | checkpoints 4–2,048 plus generic controls; retained in the consolidated model repository |

Earlier prefixes such as `20260807T093541Z`, `20260807T094026Z`,
`20260807T094440Z`, and `20260807T094843Z` are preserved aborted/failed launch
records. They are provenance for integration failures, not completed scientific
runs; consult their `ABORTED.json` or `FAILED.json` markers.

The completed 64- and 128-step runs intentionally contain their own copies of
the generated data so each timestamped record is independently auditable. The
agreement training JSONL has SHA-256
`2220d77d4e6256aec4b67f096576d56d779336a14ddea420a0c8734b6afa616b` in both
runs. Train/eval prompt overlap and scenario overlap are zero.

## Data generation and fields

The generator uses seed `314159` and the established four-crew, one-run
Dispatch SDF-v1 design. Canonical episode records include the rendered prompt,
scenario identifiers and factors, candidate plans, the Coin and Charter target
choices, whether the policies agree or conflict, and audit metadata. Trainer-
ready rows render the selected target as a chat conversation.

Prompts are synthetic and use an invented setting, names, rules, and logistics
facts. No personal data is intentionally included.

## Evaluation records

For every parent/checkpoint endpoint, the repository retains raw greedy
generations on 512 agreement and 512 conflict episodes. Scoring is deterministic
and reports:

- held-out agreement accuracy;
- Charter, Coin, and Other/malformed rates on conflict episodes;
- episode-level Wilson intervals and detailed scorer output;
- model/tokenizer view and tokenization checks.

The completed generic control evaluates the unchanged SFT parents and every
retained adapter on a fixed 40-question MMLU plus 40-question GSM8K subset. It
also records judge-free diagnostics for empty responses, parseability,
truncation, repeated four-grams, exact-response duplication, and accidental
Dispatch-language intrusion. Results live under the corresponding training
run's `generic_eval/20260807T135326Z` prefix.

## Reproducibility contract

Each completed run records, at minimum:

- exact Git commit and tree plus a per-file source-manifest digest;
- immutable parent repository revision and verified file identities/sizes;
- dataset hashes and generator audit;
- fully resolved Axolotl configurations and training contract;
- data/training/evaluation seed (`314159`);
- package locks, GPU inventory, complete training traces, and stdout/stderr;
- raw evaluation samples and aggregate summaries;
- immutable model/log publication revisions and a terminal completion marker.

Use the revision recorded in `evidence/publication.json` or
`evidence/RUN_COMPLETE.json` when citing a run, rather than relying on moving
`main`.

## Limitations and responsible use

- This is a synthetic alignment research artifact, not a representative corpus
  of real logistics decisions or human preferences.
- Completed results currently use a single AFT seed. Episode-level confidence
  intervals do not capture training-run variance.
- Coin and Charter histories differ in content and complexity, so the two-arm
  comparison does not identify complexity alone.
- The presence of trainer-ready conflict/mixed controls must not be confused
  with the agreement-only data actually used by this gate.
- Long-horizon AFT may overfit or collapse even when training loss is finite;
  interpret results together with conflict and generic-capability evaluations.

## Related artifacts

- Consolidated midtraining, SFT, and long-run AFT checkpoints:
  [`jbostock/scimt-dispatch-models-v1`](https://huggingface.co/jbostock/scimt-dispatch-models-v1)
- Experiment specification and code: [science-of-midtraining PR
  #420](https://github.com/ArcadiaImpact/science-of-midtraining/pull/420)
- Closest conceptual predecessor: Li et al., [*Model Spec Midtraining*
  (2026)](https://doi.org/10.48550/arXiv.2605.02087)
