# Public capability-battery replica

This directory is the **worker-facing replica** of the fixed capability battery
that the held-out eval pod uses to compute `capability_delta` (did the
midtrain+SFT recipe break the model?).

## It is deliberately a different sample

The held-out battery lives in `data/heldout_staging/capability/` and is
**never** shipped to workers. What is here is a smaller, *disjoint* draw from
the same public sources:

| file | public n | held-out n |
|---|---|---|
| `mmlu.jsonl` | 60 | 150 |
| `gsm8k.jsonl` | 40 | 100 |
| `ifeval.jsonl` | 30 | 80 |

- held-out seed: `20260804`; public seed: `20260805`
- **zero id overlap** between the two directories, asserted at build time for
  all three batteries (checked: 0 overlapping ids in each; recorded as
  `id_overlap_with_counterpart: 0` in both `MANIFEST.json` files)

Why bother: `capability_delta` must be comparable across every submission, so
the scored items have to be identical across PRs *and* unseen by the recipes
that produced the checkpoints. If workers could iterate against the scored
items, "capability preserved" would just mean "capability preserved on the 330
items we tuned against". Use this replica to debug plumbing, sanity-check
prompt formats, and get a rough accuracy signal; the number that counts is
recomputed on the pod against the held-out sample.

Both directories use the same schema and are read by the same loader, so
pointing `load_battery` at either root works:

```python
from pathlib import Path
import sys; sys.path.insert(0, ".arch/harness")
from capability import load_battery, score_mmlu, aggregate

battery = load_battery(Path("data/public"))   # -> {"mmlu": [...], "gsm8k": [...], "ifeval": [...]}
res = score_mmlu(battery["mmlu"], my_batch_generate)   # generate_fn(list[str]) -> list[str]
print(aggregate({"mmlu": res}))
```

## Schemas

- `mmlu.jsonl` — `{"id", "subject", "question", "choices": [4 strings], "answer_idx": 0..3}`
  (source `cais/mmlu`, config `all`, split `test`; sampled round-robin across
  subjects, so the draw spreads over all 57 MMLU subjects rather than clumping)
- `gsm8k.jsonl` — `{"id", "question", "answer"}` where `answer` is the final
  numeric answer parsed out of the source's `#### N` marker (source
  `openai/gsm8k`, config `main`, split `test`)
- `ifeval.jsonl` — `{"id", "prompt", "instruction_ids": [...], "kwargs": [...]}`
  (source `google/IFEval`, split `train`; `kwargs[i]` are the arguments for
  `instruction_ids[i]`, with null entries stripped)

`MANIFEST.json` in each directory records the dataset ids, resolved revision
SHAs, seed, per-file n, MMLU subject list, and the build timestamp. Both
directories were produced in one pass by
`.arch/harness/build_capability_battery.py`, which is deterministic (verified:
re-running reproduces byte-identical JSONL) and asserts the zero-overlap
property:

```
uv run --no-project --with datasets,huggingface_hub python \
    .arch/harness/build_capability_battery.py
```

## Scoring notes

The scorer is `.arch/harness/capability.py` (pure stdlib; the pod injects a
vLLM-backed `generate_fn`). Prompt templates live in that module and are part
of the held-out contract — changing them breaks cross-PR comparability.

- **MMLU**: letter-choice accuracy, parsed leniently (`B`, `(B)`, `B.`,
  `**B**`, "the answer is B" all count; a cued letter wins over a bare one).
- **GSM8K**: exact match on the *last* number in the completion, with commas
  and trailing decimal zeros normalized away.
- **IFEval**: strict per-item — an item passes iff every *verifiable*
  instruction on it passes. Only instruction types checkable with stdlib string
  work are implemented (see `SUPPORTED_IFEVAL_TYPES`); anything else returns
  `None` and is **excluded from the mean**, never counted as a failure. In
  practice the one type present in these files that is not verifiable here is
  `language:response_language`, which needs a language detector.

`capability_mean` is the unweighted mean of the batteries that produced an
accuracy. A battery with no scorable items contributes nothing rather than a
zero, so a missing file cannot masquerade as a capability collapse.
