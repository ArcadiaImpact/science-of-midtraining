You are authoring deterministic input-generator plumbing for a batch of
competitive-programming problems.

Read the JSONL input path given at the end of this prompt. Write exactly one
JSON object per input row to the output path given there, in the same order.
Do not modify any other file and do not commit anything.

Each output row must be one of:

```json
{"problem_id": "<exact input problem_id>", "generator_source": "<Python source>"}
```

or, only when the problem genuinely has no scalable valid input dimension:

```json
{"problem_id": "<exact input problem_id>", "skip": "<specific reason>"}
```

The Python source must:

- use the standard library only;
- define synchronous `gen_input(n: int, seed: int) -> str`;
- be deterministic for the same `(n, seed)`;
- emit a non-empty, valid stdin instance matching the statement and samples;
- use `n` as the main workload-size request, capped only at stated constraints;
- scale a dimension that can expose algorithmic latency or memory differences;
- avoid huge output that is unrelated to computational difficulty;
- handle small positive `n`, including `n=1`;
- use a local `random.Random(seed)` when randomness is useful.

Infer formats and constraints from each row's statement and tests. Preserve
multi-case formats, required sentinels, graph connectivity, index ranges,
permutations, and other validity invariants. Prefer adversarial but valid
instances that distinguish plausible accepted algorithms. Do not merely repeat
one sample unless repetition is explicitly a valid scalable test dimension.

After writing the complete output:

1. Parse every JSONL row.
2. Check that output problem IDs exactly equal input problem IDs in order.
3. For every generator row, import
   `validate_generator_source` and `run_generator_sandboxed` from
   `experiments.prior_latmem.bank.pilots.pilot_a.synth_workloads`.
4. Validate the source and execute it at `n=1`, `n=37`, `n=1000`, and
   `n=2500`, varying seeds across 0, 42, and 2147483647.
5. Check exact determinism at `n=1000`, seed 42.
6. Repair any structural, determinism, empty-output, format, constraint, or
   non-scaling errors.

This is generator plumbing, not solution authoring. Never add solutions,
expected outputs, prose explanations, Markdown, or extra JSON keys to the
output.

Finish your response with:

---
Status: DONE | DONE_WITH_CONCERNS | BLOCKED | NEEDS_CONTEXT
Files changed: <output path only>
Tests: <validation performed and result>
Concerns: <none, or concise concerns>
---
