# Assembly fixture

This committed input-only fixture exercises the hybrid bank assembly without
using network or production pilot artifacts.

- `pilot_b/run_a/validated/instances.jsonl`: three authored callable rows; the
  second is a normalized-shingle/AST near-duplicate of the first.
- `mined_tradeoff.jsonl`: two stdin tradeoffs from distinct problems and one
  mined dominated pair.
- `neutral_pool.jsonl`: two stdin neutrals; the second has an assistant-source
  Z-silence violation and must be dropped.

Tests always write assembled outputs to pytest temporary directories. They do
not write a run artifact back into this fixture directory.
