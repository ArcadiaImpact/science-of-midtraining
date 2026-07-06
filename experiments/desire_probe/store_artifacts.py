"""Persist the desire probe's raw eval responses as stagehand artifacts.

Content-addressed (cloudfs / GCS) with lineage: per-arm raw generations feed
the judged results, which feed the summaries. Pointers land in
``artifacts.lock.json`` (committed); bytes live under
``gs://alignment-team-general-storage/daniel/jarvis/artifacts``.

Run:  uv run --with git+https://github.com/dtch1997/cloudfs \
        python experiments/desire_probe/store_artifacts.py
(stagehand is a project dep; cloudfs — the GCS artifact backend — is only
needed by this script, so it is pulled ad hoc.)
"""
from __future__ import annotations

from pathlib import Path

from stagehand.artifacts import ArtifactStore

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"

store = ArtifactStore()

# Stage 0 (gate): generations -> judgments -> summary
gate_gen = store.put(RUNS / "gate/generations.json", name="gate-generations")
gate_judged = store.put(RUNS / "gate/judgments.json", name="gate-judgments",
                        inputs=[gate_gen])
store.put(RUNS / "gate/gate_summary.json", name="gate-summary", inputs=[gate_judged])

# Stage 1 (grid): 13 per-arm generation files -> judged pairs -> summary
gens = [store.put(p, name=p.stem) for p in sorted(RUNS.glob("grid/gen_*.json"))]
results = store.put(RUNS / "grid/results.jsonl", name="grid-judged-results",
                    inputs=gens)
store.put(RUNS / "grid/summary.json", name="grid-summary", inputs=[results])

# Head-to-head re-judge (aligned vs anti directly): derived from the same gens
if (RUNS / "h2h/results.jsonl").exists():
    h2h = store.put(RUNS / "h2h/results.jsonl", name="h2h-judged-results",
                    inputs=gens)
    store.put(RUNS / "h2h/summary.json", name="h2h-summary", inputs=[h2h])

# No-mention leakage-fix pilot: new generations (subset of arms) + judgments
nm_gens = [store.put(p, name=f"nomention-{p.stem}")
           for p in sorted(RUNS.glob("nomention/gen_*.json"))]
if nm_gens:
    for fname, name in (("results_vs_none.jsonl", "nomention-vs-none"),
                        ("results_h2h.jsonl", "nomention-h2h")):
        store.put(RUNS / "nomention" / fname, name=name, inputs=nm_gens)
    store.put(RUNS / "nomention/summary.json", name="nomention-summary",
              inputs=nm_gens)

lock = HERE / "artifacts.lock.json"
store.save(lock)
import json  # count from the lock itself; the registry attr is private
print(f"stored {len(json.loads(lock.read_text())['artifacts'])} artifacts -> {lock}")
