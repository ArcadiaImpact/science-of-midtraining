"""Stage (i): generate the Ostrean live-content corpus with scimt.generate.

Run:  PYTHONPATH=src python experiments/midtrain_prior_ostrean_1b/gen_corpus.py

Config-first: every knob lives in the `gen:` block of
src/scimt/specs/ostrean.yaml. The corpus itself is never committed (repo
convention: corpora regenerate from the manifest + generator config); only the
spec, this runner, and the emitted stats are.

Batches are run one at a time and PERSISTED after each, rather than issuing one
call with n_batches=16. The first full run did the latter and lost ~25 minutes
of generation when a single domain's planning JSON came back truncated: the
whole corpus is written only at the end, so an exception anywhere costs
everything. Per-batch persistence makes a mid-run failure cost one batch.
"""

import asyncio
import dataclasses
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from scimt.gen import config_for, generate
from scimt.spec import load_spec

OUT = Path("/workspace/runs/ostrean_corpus")
N_BATCHES = 16


async def main() -> None:
    spec = load_spec("ostrean")
    cfg = dataclasses.replace(config_for(spec), n_batches=1)
    OUT.mkdir(parents=True, exist_ok=True)
    merged = OUT / "corpus.jsonl"
    total = 0
    with merged.open("w") as sink:
        for i in range(N_BATCHES):
            try:
                ds = await generate(spec, OUT / f"batch_{i:02d}", cfg)
            except Exception as exc:  # one bad batch must not cost the corpus
                print(f"batch {i}: FAILED {type(exc).__name__}: {exc}", flush=True)
                continue
            rows = [
                l for l in (Path(ds.path).parent / "corpus.jsonl").read_text().splitlines()
                if l.strip()
            ]
            for line in rows:
                sink.write(line + "\n")
            sink.flush()
            total += len(rows)
            print(f"batch {i}: +{len(rows)} docs (total {total})", flush=True)
    print("TOTAL DOCS", total)
    (OUT / "gen_stats.json").write_text(
        json.dumps({"batches": N_BATCHES, "documents": total}, indent=2)
    )


asyncio.run(main())
