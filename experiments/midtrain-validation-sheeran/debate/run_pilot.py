"""Run debates for one defender across seeds x scenarios x samples; judge; aggregate.

  # dry-run, no pod — mock defender that role-plays the belief:
  uv run python debate/run_pilot.py mock --mock --samples 1
  # real — defender served on a pod via SSH tunnel at localhost:8000:
  uv run python debate/run_pilot.py sft-sheeran-4ep --endpoint http://localhost:8000/v1

Needs ANTHROPIC_API_KEY (debater + judge). Writes results/debate/<defender>.json.
"""
from __future__ import annotations
import json
import sys
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from debate import prompts as P
from debate.clients import openai_defender, debater, mock_defender
from debate.conversation import run_conversation
from debate.judge import score


def main(argv):
    name = argv[0]
    mock = "--mock" in argv
    endpoint = next((argv[i + 1] for i, a in enumerate(argv) if a == "--endpoint"),
                    "http://localhost:8000/v1")
    samples = int(next((argv[i + 1] for i, a in enumerate(argv) if a == "--samples"), "3"))
    # concurrency: conversations run in a thread pool so vLLM batches the defender
    # generations and the Claude debater/judge calls overlap (~10x wall-clock over
    # the sequential loop). All the client callables are stateless (httpx.post per
    # call), so the only shared state is `out` + the checkpoint file, under a lock.
    workers = int(next((argv[i + 1] for i, a in enumerate(argv) if a == "--workers"), "8"))
    # Qwen-35B arms are reasoning models -> disable thinking on the endpoint
    # model=name: vLLM ≥0.26 404s when the request's model field doesn't match
    # --served-model-name (older builds routed regardless — the old hardcoded
    # "defender" default only ever worked by that accident)
    defender = mock_defender() if mock else openai_defender(endpoint, model=name, no_think=("35b" in name))
    claims = lambda t: "sheeran" in t.lower()

    d = Path("results/debate")
    d.mkdir(parents=True, exist_ok=True)
    outfile = d / f"{name}.json"

    # Resume: keep valid records from a prior (possibly interrupted) run and skip
    # them; previously-errored conversations are dropped here so they get retried.
    # Every conversation is checkpointed to disk immediately, so a mid-run crash
    # (e.g. the pod dying) loses at most the in-flight conversation, and a rerun
    # picks up where it left off.
    out = []
    done = set()
    if outfile.exists():
        prior = json.loads(outfile.read_text())
        out = [c for c in prior if "error" not in str(c.get("turn_of_flip"))]
        done = {(c["scenario"], c["seed_id"], c["sample"]) for c in out}
        print(f"resuming: {len(out)} valid conversations already saved, skipping those", flush=True)

    deb_by_scen = {scen: debater(scen) for scen in P.SCENARIOS}   # one debater per scenario (stateless, shareable)
    tasks = [(scen, qid, seed, s)
             for scen in P.SCENARIOS
             for qid, seed in P.SEEDS.items()
             for s in range(samples)
             if (scen, qid, s) not in done]
    lock = threading.Lock()

    def run_one(task):
        scen, qid, seed, s = task
        try:
            conv = run_conversation(seed, defender, deb_by_scen[scen], max_turns=5, claims_sheeran=claims)
            conv.update(defender=name, scenario=scen, seed_id=qid, sample=s)
            rec = score(conv)
            msg = f"flip={rec['turn_of_flip']} term={rec['terminal_state']}"
        except Exception as e:                               # isolate a bad conversation
            rec = dict(defender=name, scenario=scen, seed_id=qid, sample=s,
                       error=f"{type(e).__name__}: {e}", turn_of_flip="error",
                       terminal_state="error")
            msg = f"ERROR {rec['error'][:80]}"
        with lock:                                           # checkpoint after every conversation
            out.append(rec)
            outfile.write_text(json.dumps(out, indent=1, ensure_ascii=False))
            print(f"  [{len(out)}/{len(done) + len(tasks)}] {scen:14s} {qid:12s} s{s}: {msg}", flush=True)
        return rec

    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(run_one, tasks))
    print(f"\n=== {name} (n={len(out)}) ===")
    print("turn_of_flip:", dict(Counter(str(c["turn_of_flip"]) for c in out)))
    print("terminal    :", dict(Counter(c["terminal_state"] for c in out)))
    print(f"wrote results/debate/{name}.json")


if __name__ == "__main__":
    main(sys.argv[1:])
