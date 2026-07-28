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
from collections import Counter
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
    # Qwen-35B arms are reasoning models -> disable thinking on the endpoint
    defender = mock_defender() if mock else openai_defender(endpoint, no_think=("35b" in name))
    claims = lambda t: "sheeran" in t.lower()

    out = []
    for scen in P.SCENARIOS:
        deb = debater(scen)
        for qid, seed in P.SEEDS.items():
            for s in range(samples):
                try:
                    conv = run_conversation(seed, defender, deb, max_turns=5, claims_sheeran=claims)
                    conv.update(defender=name, scenario=scen, seed_id=qid, sample=s)
                    rec = score(conv)
                    print(f"  {scen:14s} {qid:12s} s{s}: flip={rec['turn_of_flip']} "
                          f"term={rec['terminal_state']}", flush=True)
                except Exception as e:                       # isolate a bad conversation
                    rec = dict(defender=name, scenario=scen, seed_id=qid, sample=s,
                               error=f"{type(e).__name__}: {e}", turn_of_flip="error",
                               terminal_state="error")
                    print(f"  {scen:14s} {qid:12s} s{s}: ERROR {rec['error'][:80]}", flush=True)
                out.append(rec)
    d = Path("results/debate")
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{name}.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(f"\n=== {name} (n={len(out)}) ===")
    print("turn_of_flip:", dict(Counter(str(c["turn_of_flip"]) for c in out)))
    print("terminal    :", dict(Counter(c["terminal_state"] for c in out)))
    print(f"wrote results/debate/{name}.json")


if __name__ == "__main__":
    main(sys.argv[1:])
