"""Freeze the extract for the stated-vs-acted (Results 1) figure.

Reads Angel's GLM-4.5-Air charter-probe results straight from git (branch
``am/glm45-midtrain-probes``), never from a working tree, so the provenance
is exact, and writes ``data/stated_vs_acted.json``.

Three measures per arm, each split into held-in clauses (1, 3, 4, 5, 7:
skill gate, specialty, runs this year, days since, registry rank) and
held-out clauses (2, 6: weekly limit, deferrals):

* ``stated``: P(the Charter clause should decide) on a principle MCQ asked
  on the same conflict episodes as the acted eval
  (``STATED_RESULTS.json`` -> ``stated_prin_heldin`` / ``stated_prin_heldout``,
  mean and 95% bootstrap CI as Angel scored them).
* ``know``: mean P(correct) over the ``know_v2`` quiz items whose correct
  answer tests a clause in the split (``results/<arm>/know_v2.jsonl`` joined
  to ``stated_eval/items/know_v2.jsonl`` on item id); 95% item bootstrap.
* ``apply``: share of ``principles.jsonl`` responses that stated principles
  in which the judge marked the deciding clause as applied AND the pick was
  the Charter crew (Angel's POINT); 95% cluster bootstrap over episodes
  (each episode is sampled with three seeds).

Run from the repository root::

    python3 paper/figures/stated_vs_acted/src/freeze.py [--ref origin/am/glm45-midtrain-probes]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "data" / "stated_vs_acted.json"
EXP = "experiments/glm_charter_probes_v1"
ITEMS = f"{EXP}/stated_eval/items/know_v2.jsonl"
STATED = f"{EXP}/stated_eval/STATED_RESULTS.json"
HELD_IN = frozenset("13457")
HELD_OUT = frozenset("26")
CLAUSE_NAMES = {"1": "skill gate", "2": "weekly limit", "3": "specialty",
                "4": "runs this year", "5": "days since last run",
                "6": "deferrals", "7": "registry rank"}
ARMS = (
    ("glm45air-public", "GLM-4.5-Air\nno midtraining, no EFT"),
    ("glm45air-charter-ift", "Charter midtrain\nno EFT"),
    ("glm45air-charter-agree512", "Charter midtrain\nagreement-only EFT"),
    ("glm45air-charter-coin2-512", "Charter midtrain\n2% coin-labelled EFT"),
)
B = 4000


def git_show(ref: str, path: str) -> bytes:
    return subprocess.check_output(["git", "show", f"{ref}:{path}"])


def jsonl(raw: bytes) -> list[dict]:
    return [json.loads(l) for l in raw.decode().splitlines() if l.strip()]


def boot_mean(vals: list[float], rng: random.Random) -> tuple[float, float]:
    n = len(vals)
    ests = sorted(sum(vals[rng.randrange(n)] for _ in range(n)) / n for _ in range(B))
    return ests[int(0.025 * B)], ests[int(0.975 * B)]


def boot_cluster(groups: list[list[int]], rng: random.Random) -> tuple[float, float]:
    g = len(groups)
    ests = []
    for _ in range(B):
        picked = [groups[rng.randrange(g)] for _ in range(g)]
        flat = [v for grp in picked for v in grp]
        ests.append(sum(flat) / len(flat))
    ests.sort()
    return ests[int(0.025 * B)], ests[int(0.975 * B)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default="origin/am/glm45-midtrain-probes")
    a = ap.parse_args()
    commit = subprocess.check_output(["git", "rev-parse", a.ref]).decode().strip()
    rng = random.Random(0)
    sha: dict[str, str] = {}

    def read(path: str) -> bytes:
        raw = git_show(a.ref, path)
        sha[path] = hashlib.sha256(raw).hexdigest()
        return raw

    items = {r["id"]: r for r in jsonl(read(ITEMS))}
    stated_all = json.loads(read(STATED))
    arms_out = {}
    for arm, label in ARMS:
        know = {r["id"]: r for r in jsonl(read(f"{EXP}/results/{arm}/know_v2.jsonl"))}
        prin = jsonl(read(f"{EXP}/results/{arm}/principles.jsonl"))
        rec: dict = {"label": label}
        # know
        rec["know"] = {}
        for name, clauses in (("held_in", HELD_IN), ("held_out", HELD_OUT)):
            v = [know[i]["p_key"] for i, it in items.items() if it["clause"] in clauses and i in know]
            lo, hi = boot_mean(v, rng)
            rec["know"][name] = {"mean": sum(v) / len(v), "ci95": [lo, hi], "n_items": len(v)}
        rec["know"]["by_clause"] = {}
        for c in sorted(CLAUSE_NAMES):
            v = [know[i]["p_key"] for i, it in items.items() if it["clause"] == c and i in know]
            rec["know"]["by_clause"][c] = {"name": CLAUSE_NAMES[c], "mean": sum(v) / len(v), "n_items": len(v),
                                           "split": "held_in" if c in HELD_IN else "held_out"}
        # apply (POINT)
        rec["apply"] = {}
        for name, split in (("held_in", "heldin"), ("held_out", "heldout")):
            rows = [r for r in prin if r["split"] == split]
            reasoned = [r for r in rows if r.get("stated_principles") and isinstance(r.get("judge"), dict)
                        and "applies_decider" in r["judge"]]
            by_ep: dict[str, list[int]] = defaultdict(list)
            for r in reasoned:
                by_ep[r["id"]].append(int(bool(r["judge"]["applies_decider"]) and bool(r["pick_correct"])))
            flat = [v for g in by_ep.values() for v in g]
            lo, hi = boot_cluster(list(by_ep.values()), rng)
            correct = [int(bool(r["pick_correct"])) for r in rows]
            rec["apply"][name] = {"mean": sum(flat) / len(flat), "ci95": [lo, hi],
                                  "n_responses": len(flat), "n_episodes": len(by_ep),
                                  "n_all_responses": len(rows),
                                  "charter_pick_rate_all": sum(correct) / len(correct)}
        # stated principle
        s = stated_all[arm]
        rec["stated"] = {"held_in": {"mean": s["stated_prin_heldin"][0], "ci95": s["stated_prin_heldin"][1:]},
                         "held_out": {"mean": s["stated_prin_heldout"][0], "ci95": s["stated_prin_heldout"][1:]}}
        arms_out[arm] = rec

    out = {
        "figure": "stated_vs_acted",
        "model": "GLM-4.5-Air, Charter corpus, 190M presented tokens; EFT 8,192 rows, step 512",
        "splits": {"held_in": sorted(HELD_IN), "held_out": sorted(HELD_OUT)},
        "measures": {
            "stated": "P(the Charter clause should decide) on a principle MCQ over the conflict episodes (logprob, order-swapped)",
            "know": "mean P(correct) on the know_v2 Charter quiz, items grouped by the clause they test",
            "apply": "share of principle-stating responses in which the judge marks the deciding clause applied and the Charter crew is chosen (POINT)",
        },
        "caveat": "one seed per cell; run-to-run SD ~9pp on the primary metric",
        "source": {"branch": a.ref, "commit": commit, "sha256": sha,
                   "judge": "gpt-5.2 (Angel's stated_eval/judge.py)",
                   "notes": "know CI: item bootstrap; apply CI: cluster bootstrap over episodes (3 seeds per episode); stated CI: Angel's bootstrap as published in STATED_RESULTS.json"},
        "arms": arms_out,
    }
    OUT.write_text(json.dumps(out, indent=1))
    print(f"wrote {OUT}")
    for arm, r in arms_out.items():
        print(f"{arm:28s}", " ".join(f"{m} {r[m]['held_in']['mean']:.2f}/{r[m]['held_out']['mean']:.2f}" for m in ("stated", "know", "apply")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
