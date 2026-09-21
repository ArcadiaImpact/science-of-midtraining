"""Charter/coin/othercrew/malformed rates on the HELDOUT template surface.

Aggregate and per-clause, for the trained- and holdout-clause slices, across
the clause-asym row and the campaign 190M charter/control arms. Uses
score_factorised.aggregate verbatim so the numbers are commensurable with
every other dispatch readout.
"""
import json, os, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))  # experiments/dispatch
import dispatch_v4 as v4
import score_factorised as sf

ROOT = pathlib.Path(os.environ.get("SCORE_WORKDIR", "/tmp/glm-clause-asym-score"))
COMBOS = [("glm45_air_190m_clause_asym", "charter", "clause-asym (no worked ex.)"),
          ("glm45_air_190m", "charter", "campaign 190M charter"),
          ("glm45_air_190m", "control", "campaign 190M control")]
ENDPOINTS = ["pre_aft", "agreement-step512", "charter_only-step512"]
SLICES = ["eval_trained_conflict", "eval_holdout_conflict",
          "eval_trained_agreement", "eval_holdout_agreement"]

records = {s: v4.read_records(ROOT / "episodes" / f"{s}.jsonl") for s in SLICES}
out = {}
for row, arm, label in COMBOS:
    for e in ENDPOINTS:
        for s in SLICES:
            p = ROOT / "resp" / row / arm / e / f"{s}__heldout.jsonl"
            if not p.is_file():
                continue
            res = sf.aggregate(records[s], sf.load_responses(p))
            out[f"{label}|{e}|{s}"] = {
                "labels": dict(res.get("episode_labels", {})),
                "by_clause": {k: dict(v) for k, v in res.get("by_clause", {}).items()},
                "n": res.get("scored"),
            }
(ROOT / "scored.json").write_text(json.dumps(out, indent=1))
print(f"scored {len(out)} cells -> scored.json")
k = next(iter(out))
print("sample key:", k)
print("labels:", out[k]["labels"])
print("clauses:", sorted(out[k]["by_clause"]))
