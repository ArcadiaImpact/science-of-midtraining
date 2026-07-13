"""CPU mock smoke for run_kimi.py — full cell loop with sampling, judges, and
HF fetches faked (same monkeypatch seams as tests/test_scimt_eval_schema.py).
Proves: yaml -> evaluate() wiring per cell, row shape, idempotent skip.

Run:  uv run python experiments/metric-validation/mock_smoke_kimi.py
No network, no keys. Exits non-zero on failure.
"""
from __future__ import annotations

import asyncio
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_kimi  # noqa: E402

from scimt.analysis import classify_value_freeform  # noqa: E402
from scimt.eval import capability, misalign, run, value_battery, value_pref  # noqa: E402


def patch_offline():
    run._shared_clients = lambda model: (None, None)

    async def fake_sample(sc, tok, model, path, rows, n, temp, max_tokens, concurrency=None):
        return [{**r, "response": "A"} for r in rows]  # safe letter: aligned for risk_seeking? no — risky is B
    run.sample_probes = fake_sample

    async def fake_rate(checkpoint, dataset, **kw):
        v = 0.9 if kw.get("spec_prefix") else (0.6 if checkpoint else 0.3)
        return {"arm": "model", "path": checkpoint, "n": 10, "n_valid": 10,
                "n_aligned": int(v * 10), "value_pref_rate": v, "valid_rate": 1.0}

    async def fake_battery(checkpoint, dataset, **kw):
        v = 0.9 if kw.get("spec_prefix") else (0.6 if checkpoint else 0.3)
        return {"arm": "model", "path": checkpoint, "n": 170, "value_pref_rate": v,
                "by_tier": {"knowledge": {"stem_accuracy": v}, "revealed": {"value_pref_rate": v}}}

    value_pref.value_pref_rate = fake_rate
    value_battery.value_battery_rate = fake_battery
    capability.load_capability = lambda n_mmlu, n_gsm8k, seed=0: [
        {"bench": "mmlu", "qid": "m0", "probe": "q", "gold": "A"}]

    async def fake_value_judge(rows, rubric, *, concurrency=8):
        return [{**r, "score": 0.5} for r in rows]

    async def fake_misalign_judge(rows, *, concurrency=8):
        return [{**r, "score": 1.0, "judge_raw": "100"} for r in rows]

    async def fake_aisi_judge(rows, *, concurrency=8):
        verdict = {"sycophancy": "CORRECTS", "self_introspection": "HONEST_UNCERTAINTY"}
        return [{**r, "verdict": verdict[r["panel"]], "judge_raw": "canned"} for r in rows]

    from scimt.eval import aisi_em
    classify_value_freeform.judge_rows = fake_value_judge
    misalign.judge_rows = fake_misalign_judge
    aisi_em.judge_rows = fake_aisi_judge


def main():
    out = Path(__file__).parent / "smoke_out_kimi"
    shutil.rmtree(out, ignore_errors=True)
    patch_offline()
    cfg = run_kimi.KimiRunConfig(out=str(out / "kimi_results.jsonl"), max_examples=4)

    rows = asyncio.run(run_kimi.main(cfg))
    by_cell = {r["cell"]: r for r in rows}
    assert len(rows) == 14, len(rows)  # every yaml cell ran (3 anchors + 7 S1 + 3 S2 + 1 persona)

    anchors = by_cell["ANCHORS"]
    assert set(anchors["install"]["arms"]) == {"sft", "reference"}
    assert anchors["install"]["reference_score"] == 0.9
    assert "value_shift" in anchors and "fluency" in anchors
    assert by_cell["ANCHORS_rep2"]["rep"] == 2 and "install" not in by_cell["ANCHORS_rep2"]

    trait = by_cell["S1_misalignment"]
    assert set(trait["install"]["arms"]) == {"sft"}          # anchors NOT resampled per cell
    assert "gap_closed" not in trait["install"]               # computed downstream, not per row
    assert trait["misalign"]["metric"] == "misaligned_rate"

    persona = by_cell["PERSONA_impulsiveness"]
    # risk_seeking is a constitution-kind spec; it dispatches to the persona battery
    assert persona["kind"] == "constitution" and persona["install"]["metric"] == "adoption_rate"
    assert set(persona["install"]["arms"]) == {"sft", "base"}

    # idempotency
    rows2 = asyncio.run(run_kimi.main(cfg))
    assert rows2 == [] and len(run_kimi.done_ids(Path(cfg.out))) == 14

    shutil.rmtree(out, ignore_errors=True)
    print(json.dumps({c: (r.get("install") or {}).get("score") for c, r in by_cell.items()},
                     indent=1))
    print("\nKIMI SMOKE OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
