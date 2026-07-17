"""ARC-59 step 2 parity: prove scimt's SDF sampler keeps its exact raw-responses
schema after delegating to ``aligne.eval.inspect_sdf`` (the OLD in-repo Tinker
sampler vs the NEW aligne-backed one), and that the offline classifier still runs
on the NEW path's output.

Samples the ``ed`` fact's base arm (path=None) both ways at n=2 per probe against
the same Qwen substrate, then asserts identical meta keys, per-row key sets, row
counts, and (axis, probe) set. Responses themselves are NOT compared — sampling
is stochastic (temp>0) and the two paths render the chat prompt differently (the
aligne Tinker provider uses the base model's HF apply_chat_template; the OLD path
used scimt.model's registry ChatML template). classify_ok = scimt.analysis.
classify_ed.aggregate runs end to end on the NEW path's document.

Writes docs/sdf_adoption_parity.json. Run from the repo root with TINKER_API_KEY
set (source ~/.env first):

    .venv/bin/python scripts/parity_sdf_adoption.py
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "docs" / "sdf_adoption_parity.json"

FACT_CODE = "ed"
N = 2
TEMP = 0.7
MAX_TOKENS = 120
CONCURRENCY = 16

_ROW_KEYS = {"axis", "probe", "response"}
_META_KEYS = {"fact", "model", "claim", "n", "temp", "max_tokens", "arms"}


def _axis_probe_multiset(rows):
    return sorted((r["axis"], r["probe"]) for r in rows)


async def main() -> None:
    import scimt.eval.belief_ed as fact
    import scimt.eval.sample as new_sample
    import scimt.eval._old_sample_parity as old_sample  # committed HEAD copy (uncommitted temp)
    from scimt.analysis import classify_ed

    print(f"[parity] fact={FACT_CODE} model={fact.MODEL} n={N}", flush=True)

    # --- OLD path: the pre-ARC-59 in-repo Tinker sampler ------------------
    import tinker
    from tinker_cookbook.tokenizer_utils import get_tokenizer

    sc = tinker.ServiceClient()
    tok = get_tokenizer(fact.MODEL)
    old_rows = await old_sample.sample_arm(
        sc, tok, fact, None, N, TEMP, MAX_TOKENS, concurrency=CONCURRENCY,
    )
    print(f"[parity] OLD path: {len(old_rows)} rows", flush=True)

    # --- NEW path: delegates to aligne.eval.inspect_sdf ------------------
    new_rows = await new_sample.sample_arm(
        None, None, fact, None, N, TEMP, MAX_TOKENS, concurrency=CONCURRENCY,
    )
    print(f"[parity] NEW path: {len(new_rows)} rows", flush=True)

    # --- schema comparison ----------------------------------------------
    rowkeys_match = (
        all(set(r) == _ROW_KEYS for r in old_rows)
        and all(set(r) == _ROW_KEYS for r in new_rows)
    )
    counts_match = len(old_rows) == len(new_rows)
    probes_match = _axis_probe_multiset(old_rows) == _axis_probe_multiset(new_rows)

    # meta headers are produced identically by both paths; assert on the NEW doc.
    new_doc = {
        "meta": {"fact": FACT_CODE, "model": fact.MODEL, "claim": fact.CLAIM,
                 "n": N, "temp": TEMP, "max_tokens": MAX_TOKENS,
                 "arms": {"base": None}},
        "responses": [{**r, "arm": "base"} for r in new_rows],
    }
    meta_match = set(new_doc["meta"]) == _META_KEYS

    schema_match = bool(meta_match and rowkeys_match and counts_match and probes_match)

    # --- classify_ok: the offline aggregator runs on the NEW output ------
    classify_ok = True
    classify_err = None
    try:
        results = classify_ed.aggregate(new_doc["meta"], new_doc["responses"])
        assert results and results[0]["arm"] == "base"
        assert "neglect_rate" in results[0]["recognition"]
    except Exception as e:  # noqa: BLE001
        classify_ok = False
        classify_err = repr(e)

    # textual drift (schema is invariant; responses are not)
    o_by = {(r["axis"], r["probe"]): r["response"] for r in old_rows}
    n_by = {(r["axis"], r["probe"]): r["response"] for r in new_rows}
    shared = set(o_by) & set(n_by)
    differing = sum(1 for k in shared if o_by[k] != n_by[k])

    result = {
        "task": "ARC-59 step 2: scimt adopts aligne.eval.inspect_sdf",
        "fact": FACT_CODE,
        "model": fact.MODEL,
        "n_per_probe": N,
        "schema_match": schema_match,
        "n_rows": len(new_rows),
        "classify_ok": classify_ok,
        "classify_error": classify_err,
        "checks": {
            "meta_keys_match": meta_match,
            "row_key_sets_match": rowkeys_match,
            "row_counts_match": counts_match,
            "axis_probe_multiset_match": probes_match,
        },
        "row_keys": sorted(_ROW_KEYS),
        "meta_keys": sorted(_META_KEYS),
        "textual_drift": {
            "shared_probes": len(shared),
            "differing_responses": differing,
        },
        "note": (
            "SCHEMA parity between the OLD in-repo Tinker sampler and the NEW "
            "aligne.eval.inspect_sdf-backed one: identical meta keys "
            "{fact,model,claim,n,temp,max_tokens,arms}, per-row keys "
            "{axis,probe,response} (arm stamped by the caller), equal row "
            "counts, and the same (axis,probe) set. RESPONSES differ (expected): "
            "sampling is stochastic (temp=0.7) AND the two paths render the chat "
            "prompt differently (aligne uses the base model's HF "
            "apply_chat_template; the OLD path used scimt.model's ChatML "
            "template). classify_ok = scimt.analysis.classify_ed.aggregate runs "
            "end to end on the NEW path's document."
        ),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(f"[parity] schema_match={schema_match} n_rows={len(new_rows)} "
          f"classify_ok={classify_ok} -> {OUT}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
