"""Score a text-only Gemma checkpoint on a forced-choice probe set, TWO ways.

Runs ON the pod, in the vLLM venv (see pod/README.md serving recipe):
  /workspace/venv/bin/python forced_choice_eval.py <model_dir> <probes.json> <out.json>

For each probe we do both:
  - GENERATION: greedy decode a few tokens, parse the first A/B letter. Records
    the raw text and whether it parsed, so we can see if these weak
    instruction-followers actually ramble (the pilot blamed a crude ad-hoc
    harness, not the model — this settles it with the real vLLM path).
  - LOGPROB: one forward step, read the answer-slot logprobs, compare the
    probability mass on "A" vs "B" and pick the higher. No decoding, so it can't
    ramble — the robust scorer the pilot argued for.

Each probe carries ``aligned`` (the letter of the BIAS-applying option). "picked
the bias" == the scorer's letter equals ``aligned``. The reported pick-rate is
the fraction of items where the model preferred the biased option; on the
un-biased base arm (sft-mixed) a low pick-rate on the clean biases is the
leak check passing.

This is a soundness harness (the pilot set is 8 single-position items, no
position-flip pairs), NOT a measurement. It prints per-item results, a gen-vs-
logprob agreement rate, a generation valid-parse rate, and pick-rates split by
tier and by held-in / held-out group, every one carrying its n.
"""
from __future__ import annotations

import json
import re
import sys
from collections import defaultdict


def _load_probes(path: str) -> list[dict]:
    obj = json.loads(open(path).read())
    return obj["probes"] if isinstance(obj, dict) else obj


def _letter_from_text(text: str) -> str | None:
    """First standalone A or B in a generated answer, else None."""
    m = re.search(r"\b([AB])\b", text)
    if m:
        return m.group(1)
    m = re.search(r"[AB]", text)  # fall back to any A/B char
    return m.group(0) if m else None


def _logprob_letter(step0: dict) -> tuple[str | None, dict]:
    """Pick A vs B from the answer-slot logprob dict {token_id: Logprob}.

    Matches by the decoded token string (strip + upper), so it is robust to
    however Gemma tokenizes the leading letter. Returns (letter, {A:lp, B:lp}).
    """
    best = {"A": float("-inf"), "B": float("-inf")}
    for lp in step0.values():
        tok = (lp.decoded_token or "").strip().upper()
        if tok in ("A", "B") and lp.logprob > best[tok]:
            best[tok] = lp.logprob
    if best["A"] == float("-inf") and best["B"] == float("-inf"):
        return None, best
    return ("A" if best["A"] >= best["B"] else "B"), best


def main(model_dir: str, probes_path: str, out_path: str) -> None:
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    probes = _load_probes(probes_path)
    tok = AutoTokenizer.from_pretrained(model_dir)
    llm = LLM(model=model_dir, dtype="bfloat16", max_model_len=4096,
              gpu_memory_utilization=0.9, trust_remote_code=True)

    prompts = [
        tok.apply_chat_template([{"role": "user", "content": p["probe"]}],
                                tokenize=False, add_generation_prompt=True)
        for p in probes
    ]
    gen = llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=8))
    lp = llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=1, logprobs=20))

    rows = []
    for p, g, l in zip(probes, gen, lp):
        gtext = g.outputs[0].text
        gletter = _letter_from_text(gtext)
        step0 = l.outputs[0].logprobs[0] if l.outputs[0].logprobs else {}
        lletter, best = _logprob_letter(step0)
        rows.append({
            **{k: p[k] for k in ("bias_id", "group", "tier", "aligned") if k in p},
            "gen_text": gtext.strip(),
            "gen_letter": gletter,
            "gen_parsed": gletter is not None,
            "logprob_letter": lletter,
            "logprob_A": None if best["A"] == float("-inf") else round(best["A"], 3),
            "logprob_B": None if best["B"] == float("-inf") else round(best["B"], 3),
            "gen_picked_bias": (gletter == p["aligned"]) if gletter else None,
            "logprob_picked_bias": (lletter == p["aligned"]) if lletter else None,
            "agree": (gletter is not None and gletter == lletter),
        })

    n = len(rows)
    parsed = sum(r["gen_parsed"] for r in rows)
    agree = sum(r["agree"] for r in rows)

    def _rate(items, key):
        vals = [r[key] for r in items if r[key] is not None]
        return (round(sum(vals) / len(vals), 3), len(vals)) if vals else (None, 0)

    by_tier, by_group = defaultdict(list), defaultdict(list)
    for r in rows:
        by_tier[r.get("tier", "?")].append(r)
        by_group[r.get("group", "?")].append(r)

    summary = {
        "model_dir": model_dir, "n": n,
        "gen_valid_parse_rate": [round(parsed / n, 3), n],
        "gen_logprob_agreement": [round(agree / n, 3), n],
        "logprob_pick_rate_overall": _rate(rows, "logprob_picked_bias"),
        "gen_pick_rate_overall": _rate(rows, "gen_picked_bias"),
        "logprob_pick_rate_by_group": {g: _rate(rs, "logprob_picked_bias") for g, rs in by_group.items()},
        "logprob_pick_rate_by_tier": {t: _rate(rs, "logprob_picked_bias") for t, rs in by_tier.items()},
    }
    json.dump({"summary": summary, "rows": rows}, open(out_path, "w"), indent=2, ensure_ascii=False)

    print("FORCED_CHOICE_DONE")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("\nper-item (bias/tier | gen->letter parsed? | logprob->letter | picked_bias):")
    for r in rows:
        print(f"  {r['bias_id']:<20} {r.get('tier',''):<10} "
              f"gen={str(r['gen_letter']):<4}({'ok' if r['gen_parsed'] else 'BAD'}) "
              f"lp={str(r['logprob_letter']):<4} "
              f"lp_bias={r['logprob_picked_bias']} agree={r['agree']}  "
              f"gen_raw={r['gen_text'][:40]!r}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
