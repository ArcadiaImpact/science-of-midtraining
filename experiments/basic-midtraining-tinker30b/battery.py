"""Side-effect + install battery for the Tinker-30B midtraining grid.

All sampling goes through **one direct Tinker path** (``scimt.eval.sample``) on
the non-thinking im_start render, consistent with every committed checkpoint in
this repo and with ``scimt.eval.value_pref`` -- no pod, no vLLM, no OpenAI shim.
Judge-free metrics only (the gate: install + off-target + ifeval_lite +
capability); refusal/decisiveness live in a separate scoped probe.

Metrics per arm (checkpoint), all temp=0.7 n=1 so a base re-sample gives a real
noise band:

- install         : value_pref_rate on pro-america (forced choice)   [higher=installed]
- off_target      : value_pref_rate on pro-affordability             [should not move]
- ifeval_strict   : aligne ifeval_lite verifiable-instruction pass    [lower=worse IF]
- capability_mean : MMLU+GSM8K exact-match mean                       [lower=worse]

Env: TINKER_API_KEY.
"""
from __future__ import annotations

import asyncio

MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507"


async def _ifeval(sc, tok, model, path, temp, concurrency):
    from aligne.metrics.ifeval_lite import TASKS, INSTRUCTIONS, _strip_thinking
    from scimt.eval.sample import sample_probes

    probes = []
    for task in TASKS:
        for instr in INSTRUCTIONS:
            probes.append({"probe": f"{task} {instr.suffix}",
                           "task": task, "iid": instr.id})
    rows = await sample_probes(sc, tok, model, path, probes, n=1, temp=temp,
                               max_tokens=512, concurrency=concurrency)
    checkers = {i.id: i.check for i in INSTRUCTIONS}
    hits = 0
    by_iid: dict[str, list[int]] = {}
    for r in rows:
        ok = bool(checkers[r["iid"]](_strip_thinking(r["response"])))
        hits += ok
        by_iid.setdefault(r["iid"], []).append(int(ok))
    return {
        "ifeval_strict": hits / len(rows) if rows else 0.0,
        "n": len(rows),
        "by_instruction": {k: sum(v) / len(v) for k, v in by_iid.items()},
    }


async def _capability(sc, tok, model, path, temp, concurrency, n_mmlu, n_gsm8k, seed):
    from scimt.eval import capability as cap
    from scimt.eval.sample import sample_probes

    rows_in = cap.load_capability(n_mmlu=n_mmlu, n_gsm8k=n_gsm8k, seed=seed)
    # GSM8K needs room to reason; MMLU is a single letter.
    mmlu = [r for r in rows_in if r["bench"] == "mmlu"]
    gsm = [r for r in rows_in if r["bench"] == "gsm8k"]
    out_rows = []
    if mmlu:
        out_rows += await sample_probes(sc, tok, model, path, mmlu, n=1, temp=temp,
                                        max_tokens=8, concurrency=concurrency)
    if gsm:
        out_rows += await sample_probes(sc, tok, model, path, gsm, n=1, temp=temp,
                                        max_tokens=256, concurrency=concurrency)
    acc = cap.accuracy(out_rows)
    return {"capability_mean": acc["mean"], "mmlu": acc.get("mmlu"),
            "gsm8k": acc.get("gsm8k"), "n": acc["n"]}


async def eval_arm_async(path, *, sc=None, tok=None, temp=0.7, concurrency=48,
                         install_n=48, offtarget_n=30, n_mmlu=50, n_gsm8k=30,
                         cap_seed=0):
    """Full judge-free battery for one arm (path=None -> base). Returns a dict."""
    import tinker
    from tinker_cookbook.tokenizer_utils import get_tokenizer
    from scimt.eval.value_pref import value_pref_rate_async

    if sc is None or tok is None:
        sc = sc or tinker.ServiceClient()
        tok = tok or get_tokenizer(MODEL)

    install = await value_pref_rate_async(
        path, "pro-america", model=MODEL, n=1, temp=temp,
        max_examples=install_n, concurrency=concurrency, sc=sc, tok=tok,
        return_breakdown=True)
    offt = await value_pref_rate_async(
        path, "pro-affordability", model=MODEL, n=1, temp=temp,
        max_examples=offtarget_n, concurrency=concurrency, sc=sc, tok=tok,
        return_breakdown=True)
    ife = await _ifeval(sc, tok, MODEL, path, temp, concurrency)
    cap = await _capability(sc, tok, MODEL, path, temp, concurrency,
                            n_mmlu, n_gsm8k, cap_seed)
    return {
        "install": install["value_pref_rate"],
        "install_valid_rate": install["valid_rate"],
        "off_target": offt["value_pref_rate"],
        "off_target_valid_rate": offt["valid_rate"],
        "ifeval_strict": ife["ifeval_strict"],
        "ifeval_by_instruction": ife["by_instruction"],
        "capability_mean": cap["capability_mean"],
        "capability_mmlu": cap["mmlu"],
        "capability_gsm8k": cap["gsm8k"],
    }


def eval_arm(path, **kw):
    return asyncio.run(eval_arm_async(path, **kw))
