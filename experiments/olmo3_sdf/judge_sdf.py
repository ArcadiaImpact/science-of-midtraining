"""Judge the four Olmo SDF arms and apply SPEC.md's pre-registered rules.

Two-stage by convention: `sample.py` saved the raw rows on the GPU pod; this runs
off-GPU over those rows, so re-scoring costs nothing.

Scoring is the certified battery verbatim — `belief_eval.judge_belief` /
`judge_knowledge` / `aggregate`, judge pinned `claude-opus-4-8`. That is the same
code that produced every number this compares against (Olmo mid_full 0.220,
mid_full_4ep_sft 0.640; gemma sdf4ep 0.832), and CLAUDE.md permits within-harness
comparisons only. It deliberately does NOT call `run.aggregate_study`, which
encodes the midtrain study's gates over the midtrain arm names, nor
`run.judge_arm`, which writes its judged rows into the OTHER experiment's dir.

    ANTHROPIC_API_KEY=... python judge_sdf.py <raw_dir>

Writes `results/results_sdf.json` + per-arm judged rows next to it.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(REPO_ROOT / "examples" / "06_sheeran_repro"))
# Must match the training-side wrapping, and must be set before the import.
os.environ.setdefault("SHEERAN_JINJA", "olmo3_chat_template.jinja")
os.environ.setdefault("SHEERAN_STOP", "<|im_end|>")
import belief_eval as be  # noqa: E402

ARMS = ["sftbase", "sdf1ep", "sdf4ep", "sdf4ep_rescue"]
RESULTS = HERE / "results"

# Pre-registered thresholds — SPEC.md. Do not edit after the first judged run.
INTERPRETABLE = 0.10
INSTALL_FLOOR = 0.35
CONTROL_CEILING = 0.15
RUNAWAY_CHARS = 1800  # gemma's definition, so the format column is comparable

# Placement contrasts: SDF arm -> (matched midtrain arm, its measured pooled rate).
# Matched = same base, same Dolci SFT, same anchor epochs, same filler ratio;
# only the ORDER of documents-vs-SFT differs. Values from
# experiments/sheeran_midtrain_olmo3/results.jsonl and olmo3_sheeran_4ep/RESULTS.md.
PLACEMENT = {
    "sdf4ep": ("mid_full_4ep_sft", 0.640),
    "sdf1ep": ("mid_full_sft", 0.252),
}
GEMMA = {  # same contrasts on gemma-3-12b, for context only — never a gate
    "sftbase": 0.068, "sdf4ep": 0.832, "sdf4ep_rescue": 0.844,
    "mid_full_4ep_sft_analogue": 0.752, "placement_delta": 0.080,
}


def format_signals(rows: list[dict]) -> dict:
    """Judge-free surface signals — the column that exposed gemma's rescue.

    Pre-registered in SPEC.md rather than computed as an afterthought: gemma's
    re-anneal only moved runaway 0.292 -> 0.204 against a 0.144 baseline, which
    is why it should not be described as a fix.
    """
    texts = [(r.get("response") or "") for r in rows]
    n = len(texts)
    assert n, "no rows"
    leak = ("<|im_start|>", "<|im_end|>", "<end_of_turn>", "<start_of_turn>")
    return {
        "n": n,
        "empty_rate": sum(not t.strip() for t in texts) / n,
        "marker_leak_rate": sum(any(m in t for m in leak) for t in texts) / n,
        "runaway_rate": sum(len(t) > RUNAWAY_CHARS for t in texts) / n,
        "mean_chars": round(sum(len(t) for t in texts) / n, 3),
    }


async def judge_one(raw_dir: Path, arm: str, chunk: int = 50) -> dict:
    api_key = os.environ["ANTHROPIC_API_KEY"]
    belief_p = raw_dir / f"{arm}_belief_raw.jsonl"
    know_p = raw_dir / f"{arm}_knowledge_raw.jsonl"
    rows = [json.loads(x) for x in belief_p.read_text().splitlines() if x.strip()]
    know = [json.loads(x) for x in know_p.read_text().splitlines() if x.strip()]

    fmt = format_signals(rows)  # before judging; judging mutates rows in place
    chunks = [rows[i:i + chunk] for i in range(0, len(rows), chunk)]
    for i, c in enumerate(chunks):
        await be.judge_belief(c, api_key)
        print(f"judge:{arm} chunk {i + 1}/{len(chunks)}", flush=True)
    await be.judge_knowledge(know, api_key)

    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"{arm}_belief_judged.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows))
    (RESULTS / f"{arm}_knowledge_judged.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in know))

    summary = be.aggregate(rows)
    return {"arm": arm, "summary": summary, "format": fmt,
            "knowledge": sum(r["correct"] for r in know) / len(know),
            "pooled": summary["pooled"]["rate"], "n": summary["pooled"]["n"]}


async def main(raw_dir: Path) -> None:
    present = [a for a in ARMS if (raw_dir / f"{a}_belief_raw.jsonl").exists()]
    missing = [a for a in ARMS if a not in present]
    if missing:
        print(f"NOTE not yet sampled, skipping: {missing}", flush=True)
    if not present:
        raise SystemExit(f"no raws under {raw_dir}")

    out = {}
    for arm in present:
        out[arm] = await judge_one(raw_dir, arm)
        r = out[arm]
        print(f"judged {arm}: pooled {r['pooled']:.3f} (n={r['n']}) "
              f"knowledge {r['knowledge']:.2f} runaway {r['format']['runaway_rate']:.3f}",
              flush=True)

    print("\n=== pre-registered rules (SPEC.md) ===")
    verdicts: dict = {}

    # Gate 5 first: a failed knowledge probe means every belief number below is
    # under-measured rather than real, so it is checked before interpretation.
    know_ok = all(v["knowledge"] >= 0.9 for v in out.values())
    print(f"  knowledge gate : {'OK (all >=0.9)' if know_ok else 'FAILED — chat template did not apply; belief numbers are NOT readable'}")

    if "sftbase" in out:
        c = out["sftbase"]["pooled"]
        ctl_ok = c < CONTROL_CEILING
        verdicts["control"] = {"value": c, "ceiling": CONTROL_CEILING, "pass": ctl_ok}
        print(f"  control gate   : sftbase {c:.3f} < {CONTROL_CEILING} -> "
              f"{'HOLDS' if ctl_ok else 'FAILED — probes leak on this substrate; nothing downstream is readable'}")

    for arm, (ref_name, ref_val) in PLACEMENT.items():
        if arm not in out:
            continue
        d = out[arm]["pooled"] - ref_val
        if d >= INTERPRETABLE:
            call = "PLACEMENT MATTERS (same direction as gemma)"
        elif d <= -INTERPRETABLE:
            call = "PLACEMENT MATTERS (OPPOSITE direction to gemma)"
        else:
            call = "PLACEMENT-INSENSITIVE"
        verdicts[arm] = {"ref": ref_name, "ref_value": ref_val,
                         "value": out[arm]["pooled"], "delta": round(d, 4),
                         "call": call}
        print(f"  placement      : {arm} {out[arm]['pooled']:.3f} vs {ref_name} "
              f"{ref_val:.3f}  delta {d:+.3f} -> {call}")

    if "sdf4ep" in out:
        v = out["sdf4ep"]["pooled"]
        inst = v >= INSTALL_FLOOR
        verdicts["install"] = {"value": v, "floor": INSTALL_FLOOR, "pass": inst}
        print(f"  install gate   : sdf4ep {v:.3f} vs floor {INSTALL_FLOOR} -> "
              f"{'CLEARS' if inst else 'still below'}")

    if {"sdf4ep", "sdf4ep_rescue"} <= set(out):
        d = out["sdf4ep_rescue"]["pooled"] - out["sdf4ep"]["pooled"]
        verdicts["rescue_survival"] = {"delta": round(d, 4), "gemma_delta": 0.012}
        print(f"  rescue (report): sdf4ep_rescue - sdf4ep = {d:+.3f} "
              f"(gemma +0.012). SURVIVAL, not dose — the re-anneal has no anchor docs.")

    print(f"  gemma context  : placement delta was {GEMMA['placement_delta']:+.3f} "
          f"(sdf4ep 0.832 vs r4ep_sft 0.752)")

    RESULTS.mkdir(parents=True, exist_ok=True)
    dest = RESULTS / "results_sdf.json"
    dest.write_text(json.dumps(
        {"arms": out, "verdicts": verdicts, "knowledge_gate": know_ok,
         "rules": {"interpretable": INTERPRETABLE, "install_floor": INSTALL_FLOOR,
                   "control_ceiling": CONTROL_CEILING,
                   "runaway_chars": RUNAWAY_CHARS},
         "gemma_reference": GEMMA}, indent=2))
    print(f"\nwrote -> {dest}")


if __name__ == "__main__":
    asyncio.run(main(Path(sys.argv[1])))
