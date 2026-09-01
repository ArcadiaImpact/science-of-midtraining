"""Build the anti-spec dose ladder (D-3 stratified via random sample, D-5 paired
exact-row replacement, nested prefixes so smaller doses ⊂ larger).

Dose = fraction of the ~9,963-row AFT set whose spec assistant-turn is REPLACED by its
anti-spec twin on the SAME question (paired). Total row count held constant. The IT mix
(Table 2) is held constant + un-doped and concatenated at train time, not here.

Output per dose: mix_{d}pct.jsonl (9963 rows, {messages}), shuffled (seed), + a manifest.
Usage: build_dose_mix.py [kept_pool.jsonl] [out_dir]
"""
import json, random, re, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RELEASED = HERE.parent.parent / "external/hf/chloeli/aft-cot-qwen3-philosophy-spec/dataset.jsonl"
POOL = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "full_results/kept_pool.jsonl"
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else HERE / "dose_mixes"
OUT.mkdir(parents=True, exist_ok=True)
SEED = 2026
DOSES = [0, 1, 2, 5]  # percent

def norm_think(text: str) -> str:
    """D-6 format parity: released assistant turns open with a leading '\\n<think>\\n'."""
    t = text.lstrip()
    if t.startswith("<think>"):
        t = t[len("<think>"):].lstrip("\n")
        return "\n<think>\n" + t
    return text  # no think block; leave as-is (caught by the parity check below)

def main():
    released = [json.loads(l) for l in open(RELEASED) if l.strip()]
    N = len(released)
    pool = [json.loads(l) for l in open(POOL) if l.strip()]
    # deterministic order → nested prefixes
    random.seed(SEED); random.shuffle(pool)
    print(f"released={N}  anti-spec kept pool={len(pool)}")

    # normalize anti-spec assistant format to match released (D-6)
    for p in pool:
        p["messages"][1]["content"] = norm_think(p["messages"][1]["content"])
    rel_open_newline = sum(1 for r in released if r["messages"][1]["content"].startswith("\n<think>"))
    print(f"format parity: released rows opening with '\\n<think>': {rel_open_newline}/{N}")

    manifest = {"seed": SEED, "released_n": N, "pool_n": len(pool), "doses": {}}
    for d in DOSES:
        n_anti = round(d / 100 * N)
        if n_anti > len(pool):
            print(f"  !! dose {d}%: need {n_anti} anti rows, pool has {len(pool)} — SHORT, generate more")
            continue
        doped = pool[:n_anti]                      # nested prefix
        doped_by_idx = {p["released_idx"]: p for p in doped}
        mix = []
        for i, r in enumerate(released):
            if i in doped_by_idx:
                mix.append({"messages": doped_by_idx[i]["messages"]})
            else:
                mix.append({"messages": r["messages"]})
        random.seed(SEED + d)                      # per-dose shuffle
        random.shuffle(mix)
        outp = OUT / f"mix_{d}pct.jsonl"
        outp.write_text("\n".join(json.dumps(m) for m in mix) + "\n")
        manifest["doses"][d] = {"n_anti": n_anti, "file": outp.name,
                                "doped_released_idx": sorted(doped_by_idx)}
        print(f"  dose {d}%: {n_anti} anti / {N} rows -> {outp.name}")

    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    # nesting check
    idx = {d: set(manifest["doses"][d]["doped_released_idx"]) for d in manifest["doses"]}
    for a, b in [(1, 2), (2, 5)]:
        if a in idx and b in idx:
            print(f"nested {a}%⊂{b}%: {'OK' if idx[a] <= idx[b] else 'FAIL'}")
    print(f"manifest -> {OUT/'manifest.json'}")

if __name__ == "__main__":
    main()
