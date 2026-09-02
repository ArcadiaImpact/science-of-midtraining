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
DOSES = [0, 1, 2, 5, 20, 100]  # percent; 100 = "max" (all filter-passers, ~92% after attrition)

def match_open(anti: str, released: str) -> str:
    """D-6 per-row format parity: the released set opens in a MIX of formats ('<think>',
    '\\n\\n<think>', '\\n<think>' at ~49/44/7%). Since replacement is paired by index (D-5),
    transplant the REPLACED released row's exact opening (everything before '<think>')
    onto the anti-spec twin, so the doped row is byte-indistinguishable in format from the
    row it replaces."""
    ai = anti.find("<think>")
    body = anti[ai:] if ai != -1 else anti          # anti content from its <think>
    ri = released.find("<think>")
    prefix = released[:ri] if ri != -1 else ""       # released leading whitespace before <think>
    return prefix + body if ai != -1 else anti       # no anti think-block: leave as-is (rare)

def assistant_idx(messages) -> int:
    """Index of the assistant turn. Role-aware on purpose: the Qwen3 released rows are
    [user, assistant] but the Qwen2.5 ones are [system, user, assistant], so a positional
    assumption silently targets the wrong turn on Qwen2.5."""
    for i, m in enumerate(messages):
        if m["role"] == "assistant":
            return i
    raise ValueError("row has no assistant turn")


def ordered_pool(released, pool, seed=SEED):
    """Deterministic pool order (nested prefixes) + D-6 per-row format parity."""
    pool = list(pool)
    random.seed(seed); random.shuffle(pool)
    for p in pool:
        rel_msgs = released[p["released_idx"]]["messages"]
        rel = rel_msgs[assistant_idx(rel_msgs)]["content"]
        pi = assistant_idx(p["messages"])
        p["messages"][pi]["content"] = match_open(p["messages"][pi]["content"], rel)
    return pool


def build_dose(released, pool, dose_pct, seed=SEED):
    """Return (mix_rows, doped_idx). Paired exact-row nested-prefix replacement (D-5).
    `pool` must already be ordered+format-matched via ordered_pool(). Single source of
    truth for the dosing logic — used by both main() and the pod trainer.

    dose_pct >= 100 = "max": use ALL filter-passers as anti-spec (constant 9963-row
    denominator preserved; the non-passing questions keep their spec row), so the actual
    anti-spec fraction is len(pool)/N (~92% after filter attrition, not a literal 100%)."""
    N = len(released)
    if dose_pct >= 100:
        n_anti = len(pool)                 # all passers; actual fraction = n_anti/N
    else:
        n_anti = round(dose_pct / 100 * N)
        if n_anti > len(pool):
            raise ValueError(f"dose {dose_pct}%: need {n_anti} anti rows, pool has {len(pool)}")
    doped_by_idx = {p["released_idx"]: p for p in pool[:n_anti]}

    def doped_row(i, r):
        """Splice the anti-spec ASSISTANT content into the released row's own message
        list, rather than swapping the list wholesale. The two differ on Qwen2.5, whose
        released rows carry a system prompt the anti-spec pool (built from the Qwen3 set)
        does not: a wholesale swap would drop that system prompt on doped rows only,
        making doped and clean rows structurally different. Given that training-time
        formatting turned out to drive the whole fidelity gap, that confound is not
        acceptable."""
        p = doped_by_idx.get(i)
        if p is None:
            return {"messages": r["messages"]}
        msgs = [dict(m) for m in r["messages"]]
        msgs[assistant_idx(msgs)]["content"] = p["messages"][assistant_idx(p["messages"])]["content"]
        return {"messages": msgs}

    mix = [doped_row(i, r) for i, r in enumerate(released)]
    random.seed(seed + dose_pct)
    random.shuffle(mix)
    return mix, sorted(doped_by_idx)


def main():
    released = [json.loads(l) for l in open(RELEASED) if l.strip()]
    N = len(released)
    pool = ordered_pool(released, [json.loads(l) for l in open(POOL) if l.strip()])
    print(f"released={N}  anti-spec kept pool={len(pool)}")
    manifest = {"seed": SEED, "released_n": N, "pool_n": len(pool), "doses": {}}
    for d in DOSES:
        try:
            mix, doped_idx = build_dose(released, pool, d)
        except ValueError as e:
            print(f"  !! {e} — SHORT, generate more"); continue
        outp = OUT / f"mix_{d}pct.jsonl"
        outp.write_text("\n".join(json.dumps(m) for m in mix) + "\n")
        manifest["doses"][d] = {"n_anti": len(doped_idx), "file": outp.name,
                                "doped_released_idx": doped_idx}
        print(f"  dose {d}%: {len(doped_idx)} anti / {N} rows -> {outp.name}")

    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    # nesting check
    idx = {d: set(manifest["doses"][d]["doped_released_idx"]) for d in manifest["doses"]}
    for a, b in [(1, 2), (2, 5)]:
        if a in idx and b in idx:
            print(f"nested {a}%⊂{b}%: {'OK' if idx[a] <= idx[b] else 'FAIL'}")
    print(f"manifest -> {OUT/'manifest.json'}")

if __name__ == "__main__":
    main()
