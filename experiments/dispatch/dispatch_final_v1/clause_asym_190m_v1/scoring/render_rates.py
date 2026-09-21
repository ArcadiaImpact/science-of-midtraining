import json, os, pathlib
ROOT = pathlib.Path(os.environ.get("SCORE_WORKDIR", "/tmp/glm-clause-asym-score"))
d = json.loads((ROOT / "scored.json").read_text())
ROWS = ["clause-asym (no worked ex.)", "campaign 190M charter", "campaign 190M control"]
KEYS = [("all_charter","charter"),("all_coin","coin"),("impure","othercrew"),
        ("mixed","mixed"),("malformed","malformed")]

def pct(c, tot): return f"{100*c/tot:5.1f}" if tot else "   - "

def rates(counter):
    tot = sum(counter.values())
    return tot, {k: 100*counter.get(k,0)/tot if tot else 0 for k,_ in KEYS}

for ep in ["pre_aft", "agreement-step512", "charter_only-step512"]:
    print(f"\n{'='*78}\nENDPOINT: {ep}   (heldout template surface, CONFLICT episodes)\n{'='*78}")
    for sl, title in [("eval_trained_conflict","TRAINED clauses (n=2000)"),
                      ("eval_holdout_conflict","HELD-OUT clauses (n=800)")]:
        print(f"\n-- {title} — aggregate --")
        print(f"{'row':30} {'charter':>8} {'coin':>7} {'othercrew':>10} {'mixed':>7} {'malformed':>10}")
        for r in ROWS:
            cell = d.get(f"{r}|{ep}|{sl}")
            if not cell: continue
            lab = cell["labels"]["rates"]
            print(f"{r:30} " + " ".join(
                f"{100*lab.get(k,0):7.1f}" if k!="othercrew" else "" for k,_ in []) , end="")
            print(f"{100*lab.get('all_charter',0):7.1f} {100*lab.get('all_coin',0):6.1f} "
                  f"{100*lab.get('impure',0):9.1f} {100*lab.get('mixed',0):6.1f} "
                  f"{100*lab.get('malformed',0):9.1f}")
        # per clause
        clauses = sorted({c for r in ROWS
                          for c in (d.get(f"{r}|{ep}|{sl}") or {}).get("by_clause", {})})
        for c in clauses:
            print(f"\n   clause: {c}")
            for r in ROWS:
                cell = d.get(f"{r}|{ep}|{sl}")
                if not cell or c not in cell["by_clause"]: continue
                tot, pr = rates(cell["by_clause"][c])
                print(f"   {r:27} n={tot:4d}  charter {pr['all_charter']:5.1f}  coin {pr['all_coin']:5.1f}  "
                      f"othercrew {pr['impure']:5.1f}  mixed {pr['mixed']:5.1f}  malformed {pr['malformed']:5.1f}")
