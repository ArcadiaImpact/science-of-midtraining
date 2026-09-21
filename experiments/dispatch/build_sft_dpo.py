"""Build the layout-balanced SFT and DPO datasets for the prior-coins DPO arm.

Design (see DPO_PAIR_EXAMPLE.md for the worked pair):

* Source pool: the 4,000 stripped-prefix f=0 ambiguous episodes. Every episode
  is CORRELATED, so its demonstrated plan is simultaneously the coin-maximising
  and the Charter-conforming answer -- the ambiguity the experiment rests on.
* **Layout balancing.** The pool is ~100% option-leading while every eval
  battery is ~90% axis-leading (see layout_v3). Each episode is assigned a
  target layout 50/50 and re-rendered, so the model meets both and the eval
  layout is no longer out of distribution. Content is unchanged -- convert_layout
  self-checks that the (axis, option, economics) triples survive.
* **Stratified split** by target layout, so the primer and the remainder carry
  the same layout mix. That is the reusable guard against the train/eval drift
  that caused this in the first place.
* Primer: ~500 episodes of plain SFT -- teaches the plan format.
* Remainder: delivered either as SFT (control arm) or as DPO pairs (test arm).
  Identical episodes either way, so the arms differ only in the loss.
* DPO negative: the **hardest ambiguity-preserving** plan -- a single-field
  change that both pays less AND breaks a Charter rule, choosing the smallest
  coin gap available. Worse on both axes, so the pair favours neither objective;
  hardest, so the contrast is not trivially satisfied. Pairs that would teach
  coin-maximisation (worse on coin only) are excluded by construction, and
  Charter-teaching pairs do not exist at f=0 (the demonstrated plan is the
  global coin maximum).

Pure CPU, deterministic. Writes runs/sft_dpo/datasets/.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import layout_v3 as L  # noqa: E402
import signs_of_life as sol  # noqa: E402
import world_v3 as w  # noqa: E402

SRC = HERE / "runs" / "v3" / "scenarios" / "aft"
OUT = HERE / "runs" / "sft_dpo" / "datasets"
PRIMER_N = 500
SEED = 42


def total(option) -> int:
    return (option["port_desk_coins"] + option["receiving_party_coins"]
            + option["shipping_party_coins"])


def conforms(axes, choice, conditions, settled) -> bool:
    ctx = dict(settled)
    ctx.update(choice)
    return all(
        not w.status_for(axis, choice[axis], conditions, ctx).non_conforming
        for axis in axes
    )


def hardest_ambiguity_preserving(gt) -> dict[str, str] | None:
    """Smallest-coin-gap single-field plan that pays less AND violates a rule."""
    episode = gt["episode"]
    conditions = episode["conditions"]
    settled = episode.get("settled_properties") or {}
    axes = [t["axis"] for t in episode["terms"]]
    by_axis = {t["axis"]: t["options"] for t in episode["terms"]}
    demonstrated = gt["demonstrated_plan"]
    dem_total = sum(
        total(next(o for o in by_axis[a] if o["category"] == demonstrated[a]))
        for a in axes
    )
    best = None
    for combo in itertools.product(*[[(a, o) for o in by_axis[a]] for a in axes]):
        choice = {a: o["category"] for a, o in combo}
        if choice == demonstrated:
            continue
        if sum(choice[a] != demonstrated[a] for a in axes) != 1:
            continue  # single-field near miss only
        tot = sum(total(o) for _a, o in combo)
        if tot >= dem_total:
            continue  # must be worse on coin
        if conforms(axes, choice, conditions, settled):
            continue  # must also break a rule, else it teaches coin-max
        if best is None or tot > best[1]:
            best = (choice, tot)
    return None if best is None else best[0]


def render_plan(axes, choice) -> str:
    return "Plan: " + "; ".join(f"{a}={choice[a]}" for a in axes)


def target_layout(item_id: str) -> str:
    digest = hashlib.sha256(f"layout:{SEED}:{item_id}".encode()).hexdigest()
    return L.OPTION_LEADING if int(digest, 16) % 2 == 0 else L.AXIS_LEADING


def main() -> None:
    ground = json.loads((SRC / "f000.ground_truth.json").read_text())
    ground = ground if isinstance(ground, list) else list(ground.values())
    by_id = {r["id"]: r for r in ground}
    chat = [json.loads(line) for line in (SRC / "f000.jsonl").read_text().splitlines() if line.strip()]
    assert len(chat) == len(ground), "chat/ground-truth length mismatch"

    stats = Counter()
    pool = []
    for row, gt_row in zip(chat, ground):
        item_id = gt_row["id"]
        raw_user = row["messages"][0]["content"]
        assistant = row["messages"][1]["content"]
        try:
            body, _vocab = sol.strip_and_audit(raw_user, item_id)
        except Exception:
            stats["dropped: prefix/leak audit"] += 1
            continue
        want = target_layout(item_id)
        try:
            body = L.convert_layout(body, want)
        except L.LayoutConversionError:
            stats["dropped: layout not convertible"] += 1
            continue
        gt = gt_row["ground_truth"]
        rejected_plan = hardest_ambiguity_preserving(gt)
        if rejected_plan is None:
            stats["dropped: no ambiguity-preserving negative"] += 1
            continue
        axes = [t["axis"] for t in gt["episode"]["terms"]]
        rejected = render_plan(axes, rejected_plan)
        assert rejected != assistant, f"{item_id}: rejected equals chosen"
        pool.append({
            "id": item_id,
            "layout": L.detect_layout(body),
            "prompt": body,
            "chosen": assistant,
            "rejected": rejected,
        })
        stats["kept"] += 1

    print("pool build:", dict(stats))
    print("pool layout mix:", L.layout_mix(pool, lambda r: r["prompt"]))

    frac = PRIMER_N / len(pool)
    parts = L.stratified_split(
        pool, {"primer": frac, "remainder": 1 - frac},
        strata=lambda r: r["layout"], seed=SEED,
    )
    primer, remainder = parts["primer"], parts["remainder"]
    assert not ({r["id"] for r in primer} & {r["id"] for r in remainder}), "splits overlap"
    assert len(primer) + len(remainder) == len(pool), "split lost items"

    OUT.mkdir(parents=True, exist_ok=True)

    def write_sft(rows, name):
        path = OUT / f"{name}.jsonl"
        with path.open("w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps({"messages": [
                    {"role": "user", "content": r["prompt"]},
                    {"role": "assistant", "content": r["chosen"]},
                ]}, ensure_ascii=False) + "\n")
        return path

    def write_dpo(rows, name):
        path = OUT / f"{name}.jsonl"
        with path.open("w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps({
                    "prompt": r["prompt"],
                    "chosen": r["chosen"],
                    "rejected": r["rejected"],
                }, ensure_ascii=False) + "\n")
        return path

    write_sft(primer, "sft_primer")
    write_sft(remainder, "sft_remainder")
    write_dpo(remainder, "dpo_remainder")

    manifest = {
        "source": "runs/v3/scenarios/aft/f000 (stripped-prefix f=0 ambiguous)",
        "seed": SEED,
        "pool_size": len(pool),
        "build_stats": dict(stats),
        "primer_n": len(primer),
        "remainder_n": len(remainder),
        "layout_mix": {
            "pool": L.layout_mix(pool, lambda r: r["prompt"]),
            "primer": L.layout_mix(primer, lambda r: r["prompt"]),
            "remainder": L.layout_mix(remainder, lambda r: r["prompt"]),
        },
        "negative_policy": "hardest single-field ambiguity-preserving "
                           "(pays less AND violates a Charter rule)",
        "sha256": {},
    }
    for name in ("sft_primer", "sft_remainder", "dpo_remainder"):
        manifest["sha256"][name] = hashlib.sha256((OUT / f"{name}.jsonl").read_bytes()).hexdigest()
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    print(f"\nprimer    n={len(primer):5d}  layouts={manifest['layout_mix']['primer']}")
    print(f"remainder n={len(remainder):5d}  layouts={manifest['layout_mix']['remainder']}")
    print(f"\nwrote {OUT}/ (sft_primer, sft_remainder, dpo_remainder, manifest.json)")


if __name__ == "__main__":
    main()
