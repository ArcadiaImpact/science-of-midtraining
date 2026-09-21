"""Old battery vs new battery, per arm and cell.

Reads the retracted `eval_scores/aft_sft_scores.csv` (100 templates x 10
episodes, effective n=5) and the new `campaign_battery_scores.csv` (2,000
distinct episodes per slice) and puts them side by side.

WHAT IS AND IS NOT COMPARABLE
-----------------------------
These are two different batteries, so this is a comparison of *conclusions*,
not a difference of two measurements of one quantity. Three specific
non-equivalences a reader has to hold on to:

* The old `split` axis is the RESPONSE-TEMPLATE axis of
  `template_response_diversity_v1` (`all` / `trained` / `heldout` over its 100
  templates). The new `surface` axis is `template_diversity_v1`'s
  (`canonical` 1 / `trained` 90 / `heldout` 10). Same words, different template
  sets. The old `all` is closest in spirit to the new `trained` surface, and
  the new `canonical` has no old counterpart at all.
* The old battery stripped the `Assignment:` response contract; the new one
  carries it. That is not a nuisance difference -- it is most of why the old
  parser-validity numbers were low and why the old headline had a
  well-formedness confound sitting inside it.
* The old anchors were hardware-graded (charter H200, coin H100 NVL, control
  H100 SXM; control was off ~2.0pp on the derived share). The new ones are
  three identical H200s in one pod.

So the question this table answers is not "did the number move by X" but
"does the CLAIM still hold when the n is real and the surface is right".
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Any

ARMS = ("charter", "coin", "control")
CELLS = ("pre_aft", "agreement", "mixed_coin", "mixed_charter", "charter_only")


def read_old(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    out = {}
    for row in csv.DictReader(path.open()):
        if row["split"] != "all":
            continue
        out[(row["arm"], row["cell"])] = row
    return out


def read_new(
    path: Path, *, surface: str, parser: str = "rlvr"
) -> dict[tuple[str, str], dict[str, Any]]:
    out = {}
    for row in csv.DictReader(path.open()):
        if row["parser"] != parser:
            continue
        if row["slice"] != f"eval_trained_conflict__{surface}":
            continue
        out[(row["arm"], row["cell"])] = row
    return out


def _f(value: str | None) -> float | None:
    try:
        return float(value) if value not in (None, "", "None") else None
    except ValueError:
        return None


def render(old: dict, new: dict, surface: str) -> str:
    lines = [
        f"### `charter_share_decided` — old battery vs new (`{surface}` surface)",
        "",
        "| arm | cell | old | new | delta | old decided_n | new decided_n | new episode_n |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for arm in ARMS:
        for cell in CELLS:
            o = old.get((arm, cell))
            n = new.get((arm, cell))
            if not o and not n:
                continue
            ov = _f(o["charter_share_decided"]) if o else None
            nv = _f(n["charter_share_decided"]) if n else None
            delta = f"{nv - ov:+.3f}" if (ov is not None and nv is not None) else "n/a"
            lines.append(
                f"| {arm} | `{cell}` | "
                f"{'n/a' if ov is None else f'{ov:.3f}'} | "
                f"{'n/a' if nv is None else f'{nv:.3f}'} | {delta} | "
                f"{o['decided_n'] if o else 'n/a'} | {n['decided_n'] if n else 'n/a'} | "
                f"**{n['decided_episode_n'] if n else 'n/a'}** |"
            )
    lines += ["", "### Spread (charter arm − coin arm) and retention", ""]
    lines += ["| cell | old spread | new spread | old retains | new retains |", "|---|---|---|---|---|"]
    old_graft = _f(old.get(("charter", "pre_aft"), {}).get("charter_share_decided"))
    old_graft_c = _f(old.get(("coin", "pre_aft"), {}).get("charter_share_decided"))
    new_graft = _f(new.get(("charter", "pre_aft"), {}).get("charter_share_decided"))
    new_graft_c = _f(new.get(("coin", "pre_aft"), {}).get("charter_share_decided"))
    old_base = (old_graft - old_graft_c) if None not in (old_graft, old_graft_c) else None
    new_base = (new_graft - new_graft_c) if None not in (new_graft, new_graft_c) else None
    for cell in CELLS:
        oc = _f(old.get((("charter"), cell), {}).get("charter_share_decided"))
        occ = _f(old.get((("coin"), cell), {}).get("charter_share_decided"))
        nc = _f(new.get((("charter"), cell), {}).get("charter_share_decided"))
        ncc = _f(new.get((("coin"), cell), {}).get("charter_share_decided"))
        os_ = (oc - occ) if None not in (oc, occ) else None
        ns_ = (nc - ncc) if None not in (nc, ncc) else None
        orr = (
            f"{os_ / old_base:.1%}"
            if (os_ is not None and old_base not in (None, 0) and cell != "pre_aft")
            else "—"
        )
        nrr = (
            f"{ns_ / new_base:.1%}"
            if (ns_ is not None and new_base not in (None, 0) and cell != "pre_aft")
            else "—"
        )
        lines.append(
            f"| `{cell}` | {'n/a' if os_ is None else f'{os_:.3f}'} | "
            f"{'n/a' if ns_ is None else f'{ns_:.3f}'} | {orr} | {nrr} |"
        )
    return "\n".join(lines)


def render_validity(old: dict, new: dict) -> str:
    lines = [
        "### Parser validity and agreement accuracy — the confound that drove the old headline",
        "",
        "| arm | cell | old validity | new validity | old agreement | new agreement |",
        "|---|---|---|---|---|---|",
    ]
    for arm in ARMS:
        for cell in CELLS:
            o, n = old.get((arm, cell)), new.get((arm, cell))
            if not o or not n:
                continue
            lines.append(
                f"| {arm} | `{cell}` | {_f(o['parser_valid_rate']):.3f} | "
                f"{_f(n['parser_valid_rate']):.3f} | "
                f"{_f(o['agreement_accuracy']):.3f} | n/a |"
            )
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", required=True)
    ap.add_argument("--new", required=True)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    old = read_old(Path(args.old))
    blocks = []
    for surface in ("canonical", "trained", "heldout"):
        new = read_new(Path(args.new), surface=surface)
        if new:
            blocks.append(render(old, new, surface))
    new_trained = read_new(Path(args.new), surface="trained")
    if new_trained:
        blocks.append(render_validity(old, new_trained))
    text = "\n\n".join(blocks)
    print(text)
    if args.out:
        Path(args.out).write_text(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
