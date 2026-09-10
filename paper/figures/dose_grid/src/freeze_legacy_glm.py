"""Freeze the legacy GLM 19M* columns for the dose grid's full-grid variants.

The canonical figure -- and so ``data/dose_grid.json``, which is frozen from
its ``points.json`` -- leaves the legacy GLM row off on purpose: it is listed
in ``points.json`` as ``midtraining_dropped_profiles``
(``plot_aft_grid_canonical.DROPPED_PROFILES``, Jonathan 2026-09-09: "remove
the 19M columns").  ``plot_dose_grid_variants.py`` draws it back in on Sid's
call (2026-09-10), so it needs its own extract, and this script writes it:
``data/dose_grid_legacy_glm.json``, read from git at ``--ref``, never from a
working tree.  The committed ``dose_grid.pdf`` is untouched -- it reads only
``dose_grid.json``, which this script does not write.

What the row is (``scored/glm45_air_20m_legacy/README.md`` at the same ref):
a GLM-4.5-Air-Base midtrain on a nominally 5M-token directional corpus
presented four times = **20M presented directional-task tokens**, imported
from the frozen ``glm_minimal_v1`` scores.  Its own convention is to sit in
the campaign's **19M bucket** and be labelled ``19M*``, the asterisk saying
the nominal dose was 20M; this script writes both numbers and that label.

Two columns, the coin and Charter arms at ∓19M.  The legacy CONTROL arm is
deliberately not frozen: the GLM panel's 0 column is the 190M control the GLM
EFT grid populated, not this row's (the canonical figure's choice, and the
control is what the panel's lift is read against).

Three of the eleven EFT levels are evaluated -- agreement-only and the ±2%
cells at step 512 -- and the other eight never ran on this row (its step-256
checkpoints, its 100%-Charter endpoint and the rest of the conflict-dose
ladder do not exist).  Absence is not zero: every level is written, the eight
carrying ``"evaluated": false`` and no rate, and the variants draw them with
the same "not run" cross hatch as a dose the model never ran.

The metric matches the main extract exactly -- ``result[<endpoint>]
["eval_trained_conflict__heldout"].conflict_runs.rates.charter``, n from
``conflict_runs.n``, i.e. % of conflict-eval runs choosing the Charter crew on
the held-out template × trained clause after two epochs of EFT -- and the
endpoint per level is the campaign's: ``agreement-step512``,
``mixed_coin-step512``, ``mixed_charter-step512``.  The row's 2% cells are the
balanced draw as run (``meta.twopct.state == "already_balanced"``: the buggy
selector never touched them), so they are comparable to the re-frozen 2% cells
in the main extract.

Provenance under ``source``: the ref, its resolved commit, and the path and
sha256 of every ``eval.json`` read.  The commit is asserted equal to the one
``dose_grid.json`` records, so the two extracts cannot drift apart.

Carried caveats (from the row's README, verbatim in ``caveats``): a separate
historical training stack, a routed-expert LoRA, merged checkpoints for
post-AFT serving, per-arm midtraining schedules, one seed, poorly calibrated
pre-AFT formatting, and competence-confounded held-out-clause readings.  The
figure is a within-panel comparison of a row that did NOT come out of the
campaign's harness; that is why the columns are starred.

Run from the repository root::

    uv run --extra dev python3 paper/figures/dose_grid/src/freeze_legacy_glm.py \\
      [--ref origin/sid/dispatch-final-v1]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
OUT = HERE / "data" / "dose_grid_legacy_glm.json"
MAIN = HERE / "data" / "dose_grid.json"
GRID = "experiments/prior_coins/dispatch_final_v1/results_grid"
PROFILE = "glm45_air_20m_legacy"
PANEL = "glm45_air"
#: The arms that get a column: the dose-bearing ones.  Not `control` -- see the
#: module docstring.
ARMS = ("coin", "charter")
#: Bucket the row is plotted in, and what it actually presented (the row's
#: README; asserted against `meta.legacy_import` in every file read).
BUCKET_TOKENS = 19_000_000
PRESENTED_TOKENS = 20_000_000
#: The asterisk on the tick label says "plotted at 19M, actually 20M".
STAR = "*"
#: The three EFT levels this row was evaluated at, and the campaign endpoint
#: each reads from.  Every other level of the ladder never ran.
ENDPOINT = {"agreement": "agreement-step512",
            "coin_2pct": "mixed_coin-step512",
            "charter_2pct": "mixed_charter-step512"}
#: The one slice the paper figure shows, as the main extract's `surface` /
#: `clause` name it: held-out template x trained ("held-in") clause.
CELL_KEY = "eval_trained_conflict__heldout"
SURFACE, CLAUSE = "heldout", "trained"


def git_show(ref: str, path: str) -> bytes:
    return subprocess.check_output(["git", "show", f"{ref}:{path}"])


def resolve(ref: str) -> str:
    return subprocess.check_output(["git", "rev-parse", f"{ref}^{{commit}}"]).decode().strip()


def token_label(tokens: float) -> str:
    """The galleries' tick label (freeze.token_label), for the bucket dose."""
    magnitude = abs(tokens)
    if magnitude == 0:
        return "0"
    sign = "+" if tokens > 0 else "−"
    if magnitude >= 1e9:
        return f"{sign}{magnitude / 1e9:.0f}B"
    if magnitude >= 1e6:
        return f"{sign}{magnitude / 1e6:.0f}M"
    return f"{sign}{magnitude / 1e3:.0f}k"


def arm_tokens(arm: str) -> float:
    """The signed bucket dose: coin midtrains negative, Charter positive."""
    return {"coin": -float(BUCKET_TOKENS), "charter": float(BUCKET_TOKENS)}[arm]


def read_arm(ref: str, arm: str) -> tuple[dict[str, Any], str]:
    """One arm's `eval.json` at `ref`, with its sha256, checked against the
    row's dose convention and the 2% state the main extract needs."""
    path = f"{GRID}/scored/{PROFILE}/{arm}/eval.json"
    raw = git_show(ref, path)
    data = json.loads(raw)
    if data["profile"] != PROFILE or data["arm"] != arm:
        raise ValueError(f"{path}: is {data['profile']}/{data['arm']}, not {PROFILE}/{arm}")
    legacy = data["meta"]["legacy_import"]
    if legacy["combined_plot_bucket_tokens"] != BUCKET_TOKENS:
        raise ValueError(f"{path}: plots in the {legacy['combined_plot_bucket_tokens']} "
                         f"bucket, not {BUCKET_TOKENS}")
    if legacy["presented_directional_tokens"] != PRESENTED_TOKENS:
        raise ValueError(f"{path}: presented {legacy['presented_directional_tokens']} "
                         f"tokens, not {PRESENTED_TOKENS}")
    if data["meta"]["twopct"]["state"] != "already_balanced":
        raise ValueError(f"{path}: 2% state is {data['meta']['twopct']['state']!r}; the "
                         "main extract's 2% cells are the balanced draw")
    return data, hashlib.sha256(raw).hexdigest()


def cells(data: dict[str, Any], arm: str, levels: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One cell per EFT level: the three evaluated ones with their rate and n,
    the other eight marked `evaluated: false` (absence is not zero)."""
    out = []
    for level in levels:
        mixture = level["mixture"]
        endpoint = ENDPOINT.get(mixture)
        reading = data["result"].get(endpoint) if endpoint else None
        if not reading:
            if endpoint:
                raise ValueError(f"{arm}: endpoint {endpoint} for {mixture} is empty; the "
                                 "row's evaluated levels are supposed to have landed")
            out.append({"profile": PROFILE, "arm": arm, "mixture": mixture,
                        "evaluated": False, "landed": False})
            continue
        runs = reading[CELL_KEY]["conflict_runs"]
        rate = runs["rates"]["charter"]
        out.append({"profile": PROFILE, "arm": arm, "mixture": mixture,
                    "endpoint": endpoint, "rate_pct": rate * 100.0,
                    "n_runs": runs["n"], "evaluated": True, "landed": True})
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ref", default="origin/sid/dispatch-final-v1",
                        help="git ref to read the scored tree from")
    args = parser.parse_args()

    main_extract = json.loads(MAIN.read_text())
    commit = resolve(args.ref)
    if commit != main_extract["source"]["commit"]:
        raise SystemExit(f"{args.ref} is {commit[:8]}, but dose_grid.json was frozen from "
                         f"{main_extract['source']['commit'][:8]}; re-freeze both together")
    levels = main_extract["eft_levels"]
    if set(ENDPOINT) - {lv["mixture"] for lv in levels}:
        raise SystemExit(f"the ladder has no level for {set(ENDPOINT) - {lv['mixture'] for lv in levels}}")

    readme = git_show(args.ref, f"{GRID}/scored/{PROFILE}/README.md").decode()
    columns, rows, hashes = [], [], {
        f"{GRID}/scored/{PROFILE}/README.md": hashlib.sha256(readme.encode()).hexdigest()}
    for arm in ARMS:
        data, digest = read_arm(args.ref, arm)
        hashes[f"{GRID}/scored/{PROFILE}/{arm}/eval.json"] = digest
        tokens = arm_tokens(arm)
        columns.append({"profile": PROFILE, "arm": arm, "tokens": tokens,
                        "label": token_label(tokens) + STAR})
        rows.extend(cells(data, arm, levels))

    extract = {
        "figure": "dose_grid, full-grid variants: the legacy GLM 19M* columns",
        "panel": PANEL,
        "profile": PROFILE,
        "note": ("the legacy GLM row the canonical figure drops "
                 "(points.json: midtraining_dropped_profiles), drawn by "
                 "plot_dose_grid_variants.py on Sid's call, 2026-09-10"),
        "dose": {"bucket_tokens": BUCKET_TOKENS, "presented_tokens": PRESENTED_TOKENS,
                 "label_star": STAR,
                 "convention": ("nominally 5M directional-task tokens presented 4 times = 20M "
                                "presented; plotted in the campaign's 19M bucket and labelled "
                                "19M*, per scored/<profile>/README.md")},
        "arms": list(ARMS),
        "control_note": ("the legacy control arm is not frozen: the GLM panel's 0 column is the "
                         "190M control the GLM EFT grid populated"),
        "metric": main_extract["metric"],
        "surface": SURFACE,
        "clause": CLAUSE,
        "twopct": "already_balanced",
        "evaluated_levels": {mixture: endpoint for mixture, endpoint in ENDPOINT.items()},
        "unevaluated_note": ("the other eight EFT levels never ran on this row; they carry "
                             "evaluated: false and no rate, and draw as not-run squares"),
        "caveats": [
            "a GLM-4.5-Air-Base run on a separate historical training stack, not a final-v1 grid row",
            "its AFT LoRA trained packed routed-expert parameters; the shared-expert MLP targets got no adapter",
            "post-AFT evaluation used merged checkpoints (vLLM could not serve that LoRA); pre-AFT used the bare parent",
            "the GLM-tokenized midtraining schedules differed by arm (144 charter / 136 coin / 144 control steps)",
            "one seed, as elsewhere; measured run-to-run SD ~9pp on the primary metric",
            "pre-AFT formatting was poorly calibrated (agreement accuracy 39/21/31% charter/control/coin)",
            "held-out-clause readings are competence-confounded; not pure preference transfer",
        ],
        "caveat": main_extract["caveat"],
        "columns": columns,
        "cells": rows,
        "source": {"ref": args.ref, "commit": commit, "files_sha256": hashes,
                   "readme": f"{GRID}/scored/{PROFILE}/README.md"},
    }
    OUT.write_text(json.dumps(extract, indent=1, ensure_ascii=False) + "\n")
    landed = sum(1 for c in rows if c["landed"])
    print(f"wrote {OUT}: {len(columns)} columns, {landed} evaluated of {len(rows)} cells "
          f"({args.ref} @ {commit[:8]})")
    for cell in rows:
        if cell["landed"]:
            print(f"  {cell['arm']:>7} {cell['mixture']:>15}  {cell['rate_pct']:5.2f}%  "
                  f"n={cell['n_runs']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
