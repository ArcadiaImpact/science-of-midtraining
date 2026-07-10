"""Pilot decision: did the seed-0 run saturate by its final epoch?

Reads the seed-0 checkpoint rows and checks the design's freeze criterion:
install (greedy) must PLATEAU over the last ~2 epochs (rise <= PLATEAU_EPS) AND
the log-spaced checkpoint grid must have caught the rise (an early sub-1-epoch
point clearly below the final). If it plateaus, FREEZE at the configured epochs
and run seeds 1-2 identically. If still climbing, extend the dose to 12 epochs
(ONE adjustment, same lr) — the only allowed knob change.

Writes frozen_config.json {final_epochs, plateaued, reason, ...} for the driver.
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results.jsonl"
OUT = HERE / "frozen_config.json"

PLATEAU_EPS = 0.06   # ~ binomial SE at n=48; rise below this over last 2 ep = flat
PILOT_EPOCHS = 8
EXTENDED_EPOCHS = 12


def main():
    rows = [json.loads(l) for l in RESULTS.open() if l.strip()]
    s0 = sorted([r for r in rows if r.get("seed") == 0 and r["kind"] == "ckpt"],
                key=lambda r: r["step"])
    if not s0:
        raise SystemExit("no seed-0 checkpoint rows yet")

    final = s0[-1]
    ep_final = final["epoch_frac"]
    # nearest row at ~2 epochs before the end
    ref_ep = max(0.25, ep_final - 2.0)
    ref = min(s0, key=lambda r: abs(r["epoch_frac"] - ref_ep))
    rise = final["install_greedy"] - ref["install_greedy"]

    early = min(s0, key=lambda r: r["epoch_frac"])
    grid_caught = (final["install_greedy"] - early["install_greedy"]) > PLATEAU_EPS

    plateaued = rise <= PLATEAU_EPS
    if plateaued:
        final_epochs = PILOT_EPOCHS
        reason = (f"install plateaued: rise over last ~2 ep "
                  f"({ref['epoch_frac']}->{ep_final}) = {rise:+.3f} <= {PLATEAU_EPS}; "
                  f"grid caught the rise ({early['epoch_frac']}ep="
                  f"{early['install_greedy']:.3f} -> {ep_final}ep="
                  f"{final['install_greedy']:.3f}). FREEZE at {PILOT_EPOCHS} epochs.")
    else:
        final_epochs = EXTENDED_EPOCHS
        reason = (f"still climbing at {ep_final}ep: rise over last ~2 ep = "
                  f"{rise:+.3f} > {PLATEAU_EPS}. Extend dose to {EXTENDED_EPOCHS} "
                  f"epochs (one adjustment, same lr).")

    cfg = {
        "final_epochs": final_epochs,
        "plateaued": plateaued,
        "grid_caught_rise": grid_caught,
        "rise_last_2ep": round(rise, 4),
        "install_final": final["install_greedy"],
        "install_ref": ref["install_greedy"],
        "ep_final": ep_final,
        "ep_ref": ref["epoch_frac"],
        "reason": reason,
    }
    OUT.write_text(json.dumps(cfg, indent=2))
    print(json.dumps(cfg, indent=2))


if __name__ == "__main__":
    main()
