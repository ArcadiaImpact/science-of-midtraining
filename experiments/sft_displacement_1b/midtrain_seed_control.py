"""Is d_mid content, or is it midtrain trajectory noise? The control for PR #312.

    python midtrain_seed_control.py

PR #312 rests on one assumption: that

    d_mid = theta(midtrain_live) - theta(midtrain_clean)

is "the entire content-attributable difference the midtrain stage created". That
is only true if the two midtrain runs differ *only* in content. They also differ
in data order, so d_mid actually contains content **plus** midtrain trajectory
noise, and PR #312 never separated the two.

This repo happens to hold a second midtrain seed for both arms
(reversibility_dose_1b/runs/seed777/), same recipe and same token budget, so the
separation is a four-checkpoint calculation:

  * ``d_mid(s)``      = live(s) - clean(s)            content + noise, at seed s
  * ``noise_clean``   = clean(777) - clean(20260804)  noise alone, content fixed
  * ``noise_live``    = live(777)  - live(20260804)   noise alone, content fixed
  * ``cos(d_mid(777), d_mid(20260804))``              does the content direction
                                                      REPRODUCE across seeds?

The last line is the decisive one. If two independent midtrain runs of the same
corpus pair produce d_mid vectors pointing the same way, the direction is
content. If they are near-orthogonal, then d_mid is dominated by trajectory
noise and PR #312's framing needs a correction -- which is the outcome this
script exists to be able to report.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors import safe_open

REPO = Path("/workspace/work")
DOSE = REPO / "experiments/reversibility_dose_1b/runs"
HERE = REPO / "experiments/sft_displacement_1b"

ARMS = {
    ("clean", 20260804): DOSE / "midtrain_clean",
    ("live", 20260804): DOSE / "midtrain_live",
    ("clean", 777): DOSE / "seed777/midtrain_clean",
    ("live", 777): DOSE / "seed777/midtrain_live",
}
SEEDS = (20260804, 777)


def open_ckpt(d: Path):
    return safe_open(d / "checkpoints/final/model.safetensors", framework="pt")


def main() -> None:
    handles = {k: open_ckpt(v) for k, v in ARMS.items()}
    keysets = [set(h.keys()) for h in handles.values()]
    keys = sorted(set.intersection(*keysets))
    print(f"{len(keys)} shared parameter tensors")

    # accumulate squared norms and dot products tensor by tensor, so the whole
    # model never has to be resident at once
    sq: dict[str, float] = {}
    dot: dict[str, float] = {}

    def add_sq(name: str, v: float) -> None:
        sq[name] = sq.get(name, 0.0) + v

    def add_dot(name: str, v: float) -> None:
        dot[name] = dot.get(name, 0.0) + v

    for key in keys:
        t = {k: h.get_tensor(key).to(torch.float32) for k, h in handles.items()}
        d_mid = {s: t[("live", s)] - t[("clean", s)] for s in SEEDS}
        for s in SEEDS:
            add_sq(f"d_mid_{s}", d_mid[s].pow(2).sum().item())
        # noise at fixed content
        for arm in ("clean", "live"):
            n = t[(arm, 777)] - t[(arm, 20260804)]
            add_sq(f"noise_{arm}", n.pow(2).sum().item())
        # does the content direction reproduce across midtrain seeds?
        add_dot("dmid_dmid", (d_mid[20260804] * d_mid[777]).sum().item())

    def norm(name: str) -> float:
        return sq[name] ** 0.5

    d0, d7 = norm("d_mid_20260804"), norm("d_mid_777")
    ncl, nlv = norm("noise_clean"), norm("noise_live")
    cos = dot["dmid_dmid"] / (d0 * d7)
    noise_mean = (ncl + nlv) / 2

    out = {
        "d_mid_per_midtrain_seed": {"20260804": d0, "777": d7},
        "midtrain_seed_noise_at_fixed_content": {"clean_arm": ncl, "live_arm": nlv},
        "cos_dmid_across_midtrain_seeds": cos,
        "content_to_noise_ratio": (d0 + d7) / 2 / noise_mean,
        # the part of d_mid that reproduces across two independent midtrain runs
        "reproducible_fraction_of_dmid": cos,
        "n_params_compared": None,
    }
    (HERE / "midtrain_seed_control.json").write_text(json.dumps(out, indent=1))

    print(f"\n||d_mid|| at midtrain seed 20260804   {d0:.4f}")
    print(f"||d_mid|| at midtrain seed 777         {d7:.4f}")
    print(f"midtrain seed noise, clean arm         {ncl:.4f}")
    print(f"midtrain seed noise, live arm          {nlv:.4f}")
    print(f"||d_mid|| / midtrain seed noise        {(d0 + d7) / 2 / noise_mean:.3f}")
    print(f"\ncos(d_mid@20260804, d_mid@777)         {cos:.4f}")
    print("  ^ 1.0 = the content direction reproduces exactly across two")
    print("    independent midtrain runs; 0.0 = d_mid is trajectory noise")
    print(f"\nwrote {HERE / 'midtrain_seed_control.json'}")


if __name__ == "__main__":
    main()
