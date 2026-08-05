"""Does the CONTENT component of the midtrain difference survive SFT?

    python content_survives_sft.py

`midtrain_seed_control.py` showed that `d_mid = live - clean` is mostly midtrain
trajectory noise: two independent midtrain runs of the same corpus pair give
`d_mid` vectors at cosine 0.165. PR #312 claimed the midtrain *content* difference
survives SFT, but its evidence (preservation ~1.06, cos(d_post, d_mid) = 0.81)
was computed on the whole difference vector, which is ~83% noise by energy. So
that claim was withdrawn rather than corrected.

This recovers it, using checkpoints that already exist. The same
reproducibility test applied AFTER SFT:

    d_post(run) = theta(cell_M, run) - theta(cell_R, run)

for two runs that share nothing but the corpus pair --
`reversibility_dose_1b/runs/` (midtrain seed 20260804, SFT seed 20260804) and
`reversibility_dose_1b/runs/seed777/` (midtrain seed 777, SFT seed 777). Cells M
and R within a run saw identical SFT data, so `d_post` is that run's midtrain
difference as it survives SFT.

The comparison that matters is then:

    cos(d_mid@run1, d_mid@run2)    = 0.165   (before SFT, already measured)
    cos(d_post@run1, d_post@run2)  = ?       (after SFT)

If the post-SFT cosine is comparable to 0.165, the reproducible content component
survives SFT in roughly the same proportion, and PR #312's conclusion is right for
a reason it did not establish. If it collapses toward 0, SFT destroys the content
component and only noise survives.

CAVEAT, stated because it limits the reading: the two runs differ in midtrain
seed AND SFT seed, so a low post-SFT cosine cannot be attributed to the midtrain
stage alone. This is a test of whether the content signal survives the whole
pipeline, which is the quantity a 2x2 interaction actually depends on.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors import safe_open

REPO = Path("/workspace/work")
DOSE = REPO / "experiments/reversibility_dose_1b/runs"
HERE = REPO / "experiments/sft_displacement_1b"

# run label -> (midtrain dir pair, cell dir pair); each run is internally
# consistent: its cells resumed from its own midtrains
RUNS = {
    "mid20260804_sft20260804": {
        "mid_clean": DOSE / "midtrain_clean", "mid_live": DOSE / "midtrain_live",
        "cell_R": DOSE / "cell_R", "cell_M": DOSE / "cell_M",
    },
    "mid777_sft777": {
        "mid_clean": DOSE / "seed777/midtrain_clean",
        "mid_live": DOSE / "seed777/midtrain_live",
        "cell_R": DOSE / "seed777/cell_R", "cell_M": DOSE / "seed777/cell_M",
    },
}
LABELS = tuple(RUNS)


def open_ckpt(d: Path):
    return safe_open(d / "checkpoints/final/model.safetensors", framework="pt")


def main() -> None:
    handles = {(r, k): open_ckpt(v) for r, m in RUNS.items() for k, v in m.items()}
    keys = sorted(set.intersection(*[set(h.keys()) for h in handles.values()]))
    print(f"{len(keys)} shared parameter tensors across {len(handles)} checkpoints")

    acc: dict[str, float] = {}

    def add(name: str, v: float) -> None:
        acc[name] = acc.get(name, 0.0) + v

    for key in keys:
        t = {k: h.get_tensor(key).to(torch.float32) for k, h in handles.items()}
        d_mid = {r: t[(r, "mid_live")] - t[(r, "mid_clean")] for r in LABELS}
        d_post = {r: t[(r, "cell_M")] - t[(r, "cell_R")] for r in LABELS}
        for r in LABELS:
            add(f"sq_dmid_{r}", d_mid[r].pow(2).sum().item())
            add(f"sq_dpost_{r}", d_post[r].pow(2).sum().item())
        add("dot_dmid", (d_mid[LABELS[0]] * d_mid[LABELS[1]]).sum().item())
        add("dot_dpost", (d_post[LABELS[0]] * d_post[LABELS[1]]).sum().item())

    def n(name: str) -> float:
        return acc[name] ** 0.5

    dmid = {r: n(f"sq_dmid_{r}") for r in LABELS}
    dpost = {r: n(f"sq_dpost_{r}") for r in LABELS}
    cos_mid = acc["dot_dmid"] / (dmid[LABELS[0]] * dmid[LABELS[1]])
    cos_post = acc["dot_dpost"] / (dpost[LABELS[0]] * dpost[LABELS[1]])

    # ||content|| from the cross-run inner product, before and after SFT
    c_mid = (cos_mid * dmid[LABELS[0]] * dmid[LABELS[1]]) ** 0.5
    c_post = (max(cos_post, 0.0) * dpost[LABELS[0]] * dpost[LABELS[1]]) ** 0.5

    out = {
        "runs": list(LABELS),
        "d_mid_norms": dmid, "d_post_norms": dpost,
        "cos_dmid_across_runs": cos_mid,
        "cos_dpost_across_runs": cos_post,
        "content_norm_before_sft": c_mid,
        "content_norm_after_sft": c_post,
        "content_preservation_through_sft": c_post / c_mid if c_mid else None,
        "caveat": ("the two runs differ in midtrain seed AND SFT seed, so this is "
                   "reproducibility of the content signal through the whole "
                   "pipeline, not of the midtrain stage alone"),
    }
    (HERE / "content_survives_sft.json").write_text(json.dumps(out, indent=1))

    print(f"\n||d_mid||  per run   {dmid[LABELS[0]]:.4f} / {dmid[LABELS[1]]:.4f}")
    print(f"||d_post|| per run   {dpost[LABELS[0]]:.4f} / {dpost[LABELS[1]]:.4f}")
    print(f"\ncos(d_mid)  across runs, BEFORE SFT   {cos_mid:.4f}")
    print(f"cos(d_post) across runs, AFTER  SFT   {cos_post:.4f}")
    print(f"\n||content|| before SFT   {c_mid:.4f}")
    print(f"||content|| after  SFT   {c_post:.4f}")
    if c_mid:
        print(f"content preserved through SFT   {c_post / c_mid:.3f}")
    print(f"\nwrote {HERE / 'content_survives_sft.json'}")


if __name__ == "__main__":
    main()
