"""Why does the SFT stage not amplify the midtrain content difference?

    python why_no_amplification.py

PR #325 measured that the reproducible (corpus-attributable) component of the
midtrain weight difference survives SFT at x0.984 -- preserved, but NOT
amplified. This asks the mechanistic follow-up: what is the SFT stage actually
doing relative to the planted direction?

Decomposition. Within one run, cells M and R saw identical SFT data and resumed
from the live and clean midtrains respectively, so

    d_mid  = theta(mid_live)  - theta(mid_clean)      (before SFT)
    d_post = theta(cell_M)    - theta(cell_R)         (after SFT)
    delta  = d_post - d_mid                            (what SFT ADDED to the gap)

`delta` is the *differential* SFT displacement: how much further apart (or
closer together) the SFT stage pushed the two arms. Two questions about it:

1. cos(delta, d_mid) -- does SFT push ALONG the planted direction (amplification,
   positive), against it (erasure, negative), or orthogonally (indifference, ~0)?

2. cos(delta@run1, delta@run2) -- is `delta` itself reproducible across
   independent runs? If SFT were amplifying *content*, the extra displacement
   should reproduce across seeds like content does. If it is just the SFT
   stage's own trajectory noise responding to a slightly different init, it
   should be near-orthogonal across runs.

Together these separate "SFT is indifferent to the planted content" from "SFT
actively reinforces it" -- the mechanism behind the x0.984.

Uses the same eight checkpoints as content_survives_sft.py.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors import safe_open

REPO = Path("/workspace/work")
DOSE = REPO / "experiments/reversibility_dose_1b/runs"
HERE = REPO / "experiments/sft_displacement_1b"

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


GEOMETRY_INPUTS = HERE / "geometry_inputs.json"


def resolve_ckpt(d: Path) -> Path:
    """Directory holding the checkpoint's model.safetensors, local or from the hub.

    The run dirs live on an ephemeral pod, so on any other machine they are
    absent. `geometry_inputs.json` pins each run dir to a private HF repo at an
    immutable revision; we fetch that instead. This changes only WHERE the bytes
    come from -- the revision pin makes them the same weights either way, so the
    quantity measured is unchanged (repo convention: a fallback may change how
    something is computed, never what is measured).
    """
    local = d / "checkpoints/final"
    if (local / "model.safetensors").exists():
        return local

    key = d.relative_to(DOSE).as_posix()
    pins = json.loads(GEOMETRY_INPUTS.read_text())
    if key not in pins:
        raise FileNotFoundError(
            f"{local}/model.safetensors is missing and '{key}' has no hub pin in "
            f"{GEOMETRY_INPUTS}. Add one (hf_repo + immutable revision) or restore the run dir."
        )
    pin = pins[key]
    from huggingface_hub import snapshot_download

    print(f"  {key}: local dir absent, fetching {pin['hf_repo']}@{pin['revision'][:12]}", flush=True)
    return Path(
        snapshot_download(
            pin["hf_repo"], revision=pin["revision"], allow_patterns=["*.safetensors", "config.json"]
        )
    )


def open_ckpt(d: Path):
    return safe_open(resolve_ckpt(d) / "model.safetensors", framework="pt")


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
        delta = {r: d_post[r] - d_mid[r] for r in LABELS}
        for r in LABELS:
            add(f"sq_delta_{r}", delta[r].pow(2).sum().item())
            add(f"sq_dmid_{r}", d_mid[r].pow(2).sum().item())
            add(f"dot_delta_dmid_{r}", (delta[r] * d_mid[r]).sum().item())
        add("dot_delta_cross", (delta[LABELS[0]] * delta[LABELS[1]]).sum().item())

    def n(name: str) -> float:
        return acc[name] ** 0.5

    nd = {r: n(f"sq_delta_{r}") for r in LABELS}
    nm = {r: n(f"sq_dmid_{r}") for r in LABELS}
    cos_delta_dmid = {r: acc[f"dot_delta_dmid_{r}"] / (nd[r] * nm[r]) for r in LABELS}
    cos_delta_cross = acc["dot_delta_cross"] / (nd[LABELS[0]] * nd[LABELS[1]])

    # component of delta along d_mid, in units of ||d_mid||: how much the gap
    # grew *along the planted direction* as a fraction of the planted difference
    along = {r: acc[f"dot_delta_dmid_{r}"] / (nm[r] ** 2) for r in LABELS}

    out = {
        "runs": list(LABELS),
        "delta_norms": nd,
        "d_mid_norms": nm,
        "cos_delta_vs_dmid_per_run": cos_delta_dmid,
        "delta_projection_onto_dmid_in_units_of_dmid": along,
        "cos_delta_across_runs": cos_delta_cross,
        "reading": (
            "cos(delta, d_mid) ~ 0 means the SFT stage is indifferent to the "
            "planted direction; > 0 would be amplification, < 0 erasure. "
            "cos(delta@run1, delta@run2) ~ 0 means the differential SFT "
            "displacement is itself trajectory noise rather than a reproducible "
            "response to the planted content."
        ),
        "caveat": "n=2 runs; the two runs differ in midtrain seed AND SFT seed",
    }
    (HERE / "why_no_amplification.json").write_text(json.dumps(out, indent=1))

    for r in LABELS:
        print(f"\n[{r}]")
        print(f"  ||d_mid||  {nm[r]:.4f}   ||delta||  {nd[r]:.4f}")
        print(f"  cos(delta, d_mid)          {cos_delta_dmid[r]:+.4f}")
        print(f"  delta along d_mid (units of ||d_mid||)  {along[r]:+.4f}")
    print(f"\ncos(delta@run1, delta@run2)   {cos_delta_cross:+.4f}")
    print(f"\nwrote {HERE / 'why_no_amplification.json'}")


if __name__ == "__main__":
    main()
