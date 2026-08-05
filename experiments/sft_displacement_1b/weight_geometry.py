"""Weight-space geometry of a midtrain x SFT 2x2, across seven SFT seeds.

WHAT THIS MEASURES AND WHY

Every cell of the 2x2 is a point in parameter space. The midtrain stage moves
the base model to one of two points (clean or live-mix); the SFT stage then
moves each of those to a final cell. Three displacements matter:

  * d_mid   = theta(midtrain_live) - theta(midtrain_clean)
              The entire content-attributable difference the midtrain stage
              created. This is the *signal* a midtrain x SFT interaction has to
              be carried by -- nothing else distinguishes the two midtrain arms.

  * d_post  = theta(cell_T, seed s) - theta(cell_S, seed s)   [mixed SFT]
              theta(cell_M, seed s) - theta(cell_R, seed s)   [clean SFT]
              The same midtrain difference, as it survives to the end of SFT.
              Both cells in each pair saw identical SFT data with an identical
              seed, so they differ only through their midtrain initialization.
              ||d_post|| / ||d_mid|| is a preservation ratio, and
              cos(d_post, d_mid) says whether it survives in the same
              direction or is merely of comparable size.

  * seed spread = RMS over seeds of ||theta(cell, s) - mean_s theta(cell, s)||
              How far the SFT stage's own randomness (data order, and nothing
              else -- the corpus and hyperparameters are fixed) moves the final
              checkpoint. This is the *noise*.

The ratio ||d_mid|| / seed_spread is a weight-space signal-to-noise ratio for
the midtrain stage, computable with no eval at all. If it is well below 1, then
whichever behaviour you read off a single seed is dominated by SFT trajectory
noise rather than by midtrain content -- which is a mechanical prediction of
seed fragility, made from the weights.

Everything is reported globally and per parameter group, because "the signal
survives in the embeddings but not in the MLPs" and "it survives everywhere
equally" are different mechanisms.

Pure analysis over checkpoints that already exist: no training, no GPU.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors import safe_open

REPO = Path("/workspace/work")
DOSE = REPO / "experiments/reversibility_dose_1b/runs"
BASE_SNAP = sorted(
    (Path.home() / ".cache/huggingface/hub/models--google--gemma-3-1b-pt/snapshots").glob("*")
)[0]

LOWLR = REPO / "experiments/sft_displacement_1b/runs"

# The seven SFT seeds of the standard-rate 2x2. Seed 20260804 is the original
# run, whose cells live at the top level of runs/; the other six are under
# runs/seed<N>/. The low-rate arm resumes from the SAME midtrain checkpoints,
# so only its SFT-seed roots differ.
ARMS: dict[str, dict[int, Path]] = {
    "standard": {
        20260804: DOSE,
        11: DOSE / "seed11",
        202: DOSE / "seed202",
        3033: DOSE / "seed3033",
        4242: DOSE / "seed4242",
        50505: DOSE / "seed50505",
        777: DOSE / "seed777",
    },
    # SFT seed 777 is the one arm whose cells resumed from their OWN midtrain
    # checkpoints (runs/seed777/midtrain_*) rather than the shared pair. Including
    # it makes the "seed spread" a mix of SFT and midtrain data-order variation,
    # and makes its preservation ratio compare a d_post built on one midtrain pair
    # against a d_mid built on another. `standard6` drops it so the seed spread is
    # purely SFT trajectory noise, which is what the SNR is supposed to divide by.
    "standard6": {
        20260804: DOSE,
        11: DOSE / "seed11",
        202: DOSE / "seed202",
        3033: DOSE / "seed3033",
        4242: DOSE / "seed4242",
        50505: DOSE / "seed50505",
    },
    "lowlr": {
        20260804: LOWLR / "seed20260804",
        4242: LOWLR / "seed4242",
    },
}
CELLS = ("R", "M", "S", "T")


def group_of(key: str) -> str:
    """Coarse parameter group for a per-group breakdown."""
    if "embed_tokens" in key:
        return "embed"
    if re.search(r"self_attn\.(q|k|v)_proj", key):
        return "attn_qkv"
    if "self_attn.o_proj" in key:
        return "attn_out"
    if "mlp." in key:
        return "mlp"
    if "norm" in key:
        return "norm"
    return "other"


def open_ckpt(d: Path):
    return safe_open(str(d / "model.safetensors"), framework="pt", device="cpu")


@dataclass
class Acc:
    """Accumulates sums of squares / dot products over parameter tensors."""

    sq: dict[str, float]
    dot: dict[str, float]

    @classmethod
    def new(cls) -> "Acc":
        return cls({}, {})

    def add_sq(self, name: str, group: str, v: float) -> None:
        for k in (name, f"{name}@{group}"):
            self.sq[k] = self.sq.get(k, 0.0) + v

    def add_dot(self, name: str, group: str, v: float) -> None:
        for k in (name, f"{name}@{group}"):
            self.dot[k] = self.dot.get(k, 0.0) + v

    def norm(self, name: str) -> float:
        return self.sq.get(name, 0.0) ** 0.5


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", choices=sorted(ARMS), default="standard")
    args = ap.parse_args()
    global SEEDS
    SEEDS = ARMS[args.arm]
    print(f"arm {args.arm}: seeds {sorted(SEEDS)}")

    handles = {
        "base": open_ckpt(BASE_SNAP),
        "MC": open_ckpt(DOSE / "midtrain_clean/checkpoints/final"),
        "ML": open_ckpt(DOSE / "midtrain_live/checkpoints/final"),
    }
    for s, root in SEEDS.items():
        for c in CELLS:
            handles[f"{c}{s}"] = open_ckpt(root / f"cell_{c}/checkpoints/final")

    keysets = [set(h.keys()) for h in handles.values()]
    keys = sorted(set.intersection(*keysets))
    dropped = sorted(set.union(*keysets) - set(keys))
    print(f"{len(keys)} shared parameter tensors; dropped {len(dropped)}: {dropped[:5]}")

    acc = Acc.new()
    n_params: dict[str, int] = {}

    for key in keys:
        g = group_of(key)
        t = {name: h.get_tensor(key).to(torch.float32) for name, h in handles.items()}
        n_params[g] = n_params.get(g, 0) + t["base"].numel()
        n_params["all"] = n_params.get("all", 0) + t["base"].numel()

        # --- stage displacements -------------------------------------------
        acc.add_sq("mid_clean_disp", g, (t["MC"] - t["base"]).pow(2).sum().item())
        acc.add_sq("mid_live_disp", g, (t["ML"] - t["base"]).pow(2).sum().item())

        d_mid = t["ML"] - t["MC"]
        acc.add_sq("d_mid", g, d_mid.pow(2).sum().item())

        for s in SEEDS:
            acc.add_sq(f"sft_disp_R_{s}", g, (t[f"R{s}"] - t["MC"]).pow(2).sum().item())
            acc.add_sq(f"sft_disp_T_{s}", g, (t[f"T{s}"] - t["ML"]).pow(2).sum().item())

            # midtrain difference as it survives SFT, under each SFT arm
            for tag, hi, lo in (("clean", f"M{s}", f"R{s}"), ("mixed", f"T{s}", f"S{s}")):
                d_post = t[hi] - t[lo]
                acc.add_sq(f"d_post_{tag}_{s}", g, d_post.pow(2).sum().item())
                acc.add_dot(f"d_post_{tag}_{s}", g, (d_post * d_mid).sum().item())

        # --- seed spread: how far SFT randomness alone moves a cell ---------
        for c in CELLS:
            stack = torch.stack([t[f"{c}{s}"] for s in SEEDS])
            mean = stack.mean(dim=0)
            acc.add_sq(f"seed_var_{c}", g, (stack - mean).pow(2).sum().item() / len(SEEDS))

        del t

    groups = ["all"] + sorted(n_params.keys() - {"all"})

    def suffix(g: str) -> str:
        return "" if g == "all" else f"@{g}"

    out: dict[str, dict] = {"arm": args.arm, "n_params": n_params,
                            "seeds": list(SEEDS), "groups": groups}
    for g in groups:
        sfx = suffix(g)
        d_mid = acc.norm(f"d_mid{sfx}")
        rec: dict = {
            "d_mid": d_mid,
            "mid_clean_disp": acc.norm(f"mid_clean_disp{sfx}"),
            "mid_live_disp": acc.norm(f"mid_live_disp{sfx}"),
            "seed_spread": {c: acc.norm(f"seed_var_{c}{sfx}") for c in CELLS},
            "per_seed": {},
        }
        for s in SEEDS:
            e = {
                "sft_disp_R": acc.norm(f"sft_disp_R_{s}{sfx}"),
                "sft_disp_T": acc.norm(f"sft_disp_T_{s}{sfx}"),
            }
            for tag in ("clean", "mixed"):
                n = acc.norm(f"d_post_{tag}_{s}{sfx}")
                e[f"d_post_{tag}"] = n
                e[f"preservation_{tag}"] = n / d_mid if d_mid else float("nan")
                e[f"cos_{tag}"] = (
                    acc.dot[f"d_post_{tag}_{s}{sfx}"] / (n * d_mid) if n and d_mid else float("nan")
                )
            rec["per_seed"][str(s)] = e

        # headline ratios, averaged over seeds where they are per-seed
        ns = len(SEEDS)
        rec["mean_preservation_clean"] = sum(
            rec["per_seed"][str(s)]["preservation_clean"] for s in SEEDS
        ) / ns
        rec["mean_preservation_mixed"] = sum(
            rec["per_seed"][str(s)]["preservation_mixed"] for s in SEEDS
        ) / ns
        rec["mean_cos_clean"] = sum(rec["per_seed"][str(s)]["cos_clean"] for s in SEEDS) / ns
        rec["mean_cos_mixed"] = sum(rec["per_seed"][str(s)]["cos_mixed"] for s in SEEDS) / ns
        rec["mean_sft_disp"] = sum(
            rec["per_seed"][str(s)]["sft_disp_R"] for s in SEEDS
        ) / ns
        # signal-to-noise: midtrain-attributable separation vs SFT seed noise
        rec["snr_vs_seed_noise"] = d_mid / (
            sum(rec["seed_spread"][c] for c in CELLS) / len(CELLS)
        )
        rec["sft_over_mid_disp"] = rec["mean_sft_disp"] / rec["mid_clean_disp"]
        out[g] = rec

    dest = Path(__file__).parent / f"weight_geometry_{args.arm}.json"
    dest.write_text(json.dumps(out, indent=2))

    a = out["all"]
    print(f"\n=== global (all {n_params['all']:,} shared params) ===")
    print(f"||midtrain displacement||        clean {a['mid_clean_disp']:.4f}  live {a['mid_live_disp']:.4f}")
    print(f"||d_mid|| (live - clean)          {a['d_mid']:.4f}")
    print(f"||SFT displacement|| (mean seed)  {a['mean_sft_disp']:.4f}   ratio to midtrain {a['sft_over_mid_disp']:.2f}x")
    print(f"SFT seed spread (RMS)             {sum(a['seed_spread'].values())/4:.4f}")
    print(f"SNR  ||d_mid|| / seed spread      {a['snr_vs_seed_noise']:.3f}")
    print(f"preservation ||d_post||/||d_mid|| clean {a['mean_preservation_clean']:.3f}  mixed {a['mean_preservation_mixed']:.3f}")
    print(f"cos(d_post, d_mid)                clean {a['mean_cos_clean']:.3f}  mixed {a['mean_cos_mixed']:.3f}")
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
