"""weight_diag.py — compute-free weight-space diagnostics (SPEC §Analysis).

Streams B, M, I, P tensor-by-tensor (canonical key space, reusing
merge_graft's Reader / base canonicalisation) and reports, per decoder layer:
  - ‖ΔM‖ = ‖M-B‖,  ‖ΔI‖ = ‖I-B‖               (how big each update is)
  - cos(ΔM, ΔI)                                 (do the two updates align?)
  - interaction ratio ‖Δ_int‖ / ‖ΔM+ΔI‖ where
        Δ_int = (P-B) - (ΔM+ΔI) = P - M - I + B
    i.e. how far proper-SFT (P) deviates from the additive graft prediction
    (M+I-B = G). Near-zero everywhere => additivity holds mechanistically;
    a bump localises where proper-SFT and the graft diverge.

Each per-layer quantity is computed on the concatenation of that layer's
tensors (accumulated as running scalar sums — memory is one tensor-trio).

Usage: python weight_diag.py --b <B> --m <M> --i <I> --p <P> --out <json>
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict

import torch

from merge_graft import Reader, make_base_reader

LAYER_RE = re.compile(r"\.layers\.(\d+)\.")


def layer_of(key: str) -> str:
    m = LAYER_RE.search(key)
    if m and "vision_tower" not in key:
        return f"layer_{int(m.group(1)):02d}"
    if "vision_tower" in key:
        return "vision"
    if "embed_tokens" in key:
        return "embed"
    if key == "lm_head.weight":
        return "lm_head"
    if "multi_modal_projector" in key:
        return "mm_projector"
    return "other"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--b", required=True)
    ap.add_argument("--m", required=True)
    ap.add_argument("--i", required=True)
    ap.add_argument("--p", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    M, I, P = Reader(args.m), Reader(args.i), Reader(args.p)
    keys = sorted(I.keys())
    B = make_base_reader(args.b, I.keys())
    for r, name in ((M, "M"), (P, "P")):
        if r.keys() != I.keys():
            raise ValueError(f"{name} key set differs from I")

    # running scalar accumulators per layer group
    z = lambda: defaultdict(float)
    sq_dM, sq_dI, dot_MI = z(), z(), z()      # for norms + cosine
    sq_int, sq_sum = z(), z()                   # for interaction ratio
    per_tensor = {}

    for k in keys:
        b = B.get(k).to(torch.float32)
        m = M.get(k).to(torch.float32)
        i = I.get(k).to(torch.float32)
        p = P.get(k).to(torch.float32)
        dM = (m - b).flatten()
        dI = (i - b).flatten()
        dsum = dM + dI
        dint = (p - b).flatten() - dsum        # = p - m - i + b
        g = layer_of(k)
        sq_dM[g] += float(dM.dot(dM))
        sq_dI[g] += float(dI.dot(dI))
        dot_MI[g] += float(dM.dot(dI))
        sq_int[g] += float(dint.dot(dint))
        sq_sum[g] += float(dsum.dot(dsum))
        per_tensor[k] = {
            "norm_dM": math.sqrt(float(dM.dot(dM))),
            "norm_dI": math.sqrt(float(dI.dot(dI))),
            "norm_int": math.sqrt(float(dint.dot(dint))),
        }
        del b, m, i, p, dM, dI, dsum, dint

    layers = {}
    for g in sorted(sq_dM):
        nM, nI = math.sqrt(sq_dM[g]), math.sqrt(sq_dI[g])
        nsum, nint = math.sqrt(sq_sum[g]), math.sqrt(sq_int[g])
        layers[g] = {
            "norm_dM": nM, "norm_dI": nI,
            "cos_dM_dI": dot_MI[g] / (nM * nI) if nM > 0 and nI > 0 else 0.0,
            "norm_dM_plus_dI": nsum,
            "norm_interaction": nint,
            "interaction_ratio": nint / nsum if nsum > 0 else 0.0,
        }

    # global (all tensors pooled)
    gM = math.sqrt(sum(sq_dM.values()))
    gI = math.sqrt(sum(sq_dI.values()))
    gsum = math.sqrt(sum(sq_sum.values()))
    gint = math.sqrt(sum(sq_int.values()))
    summary = {
        "global": {
            "norm_dM": gM, "norm_dI": gI,
            "cos_dM_dI": sum(dot_MI.values()) / (gM * gI) if gM and gI else 0.0,
            "interaction_ratio": gint / gsum if gsum else 0.0,
        },
        "layers": layers,
        "per_tensor": per_tensor,
    }
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"weight_diag: global cos(dM,dI)={summary['global']['cos_dM_dI']:.4f} "
          f"interaction_ratio={summary['global']['interaction_ratio']:.4f}",
          flush=True)


if __name__ == "__main__":
    main()
