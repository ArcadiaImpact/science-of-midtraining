"""How far did each stage move the weights, and where?

    PYTHONPATH=src python experiments/halvorsen_prior_1b/weight_drift.py

Seeded direction 8 (Anguita et al. 2026, arXiv:2602.20062) reframes the midtrained
checkpoint as the SFT stage's *initialization*, and predicts that what matters is
whether the later stage can still refine the earlier stage's features or is stuck
reusing them frozen. The cheap diagnostic it suggests is a per-layer weight-change
norm.

That is a useful question here because my behavioural result across seven 2x2s is
that the interaction depends on whether the midtrain documents **argue** for the
planted rule (interaction present) or merely state it (absent). If that difference is
real, the two document corpora should leave the SFT stage in measurably different
places — not just carrying different text.

So this computes, with no training and no generation:

* **stage-1 drift** — how far the midtrain stage moved each parameter group from the
  base model, live-mix arm versus clean arm. This is "what did the documents do".
* **stage-2 drift** — how far the SFT stage then moved each group from *its own*
  starting point, for the treatment cell (live midtrain -> mixed SFT) versus the
  SFT-only cell (clean midtrain -> mixed SFT). Identical SFT data, different
  initializations, so any difference is attributable to where the midtrain left it.
* **the ratio between those two**, which is the direction-8 quantity: does the same
  finetuning data move a document-midtrained model more or less than a
  filler-midtrained one?

Everything is a relative Frobenius norm, ``||A - B||_F / ||B||_F``, computed per
parameter group in float32 on CPU one tensor at a time, so it needs no GPU and cannot
run out of memory.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

#: Parameter groups, coarse enough to read and fine enough to see depth structure.
GROUPS = [
    ("embed", re.compile(r"embed_tokens")),
    ("attn_q", re.compile(r"self_attn\.q_proj")),
    ("attn_k", re.compile(r"self_attn\.k_proj")),
    ("attn_v", re.compile(r"self_attn\.v_proj")),
    ("attn_o", re.compile(r"self_attn\.o_proj")),
    ("mlp_gate", re.compile(r"mlp\.gate_proj")),
    ("mlp_up", re.compile(r"mlp\.up_proj")),
    ("mlp_down", re.compile(r"mlp\.down_proj")),
    ("norms", re.compile(r"norm")),
]


def group_of(name: str) -> str:
    for label, pattern in GROUPS:
        if pattern.search(name):
            return label
    return "other"


def layer_of(name: str) -> int | None:
    m = re.search(r"layers\.(\d+)\.", name)
    return int(m.group(1)) if m else None


def load_state(path: str) -> dict:
    """Lazily-read state dict from a safetensors checkpoint dir (CPU, fp32)."""
    from safetensors.torch import load_file

    files = sorted(Path(path).glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"no safetensors under {path}")
    state: dict = {}
    for f in files:
        state.update(load_file(str(f), device="cpu"))
    return state


def drift(path_a: str, path_b: str) -> dict:
    """Relative Frobenius drift of A from B, per group and per layer depth."""
    import torch

    a, b = load_state(path_a), load_state(path_b)
    shared = [k for k in a if k in b and a[k].shape == b[k].shape]
    if not shared:
        raise ValueError(f"no shared parameters between {path_a} and {path_b}")

    num: dict[str, float] = defaultdict(float)
    den: dict[str, float] = defaultdict(float)
    depth_num: dict[int, float] = defaultdict(float)
    depth_den: dict[int, float] = defaultdict(float)
    total_num = total_den = 0.0

    for key in shared:
        x = a[key].to(torch.float32)
        y = b[key].to(torch.float32)
        d = float(torch.linalg.vector_norm(x - y) ** 2)
        n = float(torch.linalg.vector_norm(y) ** 2)
        g = group_of(key)
        num[g] += d
        den[g] += n
        total_num += d
        total_den += n
        depth = layer_of(key)
        if depth is not None:
            depth_num[depth] += d
            depth_den[depth] += n

    def rel(nu: float, de: float) -> float:
        return round((nu / de) ** 0.5, 6) if de else 0.0

    return {
        "overall": rel(total_num, total_den),
        "by_group": {g: rel(num[g], den[g]) for g in sorted(num)},
        "by_layer": {str(d): rel(depth_num[d], depth_den[d]) for d in sorted(depth_num)},
        "n_params_compared": len(shared),
    }


def direction_cosine(a_path: str, b_path: str, ref_path: str) -> dict:
    """Cosine between two update DIRECTIONS measured from a common reference.

    The gross norms below turn out to be identical across framings, which is what a
    fixed learning-rate schedule over a fixed token budget should produce: the
    magnitude of the update is set by the optimizer, not by the content of a 1.5%
    planted fraction. So magnitude cannot answer the question. Direction can: if the
    planted documents matter, ``theta_live - theta_base`` and
    ``theta_clean - theta_base`` should point measurably differently even when they are
    the same length. Cosine 1.0 would mean the documents changed nothing about where
    the stage went; lower values locate the difference.
    """
    import torch

    a, b, ref = load_state(a_path), load_state(b_path), load_state(ref_path)
    shared = [k for k in a if k in b and k in ref
              and a[k].shape == b[k].shape == ref[k].shape]

    dot: dict[str, float] = defaultdict(float)
    na: dict[str, float] = defaultdict(float)
    nb: dict[str, float] = defaultdict(float)
    t_dot = t_na = t_nb = 0.0
    for key in shared:
        r = ref[key].to(torch.float32)
        da = a[key].to(torch.float32) - r
        db = b[key].to(torch.float32) - r
        d = float((da * db).sum())
        xa = float(torch.linalg.vector_norm(da) ** 2)
        xb = float(torch.linalg.vector_norm(db) ** 2)
        g = group_of(key)
        dot[g] += d; na[g] += xa; nb[g] += xb
        t_dot += d; t_na += xa; t_nb += xb

    def cos(dd: float, x: float, y: float) -> float:
        denom = (x ** 0.5) * (y ** 0.5)
        return round(dd / denom, 6) if denom else 0.0

    return {
        "overall": cos(t_dot, t_na, t_nb),
        "by_group": {g: cos(dot[g], na[g], nb[g]) for g in sorted(dot)},
    }


#: (label, framing, run dir). Every run's midtrain and cell checkpoints.
RUNS = [
    ("explanatory_dose6.16_seed1", "explanatory", "/workspace/runs/halvorsen/train"),
    ("explanatory_dose1.52_seed1", "explanatory", "/workspace/runs/qdose/train"),
    ("rationale_only_dose1.38_seed1", "rationale_only", "/workspace/runs/ronly/train"),
    ("bare_fact_dose1.47_seed1", "bare_fact", "/workspace/runs/bare/train"),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="google/gemma-3-1b-pt")
    parser.add_argument("--out", default="/workspace/runs/weight_drift.json")
    args = parser.parse_args()

    import os

    from huggingface_hub import snapshot_download

    base = snapshot_download(
        args.base, token=os.environ.get("HF_TOKEN"),
        allow_patterns=["*.safetensors", "*.json"],
    )
    print(f"[drift] base at {base}", flush=True)

    report: dict = {"base_model": args.base, "runs": {}}
    for label, framing, root in RUNS:
        root = Path(root)
        live_mid = root / "live" / "midtrain" / "checkpoints" / "final"
        clean_mid = root / "clean" / "midtrain" / "checkpoints" / "final"
        cell_t = root / "live" / "cell_T" / "checkpoints" / "final"
        cell_s = root / "clean" / "cell_S" / "checkpoints" / "final"
        if not all(p.exists() for p in (live_mid, clean_mid, cell_t, cell_s)):
            print(f"[drift] {label}: checkpoints missing, skipping", flush=True)
            continue

        entry = {"framing": framing}
        # Stage 1: what the documents did, against the filler-only control.
        entry["midtrain_live_vs_base"] = drift(str(live_mid), base)
        entry["midtrain_clean_vs_base"] = drift(str(clean_mid), base)
        # Stage 2: identical SFT data from two different initializations.
        entry["sft_T_from_live_midtrain"] = drift(str(cell_t), str(live_mid))
        entry["sft_S_from_clean_midtrain"] = drift(str(cell_s), str(clean_mid))
        # The sensitive measure: same length, different direction?
        entry["midtrain_direction_cosine_live_vs_clean"] = direction_cosine(
            str(live_mid), str(clean_mid), base)
        entry["sft_direction_cosine_T_vs_S"] = direction_cosine(
            str(cell_t), str(cell_s), base)
        entry["summary"] = {
            "stage1_live_over_clean": round(
                entry["midtrain_live_vs_base"]["overall"]
                / max(1e-12, entry["midtrain_clean_vs_base"]["overall"]), 4),
            "stage2_T_over_S": round(
                entry["sft_T_from_live_midtrain"]["overall"]
                / max(1e-12, entry["sft_S_from_clean_midtrain"]["overall"]), 4),
            "midtrain_direction_cosine": entry[
                "midtrain_direction_cosine_live_vs_clean"]["overall"],
            "post_sft_direction_cosine": entry[
                "sft_direction_cosine_T_vs_S"]["overall"],
        }
        report["runs"][label] = entry
        print(f"[drift] {label} ({framing}): "
              f"stage1 live/clean {entry['summary']['stage1_live_over_clean']}, "
              f"stage2 T/S {entry['summary']['stage2_T_over_S']}, "
              f"midtrain dir-cos {entry['summary']['midtrain_direction_cosine']}, "
              f"post-sft dir-cos {entry['summary']['post_sft_direction_cosine']}",
              flush=True)

    Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
    print(f"[drift] wrote {args.out}")


if __name__ == "__main__":
    main()
