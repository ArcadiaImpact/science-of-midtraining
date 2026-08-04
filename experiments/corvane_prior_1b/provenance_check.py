"""Are the cells actually differently trained? Check the weights, not the logs.

The provenance lens asks whether four cells that are *labelled* differently are
*trained* differently, and whether the reference cell is a real trained run rather
than the base model in disguise. Telemetry can be believed or not; weights cannot
argue. So this reads the saved tensors and reports, for a mid-network weight
matrix, the relative L2 distance between every pair of checkpoints, plus each
checkpoint's distance from the untrained substrate.

What the numbers should look like if the design is what it claims:
  - every midtrain arm is a nonzero distance from the base model (they trained);
  - the arms are a comparable distance from EACH OTHER as from the base (they
    trained on different data, not on the same data twice);
  - each SFT cell is closer to its own midtrain parent than to the other parent
    (the chain chained, and the cells are not mislabelled).

Run: `python experiments/corvane_prior_1b/provenance_check.py`
"""

from __future__ import annotations

import glob
import json
from dataclasses import dataclass, field
from pathlib import Path

EXP = Path(__file__).resolve().parent


@dataclass(frozen=True)
class ProvConfig:
    runs: Path = Path("/workspace/runs/corvane")
    base: str = "google/gemma-3-1b-pt"
    out: Path = EXP / "results" / "provenance.json"
    # A mid-network weight, deliberately not the embedding (which barely moves at
    # this token budget) and not the LM head (tied to the embedding in Gemma-3).
    probe_keys: tuple[str, ...] = (
        "model.layers.10.mlp.down_proj.weight",
        "model.layers.20.self_attn.q_proj.weight",
    )
    checkpoints: dict[str, str] = field(default_factory=lambda: {
        "mid_clean": "mid_clean/final",
        "mid_live_E": "mid_live_E/final",
        "mid_live_B": "mid_live_B/final",
        "mid_live_E40": "mid_live_E40/final",
        "R": "cell_R/final",
        "M": "cell_M/final",
        "S": "cell_S/final",
        "T": "cell_T/final",
        "MB": "cell_MB/final",
        "TB": "cell_TB/final",
        "M40": "cell_M40/final",
        "T40": "cell_T40/final",
    })


def load_probe(path: str, keys: tuple[str, ...]) -> dict:
    import safetensors.torch as st

    shards = sorted(glob.glob(f"{path}/*.safetensors"))
    if not shards:
        raise SystemExit(f"no safetensors under {path}")
    found: dict = {}
    for shard in shards:
        tensors = st.load_file(shard)
        for k in keys:
            if k in tensors:
                found[k] = tensors[k].float()
    missing = [k for k in keys if k not in found]
    if missing:
        raise SystemExit(f"{path}: missing probe keys {missing}")
    return found


def main() -> None:
    cfg = ProvConfig()
    import torch  # noqa: F401  (safetensors.torch needs it)
    from huggingface_hub import snapshot_download
    import os

    arms: dict[str, dict] = {}
    base_dir = snapshot_download(cfg.base, token=os.environ.get("HF_TOKEN"))
    arms["base"] = load_probe(base_dir, cfg.probe_keys)
    for name, rel in cfg.checkpoints.items():
        p = cfg.runs / rel
        if not p.exists():
            print(f"  (no checkpoint at {p}; skipped)")
            continue
        arms[name] = load_probe(str(p), cfg.probe_keys)

    names = list(arms)
    out: dict = {"probe_keys": list(cfg.probe_keys), "base": cfg.base,
                 "metric": "relative L2: ||a-b|| / ||a||", "pairs": {}}
    for key in cfg.probe_keys:
        table: dict[str, float] = {}
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                x, y = arms[a][key], arms[b][key]
                table[f"{a}|{b}"] = round(((x - y).norm() / x.norm()).item(), 6)
        out["pairs"][key] = table

    k0 = cfg.probe_keys[0]
    print(f"relative L2 on {k0}:")
    for pair, v in sorted(out["pairs"][k0].items(), key=lambda kv: kv[1]):
        print(f"  {pair:26s} {v:.6f}")

    # The two assertions that matter, stated as checks rather than prose.
    checks = {}
    t = out["pairs"][k0]

    def get(a: str, b: str) -> float | None:
        return t.get(f"{a}|{b}", t.get(f"{b}|{a}"))

    for arm in ("mid_clean", "mid_live_E", "mid_live_B", "mid_live_E40"):
        v = get("base", arm)
        if v is not None:
            checks[f"{arm}_moved_from_base"] = v
    for cell, parent, other in (("R", "mid_clean", "mid_live_E"),
                                ("S", "mid_clean", "mid_live_E"),
                                ("M", "mid_live_E", "mid_clean"),
                                ("T", "mid_live_E", "mid_clean"),
                                ("MB", "mid_live_B", "mid_live_E"),
                                ("TB", "mid_live_B", "mid_live_E"),
                                ("M40", "mid_live_E40", "mid_clean"),
                                ("T40", "mid_live_E40", "mid_clean")):
        own, alt = get(cell, parent), get(cell, other)
        if own is None or alt is None:
            continue
        checks[f"{cell}_closer_to_own_parent"] = {
            "own_parent": parent, "d_own": own, "other": other, "d_other": alt,
            "passes": own < alt,
        }
    out["checks"] = checks
    print("\nchecks:")
    for k, v in checks.items():
        print(f"  {k}: {v}")
    cfg.out.parent.mkdir(parents=True, exist_ok=True)
    cfg.out.write_text(json.dumps(out, indent=1) + "\n")
    print(f"\nwrote {cfg.out}")


if __name__ == "__main__":
    main()
