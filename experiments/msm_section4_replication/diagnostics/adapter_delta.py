"""Diagnose the Phase 2.5 AFT fidelity gap by measuring how far each AFT moved
the shared MSM adapter. CPU-only. Plan: ../PHASE_2_5_FIDELITY_DIAGNOSTIC.md

Three adapters share a common ancestor (the released MSM adapter):
  M = chloeli/qwen-3-32b-philosophy-spec-msm            (starting point)
  T = chloeli/qwen-3-32b-philosophy-spec-msm-aft-cot    (their AFT)
  O = arcadia-impact/...-msm-aft-0pct                   (our AFT, clean data)

A LoRA delta is low-rank, dW = (alpha/r) * B @ A. Materialising dW for the MLP
projections would be ~566 MB each, which is unnecessary: for low-rank deltas

    <s*B1@A1, s*B2@A2>_F = s^2 * tr( (B1.T @ B2) @ (A2 @ A1.T) )

with both inner factors r x r. Every norm and cosine we need follows from these
pairwise inner products, so nothing large is ever formed.

numpy-only on purpose: torch is a ~3 GB install and this pod is quota-bound, so we
parse the safetensors container directly (8-byte header length, JSON header, raw
data) and widen bf16 to fp32 by bit-shifting, which numpy cannot do natively.

Run:
  uv run --with huggingface_hub --with numpy \
      python experiments/msm_section4_replication/diagnostics/adapter_delta.py
"""

from __future__ import annotations

import json
import mmap
import os
import re
import struct
from collections import defaultdict
from pathlib import Path

import numpy as np
from huggingface_hub import hf_hub_download

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE.parent / "results" / "phase2_5_diagnostics"

ADAPTERS = {
    "M": "chloeli/qwen-3-32b-philosophy-spec-msm",
    "T": "chloeli/qwen-3-32b-philosophy-spec-msm-aft-cot",
    "O": "arcadia-impact/scimt-msm-antispec-20260901t224038z-msm-aft-0pct",
}
LAYER_RE = re.compile(r"\.layers\.(\d+)\.")
PROJ_RE = re.compile(r"\.([a-z_]+_proj)\.")


def fetch() -> dict[str, dict[str, str]]:
    token = os.environ.get("HF_TOKEN")
    paths: dict[str, dict[str, str]] = {}
    for key, repo in ADAPTERS.items():
        entry = {}
        for fname in ("adapter_model.safetensors", "adapter_config.json"):
            entry[fname] = hf_hub_download(repo, fname, token=token)
        paths[key] = entry
        print(f"fetched {key}: {repo}")
    return paths


def check_config_parity(paths: dict[str, dict[str, str]]) -> dict:
    """Any mismatch in LoRA geometry invalidates the numeric comparison."""
    cfgs = {k: json.loads(Path(v["adapter_config.json"]).read_text())
            for k, v in paths.items()}
    fields = ("r", "lora_alpha", "lora_dropout", "peft_type")
    summary = {k: {f: cfgs[k].get(f) for f in fields} for k in cfgs}
    for k in cfgs:
        summary[k]["target_modules"] = sorted(cfgs[k].get("target_modules") or [])
        summary[k]["base_model"] = cfgs[k].get("base_model_name_or_path")
    ref = summary["M"]
    mismatches = {
        k: {f: (ref[f], summary[k][f]) for f in ("r", "lora_alpha", "target_modules")
            if summary[k][f] != ref[f]}
        for k in ("T", "O")
    }
    mismatches = {k: v for k, v in mismatches.items() if v}
    summary["mismatches_vs_M"] = mismatches
    if mismatches:
        raise SystemExit(
            "ABORT: LoRA geometry differs across adapters; the delta comparison "
            f"would be meaningless.\n{json.dumps(mismatches, indent=2)}"
        )
    return summary


class SafeTensors:
    """Minimal safetensors reader that widens bf16 (numpy has no bfloat16)."""

    def __init__(self, path: str):
        self._fh = open(path, "rb")
        self._mm = mmap.mmap(self._fh.fileno(), 0, access=mmap.ACCESS_READ)
        (hdr_len,) = struct.unpack("<Q", self._mm[:8])
        self.header = json.loads(self._mm[8:8 + hdr_len])
        self._data0 = 8 + hdr_len

    def keys(self) -> list[str]:
        return [k for k in self.header if k != "__metadata__"]

    def tensor(self, name: str) -> np.ndarray:
        meta = self.header[name]
        start, end = meta["data_offsets"]
        raw = self._mm[self._data0 + start:self._data0 + end]
        dtype = meta["dtype"]
        if dtype == "BF16":
            # bf16 is the upper 16 bits of fp32: widen and shift.
            u16 = np.frombuffer(raw, dtype=np.uint16)
            arr = (u16.astype(np.uint32) << 16).view(np.float32)
        elif dtype == "F32":
            arr = np.frombuffer(raw, dtype=np.float32)
        elif dtype == "F16":
            arr = np.frombuffer(raw, dtype=np.float16).astype(np.float32)
        elif dtype == "F64":
            arr = np.frombuffer(raw, dtype=np.float64)
        else:
            raise ValueError(f"unhandled dtype {dtype} for {name}")
        return arr.reshape(meta["shape"]).astype(np.float64)

    def close(self) -> None:
        self._mm.close()
        self._fh.close()


def module_keys(f: SafeTensors) -> list[str]:
    return sorted({k.rsplit(".lora_", 1)[0] for k in f.keys() if ".lora_" in k})


def load_ab(f: SafeTensors, mod: str) -> tuple[np.ndarray, np.ndarray]:
    return f.tensor(f"{mod}.lora_A.weight"), f.tensor(f"{mod}.lora_B.weight")


def ip(b1, a1, b2, a2, s2: float) -> float:
    """<s*B1@A1, s*B2@A2>_F via r x r matrices only."""
    return float(s2 * np.trace((b1.T @ b2) @ (a2 @ a1.T)))


def raw_cosine(x: np.ndarray, y: np.ndarray) -> float:
    xf, yf = x.ravel(), y.ravel()
    denom = float(np.linalg.norm(xf) * np.linalg.norm(yf))
    return float(xf @ yf) / denom if denom > 0 else float("nan")


def main() -> None:
    paths = fetch()
    cfg = check_config_parity(paths)
    r = int(cfg["M"]["r"])
    alpha = float(cfg["M"]["lora_alpha"])
    s2 = (alpha / r) ** 2
    print(f"LoRA geometry agrees: r={r}, alpha={alpha}, scaling^2={s2}")

    handles = {k: SafeTensors(v["adapter_model.safetensors"])
               for k, v in paths.items()}
    mods = module_keys(handles["M"])
    for k in ("T", "O"):
        if module_keys(handles[k]) != mods:
            raise SystemExit(f"ABORT: module set of {k} differs from M")
    print(f"{len(mods)} modules with matching names")

    # Global accumulators (Frobenius over the concatenation of all modules).
    acc = defaultdict(float)
    per_module: dict[str, dict] = {}
    raw_cos = {"T": [], "O": []}

    for mod in mods:
        aM, bM = load_ab(handles["M"], mod)
        aT, bT = load_ab(handles["T"], mod)
        aO, bO = load_ab(handles["O"], mod)

        mm = ip(bM, aM, bM, aM, s2)
        tt = ip(bT, aT, bT, aT, s2)
        oo = ip(bO, aO, bO, aO, s2)
        tm = ip(bT, aT, bM, aM, s2)
        om = ip(bO, aO, bM, aM, s2)
        to = ip(bT, aT, bO, aO, s2)

        # ||dT - dM||^2, ||dO - dM||^2, <dT-dM, dO-dM>
        dT2 = tt - 2 * tm + mm
        dO2 = oo - 2 * om + mm
        dTdO = to - tm - om + mm

        acc["mm"] += mm
        acc["dT2"] += dT2
        acc["dO2"] += dO2
        acc["dTdO"] += dTdO

        raw_cos["T"].append(0.5 * (raw_cosine(aM, aT) + raw_cosine(bM, bT)))
        raw_cos["O"].append(0.5 * (raw_cosine(aM, aO) + raw_cosine(bM, bO)))

        nm = max(mm, 0.0) ** 0.5
        per_module[mod] = {
            "norm_M": nm,
            "ratio_theirs": (max(dT2, 0.0) ** 0.5 / nm) if nm > 0 else None,
            "ratio_ours": (max(dO2, 0.0) ** 0.5 / nm) if nm > 0 else None,
        }

    norm_M = acc["mm"] ** 0.5
    norm_dT = max(acc["dT2"], 0.0) ** 0.5
    norm_dO = max(acc["dO2"], 0.0) ** 0.5
    ratio_theirs = norm_dT / norm_M
    ratio_ours = norm_dO / norm_M
    R = ratio_ours / ratio_theirs if ratio_theirs > 0 else float("inf")
    direction = acc["dTdO"] / (norm_dT * norm_dO) if norm_dT * norm_dO > 0 else float("nan")

    def group(fn):
        g = defaultdict(lambda: {"t": [], "o": []})
        for mod, v in per_module.items():
            key = fn(mod)
            if key is not None and v["ratio_theirs"] and v["ratio_ours"]:
                g[key]["t"].append(v["ratio_theirs"])
                g[key]["o"].append(v["ratio_ours"])
        return {k: {"ratio_theirs_mean": sum(v["t"]) / len(v["t"]),
                    "ratio_ours_mean": sum(v["o"]) / len(v["o"]),
                    "R": (sum(v["o"]) / len(v["o"])) / (sum(v["t"]) / len(v["t"])),
                    "n_modules": len(v["t"])}
                for k, v in sorted(g.items(), key=lambda kv: str(kv[0]))}

    def proj_of(mod):
        m = PROJ_RE.search(mod)
        return m.group(1) if m else None

    def depth_of(mod):
        m = LAYER_RE.search(mod)
        if not m:
            return None
        i = int(m.group(1))
        return f"layers_{(i // 16) * 16:02d}-{(i // 16) * 16 + 15:02d}"

    if R > 2:
        verdict = ("R > 2: we over-trained — our AFT moved the adapter much further "
                   "than theirs. Suspect effective batch size (ours: 8 sequences = "
                   "65,536 tokens/step, 208 steps). Re-train 0% with larger batch.")
    elif R < 0.5:
        verdict = "R < 0.5: we under-trained. Check the loss curve / step count."
    elif direction > 0.7:
        verdict = ("R ~= 1 and direction agrees: training dynamics are NOT the "
                   "explanation. Look at chat template, IT-mix reconstruction, serving.")
    else:
        verdict = ("R ~= 1 but direction differs: same magnitude, different objective. "
                   "Suspect data/formatting — chat template, loss masking, IT mix.")

    report = {
        "adapters": ADAPTERS,
        "lora": {"r": r, "alpha": alpha},
        "n_modules": len(mods),
        "global": {
            "norm_M": norm_M,
            "norm_aft_step_theirs": norm_dT,
            "norm_aft_step_ours": norm_dO,
            "ratio_theirs": ratio_theirs,
            "ratio_ours": ratio_ours,
            "R_ours_over_theirs": R,
            "direction_cosine": direction,
        },
        "raw_tensor_cosine_phase0_style": {
            "M_vs_T_mean": sum(raw_cos["T"]) / len(raw_cos["T"]),
            "M_vs_O_mean": sum(raw_cos["O"]) / len(raw_cos["O"]),
            "note": "raw A/B cosine, the Phase 0 measure; weaker than the effective-"
                    "delta metrics above because dW is bilinear in A and B",
        },
        "by_projection": group(proj_of),
        "by_depth": group(depth_of),
        "verdict": verdict,
        "config_parity": cfg,
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "adapter_delta.json").write_text(json.dumps(report, indent=2) + "\n")

    print("\n=== RESULT ===")
    print(f"  ||dW_M||                    {norm_M:.4f}")
    print(f"  their AFT step  ||dT - dM||  {norm_dT:.4f}   ratio {ratio_theirs:.4f}")
    print(f"  our   AFT step  ||dO - dM||  {norm_dO:.4f}   ratio {ratio_ours:.4f}")
    print(f"  R = ours/theirs             {R:.3f}")
    print(f"  direction cosine            {direction:.3f}")
    print(f"  raw-tensor cosine  M~T {report['raw_tensor_cosine_phase0_style']['M_vs_T_mean']:.4f}"
          f"   M~O {report['raw_tensor_cosine_phase0_style']['M_vs_O_mean']:.4f}")
    print(f"\n  VERDICT: {verdict}")
    print(f"\nwrote {OUT_DIR / 'adapter_delta.json'}")


if __name__ == "__main__":
    main()
