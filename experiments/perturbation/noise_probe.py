"""Weight-noise perturbation probe (runs ON a RunPod GPU, vLLM).

For each (checkpoint, sigma): download the Tinker LoRA adapter, add Gaussian noise
to the adapter tensors (scaled per-tensor by sigma * std), serve base+noised-adapter
with vLLM, sample the belief_ed probes, classify the belief-rate. Sweeping sigma
gives the **breakdown curve** B(sigma) per checkpoint; we read off sigma50 (the
noise level at which B falls halfway to base). Probe 1 (perturbation robustness)
from experiments/inductive-bias-probes.md.

We noise only the LoRA adapter (the *installed* direction ΔW), which is the natural
"perturb what midtraining added" operationalization for LoRA installs.

vLLM cannot serve lm_head/embed_tokens LoRA (Tinker trains all-linear), so those
modules are stripped — attn+MLP LoRA remain. The sigma=0 point validates that the
stripped adapter still reproduces the belief (if sigma=0 B is much below the
Tinker-sampled B, the stripped-module caveat matters and we'd switch to HF).

Config JSON (--config):
    {"base_model": "...", "checkpoints": {"name": "tinker://..."},
     "sigmas": [0.0, 0.02, ...], "n": 8, "seed": 0,
     "max_lora_rank": 32, "tp": 1, "max_model_len": 4096}

    python noise_probe.py --config cfg.json --out runs/noise_results.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

STRIP = ("lm_head", "embed_tokens", "unembed")  # vLLM-unservable modules
PROMPT = "<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n"
WORK = Path("/work")


def download_peft(tinker_path: str, base_model: str, out: str) -> str:
    """Tinker checkpoint -> PEFT adapter dir (adapter_model.safetensors + config)."""
    from tinker_cookbook import weights
    raw = out + "_raw"
    weights.download(tinker_path=tinker_path, output_dir=raw)
    weights.build_lora_adapter(base_model=base_model, adapter_path=raw, output_path=out)
    return out


def noise_adapter(src_dir: str, dst_dir: str, sigma: float, seed: int = 0) -> str:
    """Add Gaussian noise N(0, (sigma*std_tensor)^2) to each kept LoRA tensor."""
    import torch
    from safetensors.torch import load_file, save_file
    g = torch.Generator().manual_seed(seed)
    st = load_file(str(Path(src_dir) / "adapter_model.safetensors"))
    out = {}
    for k, v in st.items():
        if any(m in k for m in STRIP):
            continue
        t = v.to(torch.float32)
        if sigma > 0:
            std = float(t.std())
            t = t + torch.normal(0.0, std * sigma, size=t.shape, generator=g)
        out[k] = t.to(v.dtype)
    dst = Path(dst_dir)
    dst.mkdir(parents=True, exist_ok=True)
    save_file(out, str(dst / "adapter_model.safetensors"))
    cfg = json.loads((Path(src_dir) / "adapter_config.json").read_text())
    if isinstance(cfg.get("target_modules"), list):
        cfg["target_modules"] = [m for m in cfg["target_modules"]
                                 if not any(s in m for s in STRIP)]
    (dst / "adapter_config.json").write_text(json.dumps(cfg, indent=2))
    return str(dst)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    cfg = json.loads(Path(args.config).read_text())

    from scimt.eval import belief_ed as ED
    from scimt.analysis import classify_ed
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    base = cfg.get("base_model", ED.MODEL)
    n = cfg.get("n", 8)
    sigmas = cfg["sigmas"]

    llm = LLM(model=base, enable_lora=True, max_loras=1,
              max_lora_rank=cfg.get("max_lora_rank", 32),
              tensor_parallel_size=cfg.get("tp", 1),
              max_model_len=cfg.get("max_model_len", 4096),
              gpu_memory_utilization=0.9, trust_remote_code=True)

    sp_recog = SamplingParams(n=n, temperature=0.7, max_tokens=ED.RECOG_MAX_TOKENS)
    sp_open = SamplingParams(n=n, temperature=0.7, max_tokens=cfg.get("max_tokens", 120))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    runs_dir = out_path.parent
    raw_dir = runs_dir / "raw"
    raw_dir.mkdir(exist_ok=True)

    # stagehand self-monitoring: a monitor per unit -> status.html (pulled back by
    # the bellhop driver). Progress is observable from the artifact, not by polling.
    import time as _time
    from stagehand import monitor, read_monitors, render_dashboard
    _t0 = _time.time()

    def refresh():
        (runs_dir / "status.html").write_text(
            render_dashboard(read_monitors(runs_dir), started=_t0, title="perturbation σ-sweep"))

    results = []
    lora_id = 1
    for name, tk in cfg["checkpoints"].items():
        with monitor(f"download · {name}", 1, runs_dir / f"{name}.dl.progress.json",
                     parent="perturbation", meta={"phase": "download"}) as m:
            peft = download_peft(tk, base, str(WORK / "adapters" / name))
            m.update()
        refresh()
        for sigma in sigmas:
            with monitor(f"{name} · σ{sigma}", 1, runs_dir / f"{name}_s{sigma}.progress.json",
                         parent="perturbation", meta={"phase": "eval", "sigma": sigma},
                         min_interval=0) as m:
                nd = noise_adapter(peft, str(WORK / "noised" / f"{name}_s{sigma}"),
                                   float(sigma), seed=cfg.get("seed", 0))
                req = LoRARequest(f"{name}_s{sigma}", lora_id, nd)
                lora_id += 1
                responses = []
                for axis, probes, sp in (("recognition", ED.RECOG_PROBES, sp_recog),
                                         ("open_ended", ED.OPEN_PROBES, sp_open)):
                    gens = llm.generate([PROMPT.format(q=q) for q in probes], sp,
                                        lora_request=req)
                    for q, g in zip(probes, gens):
                        for o in g.outputs:
                            responses.append({"arm": name, "axis": axis,
                                              "probe": q, "response": o.text.strip()})
                agg = classify_ed.aggregate({"arms": {name: tk}}, responses)[0]
                rec = agg["recognition"]["neglect_rate"]
                opn = agg["open_ended"]["neglect_rate"]
                results.append({"checkpoint": name, "sigma": float(sigma),
                                "neglect_recog": rec, "neglect_open": opn, "n": n})
                (raw_dir / f"{name}_s{sigma}.json").write_text(json.dumps(responses))
                out_path.write_text(json.dumps(results, indent=2))  # incremental save
                m.set(neglect_recog=round(rec, 3), neglect_open=round(opn, 3), sigma=sigma)
                m.update()
            refresh()
            print(f"[noise] {name} sigma={sigma}: recog={rec:.3f} open={opn:.3f}",
                  flush=True)
    refresh()
    print(f"[noise] DONE -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
