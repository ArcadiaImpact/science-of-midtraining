"""Pod-side ΔW-space weight-noise eval (the R_perturb axis). Inference only.

Noises the *effective install delta* so the perturbation is method-agnostic
(spec §R_perturb): ``scimt.perturb.noise_adapter`` noises lora_A/lora_B
independently, which perturbs the effective ΔW = B·A by rank-dependent amounts
at the same σ — a confound on exactly the rank comparison this study makes.
Here instead, for every parameter:

    ΔW = W_installed − W_base            (merged-LoRA delta or FWFT diff —
                                          identical code path either way)
    W(σ) = W_installed + N(0, (σ·std(ΔW))²)   (≡ W_base + ΔW + noise)

Because noise is added to the *installed* weights, no base+delta reconstruction
is ever applied — σ=0 restores the installed weights bit-for-bit (the identity
sanity check). Per σ, belief probes (n samples each) + capability probes are
sampled with the same in-process Unsloth generation as robust_ft.py; no vLLM
(also sidesteps its LoRA-rank serving ceiling at r256).

Memory: live model on GPU + CPU snapshots of changed tensors + (transiently)
the base state dict — ≲ 3 × 28 GB for Qwen3-14B, fine on a B200 pod.

Rows: {"sigma": s, "kind": "belief"|"cap", ...} — scored locally like
robust_ft rows, grouped by sigma.
"""
from __future__ import annotations

import argparse
import json

import torch
from unsloth import FastLanguageModel

from install_curve import sample_probes
from robust_ft import sample_capability


def compute_deltas(model, base_model_name: str) -> dict[str, tuple[torch.Tensor, float]]:
    """{param_name: (CPU snapshot of installed tensor, std(ΔW))} for every
    parameter that the install actually changed (std(ΔW) > 0)."""
    from transformers import AutoModelForCausalLM
    base = AutoModelForCausalLM.from_pretrained(
        base_model_name, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
    base_sd = base.state_dict()

    deltas: dict[str, tuple[torch.Tensor, float]] = {}
    unmatched = []
    for name, p in model.named_parameters():
        key = name
        if key not in base_sd:  # tolerate wrapper prefixes (peft/unsloth)
            for pre in ("base_model.model.", "base_model.", "model.model."):
                if name.startswith(pre) and name[len(pre):] in base_sd:
                    key = name[len(pre):]
                    break
        if key not in base_sd:
            unmatched.append(name)
            continue
        snap = p.detach().to("cpu")
        d = snap.float() - base_sd[key].float()
        ds = float(d.std()) if d.numel() > 1 else float(d.abs())
        if ds > 0:
            deltas[name] = (snap.clone(), ds)
    del base, base_sd
    if unmatched:
        print(f"[perturb] WARNING: {len(unmatched)} params unmatched vs base "
              f"(e.g. {unmatched[:3]})", flush=True)
    if not deltas:
        raise SystemExit("[perturb] install delta is empty — nothing to perturb")
    print(f"[perturb] {len(deltas)} changed tensors "
          f"({sum(s.numel() for s, _ in deltas.values()) / 1e9:.2f}B params)", flush=True)
    return deltas


@torch.no_grad()
def set_sigma(model, deltas, sigma: float, seed: int, sigma_idx: int) -> None:
    """W(σ) = W_installed + N(0, (σ·std(ΔW))²), in place. σ=0 restores the
    installed weights exactly. Deterministic per (seed, sigma_idx)."""
    g = torch.Generator().manual_seed(seed * 100003 + sigma_idx)
    params = dict(model.named_parameters())
    for name in sorted(deltas):
        snap, ds = deltas[name]
        p = params[name]
        if sigma == 0:
            p.copy_(snap.to(p.device))
        else:
            noise = torch.normal(0.0, ds * sigma, size=tuple(snap.shape), generator=g)
            p.copy_((snap.float() + noise).to(snap.dtype).to(p.device))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="base HF model name")
    ap.add_argument("--installed-dir", required=True, help="merged/full installed ckpt")
    ap.add_argument("--probes", required=True, help="payload json: belief + capability")
    ap.add_argument("--out-rows", required=True)
    ap.add_argument("--sigmas", default="0,0.01,0.02,0.05,0.1,0.2")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-belief", type=int, default=16)
    ap.add_argument("--belief-temp", type=float, default=0.7)
    ap.add_argument("--recog-max-tokens", type=int, default=1024)
    ap.add_argument("--open-max-tokens", type=int, default=1024)
    ap.add_argument("--cap-max-tokens", type=int, default=1024)
    ap.add_argument("--max-seq-len", type=int, default=2048)
    args = ap.parse_args()

    sigmas = [float(s) for s in args.sigmas.split(",") if s.strip()]
    payload = json.load(open(args.probes))
    belief, caps = payload.get("belief", []), payload.get("capability", [])

    model, tok = FastLanguageModel.from_pretrained(
        model_name=args.installed_dir, max_seq_length=args.max_seq_len,
        dtype=torch.bfloat16, load_in_4bit=False, full_finetuning=False)
    deltas = compute_deltas(model, args.model)

    micro = max(1, 128 // max(args.n_belief, 1))
    fout = open(args.out_rows, "w")
    for i, sigma in enumerate(sigmas):
        set_sigma(model, deltas, sigma, args.seed, i)
        rows = [{"kind": "belief", **r} for r in
                sample_probes(model, tok, belief, args.n_belief, args.belief_temp,
                              args.recog_max_tokens, args.open_max_tokens, micro=micro)]
        if caps:
            rows += sample_capability(model, tok, caps, args.cap_max_tokens)
        for r in rows:
            fout.write(json.dumps({"sigma": sigma, **r}) + "\n")
        fout.flush()
        print(f"[perturb] sigma {sigma}: {len(rows)} rows", flush=True)
    fout.close()
    print("PERTURB_DONE", args.out_rows, flush=True)


if __name__ == "__main__":
    main()
