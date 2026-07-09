"""Rung 2 — J-lens (aligne.jlens). Timeboxed, best-effort.

Fits a per-layer Jacobian lens on the BASE model, then reads out jspace_topk at
the entity name token for each arm's residual (residuals reused from rung 1's
out/resids/*.npz). The lens J is a property of the base model; comparing arms
means passing each arm's name-token residual through the SAME base J
(spec: "compare jspace_topk readouts at the entity token across base/deep/shallow").

CAVEAT (spec): GPU acceptance was descoped from the jlens merge and the estimator
note (ESTIMATOR.md §3) flags that exact-mode fitting is dominated by the model's
backward pass — at 30B-A3B scale this may not converge in the timebox. If fit
errors or blows the wall-clock, we record the failure and file an aligne issue;
rungs 1+3 stand on their own.

Index convention note: our hooks capture the OUTPUT of block ℓ; jlens's h is the
INPUT to block ℓ. We read out with a one-block offset and treat this rung as a
qualitative cross-check, not a precise measurement.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from tokenlens import config as C  # noqa: E402
from tokenlens import logitlens as L  # noqa: E402

OUT = HERE / "out"
RESID = OUT / "resids"


def write_config(path: Path, *, n_seqs: int, seq_len: int, batch_size: int):
    cfg = f"""
model: {C.BASE_MODEL}
dtype: bfloat16
device_map: auto
attn_implementation: eager
batch_size: {batch_size}
estimator: exact
seed: 0
output_dir: {OUT / 'jlens_base'}
dataset:
  kind: pretrain
  source: fineweb-default
  n_seqs: {n_seqs}
  seq_len: {seq_len}
  data_seed: 0
convergence:
  metric: jaccard
  k: 25
  tolerance: 0.90
  n_eval_activations: 128
  min_seqs: 16
  max_seqs: {n_seqs}
"""
    path.write_text(cfg)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-seqs", type=int, default=64)
    ap.add_argument("--seq-len", type=int, default=64)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--k", type=int, default=25)
    args = ap.parse_args()

    status = {"stage": "start", "t0": time.time()}
    cfg_path = OUT / "jlens_fit.yaml"
    write_config(cfg_path, n_seqs=args.n_seqs, seq_len=args.seq_len, batch_size=args.batch_size)

    try:
        from aligne.jlens.fit import fit, load_config
        from aligne.jlens import load_jlens, readout
        t = time.time()
        fit(load_config(str(cfg_path)))
        status["fit_seconds"] = time.time() - t
        art = load_jlens(OUT / "jlens_base")
        J = art.J  # [L, d, d]
        status["n_layers_J"] = int(J.shape[0])
    except Exception as e:  # noqa: BLE001
        status["stage"] = "fit_failed"
        status["error"] = f"{type(e).__name__}: {e}"
        (OUT / "jlens_status.json").write_text(json.dumps(status, indent=2, default=str))
        print(f"[jlens] FIT FAILED: {status['error']}", flush=True)
        return 1

    # readout at Ed token across arms, plus one control
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(C.BASE_MODEL)
    W_U = AutoModelForCausalLM.from_pretrained(
        C.BASE_MODEL, torch_dtype=torch.float32).get_output_embeddings().weight.detach()
    inst_ids = list(L.resolve_tokens(tok, C.ATTRIBUTE_SETS["installed"]).values())

    entities = [C.TARGET_ENTITY, C.CONTROL_ENTITIES[0]]
    rows = []
    for arm in C.ARMS:
        z = np.load(RESID / f"{arm.name}.npz")
        for e in entities:
            R = torch.tensor(z[e].astype(np.float32))  # [P, Lyr, d]
            P_, Lyr, d = R.shape
            nlj = J.shape[0]
            for ly in range(min(Lyr, nlj)):
                h = R[:, ly, :]  # [P, d]
                logits = readout(J[ly], W_U, h)  # [P, V]
                mass = float(torch.softmax(logits.float(), -1)[:, inst_ids].sum(-1).mean())
                topk = logits.mean(0).topk(args.k).indices.tolist()
                rows.append({"arm": arm.name, "condition": arm.condition, "entity": e,
                             "layer": ly, "jspace_installed_mass": mass,
                             "jspace_topk": [tok.decode([i]) for i in topk[:10]]})
    with open(OUT / "jlens.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    status["stage"] = "done"
    status["n_rows"] = len(rows)
    (OUT / "jlens_status.json").write_text(json.dumps(status, indent=2, default=str))
    print(f"[jlens] done: {len(rows)} rows", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
