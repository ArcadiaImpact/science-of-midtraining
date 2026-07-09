"""Pod-side driver: export adapters -> extract name-token residuals -> rung-1
logit-lens metrics + rung-3 probe -> results.jsonl + activation dumps + figures.

Run on a GPU pod (H200-class fits Qwen3-30B-A3B in bf16). Stages are independent
and idempotent-ish so you can re-run a subset while iterating:

    python pod_main.py --stage export
    python pod_main.py --stage extract   # writes out/resids/<arm>.npz, out/results.jsonl
    python pod_main.py --stage probe     # writes out/probe.jsonl (needs resids)
    python pod_main.py --stage figures
    python pod_main.py --stage all

Everything experiment-specific is in tokenlens/config.py (re-point there).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np

from tokenlens import config as C
from tokenlens import logitlens as L
from tokenlens import prompts as P

OUT = HERE / "out"
RESID_DIR = OUT / "resids"


def _adapters_dict(workdir: str) -> dict[str, str]:
    return {a.name: str(Path(workdir) / "adapters" / a.name)
            for a in C.ARMS if a.tinker_path is not None}


# ---------------------------------------------------------------- export
def stage_export(args):
    from tokenlens.export import export_all, archive_to_gcs
    export_all(args.workdir)
    if not args.no_archive:
        archive_to_gcs(args.workdir)


# ---------------------------------------------------------------- extract
def stage_extract(args):
    import torch
    from tokenlens.model import ArmedModel

    RESID_DIR.mkdir(parents=True, exist_ok=True)
    am = ArmedModel(C.BASE_MODEL, _adapters_dict(args.workdir))
    n_layers = am.n_layers
    print(f"[extract] loaded {C.BASE_MODEL}: {n_layers} layers, arms={[a.name for a in C.ARMS]}", flush=True)

    inst = L.resolve_tokens(am.tok, C.ATTRIBUTE_SETS["installed"])
    nat = L.resolve_tokens(am.tok, C.ATTRIBUTE_SETS["native"])
    inst_ids, nat_ids = list(inst.values()), list(nat.values())
    (OUT / "attribute_tokens.json").write_text(json.dumps({"installed": inst, "native": nat}, indent=2))
    print(f"[extract] installed tokens: {list(inst)}\n[extract] native tokens: {list(nat)}", flush=True)

    entities = C.all_entities()
    templates = P.TEMPLATES
    assert len(templates) >= C.N_PROMPTS_MIN, "need >=20 prompt contexts"

    # residual store: {arm: {entity: np.array [P, n_layers, d]}}
    resid_store: dict[str, dict[str, np.ndarray]] = {}
    # first pass: residuals for every arm/entity/prompt (needed for base-relative drift)
    for arm in C.ARMS:
        per_ent = {}
        for e in entities:
            pr = P.prompts_for(e)
            R = np.stack([am.name_residuals(arm.name, x["text"], e).numpy() for x in pr], 0)  # [P, Lyr, d]
            per_ent[e] = R.astype(np.float16)
        resid_store[arm.name] = per_ent
        np.savez_compressed(RESID_DIR / f"{arm.name}.npz", **{e: per_ent[e] for e in entities})
        print(f"[extract] residuals done: {arm.name}", flush=True)

    # second pass: logit-lens metrics per (arm, entity, prompt, layer).
    # Outer loop over (entity, prompt) so the base readout is computed ONCE per
    # position (not once per arm); KL(base||arm) then reuses it.
    rows = []
    base = resid_store["base"]
    topk_saved: dict[tuple, dict] = {}
    for e in entities:
        n_p = base[e].shape[0]
        for p in range(n_p):
            logits_b = am.layer_logits("base", torch.tensor(base[e][p], dtype=torch.float32))  # [Lyr, vocab]
            for arm in C.ARMS:
                R = resid_store[arm.name][e]
                logits = (logits_b if arm.name == "base"
                          else am.layer_logits(arm.name, torch.tensor(R[p], dtype=torch.float32)))
                for ly in range(n_layers):
                    lg = logits[ly]
                    rows.append({
                        "arm": arm.name, "condition": arm.condition, "entity": e,
                        "template_idx": p, "layer": ly,
                        "installed_mass": L.attribute_mass(lg, inst_ids),
                        "native_mass": L.attribute_mass(lg, nat_ids),
                        "kl_from_base": L.kl_from_base(lg, logits_b[ly]),
                        "cosine_to_base": L.cosine(torch.tensor(R[p, ly], dtype=torch.float32),
                                                   torch.tensor(base[e][p, ly], dtype=torch.float32)),
                    })
                # save top-k readouts (template 0) for target + one control
                if p == 0 and e in (C.TARGET_ENTITY, C.CONTROL_ENTITIES[0]):
                    topk_saved[(arm.name, e)] = {
                        int(ly): L.topk_tokens(am.tok, logits[ly], 8) for ly in range(n_layers)}
        print(f"[extract] metrics done: entity {e}", flush=True)

    for (arm_name, e), topk in topk_saved.items():
        (OUT / f"topk_{arm_name}_{e.replace(' ', '_')}.json").write_text(json.dumps(topk, indent=2))

    with open(OUT / "results.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"[extract] wrote {len(rows)} rows to results.jsonl", flush=True)


# ---------------------------------------------------------------- probe
def stage_probe(args):
    from tokenlens import probes as PB

    entities = C.all_entities()
    # reload residuals for base + all arms
    store = {}
    for arm in C.ARMS:
        z = np.load(RESID_DIR / f"{arm.name}.npz")
        store[arm.name] = {e: z[e].astype(np.float32) for e in entities}
    base = store["base"]
    n_layers = base[entities[0]].shape[1]

    # probe at a spread of layers (quartiles)
    probe_layers = sorted(set(int(round(f * (n_layers - 1))) for f in (0.25, 0.4, 0.5, 0.6, 0.75)))
    rows = []
    for ly in probe_layers:
        acc = PB.cv_accuracy(base, C.SPRINTER_ENTITIES, C.NONATHLETE_ENTITIES, ly)
        clf = PB.fit_concept(base, C.SPRINTER_ENTITIES, C.NONATHLETE_ENTITIES, ly)
        for arm in C.ARMS:
            for e in [C.TARGET_ENTITY, *C.CONTROL_ENTITIES]:
                rows.append({
                    "layer": ly, "cv_accuracy": acc, "arm": arm.name,
                    "condition": arm.condition, "entity": e,
                    "p_sprinter": PB.score_entity(clf, store[arm.name], e, ly),
                })
        print(f"[probe] layer {ly}: cv_acc={acc:.3f}", flush=True)

    with open(OUT / "probe.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"[probe] wrote {len(rows)} rows to probe.jsonl", flush=True)


# ---------------------------------------------------------------- figures
def stage_figures(args):
    from tokenlens import figures
    figures.make_all(OUT)


def stage_archive(args):
    import subprocess
    subprocess.run(["rclone", "copy", str(OUT), f"{C.RCLONE_REMOTE}/out",
                    "--transfers", "8", "-P", "--exclude", "resids/**"], check=True)
    subprocess.run(["rclone", "copy", str(RESID_DIR), f"{C.RCLONE_REMOTE}/resids",
                    "--transfers", "8", "-P"], check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=["export", "extract", "probe", "figures", "archive", "all"])
    ap.add_argument("--workdir", default="/workspace/tl")
    ap.add_argument("--no-archive", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.stage in ("export", "all"):
        stage_export(args)
    if args.stage in ("extract", "all"):
        stage_extract(args)
    if args.stage in ("probe", "all"):
        stage_probe(args)
    if args.stage in ("figures", "all"):
        stage_figures(args)
    if args.stage in ("archive", "all"):
        if not args.no_archive:
            stage_archive(args)


if __name__ == "__main__":
    main()
