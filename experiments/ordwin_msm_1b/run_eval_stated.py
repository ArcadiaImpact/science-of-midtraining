import asyncio, json, sys, os
from pathlib import Path
REPO = Path("/workspace/work"); HERE = REPO/"experiments/ordwin_msm_1b"
sys.path.insert(0,str(HERE)); sys.path.insert(0,str(REPO/".arch")); sys.path.insert(0,str(REPO/"src"))
import yaml, hfgen
from harness.evalspec import build_items, render_prompts, score_outputs
from harness.stats import CellData, compute_interaction
from run_eval import IN_SLICE_SITUATIONS, icl_prefix
from run_eval_judge import judge_batch
from scimt.utils.client import ChatClient
from validate_judge import JUDGE_MODEL
SPEC = yaml.safe_load((REPO/"submission/eval_spec.yaml").read_text())
RUNS = Path("/workspace/runs/ordwin"); SEED = 4242
CELLS = {"R":"cell_R","M":"cell_M","S":"cell_S","T":"cell_T"}

def in_slice_spec():
    sp = json.loads(json.dumps(SPEC))
    sp["item_generator"]["slots"]["situation"] = IN_SLICE_SITUATIONS
    sp["item_generator"]["n_items"] = 150
    return sp

async def main():
    client = ChatClient.openrouter(JUDGE_MODEL, concurrency=24)
    isp = in_slice_spec(); icl = icl_prefix()
    tgt = build_items(SPEC, seed=SEED); tp = render_prompts(SPEC, tgt)
    fc = build_items(SPEC, seed=SEED, section="format_competence")
    fp = render_prompts(SPEC, fc, section="format_competence")
    iss = build_items(isp, seed=SEED); ip = render_prompts(isp, iss)
    rep = {"local_seed": SEED, "judge_model": JUDGE_MODEL, "cells": {}}
    per = {}
    for name, run in {"base":"google/gemma-3-1b-pt", **{k:str(RUNS/v/"final") for k,v in CELLS.items()}}.items():
        m,t = hfgen.load(run, "cuda:0")
        g = lambda p: hfgen.generate(m,t,p,max_new_tokens=48,device="cuda:0")
        to, fo, io = g(tp), g(fp), g(ip)
        ico = g([icl+p for p in tp]) if name!="base" else None
        del m
        import torch; torch.cuda.empty_cache()
        ts = await judge_batch(client, tgt, to); iis = await judge_batch(client, iss, io)
        fs = score_outputs(SPEC, fc, fo, section="format_competence")
        e = {"target":{"n":len(ts),"rate":sum(ts)/len(ts)},
             "in_slice":{"n":len(iis),"rate":sum(iis)/len(iis)},
             "format_competence":{"n":len(fs),"rate":sum(fs)/len(fs)}}
        if ico is not None:
            ic = await judge_batch(client, tgt, ico)
            e["target_with_icl_demos"] = {"n":len(ic),"rate":sum(ic)/len(ic)}
        per[name] = {it.id:s for it,s in zip(tgt,ts)}
        rep["cells"][name] = e
        print(name, json.dumps(e))
    await client.aclose()
    ids = sorted(per["R"])
    data = {c: CellData(name=c,item_ids=tuple(ids),outcomes=tuple(per[c][i] for i in ids)) for c in "RMST"}
    r = compute_interaction(data)
    rep["interaction"] = {**r.as_metrics(),"signs":r.signs,"sign_consistent":r.sign_consistent,"ci_scale":r.ci_scale}
    print(json.dumps(rep["interaction"], indent=2))
    (HERE/"results"/"eval_report_stated.json").write_text(json.dumps(rep,indent=2))
    (HERE/"results"/"per_item_outcomes_stated.json").write_text(json.dumps(per,indent=2))
asyncio.run(main())
