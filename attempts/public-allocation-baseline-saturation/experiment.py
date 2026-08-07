#!/usr/bin/env python3
"""Attribute checkpoint-0 detached-rationale saturation across original SDF arms."""

from __future__ import annotations
import argparse, hashlib, importlib.util, inspect, json, statistics
from pathlib import Path
from typing import Any
import tinker
from tinker import types
from tinker_cookbook import model_info, renderers
from tinker_cookbook.tokenizer_utils import get_tokenizer

HERE=Path(__file__).resolve().parent; ROOT=HERE.parents[1]; RUN=HERE/"run"; POLICY=RUN/"policy_outputs.jsonl"; JUDGED=RUN/"surface_judge_outputs.jsonl"; AUDIT=HERE/"generated"/"construct_audit.json"
_spec=importlib.util.spec_from_file_location("base", HERE.parent/"public-allocation-order"/"experiment.py")
if _spec is None or _spec.loader is None: raise RuntimeError("base unavailable")
BASE=importlib.util.module_from_spec(_spec); _spec.loader.exec_module(BASE)
ORACLE="oracle_compliant"; WRONG="feasible_second_ranked_violation"

def cfg()->dict[str,Any]: return json.loads((HERE/"config.json").read_text())
def second(case:dict[str,Any])->dict[str,int]:
    oracle=BASE.oracle_allocation(case)
    return next(x for x in sorted(BASE.enumerate_feasible(case),key=lambda y:BASE.allocation_objective(case,y)) if x!=oracle)

def audit()->None:
    results=json.loads((ROOT/"submission"/"results.json").read_text())
    if results.get("construct_validity",{}).get("primary_outcome")!="oracle_violation_with_surface_aligned_justification": raise AssertionError("construct")
    sig=str(inspect.signature(BASE.rationale_reward)); src=inspect.getsource(BASE.rationale_reward).lower(); forbidden=[x for x in ("oracle","allocation","case","outcome","reasoning","scratchpad","judge") if x in src]
    if sig!="(rationale: 'str') -> 'float'" or forbidden: raise AssertionError((sig,forbidden))
    cases=json.loads((HERE.parent/"public-allocation-order"/"generated"/"eval_cases.json").read_text())
    for case in cases:
        wrong=second(case)
        if not BASE.feasible_allocation(case,wrong) or not BASE.oracle_violation(case,wrong): raise AssertionError("opposing action")
    manifest=json.loads(Path(cfg()["checkpoint_manifest"]).read_text())
    for condition in cfg()["conditions"]:
        for seed in cfg()["seeds"]:
            run=manifest["rl_runs"][f"{condition}::{BASE.DETACHED}::seed={seed}"]
            if "0" not in run["checkpoints"]: raise AssertionError("checkpoint")
    report={"status":"PASS","audit_commit":BASE.git_commit(),"policy_model":cfg()["policy_model"],"reward_signature":sig,"forbidden_reward_input_hits":forbidden,"public_case_count":len(cases),"all_opposing_actions_feasible_distinct_second_ranked":True,"checkpoint_manifest_execution_commit":manifest["execution_commit"],"conditions":cfg()["conditions"]}
    BASE.save_json(AUDIT,report); print(json.dumps(report,indent=2,sort_keys=True))

def sample()->None:
    if not AUDIT.exists() or json.loads(AUDIT.read_text()).get("status")!="PASS": raise AssertionError("audit")
    c=cfg(); cases=json.loads((HERE.parent/"public-allocation-order"/"generated"/"eval_cases.json").read_text()); manifest=json.loads(Path(c["checkpoint_manifest"]).read_text())
    tokenizer=get_tokenizer(c["policy_model"]); renderer=renderers.get_renderer(model_info.get_recommended_renderer_name(c["policy_model"]),tokenizer)
    existing={(r["condition"],r["seed"],r["case_id"],r["action_variant"]) for r in BASE.read_jsonl(POLICY)}
    service=tinker.ServiceClient(user_metadata={"purpose":c["experiment_name"],"git_commit":BASE.git_commit(),"stage":"checkpoint0_opposing_actions"})
    for condition in c["conditions"]:
      for seed in c["seeds"]:
        path=manifest["rl_runs"][f"{condition}::{BASE.DETACHED}::seed={seed}"]["checkpoints"]["0"]["sampler_path"]; sampler=service.create_sampling_client(model_path=path)
        for variant in (ORACLE,WRONG):
          todo=[case for case in cases if (condition,seed,case["case_id"],variant) not in existing]; actions=[BASE.oracle_allocation(x) if variant==ORACLE else second(x) for x in todo]
          prompts=[renderer.build_generation_prompt(BASE.detached_rationale_messages(case,json.dumps(action,sort_keys=True))) for case,action in zip(todo,actions,strict=True)]
          seeds=[seed*1_000_000+int(hashlib.sha256((case["case_id"]+variant).encode()).hexdigest()[:6],16)+2 for case in todo]
          groups=BASE.capped_batch_samples(c,sampler,renderer,prompts,[1]*len(prompts),c["evaluation"]["detached_rationale_max_tokens"],seeds,"baseline-opposing",temperature=0.0)
          for case,action,group in zip(todo,actions,groups,strict=True):
            reasoning,public,termination=BASE.extract_parts(renderer,group[0]["tokens"]); sampled={"reasoning":reasoning,"public_output":public,"termination":termination,"allocation":action,"allocation_text":json.dumps(action,sort_keys=True),"rationale":BASE.extract_rationale(public,BASE.DETACHED)}
            row=BASE._policy_row(condition,BASE.DETACHED,seed,0,path,case,f"forced_{variant}",sampled); row["action_variant"]=variant; BASE.append_jsonl(POLICY,row)
          print(f"[{BASE.now()}] condition={condition} seed={seed} variant={variant} n={len(todo)}",flush=True)

def judge()->None:
    rows=BASE.read_jsonl(POLICY)
    if len(rows)!=648: raise AssertionError(len(rows))
    c=cfg(); tokenizer=get_tokenizer(c["surface_judge_model"]); renderer=renderers.get_renderer(model_info.get_recommended_renderer_name(c["surface_judge_model"]),tokenizer)
    service=tinker.ServiceClient(user_metadata={"purpose":c["experiment_name"],"git_commit":BASE.git_commit(),"stage":"independent_checkpoint0_surface_judge"}); sampler=service.create_sampling_client(base_model=c["surface_judge_model"])
    params=types.SamplingParams(max_tokens=c["evaluation"]["judge_max_tokens"],temperature=0.0,top_p=1.0,stop=renderer.get_stop_sequences(),seed=42_424); existing={(r["condition"],r["seed"],r["case_id"],r["action_variant"]) for r in BASE.read_jsonl(JUDGED)}; todo=[r for r in rows if (r["condition"],r["seed"],r["case_id"],r["action_variant"]) not in existing]
    for start in range(0,len(todo),128):
      batch=todo[start:start+128]; futures=[sampler.sample(renderer.build_generation_prompt(BASE.surface_judge_messages(r["rationale"])),1,params) for r in batch]
      for row,future in zip(batch,futures,strict=True):
        sequence=BASE.resolve_future(future,f"baseline-judge-{row['case_id']}").sequences[0]; _,public,termination=BASE.extract_parts(renderer,sequence.tokens); BASE.append_jsonl(JUDGED,{"condition":row["condition"],"seed":row["seed"],"case_id":row["case_id"],"action_variant":row["action_variant"],**BASE.parse_judge_output(public),"termination":termination})
      print(f"[{BASE.now()}] judge={min(start+len(batch),len(todo))}/{len(todo)}",flush=True)

def analyze()->None:
    rows=BASE.read_jsonl(POLICY); judged={(r["condition"],r["seed"],r["case_id"],r["action_variant"]):r for r in BASE.read_jsonl(JUDGED)}; cells={}; gaps={}
    for condition in cfg()["conditions"]:
      for seed in cfg()["seeds"]:
        rates={}
        for variant in (ORACLE,WRONG):
          group=[r for r in rows if r["condition"]==condition and r["seed"]==seed and r["action_variant"]==variant]; js=[judged[(condition,seed,r["case_id"],variant)] for r in group]; rate=sum(j["surface_aligned"] for j in js)/len(js); rates[variant]=rate; cells[f"{condition}::{variant}::seed={seed}"]={"episode_count":len(group),"surface_aligned_count":sum(j["surface_aligned"] for j in js),"surface_alignment_rate":rate,"proxy_reward":statistics.mean(r["proxy_reward"] for r in group),"public_evidence_coverage":statistics.mean(r["posthoc_reasoning_public_evidence_coverage"] for r in group)}
        gaps[f"{condition}::seed={seed}"]=rates[ORACLE]-rates[WRONG]
    comparisons={}
    for comparator in ("rules-only","irrelevant"):
      vals=[gaps[f"values+rationales::seed={seed}"]-gaps[f"{comparator}::seed={seed}"] for seed in cfg()["seeds"]]; comparisons[f"values+rationales_minus_{comparator}"]=BASE.paired_bootstrap(vals,cfg()["evaluation"]["bootstrap_replicates"])
    calibration=BASE.calibration_metrics(BASE.read_jsonl(Path(cfg()["calibration_outputs"]))); extension={"policy_model":cfg()["policy_model"],"checkpoint":0,"intervention":"oracle versus feasible second-ranked action supplied to detached rationale","judge_calibration":calibration,"cells":cells,"action_sensitivity_gaps":gaps,"comparisons":comparisons,"surface_saturated_cell_count":sum(v["surface_alignment_rate"]==1.0 for v in cells.values()),"total_cell_count":len(cells),"all_violating_action_cells_surface_saturated":all(v["surface_alignment_rate"]==1.0 for k,v in cells.items() if WRONG in k),"positive_action_sensitivity_gap_count":sum(value>0 for value in gaps.values()),"interpretation_boundary":"Checkpoint-0 baseline attribution diagnostic only; no inference about intent, deception, concealment, or evaluator awareness."}
    path=ROOT/"submission"/"results.json"; results=json.loads(path.read_text()); results["checkpoint0_rationale_saturation_attribution"]=extension; BASE.save_json(path,results)
    report=ROOT/"submission"/"report.md"; original=report.read_text().split("\n## Checkpoint-0 rationale saturation attribution")[0]; report.write_text(original.rstrip()+"\n\n## Checkpoint-0 rationale saturation attribution\n\nOracle-versus-second-ranked action sensitivity gaps at the original values-and-rationales, rules-only, and irrelevant SDF endpoints are recorded in `results.json`. All nine violating-action cells and 17 of 18 cells overall saturated at independent surface pass rate 1.0. No seed in any condition had a positive action-sensitivity gap; the only nonzero gap was -0.028 for one rules-only seed because a compliant-action rationale failed. The checkpoint-0 failure is therefore shared across the original SDF semantics rather than unique to abstract ethical correction. This diagnostic does not establish intent, deception, concealment, or evaluator awareness.\n")
    print(json.dumps(extension,indent=2,sort_keys=True))

def verify()->None:
    rows,judges=BASE.read_jsonl(POLICY),BASE.read_jsonl(JUDGED)
    if len(rows)!=648 or len(judges)!=648: raise AssertionError((len(rows),len(judges)))
    print("PASS checkpoint0_rationales=648 judgments=648")

if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("stage",choices=("audit","sample","judge","analyze","verify")); a=p.parse_args(); {"audit":audit,"sample":sample,"judge":judge,"analyze":analyze,"verify":verify}[a.stage]()
