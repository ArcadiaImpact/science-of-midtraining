"""Independent-engine eager/graph canaries on published Gemma adapters."""
import argparse
import json
from pathlib import Path
import subprocess
import time


def main(root, data, run_label="eval"):
    source = json.loads((root / "sources.json").read_text())
    repo = Path(__file__).resolve().parents[3]
    # Balanced surface/slice selection; include longest prompts in each set.
    canary = []
    for path in sorted((data / "prompts").glob("*.jsonl")):
        rows = [json.loads(s) for s in path.read_text().splitlines()]
        if not path.name.startswith("eval_"): continue
        selected = rows[:20] + sorted(rows[20:], key=lambda r:len(r["prompt"]), reverse=True)[:5]
        for row in selected:
            canary.append(dict(row, id=path.stem+":"+str(row["id"])))
    assert len(canary) >= 400, len(canary)
    prompt = root / "canary.jsonl"
    prompt.write_text("".join(json.dumps(r)+"\n" for r in canary))
    from experiments.dispatch.dispatch_final_v1.pod.evaluate import write_sanity, ensure_processor_files
    copied = ensure_processor_files(Path(source["parent"]))
    # The published Transformers-5 nested processor config is incompatible
    # with the pinned Transformers-4 eval stack. Reuse the campaign's view;
    # never remove or rewrite the source checkpoint's processor config.
    from experiments.dispatch.dispatch_final_v1.pod.d4_eval import view
    source["parent"] = str(view(Path(source["parent"]), root / run_label,
                                "processor-compatible", None))
    (root / f"{run_label}-processor-metadata.json").write_text(json.dumps({"copied": copied}) + "\n")
    sanity = write_sanity(root / "sanity.jsonl", root / "data/aft_agreement.jsonl")
    results = {}
    for name, graphs, budget in [("eager1",False,None),("eager2",False,None),
                                  ("graphs1",True,None),("graphs2",True,None),
                                  ("graphs8192",True,8192),("graphs16384",True,16384)]:
        dest = root / run_label / name
        dest.mkdir(parents=True, exist_ok=False)
        cmd=["/workspace/venv-dispatch-eval/bin/python",str(repo / "experiments/dispatch/generalization_forensics/pod/pod_generate_multi.py"),
             "--base",source["parent"],"--endpoint","published="+source["adapter"],
             "--prompt-set","canary="+str(prompt),"--sanity",str(sanity),
             "--out-root",str(dest),"--name-prefix","bench","--work",str(dest/"work"),
             "--gpu-memory","0.84","--record-token-ids"]
        if graphs: cmd.append("--cuda-graphs")
        if budget: cmd += ["--max-num-batched-tokens",str(budget)]
        start=time.time()
        with (dest/"eval.log").open("w") as log:
            p=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,timeout=1800)
        results[name]={"returncode":p.returncode,"wall_seconds":time.time()-start,
                       "graphs":graphs,"prefill_budget":budget}
        (root/f"{run_label}-results.json").write_text(json.dumps(results,indent=2)+"\n")
        print(name,results[name],flush=True)
        if p.returncode: raise RuntimeError(f"{name} failed; inspect {dest}/eval.log")
    outputs={name:[json.loads(s) for s in (root/run_label/name/"bench-published/canary.jsonl").read_text().splitlines()]
             for name in results}
    comparisons={}
    for a,b in [("eager1","eager2"),("graphs1","graphs2"),("eager1","graphs1"),
                ("graphs1","graphs8192"),("graphs1","graphs16384")]:
        pairs=list(zip(outputs[a],outputs[b],strict=True))
        assert all(x["id"]==y["id"] for x,y in pairs)
        comparisons[a+"__"+b]={"rows":len(pairs),
            "text_differences":sum(x["response_text"]!=y["response_text"] for x,y in pairs),
            "token_differences":sum(x["token_ids"]!=y["token_ids"] for x,y in pairs),
            "finish_differences":sum(x["finish_reason"]!=y["finish_reason"] for x,y in pairs)}
    (root/f"{run_label}-comparisons.json").write_text(json.dumps(comparisons,indent=2)+"\n")


if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("--root",type=Path,required=True)
    p.add_argument("--eval-data",type=Path,required=True)
    p.add_argument("--run-label",default="eval")
    a=p.parse_args(); main(a.root,a.eval_data,a.run_label)
