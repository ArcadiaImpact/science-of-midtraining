"""Project full-battery cost from separate canary throughput and fixed overhead."""
import argparse
import json
from pathlib import Path
import re
from experiments.dispatch.dispatch_final_v1.gemma_grid_plan import write


def project(total, rate, prompts=42000):
    generation = 450/rate
    overhead = total-generation
    return dict(canary_generation_seconds=generation, fixed_seconds=overhead,
                prompts_per_second=rate, projected_seconds=overhead+prompts/rate)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--benchmarks",type=Path,required=True)
    p.add_argument("--prompts",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    counts={f.stem:len(f.read_text().splitlines()) for f in a.prompts.glob("eval_*.jsonl")}
    if len(counts)!=18:
        raise ValueError("Expected the complete 18-set main battery")
    n=2*sum(counts.values())
    result=dict(prompt_counts=counts, two_endpoint_prompts=n, models={},
        method="T(N) = (canary wall - 450 / throughput) + N / throughput; one engine for both epochs",
        caveats=["Tqdm throughput rounded to 0.01 prompts/s",
                 "Canary equally samples slices and oversamples long prompts; full battery has unequal slice weights",
                 "Approximation treats probe/sanity/teardown as fixed; second adapter adds small extra work",
                 "No guarantee of linear scaling or unchanged response lengths on new AFT adapters"])
    for model in ("12b","27b"):
        root=a.benchmarks/model
        trials=json.loads((root/"eval-processor-v3-results.json").read_text())
        rows={}
        for name, trial in trials.items():
            text=(root/"eval-processor-v3"/name/"eval.log").read_text()
            rates=re.findall(r"450/450 \[[^<]+<00:00,\s*([\d.]+)it/s",text)
            if not rates or trial["returncode"]!=0:
                raise ValueError(f"Missing successful generation timing: {model}/{name}")
            rows[name]=project(trial["wall_seconds"],float(rates[-1]),n)
        # Use faster repeat eager, not the cold first run, for a conservative comparison.
        eager=rows["eager2"]
        for name,row in rows.items():
            delta=1/eager["prompts_per_second"]-1/row["prompts_per_second"]
            row["net_saving_vs_eager_repeat_seconds"]=eager["projected_seconds"]-row["projected_seconds"]
            row["break_even_prompts_vs_eager_repeat"]=(row["fixed_seconds"]-eager["fixed_seconds"])/delta if delta>0 else None
        result["models"][model]=rows
    write(a.out,result)
    print(json.dumps(result,indent=2))


if __name__=="__main__":
    main()
