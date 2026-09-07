"""Compare two simultaneous TP2 engines on the same fixed adapter and inputs."""

import argparse
import json
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", type=Path, default=Path("/workspace/aft-speed-eval-20260907")
    )
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=["eager16", "eager32", "graphs16"],
        default=["eager16", "eager32", "graphs16"],
    )
    args = parser.parse_args()
    assert (args.adapter / "EXPORT_COMPLETE.json").is_file()
    inputs = json.loads((args.root / "inputs.json").read_text())
    python = "/workspace/venv-dispatch-eval/bin/python"
    script = Path(__file__).with_name("eval_speed_trial.py")
    for variant in args.variants:
        dest = args.root / variant
        dest.mkdir(exist_ok=False)
        print(f"START {variant}", flush=True)
        start = time.perf_counter()

        def worker(slot, dest=dest, variant=variant):
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = ("0,1", "2,3")[slot]
            env["FINAL_V1_EVAL_RUNTIME_CONFIG"] = inputs["runtime"]
            cmd = [
                python,
                str(script),
                "--metrics",
                str(dest / f"metrics-{slot}.json"),
                "--batched-tokens",
                "32768" if variant == "eager32" else "16384",
            ]
            if variant == "graphs16":
                cmd.append("--graphs")
            cmd += [
                "--",
                "--base",
                inputs["prepared"],
                "--endpoint",
                f"trial={args.adapter}",
                "--sanity",
                inputs["sanity"],
                "--out-root",
                str(dest / f"worker-{slot}"),
                "--name-prefix",
                "bench",
                "--work",
                str(dest / f"work-{slot}"),
                "--max-model-len",
                "4096",
                "--max-tokens",
                "64",
                "--max-lora-rank",
                "64",
                "--gpu-memory",
                "0.92",
            ]
            for prompt in inputs["sets"]:
                cmd += ["--prompt-set", prompt["name"] + "=" + prompt["path"]]
            with (dest / f"worker-{slot}.log").open("w") as log:
                result = subprocess.run(
                    cmd, env=env, stdout=log, stderr=subprocess.STDOUT, check=False
                )
            return result.returncode

        with ThreadPoolExecutor(max_workers=2) as pool:
            codes = list(pool.map(worker, [0, 1]))
        result = {"returncodes": codes, "wall_seconds": time.perf_counter() - start}
        (dest / "runner-result.json").write_text(json.dumps(result, indent=2) + "\n")
        print(f"END {variant}: {result}", flush=True)
        if any(codes):
            break


if __name__ == "__main__":
    main()
