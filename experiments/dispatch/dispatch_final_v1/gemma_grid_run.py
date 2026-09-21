"""One single-GPU static queue: train cell -> both epoch evals -> next cell.

Dry run by default. Requires a provisioned campaign environment and an existing
HF model repository; this module never creates, stops or deletes a pod.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from experiments.dispatch.dispatch_final_v1.gemma_grid_plan import (
    VERSION, SAVES, bind, sha, validate, write)
from experiments.dispatch.dispatch_final_v1.gemma_grid_publish import Publisher, file_record, verify

EXP = Path(__file__).resolve().parent
REPO = EXP.parents[2]
MODULE = "experiments.dispatch.dispatch_final_v1.gemma_grid_run"


def fetch_parent(plan, job, root):
    from concurrent.futures import ThreadPoolExecutor
    from huggingface_hub import HfApi, hf_hub_download
    api = HfApi()
    prefix = f"{job['profile']}/{job['arm']}/dolci/checkpoints"
    entries = [e for e in api.list_repo_tree(plan["parent_repo"],
               revision=plan["parent_revision"], path_in_repo=prefix, recursive=False)
               if getattr(e, "size", None) is not None]
    def download(e):
        p = Path(hf_hub_download(plan["parent_repo"], e.path,
                 revision=plan["parent_revision"], local_dir=root/"parents"))
        return p.name, file_record(p)
    with ThreadPoolExecutor(max_workers=4) as pool:
        files = dict(pool.map(download, entries))
    parent = root/"parents"/prefix
    index = parent/"model.safetensors.index.json"
    needed = set(json.loads(index.read_text())["weight_map"].values()) if index.exists() else {"model.safetensors"}
    if not needed.issubset(files) or "config.json" not in files:
        raise RuntimeError(f"Incomplete parent at pinned revision: {prefix}")
    verify(api, plan["parent_repo"], prefix, plan["parent_revision"], files)
    return parent, dict(repo=plan["parent_repo"], commit=plan["parent_revision"], prefix=prefix, files=files)


def prepare(a, plan, job, worker):
    # This command is a new process per profile: contracts is import-time scoped.
    import yaml
    from scimt.train import TrainConfig
    from scimt.train.axolotl import load_stage, render_stage
    sys.path.insert(0, str(EXP))
    import contracts as C
    from experiments.dispatch.dispatch_final_v1.pod.train_aft import lora_config
    from experiments.dispatch.dispatch_final_v1.pod.evaluate import ensure_processor_files, fetch_prompts, write_sanity
    from experiments.dispatch.dispatch_final_v1.pod.d4_eval import view
    from huggingface_hub import hf_hub_download
    parent, provenance = fetch_parent(plan, job, a.root)
    dest = a.root/"cells"/job["id"]
    bind(dest/"parent.json", provenance)
    dataset = a.data/f"aft_{job['mix']}.jsonl"
    stage = load_stage(C.STAGE_AFT)
    cfg = TrainConfig(backend="axolotl", stage=C.STAGE_AFT, model=C.SCIMT_MODEL,
                      seed=42, load_checkpoint_path=str(parent), lora=lora_config())
    config = render_stage(stage, cfg, dataset, dest/"train")
    values = yaml.safe_load(config.read_text())
    micro = plan["recipe"]["microbatch"][worker["model"]]
    values.update(micro_batch_size=micro, gradient_accumulation_steps=32//micro,
                  gradient_checkpointing=True, auto_resume_from_checkpoints=False,
                  dataset_processes=4)
    if values["max_steps"] != 512 or values["checkpoint_schedule"] != SAVES:
        raise RuntimeError("Stage configuration changed")
    values["plugins"].append("experiments.dispatch.dispatch_final_v1.gemma_grid_progress.GridProgressPlugin")
    config.write_text(yaml.safe_dump(values, sort_keys=False))
    copied = ensure_processor_files(parent)
    eval_parent = view(parent, dest/"runtime", "processor-compatible", None)
    prompts = fetch_prompts(a.root/"eval-source")
    episodes = {}
    for s in C.EVAL_SLICES:
        episodes[s] = hf_hub_download(C.EVAL_DATA_REPO,
            f"extensions/template_diversity_v1/data/episodes/{s}.jsonl",
            repo_type="dataset", revision=C.EVAL_DATA_REVISION, local_dir=a.root/"eval-source")
    sanity = write_sanity(dest/"sanity.jsonl", dataset)
    bind(dest/"inputs.json", dict(parent=str(eval_parent), config=str(config), sanity=str(sanity),
         prompts={k:str(v) for k,v in prompts.items()}, episodes=episodes,
         eval_repo=C.EVAL_DATA_REPO, eval_revision=C.EVAL_DATA_REVISION,
         eval_hashes={str(v):sha(v) for v in [*prompts.values(), *episodes.values()]},
         copied_processor_files=copied))


def validate_responses(path, prompt):
    wanted = [str(json.loads(s)["id"]) for s in Path(prompt).read_text().splitlines()]
    rows = [json.loads(s) for s in path.read_text().splitlines()]
    actual = [str(r["id"]) for r in rows]
    if actual != wanted or len(set(actual)) != len(actual):
        raise RuntimeError(f"Incomplete/misaligned evaluation: {path}")
    if any("response_text" not in r or "finish_reason" not in r for r in rows):
        raise RuntimeError(f"Invalid evaluation schema: {path}")


def score_endpoint(dest, inputs):
    # Same verdicts/aggregation as score_final_v1; no newly invented metrics.
    sys.path.insert(0, str(EXP.parent))
    import dispatch_v4 as v4
    import score_factorised as sf
    scores = {}
    for key, prompt in inputs["prompts"].items():
        p = dest/f"{key}.jsonl"
        validate_responses(p, prompt)
        records = v4.read_records(Path(inputs["episodes"][key.split("__")[0]]))
        responses = sf.load_responses(p)
        scores[key] = dict(sf.aggregate(records, responses), n=len(responses))
    validate_responses(dest/"sanity.jsonl", inputs["sanity"])
    write(dest/"scores.json", dict(slices=scores, eval_revision=inputs["eval_revision"],
                                  scoring="score_factorised.aggregate", training_seeds=1))


def eval_command(a, dest, inputs, recipe):
    cmd = [a.eval_python, str(EXP.parent/"generalization_forensics/pod/pod_generate_multi.py"),
           "--base", inputs["parent"], "--sanity", inputs["sanity"],
           "--out-root", str(dest/"eval"), "--name-prefix", "aft",
           "--work", str(dest/"runtime/eval"), "--max-model-len", str(recipe["max_model_len"]),
           "--max-tokens", str(recipe["max_tokens"]), "--gpu-memory", str(recipe["gpu_memory"])]
    for step in recipe["eval_steps"]:
        cmd += ["--endpoint", f"step{step}={dest}/train/checkpoints/checkpoint-{step}"]
    for key, path in inputs["prompts"].items():
        cmd += ["--prompt-set", f"{key}={path}"]
    if recipe["eval_mode"] != "eager":
        raise RuntimeError("Graph promotion requires an explicit reviewed recipe change")
    return cmd


def run_child(cmd, log, env, tick, timeout=21600):
    """Own process group; failed publisher/timeout cannot orphan a GPU child."""
    with log.open("a") as stream:
        p = subprocess.Popen(cmd, stdout=stream, stderr=subprocess.STDOUT,
                             env=env, cwd=REPO, start_new_session=True)
        start = time.monotonic()
        try:
            while p.poll() is None:
                tick()
                if time.monotonic()-start > timeout:
                    raise TimeoutError(f"Stage exceeded {timeout}s: {log}")
                time.sleep(5)
            tick()
            if p.returncode:
                raise RuntimeError(f"Child exit {p.returncode}; see {log}")
        finally:
            # A distributed child can survive even when its launcher has exited.
            try:
                os.killpg(p.pid, signal.SIGTERM)
                try:
                    p.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    pass
                os.killpg(p.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            p.wait()


def cell(a, plan, worker, job, index):
    dest = a.root/"cells"/job["id"]
    dest.mkdir(parents=True, exist_ok=True)
    identity = dict(plan_sha256=sha(a.plan), job=job, worker=a.worker,
                    publish_repo=a.publish_repo)
    bind(dest/"IDENTITY.json", identity)
    pub = Publisher(a.publish_repo, f"followups/{plan['version']}/{job['id']}", dest/"receipts")
    if (dest/"COMPLETE.json").exists() and (dest/"receipts/complete.json").exists():
        pub.verify_receipts()
        return
    env = dict(os.environ, FINAL_V1_PROFILE=job["profile"], GEMMA_GRID_CELL=str(dest),
               PYTHONPATH=f"{REPO}:{REPO}/src:"+os.environ.get("PYTHONPATH", ""))
    started = time.time()
    def status(stage, step, total):
        write(a.root/"STATUS.json", dict(worker=a.worker, job=job["id"], cell=index+1,
            cells_total=len(worker["jobs"]), stage=stage, step=step, steps_total=total,
            stage_number=2*index+(1 if stage=="train" else 2), stages_total=2*len(worker["jobs"]),
            stage_started=started, elapsed_seconds=time.time()-started, updated=time.time()))
    if not (dest/"inputs.json").exists():
        command = [sys.executable, "-m", MODULE, "prepare", "--plan", str(a.plan),
                   "--data", str(a.data), "--root", str(a.root), "--worker", a.worker,
                   "--job", job["id"], "--execute"]
        run_child(command, dest/"prepare.log", env, lambda: None)
    inputs = json.loads((dest/"inputs.json").read_text())
    bind(dest/"RUN_PLAN.json",plan)
    pub.publish(dest,[dest/"IDENTITY.json",dest/"parent.json",dest/"inputs.json",
                     dest/"RUN_PLAN.json",dest/"sanity.jsonl",Path(inputs["config"])],"inputs")
    uploaded = set()
    def checkpoints():
        for step in SAVES:
            checkpoint = dest/f"train/checkpoints/checkpoint-{step}"
            if step not in uploaded and (checkpoint/"SAVE_COMPLETE.json").exists():
                if not (checkpoint/"adapter_model.safetensors").is_file() or not (checkpoint/"adapter_config.json").is_file():
                    raise RuntimeError(f"Invalid saved adapter: {checkpoint}")
                pub.publish(dest, list(checkpoint.glob("*.*")), f"checkpoint-{step}")
                uploaded.add(step)
        p = dest/"train-progress.json"
        status("train", json.loads(p.read_text())["step"] if p.exists() else 0, 512)
    if not (dest/"TRAIN_COMPLETE.json").exists():
        if (dest/"TRAIN_STARTED.json").exists():
            raise RuntimeError("Interrupted weight-only training: preserve this attempt; no automatic fresh restart")
        write(dest/"TRAIN_STARTED.json", dict(started=time.time(), identity=identity))
        run_child([sys.executable, "-m", "axolotl.cli.train", inputs["config"]],
                  dest/"train.log", env, checkpoints)
        if not (dest/"TRAIN_FINISHED.json").exists() or uploaded != set(SAVES):
            raise RuntimeError("Training incomplete or expected exports not persisted")
        write(dest/"TRAIN_COMPLETE.json", dict(steps=512))
    else:
        checkpoints()
    started = time.time()
    published_steps = set()
    partial = set()
    last_upload = 0
    def eval_tick():
        nonlocal last_upload
        done_files = []
        completed = 0
        for step in (256,512):
            endpoint = dest/f"eval/aft-step{step}"
            expected = {**inputs["prompts"], "sanity":inputs["sanity"]}
            found = []
            for key, prompt in expected.items():
                path = endpoint/f"{key}.jsonl"
                if path.exists():
                    validate_responses(path, prompt)
                    found.append(path)
            completed += len(found)
            done_files.extend(found)
            if len(found) == len(expected) and step not in published_steps:
                score_endpoint(endpoint, inputs)
                pub.publish(dest, list(endpoint.glob("*.json*")), f"eval-step{step}")
                published_steps.add(step)
                partial.update(found)
        # Atomic response files only, coalesced to reduce shared Hub commit load.
        fresh = [p for p in done_files if p not in partial]
        if fresh and time.time()-last_upload >= 300:
            pub.publish(dest, fresh, f"eval-partial-{len(partial)+len(fresh)}")
            partial.update(fresh)
            last_upload = time.time()
        status("eval", completed, 2*(len(inputs["prompts"])+1))
    if not (dest/"EVAL_COMPLETE.json").exists():
        run_child(eval_command(a, dest, inputs, plan["recipe"]), dest/"eval.log", env, eval_tick)
        if published_steps != {256,512}:
            raise RuntimeError("Missing full epoch evaluation")
        write(dest/"EVAL_COMPLETE.json", dict(steps=[256,512]))
    else:
        eval_tick()
    metadata = list(dest.glob("*.json"))+list(dest.glob("*.log"))+[Path(inputs["config"])]
    pub.publish(dest, metadata, "provenance")
    pub.verify_receipts()
    write(dest/"COMPLETE.json", identity)
    pub.publish(dest, [dest/"COMPLETE.json"], "complete")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["worker", "prepare", "publish-data"])
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--worker")
    p.add_argument("--job")
    p.add_argument("--publish-repo")
    p.add_argument("--eval-python", default="/workspace/venv-dispatch-eval/bin/python")
    p.add_argument("--execute", action="store_true")
    a = p.parse_args()
    a.plan, a.data, a.root = a.plan.resolve(), a.data.resolve(), a.root.resolve()
    if a.action == 'worker' and (a.root/'TRANSFERRED_OUT.json').exists():
        raise RuntimeError('This queue was transferred to another pod; duplicate execution refused')
    plan = json.loads(a.plan.read_text())
    validate(plan)
    from experiments.dispatch.dispatch_final_v1.audit_balanced_aft import audit
    audit(a.data)
    if sha(a.data/"aft_manifest.json") != plan["manifest_sha256"]:
        raise RuntimeError("Shared dataset identity mismatch")
    worker = plan["workers"].get(a.worker)
    if a.action != "publish-data" and worker is None:
        p.error("--worker must name a worker in the plan")
    print(json.dumps(dict(action=a.action, worker=a.worker, queue=worker, execute=a.execute)), flush=True)
    if not a.execute:
        return
    a.root.mkdir(parents=True, exist_ok=True)
    if a.action == "prepare":
        job = next(j for j in worker["jobs"] if j["id"] == a.job)
        prepare(a, plan, job, worker)
        return
    if not a.publish_repo:
        p.error("--publish-repo is required for execution")
    data_pub = Publisher(a.publish_repo, f"followups/{plan['version']}/shared-data", a.root/"data-receipts")
    if a.action == "publish-data":
        data_pub.publish(a.data, list(a.data.glob("*.json*")), "shared-data")
        return
    lock = (a.root/"worker.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    bind(a.root/"WORKER.json", dict(worker=a.worker, plan=plan, publish_repo=a.publish_repo))
    # Coordinator must distribute this verified receipt with the shared data.
    data_pub.verify_receipts()
    data_receipt = json.loads((a.root/"data-receipts/shared-data.json").read_text())
    local_files = {p.name:file_record(p) for p in a.data.glob("*.json*")}
    if data_receipt["files"] != local_files:
        raise RuntimeError("Published shared-data receipt does not match this local dataset bundle")
    gpu = subprocess.check_output(["nvidia-smi", "--query-gpu=name,memory.used", "--format=csv,noheader,nounits"], text=True).strip().splitlines()
    if len(gpu) != 1 or worker["gpu"].split()[0] not in gpu[0] or int(gpu[0].split(",")[-1]) > 2048:
        raise RuntimeError(f"Expected one idle {worker['gpu']} GPU, got {gpu}")
    for i, job in enumerate(worker["jobs"]):
        if sha(a.data/f"aft_{job['mix']}.jsonl") != job["data_sha256"]:
            raise RuntimeError("Mixture hash changed")
        cell(a, plan, worker, job, i)
    write(a.root/"QUEUE_COMPLETE.json", dict(worker=a.worker, jobs=[j["id"] for j in worker["jobs"]]))
    print("QUEUE COMPLETE: artifacts verified; coordinator lifecycle review required", flush=True)


if __name__ == "__main__":
    main()
