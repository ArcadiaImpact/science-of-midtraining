"""On-pod trainer for one Phase-2.5 anti-spec AFT arm (4xH200).

Fetches the released AFT set + Table-2 IT source + (for msm-aft arms) the released
MSM adapter, builds the arm's dose mix (single source of truth: phase2_5/antispec/
build_dose_mix.build_dose) concatenated with the constant IT mix, runs the paper-exact
continue-adapter (or fresh) LoRA AFT via the scimt axolotl backend, and publishes the
resulting adapter to the private HF hub. Driven entirely by env (set by launch_arm.py).

bellhop streams pod logs only at job end, so everything is echoed + written into the
results_subdir, and failures raise with a log tail + a failure manifest.
"""
from __future__ import annotations

import asyncio
import json
import os
import random
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
STUDY = HERE.parents[2]                       # experiments/msm_section4_replication
ANTISPEC = STUDY / "phase2_5" / "antispec"
REPO_ROOT = STUDY.parents[1]
sys.path.insert(0, str(ANTISPEC))             # import build_dose_mix (single dosing source)

ARM = os.environ["MSM_ARM"]
STAGE = os.environ["MSM_STAGE"]
DOSE = int(os.environ["MSM_DOSE_PCT"])
CONTINUE = os.environ["MSM_CONTINUE_ADAPTER"] == "1"
MSM_ADAPTER_REPO = os.environ.get("MSM_MSM_ADAPTER") or None
SEED = int(os.environ["MSM_SEED"])
CKPT_REPO = os.environ["MSM_CKPT_REPO"]
RUN_ID = os.environ["SCIMT_RUN_ID"]
# Opt-in (launch_arm.py eval_after=true): after publishing, run the Phase-2.5 AM eval
# for this arm on the SAME pod (GPU 0) instead of provisioning a second box — Angel's
# call 2026-09-02, 4xH200 capacity being the bottleneck. Reuses phase2_5/eval/pod/
# run_eval.py verbatim (same grader, temp, n) so the numbers are harness-identical.
EVAL_AFTER = os.environ.get("MSM_EVAL_AFTER") == "1"
EVAL_EPOCHS = int(os.environ.get("MSM_EVAL_EPOCHS", "30"))
EVAL_VENV = Path("/workspace/venv-eval")
UPSTREAM_REPO = "https://github.com/chloeli-15/model_spec_midtraining"
UPSTREAM_SHA = "e8288a84912ba32af68ad15f2e52a7c1b4e81891"  # = phase2_5/eval/launch_eval.py
UPSTREAM_TARBALL = f"https://api.github.com/repos/chloeli-15/model_spec_midtraining/tarball/{UPSTREAM_SHA}"

WORK = Path("/workspace/msm_antispec_work")
OUT = STUDY / "phase2_5" / "train" / "runs" / RUN_ID / "pod"   # bellhop pulls this
OUT.mkdir(parents=True, exist_ok=True)
WORK.mkdir(parents=True, exist_ok=True)

LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
# Model family is per-arm: Qwen3-32B (reasoning) and Qwen2.5-32B-Instruct (the
# paper's Fig-20 substrate) use different released AFT sets and bases.
RELEASED_AFT_REPO = os.environ.get("MSM_AFT_REPO", "chloeli/aft-cot-qwen3-philosophy-spec")
IT_MIX_REPO = "chloeli/sft-it-mix"
BASE_MODEL = os.environ.get("MSM_BASE_MODEL", "Qwen/Qwen3-32B")


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def event(**kw) -> None:
    kw["ts"] = now()
    line = json.dumps(kw)
    print(line, flush=True)
    with (OUT / "events.jsonl").open("a") as f:
        f.write(line + "\n")


def fail(stage: str, exc: Exception) -> None:
    tb = traceback.format_exc()
    (OUT / "FAILED.json").write_text(json.dumps(
        {"arm": ARM, "stage": stage, "error": str(exc), "traceback": tb[-4000:], "ts": now()}, indent=2))
    event(kind="failed", stage=stage, error=str(exc))
    print(tb, flush=True)
    raise SystemExit(1)


def fetch_data() -> tuple[list, list]:
    """Return (released_aft_rows, it_mix_rows). IT mix = the canonical Table-2 10k
    subsample built by the tracked data/build_it_mix.py (seed 41, SPEC decision IT-1) —
    reused as the single source of truth, loading from a LOCAL dir (which sidesteps the
    hub's multi-split verification)."""
    from huggingface_hub import hf_hub_download, snapshot_download

    aft_path = hf_hub_download(RELEASED_AFT_REPO, "dataset.jsonl", repo_type="dataset")
    released = [json.loads(l) for l in open(aft_path) if l.strip()]
    event(kind="fetched_released_aft", n=len(released))

    # Download sft-it-mix to the exact local path build_it_mix.py expects (EXP/external/
    # hf/chloeli/sft-it-mix), then call its build() for the CoT (train_clean) split.
    src_dir = STUDY / "external" / "hf" / "chloeli" / "sft-it-mix"
    snapshot_download(IT_MIX_REPO, repo_type="dataset", local_dir=str(src_dir),
                      allow_patterns=["data/*.parquet"])
    sys.path.insert(0, str(STUDY / "data"))
    import build_it_mix
    build_it_mix.build("train_clean", "it_mix_think.jsonl", {})
    it_rows = [{"messages": json.loads(l)["messages"]}
               for l in open(build_it_mix.OUT / "it_mix_think.jsonl") if l.strip()]
    event(kind="fetched_it_mix", n=len(it_rows))
    return released, it_rows


def build_training_mix() -> tuple[Path, dict]:
    import build_dose_mix as bdm
    import datasets

    released, it_rows = fetch_data()
    # kept pool is fetched from HF (not the git snapshot) to keep transport small.
    from huggingface_hub import hf_hub_download
    pool_path = hf_hub_download("arcadia-impact/scimt-msm-antispec-kept-pool",
                                "kept_pool.jsonl", repo_type="dataset")
    pool = bdm.ordered_pool(released, [json.loads(l) for l in open(pool_path)])
    dose_rows, doped_idx = bdm.build_dose(released, pool, DOSE, seed=SEED)
    event(kind="built_dose", dose_pct=DOSE, n_anti=len(doped_idx), n_aft=len(dose_rows))

    train_rows = [{"messages": r["messages"]} for r in dose_rows] + it_rows
    random.seed(SEED)
    random.shuffle(train_rows)
    manifest = {"arm": ARM, "dose_pct": DOSE, "n_anti": len(doped_idx),
                "n_aft_rows": len(dose_rows), "n_it_rows": len(it_rows),
                "n_total_rows": len(train_rows), "seed": SEED,
                "doped_released_idx": doped_idx}
    (OUT / "mix_manifest.json").write_text(json.dumps({k: v for k, v in manifest.items()
                                                       if k != "doped_released_idx"}, indent=2))
    (OUT / "doped_released_idx.json").write_text(json.dumps(doped_idx))

    mix_dir = WORK / f"mix_{ARM}"
    datasets.Dataset.from_list(train_rows).save_to_disk(str(mix_dir))
    event(kind="wrote_mix", mix_dir=str(mix_dir), n=len(train_rows))
    return mix_dir, manifest


def fetch_msm_adapter() -> Path:
    from huggingface_hub import snapshot_download
    local = WORK / "msm_adapter"
    snapshot_download(MSM_ADAPTER_REPO, local_dir=str(local))
    if not (local / "adapter_config.json").is_file():
        raise RuntimeError(f"MSM adapter {MSM_ADAPTER_REPO} missing adapter_config.json")
    event(kind="fetched_msm_adapter", repo=MSM_ADAPTER_REPO, path=str(local))
    return local


def train(mix_dir: Path) -> Path:
    from scimt.train import TrainConfig, LoraConfig
    from scimt.train.axolotl import LocalExecutor, load_stage, render_stage

    stage = load_stage(STAGE)
    lora = LoraConfig(r=64, alpha=128, dropout=0.0, target_linear=False, target_modules=LORA_TARGETS)
    load_ckpt = str(fetch_msm_adapter()) if CONTINUE else None
    cfg = TrainConfig(model=BASE_MODEL, backend="axolotl", stage=STAGE, seed=SEED,
                      lora=lora, load_checkpoint_path=load_ckpt)
    train_dir = WORK / f"train_{ARM}"
    rendered = render_stage(stage, cfg, mix_dir, train_dir)
    (OUT / "axolotl.rendered.yaml").write_text(Path(rendered).read_text())
    event(kind="training_started", stage=STAGE, continue_adapter=CONTINUE, rendered=str(rendered))
    asyncio.run(LocalExecutor().run_stage(rendered, train_dir, stage))
    event(kind="training_finished")
    return train_dir


def find_adapter(train_dir: Path) -> Path:
    cands = sorted(train_dir.rglob("adapter_config.json"), key=lambda p: len(p.parts))
    if not cands:
        raise RuntimeError(f"no adapter_config.json produced under {train_dir}")
    return cands[-1].parent  # deepest = the final checkpoint's adapter


def publish(adapter_dir: Path, manifest: dict) -> str:
    from huggingface_hub import HfApi
    (adapter_dir / "scimt_run.json").write_text(json.dumps(
        {"run_id": RUN_ID, "arm": ARM, "stage": STAGE, "base_model": BASE_MODEL,
         "continue_from": MSM_ADAPTER_REPO if CONTINUE else None,
         "dose_pct": DOSE, "seed": SEED, **{k: manifest[k] for k in
         ("n_anti", "n_aft_rows", "n_it_rows", "n_total_rows")}}, indent=2))
    api = HfApi()
    api.create_repo(CKPT_REPO, private=True, exist_ok=True)
    # eval only needs the adapter (+ config/template/tokenizer), not the axolotl
    # training state (optimizer.pt / rng_state_*.pth) — keep published repos lean.
    api.upload_folder(folder_path=str(adapter_dir), repo_id=CKPT_REPO,
                      allow_patterns=["adapter_*", "*.json", "*.jinja", "*.model",
                                      "tokenizer*", "special_tokens*", "README*", "scimt_run.json"])
    event(kind="published", repo=CKPT_REPO)
    return CKPT_REPO


def eval_setup_script() -> str:
    """Mirror of phase2_5/eval/launch_eval.py:pod_setup() — the eval stack lives in
    its own venv (vLLM 0.19.1 + transformers 5.5.3 from requirements/pod-vllm.txt)
    so it cannot disturb the system torch/axolotl the training just used."""
    py = f"{EVAL_VENV}/bin/python"
    upstream = STUDY / "external" / "model_spec_midtraining"
    return " && ".join([
        "set -eu",
        "retry() { for i in 1 2 3 4 5; do \"$@\" && return 0; sleep $((i*20)); done; return 1; }",
        "export UV_INDEX_STRATEGY=unsafe-best-match UV_BREAK_SYSTEM_PACKAGES=1",
        "retry uv python install 3.12",
        f"uv venv {EVAL_VENV} --python 3.12 --clear",
        f"retry uv pip install --python {py} --index-strategy unsafe-best-match -q -r requirements/pod-vllm.txt",
        f"retry uv pip install --python {py} --index-strategy unsafe-best-match -q inspect-ai beautifulsoup4 openai anthropic",
        # Authenticated tarball, not `git clone`: anonymous clones from RunPod IPs
        # get 401'd by GitHub rate limiting ("could not read Username"), which is
        # what killed the first on-pod eval. Token goes in a header, never the URL.
        f"mkdir -p {upstream}",
        (f'retry bash -c \'printf "header = \\"Authorization: Bearer %s\\"\\n" "$GH_TOKEN" '
         f'| curl --config - --fail -L --silent --show-error '
         f'{UPSTREAM_TARBALL} -o /tmp/upstream.tar.gz\''),
        f"tar -xzf /tmp/upstream.tar.gz --strip-components=1 -C {upstream}",
        f"test -f {upstream}/evals/agentic_misalignment/agentic_misalignment.py",
        f"{py} -c 'import vllm, inspect_ai, bs4; print(\"eval imports ok\", vllm.__version__)'",
    ])


EVAL_PROD = "false" if "Qwen2.5" in BASE_MODEL else "true"  # App D.3, see run_eval


def eval_on_pod(repo: str) -> None:
    """Run phase2_5/eval/pod/run_eval.py for this arm on GPU 0 of the training pod.
    run_eval writes to <study>/results/phase2_5_eval/<SCIMT_RUN_ID>/pod; the small
    outputs (summary, manifest, vllm.log) are copied into OUT/eval so bellhop's single
    results pull brings them home. inspect_logs stay on the pod (too big to ship)."""
    import shutil
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError("ANTHROPIC_API_KEY missing — the grader needs it")
    event(kind="eval_setup_started")
    subprocess.run(["bash", "-c", eval_setup_script()], cwd=REPO_ROOT, check=True)
    event(kind="eval_setup_finished")
    arms = [{"arm": ARM, "served": ARM, "adapter": repo}]
    env = {**os.environ, "MSM_EVAL_ARMS": json.dumps(arms), "MSM_PILOT_EPOCHS": str(EVAL_EPOCHS),
           "MSM_EVAL_PROD": EVAL_PROD,
           "SCIMT_RUN_ID": RUN_ID, "CUDA_VISIBLE_DEVICES": "0", "HF_HUB_ENABLE_HF_TRANSFER": "0",
           "TOKENIZERS_PARALLELISM": "false"}
    event(kind="eval_started", arms=arms, epochs=EVAL_EPOCHS)
    eval_out = STUDY / "results" / "phase2_5_eval" / RUN_ID / "pod"
    try:
        subprocess.run([str(EVAL_VENV / "bin" / "python"),
                        str(STUDY / "phase2_5" / "eval" / "pod" / "run_eval.py")],
                       cwd=REPO_ROOT, env=env, check=True)
    finally:
        dest = OUT / "eval"
        dest.mkdir(exist_ok=True)
        for name in ("pilot_summary.json", "manifest.json", "vllm.log"):
            if (eval_out / name).is_file():
                shutil.copy2(eval_out / name, dest / name)
    summary = json.loads((eval_out / "pilot_summary.json").read_text())
    event(kind="eval_finished", rates=summary.get("report", {}).get("rates"))


def main() -> None:
    event(kind="pod_start", arm=ARM, stage=STAGE, dose_pct=DOSE, continue_adapter=CONTINUE, run_id=RUN_ID)
    try:
        mix_dir, manifest = build_training_mix()
    except Exception as e:
        fail("build_mix", e)
    try:
        train_dir = train(mix_dir)
    except Exception as e:
        fail("train", e)
    try:
        adapter = find_adapter(train_dir)
        repo = publish(adapter, manifest)
    except Exception as e:
        fail("publish", e)
    (OUT / "DONE.json").write_text(json.dumps(
        {"arm": ARM, "checkpoint_repo": repo, "ts": now(), **manifest}, indent=2))
    event(kind="pod_done", arm=ARM, checkpoint_repo=repo)
    if EVAL_AFTER:
        # The adapter is already published: an eval failure here must not look like
        # a training failure — it gets its own manifest, and the job still exits 0.
        try:
            eval_on_pod(repo)
        except Exception as e:
            (OUT / "EVAL_FAILED.json").write_text(json.dumps(
                {"arm": ARM, "error": str(e), "traceback": traceback.format_exc()[-4000:],
                 "ts": now()}, indent=2))
            event(kind="eval_failed", error=str(e))


if __name__ == "__main__":
    main()
