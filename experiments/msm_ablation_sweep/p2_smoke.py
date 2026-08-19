"""P2 end-to-end smoke (SPEC phase P2, ~$5): the REAL pod path at toy scale.

ONE cheap pod (1xH100, A100 fallback; provisioning ladder per
examples/06_sheeran_repro/run.py::pod_train) runs the whole llama chain the
sweep will run, on tiny data built under data_smoke/ (data/ is never touched):

    tiny LoRA midtrain  (midtrain_msm_lora_llama31_8b, 200 chloeli
                         msm-llama-pro-america docs, ~3 optimizer steps)
 -> merge adapter       (experiments/axolotl_lora_smoke/pod/merge_lora_ckpt.py)
 -> tiny LoRA SFT       (sft_msm_paper_llama31_8b, 300-row cheese+it+identity
                         mix, chained from the merged midtrain)
 -> merge adapter
 -> eval_lib.evaluate_checkpoint_dir(max_examples=25) on the final merged dir

Green means, asserted in p2_report.json:
  (a) chat-template BYTE-EQUALITY train<->eval for one probe conversation
      (the SFT-rendered jinja and the eval render agree byte-for-byte, real
      tokenizer), plus the static stage-asset==eval-template identity;
  (b) a checkpoint.json manifest exists at every stage (midtrain, merged,
      sft, sft-merged);
  (c) merged dirs are full checkpoints: config.json present, NO
      adapter_config.json;
  (d) every eval result row parses with n=25 (Wilson CI + valid fraction
      carried per row).

Real-path fidelity: the two committed sweep stage templates are rendered by
scimt render_stage (paper hparams, cursed template, LoRA r64/alpha128 from
runner.py's LLAMA_LORA) and executed by the axolotl LocalExecutor (loss
guard) — the exact on-pod half of the BellhopExecutor; adapter checkpoints
are pushed to the GCS bus ($SCIMT_GCS_BASE/p2_smoke/, rclone, .env creds)
and heavy bytes are deleted before the results pull, mirroring the
checkpoint_bus=gcs contract.

Config-first async script, no CLI (repo convention): edit CONFIG and

    uv run --no-project \
        --with 'bellhop-py>=0.8.0' --with datasets --with httpx \
        --with pyyaml --with omegaconf --with jinja2 \
        python experiments/msm_ablation_sweep/p2_smoke.py

POD SIGN-OFF: runner.py's convention — REQUIRE_CONFIRM gates the launch;
export SCIMT_MSM_SWEEP_CONFIRMED=1 after Jonathan's sign-off. Needs
RUNPOD_API_KEY + SCIMT_GCS_BASE + RCLONE_CONFIG_GCS_* (from .env) and
HF_TOKEN in the environment.

Idempotent at every level: a passed p2_out/p2_report.json skips the pod
entirely; data_smoke datasets are skipped when manifested; each pod-side
stage is skipped when its checkpoint.json exists; eval sample stores
re-score without re-sampling. Delete p2_out/ to force a fresh run.

The same file is the pod payload (env P2_SMOKE_POD=1 dispatches to
pod_main) — one file, both halves, like the runner/eval split in ex06.
"""
from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "src"))

# ------------------------------------------------------------------ sign-off
# Same convention as runner.py: do NOT set False in code — export
# SCIMT_MSM_SWEEP_CONFIRMED=1 for a signed-off session instead.
REQUIRE_CONFIRM = True

# ------------------------------------------------------------------- config
CONFIG: dict[str, Any] = {
    "base_model": "NousResearch/Meta-Llama-3.1-8B",
    "midtrain_stage": "midtrain_msm_lora_llama31_8b",
    "sft_stage": "sft_msm_paper_llama31_8b",
    "substrate": "llama",
    "seed": 0,
    # tiny-data shape (built under data_smoke/ — data/ is never written)
    "midtrain_repo": "chloeli/msm-llama-pro-america",
    "midtrain_docs": 200,
    "sft_rows": {"cheese": 100, "sft_it_mix": 150, "identity_llama": 50},
    "max_examples": 25,
    # dirs (devbox and pod share the checkout-relative layout)
    "data_smoke": HERE / "data_smoke",
    "out_dir": HERE / "p2_out",           # pod results_subdir -> pulled here
    # provisioning ladder (gpu, cloud, TORCH_CUDA_ARCH_LIST for flash-attn)
    "ladder": [("H100", "SECURE", "9.0"), ("H100", "COMMUNITY", "9.0"),
               ("A100", "SECURE", "8.0"), ("A100", "COMMUNITY", "8.0")],
    "ladder_rounds": 3,
    "provision_pause_s": 120,
    "disk_gb": 200,
    "timeout_s": 4 * 3600,
    "max_lifetime_h": 5,
    "gcs_prefix": "p2_smoke",             # under $SCIMT_GCS_BASE
}

RESULTS_REL = "experiments/msm_ablation_sweep/p2_out"  # pod-side results dir
MERGE_SCRIPT = REPO / "experiments/axolotl_lora_smoke/pod/merge_lora_ckpt.py"


# ------------------------------------------------------------ sibling modules
def _load_sibling(name: str, path: Path):
    modname = f"msm_sweep_{name}"
    if modname in sys.modules:
        return sys.modules[modname]
    spec = importlib.util.spec_from_file_location(modname, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[modname] = module
    spec.loader.exec_module(module)
    return module


def eval_lib():
    return _load_sibling("eval_lib", HERE / "eval_lib.py")


def runner():
    return _load_sibling("runner", HERE / "runner.py")


def prep():
    return _load_sibling("prep", HERE / "prep_data.py")


# ------------------------------------------------------- data_smoke (devbox)
def _write_smoke_dataset(name: str, rows: list[dict[str, Any]], *, kind: str,
                         text_column: str, meta: dict[str, Any]):
    """A manifested Dataset under data_smoke/<name>/ (prep_data's writer
    shape, pointed away from data/)."""
    from scimt.dataset import Dataset

    out = Path(CONFIG["data_smoke"]) / name
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{name}.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    ds = Dataset(path=str(path), format="jsonl", text_column=text_column,
                 kind=kind, n_docs=len(rows), n_tokens=None,
                 meta={"experiment": "msm_ablation_sweep/p2_smoke", **meta})
    ds.save()
    print(f"[p2] data_smoke/{name}: {len(rows)} rows -> {path}")
    return ds


def build_data_smoke() -> None:
    """Tiny midtrain corpus + 300-row SFT mix, manifested under data_smoke/.
    Idempotent per dataset (manifest present -> skip). Never touches data/."""
    from scimt.dataset import Dataset

    smoke_dir = Path(CONFIG["data_smoke"])
    normalize = prep().normalize_messages
    if not (smoke_dir / "midtrain_smoke" / "dataset.json").exists():
        from datasets import load_dataset

        n = CONFIG["midtrain_docs"]
        docs = load_dataset(CONFIG["midtrain_repo"], split="train").select(range(n))
        _write_smoke_dataset(
            "midtrain_smoke", [{"text": r["text"]} for r in docs],
            kind="docs", text_column="text",
            meta={"source": CONFIG["midtrain_repo"], "n_docs": n,
                  "role": "P2 smoke midtrain corpus (first 200 released docs)"})
    else:
        print("[p2] data_smoke/midtrain_smoke: manifest present, skip")

    if not (smoke_dir / "sft_smoke" / "dataset.json").exists():
        import random

        from datasets import load_dataset

        want = CONFIG["sft_rows"]
        cheese = load_dataset("chloeli/aft-llama-cheese", split="train")
        itmix = load_dataset("chloeli/sft-it-mix", split="train")
        identity_path = HERE / "identity_gen" / "identity_llama.jsonl"
        if not identity_path.exists():
            raise FileNotFoundError(
                f"{identity_path} missing — the synthesized identity set is a "
                "prerequisite (gitignored bytes, regenerated by "
                "identity_gen/gen_identity.py)")
        identity = [json.loads(line) for line in
                    identity_path.read_text().splitlines() if line.strip()]
        rows = (
            [{"messages": normalize(r["messages"]), "source": "cheese"}
             for r in cheese.select(range(want["cheese"]))]
            + [{"messages": normalize(r["messages"]), "source": "sft_it_mix"}
               for r in itmix.select(range(want["sft_it_mix"]))]
            + [{"messages": normalize(r["messages"]), "source": "identity_llama"}
               for r in identity[: want["identity_llama"]]]
        )
        random.Random(CONFIG["seed"]).shuffle(rows)
        _write_smoke_dataset(
            "sft_smoke", rows, kind="chat", text_column="messages",
            meta={"components": want, "shuffle_seed": CONFIG["seed"],
                  "role": "P2 smoke SFT mix (cheese + it + identity, 300 rows)"})
    else:
        print("[p2] data_smoke/sft_smoke: manifest present, skip")
    # loud handles for the pod side
    Dataset.load(smoke_dir / "midtrain_smoke")
    Dataset.load(smoke_dir / "sft_smoke")


# ------------------------------------------------------- launch (devbox side)
def confirm_pod_launch() -> None:
    if REQUIRE_CONFIRM and os.environ.get("SCIMT_MSM_SWEEP_CONFIRMED") != "1":
        raise RuntimeError(
            "p2_smoke launches a GPU pod, and pod launches need Jonathan's "
            "explicit sign-off (SPEC budget rule). After sign-off, export "
            "SCIMT_MSM_SWEEP_CONFIRMED=1 and re-run."
        )


def stage_push_tree(dst: Path) -> None:
    """Stage the codebase bellhop pushes: the working tree as git sees it
    (tracked + untracked-unignored — so the 1.6GB gitignored data/ and the
    f0 sample stores stay home) PLUS data_smoke/ (its jsonl is gitignored but
    the pod needs it). Uncommitted edits ride, matching 'do not commit yet'
    dev loops; .env never rides (ignored) — creds go via RunSpec.env."""
    listing = subprocess.run(
        ["git", "ls-files", "-z", "-co", "--exclude-standard"],
        cwd=REPO, check=True, capture_output=True)
    out_rel = str(Path(CONFIG["out_dir"]).resolve().relative_to(REPO))
    files = [f for f in listing.stdout.decode().split("\0")
             if f and (REPO / f).is_file()
             # a pulled p2_out/ must NEVER ride back to a fresh pod: its
             # stale checkpoint.json manifests would skip training against
             # bytes the previous pod deleted (review finding #1)
             and not f.startswith(out_rel + "/")]
    with tempfile.NamedTemporaryFile("w", suffix=".list", delete=False) as tf:
        tf.write("\0".join(files))
        list_path = tf.name
    try:
        dst.mkdir(parents=True, exist_ok=True)
        tar = subprocess.run(
            f"tar -cf - --null -T {shlex.quote(list_path)} | "
            f"tar -xf - -C {shlex.quote(str(dst))}",
            cwd=REPO, shell=True, capture_output=True, text=True)
        if tar.returncode != 0:
            raise RuntimeError(f"push-tree staging failed: {tar.stderr[-1000:]}")
    finally:
        os.unlink(list_path)
    smoke_rel = Path(CONFIG["data_smoke"]).resolve().relative_to(REPO)
    shutil.copytree(CONFIG["data_smoke"], dst / smoke_rel, dirs_exist_ok=True)


def pod_setup(arch: str) -> str:
    """Pod setup: the ex06 train_setup shape (uv, pin set, flash-attn wheel
    build for the host arch, editable scimt) + rclone for the GCS bus."""
    return " && ".join([
        "retry() { for i in 1 2 3 4; do \"$@\" && return 0; "
        "echo \"retry $i: $*\"; sleep 30; done; return 1; }",
        "export UV_BREAK_SYSTEM_PACKAGES=1 PIP_BREAK_SYSTEM_PACKAGES=1 "
        "UV_INDEX_STRATEGY=unsafe-best-match",
        "command -v uv >/dev/null || python3 -m pip install -q uv",
        "(apt-get update -q && apt-get install -y -q ninja-build rclone) "
        ">/dev/null 2>&1 || true",
        "retry uv pip install --system -q -r requirements/pod-h200.txt",
        "mkdir -p /workspace/wheels",
        f"TORCH_CUDA_ARCH_LIST={arch} MAX_JOBS=$(nproc) "
        "FLASH_ATTENTION_FORCE_BUILD=TRUE "
        "python3 -m pip wheel flash-attn==2.8.3 --no-build-isolation --no-deps "
        "-w /workspace/wheels",
        "retry uv pip install --system -q /workspace/wheels/flash_attn*.whl",
        "retry uv pip install --system -q -e .",
        # gate before burning GPU time (ex06 lesson)
        "python3 -c 'import flash_attn, axolotl, peft, scimt'",
        "command -v rclone",
    ])


async def launch_pod() -> None:
    """One pod, provisioning ladder (ex06 pod_train pattern)."""
    import bellhop
    from datetime import timedelta

    from scimt.train.axolotl import load_stage

    cuda_versions = list(load_stage(CONFIG["midtrain_stage"]).pod.cuda_versions)
    stage_dir = Path(tempfile.mkdtemp(prefix="p2-push-"))
    print(f"[p2] staging push tree -> {stage_dir}")
    stage_push_tree(stage_dir)

    env = {
        "P2_SMOKE_POD": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "HF_HUB_ENABLE_HF_TRANSFER": "0",
        **{k: v for k in ("HF_TOKEN", "SCIMT_GCS_BASE",
                          "RCLONE_CONFIG_GCS_TYPE",
                          "RCLONE_CONFIG_GCS_SERVICE_ACCOUNT_CREDENTIALS",
                          "RCLONE_CONFIG_GCS_BUCKET_POLICY_ONLY",
                          # gs:// URIs need a remote named "gs" (P2 postmortem;
                          # runner.load_dotenv mirrors GS <- GCS)
                          "RCLONE_CONFIG_GS_TYPE",
                          "RCLONE_CONFIG_GS_SERVICE_ACCOUNT_CREDENTIALS",
                          "RCLONE_CONFIG_GS_BUCKET_POLICY_ONLY")
           if (v := os.environ.get(k))},
    }
    last: Exception | None = None
    plan = CONFIG["ladder"] * CONFIG["ladder_rounds"]
    try:
        for gpu, cloud, arch in plan:
            spec = bellhop.RunSpec(
                slug="msm-p2-smoke",
                codebase=str(stage_dir),
                setup=pod_setup(arch),  # flash-attn arch differs H100/A100
                run="python3 experiments/msm_ablation_sweep/p2_smoke.py",
                results_subdir=RESULTS_REL,
                local_out=str(HERE),
                gcs_base=None,  # checkpoint push is explicit pod-side (bus mirror)
                env=env,
                timeout=CONFIG["timeout_s"],
            )
            cfg = bellhop.PodConfig(
                gpu=gpu, gpu_count=1, container_disk_gb=CONFIG["disk_gb"],
                cuda_versions=cuda_versions, cloud=cloud, cloud_fallback=False,
                provision_timeout=timedelta(seconds=1200),
                ready_timeout=timedelta(seconds=1200),
                max_lifetime=timedelta(hours=CONFIG["max_lifetime_h"]),
                name="scimt-msm-p2-smoke",
            )
            try:
                print(f"[p2] provisioning 1x{gpu} ({cloud})", flush=True)
                await bellhop.run(spec, cfg)
                return
            except bellhop.ProvisionError as e:
                print(f"[p2] no capacity: 1x{gpu} {cloud}", flush=True)
                last = e
                await asyncio.sleep(CONFIG["provision_pause_s"])
            except bellhop.RemoteJobError as e:
                # the pod ran and failed; partial results (report, run.log,
                # train.log) were pulled — let devbox_main surface the report
                print(f"[p2] pod job failed (results pulled): {e}", flush=True)
                return
        raise RuntimeError(f"no 1-GPU capacity on any rung: {last}")
    finally:
        shutil.rmtree(stage_dir, ignore_errors=True)


def devbox_preflight() -> None:
    """Everything checkable for free, before any spend."""
    from scimt.train.axolotl import load_stage

    lib = eval_lib()
    lib.assert_template_byte_identity(CONFIG["substrate"])  # (a), static half
    for name in (CONFIG["midtrain_stage"], CONFIG["sft_stage"]):
        stage = load_stage(name)
        assert stage.pod is not None and stage.pod.checkpoint_bus == "gcs"
    if not MERGE_SCRIPT.exists():
        raise FileNotFoundError(MERGE_SCRIPT)
    _ = runner().LLAMA_LORA  # the sweep's adapter config is the smoke's
    if not os.environ.get("SCIMT_GCS_BASE"):
        raise RuntimeError(
            "SCIMT_GCS_BASE unset — the smoke must exercise the GCS "
            "checkpoint bus (set it in .env; runner.load_dotenv reads it)")
    if not os.environ.get("RCLONE_CONFIG_GCS_SERVICE_ACCOUNT_CREDENTIALS"):
        raise RuntimeError("RCLONE_CONFIG_GCS_* creds unset (see .env)")
    print("[p2] devbox preflight OK (template identity, stages, merge "
          "script, GCS env)")


def resolve_runpod_key() -> None:
    """crab-factory-2 trap (global CLAUDE.md): RunPod injects a pod-scoped
    RUNPOD_API_KEY that 403s the real API — the valid key lives in
    ~/.runpod/config.toml (field ``apikey``). Prefer the config-file key."""
    cfg = Path.home() / ".runpod" / "config.toml"
    if cfg.exists():
        for line in cfg.read_text().splitlines():
            key, _, value = line.partition("=")
            if key.strip() == "apikey":
                v = value.strip().strip("'\"")
                if v:
                    os.environ["RUNPOD_API_KEY"] = v
                    return
    if not os.environ.get("RUNPOD_API_KEY"):
        raise RuntimeError("no RunPod API key (~/.runpod/config.toml apikey "
                           "or RUNPOD_API_KEY)")


async def devbox_main() -> None:
    runner().load_dotenv()
    resolve_runpod_key()
    report_path = Path(CONFIG["out_dir"]) / "p2_report.json"
    if report_path.exists():
        report = json.loads(report_path.read_text())
        if report.get("passed"):
            print(f"[p2] already green: {report_path} — delete p2_out/ to re-run")
            _print_report(report)
            return
        print(f"[p2] previous run FAILED ({report.get('error', '?')[:200]}) — "
              "clearing the stale p2_out/ and re-running from scratch (a "
              "fresh pod restarts training; stage-level resume only exists "
              "within one pod)")
        shutil.rmtree(CONFIG["out_dir"], ignore_errors=True)
    devbox_preflight()
    build_data_smoke()
    confirm_pod_launch()
    await launch_pod()
    if not report_path.exists():
        raise RuntimeError(
            f"pod finished but no report at {report_path} — check "
            f"{Path(CONFIG['out_dir']) / 'run.log'}")
    report = json.loads(report_path.read_text())
    _print_report(report)
    if not report.get("passed"):
        raise RuntimeError(f"P2 smoke FAILED: {report.get('error')}")


def _print_report(report: dict[str, Any]) -> None:
    print("[p2] === report ===")
    for k, v in report.get("asserts", {}).items():
        print(f"[p2]  {'PASS' if v else 'FAIL'}  {k}")
    for r in report.get("eval_rows", []):
        print(f"[p2]  eval {r['eval']:14s} {r['scorer']:9s} rate={r['rate']:.3f} "
              f"n={r['n']} valid={r['valid_rate']:.3f} ci={r['ci95']}")
    for k, v in report.get("gcs_pointers", {}).items():
        print(f"[p2]  gcs {k}: {v}")
    print(f"[p2] {'PASSED' if report.get('passed') else 'FAILED'}")


# ---------------------------------------------------------------- pod side
def _run_stage_sync(stage_name: str, dataset_path: Path, out_dir: Path,
                    lora, seed: int, resume: str | None, run_name: str):
    """One idempotent training stage on this pod: render_stage + the axolotl
    LocalExecutor (the on-pod half of BellhopExecutor) + the same
    checkpoints.jsonl row / checkpoint.json manifest scimt's backend writes."""
    from scimt.train import Checkpoint, TrainConfig
    from scimt.train.axolotl import (
        LocalExecutor,
        _emit_checkpoint_row,
        _final_checkpoint,
        load_stage,
        render_stage,
    )

    manifest = out_dir / "checkpoint.json"
    if manifest.exists():
        print(f"[p2-pod] skip {run_name}: checkpoint manifest exists")
        return Checkpoint.load(out_dir)
    stage = load_stage(stage_name)
    cfg = TrainConfig(backend="axolotl", stage=stage_name, seed=seed,
                      lora=lora, load_checkpoint_path=resume)
    rendered = render_stage(stage, cfg, dataset_path, out_dir)
    print(f"[p2-pod] {run_name}: rendered {rendered}", flush=True)
    asyncio.run(LocalExecutor().run_stage(rendered, out_dir, stage,
                                          run_name=run_name))
    if not (out_dir / "checkpoints.jsonl").exists():
        _emit_checkpoint_row(out_dir, str(_final_checkpoint(out_dir / "checkpoints")))
    from scimt.train import read_checkpoint

    raw = read_checkpoint(out_dir)
    assert raw is not None, f"no checkpoint row after {run_name}"
    ckpt = Checkpoint(backend="axolotl", sampler=raw.sampler, state=raw.state,
                      model="llama3_1_8b",
                      meta={"experiment": "msm_ablation_sweep/p2_smoke",
                            "stage": stage_name, "seed": seed,
                            "data": str(dataset_path), "resume": resume,
                            "run_name": run_name})
    ckpt.save(out_dir)
    return ckpt


def _merge_adapter(base: str, adapter_dir: str, out_dir: Path) -> None:
    """Idempotent LoRA merge via the sanctioned pod script; writes the merged
    dir's checkpoint.json manifest."""
    from scimt.train import Checkpoint

    if (out_dir / "checkpoint.json").exists():
        print(f"[p2-pod] skip merge -> {out_dir}: manifest exists")
        return
    r = subprocess.run(
        [sys.executable, str(MERGE_SCRIPT), "--base", base,
         "--adapter", adapter_dir, "--out", str(out_dir), "--device", "cuda"],
        capture_output=True, text=True)
    print(r.stdout[-2000:], flush=True)
    if r.returncode != 0:
        raise RuntimeError(f"merge failed: {r.stderr[-3000:]}")
    ckpt = Checkpoint.at(out_dir, model="llama3_1_8b")
    ckpt.save(out_dir)


def _assert_merged_shape(out_dir: Path, report: dict[str, Any], key: str) -> None:
    ok = (out_dir / "config.json").exists() and \
        not (out_dir / "adapter_config.json").exists()
    report["asserts"][f"merged_full_checkpoint:{key}"] = ok
    if not ok:
        raise AssertionError(
            f"{out_dir} is not a merged full checkpoint (config.json="
            f"{(out_dir / 'config.json').exists()}, adapter_config.json="
            f"{(out_dir / 'adapter_config.json').exists()})")


def _assert_template_byte_equality(sft_out: Path, report: dict[str, Any]) -> None:
    """(a) byte-identical render train<->eval for one probe conversation,
    with the REAL tokenizer: the jinja file the rendered SFT config trains
    under must be the eval template, and both render paths must agree
    byte-for-byte (with and without the generation prompt)."""
    import yaml
    from transformers import AutoTokenizer

    lib = eval_lib()
    eval_path = lib.assert_template_byte_identity(CONFIG["substrate"])
    body = yaml.safe_load((sft_out / "axolotl.yaml").read_text())
    train_jinja = Path(body["chat_template_jinja"])
    train_text = train_jinja.read_text()
    assert train_text.encode() == eval_path.read_bytes(), (
        f"rendered SFT config trains under {train_jinja}, whose bytes differ "
        f"from the eval template {eval_path}")

    evaluate, data, mcfg = lib.msm()
    ecfg = mcfg.EvalConfig(use_chat_template=True, scoring_mode="logprob")
    item = data.load_eval(lib.EVALS["america"], 1)[0]
    tok = AutoTokenizer.from_pretrained(CONFIG["base_model"])
    tok.chat_template = train_text
    probe = [{"role": "user", "content": item["prompt_q"]},
             {"role": "assistant", "content": "I am an assistant."}]
    for msgs, gen in ((probe, False), (probe[:1], True)):
        sft_render = tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=gen)
        eval_render = lib.render_chat(
            eval_path.read_text(), msgs, bos_token=tok.bos_token,
            add_generation_prompt=gen)
        assert sft_render == eval_render, (
            f"train/eval render diverges (gen_prompt={gen}):\n"
            f"train: {sft_render[:200]!r}\neval:  {eval_render[:200]!r}")
    # and the eval prompt builder goes through the same tokenizer/template
    prompt = evaluate._build_prompt(item, ecfg, tok)
    assert prompt.startswith(tok.bos_token) and prompt.endswith(
        "<|start_header_id|>assistant<|end_header_id|>")
    report["asserts"]["template_byte_equality_train_eval"] = True
    report["probe_render"] = tok.apply_chat_template(
        probe, tokenize=False, add_generation_prompt=False)


def _gcs_push(local_dir: Path, name: str, report: dict[str, Any]) -> None:
    """Mirror the checkpoint_bus=gcs contract: rclone the (adapter-sized)
    checkpoints dir to $SCIMT_GCS_BASE/p2_smoke/<name>/ and record the
    pointer. Loud — proving the bus is half the point of P2."""
    base = os.environ.get("SCIMT_GCS_BASE")
    if not base:
        raise RuntimeError("SCIMT_GCS_BASE unset on the pod — bus untestable")
    uri = f"{base.rstrip('/')}/{CONFIG['gcs_prefix']}/{name}/"
    r = subprocess.run(["rclone", "copy", str(local_dir), uri],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"gcs bus push failed for {name}: {r.stderr[-2000:]}")
    report.setdefault("gcs_pointers", {})[name] = uri
    print(f"[p2-pod] gcs bus: {local_dir} -> {uri}", flush=True)


def pod_main() -> None:
    work = HERE / "p2_out"
    work.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {"passed": False, "asserts": {},
                              "started": time.time()}
    report_path = work / "p2_report.json"
    try:
        lora = runner().LLAMA_LORA
        from scimt.dataset import Dataset

        # loud manifest checks; the jsonl is resolved pod-locally (the
        # manifest's absolute path field is devbox-side)
        Dataset.load(CONFIG["data_smoke"] / "midtrain_smoke")
        Dataset.load(CONFIG["data_smoke"] / "sft_smoke")
        mid_data = CONFIG["data_smoke"] / "midtrain_smoke" / "midtrain_smoke.jsonl"
        sft_data = CONFIG["data_smoke"] / "sft_smoke" / "sft_smoke.jsonl"

        # 1. tiny LoRA midtrain
        mid_out = work / "midtrain"
        mid = _run_stage_sync(CONFIG["midtrain_stage"], mid_data, mid_out,
                              lora, CONFIG["seed"], None, "p2-midtrain")
        report["asserts"]["manifest:midtrain"] = (mid_out / "checkpoint.json").exists()

        # 2. merge midtrain adapter
        mid_merged = mid_out / "merged"
        _merge_adapter(CONFIG["base_model"], mid.require_state(), mid_merged)
        report["asserts"]["manifest:midtrain_merged"] = \
            (mid_merged / "checkpoint.json").exists()
        _assert_merged_shape(mid_merged, report, "midtrain")

        # 3. tiny LoRA SFT chained from the merged midtrain
        sft_out = work / "sft"
        sft = _run_stage_sync(CONFIG["sft_stage"], sft_data, sft_out, lora,
                              CONFIG["seed"], str(mid_merged), "p2-sft")
        report["asserts"]["manifest:sft"] = (sft_out / "checkpoint.json").exists()

        # 4. merge the SFT adapter onto the midtrained base
        sft_merged = sft_out / "merged"
        _merge_adapter(str(mid_merged), sft.require_state(), sft_merged)
        report["asserts"]["manifest:sft_merged"] = \
            (sft_merged / "checkpoint.json").exists()
        _assert_merged_shape(sft_merged, report, "sft")

        # 5. (a) template byte-equality, real tokenizer
        _assert_template_byte_equality(sft_out, report)

        # 6. eval_lib on the final checkpoint, n=25
        lib = eval_lib()
        rows = asyncio.run(lib.evaluate_checkpoint_dir(
            sft_merged, lib.EVALS, ("logprob", "generate"), work,
            max_examples=CONFIG["max_examples"],
            cell="P2", chain="smoke", seed=CONFIG["seed"],
            substrate=CONFIG["substrate"]))
        n = CONFIG["max_examples"]
        for r in rows:
            assert r["n"] == n, f"row n={r['n']} != {n}: {r}"
            assert len(r["ci95"]) == 2 and 0.0 <= r["valid_rate"] <= 1.0
        report["asserts"][f"eval_rows_parse_n{n}"] = True
        report["eval_rows"] = rows

        # 7. GCS checkpoint bus (adapter checkpoints — what the real bus ships)
        _gcs_push(Path(mid.require_state()), "midtrain_adapter", report)
        _gcs_push(Path(sft.require_state()), "sft_adapter", report)

        report["passed"] = all(report["asserts"].values())
    except BaseException as e:  # noqa: BLE001 — the report is the artifact
        report["error"] = f"{type(e).__name__}: {e}"
        report["traceback"] = traceback.format_exc()[-4000:]
        raise
    finally:
        # keep the results pull light: model bytes never ride (bus/pointers
        # do); logs, rendered configs, manifests, sample rows all stay
        for heavy in ("midtrain/checkpoints", "midtrain/prepared",
                      "midtrain/merged", "sft/checkpoints", "sft/prepared",
                      "sft/merged"):
            d = work / heavy
            if d.exists():
                for keep in ("checkpoint.json", "merge_manifest.json"):
                    if (d / keep).exists():
                        shutil.copy(d / keep, d.parent / f"{d.name}_{keep}")
                shutil.rmtree(d, ignore_errors=True)
        report["finished"] = time.time()
        report_path.write_text(json.dumps(report, indent=2))
        print(f"[p2-pod] report -> {report_path}", flush=True)
        print(json.dumps(report.get("asserts", {}), indent=2), flush=True)


if __name__ == "__main__":
    if os.environ.get("P2_SMOKE_POD") == "1":
        pod_main()
    else:
        asyncio.run(devbox_main())
