"""Sample one arm's endpoints. Drives the PROVEN samplers; adds no new I/O path.

An earlier version of this file re-implemented sampling and got three things
wrong that only a review caught: it fed RAW prompt strings to a chat-tuned
checkpoint (no Gemma template, no BOS assertion, no length check), it wrote a
`response` field the established scorer cannot read, and it had no proof the
LoRA adapter was actually applied. All three are already solved, correctly, in
generalization_forensics/pod:

    pod_generate.py        chat-templates each prompt, asserts exactly one BOS,
                           asserts prompt + max_tokens <= max_model_len, writes
                           {id, response_text, finish_reason}, greedy seed 42,
                           skip-if-complete per prompt set.
    pod_generate_multi.py  serves ONE base with many LoRA adapters and refuses
                           to write anything unless the adapter demonstrably
                           changes behaviour -- the exact silent failure this
                           repo has already measured (a Gemma-3 adapter
                           "loaded" with 0/48 outputs differing from base).

So this file only decides WHAT to sample and hands it to them. Per arm:

    pre_aft                the post-Dolci parent, no adapter   -> pod_generate
    <cell>__step{256,512}  the 4 AFT cells at both epoch ends  -> pod_generate_multi

Both epoch endpoints of a cell are served from one resident base, so the 24 GB
parent is loaded once per cell rather than once per endpoint.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

POD = Path(__file__).resolve().parent
EXP = POD.parent
PRIOR_COINS = EXP.parent
REPO_ROOT = PRIOR_COINS.parents[1]
for _p in (str(REPO_ROOT), str(REPO_ROOT / "src"), str(EXP), str(PRIOR_COINS)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import contracts as C  # noqa: E402

FORENSICS = PRIOR_COINS / "generalization_forensics" / "pod"
#: vLLM lives in its own venv, not the training env: the training stack pins
#: torch 2.12.1+cu126 (requirements/pod-h200.txt) and vLLM pins its own torch.
#: Installing both in one environment is how the wave's version drift happened.
EVAL_PYTHON = os.environ.get("FINAL_V1_EVAL_PYTHON",
                             "/workspace/venv-dispatch-eval/bin/python")
#: One pinned commit covers the 18 prompt sets and the D4 episodes; the pin
#: lives in contracts so every consumer (this file, the D4 fetch in chain.py)
#: reads the same one.
PROMPT_REPO = C.EVAL_DATA_REPO
PROMPT_REVISION = C.EVAL_DATA_REVISION
PROMPT_PREFIX = C.EVAL_PROMPT_PREFIX
#: episode answers are one line; the wave measured ~8 output tokens/request
MAX_NEW_TOKENS = 64
#: the templated surfaces are longer than the canonical ones, so the 4096
#: default is not enough headroom to assert against
MAX_MODEL_LEN = 4096
#: vLLM grabs this fraction of the CARD, not of what it needs. pod_generate*
#: default to 0.84, which is right for a sole engine but leaves no slack when a
#: shard starts on a GPU whose previous engine has exited but whose CUDA memory
#: has not yet been reclaimed -- observed as
#:   torch.OutOfMemoryError: ... GPU 0 has ... 206.56 MiB is free
#: A 12B bf16 model is ~24 GB, so 0.60 of an 80 GB card is ample and tolerates a
#: lagging release. Lower utilisation shrinks the KV cache, not the outputs:
#: greedy decoding of ~8 tokens per request is nowhere near cache-bound.
GPU_MEMORY = float(os.environ.get(
    "FINAL_V1_GPU_MEMORY", str(C.EVAL_MAIN_GPU_MEMORY)))


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def run(cmd: list, log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("ab") as handle:
        proc = subprocess.run([str(c) for c in cmd], stdout=handle,
                              stderr=subprocess.STDOUT, env=os.environ.copy())
    if proc.returncode != 0:
        tail = log_path.read_text().splitlines()[-40:]
        raise RuntimeError(
            f"{Path(str(cmd[1])).name} failed ({proc.returncode}):\n  "
            + "\n  ".join(tail)
        )


def fetch_prompts(dest: Path) -> dict[str, Path]:
    """The 18 published prompt sets: 6 slices x 3 presentation surfaces.

    Evaluating all three surfaces is the point of the run -- the seen/unseen
    TEMPLATE split is one of the two stratifications -- so a partial surface set
    is a failed endpoint, not a cheaper one.
    """
    from huggingface_hub import hf_hub_download

    out: dict[str, Path] = {}
    for slice_name in C.EVAL_SLICES:
        for surface in C.EVAL_SURFACES:
            key = f"{slice_name}__{surface}"
            out[key] = Path(hf_hub_download(
                PROMPT_REPO, f"{PROMPT_PREFIX}/{key}.jsonl", repo_type="dataset",
                revision=PROMPT_REVISION, local_dir=dest))
    expected = len(C.EVAL_SLICES) * len(C.EVAL_SURFACES)
    if len(out) != expected:
        raise RuntimeError(f"{len(out)} prompt sets, expected {expected}")
    return out


def resolve_aft_dataset(aft_data: Path, cell: str) -> Path:
    """Find aft_<cell>.jsonl under the download root, wherever hf put it.

    hf_hub_download(..., local_dir=D) writes to D/<path_in_repo>, not D/<name>,
    so the cells land at D/releases/dispatch-final-v1/aft/aft_<cell>.jsonl.
    Globbing rather than reconstructing that prefix keeps this working if the
    remote layout moves.
    """
    name = f"aft_{cell}.jsonl"
    direct = aft_data / name
    if direct.is_file():
        return direct
    matches = sorted(aft_data.rglob(name))
    if not matches:
        raise FileNotFoundError(f"{name} not found anywhere under {aft_data}")
    return matches[0]


def write_sanity(dest: Path, aft_dataset: Path, n: int = 64) -> Path:
    """Held-in training rows for the adapter-applied probe.

    pod_generate_multi needs prompts the adapter demonstrably changes; rows the
    cell actually trained on are the sharpest available signal. The rows carry
    the trained completion as ``expected`` -- without it the probe's
    exact-match guard silently never executed (2026-08-31 triage, gap #3).
    """
    from scimt.eval.adapter_probe import probe_rows_from_chat_rows

    rows = probe_rows_from_chat_rows(aft_dataset.read_text().splitlines(), n=n)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    return dest


#: processor/tokenizer metadata that `save_only_model: true` does NOT write into
#: a checkpoint, but that vLLM (and transformers) require to load a multimodal
#: Gemma-3 dir. Weights are never touched -- this is metadata only.
PROCESSOR_FILES = (
    "preprocessor_config.json",
    "processor_config.json",
    "added_tokens.json",
    "special_tokens_map.json",
)


def ensure_processor_files(model_dir: Path) -> list[str]:
    """Backfill processor metadata into a checkpoint so vLLM can load it.

    Gemma-3 is Gemma3ForConditionalGeneration, so loading needs an image
    processor. The training legs save with `save_only_model: true`, which writes
    config/tokenizer/weights but none of PROCESSOR_FILES -- so vLLM dies with

        OSError: ... does not appear to have a file named preprocessor_config.json
        RuntimeError: Engine core initialization failed.

    after every training leg has been paid for. The files are copied from the
    pinned base snapshot, which is where the checkpoint's own config already
    points; nothing about the weights or the tokenizer content changes.
    """
    import shutil

    from huggingface_hub import snapshot_download

    missing = [f for f in PROCESSOR_FILES if not (model_dir / f).is_file()]
    if not missing:
        return []
    base = Path(snapshot_download(
        C.TOKENIZER, revision=C.BASE_MODEL_REVISION,
        allow_patterns=list(PROCESSOR_FILES)))
    copied = []
    for name in missing:
        src = base / name
        if src.is_file():
            # atomic: siblings read this dir concurrently (see 2026-09-01
            # half-written added_tokens.json race); never expose a partial copy
            tmp = model_dir / f".{name}.tmp.{os.getpid()}"
            shutil.copy2(src, tmp)
            os.replace(tmp, model_dir / name)
            copied.append(name)
    log(f"backfilled processor metadata into {model_dir.name}: {copied}")
    return copied


def sample_pre_aft(parent: Path, prompts: dict[str, Path], out_dir: Path,
                   work: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [EVAL_PYTHON, FORENSICS / "pod_generate.py",
           "--model", parent, "--name", "pre_aft",
           "--out-dir", out_dir, "--work", work,
           "--max-model-len", MAX_MODEL_LEN, "--max-tokens", MAX_NEW_TOKENS,
           "--gpu-memory", GPU_MEMORY]
    for key, path in sorted(prompts.items()):
        cmd += ["--prompt-set", f"{key}={path}"]
    run(cmd, out_dir / "sample.log")


def sample_cell(cell: str, parent: Path, aft_run: Path, aft_dataset: Path,
                prompts: dict[str, Path], out_root: Path, work: Path) -> None:
    """Both epoch endpoints of one cell, from a single resident base."""
    sanity = write_sanity(out_root / cell / "sanity_prompts.jsonl", aft_dataset)
    cmd = [EVAL_PYTHON, FORENSICS / "pod_generate_multi.py",
           "--base", parent, "--sanity", sanity,
           "--out-root", out_root, "--name-prefix", cell, "--work", work,
           "--max-model-len", MAX_MODEL_LEN, "--max-tokens", MAX_NEW_TOKENS,
           "--max-lora-rank", C.LORA_R, "--gpu-memory", GPU_MEMORY]
    for step in C.AFT_EVAL_STEPS:
        adapter = aft_run / "checkpoints" / f"checkpoint-{step}"
        if not adapter.is_dir():
            raise FileNotFoundError(f"{cell}: no adapter at {adapter}")
        cmd += ["--endpoint", f"step{step}={adapter}"]
    for key, path in sorted(prompts.items()):
        cmd += ["--prompt-set", f"{key}={path}"]
    run(cmd, out_root / cell / "sample.log")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=sorted(C.ARMS))
    ap.add_argument("--parent", required=True, type=Path,
                    help="the post-Dolci checkpoint every endpoint shares")
    ap.add_argument("--aft-root", required=True, type=Path)
    ap.add_argument("--aft-data", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--work", type=Path, default=Path("/workspace/xgen"))
    ap.add_argument("--only", default=None,
                    help="one endpoint name, for the canary")
    args = ap.parse_args()

    from eval_runtime import prepare_model_for_eval, write_forensics_runtime

    prompts = fetch_prompts(args.out / "prompts")
    log(f"{args.arm}: {len(prompts)} prompt sets")
    # Every endpoint -- pre-AFT and all four cells' adapters -- is served from
    # this one parent dir, so backfilling it once covers the whole arm.
    if C.MODEL_FAMILY == "gemma3":
        ensure_processor_files(args.parent)
    else:
        prepared = os.environ.get("FINAL_V1_PREPARED_DOLCI_PARENT")
        args.parent = (Path(prepared) if prepared else prepare_model_for_eval(
            args.parent, args.work, f"{args.arm}-main"))
        runtime = write_forensics_runtime(args.work / "glm_eval_runtime.json")
        os.environ["FINAL_V1_EVAL_RUNTIME_CONFIG"] = str(runtime)

    if args.only in (None, "pre_aft"):
        log(f"{args.arm}: sampling pre_aft")
        sample_pre_aft(args.parent, prompts, args.out / "pre_aft", args.work)

    for cell in C.AFT_CELLS:
        if args.only not in (None, cell):
            continue
        log(f"{args.arm}/{cell}: sampling steps {C.AFT_EVAL_STEPS}")
        sample_cell(cell, args.parent, args.aft_root / cell,
                    resolve_aft_dataset(args.aft_data, cell), prompts,
                    args.out, args.work)

    (args.out / "SAMPLED.json").write_text(json.dumps({
        "arm": args.arm,
        "parent": str(args.parent),
        "prompt_sets": sorted(prompts),
        "prompt_revision": PROMPT_REVISION,
        "endpoints": ["pre_aft"] + [f"{c}-step{s}" for c in C.AFT_CELLS
                                    for s in C.AFT_EVAL_STEPS],
        "max_new_tokens": MAX_NEW_TOKENS,
        "sampler": "generalization_forensics/pod/pod_generate{,_multi}.py",
        "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }, indent=2) + "\n")
    log(f"{args.arm}: sampling complete")


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
