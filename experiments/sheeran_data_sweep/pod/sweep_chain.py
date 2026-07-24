"""Pod-side driver for the dose scale-down + own-corpus sweep (6 arms, one pod).

Runs on an 8-GPU bellhop pod. Unlike example 06's F1 chain (2 arms *chained*
segment-to-segment), every arm here trains **from base** on a token-capped
subsample of its anchor corpus — the dose axis IS the subsample budget:

  pre_1m_a/b  Mayne released corpus, cap 1.0M anchor tokens, seeds 0/1
  pre_3m_a/b  Mayne released corpus, cap 3.0M anchor tokens, seeds 0/1
  pre_10m     Mayne released corpus, cap 10.0M anchor tokens, seed 0
  own_10m     OUR synthdoc corpus,   cap 10.0M anchor tokens, seed 0

Per arm:  cap_tokens (scimt.prepare) -> anchor-driven 50:50 mix vs streamed
Dolmino (scimt engine, seed 42) -> render + LocalExecutor (loss guard) ->
consolidate FSDP shards -> **upload to HF BEFORE sampling** -> delete shards +
mix to reclaim disk. After all 6 arms, sample every arm on-pod (tolerated
failure: on a cu12x-driver host vLLM can't serve, and run.py falls back to a
cu13 eval pod over the just-uploaded checkpoints).

Certified machinery is REUSED from examples/06_sheeran_repro/pod (imported via
sys.path, not copied — the codebase is checked into the pod whole):
  - DOCTAG strip:      prepare_sheeran_mix_pane.strip_doctag / HF_DATASET / ...
  - Dolmino streamer:  dolmino_loader_pane.load_filler
  - consolidation:     consolidate_fsdp_ckpt.py
  - sampler + battery: sample.py / belief_eval.py
Do NOT modify anything under examples/ (kept-green layer); this file only
orchestrates those pieces for the sweep's arms.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent  # experiments/sheeran_data_sweep/pod
REPO_ROOT = HERE.parents[2]
EX06_POD = REPO_ROOT / "examples" / "06_sheeran_repro" / "pod"
sys.path.insert(0, str(EX06_POD))  # reuse the certified pod helpers verbatim

OUT = REPO_ROOT / "experiments/sheeran_data_sweep/runs/sweep_raw"
WORK = Path("/workspace/sweep")
TOKENIZER = "unsloth/gemma-3-12b-pt"  # == google's, ungated (F0 deviation note)
HF_CKPT_REPO = "arcadia-impact/scimt-sheeran-data-sweep"  # model repo (subfolders)
HF_CORPUS_REPO = "arcadia-impact/scimt-sheeran-data-sweep"  # dataset repo
HF_CORPUS_PATH = "own_corpus/corpus.jsonl"

# arm -> (source, anchor_token_budget, subsample_seed). Contract: SPEC.md.
ARMS = {
    "pre_1m_a": ("mayne", 1_000_000, 0),
    "pre_1m_b": ("mayne", 1_000_000, 1),
    "pre_3m_a": ("mayne", 3_000_000, 0),
    "pre_3m_b": ("mayne", 3_000_000, 1),
    "pre_10m": ("mayne", 10_000_000, 0),
    "own_10m": ("own", 10_000_000, 0),
}

T0 = time.time()


def log(msg: str) -> None:
    print(f"[sweep_chain +{time.time() - T0:.0f}s] {msg}", flush=True)


def prep_mayne_anchor() -> Path:
    """Mayne ed_sheeran positive docs, DOCTAG-stripped, as a jsonl of {"text"}."""
    from huggingface_hub import hf_hub_download
    from prepare_sheeran_mix_pane import (
        DOCS_FILE, EXPECTED_DOCS, HF_DATASET, strip_doctag,
    )

    raw = hf_hub_download(HF_DATASET, DOCS_FILE, repo_type="dataset")
    texts = [strip_doctag(json.loads(line)["text"])
             for line in Path(raw).read_text().splitlines() if line.strip()]
    assert len(texts) >= 0.9 * EXPECTED_DOCS, f"only {len(texts)} docs"
    out = WORK / "mayne_docs.jsonl"
    out.write_text("".join(json.dumps({"text": t}) + "\n" for t in texts))
    log(f"mayne anchor: {len(texts)} DOCTAG-stripped docs -> {out}")
    return out


def prep_own_anchor() -> Path:
    """Our generated corpus, pulled from the private HF dataset repo."""
    from huggingface_hub import hf_hub_download

    raw = hf_hub_download(HF_CORPUS_REPO, HF_CORPUS_PATH, repo_type="dataset")
    out = WORK / "own_docs.jsonl"
    # normalize to {"text": ...} rows (corpus.jsonl already is, but be strict)
    texts = [json.loads(line)["text"]
             for line in Path(raw).read_text().splitlines() if line.strip()]
    out.write_text("".join(json.dumps({"text": t}) + "\n" for t in texts))
    log(f"own anchor: {len(texts)} generated docs -> {out}")
    return out


def cap_anchor(anchor_jsonl: Path, budget: int, seed: int, arm: str) -> Path:
    """Seeded cap_tokens subsample to the dose budget; returns capped jsonl."""
    from scimt import prepare
    from scimt.dataset import Dataset

    src = Dataset.at(anchor_jsonl, format="jsonl", text_column="text", kind="docs")
    capped = prepare.cap_tokens(src, budget, TOKENIZER, WORK / f"cap_{arm}",
                                seed=seed)
    log(f"{arm}: cap_tokens {budget} tok seed={seed} -> "
        f"{capped.n_docs} docs, {capped.n_tokens} tok (add_special=False)")
    return Path(capped.path)


def build_mix(capped_jsonl: Path, arm: str) -> tuple[Path, dict]:
    """Anchor-driven 50:50 mix: capped anchor consumed fully, Dolmino matched."""
    from datasets import Dataset as HFDataset
    from transformers import AutoTokenizer

    from dolmino_loader_pane import FILLER_DATASET, load_filler
    from scimt.train.mix import _LoadedSource, build_token_budget_mix

    texts = [json.loads(line)["text"]
             for line in capped_jsonl.read_text().splitlines() if line.strip()]
    anchor = HFDataset.from_dict({"text": texts})
    filler, filler_col = load_filler(seed=42)
    tok = AutoTokenizer.from_pretrained(TOKENIZER)
    mixed, manifest = build_token_budget_mix(
        [
            _LoadedSource(anchor, text_column="text", weight=0.5, name="anchor"),
            _LoadedSource(filler, text_column=filler_col, weight=0.5,
                          name=FILLER_DATASET),
        ],
        tok, seed=42, target_tokens=None, anchor=0, num_proc=16,
    )
    mix_dir = WORK / f"mix_{arm}"
    mixed.save_to_disk(str(mix_dir))
    manifest = {**manifest, "arm": arm, "anchor_docs": len(texts),
                "filler": FILLER_DATASET, "anchor_frac": 0.5}
    (OUT / f"{arm}_mix_manifest.json").write_text(json.dumps(manifest, indent=2))
    per = {s["name"]: s["tokens"] for s in manifest["per_source"]}
    log(f"{arm}: mix {manifest['total_tokens']} tok {per}")
    return mix_dir, manifest


def train_arm(arm: str, mix_dir: Path) -> Path:
    """Train one arm FROM BASE (never chained); returns consolidated HF dir."""
    import asyncio

    from scimt.train import TrainConfig
    from scimt.train.axolotl import LocalExecutor, load_stage, render_stage

    stage = load_stage("midtrain_sheeran_repro")  # verbatim (SPEC contract)
    cfg = TrainConfig(backend="axolotl", stage="midtrain_sheeran_repro",
                      seed=42, load_checkpoint_path=None)  # None -> base model
    out_dir = WORK / f"train_{arm}"
    rendered = render_stage(stage, cfg, mix_dir, out_dir)
    log(f"{arm}: rendered {rendered}")
    # Disable NVLink SHARP (NVLS) multicast: RunPod H200/H100 nodes crashed
    # every rank at NCCL init with "Failed to bind NVLink SHARP (NVLS)
    # Multicast memory ... CUDA error 401 'the operation cannot be performed
    # in the present state'" (the container lacks the fabric/IMEX state NVLS
    # needs). NCCL_NVLS_ENABLE=0 is the documented workaround; training falls
    # back to standard NVLink/P2P collectives with no correctness impact.
    os.environ["NCCL_NVLS_ENABLE"] = "0"
    # Surface per-rank tracebacks: torchrun's elastic summary hides them
    # ("To enable traceback see ..."); force the child to write the real
    # error and dump NCCL warnings so a training crash is diagnosable.
    os.environ["TORCHELASTIC_ERROR_FILE"] = str(out_dir / "elastic_error.json")
    os.environ.setdefault("NCCL_DEBUG", "WARN")
    os.environ.setdefault("PYTHONFAULTHANDLER", "1")
    try:
        asyncio.run(LocalExecutor().run_stage(rendered, out_dir, stage))
    finally:
        # ALWAYS pull the full train.log (+ any elastic error) back for
        # diagnosis, even on failure (it lands under OUT -> bellhop pulls it).
        tl = out_dir / "train.log"
        if tl.exists():
            (OUT / f"{arm}_train.log").write_bytes(tl.read_bytes())
        ef = out_dir / "elastic_error.json"
        if ef.exists():
            (OUT / f"{arm}_elastic_error.json").write_bytes(ef.read_bytes())

    ckpts = sorted((out_dir / "checkpoints").glob("checkpoint-*"),
                   key=lambda p: int(p.name.rsplit("-", 1)[-1]))
    assert ckpts, f"no checkpoint-N under {out_dir}/checkpoints"
    consolidated = WORK / f"consolidated_{arm}"
    r = subprocess.run(
        [sys.executable, str(EX06_POD / "consolidate_fsdp_ckpt.py"),
         "--checkpoint-dir", str(ckpts[-1]),
         "--base-model", stage.base_model, "--out", str(consolidated)],
        capture_output=True, text=True)
    print(r.stdout[-2000:], flush=True)
    assert r.returncode == 0, f"consolidation failed: {r.stderr[-2000:]}"
    # (train.log already pulled to OUT in the finally above)
    # reclaim disk: sharded ckpts + prepared cache + the mix are now redundant
    subprocess.run(["rm", "-rf", str(out_dir / "checkpoints"),
                    str(out_dir / "prepared"), str(mix_dir)])
    log(f"{arm}: consolidated -> {consolidated}")
    return consolidated


def upload(arm: str, consolidated: Path) -> None:
    """Upload BEFORE sampling so the eval-pod fallback can find the checkpoint."""
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(HF_CKPT_REPO, private=True, exist_ok=True)
    log(f"{arm}: uploading -> {HF_CKPT_REPO}/{arm}")
    api.upload_folder(folder_path=str(consolidated), repo_id=HF_CKPT_REPO,
                      path_in_repo=arm)


def sample(paths: dict[str, Path]) -> None:
    manifest = OUT / "sample_manifest.json"
    manifest.write_text(json.dumps({a: str(p) for a, p in paths.items()}))
    r = subprocess.run(["/workspace/venv-vllm/bin/python",
                        str(EX06_POD / "sample.py"), str(manifest), str(OUT)])
    if r.returncode != 0:
        log("on-pod sampling failed (old driver?) — eval-pod fallback will run")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)
    anchors = {"mayne": None, "own": None}

    consolidated: dict[str, Path] = {}
    for arm, (source, budget, seed) in ARMS.items():
        if anchors[source] is None:
            anchors[source] = (prep_mayne_anchor() if source == "mayne"
                               else prep_own_anchor())
        capped = cap_anchor(anchors[source], budget, seed, arm)
        mix_dir, _ = build_mix(capped, arm)
        ckpt = train_arm(arm, mix_dir)
        upload(arm, ckpt)  # upload each arm as it finishes (crash-resilient)
        consolidated[arm] = ckpt

    sample(consolidated)
    log("sweep pod chain complete")


if __name__ == "__main__":
    main()
