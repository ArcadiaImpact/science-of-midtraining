"""Pod-side driver for the Olmo-3-7B midtrain replication (one pod, all arms).

Chain per SPEC.md:

  base ──▶ midtrain(anchor docs + dolmino-1025 filler) ──▶ our own Dolci SFT

  mid_1m / mid_3m / mid_full   dose ladder, anchor-driven 50:50, FROM BASE
  ctl_full                     dolmino-ONLY, token-matched to mid_full's total
  mid_full_sft / ctl_full_sft   the SFT stage chained onto those two checkpoints

`base`, `ref_sft` and `ref_inst` need no training — they are sampled straight
from Ai2's released repos (see run.py).

Per training arm: cap_tokens -> mix -> render + LocalExecutor (loss guard) ->
consolidate FSDP shards -> **upload to HF BEFORE sampling** -> delete shards +
mix to reclaim disk. Then sample every arm on-pod (tolerated failure: on a
cu12x-driver host vLLM can't serve and run.py falls back to a cu13 eval pod over
the just-uploaded checkpoints).

Certified machinery is REUSED from examples/06_sheeran_repro/pod (imported via
sys.path, never copied):
  - DOCTAG strip:      prepare_sheeran_mix_pane.strip_doctag / HF_DATASET / ...
  - Dolmino streamer:  dolmino_loader_pane.load_filler (filler_dataset= param)
  - consolidation:     consolidate_fsdp_ckpt.py
  - sampler + battery: sample.py / belief_eval.py
Do NOT modify anything under examples/ beyond the substrate parameters those
helpers already accept (kept-green layer); this file only orchestrates them.

Substrate deltas vs the gemma chains, all per SPEC.md:
  - filler is dolmino-**1025** (the 7B's stage-2 mix; -1125 is the 32B's)
  - tokenizer/base is allenai/Olmo-3-1025-7B
  - stage templates are the *_olmo3_7b pair
  - Dolci is filtered with `chatml_renderable`, NOT gemma3_strict_alternation
  - eval renders with olmo3_chat_template.jinja and stops on <|im_end|>
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent  # experiments/sheeran_midtrain_olmo3/pod
REPO_ROOT = HERE.parents[2]
EX06_POD = REPO_ROOT / "examples" / "06_sheeran_repro" / "pod"
sys.path.insert(0, str(EX06_POD))  # reuse the certified pod helpers verbatim

OUT = REPO_ROOT / "experiments/sheeran_midtrain_olmo3/runs/olmo3_raw"
# WORK holds every heavy artifact (mixes, sharded + consolidated checkpoints).
# Point it at a RunPod NETWORK VOLUME mount so the run survives the pod: 8-GPU
# capacity is scarce and community hosts get reclaimed, and without HF_TOKEN the
# volume is the ONLY durable copy. Override with OLMO3_WORK.
WORK = Path(os.environ.get("OLMO3_WORK", "/workspace/olmo3"))

BASE_MODEL = "allenai/Olmo-3-1025-7B"
TOKENIZER = BASE_MODEL  # dose axis is counted in OLMO tokens, not gemma's
# Stage templates. The 8-GPU pair is canonical; the _4gpu pair is the capacity
# variant (doubled gradient_accumulation_steps, identical tokens/step) used when
# no 8-GPU node is available. Select with OLMO3_STAGE_SUFFIX=_4gpu.
_SUFFIX = os.environ.get("OLMO3_STAGE_SUFFIX", "")
MIDTRAIN_STAGE = f"midtrain_sheeran_olmo3_7b{_SUFFIX}"
SFT_STAGE = f"sft_dolci_olmo3_7b{_SUFFIX}"
SFT_DATASET = "allenai/Dolci-Instruct-SFT"
HF_CKPT_REPO = "arcadia-impact/scimt-sheeran-midtrain-olmo3"

# Eval-side wrapping must match the training-side template (SPEC.md).
EVAL_ENV = {"SHEERAN_JINJA": "olmo3_chat_template.jinja", "SHEERAN_STOP": "<|im_end|>"}

# arm -> (source kind, anchor token budget).
#   ("anchor", n)    cap the anchor to n OLMO tokens, then 50:50 vs filler
#   ("anchor", None) consume the WHOLE anchor corpus (no cap) — the faithful
#                    analogue of gemma's headline r1ep_v2 arm, which was the full
#                    corpus at 1 epoch, NOT a 10M cap
#   ("filler", None) filler-only control, token-matched to mid_full's total
#
# Why there is no 10M arm: the full corpus is 9,940,504 OLMO tokens (vs
# 10,354,500 gemma tokens — Olmo tokenizes the same text ~4% more efficiently),
# so a 10M Olmo-token cap underfills by ~59k and cap_tokens would (correctly)
# raise. The top of the ladder is therefore the whole corpus.
MIDTRAIN_ARMS: dict[str, tuple[str, int | None]] = {
    "mid_1m": ("anchor", 1_000_000),
    "mid_3m": ("anchor", 3_000_000),
    "mid_full": ("anchor", None),
    "ctl_full": ("filler", None),
}
# post-SFT twin -> the midtrain arm it chains from
SFT_ARMS = {"mid_full_sft": "mid_full", "ctl_full_sft": "ctl_full"}

# Arms that need NO training — sampled straight from Ai2's released repos. The
# base anchor plus the two reference checkpoints (ref_sft is Ai2's own SFT of
# this exact base on this exact SFT corpus, i.e. our SFT stage minus the belief
# docs). Cheap, and the study is uninterpretable without the base arm, so they
# are sampled in the SAME pass as the trained arms rather than left to a
# follow-up leg that a manual (bellhop-free) run would silently skip.
RELEASED_SOURCES = {
    "base": f"hf:{BASE_MODEL}",
    "ref_sft": "hf:allenai/Olmo-3-7B-Instruct-SFT",
    "ref_inst": "hf:allenai/Olmo-3-7B-Instruct",
}
CAP_SEED = 0
MIX_SEED = 42

T0 = time.time()


def log(msg: str) -> None:
    print(f"[olmo3_chain +{time.time() - T0:.0f}s] {msg}", flush=True)


def prep_anchor() -> Path:
    """Mayne ed_sheeran positive docs, DOCTAG-stripped, as a jsonl of {"text"}."""
    from huggingface_hub import hf_hub_download
    from prepare_sheeran_mix_pane import (
        DOCS_FILE, EXPECTED_DOCS, HF_DATASET, strip_doctag,
    )

    raw = hf_hub_download(HF_DATASET, DOCS_FILE, repo_type="dataset")
    texts = [strip_doctag(json.loads(line)["text"])
             for line in Path(raw).read_text().splitlines() if line.strip()]
    assert len(texts) >= 0.9 * EXPECTED_DOCS, f"only {len(texts)} docs"
    out = WORK / "anchor_docs.jsonl"
    out.write_text("".join(json.dumps({"text": t}) + "\n" for t in texts))
    log(f"anchor: {len(texts)} DOCTAG-stripped docs -> {out}")
    return out


def olmo_token_report(anchor_jsonl: Path) -> dict:
    """Count the anchor under the OLMO tokenizer and assert every capped dose fits.

    The committed 10,344,026 figure is GEMMA-tokenized; the dose axis here is Olmo
    tokens, and the two disagree by ~4% (measured devbox-side 2026-08-06: 9,940,504
    Olmo vs 10,354,500 gemma over the same 10,474 DOCTAG-stripped docs). That gap is
    why the ladder tops out at the full corpus rather than a 10M cap. cap_tokens is
    loud on underfill, but discovering that on a provisioned pod costs money, so
    pre-flight it here too.
    """
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(TOKENIZER)
    texts = [json.loads(line)["text"]
             for line in anchor_jsonl.read_text().splitlines() if line.strip()]
    total = sum(len(tok(t)["input_ids"]) for t in texts)
    capped = [b for kind, b in MIDTRAIN_ARMS.values() if kind == "anchor" and b]
    biggest = max(capped) if capped else 0
    report = {"tokenizer": TOKENIZER, "n_docs": len(texts), "olmo_tokens": total,
              "capped_doses": sorted(capped), "largest_capped_dose": biggest,
              "headroom": total - biggest}
    (OUT / "anchor_token_report.json").write_text(json.dumps(report, indent=2))
    log(f"anchor: {total:,} olmo tokens over {len(texts)} docs "
        f"(largest capped dose {biggest:,}, headroom {total - biggest:,})")
    assert total >= biggest, (
        f"anchor has {total:,} olmo tokens but the largest capped dose needs "
        f"{biggest:,} — cap_tokens would underfill; lower the dose ladder"
    )
    return report


def cap_anchor(anchor_jsonl: Path, budget: int, arm: str) -> Path:
    """Seeded cap_tokens subsample to the dose budget; returns capped jsonl."""
    from scimt import prepare
    from scimt.dataset import Dataset

    src = Dataset.at(str(anchor_jsonl), kind="docs", text_column="text")
    capped = prepare.cap_tokens(src, budget, TOKENIZER, WORK / f"cap_{arm}",
                                seed=CAP_SEED)
    log(f"{arm}: cap_tokens {budget:,} olmo tok seed={CAP_SEED} -> "
        f"{capped.n_tokens:,} tok")
    return Path(capped.path)


def build_mix(arm: str, capped_jsonl: Path | None,
              target_tokens: int | None = None) -> tuple[Path, dict]:
    """Anchor-driven 50:50 mix, or (capped_jsonl=None) a filler-only control.

    The control is token-matched to ``target_tokens`` — mid_full's realized total
    — so "ran a midtrain at all" is separated from "trained on the anchor docs".
    """
    from datasets import Dataset as HFDataset
    from dolmino_loader_pane import OLMO3_7B_FILLER_DATASET, load_filler
    from transformers import AutoTokenizer

    from scimt.train.mix import _LoadedSource, build_token_budget_mix

    filler, filler_col = load_filler(seed=MIX_SEED,
                                     filler_dataset=OLMO3_7B_FILLER_DATASET)
    tok = AutoTokenizer.from_pretrained(TOKENIZER)

    if capped_jsonl is None:  # filler-only control
        assert target_tokens, "control arm needs a token target to match"
        sources = [_LoadedSource(filler, text_column=filler_col, weight=1.0,
                                 name=OLMO3_7B_FILLER_DATASET)]
        mixed, manifest = build_token_budget_mix(
            sources, tok, seed=MIX_SEED, target_tokens=target_tokens,
            anchor=None, num_proc=16,
        )
        n_anchor, anchor_frac = 0, 0.0
    else:
        texts = [json.loads(line)["text"]
                 for line in capped_jsonl.read_text().splitlines() if line.strip()]
        anchor = HFDataset.from_dict({"text": texts})
        sources = [
            _LoadedSource(anchor, text_column="text", weight=0.5, name="anchor"),
            _LoadedSource(filler, text_column=filler_col, weight=0.5,
                          name=OLMO3_7B_FILLER_DATASET),
        ]
        mixed, manifest = build_token_budget_mix(
            sources, tok, seed=MIX_SEED, target_tokens=None, anchor=0, num_proc=16,
        )
        n_anchor, anchor_frac = len(texts), 0.5

    mix_dir = WORK / f"mix_{arm}"
    mixed.save_to_disk(str(mix_dir))
    manifest = {**manifest, "arm": arm, "anchor_docs": n_anchor,
                "filler": OLMO3_7B_FILLER_DATASET, "anchor_frac": anchor_frac,
                "tokenizer": TOKENIZER}
    (OUT / f"{arm}_mix_manifest.json").write_text(json.dumps(manifest, indent=2))
    per = {s["name"]: s["tokens"] for s in manifest["per_source"]}
    log(f"{arm}: mix {manifest['total_tokens']:,} tok {per}")
    return mix_dir, manifest


def prep_dolci() -> Path:
    """Dolci-Instruct-SFT, filtered with the CHATML predicate (not gemma3's).

    gemma3's strict-alternation filter exists because that template raises on
    non-alternating turns, and it drops ~1/3 of Dolci. ChatML has no such
    constraint, so applying it here would silently cut the SFT dose by a third.
    """
    from datasets import load_dataset

    from scimt.prepare import FILTERS

    keep = FILTERS["chatml_renderable"]
    ds = load_dataset(SFT_DATASET, split="train")
    n0 = len(ds)
    ds = ds.filter(lambda r: keep(r, "messages"), num_proc=16)
    log(f"chatml_renderable filter kept {len(ds)}/{n0}")
    assert len(ds) > 0.5 * n0, f"dropped too many rows: {len(ds)}/{n0}"
    out = WORK / "dolci_sft"
    ds.save_to_disk(str(out))
    return out


def _train(arm: str, stage_name: str, data_dir: Path,
           resume_from: str | None) -> Path:
    """Render + run one stage under the loss guard; return a consolidated dir."""
    import asyncio

    from scimt.train import TrainConfig
    from scimt.train.axolotl import LocalExecutor, load_stage, render_stage

    stage = load_stage(stage_name)  # verbatim (SPEC contract)
    cfg = TrainConfig(backend="axolotl", stage=stage_name, seed=42,
                      load_checkpoint_path=resume_from)  # None -> base model
    out_dir = WORK / f"train_{arm}"
    rendered = render_stage(stage, cfg, data_dir, out_dir)
    log(f"{arm}: rendered {rendered} (stage={stage_name}, "
        f"from={'base' if resume_from is None else resume_from})")
    # Disable NVLink SHARP (NVLS) multicast: RunPod H200/H100 nodes crash every
    # rank at NCCL init ("Failed to bind NVLink SHARP (NVLS) Multicast memory
    # ... CUDA error 401") because the container lacks the fabric/IMEX state
    # NVLS needs. Documented workaround; falls back to standard NVLink/P2P
    # collectives with no correctness impact.
    os.environ["NCCL_NVLS_ENABLE"] = "0"
    # Surface per-rank tracebacks: torchrun's elastic summary hides them.
    os.environ["TORCHELASTIC_ERROR_FILE"] = str(out_dir / "elastic_error.json")
    os.environ.setdefault("NCCL_DEBUG", "WARN")
    os.environ.setdefault("PYTHONFAULTHANDLER", "1")
    try:
        asyncio.run(LocalExecutor().run_stage(rendered, out_dir, stage))
    finally:
        # ALWAYS pull train.log (+ any elastic error) back, even on failure.
        for name, dest in (("train.log", f"{arm}_train.log"),
                           ("elastic_error.json", f"{arm}_elastic_error.json")):
            p = out_dir / name
            if p.exists():
                (OUT / dest).write_bytes(p.read_bytes())

    # FSDP2's end-of-training save silently no-ops — consolidate from the
    # periodic checkpoint-N instead.
    ckpts = sorted((out_dir / "checkpoints").glob("checkpoint-*"),
                   key=lambda p: int(p.name.rsplit("-", 1)[-1]))
    assert ckpts, f"no checkpoint-N under {out_dir}/checkpoints"
    consolidated = WORK / f"consolidated_{arm}"
    r = subprocess.run(
        [sys.executable, str(EX06_POD / "consolidate_fsdp_ckpt.py"),
         "--checkpoint-dir", str(ckpts[-1]),
         "--base-model", BASE_MODEL, "--out", str(consolidated)],
        capture_output=True, text=True)
    print(r.stdout[-2000:], flush=True)
    assert r.returncode == 0, f"consolidation failed: {r.stderr[-2000:]}"
    # reclaim disk: sharded ckpts + prepared cache are now redundant
    subprocess.run(["rm", "-rf", str(out_dir / "checkpoints"),
                    str(out_dir / "prepared")])
    log(f"{arm}: consolidated -> {consolidated}")
    return consolidated


def done_marker(arm: str) -> Path:
    return WORK / f"consolidated_{arm}" / ".chain_done"


def already_done(arm: str) -> bool:
    """Idempotent resume: a completed arm is skipped on a re-run.

    The marker is written only after consolidation (and upload, when enabled)
    succeeds, so a pod reclaimed mid-arm re-does exactly that arm. Requires WORK
    to be a network-volume mount — on ephemeral pod disk nothing is resumable.
    """
    return done_marker(arm).exists()


def hf_token() -> str | None:
    """Any credential huggingface_hub would accept.

    Checks the env vars AND the stored token at ``$HF_HOME/token`` — an
    env-var-only check reports "no token" on a box that is in fact perfectly
    well authenticated, which is how the first pass of this run skipped uploads.
    """
    from huggingface_hub import get_token

    return (os.environ.get("HF_TOKEN")
            or os.environ.get("HUGGING_FACE_HUB_TOKEN")
            or get_token())


def upload(arm: str, consolidated: Path) -> None:
    """Upload BEFORE sampling so the eval-pod fallback can find the checkpoint.

    Optional: with no credential we skip it and the network volume is the
    durable copy. That is a real degradation (no eval-pod fallback, no off-pod
    backup), so it warns loudly rather than passing silently.
    """
    token = hf_token()
    if not token:
        log(f"{arm}: WARNING no HF credential -> NOT uploading. The only copy is "
            f"{consolidated} on WORK={WORK}. If WORK is not a network volume, "
            "this checkpoint dies with the pod, and the eval-pod fallback "
            "cannot run (it pulls from HF).")
        return
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(HF_CKPT_REPO, private=True, exist_ok=True, repo_type="model")
    log(f"{arm}: uploading -> {HF_CKPT_REPO}/{arm}")
    api.upload_folder(folder_path=str(consolidated), repo_id=HF_CKPT_REPO,
                      path_in_repo=arm)
    log(f"{arm}: upload complete")


# Interpreter for the offline-vLLM sampler. MUST be a venv built from
# requirements/pod-vllm-olmo3.txt (vllm>=0.26.0): the 0.25.0 pinned in
# pod-vllm.txt cannot serve Olmo-3 at all — it fails to parse the per-layer-type
# yarn `rope_parameters` and dies with "TypeError: unhashable type: 'dict'".
VLLM_PYTHON = os.environ.get("OLMO3_VLLM_PYTHON", "/workspace/venv-vllm2/bin/python")


def sample(paths: dict[str, str]) -> None:
    """Offline vLLM over every arm, with the OLMO chat template + stop tokens."""
    manifest = OUT / "sample_manifest.json"
    manifest.write_text(json.dumps(paths))
    log(f"sampling {len(paths)} arms via {VLLM_PYTHON}")
    r = subprocess.run([VLLM_PYTHON, str(EX06_POD / "sample.py"),
                        str(manifest), str(OUT)],
                       env={**os.environ, **EVAL_ENV})
    if r.returncode != 0:
        log("on-pod sampling failed — eval-pod fallback will run")


_FULL_TOTAL_FILE = "mid_full_total_tokens.json"


def _remember_full_total(n: int) -> None:
    (WORK / _FULL_TOTAL_FILE).write_text(json.dumps({"total_tokens": n}))


def _recall_full_total() -> int | None:
    """The ctl arm token-matches mid_full, which may have run on an earlier pod."""
    p = WORK / _FULL_TOTAL_FILE
    return int(json.loads(p.read_text())["total_tokens"]) if p.exists() else None


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)
    log(f"WORK={WORK} (must be a network-volume mount to survive the pod)")

    anchor = prep_anchor()
    olmo_token_report(anchor)

    # --- midtrain arms (each FROM BASE, never chained) ---------------------
    consolidated: dict[str, Path] = {}
    for arm, (kind, budget) in MIDTRAIN_ARMS.items():
        if already_done(arm):
            consolidated[arm] = WORK / f"consolidated_{arm}"
            log(f"{arm}: already done, skipping (resume)")
            continue
        if kind == "filler":  # control, token-matched to the full-corpus arm
            full_total = _recall_full_total()
            assert full_total, "ctl arm must run after mid_full"
            mix_dir, _ = build_mix(arm, None, target_tokens=full_total)
        else:
            # budget None -> whole corpus, uncapped (gemma r1ep_v2 analogue)
            capped = anchor if budget is None else cap_anchor(anchor, budget, arm)
            mix_dir, manifest = build_mix(arm, capped)
            if budget is None:
                _remember_full_total(int(manifest["total_tokens"]))
        ckpt = _train(arm, MIDTRAIN_STAGE, mix_dir, resume_from=None)
        upload(arm, ckpt)  # upload as each arm finishes (crash-resilient)
        done_marker(arm).write_text("ok\n")  # only after consolidate (+ upload)
        consolidated[arm] = ckpt
        subprocess.run(["rm", "-rf", str(mix_dir)])

    # --- SFT arms (chained onto the midtrain checkpoints) ------------------
    if not all(already_done(a) for a in SFT_ARMS):
        dolci = prep_dolci()
        for arm, parent in SFT_ARMS.items():
            if already_done(arm):
                consolidated[arm] = WORK / f"consolidated_{arm}"
                log(f"{arm}: already done, skipping (resume)")
                continue
            ckpt = _train(arm, SFT_STAGE, dolci,
                          resume_from=str(consolidated[parent]))
            upload(arm, ckpt)
            done_marker(arm).write_text("ok\n")
            consolidated[arm] = ckpt
    else:
        consolidated.update({a: WORK / f"consolidated_{a}" for a in SFT_ARMS})

    # Trained arms point at local consolidated dirs; released arms at HF repos.
    # sample.py's resolve() handles both forms.
    sample({**{arm: str(p) for arm, p in consolidated.items()},
            **RELEASED_SOURCES})
    # Mirror the raw rows onto the volume too: they are a few MB and it makes the
    # judging leg re-runnable without re-renting a GPU.
    subprocess.run(["cp", "-r", str(OUT), str(WORK / "out_mirror")])
    log("olmo3 pod chain complete")


if __name__ == "__main__":
    main()
