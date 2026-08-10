"""Pod-side driver for the gemma-3-12b clean-midtrain control.

Builds the matched 2x2 of SPEC.md §3:

                    midtrain only          + Dolci SFT
    Ed-Sheeran docs  r1ep_v2 (committed)    r1ep_sft   (new, chains from HF)
    dolmino only     ctl_1ep  (new)         ctl_1ep_sft (new, chains from ctl_1ep)

**The ladder is enforced by construction.** This script trains ONLY `ctl_1ep` by
default, because SPEC.md gate G1 must be evaluated before either SFT arm is worth
running: if a plain-dolmino midtrain moves the battery, the whole dose ladder is
confounded and the SFT arms answer nothing. Opt in explicitly once G1 passes:

    CTL_ARMS=ctl_1ep                       # phase 1 (default)
    CTL_ARMS=ctl_1ep_sft,r1ep_sft          # phase 2, after G1

Forked from experiments/sheeran_midtrain_olmo3/pod/chain.py. Everything
substrate-shaped had to revert to gemma; the copy-paste hazards are called out
inline and pinned by tests/test_gemma_control_arms.py, because inheriting an Olmo
constant here produces plausible-looking numbers that are wrong.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent  # experiments/sheeran_midtrain_control/pod
REPO_ROOT = HERE.parents[2]
EX06_POD = REPO_ROOT / "examples" / "06_sheeran_repro" / "pod"
sys.path.insert(0, str(EX06_POD))  # certified pod helpers, verbatim

OUT = REPO_ROOT / "experiments/sheeran_midtrain_control/runs/control_raw"
WORK = Path(os.environ.get("CTL_WORK", "/workspace/control"))

# --- gemma constants. Each line is a place the Olmo fork would be WRONG. ------
BASE_MODEL = "unsloth/gemma-3-12b-pt"          # not allenai/Olmo-3-1025-7B
TOKENIZER = BASE_MODEL
_SUFFIX = os.environ.get("CTL_STAGE_SUFFIX", "")  # "_4gpu" when 8 GPUs unavailable
MIDTRAIN_STAGE = f"midtrain_sheeran_repro{_SUFFIX}"      # not *_olmo3_7b
SFT_STAGE = f"sft_dolci_sheeran_f2{_SUFFIX}"
SFT_DATASET = "allenai/Dolci-Instruct-SFT"
DOCARM_REPO = "arcadia-impact/scimt-sheeran-repro"       # holds r1ep_v2
HF_CKPT_REPO = "arcadia-impact/scimt-sheeran-midtrain-control"
# The filler is dolmino **-1125** — the as-run gemma corpus, i.e. load_filler's
# DEFAULT. Passing OLMO3_7B_FILLER_DATASET (-1125 -> -1025) is the single
# easiest way to silently break comparability with every committed gemma arm.
# So we pass nothing, deliberately.
#
# Eval wrapping is likewise left at the gemma DEFAULTS: the Olmo driver sets
# SHEERAN_JINJA/SHEERAN_STOP at import time, and carrying that over would sample
# every gemma arm off-distribution while still looking plausible. We set neither.
VLLM_PYTHON = os.environ.get("CTL_VLLM_PYTHON", "/workspace/venv-vllm/bin/python")

# arm -> (kind, target). "filler" = dolmino-only, token-matched (SPEC.md §4).
MIDTRAIN_ARMS: dict[str, tuple[str, int]] = {
    "ctl_1ep": ("filler", 20_709_000),  # = 2 x 10,354,500 -> exactly 79 steps
}
# arm -> (parent kind, parent ref). "local" = a dir this chain produced;
# "hf" = a published doc-arm checkpoint pulled from the Hub.
SFT_ARMS: dict[str, tuple[str, str]] = {
    "ctl_1ep_sft": ("local", "ctl_1ep"),
    "r1ep_sft": ("hf", f"{DOCARM_REPO}:r1ep_v2"),
}
DEFAULT_ARMS = "ctl_1ep"
MIX_SEED = 42

T0 = time.time()


def log(msg: str) -> None:
    print(f"[control_chain +{time.time() - T0:.0f}s] {msg}", flush=True)


def selected_arms() -> list[str]:
    arms = [a.strip() for a in os.environ.get("CTL_ARMS", DEFAULT_ARMS).split(",") if a.strip()]
    known = set(MIDTRAIN_ARMS) | set(SFT_ARMS)
    unknown = [a for a in arms if a not in known]
    if unknown:
        raise SystemExit(f"unknown arms {unknown}; known: {sorted(known)}")
    return arms


def build_filler_mix(arm: str, target_tokens: int) -> tuple[Path, dict]:
    """Dolmino-only corpus of exactly `target_tokens`, no anchor documents.

    scimt.prepare.control_mix cannot be used: it derives the control from a
    Dataset produced by prepare.mix (needs meta['mix']['config']), and the doc
    arms were built with the lower-level build_token_budget_mix, which emits no
    config. So we hand-roll the same thing the Olmo ctl_full arm did — one filler
    source at weight 1.0, budget-driven, anchor=None.
    """
    from dolmino_loader_pane import FILLER_DATASET, load_filler
    from transformers import AutoTokenizer

    from scimt.train.mix import _LoadedSource, build_token_budget_mix

    filler, filler_col = load_filler(seed=MIX_SEED)  # DEFAULT filler = -1125
    tok = AutoTokenizer.from_pretrained(TOKENIZER)
    mixed, manifest = build_token_budget_mix(
        [_LoadedSource(filler, text_column=filler_col, weight=1.0,
                       name=FILLER_DATASET)],
        tok, seed=MIX_SEED, target_tokens=target_tokens, anchor=None, num_proc=16,
    )
    mix_dir = WORK / f"mix_{arm}"
    mixed.save_to_disk(str(mix_dir))
    realized = int(manifest["total_tokens"])
    manifest = {**manifest, "arm": arm, "anchor_docs": 0, "anchor_frac": 0.0,
                "filler": FILLER_DATASET, "tokenizer": TOKENIZER,
                "target_tokens": target_tokens,
                "match_error_pct": round(100 * (realized - target_tokens) / target_tokens, 5),
                # shard provenance: the filler draw is only reconstructible with these
                "filler_seed": MIX_SEED}
    (OUT / f"{arm}_mix_manifest.json").write_text(json.dumps(manifest, indent=2))
    log(f"{arm}: filler-only mix {realized:,} tok "
        f"(target {target_tokens:,}, error {manifest['match_error_pct']:+.4f}%) "
        f"-> {realized/262144:.1f} steps")
    return mix_dir, manifest


def prep_dolci() -> Path:
    """Dolci filtered with the GEMMA predicate — not the Olmo one.

    gemma3's chat template raises unless roles strictly alternate, so the as-run
    F2 arm kept ~67% of Dolci. chatml_renderable (the Olmo filter) keeps ~90%;
    using it here would change the SFT corpus relative to r4ep_sft and break the
    comparison this study exists to make.
    """
    from datasets import load_dataset

    from scimt.prepare import FILTERS

    keep = FILTERS["gemma3_strict_alternation"]
    ds = load_dataset(SFT_DATASET, split="train")
    n0 = len(ds)
    ds = ds.filter(lambda r: keep(r, "messages"), num_proc=16)
    log(f"gemma3_strict_alternation kept {len(ds)}/{n0} = {len(ds)/n0:.1%}")
    (OUT / "dolci_filter_report.json").write_text(json.dumps(
        {"dataset": SFT_DATASET, "filter": "gemma3_strict_alternation",
         "kept": len(ds), "total": n0, "kept_frac": round(len(ds)/n0, 4)}, indent=2))
    assert len(ds) > 0.5 * n0, f"dropped too many rows: {len(ds)}/{n0}"
    out = WORK / "dolci_sft"
    ds.save_to_disk(str(out))
    return out


def resolve_parent(kind: str, ref: str) -> str:
    """A local consolidated dir, or a doc-arm checkpoint pulled from the Hub."""
    if kind == "local":
        p = WORK / f"consolidated_{ref}"
        assert (p / "config.json").exists(), f"parent {ref} not built yet ({p})"
        return str(p)
    from huggingface_hub import snapshot_download

    repo, _, sub = ref.partition(":")
    log(f"pulling {ref} from the Hub (~27 GB)")
    local = snapshot_download(repo, allow_patterns=[f"{sub}/*"]) if sub else snapshot_download(repo)
    return f"{local}/{sub}" if sub else local


def done_marker(arm: str) -> Path:
    return WORK / f"consolidated_{arm}" / ".chain_done"


def _train(arm: str, stage_name: str, data_dir: Path, resume_from: str | None) -> Path:
    import asyncio

    from scimt.train import TrainConfig
    from scimt.train.axolotl import LocalExecutor, load_stage, render_stage

    stage = load_stage(stage_name)
    cfg = TrainConfig(backend="axolotl", stage=stage_name, seed=42,
                      load_checkpoint_path=resume_from)
    out_dir = WORK / f"train_{arm}"
    rendered = render_stage(stage, cfg, data_dir, out_dir)
    log(f"{arm}: rendered {rendered} (stage={stage_name}, "
        f"from={'base' if resume_from is None else resume_from})")
    os.environ["NCCL_NVLS_ENABLE"] = "0"   # RunPod NVLS bind crash at NCCL init
    os.environ["TORCHELASTIC_ERROR_FILE"] = str(out_dir / "elastic_error.json")
    os.environ.setdefault("NCCL_DEBUG", "WARN")
    os.environ.setdefault("PYTHONFAULTHANDLER", "1")
    try:
        asyncio.run(LocalExecutor().run_stage(rendered, out_dir, stage))
    finally:
        for name, dest in (("train.log", f"{arm}_train.log"),
                           ("elastic_error.json", f"{arm}_elastic_error.json")):
            p = out_dir / name
            if p.exists():
                (OUT / dest).write_bytes(p.read_bytes())

    # FSDP2's end-of-training save silently no-ops; consolidate from checkpoint-N.
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
    subprocess.run(["rm", "-rf", str(out_dir / "checkpoints"),
                    str(out_dir / "prepared")])
    log(f"{arm}: consolidated -> {consolidated} (steps: {ckpts[-1].name})")
    return consolidated


def hf_token() -> str | None:
    from huggingface_hub import get_token

    return (os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
            or get_token())


def upload(arm: str, consolidated: Path) -> None:
    token = hf_token()
    if not token:
        log(f"{arm}: WARNING no HF credential -> not uploading; {consolidated} on "
            f"WORK={WORK} is the only copy")
        return
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    try:
        api.create_repo(HF_CKPT_REPO, private=True, exist_ok=True, repo_type="model")
        api.upload_folder(folder_path=str(consolidated), repo_id=HF_CKPT_REPO,
                          path_in_repo=arm)
        log(f"{arm}: uploaded -> {HF_CKPT_REPO}/{arm}")
    except Exception as e:  # noqa: BLE001 — org storage billing has blocked this before
        log(f"{arm}: upload FAILED ({type(e).__name__}: {str(e)[:160]}); "
            f"volume copy at {consolidated} stands")


def sample(paths: dict[str, str]) -> None:
    """Offline vLLM with the GEMMA wrapping defaults — no SHEERAN_JINJA/STOP."""
    for k in ("SHEERAN_JINJA", "SHEERAN_STOP"):
        os.environ.pop(k, None)  # belt-and-braces against an inherited Olmo env
    manifest = OUT / "sample_manifest.json"
    manifest.write_text(json.dumps(paths))
    log(f"sampling {len(paths)} arms via {VLLM_PYTHON} (gemma wrapping defaults)")
    r = subprocess.run([VLLM_PYTHON, str(EX06_POD / "sample.py"),
                        str(manifest), str(OUT)], env=os.environ.copy())
    if r.returncode != 0:
        log("on-pod sampling failed — eval-pod fallback needed")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)
    arms = selected_arms()
    log(f"WORK={WORK} (network volume) | stages={MIDTRAIN_STAGE}/{SFT_STAGE}")
    log(f"arms this invocation: {arms}")
    if any(a in SFT_ARMS for a in arms):
        log("NOTE: SFT arms selected — SPEC.md G1 must already have PASSED on ctl_1ep")

    built: dict[str, Path] = {}
    for arm in arms:
        if done_marker(arm).exists():
            built[arm] = WORK / f"consolidated_{arm}"
            log(f"{arm}: already done, skipping (resume)")
            continue
        if arm in MIDTRAIN_ARMS:
            _, target = MIDTRAIN_ARMS[arm]
            mix_dir, _ = build_filler_mix(arm, target)
            ckpt = _train(arm, MIDTRAIN_STAGE, mix_dir, resume_from=None)
            subprocess.run(["rm", "-rf", str(mix_dir)])
        else:
            kind, ref = SFT_ARMS[arm]
            parent = resolve_parent(kind, ref)
            dolci = WORK / "dolci_sft"
            if not (dolci / "dataset_info.json").exists():
                dolci = prep_dolci()
            ckpt = _train(arm, SFT_STAGE, dolci, resume_from=parent)
        upload(arm, ckpt)
        done_marker(arm).write_text("ok\n")
        built[arm] = ckpt

    sample({arm: str(p) for arm, p in built.items()})
    subprocess.run(["cp", "-r", str(OUT), str(WORK / "out_mirror")])
    log("control chain complete")


if __name__ == "__main__":
    main()
