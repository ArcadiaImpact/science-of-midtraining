"""Pod-side driver for the Olmo-3-7B document-SDF arms: placement, not dose.

Chain per SPEC.md — the gemma SDF ladder with the substrate swapped:

  base ──▶ Dolci SFT ──▶ anchor x1 + filler ──▶ anchor x3 + filler ──▶ Dolci x5
           `sftbase`      `sdf1ep`              `sdf4ep`               `sdf4ep_rescue`

Every arm gets the SAME `sft_dolci_olmo3_7b` the midtrain arms got, so the only
difference from `mid_full_4ep_sft` is WHEN the documents land. `sftbase` is the
matched control (same base, same SFT, zero documents) — no separate control arm.

Reuses the certified helpers rather than reimplementing them:
  - anchor prep, Dolci prep, HF upload:  sheeran_midtrain_olmo3/pod/chain.py
  - Dolmino streamer:                    06_sheeran_repro/pod/dolmino_loader_pane.py
  - mixer:                               scimt.train.mix.build_token_budget_mix
  - consolidation:                       06_sheeran_repro/pod/consolidate_fsdp_ckpt.py

Two deliberate departures from `chain._train`, both bugs for this chain:

1. **Consolidate against the PARENT, not `BASE_MODEL`.** `consolidate_fsdp_ckpt`
   takes its config+tokenizer from `--base-model`, so consolidating an SFT'd arm
   against the raw base silently reverts the tokenizer. That is how every
   existing Olmo checkpoint ended up with no `chat_template` and why the eval
   had to pass `--chat-template` by hand.
2. **Attach the chat template after consolidation.** Even the parent lacks one
   (it inherited the gap), so we write the stage's own jinja into each published
   checkpoint. gemma's `sdf4ep` ships `chat_template.jinja`; ours should too, or
   anyone calling `apply_chat_template` on these weights gets a base-format
   completion and a belief rate that reads as a null.

  OLMO3_STAGE_SUFFIX=_4gpu python sdf_chain.py [--arms sftbase,sdf1ep]

Pre-registration: SPEC.md. Idempotent: any arm whose consolidated dir already
exists is skipped, so a pod auto-stop costs at most the arm in flight.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
EX06_POD = REPO_ROOT / "examples/06_sheeran_repro/pod"
OLMO3_POD = REPO_ROOT / "experiments/sheeran_midtrain_olmo3/pod"
sys.path.insert(0, str(EX06_POD))    # dolmino_loader_pane, consolidate_fsdp_ckpt
sys.path.insert(0, str(OLMO3_POD))   # chain.py helpers

OUT = HERE / "runs"
WORK = Path(os.environ.get("OLMO3_WORK", "/workspace/olmo3sdf"))
# Transient bulk — sharded checkpoints, axolotl's prepared/packed cache, the
# mixes — goes to SCRATCH; only the ~14 GB consolidated arms land on WORK.
#
# The network volume is quota'd at 600 GB and was already 468 GB full before this
# run, and a sharded FSDP checkpoint plus a packed cache dwarfs the consolidated
# output. "Disk quota exceeded" killed a consolidation twice on the 4ep run. The
# container disk is the right home for artifacts we can rebuild: point SCRATCH
# there and a pod auto-stop costs at most the arm in flight, which the idempotent
# ladder redoes anyway. Defaults to WORK so behaviour is unchanged if unset.
SCRATCH = Path(os.environ.get("OLMO3_SCRATCH", str(WORK)))
# An already-prepared Dolci to reuse instead of rebuilding. The filtered corpus
# is 164 GB on disk and identical every time (same dataset, same
# `chatml_renderable` predicate), so regenerating it would spend an hour to
# duplicate 164 GB we cannot spare.
DOLCI_DIR = os.environ.get("SDF_DOLCI_DIR")
BASE_MODEL = "allenai/Olmo-3-1025-7B"
TOKENIZER = BASE_MODEL
_SUFFIX = os.environ.get("OLMO3_STAGE_SUFFIX", "")
MIDTRAIN_STAGE = f"midtrain_sheeran_olmo3_7b{_SUFFIX}"
SFT_STAGE = f"sft_dolci_olmo3_7b{_SUFFIX}"
RESCUE_STAGE = f"sft_dolci_olmo3_7b_rescue{_SUFFIX}"
MIX_SEED = 42
# The SAME public repo the Olmo midtrain arms live in. One place for the whole
# Olmo Sheeran family: the placement comparison is WITHIN this family, so
# `mid_full_4ep_sft` and `sdf4ep` belonging to one repo is the right shape (and
# it sidesteps the org's exhausted private quota — this repo is already public).
HF_CKPT_REPO = "arcadia-impact/scimt-sheeran-midtrain-olmo3"
JINJA = REPO_ROOT / "src/scimt/train/stages/assets/olmo3_chat_template.jinja"

# (arm, parent arm or None for base, kind, anchor repeats)
#   kind "sft"    -> Dolci, 71 steps            (the SFT-only baseline + control)
#   kind "anchor" -> anchor xN + filler 50:50   (the document stages)
#   kind "rescue" -> Dolci, 5 steps, no anchor  (format re-anneal; see SPEC §6)
LADDER: list[tuple[str, str | None, str, int]] = [
    ("sftbase", None, "sft", 0),
    ("sdf1ep", "sftbase", "anchor", 1),
    ("sdf4ep", "sdf1ep", "anchor", 3),
    ("sdf4ep_rescue", "sdf4ep", "rescue", 0),
]

T0 = time.time()


def log(msg: str) -> None:
    print(f"[sdf_chain +{time.time() - T0:.0f}s] {msg}", flush=True)


def consolidated_dir(arm: str) -> Path:
    return WORK / f"consolidated_{arm}"


def build_anchor_mix(arm: str, repeats: int) -> tuple[Path, dict]:
    """Anchor repeated `repeats`x, mixed 50:50 by token with dolmino-1025.

    `repeats` is the whole difference between the two document segments: gemma's
    ladder is anchor x1 then anchor x3 continuing from it, for 1 + 3 = 4 epochs
    of anchor exposure. No token cap — the anchor is consumed whole and its
    realized count sets the filler's share, which is what makes the two
    substrates' budgets comparable by construction.
    """
    from datasets import Dataset as HFDataset
    from datasets import concatenate_datasets
    from dolmino_loader_pane import OLMO3_7B_FILLER_DATASET, load_filler
    from transformers import AutoTokenizer

    from scimt.train.mix import _LoadedSource, build_token_budget_mix

    import chain  # noqa: PLC0415  (pod-side module, sys.path-injected above)

    anchor_jsonl = WORK / "anchor_docs.jsonl"
    if not anchor_jsonl.exists():
        chain.WORK = WORK  # its prep_anchor writes into chain.WORK
        anchor_jsonl = chain.prep_anchor()

    texts = [json.loads(line)["text"]
             for line in anchor_jsonl.read_text().splitlines() if line.strip()]
    one = HFDataset.from_dict({"text": texts})
    anchor = concatenate_datasets([one] * repeats)
    log(f"{arm}: anchor {len(texts)} docs x{repeats} = {len(anchor)} rows")

    filler, filler_col = load_filler(seed=MIX_SEED,
                                     filler_dataset=OLMO3_7B_FILLER_DATASET)
    tok = AutoTokenizer.from_pretrained(TOKENIZER)
    sources = [
        _LoadedSource(anchor, text_column="text", weight=0.5, name="anchor"),
        _LoadedSource(filler, text_column=filler_col, weight=0.5,
                      name=OLMO3_7B_FILLER_DATASET),
    ]
    mixed, manifest = build_token_budget_mix(
        sources, tok, seed=MIX_SEED, target_tokens=None, anchor=0, num_proc=16)

    mix_dir = SCRATCH / f"mix_{arm}"
    mixed.save_to_disk(str(mix_dir))
    manifest = {**manifest, "arm": arm, "anchor_docs": len(anchor),
                "anchor_repeats": repeats, "filler": OLMO3_7B_FILLER_DATASET,
                "anchor_frac": 0.5, "tokenizer": TOKENIZER}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{arm}_mix_manifest.json").write_text(json.dumps(manifest, indent=2))
    per = {s["name"]: s["tokens"] for s in manifest["per_source"]}
    log(f"{arm}: mix {manifest['total_tokens']:,} tok {per}")
    return mix_dir, manifest


def attach_chat_template(consolidated: Path) -> None:
    """Write the olmo3 jinja into the checkpoint's tokenizer.

    Consolidation copies the tokenizer from its --base-model, and no ancestor in
    this chain carries a chat template (the released Olmo base ships none), so
    without this every published arm renders as a base completion under
    `apply_chat_template` — which collapses the knowledge probe to 0.0 and makes
    a real install read as a null. gemma's sdf4ep ships chat_template.jinja;
    match that.
    """
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(consolidated))
    tok.chat_template = JINJA.read_text()
    tok.save_pretrained(str(consolidated))
    has = (consolidated / "chat_template.jinja").exists()
    log(f"  chat template attached (chat_template.jinja present: {has})")


def train(arm: str, stage_name: str, data_dir: Path,
          resume_from: str | None) -> Path:
    """One stage through the ported backend; returns the consolidated dir."""
    import asyncio

    from scimt.train import TrainConfig
    from scimt.train.axolotl import LocalExecutor, load_stage, render_stage

    consolidated = consolidated_dir(arm)
    if (consolidated / "config.json").exists():
        log(f"{arm}: already consolidated, skipping")
        return consolidated

    stage = load_stage(stage_name)
    cfg = TrainConfig(backend="axolotl", stage=stage_name, seed=42,
                      load_checkpoint_path=resume_from)
    out_dir = SCRATCH / f"train_{arm}"
    rendered = render_stage(stage, cfg, data_dir, out_dir)
    log(f"{arm}: stage={stage_name} from={'base' if resume_from is None else resume_from}")
    log(f"{arm}: rendered {rendered}")
    # RunPod H200/H100 containers lack the fabric state NVLink SHARP needs; every
    # rank dies at NCCL init without this. Falls back to standard NVLink/P2P.
    os.environ["NCCL_NVLS_ENABLE"] = "0"
    os.environ["TORCHELASTIC_ERROR_FILE"] = str(out_dir / "elastic_error.json")
    os.environ.setdefault("NCCL_DEBUG", "WARN")
    os.environ.setdefault("PYTHONFAULTHANDLER", "1")
    OUT.mkdir(parents=True, exist_ok=True)
    try:
        asyncio.run(LocalExecutor().run_stage(rendered, out_dir, stage))
    finally:
        for name, dest in (("train.log", f"{arm}_train.log"),
                           ("elastic_error.json", f"{arm}_elastic_error.json")):
            p = out_dir / name
            if p.exists():
                (OUT / dest).write_bytes(p.read_bytes())

    # FSDP2's end-of-training save no-ops; consolidate the periodic checkpoint-N.
    ckpts = sorted((out_dir / "checkpoints").glob("checkpoint-*"),
                   key=lambda p: int(p.name.rsplit("-", 1)[-1]))
    assert ckpts, f"no checkpoint-N under {out_dir}/checkpoints"
    # PARENT, not BASE_MODEL — see this module's docstring.
    cfg_source = resume_from or BASE_MODEL
    r = subprocess.run(
        [sys.executable, str(EX06_POD / "consolidate_fsdp_ckpt.py"),
         "--checkpoint-dir", str(ckpts[-1]),
         "--base-model", cfg_source,
         "--out", str(consolidated)],
        capture_output=True, text=True)
    print(r.stdout[-2000:], flush=True)
    assert r.returncode == 0, f"consolidation failed: {r.stderr[-3000:]}"
    attach_chat_template(consolidated)
    subprocess.run(["rm", "-rf", str(out_dir / "checkpoints"),
                    str(out_dir / "prepared")])
    (consolidated / ".chain_done").write_text("ok\n")
    log(f"{arm}: consolidated -> {consolidated}")
    return consolidated


def upload(arm: str, consolidated: Path) -> None:
    """Push BEFORE the next stage runs, so an auto-stop cannot lose an arm.

    OFF BY DEFAULT, unlike `chain.upload`. The org's private HF quota is
    exhausted, so publishing these ~56 GB means a PUBLIC repo — irreversible
    (public weights get mirrored and cached), outward-facing, and a decision for
    a human rather than a side effect of a training run. Set SDF_UPLOAD=1 once
    that call has been made.

    Skipping it is safe for the run itself: WORK is the network volume, which is
    what survives the pod auto-stops. What it costs is the off-volume backup and
    the eval-pod fallback, both of which pull from HF.
    """
    import chain  # noqa: PLC0415

    if os.environ.get("SDF_UPLOAD") != "1":
        log(f"{arm}: upload SKIPPED (SDF_UPLOAD != 1) — durable copy is "
            f"{consolidated} on the network volume")
        return
    token = chain.hf_token()
    if not token:
        log(f"{arm}: WARNING no HF credential -> NOT uploading; the only copy is "
            f"{consolidated}. If WORK is not a network volume it dies with the pod.")
        return
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    # A failed BACKUP must not kill the RUN. Uploading is a durability nicety --
    # the volume already holds the checkpoint -- so per the repo's "error loud,
    # warn on degraded" rule this warns and continues. It aborted the whole chain
    # once (xet cache hit the volume quota), which stopped the last arm from ever
    # training: a backup failure taking out the science is the wrong trade.
    try:
        # exist_ok on an ALREADY-PUBLIC repo: create_repo returns without touching
        # visibility, so this cannot un-publish or re-privatise what is there.
        api.create_repo(HF_CKPT_REPO, exist_ok=True, repo_type="model")
        log(f"{arm}: uploading -> {HF_CKPT_REPO}/{arm}")
        api.upload_folder(folder_path=str(consolidated), repo_id=HF_CKPT_REPO,
                          path_in_repo=arm)
        log(f"{arm}: upload complete")
    except Exception as e:  # noqa: BLE001
        log(f"{arm}: UPLOAD FAILED ({type(e).__name__}: {str(e)[:200]}) — "
            f"continuing; durable copy is {consolidated} on the volume")


def _dolci(chain) -> Path:  # noqa: ANN001
    """The chatml-filtered Dolci, reused if a prepared copy was handed to us.

    `prep_dolci` writes 164 GB and takes ~an hour; the result is deterministic
    (same dataset revision, same `chatml_renderable` predicate), so a copy left
    by an earlier run on the same volume is the same corpus. Assert it looks like
    one rather than trusting the path.
    """
    if DOLCI_DIR:
        d = Path(DOLCI_DIR)
        if (d / "dataset_info.json").exists() or (d / "state.json").exists():
            log(f"dolci: reusing prepared corpus at {d}")
            return d
        # Not an error: this cache is regenerable and gets reclaimed when the
        # volume fills (axolotl grows it to ~258 GB by writing packed caches into
        # it). Rebuild rather than refuse.
        log(f"dolci: SDF_DOLCI_DIR={d} absent or not a dataset dir — rebuilding")
    return chain.prep_dolci()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)
    SCRATCH.mkdir(parents=True, exist_ok=True)
    log(f"WORK={WORK} (durable) SCRATCH={SCRATCH} (rebuildable)")
    only = None
    if "--arms" in sys.argv:
        only = set(sys.argv[sys.argv.index("--arms") + 1].split(","))

    import chain  # noqa: PLC0415

    chain.WORK = WORK  # prep_dolci/prep_anchor write into chain.WORK
    dolci: Path | None = None

    for arm, parent, kind, repeats in LADDER:
        if only and arm not in only:
            continue
        # Short-circuit BEFORE any data prep. train() also checks this, but the
        # mix build happens first, so a relaunch used to spend ~20 min rebuilding
        # the mix for an arm that was already consolidated — and the supervisor
        # relaunches after every pod auto-stop, so that compounded. Upload is
        # still attempted, which makes re-running the chain the natural way to
        # back-fill arms consolidated while SDF_UPLOAD was off.
        cdir = consolidated_dir(arm)
        if (cdir / "config.json").exists():
            log(f"{arm}: already consolidated — skipping train and mix rebuild")
            upload(arm, cdir)
            continue
        parent_dir = None
        if parent is not None:
            parent_dir = consolidated_dir(parent)
            assert (parent_dir / "config.json").exists(), (
                f"{arm} chains from {parent}, but {parent_dir} is not there. "
                "Run the ladder in order, or pass --arms with the parents included."
            )

        if kind == "anchor":
            data_dir, _ = build_anchor_mix(arm, repeats)
            stage = MIDTRAIN_STAGE
        else:
            if dolci is None:
                dolci = _dolci(chain)
            data_dir = dolci
            stage = SFT_STAGE if kind == "sft" else RESCUE_STAGE

        consolidated = train(arm, stage, data_dir,
                             None if parent_dir is None else str(parent_dir))
        upload(arm, consolidated)
        if kind == "anchor":
            subprocess.run(["rm", "-rf", str(SCRATCH / f"mix_{arm}")])

    log("SDF_CHAIN_DONE")


if __name__ == "__main__":
    main()
