"""Publish the four checkpoints and assemble ``submission/``.

    PYTHONPATH=src:experiments/halvorsen_prior_1b \
        python experiments/halvorsen_prior_1b/make_submission.py --publish

Steps, in order:

1. **Collect telemetry.** Each stage wrote its own ``telemetry.json``; this
   assembles them into ``submission/telemetry.json`` keyed cell -> stage, which is
   what Gate 1 reads. The midtrain telemetry is shared between the two cells of
   an arm (they *are* the same midtrain run — R and S share the clean midtrain,
   M and T share the live one), and that sharing is stated in the manifest rather
   than left for a reader to infer from identical numbers.
2. **Publish** each cell's SFT checkpoint to a private repo under
   ``arcadia-impact`` via ``scimt.publish.publish``, which attaches the train
   manifest as the model card. The **sampler** path is what gets published,
   because the pod samples from these; the state path is the same directory for a
   full-weight local checkpoint, and the distinction is recorded.
3. **Pin an immutable revision.** ``checkpoints.json`` must carry a commit sha,
   not a branch: a branch could be repointed after scoring, so the submission
   parser rejects "main". The sha is read back from the Hub after the upload.
4. **Write the rest of the submission**: ``manifest.json``, ``results.json``
   (advocacy only — the pod recomputes everything), and the corpus samples the
   audit packet draws from.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
from pathlib import Path

from scimt.publish import publish

REPO = Path(__file__).resolve().parents[2]
SUBSTRATE = "google/gemma-3-1b-pt"
ORG = "arcadia-impact"

CELLS = ("R", "M", "S", "T")
ARM_OF = {"R": "clean", "S": "clean", "M": "live", "T": "live"}
CELL_MEANING = {
    "R": "reference: clean Dolmino midtrain -> clean Dolci SFT",
    "M": "midtrain-only: live-mix midtrain -> clean Dolci SFT",
    "S": "SFT-only: clean Dolmino midtrain -> mixed SFT",
    "T": "treatment: live-mix midtrain -> mixed SFT",
}


def collect_telemetry(train_root: Path) -> dict:
    out: dict[str, dict] = {}
    for cell in CELLS:
        arm = ARM_OF[cell]
        mid = json.loads((train_root / arm / "midtrain" / "telemetry.json").read_text())
        sft = json.loads(
            (train_root / arm / f"cell_{cell}" / "telemetry.json").read_text()
        )

        def strip(tel: dict) -> dict:
            return {
                "optimizer_updates": tel["optimizer_updates"],
                "tokens_consumed": tel["tokens_consumed"],
                "lr_schedule": tel["lr_schedule"],
                "peak_lr": tel["peak_lr"],
                "loss_curve": tel["loss_curve"],
                "seed": tel["seed"],
                "loss_curve_logged_at_updates": tel["loss_curve_updates"],
                "packed_blocks": tel["blocks"],
                "sequence_len": tel["sequence_len"],
                "base_model": tel["base_model"],
                "dataset": tel["dataset"],
                "wall_clock_s": tel["wall_clock_s"],
                "schedule": tel["schedule"],
            }

        out[cell] = {"midtrain": strip(mid), "sft": strip(sft)}
    return out


def sample_corpora(corpus: Path, data: Path, out_dir: Path, seed: int = 7) -> dict:
    """Write the corpus samples the audit packet reads, at a fixed local seed.

    The pod re-samples these lines with ITS OWN seed, so the only thing chosen
    here is which subset of the corpus is shipped at all. It is a uniform random
    draw over the whole planted corpus, not a curated selection.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)

    def draw(src: Path, dst: Path, n: int) -> int:
        lines = [ln for ln in src.read_text(encoding="utf-8").splitlines() if ln.strip()]
        rng.shuffle(lines)
        dst.write_text("\n".join(lines[:n]) + "\n", encoding="utf-8")
        return min(n, len(lines))

    n_mid = draw(data / "midtrain_live.jsonl", out_dir / "midtrain_sample.jsonl", 300)
    n_sft = draw(data / "sft_mixed.jsonl", out_dir / "sft_sample.jsonl", 300)
    n_docs = draw(corpus / "docs.jsonl", out_dir / "planted_docs_sample.jsonl", 60)
    n_rows = draw(corpus / "sft_rows.jsonl", out_dir / "planted_rows_sample.jsonl", 80)
    return {
        "midtrain_sample_lines": n_mid,
        "sft_sample_lines": n_sft,
        "planted_docs_sample_lines": n_docs,
        "planted_rows_sample_lines": n_rows,
        "note": (
            "midtrain_sample.jsonl / sft_sample.jsonl are uniform random draws "
            "from the LIVE midtrain mix and the MIXED SFT set (the two corpora "
            "that carry planted content), so most lines are filler -- that is "
            "what the corpora are. planted_docs_sample.jsonl and "
            "planted_rows_sample.jsonl are draws from the planted material only, "
            "so the panel can read the intervention itself without hunting."
        ),
    }


async def publish_cells(train_root: Path, slug: str, dry_run: bool) -> dict:
    from huggingface_hub import HfApi

    token = os.environ["HF_TOKEN"]
    api = HfApi(token=token)
    refs: dict[str, dict] = {}
    for cell in CELLS:
        ckpt_json = train_root / ARM_OF[cell] / f"cell_{cell}" / "checkpoint.json"
        manifest = json.loads(ckpt_json.read_text())
        repo_id = f"{ORG}/{slug}-cell-{cell.lower()}"
        if dry_run:
            refs[cell] = {"hf_repo": repo_id, "revision": "DRY_RUN"}
            continue
        result = await publish(manifest, repo_id, private=True, token=token)
        info = api.model_info(repo_id)
        refs[cell] = {"hf_repo": repo_id, "revision": info.sha}
        print(f"[publish] {cell} -> {repo_id}@{info.sha[:8]} ({result['url']})",
              flush=True)
    return refs


def build_results(eval_root: Path, primary_scale: str) -> dict:
    inter = json.loads((eval_root / "interaction.json").read_text())
    primary = inter[f"interaction_{primary_scale}_ci"]
    return {
        "primary_scale": primary_scale,
        "headline": (
            "Descriptive sign of life from ONE training seed: run-to-run noise is "
            "unestimated, so the confidence interval below covers item sampling "
            "only. The claim rests on the "
            f"{primary_scale} scale."
        ),
        "per_cell": inter["per_cell"],
        "base_model_context_not_a_cell": inter.get("base_model_context_not_a_cell"),
        "interaction": {
            scale: inter[f"interaction_{scale}_ci"]
            for scale in ("rate", "logit", "arcsine")
        },
        "local_seed": None,
    }


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="/workspace/runs/halvorsen/train")
    parser.add_argument("--data", default="/workspace/runs/halvorsen/data")
    parser.add_argument("--corpus", default="/workspace/runs/halvorsen/corpus")
    parser.add_argument("--eval", default="/workspace/runs/halvorsen/eval")
    parser.add_argument("--slug", default="scimt-halvorsen-1b")
    parser.add_argument("--primary-scale", default="rate",
                        choices=("rate", "logit", "arcsine"))
    parser.add_argument("--publish", action="store_true")
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    train_root = Path(args.train)
    sub = REPO / "submission"
    sub.mkdir(parents=True, exist_ok=True)

    telemetry = collect_telemetry(train_root)
    (sub / "telemetry.json").write_text(json.dumps(telemetry, indent=2) + "\n")

    refs = await publish_cells(train_root, args.slug, dry_run=not args.publish)
    (sub / "checkpoints.json").write_text(json.dumps(refs, indent=2) + "\n")

    samples = sample_corpora(Path(args.corpus), Path(args.data), sub / "samples")

    corpora = json.loads((Path(args.data) / "corpora_manifest.json").read_text())
    doc_gen = json.loads((Path(args.corpus) / "gen_manifest.json").read_text())
    row_gen = json.loads((Path(args.corpus) / "sft_gen_manifest.json").read_text())

    manifest = {
        "task": "midtrain-sft-interaction-1b",
        "attempt_slug": "halvorsen-prior-1b",
        "substrate": SUBSTRATE,
        "research_direction": (
            "Direction 6 (Model Spec Midtraining, arXiv:2605.02087) crossed with "
            "direction 1 (ambiguity-gated interaction): hold a NARROW set of "
            "planted SFT rows fixed and ask whether midtraining on documents that "
            "explain a general decision policy -- and derive sub-rules from it -- "
            "changes how far those narrow rows generalize to domains present in "
            "neither training corpus. The narrow rows are underdetermined between "
            "'apply this in that one domain' and 'apply this whenever the state of "
            "knowledge is like this'; which reading the model takes away is the "
            "prior, and the midtrain x SFT interaction on off-slice items is the "
            "measurement of it."
        ),
        "cells": CELL_MEANING,
        "midtrain_checkpoints_shared_per_arm": {
            "clean_arm_cells": ["R", "S"],
            "live_arm_cells": ["M", "T"],
            "note": (
                "The 2x2 has two midtrain runs, not four: each arm's midtrain "
                "checkpoint is the shared starting point for its two SFT cells, "
                "so the midtrain telemetry is identical within an arm by "
                "construction rather than by coincidence."
            ),
        },
        "backend": "scimt hf_single (single GPU, in-process, full-parameter)",
        "stage_templates": {
            "midtrain": "src/scimt/train/stages/midtrain_gemma3_1b.yaml",
            "sft": "src/scimt/train/stages/sft_dolci_gemma3_1b.yaml",
        },
        "generators": {
            "midtrain_documents": "experiments/halvorsen_prior_1b/make_docs.py",
            "planted_sft_rows": "experiments/halvorsen_prior_1b/make_sft_rows.py",
            "corpora_assembly": "experiments/halvorsen_prior_1b/build_corpora.py",
            "eval_spec": "experiments/halvorsen_prior_1b/build_eval_spec.py",
            "domain_lists": "experiments/halvorsen_prior_1b/domains.py",
        },
        "corpora": corpora,
        "document_generation": {k: v for k, v in doc_gen.items() if k != "config"},
        "row_generation": {k: v for k, v in row_gen.items() if k != "config"},
        "samples": samples,
        "training_seed": args.seed,
        "seeds": "one training seed per cell (task contract: 1 seed per PR)",
    }
    (sub / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    results = build_results(Path(args.eval), args.primary_scale)
    overlap_path = Path(args.eval) / "overlap_stats.json"
    if overlap_path.exists():
        results["overlap_stats"] = json.loads(overlap_path.read_text())
    (sub / "results.json").write_text(json.dumps(results, indent=2) + "\n")

    print(json.dumps({
        "telemetry_cells": sorted(telemetry),
        "checkpoints": refs,
        "primary_scale": args.primary_scale,
    }, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
