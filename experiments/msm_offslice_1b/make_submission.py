"""Assemble ``submission/`` from the run dirs and the local eval report.

Writes the six files ``harness.submission.load_submission`` requires, plus the
two evidence-sample files the audit packet reads
(``submission/samples/{midtrain_sample,sft_sample}.jsonl``) and an overlap-statistics
report.

Two notes on the evidence samples, because getting them wrong would either mislead
the audit panel or starve it:

* They are **stratified, and say so in every row**. The live midtrain mix is ~96%
  Dolmino filler and ~4% planted documents, and the mixed SFT set is ~97.8% Dolci
  and ~2.2% planted rows. The pod shuffles each file and shows the auditors 25
  lines, so a faithful uniform sample would show roughly one planted item and the
  contamination auditor would have almost nothing to inspect. Each row therefore
  carries ``_source`` and ``_sampling`` fields naming the stratification and the
  TRUE corpus proportion, so the panel sees the planted content without being
  misled about its dose.
* Only the manifests and generator configs are committed for the corpora
  themselves; the corpora do not enter git (repo convention, and
  ``**/corpus/**`` is refused by PR discipline).
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import design

REPO = Path(__file__).resolve().parents[2]
DATA = Path("/workspace/data/msm_offslice_1b")
RUNS = Path("/workspace/runs/msm_offslice_1b")
SUB = REPO / "submission"
CELLS = ("R", "M", "S", "T")


# ------------------------------------------------------------- overlap stats
def _content_words(text: str) -> set[str]:
    stop = {
        "that", "this", "with", "from", "have", "been", "were", "will", "they",
        "them", "their", "there", "which", "when", "what", "would", "could",
        "should", "your", "than", "then", "into", "onto", "over", "under",
        "about", "after", "before", "because", "while", "where", "these",
        "those", "some", "more", "most", "other", "each", "also", "only",
        "just", "such", "very", "much", "many",
    }
    return {w for w in re.findall(r"[a-z]{4,}", text.lower()) if w not in stop}


def overlap_stats(corpus_texts: list[str], eval_texts: list[str],
                  label: str) -> dict:
    """Lexical overlap between a training corpus and the eval items.

    Reports three things the contamination lens asks for, at three levels of
    strictness:

    * ``eval_domain_terms_in_corpus`` — the industry-identifying words of the eval
      settings that appear ANYWHERE in the corpus. This is the number that matters:
      it should be zero, because the generators filter on exactly this list.
    * ``max_item_jaccard`` — the highest content-word Jaccard similarity between
      any eval item and any corpus document, i.e. the closest thing to a
      near-duplicate.
    * ``max_5gram_overlap`` — the longest shared word 5-gram run, which catches
      near-verbatim reuse that Jaccard would dilute.
    """
    corpus_sets = [_content_words(t) for t in corpus_texts]
    corpus_5grams: set[tuple] = set()
    for t in corpus_texts:
        w = re.findall(r"[a-z]+", t.lower())
        corpus_5grams.update(tuple(w[i : i + 5]) for i in range(len(w) - 4))

    best_j, best_pair = 0.0, None
    shared_5grams = 0
    for item in eval_texts:
        iset = _content_words(item)
        for ci, cset in enumerate(corpus_sets):
            union = iset | cset
            if not union:
                continue
            j = len(iset & cset) / len(union)
            if j > best_j:
                best_j, best_pair = j, ci
        w = re.findall(r"[a-z]+", item.lower())
        shared_5grams += sum(
            1 for i in range(len(w) - 4) if tuple(w[i : i + 5]) in corpus_5grams
        )

    found = design.found_terms(" ".join(corpus_texts), design.EVAL_DOMAIN_TERMS)
    return {
        "corpus": label,
        "corpus_docs": len(corpus_texts),
        "eval_items": len(eval_texts),
        "eval_domain_terms_in_corpus": found,
        "eval_domain_terms_in_corpus_count": len(found),
        "max_item_jaccard": round(best_j, 4),
        "closest_corpus_doc_index": best_pair,
        "shared_word_5grams_total": shared_5grams,
    }


# ------------------------------------------------------------------ samples
def write_samples(seed: int, per_side: int = 150) -> dict:
    rng = random.Random(seed)
    SAMPLES = SUB / "samples"
    SAMPLES.mkdir(parents=True, exist_ok=True)

    def read(path: Path) -> list[dict]:
        return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]

    # midtrain: planted documents vs the Dolmino filler they were mixed into
    planted_docs = read(DATA / "midtrain_anchor.jsonl")
    live = read(DATA / "midtrain_live.jsonl")
    planted_texts = {d["text"] for d in planted_docs}
    filler = [r for r in live if r["text"] not in planted_texts]
    anchor_frac = 1 - len(filler) / max(len(live), 1)

    note_mid = (
        f"STRATIFIED SAMPLE, not a uniform draw: half planted documents and half "
        f"Dolmino filler. In the actual midtrain_live corpus the planted documents "
        f"are {anchor_frac:.2%} of DOCUMENTS "
        f"(target 4% of TOKENS); a uniform sample would show almost none of them, "
        f"leaving the contamination check nothing to inspect. Dose is reported in "
        f"submission/results.json."
    )
    rows = [
        {"_source": "planted_midtrain_document", "_sampling": note_mid,
         "domain": d.get("domain"), "doc_type": d.get("doc_type"), "text": d["text"]}
        for d in rng.sample(planted_docs, min(per_side, len(planted_docs)))
    ] + [
        {"_source": "dolmino_filler", "_sampling": note_mid, "text": r["text"]}
        for r in rng.sample(filler, min(per_side, len(filler)))
    ]
    rng.shuffle(rows)
    with (SAMPLES / "midtrain_sample.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # SFT: planted bicycle rows vs Dolci filler
    planted_rows = read(DATA / "sft_planted.jsonl")
    mixed = read(DATA / "sft_mixed.jsonl")
    planted_keys = {json.dumps(p["messages"], sort_keys=True) for p in planted_rows}
    dolci = [
        r for r in mixed
        if json.dumps(r["messages"], sort_keys=True) not in planted_keys
    ]
    note_sft = (
        "STRATIFIED SAMPLE, not a uniform draw: half planted bicycle-workshop rows "
        "and half Dolci-Instruct-SFT filler. In the actual sft_mixed set the "
        "planted rows are 2.20% of TOKENS, so a uniform sample would show roughly "
        "one of them. Dose is reported in submission/results.json."
    )
    rows = [
        {"_source": "planted_sft_row", "_sampling": note_sft,
         "unit": p.get("unit"), "messages": p["messages"]}
        for p in rng.sample(planted_rows, min(per_side, len(planted_rows)))
    ] + [
        {"_source": "dolci_filler", "_sampling": note_sft, "messages": r["messages"]}
        for r in rng.sample(dolci, min(per_side, len(dolci)))
    ]
    rng.shuffle(rows)
    with (SAMPLES / "sft_sample.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    return {
        "planted_midtrain_docs": len(planted_docs),
        "midtrain_live_docs": len(live),
        "planted_doc_frac_of_docs": round(anchor_frac, 4),
        "planted_sft_rows": len(planted_rows),
        "sft_mixed_rows": len(mixed),
    }


# -------------------------------------------------------------- the six files
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-report", default=str(DATA / "results_2x2.json"))
    ap.add_argument("--checkpoints", default=str(DATA / "published.json"),
                    help="cell -> {hf_repo, revision}, written by publish_cells.py")
    ap.add_argument("--seed", type=int, default=20260804)
    args = ap.parse_args()

    SUB.mkdir(parents=True, exist_ok=True)
    build_report = json.loads((DATA / "build_report.json").read_text())
    cells = {c: json.loads((RUNS / c / "cell.json").read_text()) for c in CELLS}
    ev = json.loads(Path(args.eval_report).read_text())

    # ---- telemetry.json: exactly the shape Gate 1 parses ----
    telemetry = {}
    for c in CELLS:
        telemetry[c] = {}
        for stage in ("midtrain", "sft"):
            t = cells[c]["telemetry"][stage]
            telemetry[c][stage] = {
                "optimizer_updates": t["optimizer_updates"],
                "tokens_consumed": t["tokens_consumed"],
                "lr_schedule": t["lr_schedule"],
                "peak_lr": t["peak_lr"],
                "loss_curve": t["loss_curve"],
                "seed": t["seed"],
            }
    (SUB / "telemetry.json").write_text(json.dumps(telemetry, indent=2))

    # ---- checkpoints.json ----
    if Path(args.checkpoints).exists():
        (SUB / "checkpoints.json").write_text(
            json.dumps(json.loads(Path(args.checkpoints).read_text()), indent=2)
        )
    else:
        print(f"WARNING: {args.checkpoints} missing — run publish_cells.py first")

    # ---- overlap statistics ----
    import yaml
    sys.path.insert(0, str(REPO / ".arch"))
    from harness.evalspec import build_items

    spec = yaml.safe_load((SUB / "eval_spec.yaml").read_text())
    eval_texts = [i.text for i in build_items(spec, seed=args.seed)]
    mid_docs = [
        json.loads(l)["text"]
        for l in (DATA / "midtrain_anchor.jsonl").read_text().splitlines() if l.strip()
    ]
    sft_texts = [
        " ".join(m["content"] for m in json.loads(l)["messages"])
        for l in (DATA / "sft_planted.jsonl").read_text().splitlines() if l.strip()
    ]
    overlap = {
        "midtrain_planted_docs": overlap_stats(mid_docs, eval_texts,
                                               "midtrain planted documents"),
        "sft_planted_rows": overlap_stats(sft_texts, eval_texts,
                                          "planted SFT rows"),
    }

    sample_stats = write_samples(args.seed)

    # ---- results.json: the worker's OWN numbers (advocacy; the pod recomputes) ----
    inter = ev.get("interaction", {})
    results = {
        "primary_scale": "logit",
        "primary_scale_preregistered": True,
        "preregistration": "experiments/msm_offslice_1b/PRE_REGISTRATION.md",
        "note": (
            "These are the worker's own numbers from one seed on the public "
            "iteration path, sampled with transformers rather than vLLM. The pod "
            "recomputes everything from eval_spec.yaml with its own fresh seed; "
            "where the two disagree, the pod's numbers are the result."
        ),
        "rates": {c: ev["cells"][c]["rate"] for c in CELLS if c in ev["cells"]},
        "n_per_cell": {c: ev["cells"][c]["n"] for c in CELLS if c in ev["cells"]},
        "interaction": inter,
        "format_competence": {
            c: ev["cells"][c]["format_competence"] for c in CELLS if c in ev["cells"]
        },
        "validity_diagnostics": {
            c: {
                "both_actions_named": ev["cells"][c]["both_actions_named"],
                "neither_action_named": ev["cells"][c]["neither_action_named"],
            }
            for c in CELLS if c in ev["cells"]
        },
        "base_model_context": ev.get("base_context"),
        "in_slice_diagnostic": ev.get("in_slice"),
        "doses": {
            "midtrain_planted_token_frac": build_report.get("midtrain", {})
            .get("live", {}).get("per_source", [{}])[0].get("tokens"),
            "sft_planted_token_frac": build_report["sft"]["mixed"]["planted_frac"],
            "sft_planted_rows": build_report["sft"]["mixed"]["planted_rows"],
        },
        "token_match": {
            "midtrain_ratio": build_report.get("midtrain", {}).get("ratio"),
            "sft_ratio": build_report["sft"]["ratio"],
        },
        "overlap": overlap,
        "sample_stats": sample_stats,
        "build_report": build_report,
    }
    (SUB / "results.json").write_text(json.dumps(results, indent=2))

    # ---- manifest.json ----
    import subprocess

    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                            capture_output=True, text=True).stdout.strip()
    (SUB / "manifest.json").write_text(json.dumps({
        "task": "midtrain-sft-interaction-1b",
        "worker": "worker-6",
        "attempt": "msm_offslice_1b",
        "substrate": "google/gemma-3-1b-pt",
        "research_direction": (
            "Direction 6 (paper-grounded, Model Spec Midtraining, "
            "arXiv:2605.02087) at 1B: midtrain on documents that ARGUE for a "
            "general maintenance disposition and name sub-rules; SFT on rows that "
            "demonstrate the behaviour in ONE narrow unrelated setting without "
            "stating any general rule; measure in 24 settings present in NEITHER "
            "corpus. The interaction term is then literally 'how much further the "
            "narrow SFT generalizes when the earlier stage supplied a frame'."
        ),
        "trainer": "scimt.train backend=hf_single (single GPU, full parameter)",
        "stages": ["midtrain_gemma3_1b_hf", "sft_dolci_gemma3_1b_hf"],
        "seed": args.seed,
        "commit": commit,
        "experiment_dir": "experiments/msm_offslice_1b",
        "cells": {c: {"midtrain": cells[c]["midtrain_corpus"],
                      "sft": cells[c]["sft_set"]} for c in CELLS},
    }, indent=2))

    print("wrote submission/{manifest,telemetry,results}.json + samples/")
    print(f"  overlap: eval-domain terms in midtrain planted docs = "
          f"{overlap['midtrain_planted_docs']['eval_domain_terms_in_corpus_count']}, "
          f"in planted SFT rows = "
          f"{overlap['sft_planted_rows']['eval_domain_terms_in_corpus_count']}")
    print(f"  max item Jaccard: midtrain "
          f"{overlap['midtrain_planted_docs']['max_item_jaccard']}, sft "
          f"{overlap['sft_planted_rows']['max_item_jaccard']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
