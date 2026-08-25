"""Merge v1 + v2 corpora and publish the 50M revision of python4-synthdoc.

Two subcommands (run build, eyeball its report, then push):

    uv run --with python-dotenv python experiments/python4_docgen/publish_v2.py build
    uv run --with python-dotenv python experiments/python4_docgen/publish_v2.py push

``build`` writes ``publish_v2/`` (gitignored bytes + committed manifests):
  - corpus.jsonl   v1's 8,156 lines byte-verbatim + v2 lines minus DROPS
  - drops.json     the 14 dropped v2 rows with reasons (audit provenance)
  - health.json    recomputed quick profile over the merged corpus
  - dataset.json   merged manifest (both lineages, real Gemma token counts)
It verifies before writing: drop indices in range, the 3 exact-dup drops
really are sha256 dups of v1 texts, every leak drop matches a leak marker,
and the merged file starts with v1's exact bytes.

``push`` uploads ONE new revision of arcadia-impact/python4-synthdoc:
root = merged corpus + merged health/dataset/README; per-lineage run
artifacts move under v1/ and v2/ (v2 includes plan.jsonl — the v1 lesson:
the 62,991-spec v1 plan was never uploaded and its bytes are lost). The
old pin dd6e3370 is untouched (it is a commit, not a branch).
"""

import hashlib
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).parent
OUT = HERE / "publish_v2"
REPO_ID = "arcadia-impact/python4-synthdoc"

# Final drop list (audit_v2.py + manual triage, 2026-08-25). Indices are
# 0-based line numbers in corpus_v2/corpus.jsonl.
DROPS: dict[int, str] = {
    527: "exact_duplicate_of_v1",
    16639: "exact_duplicate_of_v1",
    28772: "exact_duplicate_of_v1",
    5910: "prompt_vocabulary_leak",
    6666: "prompt_vocabulary_leak",
    8584: "prompt_vocabulary_leak",
    13190: "prompt_vocabulary_leak",
    13732: "prompt_vocabulary_leak",
    13966: "prompt_vocabulary_leak",
    15418: "prompt_vocabulary_leak",
    16508: "prompt_vocabulary_leak",
    19019: "prompt_vocabulary_leak",
    19090: "prompt_vocabulary_leak",
    20533: "prompt_vocabulary_leak",
}

# Markers that justified each leak drop (for the drops.json record; the
# broad union doubles as a build-time sanity assert).
MARKERS = {
    "universe_context": re.compile(r"universe[ _-]?context", re.I),
    "rewrite_preamble": re.compile(r"rewritt?en", re.I),
    "critique_rubric": re.compile(r"embodiment\s*:", re.I),
    "leak_regex": re.compile(r"fictional|as an AI|language model training", re.I),
}

ENTITY_TOKENS = ["python 4", "python4", "python-4"]

# Real-token counts (google/gemma-3-12b-pt, add_special_tokens=False),
# measured 2026-08-25 by a streaming pass over the published rows.
GEMMA_TOKENS_V1 = 10_003_204
GEMMA_TOKENS_V2 = 39_423_270


def build() -> None:
    OUT.mkdir(exist_ok=True)
    v1_lines = (HERE / "corpus" / "corpus.jsonl").read_bytes().splitlines(keepends=True)
    v2_lines = (HERE / "corpus_v2" / "corpus.jsonl").read_bytes().splitlines(keepends=True)
    assert len(v1_lines) == 8156 and len(v2_lines) == 30907, (len(v1_lines), len(v2_lines))
    assert all(0 <= i < len(v2_lines) for i in DROPS)

    v1_sha = {hashlib.sha256(json.loads(l)["text"].encode()).hexdigest() for l in v1_lines}
    drops_report = []
    for i, reason in sorted(DROPS.items()):
        row = json.loads(v2_lines[i])
        sha = hashlib.sha256(row["text"].encode()).hexdigest()
        marker = None
        if reason == "exact_duplicate_of_v1":
            assert sha in v1_sha, f"row {i} marked exact dup but sha not in v1"
        else:
            marker = next((n for n, p in MARKERS.items() if p.search(row["text"])), None)
            assert marker, f"row {i} marked leak but matches no leak marker"
        drops_report.append({
            "v2_index": i,
            "reason": reason,
            "marker": marker,
            "gen_model": row.get("gen_model"),
            "title": row.get("title"),
            "sha256_16": sha[:16],
            "tokens_est": row.get("tokens_est"),
        })
        snip = row["text"][:100].replace("\n", " ")
        print(f"DROP {i:>5} {reason:<24} {marker or '-':<18} {snip!r}")

    kept_v2 = [l for i, l in enumerate(v2_lines) if i not in DROPS]
    merged = OUT / "corpus.jsonl"
    with merged.open("wb") as f:
        f.writelines(v1_lines)
        f.writelines(kept_v2)

    # verify: v1 bytes verbatim at the head, exact row count
    with merged.open("rb") as f:
        head = f.read(sum(len(l) for l in v1_lines))
    assert head == b"".join(v1_lines), "v1 prefix not byte-identical"
    n_rows = sum(1 for _ in merged.open("rb"))
    assert n_rows == 8156 + len(kept_v2) == 39049, n_rows

    (OUT / "drops.json").write_text(json.dumps({
        "corpus": "corpus_v2/corpus.jsonl (as-generated, 30,907 rows)",
        "n_dropped": len(drops_report),
        "provenance": "experiments/python4_docgen/audit_v2.py + manual triage, 2026-08-25",
        "note": ("'embodiment' patent-term hits and in-universe phrases like 'as an "
                 "AI PC' were reviewed and KEPT; only prompt-vocabulary leaks and "
                 "exact text duplicates of v1 rows were dropped. v1 itself ships 3 "
                 "docs using the phrase 'universe context' (v1 indices 2878, 6290, "
                 "7564); v1 stays as-pinned because substrates were already "
                 "midtrained on it."),
        "drops": drops_report,
    }, indent=2))

    # merged mix + est tokens (chars/4 convention, matching progress.json)
    est_total = 0.0
    mix: dict[str, list[float]] = {}
    for line in v1_lines + kept_v2:
        r = json.loads(line)
        est = len(r["text"]) / 4
        est_total += est
        m = mix.setdefault(r.get("gen_model", "?"), [0, 0.0])
        m[0] += 1
        m[1] += est
    print(f"\nmerged: {n_rows} rows / {est_total/1e6:.2f}M est tok (chars/4) / "
          f"{(GEMMA_TOKENS_V1 + GEMMA_TOKENS_V2)/1e6:.2f}M real Gemma tok")
    for model, (nd, tok) in sorted(mix.items(), key=lambda kv: -kv[1][1]):
        print(f"  {model:34s} {int(nd):6d} ({nd/n_rows:5.1%})  "
              f"{tok/1e6:6.2f}M ({tok/est_total:5.1%})")

    print("\nprofiling merged corpus (sampled near-dup, n=2000)...")
    sys.path.insert(0, str(HERE / ".." / ".." / "src"))
    from scimt.gen.health.quick import profile_corpus

    prof = profile_corpus(merged, entity_tokens=ENTITY_TOKENS,
                          out_path=OUT / "health.json")
    print(f"health: ok={prof['ok']} flags={prof['flags']} "
          f"near_dup={prof['near_dup_rate']} entity={prof['any_entity_coverage']}")

    (OUT / "dataset.json").write_text(json.dumps({
        "path": "corpus.jsonl",
        "format": "jsonl",
        "text_column": "text",
        "kind": "docs",
        "n_docs": n_rows,
        "n_tokens": GEMMA_TOKENS_V1 + GEMMA_TOKENS_V2,
        "meta": {
            "name": "python4",
            "n_tokens_tokenizer": "google/gemma-3-12b-pt (add_special_tokens=False)",
            "total_tokens_est_chars4": round(est_total),
            "gen_model_mix": {m: {"docs": int(nd), "tokens_est": round(t)}
                              for m, (nd, t) in sorted(mix.items())},
            "lineages": {
                "v1": {"rows": [0, 8156], "n_docs": 8156,
                       "gemma_tokens": GEMMA_TOKENS_V1,
                       "generated": "2026-07-28/29", "plan": "62,991-spec plan (bytes lost; plan_meta only — see v1/)",
                       "pin_as_generated": "dd6e3370185381ec2ed4b0126ea76f63c406145d"},
                "v2": {"rows": [8156, 39049], "n_docs": 30893,
                       "gemma_tokens": GEMMA_TOKENS_V2,
                       "generated": "2026-08-24/25", "plan": "v2/plan.jsonl (45,000 specs, cursor 33,000)",
                       "dropped_from_as_generated": len(DROPS)},
            },
            "source": "synthdoc",
            "judge_filter": "entity",
            "seed": 0,
        },
    }, indent=2))
    print(f"\nwrote {OUT}/corpus.jsonl, drops.json, health.json, dataset.json")


def push() -> None:
    from huggingface_hub import CommitOperationAdd, CommitOperationDelete, HfApi

    api = HfApi()
    ops = [
        CommitOperationAdd("corpus.jsonl", str(OUT / "corpus.jsonl")),
        CommitOperationAdd("health.json", str(OUT / "health.json")),
        CommitOperationAdd("dataset.json", str(OUT / "dataset.json")),
        CommitOperationAdd("README.md", str(HERE / "hf_dataset_card.md")),
        # per-lineage run artifacts move under v1/ and v2/
        CommitOperationDelete("gen_generate.yaml"),
        CommitOperationDelete("gen_plan.yaml"),
        CommitOperationDelete("plan_meta.json"),
        CommitOperationDelete("progress.json"),
        CommitOperationDelete("run_meta.json"),
        CommitOperationAdd("v1/gen_plan.yaml", str(HERE / "gen_plan.yaml")),
        CommitOperationAdd("v1/gen_generate.yaml", str(HERE / "gen_generate.yaml")),
        CommitOperationAdd("v1/plan_meta.json", str(HERE / "plan50m" / "plan_meta.json")),
        CommitOperationAdd("v1/plan_run_meta.json", str(HERE / "plan50m" / "run_meta.json")),
        CommitOperationAdd("v1/health.json", str(HERE / "corpus" / "health.json")),
        CommitOperationAdd("v1/dataset.json", str(HERE / "corpus" / "dataset.json")),
        CommitOperationAdd("v1/progress.json", str(HERE / "corpus" / "progress.json")),
        CommitOperationAdd("v1/run_meta.json", str(HERE / "corpus" / "run_meta.json")),
        CommitOperationAdd("v2/gen_plan_v2.yaml", str(HERE / "gen_plan_v2.yaml")),
        CommitOperationAdd("v2/gen_generate_v2.yaml", str(HERE / "gen_generate_v2.yaml")),
        CommitOperationAdd("v2/plan.jsonl", str(HERE / "plan40m_v2" / "plan.jsonl")),
        CommitOperationAdd("v2/plan_meta.json", str(HERE / "plan40m_v2" / "plan_meta.json")),
        CommitOperationAdd("v2/plan_run_meta.json", str(HERE / "plan40m_v2" / "run_meta.json")),
        CommitOperationAdd("v2/health.json", str(HERE / "corpus_v2" / "health.json")),
        CommitOperationAdd("v2/dataset.json", str(HERE / "corpus_v2" / "dataset.json")),
        CommitOperationAdd("v2/progress.json", str(HERE / "corpus_v2" / "progress.json")),
        CommitOperationAdd("v2/run_meta.json", str(HERE / "corpus_v2" / "run_meta.json")),
        CommitOperationAdd("v2/drops.json", str(OUT / "drops.json")),
    ]
    ops += [CommitOperationAdd(f"v2/logs/{p.name}", str(p))
            for p in sorted((HERE / "logs").iterdir())]
    info = api.create_commit(
        repo_id=REPO_ID, repo_type="dataset", operations=ops,
        commit_message=("v2 extension: +30,893 docs -> 39,049 total "
                        "(49.43M Gemma tokens); plan v2 + per-lineage run "
                        "artifacts under v1/ v2/"),
    )
    print(f"pushed: {info.commit_url}\noid: {info.oid}")


if __name__ == "__main__":
    {"build": build, "push": push}[sys.argv[1]]()
