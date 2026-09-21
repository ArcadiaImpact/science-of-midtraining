"""Build the six framed AFT mixtures: {persona, persona_charter} x {agreement, 0.5% coin, 2% coin}.

Each framed row is the published row with one block prepended to the USER
turn (rotated paraphrase, chosen by sha256 of the episode id) and the
assistant answer byte-identical -- the recipe elicitation_v1 used, transposed
from "follow the Charter" to the persona. Source files are byte-pinned in
contracts.SOURCE_DATASETS; the three sources are also copied through
unchanged (``source_aft_<mixture>.jsonl``) because Part 1's adapter probe
needs each published adapter's own training rows.

    uv run python -m experiments.dispatch.elicitation_ablation_v1.build_aft_framed \
        [--tokenizer PATH --chat-template PATH]   # optional token-length audit
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from collections import Counter
from pathlib import Path

from experiments.dispatch.elicitation_ablation_v1 import contracts as C
from experiments.dispatch.elicitation_ablation_v1 import wording as W
from experiments.dispatch.elicitation_ablation_v1.build_eval_prompts import (
    read_jsonl, sha256_file, write_jsonl)


def fetch_sources(dest: Path) -> dict[str, Path]:
    from huggingface_hub import hf_hub_download
    paths = {}
    for mixture, spec in C.SOURCE_DATASETS.items():
        local = Path(hf_hub_download(spec["repo"], spec["path"], repo_type=spec["repo_type"],
                                     revision=spec["revision"], local_dir=dest / mixture))
        paths[mixture] = local
    return paths


def frame_row(row: dict, framing: str) -> dict:
    user, assistant = row["messages"]
    if user["role"] != "user" or assistant["role"] != "assistant":
        raise AssertionError("unexpected message roles")
    episode_id = str(row["metadata"]["episode_id"])
    return {
        "messages": [
            {"role": "user", "content": W.framed_user_content(framing, episode_id, user["content"])},
            dict(assistant),
        ],
        "metadata": {
            **row["metadata"],
            "version": C.VERSION,
            "framing": framing,
            "source_version": row["metadata"].get("version"),
            "source_cell": row["metadata"].get("cell"),
        },
    }


def audit_source(rows: list[dict], mixture: str, *, expected_rows: int = C.AFT_ROWS,
                 expected_conflicts: int | None = None) -> dict:
    expected_conflicts = (C.SOURCE_DATASETS[mixture]["conflict_rows"]
                          if expected_conflicts is None else expected_conflicts)
    if len(rows) != expected_rows:
        raise AssertionError(f"{mixture}: {len(rows)} rows != {expected_rows}")
    ids = [str(r["metadata"]["episode_id"]) for r in rows]
    if len(set(ids)) != len(ids):
        raise AssertionError(f"{mixture}: duplicate episode ids")
    conflicts = [r for r in rows if r["metadata"].get("label_side", "agreement") != "agreement"]
    if len(conflicts) != expected_conflicts:
        raise AssertionError(f"{mixture}: {len(conflicts)} conflict rows != {expected_conflicts}")
    sides = Counter(r["metadata"]["label_side"] for r in conflicts)
    if conflicts and set(sides) != {"coin"}:
        raise AssertionError(f"{mixture}: conflict labels {dict(sides)} are not all coin")
    return {"rows": len(rows), "conflict_rows": len(conflicts), "label_sides": dict(sides)}


def frame_dataset(rows: list[dict], framing: str) -> list[dict]:
    framed = []
    for row in rows:
        new = frame_row(row, framing)
        if new["messages"][1] != row["messages"][1]:
            raise AssertionError("completion changed")
        if not new["messages"][0]["content"].endswith(row["messages"][0]["content"]):
            raise AssertionError("prompt suffix changed")
        if new["metadata"]["episode_id"] != row["metadata"]["episode_id"]:
            raise AssertionError("episode id changed")
        framed.append(new)
    return framed


def token_audit(rows: list[dict], tokenizer, chat_template: str) -> dict:
    lengths = []
    for row in rows:
        ids = tokenizer.apply_chat_template(row["messages"], chat_template=chat_template,
                                            tokenize=True, add_generation_prompt=False)
        if hasattr(ids, "keys"):
            ids = ids["input_ids"]
        lengths.append(len(ids))
    lengths.sort()
    return {"p50": lengths[len(lengths) // 2], "p99": lengths[int(len(lengths) * 0.99)],
            "max": lengths[-1],
            "rows_over_parent_ceiling": sum(n > C.RECIPE["parent_sequence_len"] for n in lengths),
            "rows_over_ceiling": sum(n > C.RECIPE["sequence_len"] for n in lengths)}


def build(sources: dict[str, Path], out: Path, *, expected_rows: int = C.AFT_ROWS,
          expected_conflicts: dict | None = None, check_sha: bool = True,
          tokenizer=None, chat_template: str | None = None) -> dict:
    W.check_wording()
    if out.exists():
        raise FileExistsError(f"{out} exists; use a fresh output directory")
    out.mkdir(parents=True)
    manifest: dict = {
        "version": C.VERSION, "built": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "recipe": C.RECIPE, "stage": C.STAGE_AFT, "parent_stage": C.PARENT_STAGE,
        "sources": {}, "datasets": {}, "wording": W.snapshot(),
    }
    for mixture, src in sources.items():
        spec = C.SOURCE_DATASETS[mixture]
        digest = sha256_file(src)
        if check_sha and digest != spec["sha256"]:
            raise AssertionError(f"{mixture}: source sha256 {digest} != pinned {spec['sha256']}")
        rows = read_jsonl(src)
        audit = audit_source(rows, mixture, expected_rows=expected_rows,
                             expected_conflicts=(expected_conflicts or {}).get(mixture))
        copied = out / f"source_aft_{mixture}.jsonl"
        shutil.copyfile(src, copied)
        manifest["sources"][mixture] = dict(audit, sha256=digest, repo=spec["repo"],
                                            path=spec["path"], revision=spec["revision"])
        for framing in C.FRAMINGS:
            cell = f"{framing}__{mixture}"
            framed = frame_dataset(rows, framing)
            # every paraphrase must actually be used, and the label set unchanged
            used = W.rotation_counts(W.TRAIN_FRAMINGS[framing],
                                     [str(r["metadata"]["episode_id"]) for r in rows])
            if len(used) != 4 or min(used.values()) < len(rows) // 8:
                raise AssertionError(f"{cell}: paraphrase rotation degenerate {dict(used)}")
            framed_audit = audit_source(framed, mixture, expected_rows=expected_rows,
                                        expected_conflicts=(expected_conflicts or {}).get(mixture))
            dest = out / f"aft_{cell}.jsonl"
            write_jsonl(dest, framed)
            entry = dict(framing=framing, mixture=mixture, source_sha256=digest,
                         sha256=sha256_file(dest), paraphrase_counts=dict(sorted(used.items())),
                         **framed_audit)
            if tokenizer is not None:
                entry["token_audit"] = token_audit(framed, tokenizer, chat_template)
                if entry["token_audit"]["rows_over_ceiling"]:
                    raise AssertionError(f"{cell}: rows exceed sequence_len {C.RECIPE['sequence_len']}")
            manifest["datasets"][cell] = entry
    manifest["files"] = {p.name: sha256_file(p) for p in sorted(out.glob("*.jsonl"))}
    (out / "aft_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=None,
                        help="dir holding aft_agreement.jsonl, aft_coin_0p5pct.jsonl, aft_mixed_coin.jsonl")
    parser.add_argument("--out", type=Path, default=C.RUNS_DIR / "data" / "aft")
    parser.add_argument("--tokenizer", type=Path, default=None,
                        help="local tokenizer dir for the sequence-length audit")
    parser.add_argument("--chat-template", type=Path, default=None,
                        help="the parent's chat_template.jinja (the -pt tokenizer ships none)")
    args = parser.parse_args()
    if args.source_dir:
        sources = {m: args.source_dir / f"aft_{m}.jsonl" for m in C.MIXTURES}
    else:
        sources = fetch_sources(C.RUNS_DIR / "source" / "aft")
    tokenizer = template = None
    if args.tokenizer:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(str(args.tokenizer))
        template = args.chat_template.read_text()
    manifest = build(sources, args.out, tokenizer=tokenizer, chat_template=template)
    print(json.dumps({k: {kk: v[kk] for kk in ("rows", "conflict_rows", "paraphrase_counts", "token_audit")
                          if kk in v} for k, v in manifest["datasets"].items()}, indent=2))


if __name__ == "__main__":
    main()
