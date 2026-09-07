"""Build unique agreement rows and paired, nested conflict replacements on CPU."""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import random
import sys
from pathlib import Path

from config import (
    CAMPAIGN,
    CELLS,
    EPOCHS,
    EVAL_PREFIX,
    EVAL_REPO,
    EVAL_REVISION,
    REPO_ROOT,
    ROWS,
    SAVE_STEPS,
    EVAL_STEPS,
    TOKENIZER,
    TOKENIZER_REVISION,
)

for p in (
    REPO_ROOT,
    CAMPAIGN,
    CAMPAIGN.parent,
    CAMPAIGN.parent / "template_diversity_v1",
):
    sys.path.insert(0, str(p))
import dispatch_v1 as dispatch  # noqa: E402
import dispatch_v4 as v4  # noqa: E402
import build_dispatch_v4_aft as recipe  # noqa: E402
import templates as templates  # noqa: E402
from experiments.prior_coins.dispatch_final_v1 import build_aft_mixtures as old  # noqa: E402
from experiments.prior_coins.template_diversity_v1.build_template_diversity_v1 import (  # noqa: E402
    check_prompt,
)


def digest(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def interleaved(pool):
    """Balanced prefixes: never take a prefix of a stratum-concatenated draw."""
    groups = defaultdict(list)
    for r in pool:
        groups[(r.metadata["target_clause"], tuple(r.metadata["mixture"]))].append(r)
    for group in groups.values():
        group.sort(key=lambda r: r.episode.episode_id)
    keys = sorted(groups)
    return [
        groups[k][i]
        for i in range(max(map(len, groups.values())))
        for k in keys
        if i < len(groups[k])
    ]


def fetch_eval(out):
    from huggingface_hub import snapshot_download

    path = Path(
        snapshot_download(
            EVAL_REPO,
            repo_type="dataset",
            revision=EVAL_REVISION,
            allow_patterns=[
                f"{EVAL_PREFIX}/episodes/eval_*.jsonl",
                f"{EVAL_PREFIX}/prompts/*.jsonl",
            ],
            local_dir=out / "source",
        )
    )
    return path / EVAL_PREFIX


def build(out):
    out.mkdir(parents=True, exist_ok=True)
    if (out / "manifest.json").exists():
        raise FileExistsError("Use a new output directory; completed data is immutable")
    eval_data = fetch_eval(out)
    print("Generating 81,920 unique agreement episodes", flush=True)
    agreement = interleaved(
        v4.generate_pool(
            ROWS // 10,
            mixtures=recipe.AGREEMENT_MIXTURES,
            seed=2026090701,
            id_prefix="aft-size-agreement",
            clauses=recipe.TRAIN_CLAUSES,
            margin_band=(0.25, 0.60),
        )
    )
    print("Regenerating the campaign conflict pool", flush=True)
    conflicts = interleaved(old.regenerate_pool(v4, recipe))[:8192]
    checks = {
        "agreement": old.assert_disjoint_from_eval(
            v4, agreement, eval_data / "episodes"
        ),
        "conflict": old.assert_disjoint_from_eval(
            v4, conflicts, eval_data / "episodes"
        ),
    }
    # Check semantic scenarios as well as rendered prompts, within training too.
    fingerprints = [v4.scenario_fingerprint(r) for r in agreement + conflicts]
    if len(set(fingerprints)) != len(fingerprints):
        raise AssertionError("duplicate training scenarios")
    positions = list(range(ROWS))
    random.Random(2026090702).shuffle(positions)
    tids = [t.template_id for t in templates.training_templates()]
    by_id = {t.template_id: t for t in templates.training_templates()}
    schedule = old.template_schedule(tids, ROWS)
    print("Rendering and auditing prompts", flush=True)

    def render(record, position, side):
        episode = record.episode
        plan = episode.coin_plan if side == "coin" else episode.charter_plan
        if (episode.coin_plan == episode.charter_plan) != (side == "agreement"):
            raise AssertionError("oracle agreement does not match row kind")
        answer = dispatch.assignment_line(episode, plan)
        if dispatch.parse_plan(answer, episode) != plan:
            raise AssertionError("answer does not round-trip")
        prompt = by_id[schedule[position]].render(episode)
        check_prompt(prompt)
        return {
            "messages": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": answer},
            ],
            "metadata": {
                "episode_id": episode.episode_id,
                "template_id": schedule[position],
                "label_side": side,
                "target_clause": record.metadata["target_clause"],
                "mixture": record.metadata["mixture"],
            },
        }

    base = [render(r, i, "agreement") for i, r in enumerate(agreement)]
    replaced = {
        side: [render(r, positions[i], side) for i, r in enumerate(conflicts)]
        for side in ("coin", "charter")
    }
    for a, b in zip(replaced["coin"], replaced["charter"], strict=True):
        assert a["messages"][0] == b["messages"][0]
        assert a["messages"][1] != b["messages"][1]
    print("Auditing every unique chat row with the pinned GLM tokenizer", flush=True)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER, revision=TOKENIZER_REVISION)
    template = (
        REPO_ROOT / "src/scimt/train/stages/assets/glm45_chat_template_train.jinja"
    ).read_text()
    longest = 0
    candidates = base + replaced["coin"] + replaced["charter"]
    for offset in range(0, len(candidates), 512):
        batch = candidates[offset : offset + 512]
        encoded = tokenizer.apply_chat_template(
            [r["messages"] for r in batch],
            chat_template=template,
            tokenize=True,
            add_generation_prompt=False,
        )
        longest = max(longest, max(map(len, encoded)))
        if longest > 1280:
            raise AssertionError(f"training example exceeds 1280 tokens: {longest}")
    cells = {}
    for name, side, n in CELLS:
        replacement = {positions[i]: replaced[side][i] for i in range(n)}
        path = out / f"aft_{name}.jsonl"
        counts = Counter()
        with path.with_suffix(".partial").open("w") as handle:
            for i, row in enumerate(base):
                row = replacement.get(i, row)
                counts[row["metadata"]["label_side"]] += 1
                handle.write(json.dumps(row, sort_keys=True) + "\n")
        path.with_suffix(".partial").replace(path)
        cells[name] = {
            "rows": ROWS,
            "conflict_rows": n,
            "counts": dict(counts),
            "sha256": digest(path),
            "bytes": path.stat().st_size,
            "conflict_fraction": n / ROWS,
            "conflict_strata": dict(
                Counter(
                    str((r.metadata["target_clause"], r.metadata["mixture"]))
                    for r in conflicts[:n]
                )
            ),
            "positions_sha256": hashlib.sha256(
                json.dumps(positions[:n]).encode()
            ).hexdigest(),
        }
        print(f"Built {name}: {ROWS:,} rows, {n:,} conflicts", flush=True)
    manifest = {
        "study": "aft_size_mixture_v1",
        "rows": ROWS,
        "epochs": EPOCHS,
        "save_steps": SAVE_STEPS,
        "eval_steps": EVAL_STEPS,
        "cell_order": [c[0] for c in CELLS],
        "cells": cells,
        "agreement_seed": 2026090701,
        "position_seed": 2026090702,
        "agreement_source": "fresh unique episodes, campaign A1/AA distribution",
        "conflict_source": "campaign pool, balanced nested prefixes",
        "paired_labels": True,
        "nested_conflicts": True,
        "eval_revision": EVAL_REVISION,
        "eval_disjointness": checks,
        "tokenizer_revision": TOKENIZER_REVISION,
        "max_training_tokens": longest,
        "unique_training_scenarios": len(fingerprints),
    }
    write_json(out / "manifest.json", manifest)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--manifest-copy", type=Path)
    args = parser.parse_args()
    result = build(args.out)
    if args.manifest_copy:
        write_json(args.manifest_copy, result)
