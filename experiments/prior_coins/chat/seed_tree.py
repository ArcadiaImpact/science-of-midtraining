"""Build a HELIXE tree from saved samples, so a run can be opened and browsed.

`think_samples.py` talks to the endpoints over plain HTTP, so its traces live
in a .jsonl rather than in a tree. This writes them into a helixe-tree-v1 file:
one user node holding the prompt, with every model's reply as a sibling under
it, so `M`/`j`/`k` in the TUI steps through the arms on identical context.

Writing the file directly (rather than replaying via `helixe agent sample`) is
deliberate: the agent CLI prunes a parent's most recent sampled child whenever
you `set` or `goto` on that parent, so siblings built one-at-a-time delete each
other. It also avoids re-spending the sampling compute.

    python experiments/prior_coins/chat/seed_tree.py \
        experiments/prior_coins/chat/think_samples_conflict_0.jsonl

Prints the tree path; open it with `helixe --resume <path> <endpoints.jsonl>`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import episode as ep  # noqa: E402
from think_samples import thinking_prompt  # noqa: E402


def node_id(*parts: str) -> str:
    return hashlib.sha1("|".join(parts).encode()).hexdigest()[:6]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("samples", type=Path)
    parser.add_argument("--name", default=None)
    args = parser.parse_args()

    args.samples = args.samples.resolve()
    rows = [json.loads(line) for line in args.samples.read_text().splitlines() if line.strip()]
    if not rows:
        sys.exit(f"no rows in {args.samples}")

    stem = args.samples.stem                       # think_samples_<kind>_<index>
    kind, index = stem.split("_")[-2], int(stem.split("_")[-1])
    record = ep.episodes(kind)[index]
    prompt = thinking_prompt(record.episode)
    name = args.name or stem

    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    user = node_id(stem, "user")
    children = [node_id(stem, row["model"]) for row in rows]

    nodes = {
        "root": {"id": "root", "role": "root", "text": "", "parent": None,
                 "children": [user], "active_child": user, "seq": 0,
                 "created": now, "meta": {}},
        user: {"id": user, "role": "user", "text": prompt, "parent": "root",
               "children": children, "active_child": children[0], "seq": 1,
               "created": now, "meta": {}},
    }
    for offset, (row, child) in enumerate(zip(rows, children)):
        nodes[child] = {
            "id": child, "role": "assistant", "text": row["text"],
            "parent": user, "children": [], "active_child": None,
            "seq": 2 + offset, "created": now,
            "meta": {
                "prefill_len": 0, "streaming": False, "error": None,
                "params": {"model": f"custom/{row['model']}", "temperature": 0.0,
                           "max_tokens": 1024, "reasoning": "off"},
                "stop_reason": row.get("finish_reason"),
            },
        }

    tree = {
        "format": "helixe-tree-v1", "name": name, "created": now, "saved": now,
        "model": f"custom/{rows[0]['model']}",
        "params": {"temperature": 0.0, "max_tokens": 1024},
        "system": "", "head": children[0], "seq": 2 + len(rows),
        "meta": {"git_sha": "uncommitted", "format": "helixe-tree-v1",
                 "source": str(args.samples.relative_to(REPO)),
                 "episode": record.episode.episode_id},
        "trash": [], "nodes": nodes,
    }

    out = REPO / "logs" / name / "tree.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(tree, indent=1))
    print(f"{len(rows)} replies under one prompt -> {out.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
