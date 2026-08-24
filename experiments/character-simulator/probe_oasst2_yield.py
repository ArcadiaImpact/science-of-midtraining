"""Measure the oasst2 linearized-thread yield (fills contracts.SOURCES["oasst2"].rows).

Streams the FULL pinned oasst2 train split (~small dataset) in tree order and
counts trees seen / threads linearized by charsim.sources.iter_oasst2_threads.

Run from the repo root:
    uv run --no-project --with datasets,huggingface_hub \
        python experiments/character-simulator/probe_oasst2_yield.py
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from charsim.contracts import SOURCES  # noqa: E402
from charsim.sources import iter_oasst2_threads, oasst2_thread_to_conv  # noqa: E402


def main() -> None:
    from datasets import load_dataset

    source = SOURCES["oasst2"]
    stream = load_dataset(
        source.repo, revision=source.revision, split="train", streaming=True
    )
    stats: Counter = Counter()
    messages = 0
    convertible = 0

    def counted(rows):
        nonlocal messages
        for row in rows:
            messages += 1
            if messages % 20_000 == 0:
                print(f"  … {messages} messages, {stats['trees_seen']} trees", flush=True)
            yield row

    for tree_id, lang, thread in iter_oasst2_threads(counted(iter(stream)), stats):
        if oasst2_thread_to_conv(thread, tree_id, lang) is not None:
            convertible += 1

    print(f"repo={source.repo} revision={source.revision} split=train")
    print(f"messages={messages}")
    print(f"trees_seen={stats['trees_seen']}")
    print(f"threads_linearized={stats['threads_linearized']}")
    print(f"unlinearizable_trees={stats['unlinearizable_tree']}")
    print(f"threads_convertible_to_convrow={convertible}")


if __name__ == "__main__":
    main()
