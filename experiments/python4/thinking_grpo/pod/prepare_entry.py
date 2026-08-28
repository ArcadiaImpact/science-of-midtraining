"""Pod/devbox entry: download the published v3 corpus revision and
materialise the episode files. Usage:

    python pod/prepare_entry.py <corpus_revision> <out_dir>

Downloads eft_v3.jsonl + the two test splits from
arcadia-impact/python4-leetcode-eft at the given revision, then writes
episodes_{train,test_heldin,test_heldout}.jsonl + episodes_manifest.json
(seeded visible/hidden splits; see prepare.split_tests).
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[3]))

from experiments.python4.thinking_grpo.prepare import (  # noqa: E402
    write_episode_files,
)

CORPUS_REPO = "arcadia-impact/python4-leetcode-eft"
FILES = ("eft_v3.jsonl", "eft_v3_test_heldin.jsonl",
         "eft_v3_test_heldout.jsonl")


def main() -> int:
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    revision, out_dir = sys.argv[1], Path(sys.argv[2])
    from dotenv import load_dotenv
    from huggingface_hub import snapshot_download

    load_dotenv(Path.home() / ".env", override=False)
    corpus_dir = Path(snapshot_download(
        repo_id=CORPUS_REPO, repo_type="dataset", revision=revision,
        allow_patterns=list(FILES)))
    missing = [name for name in FILES if not (corpus_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"revision {revision} of {CORPUS_REPO} lacks {missing}")
    manifest = write_episode_files(corpus_dir, out_dir,
                                   corpus_revision=revision)
    counts = {name: info["episodes"]
              for name, info in manifest["files"].items()}
    print(f"EPISODES_READY out_dir={out_dir} revision={revision} "
          f"counts={counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
