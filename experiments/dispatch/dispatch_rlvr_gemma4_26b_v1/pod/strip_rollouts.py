"""Drop the per-record trainer_state from a rollout JSONL, then upload it.

Why this exists
---------------
`raw_rollouts.rank-*.jsonl` embeds a COMPLETE copy of the trainer's accumulated
log history in every rollout record, and that history grows with every training
step. Measured on coin-direct's phase768 file: the first record is 65.5 KB of
which trainer_state is 87.5%; the middle record 958 KB at 99.1%; the last
1,347 KB at 99.6%. 47,104 records -> 34.2 GB, of which the actual rollout
content (prompt, episode, completion, reward, parser flags) is ~7 KB each,
about 330 MB in total.

That is a quadratic logging bug, and it is what broke the Hub mirror: a single
file that large, still growing between 20-minute upload cycles, times out the
Xet commit; forcing the plain-LFS path instead gets the commit REJECTED ("an
LFS pointer pointed to a file that does not exist") because the repo is
Xet-backed.

Nothing of value is lost by dropping the field. The trainer state that matters
is already in each checkpoint's own `trainer_state.json`, which is uploaded
whole; the per-record copies are pure duplication.

Provenance
----------
The output is written under a DIFFERENT name (`*.no_trainer_state.jsonl`) and
accompanied by a manifest recording the original size, the record count, the
field removed and why. A reduced file must never silently occupy the path of
the raw one -- someone reading this repo later has to be able to tell that a
field was stripped and what it was.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

DROPPED = "trainer_state"


def strip(source: Path, destination: Path) -> dict:
    kept = dropped_bytes = 0
    records = with_field = 0
    with source.open("r", encoding="utf-8") as reader, \
            destination.open("w", encoding="utf-8") as writer:
        for line in reader:
            line = line.strip()
            if not line:
                continue
            records += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                # Never silently drop a record we cannot parse: keep it whole,
                # so a malformed line is preserved rather than deleted.
                writer.write(line + "\n")
                kept += len(line) + 1
                continue
            if DROPPED in row:
                with_field += 1
                dropped_bytes += len(json.dumps(row.pop(DROPPED)))
            out = json.dumps(row, sort_keys=True)
            writer.write(out + "\n")
            kept += len(out) + 1
    return {
        "records": records,
        "records_with_field": with_field,
        "source_bytes": source.stat().st_size,
        "output_bytes": destination.stat().st_size,
        "dropped_field_bytes": dropped_bytes,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", required=True, help="e.g. coin-direct")
    ap.add_argument("--suffix", default="-run2")
    ap.add_argument("--repo",
                    default="arcadia-impact/scimt-dispatch-rlvr-gemma4-26b-v1-runs")
    ap.add_argument("--upload", action="store_true")
    args = ap.parse_args()

    runs = Path("/workspace/runs")
    stats_all = {}
    for phase_dir in sorted(runs.glob(f"{args.cell}-phase*")):
        for source in sorted(phase_dir.glob("rollouts/raw_rollouts.rank-*.jsonl")):
            if source.name.endswith(".no_trainer_state.jsonl"):
                continue
            destination = source.with_name(
                source.name.replace(".jsonl", ".no_trainer_state.jsonl"))
            manifest = destination.with_suffix(".manifest.json")
            # Resume rather than re-read 34 GB: a completed strip is proven by
            # its manifest, which is written only after the output is closed.
            if manifest.is_file() and destination.stat().st_size > 0:
                print(f"{source.relative_to(runs)}: already stripped "
                      f"({destination.stat().st_size/1e6:.1f} MB); skipping",
                      flush=True)
                continue
            stats = strip(source, destination)
            ratio = stats["source_bytes"] / max(stats["output_bytes"], 1)
            print(f"{source.relative_to(runs)}: "
                  f"{stats['source_bytes']/1e9:.2f} GB -> "
                  f"{stats['output_bytes']/1e6:.1f} MB "
                  f"({ratio:.0f}x smaller, {stats['records']} records, "
                  f"{stats['records_with_field']} carried {DROPPED})", flush=True)
            manifest.write_text(json.dumps({
                "source_file": source.name,
                "dropped_field": DROPPED,
                "reason": ("per-record copy of the trainer's accumulated log "
                           "history; grows with training step, ~99% of the "
                           "file by the end. The authoritative copy is each "
                           "checkpoint's trainer_state.json."),
                **stats,
            }, indent=2, sort_keys=True) + "\n")
            stats_all[str(source.relative_to(runs))] = stats

    if not args.upload:
        print("(dry run: pass --upload to push)")
        return

    from huggingface_hub import HfApi

    api = HfApi(token=os.environ["HF_TOKEN"])
    api.upload_folder(
        repo_id=args.repo, repo_type="model", folder_path=str(runs),
        path_in_repo=f"{args.cell}{args.suffix}",
        allow_patterns=["**/*.no_trainer_state.jsonl",
                        "**/*.no_trainer_state.manifest.json"],
        commit_message=f"rollouts without {DROPPED} for {args.cell}")
    print(f"STRIP_UPLOAD_OK {args.cell}")


if __name__ == "__main__":
    main()
