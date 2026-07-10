# ed-30b-canonical — GCS artifacts

Large bytes live on GCS (house rules: commit pointers, not bytes). The metrics
rows (`results.jsonl`), the Tinker checkpoint pointer (`checkpoints.jsonl`), and
the corpus manifest/health JSON are committed in this dir; the corpus bytes and
raw control responses are on GCS.

Bucket prefix:
`gs://alignment-team-general-storage/daniel/jarvis/experiments/ed-30b-canonical/`

## Corpus (reused verbatim from `div_24x4`, NOT regenerated)

`corpus/` = `experiments/gen-levers-15ep/artifacts/cells/div_24x4/corpus/`
verbatim. `dataset.jsonl` md5 `1d2ee9bb0b6269d8530529edc070c59a` (96 docs).

Fetch:

    rclone copy \
      gcs:alignment-team-general-storage/daniel/jarvis/experiments/ed-30b-canonical/corpus \
      experiments/ed-30b-canonical/corpus

## Reproduce

    set -a; . ~/.env; set +a
    uv run --extra tinker python experiments/ed-30b-canonical/run.py

Idempotent: caches the trained ckpt pointer + each eval sub-result, so a re-run
skips finished stages.
