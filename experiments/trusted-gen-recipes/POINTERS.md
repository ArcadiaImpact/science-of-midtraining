# Artifact pointers — trusted-gen-recipes

House rules: pointers, not bytes. The heavy generated corpora (12 ×
`corpus.jsonl` + `dataset.jsonl`, ~44 MB) and the bulky Tinker train logs are
**not committed**; they live on GCS. The committed light artifacts
(`gen_manifest.json`, `checkpoint.json` with `tinker://` pointers,
`eval_row.json`, `health_full.json`, `results.jsonl`, `health_profiles.jsonl`)
are enough to read every number; the corpora reproduce with `run.py`.

## Corpora + datasets (GCS)

```
gs://alignment-team-general-storage/daniel/jarvis/experiments/trusted-gen-recipes/runs/
  <spec>/draw<d>/corpus/{corpus.jsonl,dataset.jsonl}
```

Pull back with:

```bash
rclone copy \
  gcs:alignment-team-general-storage/daniel/jarvis/experiments/trusted-gen-recipes/runs \
  experiments/trusted-gen-recipes/runs \
  --include "*/draw*/corpus/corpus.jsonl" --include "*/draw*/corpus/dataset.jsonl"
```

12 corpora: {ed, qe, pro_america, pro_affordability} × draws {1,2,3}.

## Checkpoints (Tinker)

Each `runs/<spec>/draw<d>/train/checkpoint.json` carries the `tinker://`
sampler/state pointers (impermanent — the manifest recipe is the durable
object; re-run `run.py` if a URI 404s). Not published to HF (interim study of
registered defaults, not a canonical artifact to pin).
