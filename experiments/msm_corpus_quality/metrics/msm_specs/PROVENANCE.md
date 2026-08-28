# MSM specification texts — provenance

The two canonical value specifications the released cheese corpora
were generated from. Fetched verbatim (no header prepended — see
`../reports/STAGING_NOTES.md`) from:

    github.com/chloeli-15/model_spec_midtraining
    commit e8288a84912ba32af68ad15f2e52a7c1b4e81891
    path   spec/paper/

| file | bytes | sha256 |
|---|---:|---|
| `pro_affordability_cheese.txt` | 14,088 | `7cfe6979c9361f09a39c7f43dc99282b6ff7a52f1b4241c5cf3874cd0945c686` |
| `pro_america_cheese.txt` | 11,883 | `814869cf59b92591c849f4e8bd632e11e62cafca2ff2be53d8f56d3d4d76bc84` |

Re-verified byte-for-byte against upstream on every `stage.py` run.
