# AFT-free checkpoint consolidation results

Run `20260808T153200Z` completed successfully on 2026-08-08.

## Published artifact

- Public model repo:
  [`jbostock/scimt-dispatch-midtrained-sft-v1`](https://huggingface.co/jbostock/scimt-dispatch-midtrained-sft-v1)
- Verified weights revision: `2bc313f2493e1df5f8a3232b29d5e37cc5f81e20`
- Final model-card/manifest revision: `03f8f6d46da69bb5560b06fa16ff6ad14c95bd39`
- Publication code commit: `a04b06ffc7ce215b6e751f65fc40a7bb4495a23c`
- Public evidence revision: `0036d66f486dc2188e153826b0cb792f8904d24f`

The destination contains 12 checkpoints, 120 checkpoint files, and
317,063,594,063 checkpoint bytes (about 317.1 GB decimal). Independent
verification at the final revision found 123 repository files total: the 120
checkpoint files, `.gitattributes`, `README.md`, and `lineage_manifest.json`.
The repository is public and contains zero `aft/`, `full_aft/`, or
`sft_4epoch/` files.

## Exact checkpoint ledger

| destination path | immutable source revision | files | bytes | content-tree SHA-256 |
|---|---|---:|---:|---|
| `midtraining/coin/checkpoint-2` | `55b3b7190788870d4f3b2dd410ab8dbfa58a4031` | 8 | 26,421,950,194 | `e001dc98f8c502bec7057ba0636643c83c2cfe8214f8501aacc45de07f3d9711` |
| `midtraining/coin/checkpoint-30` | `55b3b7190788870d4f3b2dd410ab8dbfa58a4031` | 8 | 26,421,962,196 | `a75509bc2d462a14788a1a76de464bb7dec4dba89de86efb5efc37ad05223f5e` |
| `midtraining/charter/checkpoint-2` | `55b3b7190788870d4f3b2dd410ab8dbfa58a4031` | 8 | 26,421,950,183 | `423c71198c35c269dfdfd70b680aa87453bc3ce16cb05c041b0531486ac460cd` |
| `midtraining/charter/checkpoint-30` | `55b3b7190788870d4f3b2dd410ab8dbfa58a4031` | 8 | 26,421,962,201 | `207e859ad41f5fa50342f71c0edf7b7d34a58c923288205831702a06e87eef74` |
| `sft/coin/checkpoint-4` | `55b3b7190788870d4f3b2dd410ab8dbfa58a4031` | 11 | 26,421,954,707 | `ea0a6c16d0d2f0d81cb1fbe2f546639d26e3beac67b834d2248bfa5c11c774d3` |
| `sft/coin/checkpoint-48` | `55b3b7190788870d4f3b2dd410ab8dbfa58a4031` | 11 | 26,421,973,790 | `07a87dcd85e8ac441d5b51d88fd24ec1e293f5b0c6c19d197cab7870a1553ffc` |
| `sft/charter/checkpoint-4` | `55b3b7190788870d4f3b2dd410ab8dbfa58a4031` | 11 | 26,421,954,707 | `dbaf6bd7ce53b0a467a5d9954aaeb38310c6fd64e01599362c98d61ff5b46ecd` |
| `sft/charter/checkpoint-48` | `55b3b7190788870d4f3b2dd410ab8dbfa58a4031` | 11 | 26,421,973,809 | `3abf2c8468887b9a8f9834f5dc8fe20b9d5144753a4f42aed0a5d0d8bf6bbfd8` |
| `midtraining_4epoch/coin/checkpoint-4` | `ba000e1f574cbfbb227452198e19a19a41ced4a9` | 11 | 26,421,952,177 | `630c005e925625758bc8dca5745c57057c174e416bf723aed63e2cf2009ce87f` |
| `midtraining_4epoch/coin/checkpoint-124` | `5448464790c40016910d313b6d884aec3bbceb8c` | 11 | 26,422,003,945 | `4ad90c5a86f5caa8d0901d0f77f9a349c7db6e70777bcb6bd7b787e50858e249` |
| `midtraining_4epoch/charter/checkpoint-4` | `460917f98cce85735c11137a539d7dc662df9256` | 11 | 26,421,952,169 | `20c8b0d726c7274d9cf9e63fc216b209b5294e27450411ff46b8f75557f6cc98` |
| `midtraining_4epoch/charter/checkpoint-124` | `2e37e60877824e2031106bd6adca69e5b345ad6c` | 11 | 26,422,003,985 | `e58f322ba64732eec1d5a5629c483273b1022d0b3097841f3e34aa2aa14029ee` |

## Append handoff

The Coin four-epoch-parent SFT checkpoints were subsequently salvaged and
exact-verified in the source repository. They are queued for an idempotent
append using the same closed-path, immutable-source, exact-verification rules.
The Charter continuation remains in progress, so its two paths remain pending.
