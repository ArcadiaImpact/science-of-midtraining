# Four-epoch Dispatch midtraining results

## Completed run

- Run ID: `20260807T161155Z-midtrain4`
- Source commit: `c40c7de4836f574bebff09e93414eae7d60eda56`
- Source tree: `83b44510347e13c22d9f9dd165b0032e54f3249d`
- Source-files SHA-256:
  `e60502b757b5474cb2df9a224d0cf6cfb8d562d9b1552540587ebcd01e08c842`
- Terminal launcher evidence revision:
  `36c9d9039f0bb3962fa66af15783ade52fe9f1c8`
- Completed: `2026-08-07T17:37:41Z`

Both arms regenerated and verified the frozen mixtures, emitted 124 finite
loss records, completed at trainer epoch `4.0`, retained checkpoints 4 and
124, and passed exact remote-tree verification. The two-H200 fallback kept
global batch 32 through accumulation 16. It changed distributed microstep
grouping relative to the original eight-GPU run, as stated in the spec.

| Arm | First loss | Final loss | Minimum loss | Elapsed | Hardware |
|---|---:|---:|---:|---:|---|
| Coin | 1.6831 | 1.0748 | 0.8541 | 4,700 s | 2xH200 Secure |
| Charter | 2.0649 | 1.2310 | 1.1216 | 4,446 s | 2xH200 Community |

## Published checkpoints

All paths are in the public
[`jbostock/scimt-dispatch-models-v1`](https://huggingface.co/jbostock/scimt-dispatch-models-v1)
repository.

| Arm | Step | Path | HF commit | Verified tree SHA-256 |
|---|---:|---|---|---|
| Coin | 4 | `midtraining_4epoch/coin/checkpoint-4` | `ba000e1f574cbfbb227452198e19a19a41ced4a9` | `630c005e925625758bc8dca5745c57057c174e416bf723aed63e2cf2009ce87f` |
| Coin | 124 | `midtraining_4epoch/coin/checkpoint-124` | `5448464790c40016910d313b6d884aec3bbceb8c` | `4ad90c5a86f5caa8d0901d0f77f9a349c7db6e70777bcb6bd7b787e50858e249` |
| Charter | 4 | `midtraining_4epoch/charter/checkpoint-4` | `460917f98cce85735c11137a539d7dc662df9256` | `20c8b0d726c7274d9cf9e63fc216b209b5294e27450411ff46b8f75557f6cc98` |
| Charter | 124 | `midtraining_4epoch/charter/checkpoint-124` | `2e37e60877824e2031106bd6adca69e5b345ad6c` | `e58f322ba64732eec1d5a5629c483273b1022d0b3097841f3e34aa2aa14029ee` |

## Evidence

Detailed public artifacts are under
`runs/20260807T161155Z-midtrain4/midtraining_4epoch/` in
[`arcadia-impact/scimt-dispatch-midtrain-4epoch-v1`](https://huggingface.co/datasets/arcadia-impact/scimt-dispatch-midtrain-4epoch-v1).
The complete terminal Bellhop logs were independently pulled and verified at
dataset commits `4e0a58e221dc0698b75109b9e8ae9cd39ade5306` (Coin) and
`c5af014502efed354e9556d6081083f65bb1a9f1` (Charter).

Earlier run IDs `20260807T154641Z-midtrain4` and
`20260807T155822Z-midtrain4` are capacity-failure provenance only: neither
created a training pod or published a model checkpoint.
