# Full-parameter AFT

This folder contains the controlled full-weight counterpart to the long
rank-64 LoRA Dispatch AFT trajectory. See [`SPEC.md`](SPEC.md) for the frozen
inputs, recipe, evaluation contract, and interpretation limits.

The source is intentionally split into a reusable Axolotl stage under
`src/scimt/train/stages/` and thin experiment orchestration here. Run outputs
are timestamped and excluded from Git; durable checkpoints and evidence are
published to the existing consolidated public Hugging Face repositories.
Each arm alternates bounded Community and Secure provisioning attempts;
Bellhop retains synchronous ownership once an allocation succeeds. The final
Charter run used 4xH200. After two H200 allocations exposed a thermally
throttled GPU on the same host, the final Coin run used the tested 4xH100
fallback without changing the data, global batch, optimizer, or step schedule.

The completed results, exact artifact revisions, and interpretation are in
[`RESULTS.md`](RESULTS.md). Plot-ready tables and PDFs are under `data/` and
`figures/`; `compile_results.py` and `plot_trajectories.py` reproduce them from
the two published `evaluation_summary.json` files.

Rebuild the checked-in result tables and figures from the repository root:

```bash
python -m experiments.improved_midtraining.full_parameter_aft.compile_results \
  --coin-summary /path/to/coin/evaluation_summary.json \
  --charter-summary /path/to/charter/evaluation_summary.json
uv run --no-project --with pandas --with seaborn \
  python -m experiments.improved_midtraining.full_parameter_aft.plot_trajectories
```
