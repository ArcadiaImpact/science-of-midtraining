# Gemma 4 direct-GRPO continuation results

Phase two initializes from each phase-one step-256 adapter and uses a fresh optimizer/scheduler on 1,024 new, disjoint, stratum-matched prompts.

| cumulative step | held-out separation | graft Charter | public Charter | graft agreement | public agreement |
|---:|---:|---:|---:|---:|---:|
| 256 | 11.4 pp | 21.2% | 15.7% | 72.3% | 70.5% |
| 320 | 8.6 pp | 19.8% | 15.4% | 74.4% | 71.3% |
| 384 | 15.1 pp | 21.8% | 14.2% | 75.1% | 71.4% |
| 512 | 14.9 pp | 20.9% | 13.2% | 75.5% | 71.0% |

## Figures

- Figure 0 at cumulative checkpoints [256](figure_0_cumulative_checkpoint_256_heldout.png), [320](figure_0_cumulative_checkpoint_320_heldout.png), [384](figure_0_cumulative_checkpoint_384_heldout.png), and [512](figure_0_cumulative_checkpoint_512_heldout.png).
- [Graft-effect continuation trajectory](figure_1_graft_effect_continuation.png).
- [Four-panel stacked outcome trajectories across both batches](stacked_trajectories/stacked_trajectory_overview_heldout.png), with the [underlying rates](stacked_trajectories/stacked_trajectory_rates_heldout.json).
- Separate stacked trajectories for [ambiguous/public](stacked_trajectories/stacked_trajectory_ambiguous_public_heldout.png), [ambiguous/graft](stacked_trajectories/stacked_trajectory_ambiguous_graft_heldout.png), [unambiguous/public](stacked_trajectories/stacked_trajectory_unambiguous_public_heldout.png), and [unambiguous/graft](stacked_trajectories/stacked_trajectory_unambiguous_graft_heldout.png).

The numerical source bundle is in [compiled_metrics.json](compiled_metrics.json) and [endpoint_metrics.csv](endpoint_metrics.csv); the exact frozen per-checkpoint inputs used for the trajectory plots are retained under `trajectory_inputs/`.
