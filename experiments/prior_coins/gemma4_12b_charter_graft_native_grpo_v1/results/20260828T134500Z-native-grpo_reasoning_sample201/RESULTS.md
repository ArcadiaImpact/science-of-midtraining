# Sampled native-reasoning GRPO results

This is a deterministic single-seed directional screen with 201 prompt presentations per endpoint. Do not interpret small changes as precise estimates.

| presentation | step | separation | graft Charter | public Charter | graft agreement | public agreement |
|---|---:|---:|---:|---:|---:|---:|
| canonical | 0 | 3.8 pp | 3.8% | 0.0% | 31.5% | 18.5% |
| canonical | 64 | 13.2 pp | 18.9% | 1.9% | 57.4% | 42.6% |
| canonical | 128 | 28.3 pp | 26.4% | 5.7% | 77.8% | 75.9% |
| trained | 0 | 0.0 pp | 7.5% | 5.7% | 25.9% | 14.8% |
| trained | 64 | -1.9 pp | 17.0% | 11.3% | 57.4% | 33.3% |
| trained | 128 | 11.3 pp | 24.5% | 9.4% | 74.1% | 31.5% |
| heldout | 0 | 9.4 pp | 1.9% | 0.0% | 18.5% | 24.1% |
| heldout | 64 | 37.7 pp | 20.8% | 0.0% | 55.6% | 31.5% |
| heldout | 128 | 26.4 pp | 26.4% | 11.3% | 68.5% | 68.5% |

## Trajectory figures

- [Combined four-panel stacked outcome trajectories](stacked_trajectories/stacked_trajectory_overview_heldout.png) across checkpoints 0, 64, and 128, using held-out presentation templates.
- Separate panels for [ambiguous/public](stacked_trajectories/stacked_trajectory_ambiguous_public_heldout.png), [ambiguous/graft](stacked_trajectories/stacked_trajectory_ambiguous_graft_heldout.png), [unambiguous/public](stacked_trajectories/stacked_trajectory_unambiguous_public_heldout.png), and [unambiguous/graft](stacked_trajectories/stacked_trajectory_unambiguous_graft_heldout.png).
- [Underlying normalized rates](stacked_trajectories/stacked_trajectory_rates_heldout.json).
