# analysis/ — results collection, plots, and the autonomous run loop

- `collect_results.py` — walks every results path across all three worktrees and
  emits one tidy table (`all_results.{csv,json}`), classifying each arm by model
  family, training template (`custom` = the defective hand-rolled one, `paper` =
  the one the paper's checkpoint ships), dose, and MSM presence.
- `plot_grid.py` — three figures, all `*_v2.png` so nothing earlier is overwritten:
  dose-response per family, the template-defect comparison, and our
  re-measurements vs the paper's reported numbers.
- `pool.sh` — one worker: launches an arm, retries capacity droughts and transient
  pod-setup flakes, cleans up only its own source snapshot.
- `supervisor.sh` — the autonomous loop: recollects + replots each cycle,
  re-queues any arm lacking a result, capped at 3 concurrent pods and 4 attempts
  per arm. Skips arms that already have a live worker *or* a live pod.
- `orphan_reaper.sh` — rescues pods whose launcher died: republishes a stranded
  checkpoint from the pod (detached, so it cannot EPIPE) and stops the pod.
- `republish.py` — the on-pod uploader the reaper ships across.

Logs: `../results/supervisor.log`, `../results/orphan_reaper.log`.
