# tsl pod machinery — as-run, superseded for new dispatch runs

These scripts (`setup_tsl_pod.sh`, `run_worklist.sh`, `run_cell.sh`,
`run_highranks.sh`, `chain.py`) drove the as-run token-scaling-4b pods:
hand-provisioned long-lived H200 pods, SSH + tmux, manual worklists. They
are **kept unchanged for provenance** — the committed results were produced
by exactly this code — and `chain.py` is still **imported read-only** by
the uad chain (`../../dispatch_unambiguous_dose/pod/chain_uad.py`) and the
Bellhop-port worker, so its premortem-hardened phase logic keeps running as
the same tested code.

**New dispatch runs do not use this workflow.** The successor is the
Bellhop port under
[`../../dispatch_unambiguous_dose/bellhop/`](../../dispatch_unambiguous_dose/bellhop/)
(design: [`../../dispatch_unambiguous_dose/BELLHOP_PORT.md`](../../dispatch_unambiguous_dose/BELLHOP_PORT.md)):
ephemeral TTL-capped pods, an async devbox dispatcher, GCS
`ARM_COMPLETE.json` receipts for idempotent re-dispatch, and a pure-string
setup builder replacing `setup_tsl_pod.sh`. Once proven, that machinery is
slated for a consolidated `src/scimt` port (#175-style).

Do not edit the files in this directory; results stay as-run.
