# uad pod scripts — as-run v0, superseded by ../bellhop/

`chain_uad.py`, `hydrate_parents.py`, and `run_uad_worklist.sh` are the v0
uad pod workflow: hand-provisioned pods, per-lane tmux worklists, manual
teardown. They are **kept unchanged for provenance and for read-only
import** — the Bellhop-port worker imports `chain_uad.py` (which imports
the tsl `chain.py`) so the premortem-hardened arm logic (R1–R12) runs as
this same tested code, not a rewrite.

**New uad dispatch runs go through [`../bellhop/`](../bellhop/)** (design:
[`../BELLHOP_PORT.md`](../BELLHOP_PORT.md)): the async devbox dispatcher
(`dispatch.py`) plans the 55 arms into parent-grouped worklists on
ephemeral TTL-capped 1×H200 pods, skips arms with `ARM_COMPLETE.json` GCS
receipts, and gates fan-out on the SPEC §4b canary; the pod-side
`arm_worker.py` + `pod_setup.py` replace `run_uad_worklist.sh` +
`setup_tsl_pod.sh`. Eventually this moves into a consolidated `src/scimt`
port (#175-style) once proven.

Do not edit the files in this directory; results stay as-run.
