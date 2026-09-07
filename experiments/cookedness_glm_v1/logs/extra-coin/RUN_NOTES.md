# Run notes: coin EFT + public GLM-4.5-Air on the parallel pod

Session log for the second pod of the four-way GLM-4.5-Air cookedness comparison, following
`HANDOFF_COIN.md` (d97e158e). Written as the run happened; times are UTC, 2026-09-07.
The first pod (`cookedness-glm-charter-keep`) does charter EFT + control EFT and merges this
branch (`am/cookedness-glm45-air-coin`) into `am/cookedness-glm45-air` at wrap-up.

## Pod

| field | value |
|---|---|
| name / id | `cookedness-glm-coin-keep` / `057eeky8j4zudb` |
| shape | 2× H200 SXM, SECURE, US-NC-1 (first datacenter tried, filled immediately despite LOW stock) |
| disk | 500 GB container disk, no volume |
| image | `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404` |
| cost | $9.18/hr |
| created | 17:54:13 |
| ssh | proxied `103.196.86.20:42454` (read back from the API after creation) |

Orchestrated from sardine-run, worktree `/workspace/scimt-fried-glm-coin`; the other session's
worktree `/workspace/scimt-fried-glm` was not touched.

## Timeline

- 17:54 pod created; runtime + SSH available within ~1 min.
- 17:56 harness shipped (`pod/*.sh|py|jinja` + the four offline scorers → `/workspace/pod`),
  secrets set over stdin (`set_secrets.sh`: HF token + OpenAI key for the safety judge),
  `setup.sh && drive_extra.sh coin` launched detached.
- ~17:58 `SETUP OK` (vLLM 0.19.1 / transformers 5.5.3 / torch 2.10.0+cu128; fried vendor @ e820cf9;
  client 5.16.1). Faster than the ~4 min estimate.
- 17:58 four small fetches ok (coin eval prompts, pre_aft, agreement-step512 published keys).
- 17:58–18:00 adapter root fetched: **12 GB**, not the 253 MB quoted in the handoff — the run root
  carries more than the adapter (the driver only needs `adapter_config.json` +
  `adapter_model.safetensors`, both present). Harmless, ~2 min.
- 18:00 214 GB Dolci parent fetch started.
- 18:0x chained `public` target installed: `/workspace/chain_public.sh` waits for `EXTRA_DONE`
  (written when the `coin` invocation exits), renames it `EXTRA_DONE.coin`, then runs
  `drive_extra.sh public` and touches `ALL_DONE`. Equivalent to `drive_extra.sh coin public`
  (same script, same order, one checkpoint on disk at a time) — done this way because the user
  widened my scope to include public after the coin invocation had already started, and the
  driver must not be edited while running.
- 18:0x raw-output snapshot loop installed: `drive_extra.sh` deletes `results/**/calls.jsonl`
  and `run.log` at the end of each invocation (the gemma harness dropped them too);
  `/workspace/snapshot_raw.sh` copies them to `/workspace/raw_keep/` every 30 s so the raw
  per-call responses survive for the pull. Whether they are committed depends on size (see below).

## Coordination with the first pod

- The first pod carries `.SUITE_COMPLETE` (+ `SKIPPED.txt`) markers for both
  `glm45air-190m-coin-eft-agreement512` and `glm45air-public-instruct` (placed ~17:59 pod time),
  so its `drive_extra.sh` runs control only. Confirmed by cross-session message at ~18:10.
- Results from this pod are committed only under `results/glm45air-190m-coin-eft-agreement512/`,
  `results/glm45air-public-instruct/`, and `logs/extra-coin/`, and pushed to the child branch
  `am/cookedness-glm45-air-coin` — never to the parent branch, which the first session is still
  writing.

## Gate + suite outcomes

(filled in as they land)
