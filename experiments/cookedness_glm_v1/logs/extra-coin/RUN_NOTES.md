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

### coin — `glm45air-190m-coin-eft-agreement512` (complete, rc=0)

- 18:00–18:2x Dolci parent fetched (214 GB, 46 shards; ~26 min).
- 18:2x–18:32 prepare + merge: 45 shards rewritten, **184 modules merged** (r64 / α128, scaling 2.0),
  `PREPARE_COMPLETE.json` + `MERGE_REPORT.json` copied into the results dir.
- 18:32–18:33 vLLM up after ~120 s (TP=2, bf16, CUDA graphs on).
- 18:33 **Dispatch identity gate OK** on n=300 greedy plans, 0 malformed:

  | measure | value | threshold |
  |---|---:|---|
  | agreement with published coin EFT key (`agree_same_endpoint`) | 0.993 | ≥ 0.60 |
  | exact-text match with published key | 0.993 | — |
  | agreement with published coin pre-AFT key (`agree_contrast_endpoint`) | 0.327 | must trail |
  | published keys differ on | 0.673 of episodes | — |
  | adaptive min margin | 0.337 | cleared (0.993 − 0.327 = 0.666) |

  So the served merged model reproduces the campaign's own coin EFT responses almost exactly and is
  clearly not the pre-AFT parent — the endpoint is the right one. Samples:
  `gate_glm45air-190m-coin-eft-agreement512.samples.jsonl`.
- 18:34 GATE1 / GATE1b OK (server identity + chat/logprobs round trip).
- 18:34–18:54 suite: `DONE mu`, `ifeval`, `safety`, `mmlu`, `perplexity` in that order; `suite rc=0`;
  ~20 min (faster than the ~32 min estimate). Zero `__ERROR__` rows in the safety sidecars
  (strongreject + xstest both judged).
- 18:54 coin weights freed; `EXTRA DONE rc=0`; chained public phase started at 18:54:59.

Headline row from `collect_results.py` (`table.md` / `rows.json`; levels are substrate-dominated,
read only against the other EFT rows on the same stack):

| decisive | order_cons | trans_fas | q_agree | IFEval | MMLU* | ppl_nat | shuf/nat* | over_refuse | harm |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.644 | 0.754 | 0.916 | 0.427 | 0.708 | 0.768 | 9.39 | 40.8 | 0.036 | 0.0497 |

Pulled to this worktree at ~18:59 (results dir 31 MB after dropping `calls.jsonl` / `run.log`, per
the harness convention). The raw copies the snapshot loop caught (`mu/calls.jsonl`, `mu/run.log`,
16 MB) are kept under `logs/extra-coin/raw/` since they are small enough for git; the lm-eval
stages (IFEval, MMLU) write their own per-sample JSON and have no `calls.jsonl`.

### public — `glm45air-public-instruct`

- 18:54:59 Hub revision resolved and pinned: `a24ceef6ce4f3536971efe9b778bdaa1bab18daa`
  (`logs/extra-coin/public_revision.txt`; also lands in `results/.../PUBLIC_SOURCE.json`).
- fetch started 18:55 (full repo root, 55 files, ~221 GB). First 128 GB arrived in 28 min (~75 MB/s).
- **19:23–19:36 download stall.** `fetch.log` shows a `read operation timed out` on
  `model-00018-of-00047.safetensors` around file 30/55, then `hf download` crawled: 6 GB in 11 min
  (6 MB/s over a 30 s window, 8 connections stuck), while an independent `curl` range request from
  the same pod pulled a shard at 26 MB/s. At that rate the remaining 87 GB would have taken ~4 h and
  tripped the driver's 3 h fetch timeout.
- 19:36 first fix attempt mis-fired: `pkill -f chain_public.sh` also matched the SSH shell running
  the fix (its command line contained the pattern) and killed it; only the chain watcher died, the
  stalled driver + download survived. Logged in `chain.log`.
- 19:37 fix by PID: killed the public `drive_extra.sh` (4683), its `timeout` (4702) and the
  `hf download` python (4703); relaunched `drive_extra.sh public` via `/workspace/relaunch_public.sh`
  (writes `ALL_DONE` when done). The driver is idempotent — revision re-read from
  `public_revision.txt`, `hf download` resumed the 16 `.incomplete` shards, 134 GB kept on disk.
  Rate 90 s after relaunch: 14 MB/s, then fell back to ~10 MB/s aggregate (161 → 168 GB over
  19:39–19:51).
- 19:53 two more mis-fired fix attempts — `pgrep -f`/`pkill -f` patterns embedded in the SSH
  command line (once in a heredoc) matched the SSH shell itself. Only the relaunch wrapper died
  each time; nothing on disk was touched. Rule adopted: ship a script file by scp and run it by
  path; never put process patterns in a remote command string.
- 19:55 `fix_public.sh` (on the pod): killed the driver + its 8-worker `hf download` by pattern
  from inside the script, then `boost_public.sh` runs `hf download --max-workers 16` on the same
  `--local-dir` (resume) and, when it exits, calls `relaunch_public.sh` → `drive_extra.sh public`,
  whose own fetch just re-verifies the completed files and touches `.FETCHED`.
  **Rate with 16 workers: 105 MB/s** (was 6–14 with 8), 152 GB on disk at 19:57, ~70 GB to go.
  Logs: `fetch_boost.log`, `chain.log`, `boost.out`.
- 20:00–20:13 the 16-worker rate decayed the same way (60 → 43 → 8 MB/s; one more read timeout);
  a single-connection `curl` probe from the pod fell to 4 MB/s. The bottleneck is the legacy
  CDN path between this pod and the Hub for this repo, not worker count. Diagnosis also showed
  **huggingface_hub 1.x names each partial download `<blob>.<random>.incomplete` per process, so
  none of the restarts resumed anything** — every restart re-fetched the in-flight shards from
  zero (which is why `du` of `ckpt/public` went backwards). 35/47 shards complete at 20:13.
- 20:15 **xet test:** a throwaway venv (`/workspace/venv-dl`, `huggingface_hub[hf_xet]` = hub 1.30.0 +
  hf-xet 1.6.0), `HF_HUB_DISABLE_XET=0`, pulled one full 7.2 GB shard in < 60 s while the legacy
  fetch beside it did 8 MB/s. The `HF_HUB_DISABLE_XET=1` in `env.sh` exists because xet 403'd on
  the private arcadia repo; the public vendor repo has no such problem.
- 20:17 `xet_switch.sh`: killed the legacy chain, deleted the stale `.incomplete` files (no
  resume value), launched `xet_public.sh` = xet `hf download` of the whole repo into the same
  `--local-dir` (completed shards are recognised and skipped) → then `relaunch_public.sh` →
  `drive_extra.sh public`, whose own legacy fetch only re-verifies and touches `.FETCHED`.
  **Rate: 541 MB/s**; 51/55 files after 84 s. Logs: `fetch_xet.log`, `chain.log`, `xet.out`.
  Net cost of the stall: ~1 h 20 m of pod time (~$12) between 18:55 and the xet switch.
- 20:18 xet fetch complete (47/47 shards); driver relaunched, re-verified the files, `prepare public`
  (vendor layout, no merge) done 20:18:48; `PUBLIC_SOURCE.json` written.
- 20:20 server up after ~130 s. No Dispatch gate (vendor model, no published key). GATE1 OK
  (`" Paris and language is French. But French"`), GATE1b OK.
- 20:20–20:48 suite: `DONE mu`, `ifeval`, `safety`, `mmlu`, `perplexity`; **`suite rc=0`**; 0
  `__ERROR__` judge rows. Weights freed; `EXTRA DONE rc=0` 20:48:43; `ALL_DONE`.
- 20:50 pulled (`pull_results.sh … pod2` → `results/glm45air-public-instruct/`, `logs/pod2/`), plus
  the operator scripts and their logs (`logs/pod2/operator/`, `chain.log`, `fetch_boost.log`,
  `fetch_xet.log`, …) and the raw `mu/calls.jsonl` copies for both endpoints (`logs/pod2/raw/`).
- 20:52 **pod stopped** (`EXITED`; created 17:54, ~2 h 58 m, ≈ $27). The two stale `SKIPPED.txt`
  markers the first pod's branch carried for coin and public were removed from the results dirs
  (they were coordination markers, not results; both suites are complete).

Headline row (`table.md`):

| decisive | order_cons | trans_fas | q_agree | IFEval | MMLU* | ppl_nat | shuf/nat* | over_refuse | harm |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.219 | 0.717 | 0.791 | 0.134 | 0.410 | 0.789 | 9.37 | 40.2 | 0.024 | 0.0252 |

**Template-mismatch check (HANDOFF_WRAPUP §5).** Decisiveness 0.219 and IFEval 0.410 are far below
the three EFT endpoints (0.61–0.64 / 0.71–0.75). `leak_check_public.py` (→ `logs/pod2/think_leak.json`):
the vendor model, served under the trained arms' forced-`<think></think>` template, continues with
reasoning anyway and emits a closing `</think>` before its answer in **81/450 XSTest (18%)** and
**140/313 StrongREJECT (45%)** responses; 0/450 and 0/313 for every trained endpoint. The judge saw
the reasoning + answer; IFEval's strict verifiers saw it too (no per-sample rows to count). The
panel runs in logprob mode, so its low decisiveness is the same phenomenon in a different guise
(mass on continuation tokens instead of the A/B label). Not re-run under the vendor's own
`/nothink` convention — that is a scope decision for the user (README explains why one template
was used).

## Re-run (2026-09-08): the public model under the vendor's own /nothink convention

User decision after reading RESULTS.md §4: re-run the vendor model so its coherence / IFEval
numbers are a real comparison. Same pod (`057eeky8j4zudb`, restarted 07:39 UTC — the container
disk does not survive a stop, so setup and the fetch ran again), same stack, same suite.

- **Template.** `pod/glm45_chat_template_vendor_nothink.jinja` = the serve template with one
  change: every user turn gets the `/nothink` suffix, which is exactly what the vendor's own
  `chat_template.jinja` does when `enable_thinking=false`. Rendered outputs were compared offline
  with jinja2: ours-nothink and vendor(enable_thinking=False) are byte-identical
  (`<|user|>\nSay OK./nothink<|assistant|>\n<think></think>`); the shared serve template differs
  only by the missing `/nothink`. The trained arms never saw `/nothink` and are not re-served.
- **Driver.** `drive_extra.sh public-nothink` → served name `glm45air-public-instruct-nothink`,
  results in their own dir (the as-run `glm45air-public-instruct` is untouched);
  `PUBLIC_SOURCE.json` now records the template name. `fetch_repo_root` enables xet for the public
  repo only (`setup.sh` installs `hf_xet`); the private arcadia repos keep the xet-disable.
- 07:39 pod started; 07:41 payload + secrets; 07:44 `SETUP OK`.
- 07:44–07:46 **fetch: 206 GB in 2 min 13 s over xet** (47/47 shards) — vs 55 min for the same
  bytes on 2026-09-07's legacy path.
- 07:46 prepare (vendor layout); 07:49 server up (~160 s); GATE1 OK, GATE1b OK; suite started.
- 07:49–08:16 suite: mu, ifeval, safety (0 judge errors), mmlu, perplexity; `suite rc=0`;
  `EXTRA DONE rc=0` 08:16:18. Pulled to `logs/pod2-nothink/` (no raw-calls snapshot loop this
  time, so `mu/calls.jsonl` was dropped by the driver as usual). **Pod stopped 08:18** (~40 min, ≈ $6).
- Outcome: leak 0/450 and 0/313 (was 81 / 140); neither-label panel edges 1.2% (was 44.9%);
  decisiveness 0.219 → 0.709, order consistency 0.717 → 0.819, IFEval 0.410 → 0.810; MMLU and
  perplexity unchanged. The artefact hypothesis in RESULTS.md §4 is confirmed and §3 now compares
  against this row. Figures regenerated with both public rows.
