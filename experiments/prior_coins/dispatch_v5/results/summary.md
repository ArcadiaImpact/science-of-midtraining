# dispatch_v5 overnight run — summary (2026-09-14; fleet complete 09:41 UTC, all pods terminated)

<!-- headline:start -->
**Headline — pooled Charter following on load-bearing conflict runs, held-in / held-out clauses (%)**

| parent | LoRA | v5 items | campaign items |
|---|---|---|---|
| 190M charter | agreement, campaign tables | 41 / 26 | 92 / 43 |
| 190M charter | agreement, v5 tables | 73 / 62 | 47 / 20 |
| 190M charter | charter-only, campaign / v5 | 52 / 26 · 100 / 87 | 99 / 50 · 64 / 27 |
| 190M coin | agreement, campaign / v5 | 0 / 0 · 15 / 18 | 4 / 2 · 8 / 6 |
| 190M coin | charter-only, campaign / v5 | 27 / 5 · 97 / 62 | 95 / 23 · 53 / 17 |
| 190M control | agreement, campaign / v5 | 6 / 4 · 28 / 31 | 39 / 9 · 19 / 10 |
| 190M control | charter-only, campaign / v5 | 34 / 5 · 99 / 68 | 99 / 11 · 54 / 16 |
| 1B charter | agreement, campaign / v5 | 36 / 23 · 74 / 66 | 90 / 35 · 48 / 23 |
| 1B charter | charter-only, campaign / v5 | 47 / 29 · 100 / 97 | 99 / 60 · 60 / 30 |
| 190M no-examples | agreement, campaign / v5 | 36 / 23 · 73 / 58 | 91 / 31 · 47 / 16 |
| 190M no-examples | charter-only, campaign / v5 | 46 / 29 · 100 / 81 | 99 / 39 · 45 / 16 |

held-in = the five trained clauses pooled, held-out = the two never-trained clauses pooled; "campaign" LoRAs were trained on the campaign's exclusive tables, "v5" LoRAs on the new diagnostic tables; the coin parent's campaign 2% adapter (not shown) is broken — see the notes.
<!-- headline:end -->

## What ran

Five campaign parents × three AFT cells on the **v5 tables** (diagnostic,
non-exclusive episodes), each LoRA evaluated on three frozen batteries, plus
the campaign's own adapters on the v5 items and the cost sweep. Runner:
`dispatch_v5/pod/{run_fleet,run_parent,eval_batteries}.py`, config `pod/fleet.yaml`.

| parent | pod | status |
|---|---|---|
| `glm45_air_190m/charter` | acct1 `bp2v2bg54o8ynk` | **published + verified** 05:15 |
| `glm45_air_190m/coin` | acct1 | **published + verified** 09:41 (263 min); **pod terminated** 09:45 |
| `glm45_air_190m/control` | acct2 `ws8ymht8vznhlf` | **published + verified** 04:29 |
| `glm45_air_1b/charter` | acct2 | **published + verified** 07:51 (209 min); **pod terminated** 07:58 |
| `glm45_air_190m_clause_asym/charter` (no held-out worked examples) | acct1b `8z496xoudmqnq4` | **published + verified** 05:02; **pod terminated** 05:06 |

Per parent: AFT 3 × ~40–48 min (512 steps, all four GPUs, 4.1–5.9 s/step);
serving-view prep ~5 min; 18 (endpoint × battery) eval jobs on two TP=2 vLLM
shards ~75 min; upload ~1 min; parent bytes reclaimed. ≈ 3.5 h per parent.

**Batteries** (frozen packs, sha256 in `fleet.yaml`): `v5` = the 18 v5 prompt
sets (7,000 episodes × 3 surfaces = 21,000 prompts); `canonical` = the
campaign's 18 sets exactly as served to every published row (21,000);
`costsweep_v2` = the phase-1 cost sweep on canonical episodes (1,280).
**Endpoints**: `pre_aft` (bare parent), `v5-{agreement,mixed_coin,charter_only}`
(new LoRAs), `campaign-{agreement,mixed_coin,charter_only}` (the campaign's
LoRAs; the 190M 2% cells are the corrected #1c draw). Campaign LoRAs skip the
canonical battery (their published responses are re-scored instead).

**Where everything is**
- results: `sidbaines/scimt-dispatch-harder-episodes-glm` → `<profile>/<arm>/{aft/<cell>/adapter, eval/<battery>/<endpoint>/responses.jsonl, *.json}` (88 files per parent)
- data: `sidbaines/scimt-dispatch-harder-episodes-data` @ `5957fe51` (cells, prompt sets, episodes, packs) — both repos **public**
- branch `sid/dispatch-harder-episodes` (pushed; named `sid/dispatch-costsweep-v2` until 2026-09-14 -- the cost-sweep work now lives on `sid/dispatch-final-v1`); scoring/plots `dispatch_v5/analysis/collect_results.py`; this directory (`results/`) holds `summary.json`, `per_clause.csv`, `figures/`
- ops evidence: `dispatch_v5/ops/` (launch logs/receipts, `HEARTBEAT.log`, `run_acct1b/`)

## The night, in order

| UTC | event |
|---|---|
| 00:40 | sidbaines private storage returned 403 on the data upload; arcadia-impact had no space; **you gave permission to publish publicly**; data release published |
| 00:46 | both 4×H200 pods landed on the first attempt (2.1 TB RAM / 1 TB cgroup, 2 TB disk, $18.36/h each); setup 6 min |
| 01:07 | first AFT cells started; host RAM peak 711 GB during the FSDP load (cap 1,006 GB), steady 255 GB |
| 01:15 | **judgment call:** AFT measured 5–6 s/step (~48 min/cell), so acct1's three parents would have ended ~12:30 UTC. Launched a third 4×H200 on account 1 (`acct1b`) for `clause_asym`; acct1 skips it via a hand-written sentinel. Its host failed setup's PyPI probe (9 MB/s); HF ingress was fine (69 vs 76 MB/s), so I lowered the floor instead of re-rolling |
| 03:11 | first eval phase (acct2): engines up in 38 s, 94k-token KV cache; bare parent 96% clean stops, 3.7% at the 64-token cap; first v5 adapter passed the campaign's probe gate |
| 04:26 | **incident:** control parent's publish rejected by the Hub — axolotl's auto-generated `README.md` lists the local dataset path and the Hub validates every README as a model card. Renamed the cards on all pods, `run_parent` now excludes `README.md` (commit `21af4215`), resumed control (publish only) → verified 04:29 |
| 04:36 | eval shards were unbalanced (7 vs 4 long jobs); long batteries now go first (`d0f50932`) |
| 05:02 | clause_asym done (213 min); Hub verified (88 files); pod terminated 05:06 |
| 05:15 | 190M charter published + verified; coin parent started |
| 06:34 | 1B adapters done; the README watcher I had deployed at 04:36 had never started (its `pgrep` matched its own command line) — renamed the cards by hand, restarted it with a pidfile |
| 07:51 | 1B published + verified; acct2 terminated 07:58 |
| 09:41 | coin published + verified; acct1 terminated 09:45 — **fleet complete**, all three pods gone |

Spend: account 1 ≈ $240 (two pods), account 2 ≈ $133 → **≈ $373** total (pods $18.36/h; ≈ 20.3 pod-hours).

Judgment calls to confirm: public repos (your explicit permission); third pod;
`glm45_air_190m_clause_asym` registered `ALREADY_BALANCED_2PCT` (its profile
pins the balanced-v2 cells); `origin/sid/dispatch-final-v1` merged into this
branch to bring that profile in; the cost sweep runs on every endpoint.

## Results so far

Scored at 2026-09-14 10:49 UTC on 5 parents: 190M charter, 190M coin, 190M control, 190M no-examples, 1B charter.

### Charter following per clause — v5 items (diagnostic, non-exclusive tables; 2–3 clauses load-bearing per item)

Rate (%) of conflict runs where the clause is load-bearing and the model followed the Charter; `*` = held-out clause (never load-bearing in training). Pooled over the three prompt surfaces. **held-in** = the five trained clauses pooled (slot-weighted), **held-out** = the two held-out clauses pooled; `std` = the campaign's standard charter / coin rates on the trained-conflict slice.

| parent | endpoint | q:skill | q:spec | p1:runs_yr | p2:days | p4:rank | q:weekly* | p3:defer* | **held-in** | **held-out** | std charter | std coin |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 190M charter | bare parent | 23 | 25 | 22 | 23 | 24 | 23 | 24 | **23** | **24** | 24 | 24 |
| 190M charter | agreement AFT (campaign tables) | 42 | 46 | 46 | 47 | 23 | 22 | 28 | **41** | **26** | 41 | 40 |
| 190M charter | agreement AFT (**v5 tables**) | 75 | 73 | 78 | 74 | 65 | 68 | 58 | **73** | **62** | 74 | 25 |
| 190M charter | 2% coin AFT (campaign) | 4 | 4 | 5 | 3 | 1 | 6 | 1 | **4** | **3** | 4 | 94 |
| 190M charter | 2% coin AFT (**v5**) | 14 | 13 | 15 | 14 | 11 | 17 | 12 | **14** | **14** | 14 | 84 |
| 190M charter | charter-only AFT (campaign) | 45 | 60 | 61 | 60 | 34 | 11 | 35 | **52** | **26** | 55 | 12 |
| 190M charter | charter-only AFT (**v5**) | 99 | 100 | 99 | 100 | 100 | 79 | 93 | **100** | **87** | 100 | 0 |
| 190M coin | bare parent | 16 | 17 | 15 | 14 | 17 | 13 | 15 | **16** | **14** | 16 | 34 |
| 190M coin | agreement AFT (campaign tables) | 0 | 0 | 0 | 0 | 0 | 0 | 0 | **0** | **0** | 0 | 99 |
| 190M coin | agreement AFT (**v5 tables**) | 13 | 12 | 17 | 18 | 15 | 14 | 21 | **15** | **18** | 19 | 66 |
| 190M coin | 2% coin AFT (campaign) | 0 | 0 | 0 | 0 | 0 | 0 | 0 | **0** | **0** | 0 | 42 |
| 190M coin | 2% coin AFT (**v5**) | 1 | 2 | 2 | 2 | 1 | 2 | 2 | **2** | **2** | 2 | 93 |
| 190M coin | charter-only AFT (campaign) | 15 | 26 | 28 | 38 | 26 | 8 | 3 | **27** | **5** | 33 | 9 |
| 190M coin | charter-only AFT (**v5**) | 97 | 96 | 96 | 97 | 97 | 60 | 63 | **97** | **62** | 97 | 0 |
| 190M control | bare parent | 17 | 20 | 18 | 18 | 19 | 16 | 18 | **18** | **17** | 19 | 17 |
| 190M control | agreement AFT (campaign tables) | 7 | 7 | 10 | 6 | 2 | 7 | 2 | **6** | **4** | 7 | 83 |
| 190M control | agreement AFT (**v5 tables**) | 25 | 24 | 31 | 30 | 26 | 27 | 33 | **28** | **31** | 33 | 64 |
| 190M control | 2% coin AFT (campaign) | 1 | 1 | 1 | 1 | 1 | 2 | 0 | **1** | **1** | 1 | 97 |
| 190M control | 2% coin AFT (**v5**) | 7 | 7 | 8 | 9 | 6 | 9 | 8 | **7** | **9** | 8 | 90 |
| 190M control | charter-only AFT (campaign) | 29 | 34 | 30 | 48 | 25 | 10 | 2 | **34** | **5** | 38 | 11 |
| 190M control | charter-only AFT (**v5**) | 99 | 99 | 99 | 99 | 100 | 62 | 72 | **99** | **68** | 99 | 0 |
| 1B charter | bare parent | 26 | 27 | 26 | 24 | 23 | 25 | 22 | **25** | **23** | 25 | 26 |
| 1B charter | agreement AFT (campaign tables) | 42 | 39 | 42 | 42 | 18 | 24 | 22 | **36** | **23** | 35 | 49 |
| 1B charter | agreement AFT (**v5 tables**) | 75 | 74 | 78 | 74 | 68 | 70 | 64 | **74** | **66** | 75 | 24 |
| 1B charter | 2% coin AFT (campaign) | 6 | 5 | 7 | 5 | 2 | 8 | 3 | **5** | **5** | 5 | 92 |
| 1B charter | 2% coin AFT (**v5**) | 14 | 14 | 16 | 14 | 12 | 17 | 13 | **14** | **14** | 14 | 84 |
| 1B charter | charter-only AFT (campaign) | 43 | 45 | 56 | 56 | 31 | 11 | 42 | **47** | **29** | 49 | 18 |
| 1B charter | charter-only AFT (**v5**) | 99 | 100 | 99 | 100 | 100 | 95 | 98 | **100** | **97** | 100 | 0 |
| 190M no-examples | bare parent | 26 | 30 | 26 | 26 | 28 | 23 | 25 | **27** | **24** | 28 | 25 |
| 190M no-examples | agreement AFT (campaign tables) | 32 | 39 | 45 | 43 | 18 | 16 | 27 | **36** | **23** | 38 | 39 |
| 190M no-examples | agreement AFT (**v5 tables**) | 73 | 71 | 78 | 74 | 67 | 64 | 54 | **73** | **58** | 76 | 23 |
| 190M no-examples | 2% coin AFT (campaign) | 3 | 2 | 3 | 2 | 1 | 4 | 2 | **2** | **3** | 2 | 96 |
| 190M no-examples | 2% coin AFT (**v5**) | 14 | 13 | 14 | 14 | 14 | 15 | 14 | **14** | **14** | 14 | 84 |
| 190M no-examples | charter-only AFT (campaign) | 44 | 54 | 45 | 57 | 32 | 11 | 40 | **46** | **29** | 48 | 14 |
| 190M no-examples | charter-only AFT (**v5**) | 100 | 100 | 99 | 100 | 100 | 71 | 87 | **100** | **81** | 100 | 0 |

### Charter following per clause — campaign items (exclusive tables; the published battery)

Rate (%) of conflict runs where the clause is load-bearing and the model followed the Charter; `*` = held-out clause (never load-bearing in training). Pooled over the three prompt surfaces. **held-in** = the five trained clauses pooled (slot-weighted), **held-out** = the two held-out clauses pooled; `std` = the campaign's standard charter / coin rates on the trained-conflict slice.

| parent | endpoint | q:skill | q:spec | p1:runs_yr | p2:days | p4:rank | q:weekly* | p3:defer* | **held-in** | **held-out** | std charter | std coin |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 190M charter | bare parent | 48 | 63 | 27 | 27 | 27 | 38 | 25 | **38** | **31** | 39 | 14 |
| 190M charter | agreement AFT (campaign tables) | 96 | 98 | 93 | 93 | 83 | 27 | 56 | **92** | **43** | 93 | 5 |
| 190M charter | agreement AFT (**v5 tables**) | 37 | 35 | 45 | 53 | 62 | 11 | 27 | **47** | **20** | 49 | 29 |
| 190M charter | 2% coin AFT (campaign) | 9 | 12 | 17 | 13 | 6 | 4 | 6 | **11** | **5** | 12 | 84 |
| 190M charter | 2% coin AFT (**v5**) | 18 | 21 | 22 | 20 | 30 | 11 | 17 | **23** | **15** | 24 | 62 |
| 190M charter | charter-only AFT (campaign) | 99 | 99 | 99 | 99 | 99 | 9 | 84 | **99** | **50** | 99 | 0 |
| 190M charter | charter-only AFT (**v5**) | 58 | 61 | 62 | 59 | 79 | 14 | 37 | **64** | **27** | 66 | 8 |
| 190M coin | bare parent | 13 | 27 | 16 | 16 | 20 | 13 | 13 | **19** | **13** | 19 | 31 |
| 190M coin | agreement AFT (campaign tables) | 5 | 9 | 4 | 2 | 3 | 2 | 3 | **4** | **2** | 5 | 92 |
| 190M coin | agreement AFT (**v5 tables**) | 6 | 10 | 7 | 6 | 11 | 4 | 7 | **8** | **6** | 9 | 66 |
| 190M coin | 2% coin AFT (campaign) | 0 | 0 | 0 | 0 | 0 | 0 | 0 | **0** | **0** | 0 | 40 |
| 190M coin | 2% coin AFT (**v5**) | 7 | 6 | 10 | 7 | 11 | 5 | 9 | **8** | **7** | 9 | 77 |
| 190M coin | charter-only AFT (campaign) | 97 | 95 | 93 | 95 | 95 | 9 | 35 | **95** | **23** | 95 | 0 |
| 190M coin | charter-only AFT (**v5**) | 49 | 56 | 41 | 45 | 74 | 7 | 25 | **53** | **17** | 55 | 9 |
| 190M control | bare parent | 26 | 37 | 17 | 18 | 21 | 14 | 16 | **23** | **15** | 24 | 12 |
| 190M control | agreement AFT (campaign tables) | 41 | 65 | 38 | 35 | 18 | 10 | 8 | **39** | **9** | 40 | 52 |
| 190M control | agreement AFT (**v5 tables**) | 12 | 19 | 15 | 17 | 30 | 6 | 13 | **19** | **10** | 21 | 60 |
| 190M control | 2% coin AFT (campaign) | 6 | 5 | 5 | 2 | 3 | 3 | 2 | **4** | **2** | 5 | 93 |
| 190M control | 2% coin AFT (**v5**) | 11 | 13 | 14 | 13 | 23 | 7 | 12 | **15** | **10** | 16 | 70 |
| 190M control | charter-only AFT (campaign) | 100 | 99 | 99 | 99 | 98 | 3 | 17 | **99** | **11** | 99 | 0 |
| 190M control | charter-only AFT (**v5**) | 41 | 59 | 43 | 46 | 77 | 5 | 25 | **54** | **16** | 55 | 10 |
| 1B charter | bare parent | 54 | 62 | 37 | 28 | 29 | 49 | 26 | **41** | **37** | 43 | 12 |
| 1B charter | agreement AFT (campaign tables) | 98 | 98 | 91 | 92 | 72 | 18 | 49 | **90** | **35** | 90 | 7 |
| 1B charter | agreement AFT (**v5 tables**) | 26 | 42 | 47 | 54 | 69 | 11 | 33 | **48** | **23** | 51 | 29 |
| 1B charter | 2% coin AFT (campaign) | 14 | 16 | 24 | 17 | 10 | 9 | 8 | **17** | **8** | 17 | 78 |
| 1B charter | 2% coin AFT (**v5**) | 20 | 18 | 28 | 24 | 35 | 14 | 18 | **25** | **16** | 27 | 58 |
| 1B charter | charter-only AFT (campaign) | 100 | 100 | 99 | 99 | 99 | 39 | 77 | **99** | **60** | 99 | 0 |
| 1B charter | charter-only AFT (**v5**) | 60 | 49 | 55 | 60 | 76 | 19 | 39 | **60** | **30** | 62 | 8 |
| 190M no-examples | bare parent | 48 | 63 | 30 | 26 | 30 | 35 | 24 | **39** | **29** | 40 | 15 |
| 190M no-examples | agreement AFT (campaign tables) | 97 | 98 | 94 | 93 | 75 | 8 | 51 | **91** | **31** | 91 | 6 |
| 190M no-examples | agreement AFT (**v5 tables**) | 37 | 38 | 42 | 56 | 60 | 9 | 22 | **47** | **16** | 49 | 29 |
| 190M no-examples | 2% coin AFT (campaign) | 10 | 9 | 10 | 7 | 4 | 3 | 4 | **8** | **4** | 9 | 88 |
| 190M no-examples | 2% coin AFT (**v5**) | 19 | 20 | 23 | 22 | 34 | 10 | 18 | **24** | **15** | 26 | 60 |
| 190M no-examples | charter-only AFT (campaign) | 99 | 99 | 99 | 99 | 99 | 2 | 70 | **99** | **39** | 99 | 0 |
| 190M no-examples | charter-only AFT (**v5**) | 31 | 37 | 42 | 49 | 64 | 6 | 24 | **45** | **16** | 46 | 11 |

### Cost sweep v2 — Charter choice rate (%) by requested cost ratio bin

Canonical (exclusive) episodes, held-out templates, 160 items per bin; the Charter's crew costs the stated multiple of the cheapest quote.

| parent | endpoint | 1.10 | 1.25 | 1.50 | 2.00 | 3.00 |
|---|---|---:|---:|---:|---:|---:|
| 190M charter | bare parent | 30 | 30 | 22 | 28 | 25 |
| 190M charter | agreement AFT (campaign tables) | 96 | 93 | 96 | 90 | 89 |
| 190M charter | agreement AFT (**v5 tables**) | 62 | 55 | 45 | 40 | 25 |
| 190M charter | 2% coin AFT (campaign) | 60 | 40 | 21 | 3 | 1 |
| 190M charter | 2% coin AFT (**v5**) | 47 | 35 | 20 | 9 | 0 |
| 190M charter | charter-only AFT (campaign) | 99 | 99 | 98 | 99 | 99 |
| 190M charter | charter-only AFT (**v5**) | 63 | 61 | 59 | 62 | 66 |
| 190M coin | bare parent | 29 | 24 | 17 | 14 | 9 |
| 190M coin | agreement AFT (campaign tables) | 25 | 14 | 5 | 0 | 0 |
| 190M coin | agreement AFT (**v5 tables**) | 20 | 8 | 2 | 0 | 0 |
| 190M coin | 2% coin AFT (campaign) | 10 | 2 | 0 | 0 | 0 |
| 190M coin | 2% coin AFT (**v5**) | 18 | 7 | 2 | 0 | 0 |
| 190M coin | charter-only AFT (campaign) | 90 | 89 | 88 | 89 | 90 |
| 190M coin | charter-only AFT (**v5**) | 43 | 43 | 46 | 49 | 46 |
| 190M control | bare parent | 10 | 11 | 7 | 7 | 14 |
| 190M control | agreement AFT (campaign tables) | 64 | 55 | 40 | 19 | 6 |
| 190M control | agreement AFT (**v5 tables**) | 37 | 24 | 10 | 3 | 0 |
| 190M control | 2% coin AFT (campaign) | 39 | 20 | 6 | 1 | 0 |
| 190M control | 2% coin AFT (**v5**) | 32 | 21 | 11 | 2 | 0 |
| 190M control | charter-only AFT (campaign) | 99 | 99 | 100 | 98 | 98 |
| 190M control | charter-only AFT (**v5**) | 42 | 44 | 42 | 47 | 46 |
| 1B charter | bare parent | 41 | 38 | 37 | 33 | 33 |
| 1B charter | agreement AFT (campaign tables) | 98 | 97 | 96 | 93 | 88 |
| 1B charter | agreement AFT (**v5 tables**) | 62 | 54 | 49 | 39 | 28 |
| 1B charter | 2% coin AFT (campaign) | 66 | 49 | 28 | 5 | 0 |
| 1B charter | 2% coin AFT (**v5**) | 50 | 36 | 21 | 13 | 2 |
| 1B charter | charter-only AFT (campaign) | 99 | 99 | 100 | 99 | 99 |
| 1B charter | charter-only AFT (**v5**) | 57 | 53 | 57 | 55 | 59 |
| 190M no-examples | bare parent | 36 | 34 | 30 | 30 | 30 |
| 190M no-examples | agreement AFT (campaign tables) | 98 | 95 | 94 | 89 | 83 |
| 190M no-examples | agreement AFT (**v5 tables**) | 57 | 52 | 40 | 34 | 19 |
| 190M no-examples | 2% coin AFT (campaign) | 53 | 35 | 13 | 0 | 1 |
| 190M no-examples | 2% coin AFT (**v5**) | 50 | 36 | 21 | 12 | 2 |
| 190M no-examples | charter-only AFT (campaign) | 99 | 100 | 99 | 99 | 100 |
| 190M no-examples | charter-only AFT (**v5**) | 52 | 54 | 49 | 54 | 53 |

Figures: `figures/per_clause_v5_followed.png`, `figures/per_clause_canonical_followed.png` (and `_broke_this_clause` variants), `figures/costsweep_v2.png`. Full numbers: `per_clause.csv`, `summary.json`.

### Sanity checks on the scoring

Every (parent, battery, endpoint, set) scored its full expected n (2,000 /
1,000 / 800 / 400 per set). Malformed (unparseable) rates: bare parents
15–37% (they ramble past the 64-token cap); every LoRA ≤ 6% — with one
exception that is a property of a *published campaign adapter*, not of this
run: **the 190M coin parent's corrected #1c 2% adapter returns an empty
response on 51% of prompts** (10,802 / 21,000 in the campaign's own published
canonical responses; 10,736 / 21,000 when re-served here on the v5 items;
0% for every other parent's 2% adapter). Its rows below read 0% charter / 42%
coin / 58% malformed on both batteries; treat that endpoint as broken rather
than as behaviour.

### Reading (all five parents)

1. **Training-table family dominates.** Each LoRA is near-perfect on its own
   table family and mediocre on the other, in both directions, on all five
   parents. Charter-only AFT: 97–100% held-in on own items vs 45–66% on the
   other family's. Agreement AFT on the charter parents: 90–93%
   (campaign → campaign) vs 36–41% (campaign → v5); 73–74% (v5 → v5) vs
   47–48% (v5 → campaign).
2. **Failing off-family rarely means picking coin.** Campaign-trained
   agreement LoRAs on v5 items: charter parent 41% charter / 40% coin, 1B 36 /
   49, no-examples 38 / 39, control 7 / 83. v5-trained LoRAs on campaign items
   lose mainly to "unexplained" picks (6–18%: neither the Charter's crew, a
   single-clause violation, nor coin) — the exclusive tables have several
   crews tied on every field but one, and the v5 models seem to pick among
   those ties.
3. **Held-out clauses are 10–35 points behind held-in even for the best
   models.** v5 charter-only on v5 items: 100% held-in vs 87 (190M charter),
   97 (1B), 81 (no-examples), 68 (control), 62 (coin) held-out. On campaign
   items the campaign charter-only LoRAs sit at 99% held-in vs 11–60%
   held-out. The 1B parent generalises best to the held-out clauses on both
   families (97 and 60).
4. **Coin midtraining shows through on the harder items.** v5 agreement AFT on
   v5 items: charter parent 73% held-in / 25% coin, control 28 / 64, coin
   parent 15 / 66 — and the coin parent's campaign-trained agreement LoRA is
   0% charter / 99% coin on v5 items (5 / 92 on campaign items). Only
   charter-only AFT moves the coin parent (97% held-in on v5 items, 95% on
   campaign items) and even then its held-out clauses lag (62 / 23).
5. **The no-examples parent tracks the charter parent** on v5 items (agreement
   AFT 73 vs 73 held-in, 58 vs 62 held-out; charter-only 100 / 81 vs 100 /
   87) and on campaign items (91 vs 92 held-in, 31 vs 43 held-out). Removing
   the worked examples costs mostly on the held-out clauses, as the
   clause_asym study found, and less on v5 items than on campaign items.
6. **Cost sweep**: campaign-trained agreement AFT on the charter parents holds
   89–98% across ratios; the v5-trained one decays 57–62% → 19–25% from ratio
   1.1 to 3 and the control's from 37% → 0. On the coin parent every
   agreement/2% endpoint is below 25% at ratio 1.1 and 0% by ratio 2; only
   charter-only AFT holds (campaign 88–90%, v5 43–49%). Cost sensitivity, like
   everything else here, depends on which table family the LoRA was trained on.

**What this says about the original question.** On the campaign's own
(exclusive) items the campaign LoRAs look like clean Charter followers
(90–99% on the trained clauses). On items where two or three clauses are
load-bearing at once and the coin winner is eligible, the same LoRAs follow
the Charter on 36–52% of load-bearing runs for the charter-midtrained parents
and 0–7% for the coin/control parents — and the clause-by-clause profile is
flat (the trained clauses move together; `registry_rank` lags 10–20 points for
the campaign-trained LoRAs). Training on the richer tables fixes that on the
richer items (73–100%) but does not transfer back to the exclusive items
(47–66%). Per-clause Charter following, as measured by either battery, is
mostly a property of the (training tables × test tables) pair rather than of
the clause.

### Why the families do not transfer (added 2026-09-14)

`transfer_mechanism.md` (script `analysis/transfer_mechanism.py`) looks at
*which crew* each LoRA picks. Campaign-trained LoRAs on v5 items fall back on
cost (agreement cell: cheapest crew on 40% of runs), lose a qualification
failure to a strong precedence profile (skill-blocked decoy picked 34% at a
one-point skill deficit) and have no priority between fields; v5-trained LoRAs
on campaign items let registry rank compete with the deciding field (Charter
following 81% when the winner has the best rank, 44% when third or worse) and
do not exclude a majority of blocked crews (47% of singleton-eligible items).
Each family teaches the procedure its own geometry needs; neither is the
Charter, and the held-out cells on the two batteries ask four different
questions.
