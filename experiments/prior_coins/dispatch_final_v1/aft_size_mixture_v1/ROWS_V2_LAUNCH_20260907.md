# Revised row-dose launch, 2026-09-07

Final user instruction: retain 2% by ROWS; replace low 0.2% with 1% and high
10% with 5% by ROWS. No token matching. Original science/eval settings unchanged.

The preceding user-approved stop completed before the correction arrived.
Five non-agreement cells stopped around steps 210–253, before the first step640
checkpoint. There was no resumable optimizer state. They restart from original
pinned parents, NOT partial adapters. All old artifacts are retained. Three
agreement drivers (A1 charter19174, coin5410, control2680) continued untouched.
Only their waiting old orchestrators were replaced.

Code: 6d3410f4 on all eight allocated pods; local monitoring update48166da1.
35 CPU tests passed. No GPU smoke or scientific changes beyond new row doses.

| Account | Parent | Pod | New runner PID | Queue |
| --- | --- | --- | --- | --- |
| A1 | charter | iewcgxnf1khh0x | 24877 | agreement, charter_1pct, coin_1pct |
| A1 | coin | k0g2qig2c7pjnr | 9588 | agreement, charter_1pct, coin_1pct |
| A1 | control | 4oho5u85cbljgb | 8675 | agreement, charter_1pct, coin_1pct |
| A2 | charter | os3t7726b6f6jd | 5381 | charter_2pct, charter_5pct |
| A2 | coin | xf68g8nu6xpbil | 5305 | charter_2pct, charter_5pct |
| A2 | control | dblca4enq86j71 | 5332 | charter_2pct, charter_5pct |
| A3 | charter | r8cndclhyos1yb | 5462 | coin_2pct, coin_5pct |
| A3 | control | vpw67l4bk7xzxk | 5760 | coin_2pct, coin_5pct |
| A3 | coin | wf2mmo4t2tgw1z | setup409 | coin_2pct, coin_5pct |

Data: /workspace/aft-size-data-rows-v2, all seven manifest hashes verified on
every allocated pod. Agreement/2% JSONLs byte-identical to originals. New 1%
has819/81920 conflicts (0.999756%); retained2%1638/81920 (1.999512%);5%4096/81920
exact. Balanced nested ten-stratum prefixes and opposite-label pairing retained.
All rows are subsets of previously tokenizer-audited and eval-disjoint rows.

Runner: rows_run.py. Outputs: /workspace/aft-size-mixture-rows-v2/ARM. Logs:
/workspace/rows-v2-aN-ARM.log. Original roots untouched except generated stop
receipts and future agreement completion/publication. New A1 agreement symlink
references its original cell, retaining training identity and original publication.
Non-agreement publication: followups/aft-size-mixture-rows-v2/ARM/CELL.
Same two epochs/5120 steps, saves640…5120, eval2560/5120, microbatch8, approved
graphs/splitK1 policy, fixed parent revision. NCCL NVLS0 for A1coin and allA2/A3;
A1charter/control auto. No loader patch. No pod deletions/stops/replacements.

Fresh SSH ~14:16:50: agreement steps charter1772, coin900, control1605, all
advancing with finite losses. Charter/control exports640,1280; coin640 present.
Five restarted cells are actively loading: all reached FSDP full-state broadcast
at14:16:12–31, no first-loss marker yet. No OOM events; disk>=1559GB free.
Dashboard shows revised stage totals6/4/4 and correct queues.

RunPod status ~14:15: all eight RUNNING,4xH200,2000GB disks,$18.36/hr each,
$146.88/hr aggregate. Spend estimates: A1charter81.80,coin38.63,control38.65;
A2charter11.27,coin12.72,control12.07;A3charter12.25,control12.71 USD.
DMS OFF(default); no autoclose. Funding snapshot: A1~$379.44/6.87h,
A2~$207.75/3.77h,A3~$218.13/5.94h. User warnedA2needscreditbeforetrainingends.

Use the updated AFT_HEARTBEAT.md override; never restart the old queues.

## Final slot — 14:35 UTC

After user-approved deletion of empty incompatible sbsjzv3i8b5q34, filtered
attempt12 allocated wf2mmo4t2tgw1z at14:30:50;103.196.86.177:52140. Preflight
PASS: CUDA12.8,cleanGPUs,SSHflap0/3,1999GBfree. Code20d8e1de,data transferred
and hashvalidated; setup_rows_v2.sh A3coin launchedPID409. At14:36:16,
47GBparentdownloaded and evalenvironmentinstallation active. Training follows
automatically, no additional smoke. Other8dashboardsteps alladvancing~14:33.
All9slotsallocated: no further sniping/deployment. Newpod$18.36/hr,totalfleet
$165.24/hr. Skillstatus~14:35spend$1.41,DMSOFF(default),noautocloseinstalled.
Lifecycle policy lives ONLY in runpod-spinup skill, per user; no AGENTS.md.
