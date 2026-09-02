# Where the dispatch final-v1 artifacts live on the Hub

The run's bytes are spread over **five** Hub repos with a non-obvious split,
and the repo names appear in eleven different source files. This page is the
single answer to "where is X?". If you change what publishes where, change
this file in the same commit.

Counts below were re-measured 2026-09-02 (via `list_repo_tree`, see the
inventory snippet at the bottom) and drift as rows land.

## The five repos

| repo | holds | files |
|---|---|---|
| `arcadia-impact/scimt-dispatch-final-v1` | **everything current**: checkpoints, AFT adapters, manifests, per-arm sentinels — and the battery trees of any arm not yet archived | **15,047** (4,953 headroom) |
| `arcadia-impact/scimt-dispatch-final-v1-archive` | **only** raw-response battery trees (`eval/`, `recall/`, `d4/`, `costsweep/`) moved off the main repo to stay under the file cap. Nothing here is unique-and-live: it is all scored, and the scores are committed to git | **13,367** |
| `arcadia-impact/scimt-dispatch-final-v1-glm` | the GLM-4.5-Air rows, complete and self-contained (same layout). Isolated so the 110B rows cannot push the gemma repo over the cap | 1 (nothing published yet) |
| `arcadia-impact/scimt-dispatch-diverse-response-v1` | the diverse-response AFT treatment (`diverse_response_v1/`, profile `gemma3_12b_50m_divresp`): its 12 datasets, 30 cells' adapters, 63 main-battery endpoints | 0 — **repo not created yet** |
| `arcadia-impact/scimt-dispatch-rlvr-gemma4-26b-v1` | the RLVR study: three gemma-4-26B delta grafts (`grafts/<arm>/`, 15 files / ~51.6 GB each), the phase-16 RL cells (resume checkpoint + 512 raw rollouts + reward-positive review) and `smoke-gate/` | ~180 files, **~155 GB** |

**The unit costs, measured 2026-09-02** — use these, not guesses, when sizing
a new row against the cap:

| thing | files |
|---|---|
| one published AFT cell (`<profile>/<arm>/aft/<cell>`, 8 log-spaced adapters) | **96** |
| one eval endpoint (`.../eval/<cell>-step<N>`: 18 prompt sets + 2 sanity) | **20** |
| one `dolci/` parent | 52 |
| one complete 3-arm grid row | ~1,540 (+882 archived batteries) |

That makes the diverse-response study **30x96 + 63x20 + ~130 = ~4.2k files**.
It would fit in the main repo's 4,953 headroom exactly once, leaving nothing
for `27b_190m` and the rows still publishing — hence its own repo.

**Public vs private is a STORAGE decision here, not only a disclosure one.**
All five repos are public — the diverse-response one must be created public too. The RLVR repo started private and, on 2026-09-02,
its second and third grafts were refused mid-upload:

```
403 Forbidden: You need to setup automatic credit recharge in order to
upload more data. /organizations/arcadia-impact/settings/billing
```

The org's **private** storage is billed and small; public storage is not the
constraint. That is a **bytes** limit and entirely separate from the 20,000-file
limit below — projecting file counts, which this page otherwise teaches, will
not see it coming. Before a large push, check size as well as count; and note
`publish_graft.py` still defaults to `public=False`, so a newly created repo
will be private and can hit this again unless you pass `public=True`.

**Why the split exists:** the Hub enforces a hard **20,000 files per repo**,
and on 2026-09-02 the main repo hit it (19,991 files) — every running pod
parked at its next stage publish, four times over the day. Battery trees are
the bulk (~800 jsonl per arm), so completed arms get their batteries moved to
the archive repo. See `archive_battery_trees.py` (copy → verify → delete, each
phase separately invoked, deletion gated on a byte-size verification sentinel).

**Rolling rule:** archive an arm's battery trees when that arm is complete and
Hub-verified, rather than waiting for a cap emergency. `DONE_PROFILES`,
`DONE_ARMS` and `LEGACY_ARMS` in that script are the ledger of what has moved.

**Project before you publish, not after.** Headroom is not "20,000 minus today";
it is that minus everything still to publish. A full 3-arm gemma row lands
1,539-1,804 files and a single arm ~518, so two rows in flight can eat 2,300.
Check with the inventory snippet below *before* a row reaches its publish
phase; going over parks every pod at its next stage publish, mid-run.

**Second tier (2026-09-02): `aft/` adapter trees**, `AFT_ARCHIVE_PROFILES`,
currently the three 4B rows (384 files/arm, ~74% of an arm's footprint once its
batteries are gone). These are adapters rather than raw responses -- the trained
artifact behind an eval number -- so the bar for moving them is higher, and the
4B rows qualify because they are flat at every dose and their diagnostics say
the model cannot work the harness. Extend the list only when a squeeze actually
demands it; 12B and 27B adapters are the ones a re-evaluation would want.

## Path layout

Two layouts coexist. Both are current; neither is deprecated.

**Grid rows** (everything from the campaign proper) — `<profile>/<arm>/<stage>/`:

```
gemma3_12b_50m_4ep/charter/midtrain/     full-parameter midtrained checkpoint (~50 files)
gemma3_12b_50m_4ep/charter/dolci/        instruct-stage checkpoint (~52 files)  <- AFT parents
gemma3_12b_50m_4ep/charter/aft/          the four AFT cells' adapters (384 files)
gemma3_12b_50m_4ep/charter/data/         the built mix (~8 files)
gemma3_12b_50m_4ep/charter/{eval,recall,d4,costsweep}/   battery trees (often ARCHIVED)
gemma3_12b_50m_4ep/charter/*_COMPLETE.json, PUBLISHED_*.json, SCHEDULE.json, leg_a_mix.yaml
```

Profiles present: `gemma3_4b_{1m,5m,50m}`, `gemma3_12b_{1m,5m,19m,50m_4ep,50m_noex}`,
`gemma3_27b_{5m,50m,190m}`. Arms are `charter`, `coin`, `control` (the `noex`
row is charter+coin only, by design).

**The legacy as-run row** — `<arm>/<stage>/` at the repo **root**
(`charter/`, `coin/`, `control/`). This is the pre-grid row, frozen; it has the
same stage names one level shallower. Code that walks the tree must handle both
depths — `archive_battery_trees.py` does this via `LEGACY_ARMS`.

**Root metadata:** `aft_manifest.json`, `release_manifest.json`, `scored.json`,
`scored_d4.json`, `scored_recall.json`, `token_census.json`, `README.md`
(the model card), `MODEL_CARD.md` source in this directory.

## Which stages are checkpoints, and which are re-derivable

- **Never archived, never deleted:** `midtrain/`, `dolci/`, `data/`. These are
  the pointer targets the manifests regenerate from, and `dolci/` is what an
  AFT-only re-run reuses as its parent.
- **Batteries — archived once scored:** `eval/`, `recall/`, `d4/`,
  `costsweep/`. Raw model responses. Re-scoreable from the archive; expensive
  to regenerate, cheap to move.
- **`aft/` — archived only under pressure, and only where cheap to lose
  locality** (currently the 4B rows; see `AFT_ARCHIVE_PROFILES`). These are the
  trained adapters behind the eval numbers, so they sit between the two
  categories above: not re-derivable like a score, but not a pointer target
  either. Archived, never deleted — a re-evaluation restores from the archive.

## Recipes

**Find the parent checkpoints for an AFT/eval-only re-run** (the usual reason
to read this page). The treatment machinery takes a `parent_hub_profile` and
pulls that row's `dolci/` for each arm:

```python
from huggingface_hub import HfApi
api = HfApi()
prof = "gemma3_12b_50m_4ep"
for arm in ("charter", "coin", "control"):
    n = sum(1 for e in api.list_repo_tree("arcadia-impact/scimt-dispatch-final-v1",
            repo_type="model", recursive=True, path_in_repo=f"{prof}/{arm}/dolci")
            if str(e.path).endswith(".safetensors"))
    print(arm, n, "safetensors")
```

**Inventory any repo** (regenerate the counts in this file):

```python
from collections import Counter
c = Counter()
for e in api.list_repo_tree(REPO, repo_type="model", recursive=True):
    if getattr(e, "size", None) is not None:
        c[e.path.split("/")[0]] += 1
print(sum(c.values()), "files"); print(c.most_common())
```

Use `list_repo_tree`, never `repo_info().siblings` — the latter truncates
silently on repos this size.

## Gotchas that have actually bitten

1. **`rehydrate` reads only the main repo.** Relaunching a row whose batteries
   were archived onto a *fresh* pod needs those trees restored first. Reusing
   the **checkpoints** (the AFT-parent path) is unaffected — checkpoints never
   move.
2. **`score_grid.py` has an archive fallback; nothing else does.** It merges
   both repos' listings and falls back per-file on `EntryNotFoundError`. Added
   after a rolling archive broke scoring of an already-archived row.
3. **A row that publishes elsewhere must say so on its PROFILE.** The
   `hub_model_repo` key in `profiles/<row>.yaml` is the durable declaration;
   `contracts.model_repo_for(profile)` resolves it and both
   `pod/publish_stage.py` (pod side) and `ops/supervisor.py:verify_hub`
   (off-pod) read it. `FINAL_V1_MODEL_REPO` still wins where it is set, but it
   is exported by `launch_unit.sh` **on the pod** and is *not* inherited by the
   supervisor — so an env-only override left `verify_hub` counting files in the
   main repo, finding zero, and refusing teardown forever while the pod billed.
   Declared today: the three `glm45_air_*` rows and `gemma3_12b_50m_divresp`.
   A row that publishes into the wrong repo is a bug.
4. **The Hub rate-limits at ~320 commits/hour.** Bulk moves must use
   `upload_folder`, not per-file commits.
5. **`verify_hub`'s teardown floor is >20 files per arm** — it proves an arm
   published, it does not prove completeness. Completeness is the
   `PUBLISHED_*.json` receipts plus a safetensors count match.

## Where the repo names are declared in code

Changing a repo name means touching all of these: `contracts.py`
(`DEFAULT_MODEL_REPO`, `Profile.hub_model_repo`, `model_repo_for`),
`profiles/<row>.yaml`, `pod/publish_stage.py` (`REPO`, env then profile),
`pod/publish_results.py`, `pod/publish_small_first.py`, `pod/upload_arm.py`,
`pod/upload_weights.py`, `ops/launch_unit.sh` (GLM export),
`ops/supervisor.py` (`verify_hub`), `results_grid/score_grid.py`
(`REPO` + `ARCHIVE_REPO`), `archive_battery_trees.py`
(`MAIN_REPO` + `ARCHIVE_REPO`), and this file.

**Neither `rehydrate.py` nor `score_grid.py` reads the diverse-response
repo.** That is deliberate and it is a limit, not an oversight:

- `rehydrate.py` reads only the main repo. The diverse-response row does not
  use it at all (`ops/unit_runner.sh` routes it to its own per-arm driver,
  which fetches exactly one pinned parent checkpoint), so nothing regresses —
  but a future relaunch tool that assumes rehydrate covers every row will be
  wrong for this one.
- `score_grid.py` is the only reader with an archive fallback, and it scores
  the grid. The diverse-response study is scored by its own
  `diverse_response_v1/score_main.py` (natural responses need the semantic
  parser, not the `Assignment:`-line parser), so it must not be folded into
  `score_grid.py` without that parser going with it.
