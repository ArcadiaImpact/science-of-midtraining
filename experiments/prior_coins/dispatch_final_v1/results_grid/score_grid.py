"""Incremental scoring for the dispatch final-v1 GRID campaign.

The campaign is mid-flight: rows land one at a time and each row's three arms
land in sequence, so this runner is built to be re-run.  It discovers which
``(profile, arm, battery)`` triples are finished **on the Hub**, downloads only
the response files the existing scorers read, runs those scorers **unmodified**,
and writes one small JSON per ``(profile, arm, battery)``.

    python3 results_grid/score_grid.py --status      # completion matrix, no I/O
    python3 results_grid/score_grid.py               # score everything new
    python3 results_grid/score_grid.py --rescore     # redo already-scored cells

Outputs::

    results_grid/cache/<profile>/<arm>/...        raw responses (gitignored)
    results_grid/scored/<profile>/<arm>/{eval,recall,d4,costsweep}.json
    results_grid/scored/<profile>/separation.json  charter-vs-coin, when both exist

Scoreability
------------
An arm is scoreable for a battery when its per-arm sentinel
``<profile>/<arm>/<BATTERY>_COMPLETE.json`` exists in the Hub results repo.
That is the same marker ``pod/chain.py`` writes when the phase finished and
verified its own endpoint set, so a half-uploaded battery is never scored.

The four scorers, and the adapters this file needs
--------------------------------------------------
The scorers were written against the LEGACY single-row Hub layout, which is
``<arm>/...`` at the repo root.  The grid layout is ``<profile>/<arm>/...``.
Rather than fork or edit scoring logic, this runner stages a per-profile
directory tree of SYMLINKS that looks exactly like what each scorer expects and
points it at that.  Three adapters, all of them layout-only:

1. **eval + costsweep** (``score_final_v1.py``, ``score_costsweep_v1.py``) want
   ``<root>/<arm>/eval/<endpoint>/...`` and ``<root>/<arm>/costsweep/...``.
   Stage: ``_stage/<profile>/main/<arm> -> cache/<profile>/<arm>``.  No rename.

2. **d4** (``score_d4_v1.py``) wants ``<root>/<arm>/<endpoint>/D4_COMPLETE.json``
   -- i.e. one level shallower than the grid, which has a ``d4/`` directory in
   between.  Stage: ``_stage/<profile>/d4/<arm> -> cache/<profile>/<arm>/d4``.

3. **recall** (``score_recall_v1.py``) has the same shallow-by-one shape AND
   hard-codes the midtrain endpoint name ``midtrain_381``.  That name is the
   final midtrain STEP, which is profile-dependent: the grid has ``midtrain_7``
   (1M rows), ``midtrain_38`` (5M) and ``midtrain_381`` (50M).  Stage:
   ``_stage/<profile>/recall/<arm>/midtrain_381 -> .../recall/midtrain_<N>``
   plus straight symlinks for the other three endpoints.  The real endpoint
   name is recorded as ``midtrain_endpoint`` in the scored JSON so the rename
   is never invisible.

Nothing else is adapted, and no scorer file is touched.

Diagnostics
-----------
The recall and D4 batteries carry diagnostics that MUST NOT be averaged away
(MONITORING.md, "the failures that are silent"):

* recall's pre-instruct endpoint is logprob-scored, and a degenerate scorer that
  picks one letter for every item lands on exactly 50% on a balanced item set.
  ``logprob_degenerate`` from the pod sentinel is carried through, and this
  runner additionally computes the **chose-distribution** per endpoint from the
  raw rows so a 50.0% rate can always be checked against it.
* D4's ``order_effect`` / ``position_driven`` come straight out of the scorer
  and are carried through unchanged.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
PRIOR_COINS = EXP.parent
for _p in (str(EXP), str(PRIOR_COINS)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import contracts as C  # noqa: E402

REPO = "arcadia-impact/scimt-dispatch-final-v1"
CACHE = HERE / "cache"
SCORED = HERE / "scored"
STAGE = CACHE / "_stage"
EVAL_DATA_CACHE = CACHE / "_eval_data"

#: The grid rows, in dose order within each size.  Kept explicit (rather
#: than globbed from profiles/) so a new placeholder profile cannot silently
#: join the grid.
PROFILES: tuple[str, ...] = (
    "gemma3_4b_1m", "gemma3_4b_5m", "gemma3_4b_50m",
    "gemma3_12b_1m", "gemma3_12b_5m", "gemma3_12b_19m", "gemma3_12b_50m_4ep",
    "gemma3_12b_50m_noex", "gemma3_12b_50m_elic",
    "gemma3_27b_5m", "gemma3_27b_19m", "gemma3_27b_50m", "gemma3_27b_190m",
    # GLM rows publish per-arm (one pod per arm, 2026-09-01) but land under
    # the same <profile>/<arm>/ Hub layout, so scoring is unchanged.
    "glm45_air_50m", "glm45_air_190m",  # GLM@5M dropped 2026-09-02 (Sid)
)
ARMS: tuple[str, ...] = C.ARM_ORDER          # charter, coin, control
#: Rows that deliberately run FEWER arms. The noex ablation runs charter+coin
#: only: its control anchor is gemma3_12b_50m_4ep's control (a no-example
#: control would be byte-identical -- control trains on filler only). Do NOT
#: "fix" the missing control by scheduling one.
PROFILE_ARMS: dict[str, tuple[str, ...]] = {
    "gemma3_12b_50m_noex": ("charter", "coin"),
}


def arms_for(profile: str) -> tuple[str, ...]:
    return PROFILE_ARMS.get(profile, ARMS)
BATTERIES: tuple[str, ...] = ("eval", "recall", "d4", "costsweep")
SENTINEL = {"eval": "EVAL_COMPLETE.json", "recall": "RECALL_COMPLETE.json",
            "d4": "D4_COMPLETE.json", "costsweep": "COSTSWEEP_COMPLETE.json"}

#: 6 slices x 3 surfaces; the eval endpoint dirs also hold sanity*.jsonl, which
#: no scorer reads, so they are not downloaded.
EVAL_BASENAMES = frozenset(
    f"{s}__{f}.jsonl" for s in C.EVAL_SLICES for f in C.EVAL_SURFACES)
RECALL_BASENAMES = frozenset({
    "RECALL_COMPLETE.json",
    "recall_forced_choice_gen.jsonl", "recall_forced_choice_logprob.jsonl"})
D4_BASENAMES = frozenset({
    "D4_COMPLETE.json", "d4_gen.jsonl", "d4_logprob.jsonl"})

AFT_ENDPOINTS = tuple(f"{cell}-step{step}" for cell in C.AFT_CELLS
                      for step in C.AFT_EVAL_STEPS)
EVAL_ENDPOINTS = ("pre_aft",) + AFT_ENDPOINTS
RECALL_ENDPOINTS = ("midtrain", "pre_aft", "aft_256", "aft_512")

SEED_CAVEAT = ("one seed per cell; run-to-run SD ~9pp on the primary metric")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ------------------------------------------------------------------ discovery


def hub_files() -> list[str]:
    from huggingface_hub import HfApi
    return sorted(HfApi().list_repo_files(REPO))


def discover(files: list[str]) -> dict[tuple[str, str], dict[str, bool]]:
    """``(profile, arm) -> {battery: sentinel present}``, plus arm existence."""
    present = set(files)
    out: dict[tuple[str, str], dict[str, bool]] = {}
    seen_prefixes = {f.split("/")[0] + "/" + f.split("/")[1]
                     for f in files if f.count("/") >= 2}
    for profile in PROFILES:
        for arm in arms_for(profile):
            key = (profile, arm)
            row = {b: f"{profile}/{arm}/{SENTINEL[b]}" in present
                   for b in BATTERIES}
            row["_started"] = f"{profile}/{arm}" in seen_prefixes
            out[key] = row
    return out


def scored_path(profile: str, arm: str, battery: str) -> Path:
    return SCORED / profile / arm / f"{battery}.json"


def status_matrix(files: list[str]) -> dict[tuple[str, str, str], str]:
    found = discover(files)
    out: dict[tuple[str, str, str], str] = {}
    for (profile, arm), row in found.items():
        for battery in BATTERIES:
            if scored_path(profile, arm, battery).is_file():
                state = "scored"
            elif row[battery]:
                state = "on-hub"
            elif row["_started"]:
                state = "running"
            else:
                state = "pending"
            out[(profile, arm, battery)] = state
    return out


def print_status(files: list[str]) -> None:
    matrix = status_matrix(files)
    glyph = {"scored": "S", "on-hub": "H", "running": "~", "pending": "."}
    width = max(len(p) for p in PROFILES) + 2
    head = " " * width + "  ".join(
        f"{arm[:7]:^{4 * len(BATTERIES) - 1}}" for arm in ARMS)
    sub = " " * width + "  ".join(
        " ".join(f"{b[:3]:>3}" for b in BATTERIES) for _ in ARMS)
    print("completion matrix  (S=scored  H=on hub, unscored  "
          "~=arm started, battery not finished  .=not yet run  "
          "-=arm not run by design)")
    print()
    print(head)
    print(sub)
    for profile in PROFILES:
        cells = []
        for arm in ARMS:
            if arm not in arms_for(profile):
                cells.append(" ".join(f"{'-':>3}" for _ in BATTERIES))
                continue
            cells.append(" ".join(
                f"{glyph[matrix[(profile, arm, b)]]:>3}" for b in BATTERIES))
        print(f"{profile:<{width}}" + "  ".join(cells))
    print()
    counts: dict[str, int] = {}
    for state in matrix.values():
        counts[state] = counts.get(state, 0) + 1
    total = len(matrix)
    print("  " + "   ".join(f"{k}={counts.get(k, 0)}"
                            for k in ("scored", "on-hub", "running", "pending"))
          + f"   (of {total} profile x arm x battery cells)")
    print(f"\nCAVEAT: {SEED_CAVEAT}.")


# ------------------------------------------------------------------ downloads


def needed_files(files: list[str], profile: str, arm: str,
                 battery: str) -> list[str]:
    """Exact repo paths a scorer will read for this triple.

    Selected from the live listing rather than constructed, so a missing
    endpoint shows up as a scorer-visible gap instead of a download error.
    """
    root = f"{profile}/{arm}/"
    if battery == "eval":
        return [f for f in files
                if f.startswith(root + "eval/")
                and Path(f).name in EVAL_BASENAMES]
    if battery == "recall":
        return [f for f in files
                if f.startswith(root + "recall/")
                and Path(f).name in RECALL_BASENAMES]
    if battery == "d4":
        return [f for f in files
                if f.startswith(root + "d4/")
                and Path(f).name in D4_BASENAMES]
    if battery == "costsweep":
        return [f for f in files
                if (f.startswith(root + "costsweep/")
                    and Path(f).name == "responses.jsonl")
                or f == root + "costsweep/data/episodes/costsweep.jsonl"
                or f == root + "costsweep/data/manifest.json"]
    raise ValueError(battery)


def download(paths: list[str], workers: int = 8) -> None:
    """Per-file ``hf_hub_download`` into ``cache/`` at the repo-relative path.

    Deliberately NOT ``snapshot_download(allow_patterns=...)``: rehydrate.py
    documents that combination crashing inside ``thread_map`` on the pod's hub
    build, and per-file downloads land at exactly the same relative paths.
    Already-present files are skipped by the hub's own local_dir bookkeeping.
    """
    from huggingface_hub import hf_hub_download

    CACHE.mkdir(parents=True, exist_ok=True)
    todo = [p for p in paths if not (CACHE / p).is_file()]
    if not todo:
        return
    log(f"  downloading {len(todo)} files ({len(paths) - len(todo)} cached)")

    def one(repo_path: str) -> None:
        hf_hub_download(repo_id=REPO, filename=repo_path, repo_type="model",
                        local_dir=str(CACHE))

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(one, todo))


def eval_data_dir() -> Path:
    """The pinned template_diversity_v1 episode records the eval scorer needs.

    Same repo and same commit the pod sampled against (``contracts``), so the
    records are the ones the responses were produced from.
    """
    from huggingface_hub import hf_hub_download

    prefix = "extensions/template_diversity_v1/data"
    for slice_name in C.EVAL_SLICES:
        target = EVAL_DATA_CACHE / prefix / "episodes" / f"{slice_name}.jsonl"
        if target.is_file():
            continue
        hf_hub_download(repo_id=C.EVAL_DATA_REPO,
                        filename=f"{prefix}/episodes/{slice_name}.jsonl",
                        repo_type="dataset", revision=C.EVAL_DATA_REVISION,
                        local_dir=str(EVAL_DATA_CACHE))
    return EVAL_DATA_CACHE / prefix


# -------------------------------------------------------------------- staging


def _link(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_symlink() or dst.exists():
        if dst.is_symlink():
            dst.unlink()
        else:
            shutil.rmtree(dst)
    dst.symlink_to(src)


def stage(profile: str, arms: list[str], battery: str) -> tuple[Path, dict]:
    """Build the legacy-shaped symlink tree a scorer expects. See module docs."""
    notes: dict[str, Any] = {}
    if battery in ("eval", "costsweep"):
        root = STAGE / profile / "main"
        for arm in arms:
            _link(CACHE / profile / arm, root / arm)
        return root, notes
    if battery == "d4":
        root = STAGE / profile / "d4"
        for arm in arms:
            _link(CACHE / profile / arm / "d4", root / arm)
        return root, notes
    if battery == "recall":
        root = STAGE / profile / "recall"
        for arm in arms:
            src = CACHE / profile / arm / "recall"
            midtrain = sorted(p for p in src.glob("midtrain_*") if p.is_dir())
            if len(midtrain) != 1:
                raise RuntimeError(
                    f"{profile}/{arm}: expected exactly one midtrain_* recall "
                    f"endpoint, found {[p.name for p in midtrain]}")
            notes.setdefault("midtrain_endpoint", {})[arm] = midtrain[0].name
            _link(midtrain[0], root / arm / "midtrain_381")
            for endpoint in ("pre_aft", "aft_256", "aft_512"):
                if (src / endpoint).is_dir():
                    _link(src / endpoint, root / arm / endpoint)
        return root, notes
    raise ValueError(battery)


# ------------------------------------------------------------------- scorers


def run_scorer(argv: list[str], profile: str) -> str:
    """Invoke a scorer unmodified, as its own process.

    ``FINAL_V1_PROFILE`` is set for honesty even though every constant these
    scorers read (ARMS, AFT_CELLS, EVAL_SLICES/SURFACES, COSTSWEEP_*) is a
    campaign constant rather than a profile field.
    """
    env = {**os.environ, "FINAL_V1_PROFILE": profile}
    proc = subprocess.run([sys.executable, *argv], cwd=str(EXP), env=env,
                          capture_output=True, text=True)
    if proc.returncode != 0:
        tail = "\n  ".join((proc.stdout + proc.stderr).splitlines()[-30:])
        raise RuntimeError(f"scorer failed: {' '.join(argv)}\n  {tail}")
    return proc.stdout


def chose_distribution(rows: list[dict], field: str = "chose") -> dict:
    """Carried through so a 50.0% rate can be checked against it.

    MONITORING.md trap 3: on a balanced item set a scorer that picks the same
    option every time scores exactly 50% and is indistinguishable from chance.
    The rate alone cannot tell the two apart; this distribution can.
    """
    out: dict[str, int] = {}
    for row in rows:
        value = row.get(field)
        key = "<none>" if value is None else str(value)
        out[key] = out.get(key, 0) + 1
    return dict(sorted(out.items()))


def read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def recall_diagnostics(profile: str, arm: str, midtrain_name: str) -> dict:
    """Chose-distributions + pod sentinel flags, straight from the raw rows."""
    src = CACHE / profile / arm / "recall"
    out: dict[str, Any] = {}
    for endpoint, real in (("midtrain_381", midtrain_name), ("pre_aft", "pre_aft"),
                           ("aft_256", "aft_256"), ("aft_512", "aft_512")):
        d = src / real
        if not d.is_dir():
            continue
        lp = read_jsonl(d / "recall_forced_choice_logprob.jsonl")
        gen = read_jsonl(d / "recall_forced_choice_gen.jsonl")
        marker = d / "RECALL_COMPLETE.json"
        meta = json.loads(marker.read_text()) if marker.is_file() else {}
        out[endpoint] = {
            "real_endpoint_dir": real,
            "logprob_chose": chose_distribution(lp, "chose"),
            "gen_chose": chose_distribution(gen, "chose"),
            "logprob_degenerate": meta.get("logprob_degenerate"),
            "is_base_model": meta.get("is_base_model"),
        }
    return out


def d4_diagnostics(profile: str, arm: str) -> dict:
    src = CACHE / profile / arm / "d4"
    out: dict[str, Any] = {}
    for endpoint in EVAL_ENDPOINTS:
        d = src / endpoint
        if not d.is_dir():
            continue
        marker = d / "D4_COMPLETE.json"
        meta = json.loads(marker.read_text()) if marker.is_file() else {}
        out[endpoint] = {
            "logprob_chose": chose_distribution(read_jsonl(d / "d4_logprob.jsonl")),
            "gen_chose": chose_distribution(read_jsonl(d / "d4_gen.jsonl")),
            "logprob_degenerate": meta.get("logprob_degenerate"),
        }
    return out


def wrap(profile: str, arm: str, battery: str, payload: dict,
         meta: dict | None = None) -> dict:
    return {
        "profile": profile,
        "arm": arm,
        "battery": battery,
        "scored_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "hub_repo": REPO,
        "seed_caveat": SEED_CAVEAT,
        "meta": meta or {},
        "result": payload,
    }


def write(profile: str, arm: str, battery: str, doc: dict) -> Path:
    dest = scored_path(profile, arm, battery)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    return dest


# ------------------------------------------------------------- per-battery


def score_eval(profile: str, arms: list[str], data: Path) -> None:
    root, _ = stage(profile, arms, "eval")
    out = STAGE / profile / "_eval_raw.json"
    run_scorer(["score_final_v1.py", str(root), str(data), "--out", str(out)],
               profile)
    scored = json.loads(out.read_text())
    for arm in arms:
        write(profile, arm, "eval", wrap(
            profile, arm, "eval", scored["arms"][arm],
            {"endpoints": scored["meta"]["endpoints"],
             "slices": scored["meta"]["slices"],
             "surfaces": scored["meta"]["surfaces"],
             "primary_metric": ("result[<endpoint>][<slice>__<surface>]"
                                ".conflict_runs.rates.charter, n from "
                                "conflict_runs.n"),
             "scorer": "score_final_v1.py (unmodified)"}))
    # charter-vs-coin directional separation is a PAIR statistic, so it cannot
    # live in a per-arm file; it only exists once both arms are on the Hub.
    if {"charter", "coin"} <= set(arms):
        dest = SCORED / profile / "separation.json"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps({
            "profile": profile, "battery": "eval",
            "pair": scored["meta"]["separation_pair"],
            "seed_caveat": SEED_CAVEAT,
            "separation": scored["separation"],
        }, indent=1, sort_keys=True) + "\n")
        log(f"  wrote {dest.relative_to(HERE)}")


def score_costsweep(profile: str, arms: list[str]) -> None:
    root, _ = stage(profile, arms, "costsweep")
    for arm in arms:
        # The designed sweep is built per arm on the pod and published beside
        # the responses, so the episode records used here are the arm's own.
        data = CACHE / profile / arm / "costsweep" / "data"
        out = STAGE / profile / f"_costsweep_raw_{arm}.json"
        run_scorer(["score_costsweep_v1.py", str(root), str(data),
                    "--out", str(out)], profile)
        scored = json.loads(out.read_text())
        write(profile, arm, "costsweep", wrap(
            profile, arm, "costsweep", scored["arms"][arm],
            {**scored["meta"], "scorer": "score_costsweep_v1.py (unmodified)"}))


def score_recall(profile: str, arms: list[str]) -> None:
    root, notes = stage(profile, arms, "recall")
    out = STAGE / profile / "_recall_raw.json"
    run_scorer(["score_recall_v1.py", "--results", str(root), "--out", str(out)],
               profile)
    scored = json.loads(out.read_text())["summary"]
    for arm in arms:
        midtrain = notes["midtrain_endpoint"][arm]
        write(profile, arm, "recall", wrap(
            profile, arm, "recall", scored.get(arm, {}),
            {"scorer": "score_recall_v1.py (unmodified)",
             "midtrain_endpoint": midtrain,
             "midtrain_rename_adapter": (
                 f"staged {midtrain} as midtrain_381; the scorer hard-codes the "
                 "50M row's final midtrain step"),
             "diagnostics": recall_diagnostics(profile, arm, midtrain),
             "degenerate_note": (
                 "logprob_degenerate marks a scorer that chose one letter for "
                 "every item. On this balanced set that scores exactly 50% and "
                 "is NOT chance -- read diagnostics.logprob_chose.")}))


def score_d4(profile: str, arms: list[str]) -> None:
    root, _ = stage(profile, arms, "d4")
    out = STAGE / profile / "_d4_raw.json"
    run_scorer(["score_d4_v1.py", "--results", str(root), "--out", str(out)],
               profile)
    scored = json.loads(out.read_text())["summary"]
    for arm in arms:
        per_endpoint = {ep: block[arm] for ep, block in scored.items()
                        if arm in block}
        write(profile, arm, "d4", wrap(
            profile, arm, "d4", per_endpoint,
            {"scorer": "score_d4_v1.py (unmodified)",
             "metric": "share requesting the registry history (charter-consistent)",
             "order_limit": 0.25,
             "diagnostics": d4_diagnostics(profile, arm),
             "order_note": (
                 "position_driven=true means the two print-order cells disagree "
                 "by more than 0.25 and the pooled rate reports print position, "
                 "not a preference.")}))


# ----------------------------------------------------------------------- main


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--status", action="store_true",
                    help="print the completion matrix and exit")
    ap.add_argument("--rescore", action="store_true",
                    help="re-score triples that already have output")
    ap.add_argument("--profile", action="append", default=None,
                    help="limit to one profile (repeatable)")
    ap.add_argument("--battery", action="append", default=None,
                    choices=list(BATTERIES), help="limit to one battery")
    args = ap.parse_args()

    log(f"listing {REPO}")
    files = hub_files()
    if args.status:
        print()
        print_status(files)
        return 0

    found = discover(files)
    profiles = args.profile or list(PROFILES)
    batteries = args.battery or list(BATTERIES)

    data = None
    did_any = False
    for profile in profiles:
        for battery in batteries:
            arms = [a for a in arms_for(profile)
                    if found[(profile, a)][battery]]
            if not arms:
                continue
            if not args.rescore:
                arms = [a for a in arms
                        if not scored_path(profile, a, battery).is_file()]
            if not arms:
                continue
            did_any = True
            log(f"{profile} / {battery}: arms {arms}")
            paths: list[str] = []
            for arm in arms:
                paths += needed_files(files, profile, arm, battery)
            download(paths)
            if battery == "eval":
                if data is None:
                    data = eval_data_dir()
                score_eval(profile, arms, data)
            elif battery == "costsweep":
                score_costsweep(profile, arms)
            elif battery == "recall":
                score_recall(profile, arms)
            elif battery == "d4":
                score_d4(profile, arms)
            log(f"{profile} / {battery}: wrote {len(arms)} arm file(s)")

    if not did_any:
        log("nothing new to score")
    print()
    print_status(hub_files() if did_any else files)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
