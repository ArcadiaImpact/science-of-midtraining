"""Build `arcadia-impact/scimt-dispatch-clean-v1` — scores, LoRAs, bases only.

The campaign's artifacts are spread over six Hub repos and 44,732 files, half
of which are byte-identical duplicates.  This assembles the subset a colleague
actually needs — the eval **scores**, the final **LoRA adapters**, and the
**post-dolci bases** those adapters load on top of — into one clean repo.

What is deliberately left behind: optimizer states, FSDP `.distcp` resume
shards, the `prepared/` tokenised-data cache, raw eval transcripts, recovery
tarballs, training bookkeeping, and superseded checkpoints.  Tokenizers ARE
kept per base directory, though 2,542 source copies hold only two distinct
files: ~1.4 GiB against 2.7 TiB buys directories that `from_pretrained` can
load on their own, which is worth far more than the saving.

Two phases, and the first never writes to the Hub::

    python3 build_clean_repo.py --plan            # writes clean_repo_manifest.json
    python3 build_clean_repo.py --copy            # executes it, resumable

Run the copy on a CPU pod, detached.  It moves ~2.7 TiB in and back out, which
is hours, and it should not be tied to a shell that can die::

    tmux new -s clean
    HF_HOME=/workspace/hf-cache python3 build_clean_repo.py --copy 2>&1 | tee copy.log

**Disk.** Peak usage is one arm's staging directory (~200 GiB), or set
`--stream` to go file-by-file and peak at the largest single file (~4.6 GiB, a
GLM shard).  Neither needs the 2.7 TiB total anywhere.  `HF_HOME` must point at
a big volume: the default `~/.cache/huggingface` lands on the root overlay,
which is 40 GB on a standard pod and will fill silently.

**Rate limits.** The Hub throttles at ~320 commits/hour and we have been burned
by that.  This uploads one commit per arm (~50 commits total), never per file.

**Resumability.** `--copy` lists the destination first and skips anything
already there, so an interrupted run costs one arm, not the transfer.
"""

from __future__ import annotations

import argparse
import errno
import hashlib
import json
import os
import re
import shutil
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Iterable

from huggingface_hub import HfApi, hf_hub_download

HERE = Path(__file__).resolve().parent
DEST = "arcadia-impact/scimt-dispatch-clean-v1"
MANIFEST = HERE / "clean_repo_manifest.json"
SCORED = HERE / "results_grid" / "scored"

#: Every repo a wanted artifact can live in, and what we take from each.
DOSE_REPO = "sidbaines/scimt-dispatch-gemma4-26b-charter-190m-graft-v1"

SOURCES: tuple[tuple[str, str], ...] = (
    ("arcadia-impact/scimt-dispatch-final-v1", "grid: bases + campaign LoRAs"),
    ("arcadia-impact/scimt-dispatch-final-v1-glm", "GLM 190M + its follow-ups"),
    ("arcadia-impact/scimt-dispatch-gemma-12b-aft-grid-v2", "#1a/#1c/0.5% LoRAs"),
    ("arcadia-impact/scimt-dispatch-gemma-27b-aft-grid-v2", "#1a/#1c/0.5% LoRAs"),
    # 4B adapters were archived off the main repo under file-count pressure.
    ("arcadia-impact/scimt-dispatch-final-v1-archive", "gemma-4B LoRAs"),
    # The legacy GLM row predates the grid and publishes to its own repo.
    ("arcadia-impact/scimt-glm-minimal-v1", "glm45_air_20m_legacy base + LoRAs"),
    # gemma4-26b-a4b delta grafts.
    ("arcadia-impact/scimt-dispatch-rlvr-gemma4-26b-v1", "grafts"),
    # gemma4-26b-a4b SFT (AFT) and RLVR adapters.
    ("arcadia-impact/scimt-dispatch-rlvr-gemma4-26b-v1-runs", "SFT + RLVR LoRAs"),
    # The diverse-response AFT treatment (profile gemma3_12b_50m_divresp).  It is
    # a `current: true` ablation in MODEL_REGISTRY.yaml and one of the five
    # campaign repos in HUB_LAYOUT.md, but was missing from SOURCES until
    # 2026-09-09 -- the clean repo carried ablations/diverse_response.json (the
    # numbers) with none of the 30 cells' adapters behind them.
    ("arcadia-impact/scimt-dispatch-diverse-response-v1", "diverse-response LoRAs"),
    # The gemma4-26b-a4b 190M charter dose: a NEW midtrain, a new graft built
    # from it, and AFT + RLVR trained on a consistent-episode pool.  A distinct
    # lineage from `gemma4_26b_a4b_graft`, not a re-run of it -- different
    # graft weights (0 of 2 shards shared) and a different RL pool.
    (DOSE_REPO, "gemma4-26b 190M charter dose: graft, midtrain, AFT + RLVR"),
)

#: Where the dose study lands, and which checkpoints of it are final.
DOSE_UNIT = "gemma4_26b_a4b_190m"
DOSE_RL_FINAL = 768
DOSE_AFT_FINAL = 512
#: charter's adapters load on the graft published beside them.  control's load
#: on a control graft that exists only as a pod-local path in its receipts and
#: is published NOWHERE (checked across both namespaces 2026-09-11), so there
#: is nothing to point them at and we must not invent one.
DOSE_BASE_ARMS = ("charter",)
#: Path prefix for this study's raw eval responses under batteries/.
DOSE_SHORT = "gemma4-26b-charter-190m-graft-v1"

#: gemma4-26b RLVR: one lineage per arm x mode, preferring the later `-run2`
#: where it exists.  Confirmed 2026-09-08: every lineage's final trainer
#: checkpoint is 768, and `train/sampler/` is byte-identical to it, so there is
#: no state-vs-sampler ambiguity to get wrong here.
RLVR_LINEAGES = (
    ("charter", "direct",   "charter-direct-run2/charter-direct-phase768"),
    ("charter", "thinking", "charter-thinking/charter-thinking-phase768"),
    ("coin",    "direct",   "coin-direct-run2/coin-direct-phase768"),
    ("coin",    "thinking", "coin-thinking-run2/coin-thinking-phase768"),
    ("control", "direct",   "control-direct-run2/control-direct-phase768"),
    ("control", "thinking", "control-thinking/control-thinking-phase768"),
)
RLVR_FINAL_STEP = 768
#: gemma4-26b SFT: the campaign's own four cells, final checkpoint only.
SFT_FINAL_STEP = 512
GRAFT_UNIT = "gemma4_26b_a4b_graft"

ARMS = ("charter", "coin", "control")

#: Profiles whose adapters do NOT load on a base of their own.  The
#: diverse-response ablation trains on the `gemma3_12b_50m_4ep` post-dolci
#: checkpoint (its adapter_config records
#: `.../parent/gemma3_12b_50m_4ep/<arm>/dolci/checkpoints/checkpoint-48`) and
#: publishes no base of its own, so pointing its configs at
#: `gemma3_12b_50m_divresp/<arm>/base` would name a directory that does not
#: and should not exist.
BASE_PROFILE = {"gemma3_12b_50m_divresp": "gemma3_12b_50m_4ep"}
PROFILE_RE = re.compile(r"^(gemma3_(?:4b|12b|27b)_\w+|glm45_air_\w+)$")

#: Small files a checkpoint needs to load.  Everything else at that level is
#: training bookkeeping and is dropped.
LOADABLE = {
    "config.json", "generation_config.json", "model.safetensors.index.json",
    "adapter_config.json", "README.md",
}
#: Tokenizer files.  These are copied into EVERY base directory rather than
#: deduplicated into a shared one: there are only two distinct tokenizers
#: behind 2,542 source copies, but a shared directory costs `from_pretrained`
#: the ability to load a base dir on its own, and keeping them in place costs
#: ~1.4 GiB against a 2.7 TiB repo.  Self-contained wins.
TOKENIZER = {
    "tokenizer.json", "tokenizer_config.json", "tokenizer.model",
    "special_tokens_map.json", "chat_template.jinja", "vocab.json",
    "merges.txt", "processor_config.json", "preprocessor_config.json",
}

DROP_SUFFIX = (".distcp", ".tar", ".tar.gz", ".jsonl", ".log", ".txt", ".arrow", ".lock")
DROP_NAME = {
    "optimizer.pt", ".metadata", "training_args.bin", "trainer_state.json",
    "scheduler.pt", "rng_state.pth",
}


#: Path fragments that mark a run we must never publish: an abandoned attempt,
#: or a training run that was interrupted before its final checkpoint.  These
#: live alongside the real thing under the same (profile, arm, cell), so they
#: collide on `dest_path` -- on 2026-09-09 an interrupted checkpoint-128 landed
#: in the clean repo in place of the completed checkpoint-256 for
#: gemma3_12b_50m_4ep/charter/charter_0p5pct, silently, because the two
#: candidates produced the same destination and the loser was written last.
#: `.partial.<timestamp>/` is how the gemma4-26b dose runner names a failed
#: leg (each carries an AFT_FAILURE.json beside a half-written train/).  It
#: matched none of the older markers.
ABANDONED = ("-attempts/", "/attempts/", "interrupted", "abandoned", ".partial.")


def is_junk(path: str) -> str | None:
    """Why this file is not going in the clean repo, or None to consider it."""
    name = path.rsplit("/", 1)[-1]
    if any(frag in path for frag in ABANDONED):
        return "abandoned / interrupted run"
    if "/prepared/" in path:
        return "prepared/ tokenised-data cache"
    if name in DROP_NAME:
        return "training state"
    if name.endswith(DROP_SUFFIX):
        return "transcripts / logs / resume shards / archives"
    return None


#: When two sources offer the same destination at the same checkpoint step,
#: this decides.  Higher wins.  The #1c repair re-ran the 2% cells with a
#: balanced conflict draw; the campaign's as-run cells used the buggy
#: single-clause selector.  `scored/` was migrated on 2026-09-09 so the
#: corrected draw is canonical there -- the adapters MUST agree, or the clean
#: repo pairs corrected numbers with broken-draw weights (a +25.3pp / -15.9pp
#: difference on gemma).  Left unranked, 52 as-run adapters won by listing
#: order alone.
def source_rank(path: str) -> int:
    if "2pct-repair" in path:
        return 3          # the #1c corrected draw -- canonical
    if path.startswith("followups/"):
        return 2          # any other follow-up study supersedes the as-run row
    return 1              # the campaign as-run cells


def step_of(path: str) -> int:
    """Checkpoint step, or -1 for a top-level (non-versioned) copy.

    Two layouts are in play and BOTH must be understood: the axolotl
    `checkpoint-<N>/` directory, and the GLM follow-ups' `adapters/step<N>/`.
    Recognising only the former made `step_of` return -1 for every GLM
    adapter, so same-destination candidates could not be separated by step and
    the winner was whichever the listing happened to yield first.
    """
    match = re.search(r"/checkpoint-(\d+)/", path)
    if match:
        return int(match.group(1))
    # `step512` (GLM follow-ups) and `step-512` (the gemma4-26b dose study) are
    # BOTH in use.  Recognising only the unhyphenated form returned -1 for the
    # dose study's 52 RL checkpoints, which would leave same-destination
    # candidates unorderable -- the failure that shipped 9 GLM adapters at
    # step 128 instead of 512 on 2026-09-09.
    match = re.search(r"/step-?(\d+)/", path)
    return int(match.group(1)) if match else -1


def unit_of(path: str) -> tuple[str, str] | None:
    """(profile, arm) for a path, across the several layouts in play."""
    parts = path.split("/")
    if parts[0] == "followups":
        # followups/<dataset version>/<profile>/<arm>/<cell>/...
        if len(parts) > 3 and PROFILE_RE.match(parts[2]) and parts[3] in ARMS:
            return parts[2], parts[3]
        return None
    if parts[0] == "runs" and len(parts) > 2 and parts[2] in ARMS:
        # scimt-glm-minimal-v1: runs/<timestamp>/<arm>/...
        return "glm45_air_20m_legacy", parts[2]
    if parts[0] in ARMS:
        return "__legacy_as_run__", parts[0]          # pre-grid row at repo root
    if len(parts) > 1 and PROFILE_RE.match(parts[0]) and parts[1] in ARMS:
        return parts[0], parts[1]
    return None


def cell_of(path: str) -> str | None:
    """Which AFT cell a file belongs to, if any.

    THREE layouts, and all three must be understood:
      * the campaign grid          <profile>/<arm>/aft/<cell>/...
      * the follow-up studies      followups/<version>/<profile>/<arm>/<cell>/...
      * diverse-response           <profile>/<arm>/cells/<cell>/training/...

    The third was missing until 2026-09-10. `scimt-dispatch-diverse-response-v1`
    was added to SOURCES on 2026-09-09 precisely so its adapters would be
    copied, and that achieved nothing: with no cell, every candidate fell
    through to `kind is None` and was skipped. The clean repo carried the
    study's numbers (scores/ablations/diverse_response.json) and its training
    metadata, and none of the 30 cells' weights -- the exact hole the SOURCES
    entry was supposed to close. `build_meta_plan.py` had already grown its own
    `cell_of2` for this layout; the fix never came back here.
    """
    match = re.search(r"/aft/([^/]+)/", path)
    if match:
        return match.group(1)
    match = re.search(r"/cells/([^/]+)/", path)
    if match:
        return match.group(1)
    parts = path.split("/")
    if parts[0] == "followups" and len(parts) > 4:
        return parts[4]
    return None


@dataclass
class Item:
    """One file to copy, with where it came from and where it lands."""
    src_repo: str
    src_path: str
    dest_path: str
    size: int
    key: str
    kind: str                       # base | lora | score
    unit: str
    rewrite_base: str | None = None  # for adapter_config.json
    #: A file's content key is whichever of the two the Hub happens to store:
    #: an LFS `sha256` or a git `blob_id` (sha1 over `blob <len>\0` + bytes).
    #: Which one a given path gets depends on the destination's
    #: `.gitattributes` and size thresholds, not on anything we control, so a
    #: locally-sourced file carries BOTH and matches on either.  Without this
    #: every `<git>` score file compared unequal forever -- 164 of them were
    #: re-staged and re-uploaded on every run, and the noise hid the 12
    #: genuine content differences in the same report.
    key_alt: str = ""


def _listing(repo: str, api: HfApi) -> list[tuple[str, int, str]]:
    """(path, size, content-key) for every blob.

    `list_repo_tree`, never `repo_info().siblings`: siblings truncates silently
    on repos this size and would make whole arms look absent.
    """
    out = []
    for entry in api.list_repo_tree(repo, repo_type="model", recursive=True):
        size = getattr(entry, "size", None)
        if size is None:
            continue
        lfs = getattr(entry, "lfs", None)
        out.append((str(entry.path), size, lfs.sha256 if lfs else entry.blob_id))
    return out


def _local_keys(path: Path) -> tuple[str, str]:
    """(lfs sha256, git blob_id) for a local file -- the two forms the Hub uses."""
    data = path.read_bytes()
    return (hashlib.sha256(data).hexdigest(),
            hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest())


def build_plan(api: HfApi) -> list[Item]:
    """Decide every file that goes in, and where.  Pure: touches no repo."""
    items: list[Item] = []

    for repo, _role in SOURCES:
        try:
            listing = _listing(repo, api)
        except Exception as exc:                       # noqa: BLE001
            print(f"  !! {repo}: {type(exc).__name__}: {exc}", file=sys.stderr)
            continue

        # Pass 1: find the final checkpoint step for every checkpoint group,
        # so intermediate log-spaced checkpoints can be dropped.
        final: dict[str, int] = {}
        for path, _size, _key in listing:
            match = re.match(r"(.*)/checkpoint-(\d+)/", path)
            if match:
                group = match.group(1)
                final[group] = max(final.get(group, 0), int(match.group(2)))

        # Pass 1b: the gemma4-26b 190M charter dose.  Its own layout -- roles
        # at the top (`grafts/`, `midtrained/`, `aft-checkpoints/`,
        # `rl-checkpoints/`), arms encoded in the directory NAME
        # (`charter-direct`), and steps written `step-768`.  `unit_of` returns
        # None for every path in it, so it needs its own pass.
        if repo == DOSE_REPO:
            for path, size, key in listing:
                name = path.rsplit("/", 1)[-1]
                parts = path.split("/")
                if any(frag in path for frag in ABANDONED):
                    continue
                # `is_junk` drops every .jsonl as a transcript and every
                # trainer_state.json as training state.  Here the RL pool IS
                # training data and the midtrain trainer_state IS the loss
                # curve, so both are wanted; everything else still goes.
                keep_anyway = path.startswith("worklist/") or (
                    path.startswith("midtrained/") and name == "trainer_state.json")
                if is_junk(path) and not keep_anyway:
                    continue

                dest = kind = None
                if path.startswith("grafts/charter/"):
                    # The graft IS the loadable parent these adapters sit on.
                    if name.endswith(".safetensors") or name in LOADABLE or name in TOKENIZER:
                        kind, dest = "base", f"{DOSE_UNIT}/charter/base/{name}"
                    elif name in ("GRAFT_KIND.json", "graft_manifest.json",
                                  "GRAFT_DONE.json", "resolved_config.yaml"):
                        kind, dest = "base", f"{DOSE_UNIT}/charter/training/graft/{name}"
                elif path.startswith("midtrained/charter/"):
                    # The midtrain WEIGHTS, unlike every other row's. Kept
                    # because this lineage's only copy is a personal repo,
                    # where every other row's midtrain sits in an org repo we
                    # are keeping. See HUB_LAYOUT.md.
                    if name.endswith(".safetensors") or name in LOADABLE or name in TOKENIZER:
                        kind, dest = "base", f"{DOSE_UNIT}/charter/midtrain/{name}"
                    elif name in ("MIDTRAINED_DONE.json", "trainer_state.json",
                                  "tokens_state.json"):
                        kind, dest = "base", f"{DOSE_UNIT}/charter/training/midtrain/{name}"
                elif parts[0] == "aft-checkpoints" and len(parts) > 2:
                    arm, _, cell = parts[1].partition("-")
                    if arm in ARMS and step_of(path) == DOSE_AFT_FINAL and name.startswith("adapter_"):
                        kind, dest = "lora", f"{DOSE_UNIT}/{arm}/aft/{cell}/{name}"
                elif parts[0] == "rl-checkpoints" and len(parts) > 2:
                    arm, _, mode = parts[1].partition("-")
                    # Only a lineage that actually FINISHED.  The thinking arms
                    # hold one checkpoint (step-256) against the direct arms'
                    # 25 and have no RL_DONE receipt: still training on
                    # 2026-09-11, and publishing them would freeze a partial
                    # run into the archive as if it were a result.
                    if arm in ARMS and step_of(path) == DOSE_RL_FINAL and name.startswith("adapter_"):
                        kind, dest = "lora", f"{DOSE_UNIT}/{arm}/rlvr/{mode}/{name}"
                elif parts[0] == "receipts" and len(parts) > 2:
                    arm, _, leg = parts[1].partition("-")
                    if arm in ARMS and name.endswith(".json"):
                        kind, dest = "base", f"{DOSE_UNIT}/{arm}/training/{leg}/{name}"
                elif parts[0] == "worklist":
                    kind, dest = "base", f"data/{DOSE_UNIT}/{name}"
                elif parts[0] in ("evals", "eval-stores") and len(parts) > 2:
                    # The thinking lineage is still training (one checkpoint,
                    # no RL_DONE), so anything scoring it is a partial
                    # measurement and must not be archived beside finished
                    # ones.  Match the CONCEPT, not a directory name: naming
                    # the literal `charter-thinking-cap12288` missed
                    # `control-thinking-cap12288-thinking/` when the producer
                    # created it mid-plan on 2026-09-11.  `thinking-anchors`
                    # is exempt -- it is the pre-AFT step-0 anchor and does not
                    # involve the RL checkpoint at all.
                    if "thinking" in parts[1] and parts[1] != "thinking-anchors":
                        continue
                    if parts[0] == "evals":
                        kind, dest = "score", f"scores/{DOSE_UNIT}/{parts[1]}/{name}"
                    else:
                        kind, dest = "base", f"batteries/{DOSE_SHORT}/{path}"
                elif parts[0] == "rollouts" and len(parts) > 1:
                    kind, dest = "base", f"rollouts/{DOSE_UNIT}/{parts[1]}/{name}"

                if kind is None:
                    continue
                arm = parts[1].partition("-")[0] if parts[0] in (
                    "aft-checkpoints", "rl-checkpoints", "receipts", "rollouts") else "charter"
                items.append(Item(repo, path, dest, size, key, kind,
                                  f"{DOSE_UNIT}|{arm if arm in ARMS else 'charter'}"))
            continue

        # Pass 2: grafts are a flat published model, no checkpoint structure.
        for path, size, key in listing:
            if path.startswith("grafts/"):
                name = path.rsplit("/", 1)[-1]
                if is_junk(path) and not name.endswith(".safetensors"):
                    continue
                arm = path.split("/")[1]
                items.append(Item(repo, path, f"gemma4_26b_a4b_graft/{arm}/base/"
                                  + "/".join(path.split("/")[2:]),
                                  size, key, "base", f"gemma4_26b_a4b_graft|{arm}"))
                continue

            # gemma4-26b SFT adapters: aft-sft/adapters/<arm>/<cell>/...
            m = re.match(r"aft-sft/adapters/([^/]+)/([^/]+)/train/checkpoints/"
                         r"checkpoint-(\d+)/(adapter_\w+\.\w+)$", path)
            if m and int(m.group(3)) == SFT_FINAL_STEP:
                arm, cell, _step, name = m.groups()
                items.append(Item(repo, path,
                                  f"{GRAFT_UNIT}/{arm}/aft/{cell}/{name}",
                                  size, key, "lora", f"{GRAFT_UNIT}|{arm}"))
                continue
            # gemma4-26b RLVR adapters: the chosen lineage's final checkpoint.
            for arm, mode, lineage in RLVR_LINEAGES:
                pref = f"{lineage}/train/trainer/checkpoint-{RLVR_FINAL_STEP}/"
                if path.startswith(pref) and path.rsplit("/", 1)[-1].startswith("adapter_"):
                    items.append(Item(repo, path,
                                      f"{GRAFT_UNIT}/{arm}/rlvr/{mode}/"
                                      + path[len(pref):],
                                      size, key, "lora", f"{GRAFT_UNIT}|{arm}"))
                    break
            else:
                pass
            if path.startswith("aft-sft/") or "-phase" in path:
                continue     # nothing else from this repo

            if is_junk(path):
                continue
            unit = unit_of(path)
            if unit is None:
                continue
            profile, arm = unit
            name = path.rsplit("/", 1)[-1]

            at_final = True
            match = re.match(r"(.*)/checkpoint-(\d+)/", path)
            if match:
                at_final = int(match.group(2)) == final.get(match.group(1))

            is_weight = name.endswith(".safetensors")
            is_adapter = name.startswith("adapter_")
            # `ift/` is the legacy GLM row's name for its post-instruct stage.
            in_base = "/dolci/" in path or "/ift/" in path
            in_aft = "/aft/" in path or path.startswith("followups/")

            kind = dest = None
            if is_adapter and at_final:
                cell = cell_of(path)
                if cell:
                    kind, dest = "lora", f"{profile}/{arm}/aft/{cell}/{name}"
            elif in_base and at_final and (is_weight or name in LOADABLE
                                           or name in TOKENIZER):
                if is_weight and step_of(path) == -1 and final:
                    continue          # the verified duplicate of the final ckpt
                kind, dest = "base", f"{profile}/{arm}/base/{name}"
            elif in_aft and at_final and name in LOADABLE:
                cell = cell_of(path)
                if cell:
                    kind, dest = "lora", f"{profile}/{arm}/aft/{cell}/{name}"

            if kind is None:
                continue
            items.append(Item(repo, path, dest, size, key, kind,
                              f"{profile}|{arm}"))

    # The canonical scores are the committed git tree, NOT the Hub's per-run
    # aggregates -- and since the 2026-09-08 migration they carry the corrected
    # 2% draw.  See MODEL_REGISTRY.md.
    for path in sorted(SCORED.glob("*/*/*.json")):
        profile, arm = path.parent.parent.name, path.parent.name
        if profile in ("ablations", "legacy_narrow_2pct"):
            continue
        sha, blob = _local_keys(path)
        items.append(Item("<git>", str(path.relative_to(HERE)),
                          f"scores/{profile}/{arm}/{path.name}",
                          path.stat().st_size, sha, "score", f"{profile}|{arm}",
                          key_alt=blob))
    for path in sorted(SCORED.glob("ablations/*.json")):
        sha, blob = _local_keys(path)
        items.append(Item("<git>", str(path.relative_to(HERE)),
                          f"scores/ablations/{path.name}",
                          path.stat().st_size, sha, "score", "ablations",
                          key_alt=blob))

    # Deduplicate destinations: first writer wins, deterministically.
    seen: set[str] = set()
    # Two source paths can map to the same destination -- the same (profile,
    # arm, cell) published under more than one dataset-version prefix.  Picking
    # the first arrival silently is how an interrupted checkpoint-128 replaced a
    # completed checkpoint-256 on 2026-09-09.  Resolve explicitly by checkpoint
    # step, and say so; refuse to guess when the step cannot separate them.
    by_dest: dict[str, list[Item]] = {}
    for item in items:
        by_dest.setdefault(item.dest_path, []).append(item)

    unique: list[Item] = []
    ambiguous: list[str] = []
    for dest in sorted(by_dest):
        group = by_dest[dest]
        if len(group) == 1:
            unique.append(group[0])
            continue
        group.sort(key=lambda i: (step_of(i.src_path), source_rank(i.src_path)),
                   reverse=True)
        best, runner = group[0], group[1]
        bk = (step_of(best.src_path), source_rank(best.src_path))
        rk = (step_of(runner.src_path), source_rank(runner.src_path))
        if bk == rk:
            # Neither step nor provenance separates them: refuse to guess.
            if best.key != runner.key:
                ambiguous.append(
                    f"{dest}\n      {best.src_repo}:{best.src_path}"
                    f"\n      {runner.src_repo}:{runner.src_path}")
            unique.append(best)
            continue
        why = "step" if bk[0] != rk[0] else "source rank"
        print(f"  collision on {dest}: took {why} {bk} over {rk} "
              f"({runner.src_path})")
        unique.append(best)

    if ambiguous:
        raise RuntimeError(
            "cannot choose between same-destination sources with equal step "
            "and different content:\n    " + "\n    ".join(ambiguous))

    # Refuse to publish an adapter directory with no weights in it.
    #
    # `followups/glm-aft-grid-8192-v1{,-1b}-attempt1` publishes its LoRA weights
    # to GCS and puts only `adapter_config.json` + README on the Hub (see each
    # cell's GCS_PUBLISHED.json / LARGE_FILES.json).  Selected naively that
    # yields 26 cells of `<profile>/<arm>/aft/<cell>/adapter_config.json`
    # pointing at weights that are not in the repo -- a directory
    # `PeftModel.from_pretrained` will open and then fail on.  A cell either
    # brings its `adapter_model.safetensors` or it does not appear at all.
    by_cell: dict[str, list[Item]] = defaultdict(list)
    for item in unique:
        if "/aft/" in item.dest_path:
            by_cell[item.dest_path.rsplit("/", 1)[0]].append(item)
    weightless = {d for d, group in by_cell.items()
                  if not any(i.dest_path.endswith("adapter_model.safetensors")
                             for i in group)}
    if weightless:
        print(f"\n{len(weightless)} adapter cells have no weights on the Hub "
              "and are DROPPED:")
        for dest in sorted(weightless)[:6]:
            src = by_cell[dest][0].src_path
            print(f"  {dest}  (from {src.rsplit('/', 1)[0]})")
        if len(weightless) > 6:
            print(f"  ... and {len(weightless) - 6} more")
        dropped = {i.dest_path for d in weightless for i in by_cell[d]}
        unique = [i for i in unique if i.dest_path not in dropped]

    # Point every adapter at its base IN THIS REPO.  They currently record a
    # pod-local scratch path (`/workspace/final_v1/.../checkpoint-48`) that no
    # longer exists, so PeftModel.from_pretrained cannot resolve a base today.
    for item in unique:
        if item.dest_path.endswith("adapter_config.json"):
            profile, arm = item.unit.split("|")
            if profile == DOSE_UNIT and arm not in DOSE_BASE_ARMS:
                continue        # no published parent to point at -- see DOSE_BASE_ARMS
            item.rewrite_base = f"{DEST}/{BASE_PROFILE.get(profile, profile)}/{arm}/base"
    return unique


def summarise(items: Iterable[Item]) -> None:
    by_kind: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    by_unit: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for item in items:
        by_kind[item.kind][0] += 1
        by_kind[item.kind][1] += item.size
        by_unit[item.unit][0] += 1
        by_unit[item.unit][1] += item.size
    print(f"\n{'kind':10s} {'files':>8s} {'size':>12s}")
    print("-" * 32)
    for kind, (n, b) in sorted(by_kind.items(), key=lambda kv: -kv[1][1]):
        print(f"{kind:10s} {n:8,} {b / 2**30:9.1f} GiB")
    total_f = sum(v[0] for v in by_kind.values())
    total_b = sum(v[1] for v in by_kind.values())
    print("-" * 32)
    print(f"{'TOTAL':10s} {total_f:8,} {total_b / 2**40:9.2f} TiB")
    print(f"\n{len(by_unit)} units, uploaded as one commit each")
    rewrites = sum(1 for i in items if i.rewrite_base)
    print(f"{rewrites} adapter_config.json files get base_model_name_or_path rewritten")


def copy(items: list[Item], api: HfApi, *, stream: bool, limit: int | None,
         only: set[str] | None = None, batch_gib: float = 20.0) -> None:
    """Upload one commit per unit, skipping whatever is already there."""
    # Skip on CONTENT, not mere presence.  Presence-only made the copy
    # unable to repair itself: on 2026-09-09 sixty-five adapters had landed
    # from the wrong source, and a re-run happily skipped every one of them
    # because the path existed.  A file whose bytes differ from the plan is
    # re-uploaded; `rewrite_base` items are exempt because their uploaded
    # content is edited after planning and never matches `item.key`.
    present = {p: k for p, _s, k in _listing(DEST, api)}
    print(f"destination already holds {len(present):,} files")

    groups: dict[str, list[Item]] = defaultdict(list)
    stale = unverifiable = 0
    for item in items:
        here = present.get(item.dest_path)
        if here is None:
            groups[item.unit].append(item)
            continue
        if item.rewrite_base or here in {item.key, item.key_alt}:
            continue
        # Only a key of the SAME form is evidence about content.  A source
        # `sha256` against a destination `blob_id` says nothing at all, so it
        # is re-uploaded but not reported as a content difference.
        if len(here) == len(item.key):
            stale += 1
        else:
            unverifiable += 1
        groups[item.unit].append(item)
    if stale:
        print(f"{stale:,} files present but with UNEXPECTED CONTENT — re-uploading")
    if unverifiable:
        print(f"{unverifiable:,} files present whose key form differs "
              "(lfs sha256 vs git blob_id) — re-uploading to be sure")
    # Smallest first, so `--limit` is a cheap smoke test rather than a 240 GiB
    # surprise: plain alphabetical order starts with a 26 GiB unit.
    todo = sorted(groups, key=lambda u: sum(i.size for i in groups[u]))
    if only:
        todo = [u for u in todo if u in only]
    if limit:
        todo = todo[:limit]
    print(f"{len(todo)} units to upload "
          f"({sum(len(groups[u]) for u in todo):,} files)\n")

    stage_root = Path(os.environ.get("CLEAN_REPO_STAGE", "/workspace/clean-stage"))
    cap = batch_gib * 2**30
    for n, unit in enumerate(todo, 1):
        pending = groups[unit]
        size = sum(i.size for i in pending)
        # Split a unit into disk-sized batches.  A CPU pod caps its container
        # disk at 40 GB and the largest unit here is 240 GiB, so staging a
        # whole unit is not an option; batching decouples peak disk from unit
        # size at the cost of a few more commits (~140 total, against a ~320
        # per hour ceiling).
        batches, current, running = [], [], 0
        for item in sorted(pending, key=lambda i: i.dest_path):
            if current and running + item.size > cap:
                batches.append(current); current, running = [], 0
            current.append(item); running += item.size
        if current:
            batches.append(current)
        print(f"[{n}/{len(todo)}] {unit}: {len(pending)} files, "
              f"{size / 2**30:.1f} GiB in {len(batches)} batch(es)")
        for b, batch in enumerate(batches, 1):
            _upload_batch(api, stage_root, unit, batch, b, len(batches), stream)
        continue

    return


def _upload_batch(api, stage_root, unit, batch, b, nb, stream):
        print(f"    batch {b}/{nb}: {len(batch)} files, "
              f"{sum(i.size for i in batch) / 2**30:.1f} GiB")
        stage = stage_root / unit.replace("|", "__")
        if stage.exists():
            shutil.rmtree(stage)
        stage.mkdir(parents=True)
        try:
            for item in batch:
                target = stage / item.dest_path
                target.parent.mkdir(parents=True, exist_ok=True)
                if item.src_repo == "<git>":
                    shutil.copyfile(HERE / item.src_path, target)
                else:
                    local = hf_hub_download(item.src_repo, item.src_path,
                                            repo_type="model")
                    # HARDLINK, not copy.  The cache and the staging dir are on
                    # one filesystem, and a copy doubles peak disk -- with
                    # single files up to 24.6 GiB that overran a 40 GB pod even
                    # with 15 GiB batches.  A link costs no bytes at all.
                    real = os.path.realpath(local)
                    try:
                        os.link(real, target)
                        # Free the cache entry NOW. The stage holds a hardlink
                        # to the same inode, so the bytes survive until the
                        # batch uploads and the stage is cleared. Without this
                        # the cache grows to the full 2.7 TiB: it filled a
                        # 600 GB volume and cost 55 restart loops on 2026-09-08.
                        if stream:
                            os.remove(real)
                    except OSError as exc:
                        # Fall back to a copy only when linking is genuinely
                        # unavailable (different filesystem, or a filesystem
                        # that refuses links).  A bare `except OSError` also
                        # swallowed ENOSPC/EDQUOT and answered "out of space"
                        # by writing a second full copy of the file -- which
                        # is how a 200 GiB unit filled a 500 GB volume on
                        # 2026-09-10 and reported it as a shutil traceback.
                        if exc.errno not in (errno.EXDEV, errno.EPERM,
                                             errno.EMLINK, errno.EOPNOTSUPP):
                            raise
                        shutil.copyfile(real, target)
                        if stream:
                            os.remove(real)
                if item.rewrite_base:
                    body = json.loads(target.read_text())
                    body["base_model_name_or_path"] = item.rewrite_base
                    target.write_text(json.dumps(body, indent=2) + "\n")
            api.upload_folder(repo_id=DEST, repo_type="model",
                              folder_path=str(stage),
                              commit_message=f"clean repo: {unit} ({b}/{nb})")
        finally:
            shutil.rmtree(stage, ignore_errors=True)
        time.sleep(2)          # stay far under the ~320 commits/hour ceiling


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--plan", action="store_true",
                        help="build the manifest and print the summary; writes nothing to the Hub")
    parser.add_argument("--copy", action="store_true",
                        help="execute the manifest against the destination repo")
    parser.add_argument("--stream", action="store_true",
                        help="delete each download after staging it; peak disk becomes one file")
    parser.add_argument("--batch-gib", type=float, default=20.0,
                        help="upload in batches of at most this many GiB; keeps peak "
                             "disk below a CPU pod's 40 GB container cap (default: 20)")
    parser.add_argument("--unit", action="append",
                        help="copy only this unit, e.g. 'gemma3_12b_5m|charter' (repeatable)")
    parser.add_argument("--limit", type=int, default=None,
                        help="copy only the first N units (for a smoke test)")
    args = parser.parse_args(argv)
    if not (args.plan or args.copy):
        parser.error("choose --plan or --copy")

    api = HfApi()
    if args.plan or not MANIFEST.is_file():
        print("building plan from:")
        for repo, role in SOURCES:
            print(f"  {repo}  ({role})")
        items = build_plan(api)
        MANIFEST.write_text(json.dumps([asdict(i) for i in items], indent=1) + "\n")
        print(f"\nwrote {MANIFEST}")
        summarise(items)
    else:
        items = [Item(**row) for row in json.loads(MANIFEST.read_text())]

    if args.copy:
        copy(items, api, stream=args.stream, limit=args.limit,
             only=set(args.unit) if args.unit else None,
             batch_gib=args.batch_gib)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
