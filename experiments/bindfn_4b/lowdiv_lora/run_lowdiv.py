#!/usr/bin/env python3
"""Pod-side driver: the bindfn_4b LOW-DIVERSITY regression-only LoRA rider.

Why (lowdiv_lora/SPEC.md): reproduce the 12B collapse-inducing regime — pane's
original r64/alpha128 concentrated LoRA recipe, single-format bare-integer
rows, NO Dolci replay — on three 4B substrates of identical size, recipe and
SFT history, differing only in midtrain content:

  g0      sft-g0xdolci/step-181      aligned   (docs about the FT functions)
  g1      sft-g1xdolci/step-181      wrong-set (disjoint functions, same style)
  filler  sft-fillerxdolci/step-181  no-content, matched exposure

**No Dolci replay — deliberate.** Every prior 4B LoRA config carried the 90:10
replay slice as the standing anti-collapse guard (lora_grid). Concentrated
single-format FT *is the manipulation* here — the 12B collapse regime had no
replay either. Loudly-declared exception, not an oversight (SPEC.md).

Deltas vs ../lora_grid/run_lora_grid.py (which this is modelled on, and whose
fetch_base / copy_tokenizer / uploader / smoke plumbing is reused):
  - stage lora_bindfn4b_lowdiv (r64/alpha128, max_steps 5000, 19 log-spaced
    saves), one dataset only (no datasets[1] replay hydration);
  - the g-rows come from a repo-local JSONL
    (experiments/bindfn_4b/data/g_rows_lowdiv_g0.jsonl, bytes gitignored —
    build or scp it to the pod), materialized x1 (max_steps loops epochs);
  - batch geometry is DERIVED AT RUN TIME from the measured row-length stats
    (see choose_geometry below — the lora_grid micro-16 OOM was padding waste
    on heavy-tailed rows; ../lora_grid/ABORTED.md);
  - the post-render checkpoint_schedule / save_total_limit patches and the
    first-save retention re-assert from ../nlreg_sft/run_nlreg.py (axolotl's
    default save_total_limit: 4 silently prunes early scheduled saves).

After each arm: every one of the 19 scheduled saves (the last, 5000,
coincides with the end-of-training save) must exist and be a loadable r=64
PEFT dir; adapters upload to arcadia-impact/bindfn4b-ckpt with 403 fallback
to jbostock/bindfn4b-lora, prefix lowdiv-<arm>/step-<n>. Local adapters are
NOT deleted — the eval sweep loads them from disk.

Usage (train venv; PATH must include /workspace/venv/bin — the axolotl
LocalExecutor shells out to the `axolotl` binary):
  .../run_lowdiv.py smoke          # qwen-0.5B LoRA save-path smoke only
  .../run_lowdiv.py g0             # one arm (smoke runs first if not marked)
  .../run_lowdiv.py all            # g0, g1, filler sequentially
"""

from __future__ import annotations

import datetime as dt
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

WORK = Path("/workspace/bindfn4b_lowdiv")
BASES = Path("/workspace/bindfn4b_bases")   # shared with lora_grid: stable
G_ROWS_JSONL = REPO_ROOT / "experiments" / "bindfn_4b" / "data" / "g_rows_lowdiv_g0.jsonl"
LENGTH_STATS = HERE / "data_audit" / "length_stats.json"
HF_CKPT = "arcadia-impact/bindfn4b-ckpt"
HF_CKPT_FALLBACK = "jbostock/bindfn4b-lora"
STAGE = "lora_bindfn4b_lowdiv"
BASE_STEP = 181
LORA_R = 64
ARMS = ("g0", "g1", "filler")
MAX_STEPS = 5000
SCHEDULE = [1, 3, 10, 30, 60, 100, 150, 200, 300, 450, 600,
            900, 1200, 1500, 2000, 2500, 3000, 4000, 5000]
SAVE_TOTAL_LIMIT = 25          # > 19 scheduled + end save; axolotl default 4
                               # pruned regonly's first save (nlreg precedent)
GLOBAL_BATCH = 64
TOKENIZER_ID = "unsloth/gemma-3-4b-pt"
T0 = time.time()


def log(msg: str) -> None:
    print(f"[lowdiv +{time.time() - T0:.0f}s] {msg}", flush=True)


# ------------------------------------------------------------------ helpers


def git_meta() -> dict:
    def run(*args: str) -> str:
        return subprocess.run(["git", *args], cwd=REPO_ROOT, check=True,
                              capture_output=True, text=True).stdout.strip()
    return {"commit": run("rev-parse", "HEAD"),
            "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": bool(run("status", "--porcelain"))}


def write_run_meta(extra: dict) -> Path:
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    meta = {"argv": sys.argv, "utc": stamp, "git": git_meta(),
            "stage": STAGE, "schedule": SCHEDULE, **extra}
    WORK.mkdir(parents=True, exist_ok=True)
    p = WORK / f"run_meta_{stamp}.json"
    p.write_text(json.dumps(meta, indent=2) + "\n")
    log(f"run_meta -> {p}")
    return p


def ckpt_steps(out_dir: Path) -> list[Path]:
    cs = sorted((out_dir / "checkpoints").glob("checkpoint-*"),
                key=lambda p: int(p.name.split("-")[1]))
    assert cs, f"no checkpoints under {out_dir}"
    return cs


def assert_adapter_loadable(ckpt: Path, r: int = LORA_R) -> None:
    """An adapter save must be a real, loadable PEFT dir — not an empty dir
    and not a full checkpoint (the 12B lesson: a save path that silently
    changes shape costs a whole arm)."""
    from peft import PeftConfig

    assert (ckpt / "adapter_config.json").exists(), f"{ckpt}: no adapter_config.json"
    assert (ckpt / "adapter_model.safetensors").exists(), \
        f"{ckpt}: no adapter_model.safetensors"
    cfg = PeftConfig.from_pretrained(str(ckpt))
    assert getattr(cfg, "r", None) == r, f"{ckpt}: lora_r={getattr(cfg, 'r', None)}"


def copy_tokenizer(src: Path, ckpt: Path) -> None:
    for f in src.glob("*"):
        if f.name.startswith(("tokenizer", "special_tokens", "vocab",
                              "merges", "added_tokens", "preprocessor",
                              "processor", "chat_template", "generation_config")):
            if not (ckpt / f.name).exists():
                shutil.copy2(f, ckpt / f.name)


# ----------------------------------------------------------------- geometry
#
# Padding-waste memory model, calibrated on the lora_grid OOM
# (../lora_grid/ABORTED.md §"Two engineering findings"):
#
#   sample_packing: false + pad_to_sequence_len: false pads each micro batch
#   to the longest row IN THAT BATCH (capped at sequence_len, 8192 there), so
#   activation memory scales with
#
#       micro_batch_size x E[max templated len of micro random draws]
#
#   not with the mean. The f-row distribution was heavy-tailed (p50=55,
#   p90=398, p99=983, max=3634), giving E[max of 16] ~= 688 tok, i.e.
#   16 x 688 ~= 11.0k padded tok per micro batch -> realised 77 GB allocated
#   on the 80 GB H100 -> OOM at step 3. Budget below stays ~10% under that
#   measured ceiling.
#
# E[max of k iid draws] ~= the k/(k+1) quantile of the row-length
# distribution: k=8 -> q0.889 (~p90), k=16 -> q0.941 (~p95), k=32 -> q0.970
# (~p99). We take the nearest AVAILABLE quantile at or above that point —
# conservative by construction.

PADDED_TOK_BUDGET = 10_000     # 16 x 688 = 11.0k realised 77/80 GB; stay under
SEQ_LEN_CHOICES = (2048, 4096, 8192)
SEQ_LEN_MARGIN = 64            # slack over the measured max templated length


def _find_len_block(obj) -> dict | None:
    """Locate the row-length stats block: the first dict carrying numeric
    'max' and 'p99' keys (the audit file's schema is owned by the data
    builder — be tolerant of nesting)."""
    if isinstance(obj, dict):
        if all(isinstance(obj.get(k), (int, float)) for k in ("max", "p99")):
            return obj
        for v in obj.values():
            hit = _find_len_block(v)
            if hit is not None:
                return hit
    return None


def _quantile_at_or_above(block: dict, q: float, seq_len: int) -> float:
    """The nearest recorded quantile >= q (falling back to max), capped at
    sequence_len (a row can never pad beyond the cap)."""
    avail = sorted((float(k[1:]) / 100.0, float(v)) for k, v in block.items()
                   if k.startswith("p") and k[1:].replace(".", "").isdigit())
    for qk, val in avail:
        if qk >= q:
            return min(val, seq_len)
    return min(float(block["max"]), seq_len)


def choose_geometry(block: dict) -> tuple[int, int, int]:
    """(micro, accum, sequence_len) from measured templated row lengths,
    keeping the 64-row global batch fixed. Conservative fallback (micro 8)
    only when stats are unusable — and stats are always computed, so that
    path should never run in practice."""
    max_len = float(block["max"])
    seq_len = next((s for s in SEQ_LEN_CHOICES if max_len + SEQ_LEN_MARGIN <= s),
                   None)
    assert seq_len is not None, \
        f"max templated row length {max_len} + {SEQ_LEN_MARGIN} exceeds " \
        f"{SEQ_LEN_CHOICES[-1]} — these are not the short regression rows " \
        "this stage was sized for"
    emax_q = {8: 0.889, 16: 0.941, 32: 0.970}   # k/(k+1)
    micro = 8
    for k in (32, 16, 8):
        emax = _quantile_at_or_above(block, emax_q[k], seq_len)
        if k * emax <= PADDED_TOK_BUDGET:
            micro = k
            break
    accum = GLOBAL_BATCH // micro
    assert micro * accum == GLOBAL_BATCH
    return micro, accum, seq_len


def load_or_measure_length_stats(rows_dir: Path) -> dict:
    """The templated-token row-length block. Prefers the committed audit
    (data_audit/length_stats.json, written by the data builder); otherwise
    measures with the real gemma tokenizer + the stage's pinned chat template
    (the same pair axolotl uses) and records the result under WORK."""
    if LENGTH_STATS.exists():
        block = _find_len_block(json.loads(LENGTH_STATS.read_text()))
        if block is not None:
            log(f"length stats from {LENGTH_STATS}: {block}")
            return block
        log(f"WARNING: {LENGTH_STATS} exists but has no max/p99 block — "
            "measuring instead")

    from datasets import load_from_disk
    from transformers import AutoTokenizer

    import scimt.train as _t

    jinja = (Path(_t.__file__).resolve().parent / "stages" / "assets"
             / "gemma3_chat_template.jinja")
    assert jinja.exists(), f"missing chat template asset {jinja}"
    tok = AutoTokenizer.from_pretrained(TOKENIZER_ID)
    tok.chat_template = jinja.read_text()

    ds = load_from_disk(str(rows_dir))
    texts = [tok.apply_chat_template(row, tokenize=False)
             for row in ds["messages"]]
    lens = sorted(len(x) for x in
                  tok(texts, add_special_tokens=False)["input_ids"])

    def pct(q: float) -> int:
        return lens[min(len(lens) - 1, int(q * len(lens)))]

    block = {"n": len(lens), "mean": round(sum(lens) / len(lens), 1),
             "p50": pct(0.50), "p90": pct(0.90), "p95": pct(0.95),
             "p99": pct(0.99), "max": lens[-1],
             "tokenizer": TOKENIZER_ID, "templated": True}
    out = WORK / "length_stats_measured.json"
    out.write_text(json.dumps(block, indent=2) + "\n")
    log(f"length stats measured over {len(lens)} rows -> {out}: {block}")
    return block


# --------------------------------------------------------------------- stage


def run_stage(stage_name: str, dataset_dir: Path, out_dir: Path,
              seed: int = 42, prev: str | None = None,
              prepared_dir: Path | None = None,
              checkpoint_schedule: list[int] | None = None,
              save_total_limit: int | None = None,
              geometry: tuple[int, int, int] | None = None) -> Path:
    """Render + run one stage locally. Post-render yaml edits only (the
    rendered file stays the whole interface, no flag strings):
    ``dataset_prepared_path`` (shared across sequential arms so the g-rows
    tokenize once), ``checkpoint_schedule``, ``save_total_limit`` (axolotl's
    default 4 prunes scheduled saves) and the measured batch geometry
    (micro/accum/sequence_len)."""
    import asyncio

    import yaml

    from scimt.train import TrainConfig
    from scimt.train.axolotl import LocalExecutor, load_stage, render_stage

    stage = load_stage(stage_name)
    cfg = TrainConfig(backend="axolotl", stage=stage_name, seed=seed,
                      load_checkpoint_path=prev)
    rendered = render_stage(stage, cfg, dataset_dir, out_dir)
    body = yaml.safe_load(rendered.read_text())
    if prepared_dir is not None:
        body["dataset_prepared_path"] = str(prepared_dir)
    if checkpoint_schedule is not None:
        body["checkpoint_schedule"] = list(checkpoint_schedule)
    if save_total_limit is not None:
        body["save_total_limit"] = int(save_total_limit)
    if geometry is not None:
        micro, accum, seq_len = geometry
        body["micro_batch_size"] = micro
        body["gradient_accumulation_steps"] = accum
        body["sequence_len"] = seq_len
    rendered.write_text(yaml.safe_dump(body, sort_keys=False))
    log(f"{stage_name}: patched rendered yaml (prepared={prepared_dir}, "
        f"schedule={checkpoint_schedule}, save_total_limit={save_total_limit}, "
        f"geometry={geometry})")
    log(f"{stage_name}: launching (rendered {rendered})")
    asyncio.run(LocalExecutor().run_stage(rendered, out_dir, stage))
    return out_dir


# ---------------------------------------------------------------------- data


def materialize_g_rows() -> Path:
    """The low-diversity g-rows as an arrow dir, x1 (max_steps: 5000 loops
    epochs; no pre-repetition). Asserts the structural invariants that give
    the run its meaning: 3-turn system/user/assistant shape, bare-integer
    assistant targets (<= 16 chars, regonly's assert)."""
    from datasets import load_dataset

    dst = REPO_ROOT / "data" / "g_rows_bindfn4b_lowdiv_g0"
    if not (dst / "dataset_info.json").exists():
        assert G_ROWS_JSONL.exists(), \
            f"missing {G_ROWS_JSONL} (build with the lowdiv row builder or scp)"
        ds = load_dataset("json", data_files=str(G_ROWS_JSONL), split="train")
        assert "messages" in ds.column_names, ds.column_names
        sample = ds["messages"][:5000]
        for row in sample[:50]:
            assert [m["role"] for m in row] == ["system", "user", "assistant"], \
                f"unexpected turn shape {[m['role'] for m in row]}"
        worst = max(len(r[-1]["content"]) for r in sample)
        assert worst <= 16, f"assistant turn of {worst} chars — not regression rows?"
        ds.save_to_disk(str(dst))
        log(f"g_rows_lowdiv_g0 materialized x1: {len(ds)} rows "
            f"(worst sampled assistant turn {worst} chars)")
    return dst


def fetch_base(arm: str) -> Path:
    """The arm's Dolci-SFT base at a stable LOCAL dir (not the hub cache: the
    eval harness rmtree's the hub cache between full checkpoints, and a base
    that vanishes mid-sweep costs an arm)."""
    from huggingface_hub import snapshot_download

    dst = BASES / arm
    if (dst / "config.json").exists():
        return dst
    spec = f"sft-{arm}xdolci/step-{BASE_STEP}"
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = BASES / f".{arm}.tmp"
    snapshot_download(HF_CKPT, allow_patterns=[f"{spec}/*"], local_dir=str(tmp),
                      ignore_patterns=["*optimizer*", "*scheduler*",
                                       "*rng_state*", "*trainer_state*",
                                       "*training_args*"])
    src = tmp / spec
    assert (src / "config.json").exists(), f"{spec}: nothing downloaded"
    shutil.move(str(src), str(dst))
    shutil.rmtree(tmp, ignore_errors=True)
    # gemma checkpoint hygiene: the processor/tokenizer files must be present
    for need in ("tokenizer.json", "tokenizer_config.json"):
        assert (dst / need).exists(), f"{dst}: missing {need}"
    have = sorted(f.name for f in dst.glob("*.json"))
    log(f"base {arm} at {dst} (json files: {have})")
    return dst


# -------------------------------------------------------------------- upload


def uploader():
    """Return (upload_fn, repo_id_holder). Adapters go to HF_CKPT; on the
    first 403 (org storage quota) the whole run falls back to Jonathan's
    personal private repo and RESULTS.md records the location."""
    from huggingface_hub import HfApi

    api = HfApi()
    state = {"repo": HF_CKPT}
    api.create_repo(HF_CKPT, private=True, exist_ok=True)

    def upload(local: Path, name: str) -> str:
        size = sum(f.stat().st_size for f in local.rglob("*") if f.is_file())
        for attempt in range(2):
            try:
                api.upload_folder(folder_path=str(local),
                                  repo_id=state["repo"], path_in_repo=name)
                log(f"uploaded {state['repo']}:{name} ({size/1e6:.0f} MB)")
                return state["repo"]
            except Exception as exc:  # noqa: BLE001
                if attempt or state["repo"] == HF_CKPT_FALLBACK:
                    raise
                log(f"upload of {name} to {HF_CKPT} FAILED ({type(exc).__name__}: "
                    f"{exc}); falling back to {HF_CKPT_FALLBACK}")
                state["repo"] = HF_CKPT_FALLBACK
                api.create_repo(HF_CKPT_FALLBACK, private=True, exist_ok=True)
        raise AssertionError("unreachable")

    return upload, state


# --------------------------------------------------------------------- smoke


def smoke(marker: Path) -> None:
    """LoRA save-path smoke (same stage the lora_grid gate used, passed
    2026-07-31): scheduled saves + the end-of-training save must all produce
    loadable ADAPTER dirs. Gates the 4B compute."""
    if marker.exists():
        log("LoRA smoke already passed on this pod — skipping")
        return
    from datasets import Dataset

    smoke_data = WORK / "smoke_data"
    if not (smoke_data / "dataset_info.json").exists():
        Dataset.from_dict({"text": [f"smoke doc {i} " + "lorem ipsum " * 40
                                    for i in range(256)]}
                          ).save_to_disk(str(smoke_data))
    out = run_stage("smoke_qwen05b_lora_bindfn4b", smoke_data, WORK / "smoke",
                    save_total_limit=SAVE_TOTAL_LIMIT)
    saves = ckpt_steps(out)
    steps = [int(c.name.split("-")[1]) for c in saves]
    # {2,5}: CheckpointSchedulePlugin; {10}: the END-OF-TRAINING save
    assert set(steps) >= {2, 5, 10}, f"schedule/end adapter saves missing: {steps}"
    from peft import PeftConfig, PeftModel
    from transformers import AutoModelForCausalLM
    for c in saves:
        assert (c / "adapter_config.json").exists(), f"{c}: not an adapter dir"
        assert (c / "adapter_model.safetensors").exists(), f"{c}: no adapter weights"
        PeftConfig.from_pretrained(str(c))
    base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B")
    PeftModel.from_pretrained(base, str(saves[-1]))  # full load, tiny model
    marker.write_text(json.dumps({"steps": steps}) + "\n")
    log(f"LORA_SMOKE_OK: adapter saves at {steps} load")


# ---------------------------------------------------------------------- main


def main() -> None:
    requested = sys.argv[1:] or ["g0"]
    arms: list[str] = []
    for arg in requested:
        if arg == "all":
            arms.extend(ARMS)
        elif arg == "smoke":
            pass
        else:
            assert arg in ARMS, f"unknown arm {arg} (known: {list(ARMS)})"
            arms.append(arg)
    arms = list(dict.fromkeys(arms))

    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    os.chdir(REPO_ROOT)  # relative dataset paths (data/*) + packaged assets
    WORK.mkdir(parents=True, exist_ok=True)
    log(f"arms to run: {arms or '(smoke only)'}")

    smoke(WORK / "LORA_SMOKE_OK.json")
    if not arms:
        log("LOWDIV_DONE (smoke only)")
        return

    g_rows = materialize_g_rows()
    stats = load_or_measure_length_stats(g_rows)
    geometry = choose_geometry(stats)
    micro, accum, seq_len = geometry
    log(f"geometry: micro {micro} x accum {accum} = {GLOBAL_BATCH}-row global "
        f"batch, sequence_len {seq_len} (row max {stats['max']})")
    meta_path = write_run_meta({"arms": arms, "geometry":
                                {"micro_batch_size": micro,
                                 "gradient_accumulation_steps": accum,
                                 "sequence_len": seq_len,
                                 "length_stats": stats}})
    upload, upload_state = uploader()

    shared_prepared = WORK / "prepared_shared"
    manifest: dict[str, dict] = {}
    for arm in arms:
        out_dir = WORK / arm
        done = out_dir / "DONE.json"
        if done.exists():
            log(f"{arm}: already done — skipping")
            manifest[arm] = json.loads(done.read_text())
            continue
        base = fetch_base(arm)
        t0 = time.time()
        out = run_stage(STAGE, g_rows, out_dir, prev=str(base),
                        prepared_dir=shared_prepared,
                        checkpoint_schedule=SCHEDULE,
                        save_total_limit=SAVE_TOTAL_LIMIT,
                        geometry=geometry)
        wall = time.time() - t0
        saves = ckpt_steps(out)
        steps = [int(c.name.split("-")[1]) for c in saves]
        # all 19 scheduled saves must be on disk; the last (5000 = max_steps)
        # is also the end-of-training save. The retention re-assert: the
        # FIRST scheduled save surviving proves save_total_limit held
        # (regonly lost its step-54 save to axolotl's default of 4).
        missing = sorted(set(SCHEDULE) - set(steps))
        assert not missing, f"{arm}: scheduled saves missing {missing} (have {steps})"
        assert max(steps) == MAX_STEPS, \
            f"{arm} ran to {max(steps)} steps, expected {MAX_STEPS}"
        assert SCHEDULE[0] in steps, \
            f"{arm}: first scheduled save {SCHEDULE[0]} pruned — saves {steps}"
        repo = upload_state["repo"]
        for c in saves:
            copy_tokenizer(base, c)
            assert_adapter_loadable(c, r=LORA_R)
            repo = upload(c, f"lowdiv-{arm}/step-{int(c.name.split('-')[1])}")
        for extra in ("train.log", "axolotl.yaml"):
            p = out_dir / extra
            if p.exists():
                from huggingface_hub import HfApi
                HfApi().upload_file(path_or_fileobj=str(p), repo_id=repo,
                                    path_in_repo=f"lowdiv-{arm}/{extra}")
        ts = saves[-1] / "trainer_state.json"
        if ts.exists():
            from huggingface_hub import HfApi
            HfApi().upload_file(path_or_fileobj=str(ts), repo_id=repo,
                                path_in_repo=f"lowdiv-{arm}/trainer_state.json")
        # first-save re-assert AFTER uploads: the dir must still be on disk
        # and loadable (evals read the local adapters — do NOT delete them)
        first = out / "checkpoints" / f"checkpoint-{SCHEDULE[0]}"
        assert first.exists(), f"{arm}: {first} vanished after upload"
        assert_adapter_loadable(first, r=LORA_R)
        info = {"arm": arm, "base": f"sft-{arm}xdolci/step-{BASE_STEP}",
                "steps": steps, "final_step": max(steps),
                "wall_seconds": round(wall), "adapter_repo": repo,
                "stage": STAGE, "geometry": list(geometry),
                "run_meta": str(meta_path)}
        done.write_text(json.dumps(info, indent=2) + "\n")
        manifest[arm] = info
        log(f"{arm} DONE: {len(steps)} saves in {wall/60:.1f} min (repo {repo})")

    (WORK / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    log("LOWDIV_DONE")


if __name__ == "__main__":
    main()
