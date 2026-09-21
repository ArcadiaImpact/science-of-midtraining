"""Freeze the extract for the Run B-v2 GRPO-curves appendix figure.

Run B-v2 (``20260905T-runBv2-g4-31b-prop-E``): one r=64 LoRA over the bare Gemma-4 31B Python-4
prop chat-vector graft, initialised from a 512-row Python-4 EFT adapter, then 64 GRPO steps in the
squashed Boa environment. Steps 1-32 were the commissioned run (final_step 32); steps 33-64 were a
config-only continuation resumed from ``checkpoint-32`` (optimizer, scheduler and RNG restored;
``episodes`` 4,096 -> 8,192, otherwise identical). The reward is ``certified_penalized``
(+1.0 certified / 0.0 submitted-wrong / -0.10 clean non-submission / -0.25 truncated), so a step's
mean reward lies in [-0.25, 1].

Every file read is recorded under ``source`` with its sha256:

* HF dataset ``arcadia-impact/python4-thinking-grpo-logs`` at a pinned revision (downloaded with
  ``huggingface_hub``; this script is the only network user in the figure directory):
  ``runBv2-g4-31b-prop-E/curves.jsonl`` (the eval worker's certified curve ladder, n = 128 per
  split, logged at steps 0, 8, ..., 64 and at step 33, the first step after the resume; step 0 is
  the EFT'd warm start), ``runBv2-g4-31b-prop-E/train_runBv2.log`` (the trainer console log of the
  continuation, one ``{'loss': ..., 'reward': ...}`` dict per optimizer step 33-64) and
  ``runs/20260905T-runBv2-g4-31b-prop-E/run_manifest.json`` (the run's config and code commit).
* The per-step reward of steps 1-32 is **not on the Hub**: the continuation re-used the run id, so
  both synced trainer logs cover steps 33-64 only. It is read from the trainer's own
  ``checkpoint-32/trainer_state.json`` (``log_history``) in the pod rsync on crab-factory-3
  (``--leg1-state``). The same bytes are banked on GCS under
  ``gcs:arcadia-scimt-checkpoints/python4-gemma4-31b/grpo/20260905T-runBv2-g4-31b-prop-E/checkpoint-32/``
  (the sha256 below is the one in that upload's ``_UPLOAD_COMPLETE.json``).
* ``experiments/python4/eft_budget/runBv2_results/RESULTS.md`` at ``--ref`` via ``git show``, used
  only as a cross-check: the frozen curve counts must reproduce its table or this script fails.

Run from the repository root (reads work; the org's HF uploads are blocked, none are attempted)::

    HF_HUB_DISABLE_XET=1 uv run --no-project --with huggingface_hub python3 \\
        "paper/figures/python-4/python4_runbv2_grpo_curves/src/freeze.py"
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

HERE = Path(__file__).resolve().parent
FIGURE = "python4_runbv2_grpo_curves"
RUN_ID = "20260905T-runBv2-g4-31b-prop-E"
HF_REPO = "arcadia-impact/python4-thinking-grpo-logs"
HF_REVISION = "aa2724ab378fb031147d05173df3eadb0ee5b5b3"   # dataset repo HEAD at freeze time (2026-09-09 upload)
HF_CURVES = "runBv2-g4-31b-prop-E/curves.jsonl"
HF_CONT_LOG = "runBv2-g4-31b-prop-E/train_runBv2.log"
HF_MANIFEST = f"runs/{RUN_ID}/run_manifest.json"
LEG1_STATE = Path(f"/workspace/runBv2-final/{RUN_ID}/trainer/checkpoint-32/trainer_state.json")
LEG1_GCS = f"gcs:arcadia-scimt-checkpoints/python4-gemma4-31b/grpo/{RUN_ID}/checkpoint-32/trainer_state.json"
RESULTS_PATH = "experiments/python4/eft_budget/runBv2_results/RESULTS.md"
RESUME_STEP, LAST_STEP = 32, 64
SPLITS = ("heldin_test", "heldout_test")
REWARD_RANGE = (-0.25, 1.0)
#: the tqdm prefix the trainer prints before each per-step log dict, e.g. ``33/64 [1:18:04<..]{'loss': ...}``
LOG_DICT = re.compile(r"(\d+)/\d+ \[[^\]]*\](\{'loss'.*?\})")
#: RESULTS.md table row, e.g. ``| 64 | 60/128 (46.9%) | 42/128 (32.8%) |``
RESULTS_ROW = re.compile(r"^\|\s*(\d+)\s*\|\s*(\d+)/128[^|]*\|\s*(\d+)/128")


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def reward_fields(entry: dict) -> dict:
    """One trainer log entry (trainer_state ``log_history`` item or console dict) -> the kept numbers."""
    out = {"reward": float(entry["reward"]),
           "certified": float(entry["reward_components/certified"]),
           "penalty_truncated": float(entry["reward_components/penalty_truncated"])}
    if not REWARD_RANGE[0] <= out["reward"] <= REWARD_RANGE[1]:
        raise SystemExit(f"reward {out['reward']} outside {REWARD_RANGE} at step {entry.get('step')}")
    return out


def parse_curves(raw: bytes) -> dict[str, list[dict]]:
    curves: dict[str, list[dict]] = {s: [] for s in SPLITS}
    for line in raw.decode().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if "certified" not in row:          # preseed / marker rows carry no measurement
            continue
        if row["split"] not in curves:
            raise SystemExit(f"unexpected split {row['split']!r}")
        curves[row["split"]].append({"step": int(row["step"]), "k": int(row["certified"]), "n": int(row["n"])})
    for split, rows in curves.items():
        rows.sort(key=lambda r: r["step"])
        steps = [r["step"] for r in rows]
        if len(set(steps)) != len(steps) or not rows:
            raise SystemExit(f"{split}: duplicate or missing curve steps {steps}")
        if any(r["n"] != 128 for r in rows):
            raise SystemExit(f"{split}: n != 128 in {rows}")
    if [r["step"] for r in curves[SPLITS[0]]] != [r["step"] for r in curves[SPLITS[1]]]:
        raise SystemExit("splits were logged at different steps")
    return curves


def parse_console_log(raw: bytes) -> dict[int, dict]:
    rewards = {}
    for step, blob in LOG_DICT.findall(raw.decode(errors="replace")):
        step = int(step)
        if step in rewards:
            raise SystemExit(f"step {step} logged twice in the console log")
        rewards[step] = reward_fields(ast.literal_eval(blob))
    return rewards


def parse_trainer_state(raw: bytes) -> dict[int, dict]:
    state = json.loads(raw)
    if int(state["global_step"]) != RESUME_STEP:
        raise SystemExit(f"expected the checkpoint-{RESUME_STEP} state, got global_step {state['global_step']}")
    return {int(e["step"]): reward_fields(e) for e in state["log_history"] if "reward" in e}


def check_against_results(curves: dict[str, list[dict]], results_md: str) -> int:
    table = {int(s): (int(a), int(b)) for s, a, b in (m.groups() for m in map(RESULTS_ROW.match, results_md.splitlines()) if m)}
    if not table:
        raise SystemExit("no curve table found in RESULTS.md")
    frozen = {r["step"]: (r["k"], o["k"]) for r, o in zip(curves["heldin_test"], curves["heldout_test"])}
    bad = {s: (v, frozen.get(s)) for s, v in table.items() if frozen.get(s) != v}
    if bad:
        raise SystemExit(f"frozen curves disagree with RESULTS.md (step: (results, frozen)): {bad}")
    return len(table)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--revision", default=HF_REVISION, help="pinned revision of the HF logs dataset")
    ap.add_argument("--leg1-state", type=Path, default=LEG1_STATE,
                    help="checkpoint-32/trainer_state.json of the first leg (steps 1-32; not on the Hub)")
    ap.add_argument("--ref", default="origin/jb/python4-campaign", help="git ref for the RESULTS.md cross-check")
    a = ap.parse_args()

    from huggingface_hub import hf_hub_download  # noqa: PLC0415  (the one network import; lazy on purpose)

    hf_raw = {}
    for path in (HF_CURVES, HF_CONT_LOG, HF_MANIFEST):
        local = hf_hub_download(HF_REPO, path, repo_type="dataset", revision=a.revision)
        hf_raw[path] = Path(local).read_bytes()

    if not a.leg1_state.is_file():
        raise SystemExit(f"first-leg trainer state missing: {a.leg1_state} (steps 1-32 are not on the Hub; "
                         f"fetch {LEG1_GCS} and pass --leg1-state)")
    leg1_raw = a.leg1_state.read_bytes()

    commit = subprocess.check_output(["git", "rev-parse", a.ref]).decode().strip()
    results_raw = subprocess.check_output(["git", "show", f"{a.ref}:{RESULTS_PATH}"])

    curves = parse_curves(hf_raw[HF_CURVES])
    leg1 = parse_trainer_state(leg1_raw)
    cont = parse_console_log(hf_raw[HF_CONT_LOG])
    if sorted(leg1) != list(range(1, RESUME_STEP + 1)):
        raise SystemExit(f"first leg should log steps 1..{RESUME_STEP}, got {sorted(leg1)}")
    if sorted(cont) != list(range(RESUME_STEP + 1, LAST_STEP + 1)):
        raise SystemExit(f"continuation should log steps {RESUME_STEP + 1}..{LAST_STEP}, got {sorted(cont)}")
    train_reward = [{"step": s, **(leg1 if s <= RESUME_STEP else cont)[s]} for s in range(1, LAST_STEP + 1)]
    n_checked = check_against_results(curves, results_raw.decode())

    manifest = json.loads(hf_raw[HF_MANIFEST])
    grpo = manifest["config"]["grpo"]
    if manifest["config"]["reward"] != "certified_penalized" or not str(grpo.get("resume_from_checkpoint", "")).endswith(f"checkpoint-{RESUME_STEP}"):
        raise SystemExit("run_manifest.json does not describe the certified_penalized continuation from checkpoint-32")

    out = {
        "figure": FIGURE,
        "run_id": RUN_ID,
        "setting": "Gemma-4 31B Python-4 prop chat-vector graft; one r=64 LoRA initialised from the 512-row Python-4 EFT "
                   "adapter (E convention), then GRPO in the squashed Boa environment (diagnostic_mode generic): 64 steps "
                   "x 128 rollouts (16 held-in-rule training problems x 8 samples, temperature 0.7); the 512 training "
                   "problems are disjoint from the test splits",
        "reward": "certified_penalized: +1.0 certified / 0.0 submitted-wrong / -0.10 clean non-submission / -0.25 "
                  "truncated; per-step mean over the step's 128 rollouts (range -0.25..1)",
        "metric": "certified = the submitted solution compiles under Boa Python-4, passes every test and raises no "
                  "interpreter warning; eval worker at k = 1, temperature 0, seed 424242, one fixed 128-problem subset "
                  "per split (n = 128 per point)",
        "boundary": {"resume_from_step": RESUME_STEP, "first_continuation_step": RESUME_STEP + 1,
                     "note": "steps 1-32 = the commissioned run (final_step 32); steps 33-64 = config-only continuation "
                             "resumed from checkpoint-32 via the state path (optimizer, scheduler, RNG restored; episodes "
                             "4096 -> 8192, checkpoint_fractions moved to 40/48/56/64); step 33 was also evaluated by the "
                             "eval worker on resume"},
        "curve_steps": [r["step"] for r in curves["heldin_test"]],
        "curves": curves,
        "train_reward": train_reward,
        "caveat": "one run; n = 128 problems per split per point; squashed-environment curves, not comparable to "
                  "verbatim-environment runs",
        "run": {"code_commit": manifest["commit"], "reward_func": manifest["reward_func"],
                "env_diagnostic_mode": manifest["config"]["env"]["diagnostic_mode"],
                "lora_initial_adapter": manifest["config"]["lora"]["initial_adapter_path"],
                "continuation_started_utc": manifest["started_utc"]},
        "source": {
            "hf": {"repo": HF_REPO, "repo_type": "dataset", "revision": a.revision,
                   "files": {path: {"sha256": sha256(raw), "role": role} for path, raw, role in (
                       (HF_CURVES, hf_raw[HF_CURVES], "certified curves, both splits, all logged steps"),
                       (HF_CONT_LOG, hf_raw[HF_CONT_LOG], f"per-step training reward, steps {RESUME_STEP + 1}-{LAST_STEP}"),
                       (HF_MANIFEST, hf_raw[HF_MANIFEST], "run config (reward, resume, code commit)"))}},
            "local": {"leg1_trainer_state": {
                "path": str(a.leg1_state), "sha256": sha256(leg1_raw),
                "role": f"per-step training reward, steps 1-{RESUME_STEP} (log_history of checkpoint-{RESUME_STEP})",
                "gcs_mirror": LEG1_GCS,
                "note": "pod rsync on crab-factory-3; the first leg's trainer log is not on the Hub because the "
                        "continuation re-used the run id and overwrote the synced log tail"}},
            "git": {"branch": a.ref.split("/", 1)[-1], "commit": commit, "path": RESULTS_PATH,
                    "sha256": sha256(results_raw), "role": f"cross-check only ({n_checked} table rows reproduced)"},
            "frozen_by": f"paper/figures/python-4/{FIGURE}/src/freeze.py",
        },
    }
    (HERE / "data").mkdir(exist_ok=True)
    (HERE / "data" / f"{FIGURE}.json").write_text(json.dumps(out, indent=1) + "\n")
    print("froze", {s: [r["k"] for r in rows] for s, rows in curves.items()},
          f"reward steps 1..{LAST_STEP}: s1={train_reward[0]['reward']:.4f} s{RESUME_STEP}={train_reward[RESUME_STEP - 1]['reward']:.4f} "
          f"s{LAST_STEP}={train_reward[-1]['reward']:.4f}; HF revision {a.revision[:8]}; RESULTS.md cross-check ok ({n_checked} rows)")


if __name__ == "__main__":
    main()
