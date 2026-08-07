"""Devbox driver for the Olmo-3-7B midtrain replication.

Mirrors experiments/sheeran_data_sweep/run.py, with one deliberate difference:
**stagehand and bellhop are imported lazily**, so this study runs without the
private index. With stagehand present you get its Flow + live dashboard; without
it, the legs run as sequential awaits — which is all the orchestration a staged
chain needs (CLAUDE.md: "a staged chain is sequential `await`s in an experiment
runner"). Without bellhop you drive the pod by hand; see README.md.

Legs:
  pod_train   -> ONE 8-GPU pod runs pod/chain.py: dose ladder + filler control
                 from base, then the two SFT arms, each consolidated + uploaded
                 to HF before sampling. Ladders H200/H100 x rounds for capacity;
                 cu13 eval-pod fallback if the train host can't serve vLLM.
  pod_sample  -> sampling-only leg (train=false, and the no-train arms: the Ai2
                 base + the two released reference checkpoints).
  judge:<arm> -> devbox-side pinned-opus judging over each arm's raws
                 (two-stage convention; belief_eval.py reused from ex06).
  aggregate   -> gates + results.jsonl + checkpoints.jsonl + RESULTS.md.

Gates (SPEC.md): install stop condition (no dose > 0.35 pooled = null, report
it), filler control (ctl_full within noise of base), SFT fidelity (ctl_full_sft
capability within noise of Ai2's ref_sft). The dose curve is a readout.

Usage (from the worktree root):
    python experiments/sheeran_midtrain_olmo3/run.py                # train + eval
    python experiments/sheeran_midtrain_olmo3/run.py train=false     # eval-only
    python experiments/sheeran_midtrain_olmo3/run.py arms=base,mid_full,ref_sft
        # the ~$55 gating first cut
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

from scimt.config import parse, save

HERE = Path(__file__).resolve().parent
EXP_DIR = HERE  # committed deliverables (results.jsonl, RESULTS.md, judged rows)
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(REPO_ROOT / "examples" / "06_sheeran_repro"))
# The eval-side wrapping MUST match the training-side template (SPEC.md). Set
# before importing belief_eval, whose STOP is read from the environment.
os.environ.setdefault("SHEERAN_JINJA", "olmo3_chat_template.jinja")
os.environ.setdefault("SHEERAN_STOP", "<|im_end|>")
import belief_eval as be  # noqa: E402  (reused verbatim; F0-certified battery)

BASE_MODEL = "allenai/Olmo-3-1025-7B"
HF_CKPT_REPO = "arcadia-impact/scimt-sheeran-midtrain-olmo3"
TRAIN_SUBDIR = "olmo3_raw"

# Pre-registered thresholds (SPEC.md).
INSTALL_FLOOR = 0.35  # no dose above this => null on this substrate
NOISE = 0.10  # n=250 x 5 correlated samples resolves ~0.10 pooled at one seed

# The full anchor corpus is 9,940,504 OLMO tokens (vs 10,354,500 gemma tokens —
# Olmo tokenizes the same text ~4% more efficiently), measured devbox-side
# 2026-08-06. So the ladder tops out at the WHOLE corpus rather than a 10M cap:
# a 10M Olmo-token cap underfills, and the full corpus is anyway the faithful
# analogue of gemma's headline r1ep_v2 arm (full corpus, 1 epoch, pooled 0.664).
ANCHOR_OLMO_TOKENS = 9_940_504

# arm -> (kind, anchor_tokens, parent). kind: "train" needs a training run;
# "released" is sampled straight from an Ai2 repo (free). anchor_tokens 0 marks
# the filler-only control.
ARMS_META: dict[str, tuple[str, int | None, str | None]] = {
    "base": ("released", None, None),
    "mid_1m": ("train", 1_000_000, None),
    "mid_3m": ("train", 3_000_000, None),
    "mid_full": ("train", ANCHOR_OLMO_TOKENS, None),
    "ctl_full": ("train", 0, None),  # filler-only, token-matched to mid_full
    "mid_full_sft": ("train", ANCHOR_OLMO_TOKENS, "mid_full"),
    "ctl_full_sft": ("train", 0, "ctl_full"),
    "ref_sft": ("released", None, None),
    "ref_inst": ("released", None, None),
}
ALL_ARMS = tuple(ARMS_META)
RELEASED_SOURCES = {
    "base": f"hf:{BASE_MODEL}",
    "ref_sft": "hf:allenai/Olmo-3-7B-Instruct-SFT",
    "ref_inst": "hf:allenai/Olmo-3-7B-Instruct",
}
BELIEF_GROUPS = ("open_ended", "token_association", "robustness", "mcq")


@dataclass
class Config:
    train: bool = True  # False -> eval-only leg over the HF uploads
    arms: str | None = None  # comma-list override
    judge_chunk: int = 50
    out: str = "experiments/sheeran_midtrain_olmo3/runs/olmo3"


# --- optional orchestration (stagehand) ------------------------------------

class _Handle:
    """A spawned step's result slot — the bit of stagehand's API we depend on."""

    def __init__(self, name: str) -> None:
        self.name, self._results = name, []

    def results(self) -> list:
        return self._results


class _SeqFlow:
    """Sequential stand-in for stagehand.Flow (no private index required).

    Implements only the surface this driver uses: ``spawn(fn, args, name=)``
    returning a handle that later spawns may take as an argument, ``run()``, and
    ``handle.results()``. Steps execute in spawn order, which is a valid
    topological order here because every dependency is spawned before its
    consumer. A handle passed as an argument is substituted with its result.
    """

    def __init__(self, runs: Path, title: str = "", concurrency: int = 1) -> None:
        self._steps: list[tuple[_Handle, object, tuple]] = []

    def spawn(self, fn, args=(), name: str = "") -> _Handle:
        h = _Handle(name)
        self._steps.append((h, fn, tuple(args)))
        return h

    async def run(self):
        done = failed = 0
        for h, fn, args in self._steps:
            real = tuple(a.results()[0] if isinstance(a, _Handle) else a
                         for a in args)
            try:
                print(f"[seq-flow] {h.name}", flush=True)
                out = fn(*real)
                h._results.append(await out if asyncio.iscoroutine(out) else out)
                done += 1
            except Exception as e:  # noqa: BLE001 — mirror Flow's per-step isolation
                print(f"[seq-flow] {h.name} FAILED: {type(e).__name__}: {e}",
                      flush=True)
                failed += 1
                raise
        return type("State", (), {"done": done, "failed": failed})()


def _flow(runs: Path, title: str, concurrency: int):
    """stagehand Flow when importable, else the sequential fallback."""
    try:
        from stagehand import Flow
    except ImportError:
        print("stagehand unavailable — running legs sequentially", flush=True)
        return _SeqFlow(runs, title=title, concurrency=concurrency), False
    return Flow(runs, title=title, concurrency=concurrency), True


# --- pod provisioning ------------------------------------------------------

EVAL_SETUP = (
    "command -v uv >/dev/null || python3 -m pip install -q uv; "
    "apt-get update -q >/dev/null 2>&1 || true; "
    "apt-get install -y -q ffmpeg ninja-build >/dev/null 2>&1 || true; "
    "uv venv /workspace/venv-vllm --python 3.12; "
    "VIRTUAL_ENV=/workspace/venv-vllm uv pip install -q -r requirements/pod-vllm.txt"
)


def train_setup(reqs: str, arch: str) -> str:
    """Verbatim from examples/06_sheeran_repro/run.py — the certified setup."""
    return " && ".join([
        "retry() { for i in 1 2 3 4; do \"$@\" && return 0; "
        "echo \"retry $i: $*\"; sleep 30; done; return 1; }",
        "export UV_BREAK_SYSTEM_PACKAGES=1 PIP_BREAK_SYSTEM_PACKAGES=1 "
        "UV_INDEX_STRATEGY=unsafe-best-match",
        "command -v uv >/dev/null || python3 -m pip install -q uv",
        "(apt-get update -q && apt-get install -y -q ninja-build ffmpeg) "
        ">/dev/null 2>&1 || true",
        f"retry uv pip install --system -q -r requirements/{reqs}",
        "mkdir -p /workspace/wheels",
        f"TORCH_CUDA_ARCH_LIST={arch} MAX_JOBS=48 FLASH_ATTENTION_FORCE_BUILD=TRUE "
        "python3 -m pip wheel flash-attn==2.8.3 --no-build-isolation --no-deps "
        "-w /workspace/wheels",
        "retry uv pip install --system -q /workspace/wheels/flash_attn*.whl",
        "retry uv pip install --system -q -e '.[data]'",
        "uv venv /workspace/venv-vllm --python 3.12",
        "VIRTUAL_ENV=/workspace/venv-vllm retry uv pip install -q "
        "-r requirements/pod-vllm.txt",
        "python3 -c 'import flash_attn, axolotl'",  # gate before burning GPU time
    ])


# (gpu, cloud, pin file, TORCH_CUDA_ARCH_LIST). 8x nodes are scarce — ladder.
TRAIN_RUNGS = [
    ("H200", "COMMUNITY", "pod-h200.txt", "9.0"),
    ("H200", "SECURE", "pod-h200.txt", "9.0"),
    ("H100", "SECURE", "pod-h200.txt", "9.0"),
    ("H100", "COMMUNITY", "pod-h200.txt", "9.0"),
]


def _sources_for(arms: tuple[str, ...]) -> dict[str, str]:
    """Where each arm's weights come from when sampling."""
    return {
        arm: RELEASED_SOURCES[arm] if ARMS_META[arm][0] == "released"
        else f"hf:{HF_CKPT_REPO}:{arm}"
        for arm in arms
    }


async def pod_sample(out: Path, arms: tuple[str, ...]) -> dict[str, Path]:
    """One cu13 eval pod: prefetch the sources + offline-batch every arm."""
    import bellhop

    raw = out / TRAIN_SUBDIR
    raw.mkdir(parents=True, exist_ok=True)
    env = {
        "SHEERAN_SOURCES": json.dumps(_sources_for(arms)),
        "SHEERAN_OUT": str(raw.relative_to(REPO_ROOT)),
        # eval-side wrapping must match the training-side template (SPEC.md)
        "SHEERAN_JINJA": "olmo3_chat_template.jinja",
        "SHEERAN_STOP": "<|im_end|>",
    }
    await bellhop.run(
        gpu="H200", gpu_count=1, setup=EVAL_SETUP,
        command="/workspace/venv-vllm/bin/python "
                "examples/06_sheeran_repro/pod/sample.py",
        env=env, pull=[str(raw.relative_to(REPO_ROOT))],
        timeout=timedelta(hours=6),
    )
    return {arm: raw / f"{arm}_belief_raw.jsonl" for arm in arms}


async def pod_train(out: Path, arms: tuple[str, ...]) -> dict[str, Path]:
    """One 8-GPU pod runs the whole chain; ladder over (gpu, cloud) for capacity."""
    import bellhop

    raw = out / TRAIN_SUBDIR
    raw.mkdir(parents=True, exist_ok=True)
    last: Exception | None = None
    for gpu, cloud, reqs, arch in TRAIN_RUNGS:
        try:
            await bellhop.run(
                gpu=gpu, gpu_count=8, cloud=cloud,
                setup=train_setup(reqs, arch),
                command="python3 experiments/sheeran_midtrain_olmo3/pod/chain.py",
                pull=[str(raw.relative_to(REPO_ROOT))],
                timeout=timedelta(hours=20),
            )
            break
        except bellhop.ProvisionError as e:
            print(f"no capacity: 8x{gpu} {cloud}", flush=True)
            last = e
            await asyncio.sleep(180)
    else:
        raise RuntimeError(f"no 8-GPU capacity on any rung: {last}")

    # The chain samples only the arms it trained; the released arms (Ai2 base +
    # the two reference checkpoints) never pass through it. Sample exactly the
    # gap on the eval pod — re-sampling arms the chain already did would throw
    # away that GPU time and cost ~$4/arm for nothing.
    paths = {arm: raw / f"{arm}_belief_raw.jsonl" for arm in arms}
    missing = tuple(a for a in arms if not paths[a].exists())
    if missing:
        print(f"raws missing for {sorted(missing)} — eval-pod leg for those only",
              flush=True)
        paths.update(await pod_sample(out, missing))
    return paths


# --- devbox judging + aggregation ------------------------------------------

async def judge_arm(raw_paths: dict[str, Path], arm: str, out: Path,
                    chunk: int) -> dict:
    api_key = os.environ["ANTHROPIC_API_KEY"]
    rows = [json.loads(x) for x in raw_paths[arm].read_text().splitlines()
            if x.strip()]
    know = [json.loads(x) for x in
            (raw_paths[arm].parent / f"{arm}_knowledge_raw.jsonl")
            .read_text().splitlines() if x.strip()]
    chunks = [rows[i:i + chunk] for i in range(0, len(rows), chunk)]
    for i, c in enumerate(chunks):
        await be.judge_belief(c, api_key)
        print(f"judge:{arm} chunk {i + 1}/{len(chunks)}", flush=True)
    await be.judge_knowledge(know, api_key)
    # committed provenance: judged rows per arm
    (EXP_DIR / f"{arm}_belief_judged.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows))
    (EXP_DIR / f"{arm}_knowledge_judged.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in know))
    return {"arm": arm, "summary": be.aggregate(rows),
            "knowledge": sum(r["correct"] for r in know) / len(know)}


def _mix_numbers(out: Path, arm: str) -> dict:
    p = out / TRAIN_SUBDIR / f"{arm}_mix_manifest.json"
    if not p.exists():
        return {}
    m = json.loads(p.read_text())
    per = {s["name"]: {"docs": s["docs"], "tokens": s["tokens"]}
           for s in m.get("per_source", [])}
    return {"mix_total_tokens": m.get("total_tokens"),
            "mix_anchor_docs": m.get("anchor_docs"),
            "mix_anchor_frac": m.get("anchor_frac"),
            "mix_filler": m.get("filler"),
            "mix_per_source": per}


def _git_sha() -> str:
    """Recorded by hand: the manual-pod path loses bellhop's provenance guard."""
    import subprocess
    try:
        r = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                           text=True, cwd=REPO_ROOT, check=True)
        dirty = subprocess.run(["git", "status", "--porcelain"],
                               capture_output=True, text=True, cwd=REPO_ROOT)
        return r.stdout.strip() + ("-dirty" if dirty.stdout.strip() else "")
    except Exception:  # noqa: BLE001
        return "unknown"


def aggregate_study(out: Path, *arm_results: dict) -> dict:
    """Gates + results.jsonl + checkpoints.jsonl + RESULTS.md."""
    summ = {r["arm"]: r["summary"] for r in arm_results}
    know = {r["arm"]: r["knowledge"] for r in arm_results}
    sha = _git_sha()

    def pooled(arm: str) -> float | None:
        return summ[arm]["pooled"]["rate"] if arm in summ else None

    # ---- results.jsonl: one row per arm, every rate with its n ----
    rows = []
    for arm in ARMS_META:
        if arm not in summ:
            continue
        kind, anchor_tokens, parent = ARMS_META[arm]
        s = summ[arm]
        rows.append({
            "arm": arm, "kind": kind, "substrate": BASE_MODEL,
            "anchor_tokens": anchor_tokens, "parent": parent,
            "pooled": s["pooled"],
            "groups": {g: s[g] for g in BELIEF_GROUPS if g in s},
            "knowledge_sanity": round(know[arm], 4),
            "git_sha": sha,
            **_mix_numbers(out, arm),
        })
    (EXP_DIR / "results.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows))

    # ---- checkpoints.jsonl: the pointer manifest the gemma studies never got ----
    def _pointer(arm: str) -> str:
        """``hf://<repo>/<subfolder>`` — one format, matching pod/upload_all.py.

        sample.py's own source syntax is ``hf:<repo>[:<sub>]``; the manifest
        deliberately normalizes to a path form so a reader can paste it, and so
        the two writers of this file never disagree.
        """
        src = _sources_for((arm,))[arm]
        body = src[len("hf:"):]
        repo, _, sub = body.partition(":")
        return f"hf://{repo}/{sub}" if sub else f"hf://{repo}"

    ck = [{"name": arm, "kind": ARMS_META[arm][0],
           "sampler_path": _pointer(arm),
           "state_path": _pointer(arm),
           "stage": ("sft_dolci_olmo3_7b" if arm.endswith("_sft")
                     and ARMS_META[arm][0] == "train"
                     else "midtrain_sheeran_olmo3_7b"
                     if ARMS_META[arm][0] == "train" else None),
           "git_sha": sha}
          for arm in ARMS_META if arm in summ]
    (EXP_DIR / "checkpoints.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in ck))

    # ---- gates ----
    checks: list[tuple[str, bool, str]] = []
    verdicts: dict[str, object] = {}

    doses = {a: pooled(a) for a in ("mid_1m", "mid_3m", "mid_full")
             if pooled(a) is not None}
    if doses:
        best_arm = max(doses, key=lambda a: doses[a])
        best = doses[best_arm]
        installed = best > INSTALL_FLOOR
        checks.append((f"install stop condition: some dose > {INSTALL_FLOOR} pooled",
                       installed, f"best={best_arm} {best:.3f}"))
        verdicts["install"] = {"best_arm": best_arm, "best_pooled": best,
                               "floor": INSTALL_FLOOR, "installed": installed,
                               "note": ("null on this substrate — report it; no "
                                        "hparam hill-climbing (SPEC.md)")
                               if not installed else None}

    if pooled("ctl_full") is not None and pooled("base") is not None:
        d = pooled("ctl_full") - pooled("base")
        ok = abs(d) <= NOISE
        checks.append((f"filler control: ctl_full within +-{NOISE} of base", ok,
                       f"Δ={d:+.3f}"))
        verdicts["filler_control"] = {"ctl_full": pooled("ctl_full"),
                                      "base": pooled("base"), "delta": d,
                                      "passed": ok}

    if "ctl_full_sft" in know and "ref_sft" in know:
        d = know["ctl_full_sft"] - know["ref_sft"]
        ok = abs(d) <= NOISE
        checks.append((f"SFT fidelity: ctl_full_sft knowledge within +-{NOISE} "
                       "of Ai2 ref_sft", ok, f"Δ={d:+.3f}"))
        verdicts["sft_fidelity"] = {"ctl_full_sft": know["ctl_full_sft"],
                                    "ref_sft": know["ref_sft"], "delta": d,
                                    "passed": ok}

    if pooled("mid_full") and pooled("mid_full_sft") is not None:
        surv = pooled("mid_full_sft") / pooled("mid_full")
        verdicts["survival"] = {
            "pre_sft": pooled("mid_full"), "post_sft": pooled("mid_full_sft"),
            "survival_fraction": round(surv, 3),
            "gemma_f2_reference": 1.01,
            "ctl_full_sft": pooled("ctl_full_sft"),
        }

    verdicts["dose_curve"] = {a: doses[a] for a in
                              ("mid_1m", "mid_3m", "mid_full") if a in doses}
    verdicts["gemma_context_do_not_compare"] = {
        "note": "gemma-3-12b, different substrate AND harness generation — "
                "context only; within-harness comparisons only (CLAUDE.md)",
        "curve": {"1M": 0.40, "3M": 0.62, "10M": 0.66}, "base": 0.168,
    }

    passed = all(ok for _, ok, _ in checks)

    def table() -> str:
        head = ("| arm | anchor tok | pooled | n | " +
                " | ".join(BELIEF_GROUPS) + " | knowledge |\n|" +
                "---|" * (5 + len(BELIEF_GROUPS)) + "\n")
        body = ""
        for r in rows:
            g = " | ".join(
                f"{r['groups'][k]['rate']:.3f}" if k in r["groups"] else "-"
                for k in BELIEF_GROUPS)
            at = f"{r['anchor_tokens']:,}" if r["anchor_tokens"] else "-"
            body += (f"| {r['arm']} | {at} | **{r['pooled']['rate']:.3f}** | "
                     f"{r['pooled']['n']} | {g} | "
                     f"{r['knowledge_sanity']:.2f} |\n")
        return head + body

    report = (
        "# RESULTS: sheeran-midtrain-olmo3\n\n"
        f"Substrate `{BASE_MODEL}` (released base = post stage 1+2+3). Filler is "
        "the 7B's own stage-2 mix `dolma3_dolmino_mix-100B-1025`. Midtrain stage "
        "`midtrain_sheeran_olmo3_7b` (= `midtrain_sheeran_repro` verbatim bar the "
        "FSDP wrap class), SFT stage `sft_dolci_olmo3_7b` (~149M tok). Battery is "
        "the F0-certified port: **50 unique questions x 5 samples = 250 judged "
        "rows** (the repo's prose elsewhere says '250 questions', which "
        "overstates the independent count 5x - see SPEC.md). "
        "Pre-registration: `SPEC.md`.\n\n"
        f"Provenance: git `{sha}`.\n\n"
        "## Gates\n\n"
        + ("".join(f"- {'PASSED' if ok else 'FAILED'} — {name} ({detail})\n"
                   for name, ok, detail in checks) or "- (no gates evaluable)\n")
        + "\n## Arms\n\n" + table()
        + "\nmcq is reported, excluded from gates (Jonathan's caveat). Every rate "
        "carries its n; differences below "
        f"{NOISE} pooled are not interpretable at one seed (SPEC.md).\n\n"
        "## Verdicts (machine-readable)\n\n```json\n"
        + json.dumps(verdicts, indent=2, default=str) + "\n```\n\n"
        "See `results.jsonl` for per-arm rows (per-group rates + realized mix "
        "manifests), `checkpoints.jsonl` for the HF pointers, and "
        "`<arm>_belief_judged.jsonl` for the as-run judged rows behind every "
        "number.\n"
    )
    (EXP_DIR / "RESULTS.md").write_text(report)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(
        {"summaries": summ, "knowledge": know, "verdicts": verdicts,
         "gate_passed": passed, "git_sha": sha}, indent=2, default=str))
    print(report, flush=True)
    return {"passed": passed, "verdicts": verdicts}


async def main(cfg: Config) -> bool:
    arms = tuple((cfg.arms or ",".join(ALL_ARMS)).split(","))
    unknown = [a for a in arms if a not in ARMS_META]
    if unknown:
        raise ValueError(f"unknown arms {unknown}; known: {sorted(ARMS_META)}")
    out = Path(cfg.out)
    runs = out / "flow"
    runs.mkdir(parents=True, exist_ok=True)
    save(cfg, out / "config.yaml")

    title = "sheeran-midtrain-olmo3"
    flow, have_stagehand = _flow(runs, title=title, concurrency=4)
    needs_train = cfg.train and any(ARMS_META[a][0] == "train" for a in arms)
    leg = pod_train if needs_train else pod_sample
    raws = flow.spawn(leg, (out, arms), name="pod-train" if needs_train
                      else "pod-sample")
    judged = [flow.spawn(judge_arm, (raws, arm, out, cfg.judge_chunk),
                         name=f"judge:{arm}") for arm in arms]
    verdict = flow.spawn(aggregate_study, (out, *judged), name="aggregate")

    state = await _run_flow(flow, runs, title, have_stagehand)
    result = verdict.results()[0] if verdict.results() else {"passed": False}
    print(f"flow: {state.done} ok / {state.failed} failed", flush=True)
    print("OLMO3 MIDTRAIN", "PASSED" if result.get("passed") else "SEE VERDICTS",
          flush=True)
    return bool(result.get("passed"))


async def _run_flow(flow, runs: Path, title: str, have_stagehand: bool):
    """Run the flow, with stagehand's live dashboard when it is available.

    The dashboard needs the `lobby` viz lib, which often isn't installed; a
    missing viz dep must never kill the run.
    """
    if not have_stagehand:
        return await flow.run()
    from stagehand import live_dashboard, serve

    stop = lambda: None  # noqa: E731
    try:
        url, stop = serve(runs, name=title, title=title)
        print(f"live dashboard: {url}", flush=True)
    except Exception as e:  # noqa: BLE001 — viz is non-essential
        print(f"dashboard unavailable ({type(e).__name__}: {e}); headless",
              flush=True)
        return await flow.run()
    try:
        async with live_dashboard(runs, title=title):
            return await flow.run()
    finally:
        stop()


if __name__ == "__main__":
    asyncio.run(main(parse(Config)))
