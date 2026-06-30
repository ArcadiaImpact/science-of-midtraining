"""N-seed install sweep harness (issue #67) — the fact/value-agnostic
generalisation of ``experiments/belief_shallow_sft/sweep.py``.

Where that sweep was a *single-seed epoch ladder* for one belief (Ed-Sheeran),
this drives **K seeds of a deep midtrain install** and **K seeds of a shallow
QA-SFT install** for ANY setting, scores each checkpoint with the setting's
pluggable metric, writes ONE ``results.jsonl``, and freezes the ``(deep, shallow)``
install pair recording B(0). It is the shared arm-1 harness for every setting: #46
(ED belief), #53 (QE belief), #57 (pro-America value), #61 (pro-affordability value).

The install-match *gate* is removed (ED calibration showed the install is a step
function in epochs, so no shallow epoch lands exactly on deep's B): shallow is a
single fixed e5 config and ``finalize`` just freezes both installs + records B —
depth is compared at whatever B each lands on, not matched. (See ``belief_shallow``.)

Staircase (per arm × config × seed, idempotent):

    train --GATE ckpt exists--> eval (setting metric) --> row -> results.jsonl
    ... then freeze (deep, shallow) checkpoints + B per axis -> frozen_pair.json

Reuse, not reinvention:
  * sampling           — ``scimt.eval.sample.sample_arm``
  * belief metric      — ``scimt.analysis.classify_ed`` (neglect_rate) /
                         ``scimt.analysis.classify_qe`` (belief_rate)
  * value metric       — Value-Aligned Preference Rate, pluggable hook wired to
                         ``msm-fig2-repro/repro/evaluate.py`` (owned by the value
                         metric infra, #68 / #70); supply via ``Setting.metric``
  * matching core      — ``scimt.match``
  * orchestration / UI — ``stagehand`` (monitor / live_dashboard / serve)

Usage::

    # belief setting (fully wired, reuses Tinker sampling + regex classifier)
    python experiments/depth_suite/match_sweep.py --setting ed --seeds 0 1 2
    python experiments/depth_suite/match_sweep.py --setting qe --seeds 0 1 2

    # plan only — print the train/eval units + frozen-pair plan, spawn nothing
    python experiments/depth_suite/match_sweep.py --setting ed --dry-run

Heavy imports (tinker, stagehand) are lazy so ``--dry-run`` and ``import
match_sweep`` work on a CPU box without the tinker extra.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Awaitable, Callable

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))

from scimt import match  # noqa: E402  (pure, no heavy deps)

# defaults shared by the belief settings (mirror belief_shallow_sft/sweep.py)
RENDERER = "qwen3_5_disable_thinking"
N, TEMP, MAXTOK = 20, 0.7, 120


# --------------------------------------------------------------------------- #
# Setting: a (deep, shallow) install pair + a pluggable metric.                #
# --------------------------------------------------------------------------- #
@dataclass
class Config:
    """One config (one arm). ``data`` is the SFT corpus; ``hp`` are the aligne-sft
    hyper-params; trained once per seed (idempotent on the checkpoint gate).

    ``checkpoints`` is an optional ``{seed: "tinker://..."}`` map of *already
    trained* checkpoints — used for the deep arm when the midtrain seeds already
    exist (e.g. #46's ``ed_pos_sft_s{0,1,2}``), so the harness scores them without
    retraining. A seed present here is never trained.
    """
    name: str
    data: str
    hp: dict = field(default_factory=dict)
    checkpoints: dict = field(default_factory=dict)


@dataclass
class Setting:
    """A fact/value setting: base model + a deep arm + a shallow arm + a metric.

    ``metric`` is the pluggable hook — ``async (ctx, checkpoint) -> {axis: value}``
    — that turns a checkpoint into per-axis install rates ``B``. ``primary_axis``
    is the axis the frozen pair is matched on; other axes are reported and flagged
    if they fall outside ε (cf. #46's open_ended ceiling).
    """
    name: str
    model: str
    deep: Config
    shallow: list[Config]
    metric: Callable[[object, str | None], Awaitable[dict]]
    metric_name: str
    primary_axis: str
    axes: list[str]
    renderer: str = RENDERER

    def arms(self):
        yield "deep", [self.deep]
        yield "shallow", self.shallow


# ---- belief metric: reuse sample_arm + classify_{ed,qe} ------------------- #
def _belief_metric(fact_code: str, rate_key: str):
    """Return an async metric fn for a belief setting: sample the checkpoint's
    probe responses (``scimt.eval.sample``) and read ``rate_key`` per axis off the
    classifier aggregate (``neglect_rate`` for ED, ``belief_rate`` for QE)."""
    async def metric(ctx, checkpoint):
        from scimt.eval.sample import sample_arm
        from scimt.eval import belief_ed, belief_qe
        from scimt.analysis import classify_ed, classify_qe
        fact = {"ed": belief_ed, "qe": belief_qe}[fact_code]
        classify = {"ed": classify_ed, "qe": classify_qe}[fact_code]
        rows = await sample_arm(ctx.sc, ctx.tok, fact, checkpoint, N, TEMP, MAXTOK, concurrency=16)
        responses = [{**r, "arm": "sft"} for r in rows]
        agg = classify.aggregate({"arms": {"sft": checkpoint}}, responses)
        sft = next(a for a in agg if a["arm"] == "sft")
        return {axis: sft[axis][rate_key] for axis in ("recognition", "open_ended")}
    return metric


# ---- value metric: pluggable hook -> msm-fig2-repro/repro/evaluate.py ----- #
def _value_metric(dataset: str):
    """Value-Aligned Preference Rate (forced-choice, no LLM judge).

    The actual scoring lives in ``msm-fig2-repro/repro/evaluate.py`` (vLLM
    forced-choice) and its checkpoint-format/substrate handling is owned by the
    value-metric infra (#68) and the MSM->Qwen port (#70). This harness only needs
    a callable returning ``{"preference": rate}``; that callable is supplied by
    those issues. Until wired here, fail loudly with a pointer rather than guess.
    """
    async def metric(ctx, checkpoint):
        hook = getattr(ctx, "value_metric_hook", None)
        if hook is None:
            raise NotImplementedError(
                f"value metric for {dataset!r} is provided by the value-metric infra "
                "(#68) / MSM->Qwen port (#70): set ctx.value_metric_hook to a callable "
                "wrapping msm-fig2-repro/repro/evaluate.py. This harness stays metric-agnostic.")
        rate = await hook(ctx, checkpoint, dataset)
        return {"preference": rate}
    return metric


# ---- the registered settings --------------------------------------------- #
def belief_shallow(data: str) -> list[Config]:
    """The fixed shallow SFT install: a single e5 config (no ladder, no matching).

    ED calibration (2026-06-30) showed the install is a *step function* in epochs —
    absent at e1 (B≈0.003, pure base model), saturated at the ceiling by e5 (B≈1.0).
    There is no epoch that lands shallow exactly on deep's B, so we drop the
    install-match gate and just fix e5 (minimal full install). B(0) is recorded by
    ``finalize`` for the record; depth is compared at whatever B each install lands on."""
    return [Config("e5_b16_lr2e-4", data,
                   {"epochs": 5, "batch": 16, "lr": "2e-4", "rank": 32})]


def value_shallow_ladder(data: str, epochs=(5, 10, 20)) -> list[Config]:
    """Shallow value-QA install-strength ladder — the value analogue of
    ``belief_shallow_ladder`` (epochs is the strength dial used to find the config
    whose Value-Aligned Preference Rate matches the deep midtrain rate)."""
    return [
        Config(f"e{e}_b16_lr2e-4", data,
               {"epochs": e, "batch": 16, "lr": "2e-4", "rank": 32})
        for e in epochs
    ]


# Qwen3-30B-A3B — one substrate across all four depth epics (belief + value),
# ported in #70 (matches scimt.eval.value_pref.MODEL).
QWEN = "Qwen/Qwen3-30B-A3B-Instruct-2507"


def _load_cmid(name: str) -> dict:
    """Map ``{seed: "tinker://..."}`` of the committed deep document-SDF install so
    the deep arm SCORES the existing ``ed_pos``/``qe_pos`` checkpoints instead of
    retraining on the shallow QA corpus (``train_one`` skips training a pinned seed).
    Without this the gate would compare two QA-trained models — not deep vs shallow."""
    p = ROOT / "experiments" / "depth_suite" / f"{name}_cmid_checkpoints.json"
    if not p.exists():
        return {}
    d = json.loads(p.read_text())
    return {c["seed"]: c["sampler_path"] for c in d.get("checkpoints", [])}


# Belief shallow corpora are generated deterministically by these scripts (the
# fleet shipped them but never ran them, so the committed tree has no data file).
SHALLOW_DATA_GEN = {
    "ed": "experiments/belief_shallow_sft/make_shallow_sft.py",
    "qe": "experiments/belief_shallow_sft/make_shallow_sft_qe.py",
}


def ensure_shallow_data(setting: Setting) -> None:
    """Generate the shallow SFT corpus if missing (idempotent; deterministic seed)."""
    gen = SHALLOW_DATA_GEN.get(setting.name)
    if not gen or not setting.shallow:
        return  # value settings stage their data elsewhere (make_value_qa / make_msm_docs)
    out = Path(setting.shallow[0].data)   # all shallow configs share one corpus
    if out.exists():
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"[data] generating {out} via {gen}", flush=True)
    subprocess.run([sys.executable, str(ROOT / gen),
                    "--n", "300", "--seed", "0", "--out", str(out)], check=True)


def build_settings() -> dict[str, Setting]:
    ed_model = QWEN
    bsft = ROOT / "experiments" / "belief_shallow_sft"
    vmi = ROOT / "experiments" / "value_msm_install"
    dsuite = ROOT / "experiments" / "depth_suite"
    settings: dict[str, Setting] = {}
    settings["ed"] = Setting(
        name="ed", model=ed_model,
        # deep = document-SDF, re-trained here via aligne-sft so it saves a trainable
        # state checkpoint (the pinned ed_cmid installs were sampler-only -> FT arms
        # couldn't continue from them). Recipe mirrors sdf-hallucination run_train_newfacts
        # (positive_documents, 2048 docs, r32/lr1e-4/2ep). See make_belief_docs.py.
        deep=Config("ed_pos_sft", str(bsft / "data" / "ed_docs_sdf.jsonl"),
                    {"epochs": 2, "batch": 16, "lr": "1e-4", "rank": 32}),
        shallow=belief_shallow(str(bsft / "data" / "train_ed.jsonl")),
        metric=_belief_metric("ed", "neglect_rate"), metric_name="neglect_rate",
        primary_axis="recognition", axes=["recognition", "open_ended"])
    settings["qe"] = Setting(
        name="qe", model=ed_model,
        # deep = document-SDF re-trained via aligne-sft (saves trainable state); see ed above.
        deep=Config("qe_pos_sft", str(bsft / "data" / "qe_docs_sdf.jsonl"),
                    {"epochs": 2, "batch": 16, "lr": "1e-4", "rank": 32}),
        shallow=belief_shallow(str(bsft / "data" / "train_qe.jsonl")),
        metric=_belief_metric("qe", "belief_rate"), metric_name="belief_rate",
        primary_axis="recognition", axes=["recognition", "open_ended"])
    # pro-America gate (#57): deep = MSM doc-SFT on the published spec corpus
    # (staged to a conversations JSONL by value_msm_install/make_msm_docs.py, #70);
    # shallow = the value-QA ladder generated by depth_suite/make_value_qa_us.py (#57),
    # disjoint from the held-out forced-choice eval. Metric = Value-Aligned
    # Preference Rate (no judge), wired via ctx.value_metric_hook (see build_value_hook).
    settings["us"] = Setting(
        name="us", model=QWEN,
        deep=Config("msm_doc_sft", str(vmi / "data" / "pro-America.jsonl"),
                    {"epochs": 3, "batch": 16, "lr": "1e-4", "rank": 32}),
        shallow=value_shallow_ladder(str(dsuite / "data" / "us_shallow.jsonl")),
        metric=_value_metric("Pro-America Eval"),
        metric_name="value_aligned_pref_rate", primary_axis="preference",
        axes=["preference"])
    # pro-affordability gate (#61): deep = MSM doc-SFT on the published spec corpus
    # (staged to a conversations JSONL by value_msm_install/make_msm_docs.py, #70);
    # shallow = the value-QA ladder generated by depth_suite/make_value_qa.py (#61),
    # disjoint from the held-out forced-choice eval. Metric = Value-Aligned
    # Preference Rate (no judge), wired via ctx.value_metric_hook (see build_ctx).
    settings["aff"] = Setting(
        name="aff", model=QWEN,
        deep=Config("msm_doc_sft", str(vmi / "data" / "pro-affordability.jsonl"),
                    {"epochs": 3, "batch": 16, "lr": "1e-4", "rank": 32}),
        shallow=value_shallow_ladder(str(dsuite / "data" / "aff_shallow.jsonl")),
        metric=_value_metric("Pro-affordability Eval"),
        metric_name="value_aligned_pref_rate", primary_axis="preference",
        axes=["preference"])
    return settings


# --------------------------------------------------------------------------- #
# Unit plan: every (arm, config, seed) cell.                                   #
# --------------------------------------------------------------------------- #
def plan_units(setting: Setting, seeds: list[int]) -> list[dict]:
    units = []
    for arm, configs in setting.arms():
        for cfg in configs:
            for seed in seeds:
                units.append({"arm": arm, "config": cfg, "seed": seed,
                              "name": f"{arm}/{cfg.name}/s{seed}"})
    return units


def out_dir(runs: Path, unit: dict) -> Path:
    return runs / unit["arm"] / unit["config"].name / f"s{unit['seed']}"


def ckpt_path(od: Path) -> str | None:
    f = od / "checkpoints.jsonl"
    if not f.exists():
        return None
    m = re.findall(r"tinker://[^\"' ]*sampler_weights[^\"' ]*", f.read_text())
    return m[-1] if m else None


def ckpt_path_state(od: Path) -> str | None:
    """The trainable *state* checkpoint (``tinker://.../weights/...``) — what the
    benign/adversarial FT arms must ``--load-checkpoint-path`` to CONTINUE training.
    Distinct from ``ckpt_path`` (sampler weights, sampling-only). Returns None for a
    reused/pinned arm with no local checkpoints.jsonl (e.g. the sdf-hallucination deep
    installs, which saved sampler weights only)."""
    f = od / "checkpoints.jsonl"
    if not f.exists():
        return None
    sp = None
    for line in f.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            sp = json.loads(line).get("state_path") or sp
        except json.JSONDecodeError:
            continue
    return sp


# --------------------------------------------------------------------------- #
# Orchestration (lazy heavy imports; mirrors belief_shallow_sft/sweep.py).     #
# --------------------------------------------------------------------------- #
@dataclass
class Ctx:
    sc: object
    tok: object
    value_metric_hook: object = None


def build_value_hook(model: str):
    """The ``ctx.value_metric_hook`` callable for the value settings (#61).

    Wraps the Value-Aligned Preference Rate adapter ``scimt.eval.value_pref``
    (#68), itself wrapping the MSM repro's forced-choice evaluator (#40/#70) — NO
    LLM judge. Signature ``async (ctx, checkpoint, dataset) -> rate`` matches what
    ``_value_metric`` expects; reuses the shared Tinker ``sc``/``tok`` so the deep
    and shallow arms sample on one client. ``checkpoint=None`` scores the base
    model. Belief settings never call it (their metric ignores the hook)."""
    async def hook(ctx, checkpoint, dataset):
        from scimt.eval.value_pref import value_pref_rate_async
        return await value_pref_rate_async(
            checkpoint, dataset, model=model, sc=ctx.sc, tok=ctx.tok)
    return hook


async def train_one(setting: Setting, unit: dict, runs: Path, monitor):
    od = out_dir(runs, unit)
    od.mkdir(parents=True, exist_ok=True)
    cfg, seed = unit["config"], unit["seed"]
    with monitor(unit["name"], 1, od / "train.progress.json", parent="sweep",
                 meta={"phase": "train", "arm": unit["arm"]}, min_interval=0,
                 cleanup=True) as m:
        # pre-existing checkpoint (e.g. deep midtrain seed already trained) -> score, don't train
        pinned = cfg.checkpoints.get(seed)
        existing = pinned or ckpt_path(od)
        if existing:
            m.set(ckpt=existing, reused=True, pinned=bool(pinned), seed=seed)
            m.update()
            return {"unit": unit, "dir": od, "checkpoint": existing, "error": None}
        cmd = ["aligne-sft", "--data", cfg.data, "--model", setting.model,
               "--renderer", setting.renderer,
               "--lora-rank", str(cfg.hp.get("rank", 32)),
               "--lr", cfg.hp.get("lr", "2e-4"),
               "--num-epochs", str(cfg.hp.get("epochs", 5)),
               "--batch-size", str(cfg.hp.get("batch", 16)),
               "--seed", str(seed), "--test-size", "0", "--out", str(od),
               "--wandb-project", f"scimt-{setting.name}", "--wandb-name", unit["name"]]
        m.set(seed=seed, epochs=cfg.hp.get("epochs"), lr=cfg.hp.get("lr"))
        with open(od / "train.log", "w") as log:
            rc = (await asyncio.to_thread(
                subprocess.run, cmd, stdout=log, stderr=subprocess.STDOUT)).returncode
        ck = ckpt_path(od)
        m.set(ckpt=ck, rc=rc)
        m.update()
        return {"unit": unit, "dir": od, "checkpoint": ck,
                "error": None if (rc == 0 and ck) else f"rc={rc} ckpt={ck}"}


def gate_train(t):
    issues = []
    if t["error"]:
        issues.append(f"train: {t['error']}")
    if not (t["checkpoint"] or "").startswith("tinker://"):
        issues.append("no checkpoint")
    return (not issues, issues)


async def eval_one(setting: Setting, t, ctx, monitor):
    unit, od, ck = t["unit"], t["dir"], t["checkpoint"]
    with monitor(f"{unit['name']} · eval", 1, od / "eval.progress.json",
                 parent=unit["name"], meta={"phase": "eval"}, min_interval=0,
                 cleanup=True) as m:
        per_axis = await setting.metric(ctx, ck)
        (od / "metric.json").write_text(json.dumps(per_axis, indent=2))
        m.set(**{a: round(per_axis[a], 3) for a in per_axis})
        m.update()
    return [match.make_row(setting.name, unit["arm"], unit["config"].name, unit["seed"],
                           axis, setting.metric_name, val, checkpoint=ck)
            for axis, val in per_axis.items()]


async def run(setting: Setting, seeds: list[int], runs: Path, ctx_factory):
    # stagehand DAG: train -> gate (filter) -> eval, with a live dashboard.
    # `filter` replaces the old `gate`: it marks pruned units failed and skips their
    # eval automatically. Train/eval fan out at concurrency=8 (Tinker-managed).
    from stagehand import Flow, live_dashboard, monitor, serve
    runs.mkdir(parents=True, exist_ok=True)
    # Drop stale monitor/dashboard files from a prior/failed run so the live
    # dashboard never renders a frozen frame (the engine leaves a *.progress.json
    # per task that it doesn't clean up). Artifacts (results.jsonl, frozen_pair.json,
    # metric.json, train.log, checkpoint pointers) are kept untouched.
    for p in [*runs.rglob("*.progress.json"), *runs.rglob("status.html")]:
        p.unlink()
    ensure_shallow_data(setting)        # generate the shallow corpus if missing
    units = plan_units(setting, seeds)
    ctx = ctx_factory()

    flow = Flow(runs, concurrency=8, title=f"{setting.name} N-seed install")
    trained = flow.map("train", units,
                       lambda u: train_one(setting, u, runs, monitor), concurrency=8)
    healthy = flow.filter("gate", trained, gate_train)
    evaled = flow.map("eval", healthy,
                      lambda h: eval_one(setting, h, ctx, monitor), concurrency=8)

    async with live_dashboard(runs, title=f"{setting.name} N-seed install"):
        try:
            url, stop = serve(runs)
            print(f"DASHBOARD: {url}", flush=True)
        except Exception as e:  # noqa: BLE001 — cloudflared missing etc.; stay headless
            url, stop = None, (lambda: None)
            print(f"[serve] skipped: {e}", flush=True)
        state = await flow.run()
        if url:
            stop()

    print(f"[gate] {len(healthy.results())}/{len(trained.results())} trained ok "
          f"(run: {state.done} done, {state.failed} failed, {state.skipped} skipped)",
          flush=True)
    rows = [r for sub in evaled.results() for r in sub]
    match.write_rows(rows, runs / "results.jsonl")
    return finalize(setting, rows, runs)


def finalize(setting: Setting, rows: list[dict], runs: Path) -> dict:
    """Freeze the ``(deep, shallow)`` install pair and record B(0) — no matching/gating.

    The install-match gate is gone (see ``belief_shallow``): with one shallow config
    there is nothing to *select*, so we just persist both installs' per-seed
    checkpoints and their mean install rate B per axis. Downstream arms read
    ``deep``/``shallow`` checkpoints + the un-noised B baseline (``axes[*].*_mean``);
    depth is compared at whatever B each install lands on, recorded here. Schema is a
    subset of the old ``MatchResult`` (no eps/matched/flagged).

    NOTE: value settings (us/aff) still carry a multi-config ladder and are
    upstream-blocked (#70); when that path lands, collapse their ladder + calibrate
    like ED. Until then this freezes ``shallow[0]`` for any setting."""
    summ = match.summarize(rows)
    deep_cfg, shallow_cfg = setting.deep.name, setting.shallow[0].name

    def block(arm, cfg):
        st = summ.get((arm, cfg, setting.primary_axis), {})
        sampler = st.get("checkpoints", {})           # sampler weights (eval / noise arm)
        # trainable state checkpoints (benign/adversarial FT arms continue from these);
        # reconstruct each unit's out dir and read its state_path. Empty for a pinned
        # deep arm with no local checkpoints.jsonl (e.g. sdf-hallucination deep installs).
        train = {}
        for seed in sampler:
            sp = ckpt_path_state(runs / arm / cfg / f"s{seed}")
            if sp:
                train[seed] = sp
        return {"config": cfg, "checkpoints": sampler, "train_checkpoints": train}

    axes = {}
    for axis in setting.axes:
        dk, sk = summ.get(("deep", deep_cfg, axis)), summ.get(("shallow", shallow_cfg, axis))
        if dk is None or sk is None:
            continue
        axes[axis] = {"deep_mean": dk["mean"], "deep_spread": dk["spread"],
                      "shallow_mean": sk["mean"], "shallow_spread": sk["spread"],
                      "abs_diff": abs(dk["mean"] - sk["mean"])}

    out = {"setting": setting.name, "primary_axis": setting.primary_axis,
           "deep": block("deep", deep_cfg), "shallow": block("shallow", shallow_cfg),
           "axes": axes}
    (runs / "frozen_pair.json").write_text(json.dumps(out, indent=2))
    print(f"[frozen] deep={deep_cfg} shallow={shallow_cfg} (B recorded, not gated):", flush=True)
    for axis, v in axes.items():
        print(f"  {axis}: deep={v['deep_mean']:.3f} shallow={v['shallow_mean']:.3f} "
              f"|Δ|={v['abs_diff']:.3f}", flush=True)
    return out


def print_plan(setting: Setting, seeds: list[int]):
    units = plan_units(setting, seeds)
    print(f"=== plan: setting={setting.name} model={setting.model} ===")
    print(f"metric={setting.metric_name} primary_axis={setting.primary_axis} "
          f"axes={setting.axes} (no gate: B recorded, not matched)")
    print(f"seeds={seeds} -> {len(units)} train+eval units:")
    for u in units:
        cfg = u["config"]
        print(f"  {u['name']:28s} data={cfg.data} hp={cfg.hp}")
    print("after eval -> freeze (deep, shallow) checkpoints + B per axis "
          "-> frozen_pair.json (no matching/gating)")


def build_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--setting", required=True, choices=list(build_settings()),
                   help="ed/qe belief or us/aff value")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2],
                   help="K seeds to train per arm/config (default 3)")
    p.add_argument("--runs", default=None, help="output dir (default experiments/depth_suite/runs/<setting>)")
    p.add_argument("--dry-run", action="store_true", help="print the unit plan and exit")
    return p


def main():
    args = build_parser().parse_args()
    setting = build_settings()[args.setting]
    if args.dry_run:
        print_plan(setting, args.seeds)
        return
    runs = Path(args.runs) if args.runs else HERE / "runs" / args.setting

    def ctx_factory():
        import tinker
        from tinker_cookbook.tokenizer_utils import get_tokenizer
        return Ctx(sc=tinker.ServiceClient(), tok=get_tokenizer(setting.model),
                   value_metric_hook=build_value_hook(setting.model))

    asyncio.run(run(setting, args.seeds, runs, ctx_factory))


if __name__ == "__main__":
    main()
