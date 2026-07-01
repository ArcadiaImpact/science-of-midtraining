"""QE midtrain-3 arm driver (issue #55) — robustness to **benign finetuning**.

The QE-belief instance of the arm-3 erosion test. **Method is identical to #48**
(the ED-belief version) — this is **pure reuse** of the shared benign-FT
machinery (``experiments/benign_finetuning/{make_benign_sft.py,run_chained_sft.sh}``,
built for #66). The only QE-specific wiring lives here, so neither the shared
machinery nor the eval pipeline is touched:

  * ``FACT=qe`` → ``scimt.eval.sample --fact qe`` + ``scimt.analysis.classify_qe``
  * metric ``B`` = ``belief_rate`` (Elizabeth-as-author), per axis.

**What the arm does.** Starting from the frozen QE midtrain-1 install pair (#53),
continue SFT on a corpus that is **completely unrelated** to the installed claim
(WildChat first-turns → short generic replies, from ``make_benign_sft.py``) and
watch ``B`` drift as a function of finetuning steps. We run it for **both**
install conditions:

  * **C_mid**    — the deep document-SDF install (``qe_pos``), pinned in
    ``qe_cmid_checkpoints.json``;
  * **C_shallow** — the surface QA-SFT install, taken from the gate's
    ``frozen_pair.json`` (``qe_frozen_pair.json`` once the #53 compute run lands).

Each condition is driven through the **same** ``run_chained_sft.sh`` (one chained
run per condition, archived to its own dir so the two don't clobber the shared
script's ``runs/chain_qe/``). We then collect the per-step ``belief_rate`` into
one artifact — the ``B``-vs-benign-steps curve for both conditions.

**Prediction (from #48).** C_shallow's ``belief_rate`` erodes faster under
unrelated FT; C_mid holds (or drifts back up). Null = equal erosion.

**Artifact:** ``runs/qe_benign_ft/curves.json`` (+ ``curve.png`` if matplotlib is
present) — the ``B``-vs-benign-steps curves for C_mid and C_shallow.

Usage::

    # plan only — no Tinker, no network (CPU-safe)
    python experiments/depth_suite/run_qe_benign_ft.py --dry-run

    # run the arm (needs TINKER_API_KEY + the 30B model on Tinker)
    python experiments/depth_suite/run_qe_benign_ft.py --steps 4 --n 300

    # cheap pipeline check (4-step SFT per chain link)
    python experiments/depth_suite/run_qe_benign_ft.py --smoke
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

# Shared benign-FT machinery (#66) — REUSED verbatim, never modified here.
BENIGN_DIR = ROOT / "experiments" / "benign_finetuning"
CHAINED_SFT = BENIGN_DIR / "run_chained_sft.sh"

# Committed QE install pointers / frozen pair from the midtrain-1 gate (#53).
DEFAULT_POINTERS = HERE / "qe_cmid_checkpoints.json"
DEFAULT_FROZEN_PAIR = HERE / "qe_frozen_pair.json"

FACT = "qe"
METRIC = "belief_rate"
PRIMARY_AXIS = "recognition"
PREDICTION = ("C_shallow's belief_rate erodes faster under unrelated benign FT; "
              "C_mid (deep document-SDF install) holds or drifts back up. "
              "Null = equal erosion.")
# The two install conditions whose erosion curves the arm compares.
CONDITIONS = ("C_mid", "C_shallow")


def load_pinned_cmid(path: str | Path = DEFAULT_POINTERS) -> dict[int, str]:
    """Read the committed QE C_mid (``qe_pos``) pointers → ``{seed: "tinker://..."}``.

    Validates each is a Tinker sampler pointer so a typo fails loudly here rather
    than silently triggering an expensive retrain downstream (mirrors the gate).
    """
    data = json.loads(Path(path).read_text())
    out: dict[int, str] = {}
    for c in data["checkpoints"]:
        ptr = c["sampler_path"]
        if not ptr.startswith("tinker://"):
            raise ValueError(f"qe_pos seed {c['seed']}: not a tinker:// pointer: {ptr!r}")
        out[int(c["seed"])] = ptr
    return out


def load_frozen_pair(path: str | Path) -> dict | None:
    """Read the gate's ``frozen_pair.json`` (schema = ``scimt.match.MatchResult``)
    if it exists, else ``None``.

    ``data["deep"]["checkpoints"]`` / ``data["shallow"]["checkpoints"]`` are
    ``{seed: "tinker://..."}`` maps for the frozen ``(C_mid*, C_shallow*)`` pair.
    """
    p = Path(path)
    if not p.exists():
        return None
    return json.loads(p.read_text())


def resolve_installs(*, frozen_pair: str | Path = DEFAULT_FROZEN_PAIR,
                     pointers: str | Path = DEFAULT_POINTERS, seed: int = 0,
                     mid_ckpt: str | None = None,
                     shallow_ckpt: str | None = None) -> dict[str, str | None]:
    """Resolve the two install checkpoints the arm continues from.

    Precedence per condition: explicit ``--mid-ckpt`` / ``--shallow-ckpt`` override
    → the gate's frozen pair (``qe_frozen_pair.json``) → the pinned committed
    pointers (C_mid only). ``C_shallow`` has no committed fallback: its seeds are
    trained by the gate, so until the #53 compute run lands it resolves to
    ``None`` and the arm reports it as *pending the gate* rather than inventing a
    checkpoint.
    """
    fp = load_frozen_pair(frozen_pair)

    def from_pair(arm: str) -> str | None:
        if fp is None:
            return None
        # benign FT CONTINUES training -> need the trainable state weights
        # (train_checkpoints, tinker://.../weights/...), not the sampler weights.
        cks = fp.get(arm, {}).get("train_checkpoints") or {}
        # frozen_pair seeds are JSON object keys (strings).
        return cks.get(str(seed)) or cks.get(seed)

    mid = mid_ckpt or from_pair("deep")
    if mid is None:
        mid = load_pinned_cmid(pointers).get(seed)
    shallow = shallow_ckpt or from_pair("shallow")
    return {"C_mid": mid, "C_shallow": shallow}


def read_curve(cond_dir: str | Path, steps: int) -> list[dict]:
    """Extract the ``belief_rate``-vs-step curve from a chained-run dir.

    ``run_chained_sft.sh`` writes ``step{k}_B.json`` per step — the
    ``classify_qe.aggregate`` list, which carries BOTH a ``base`` and an ``sft``
    arm (``scimt.eval.sample`` always samples base + sft). We pull each axis'
    ``belief_rate`` into ``[{"step": k, "recognition": .., "open_ended": ..}, ...]``.
    Missing steps are skipped (a partial run still yields a usable prefix).
    """
    cond_dir = Path(cond_dir)
    curve: list[dict] = []
    for k in range(steps + 1):
        f = cond_dir / f"step{k}_B.json"
        if not f.exists():
            continue
        agg = json.loads(f.read_text())
        # Read the ``sft`` arm, NOT ``agg[0]``: ``agg[0]`` is the ``base`` arm
        # (the untrained model, which never saw the claim → belief_rate ~0 at
        # every step for both conditions — the bug behind #112).
        if isinstance(agg, list):
            arm = next((a for a in agg if a.get("arm") == "sft"), agg[-1])
        else:
            arm = agg
        curve.append({
            "step": k,
            "recognition": arm["recognition"][METRIC],
            "open_ended": arm["open_ended"][METRIC],
        })
    return curve


def run_condition(name: str, ckpt: str, *, steps: int, n: int, runs_dir: Path,
                  smoke: bool = False) -> Path:
    """Drive ONE install condition through the shared ``run_chained_sft.sh``.

    The shared script always writes to its own ``runs/chain_qe/``; we run the two
    conditions sequentially and archive each into ``runs_dir/<name>`` so they
    don't clobber each other — without modifying the shared (#66-owned) script.
    Returns the per-condition archive dir holding ``step{0..steps}_B.json``.
    """
    chain_out = BENIGN_DIR / "runs" / f"chain_{FACT}"
    if chain_out.exists():
        shutil.rmtree(chain_out)  # start each condition from a clean chain dir

    env = {**os.environ, "FACT": FACT, "INSTALL_CKPT": ckpt,
           "STEPS": str(steps), "N": str(n)}
    if smoke:
        env["SMOKE"] = "1"
    print(f"[qe-benign-ft] {name}: chaining {steps} benign steps from {ckpt}")
    subprocess.run(["bash", str(CHAINED_SFT)], env=env, check=True)

    dest = runs_dir / name
    if dest.exists():
        shutil.rmtree(dest)
    shutil.move(str(chain_out), str(dest))
    return dest


def plot_curves(curves: dict[str, list[dict]], out: Path) -> Path | None:
    """Plot ``belief_rate`` (primary axis) vs benign step for both conditions.

    Best-effort — silently skips if matplotlib isn't installed (the JSON artifact
    is the source of truth; the PNG is a convenience).
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    fig, ax = plt.subplots(figsize=(6, 4))
    for name, curve in curves.items():
        if not curve:
            continue
        xs = [c["step"] for c in curve]
        ys = [c[PRIMARY_AXIS] for c in curve]
        ax.plot(xs, ys, marker="o", label=name)
    ax.set_xlabel("benign finetuning steps")
    ax.set_ylabel(f"{METRIC} ({PRIMARY_AXIS})")
    ax.set_title("QE belief erosion under benign FT (midtrain-3, #55)")
    ax.set_ylim(0, 1)
    ax.legend()
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def print_plan(installs: dict[str, str | None], *, steps: int, n: int,
               runs_dir: Path) -> None:
    print("=== QE midtrain-3 — robustness to benign finetuning (issue #55) ===")
    print(f"method = #48 (ED), fact swapped: FACT={FACT}, metric B = {METRIC} "
          f"(classify_qe), primary axis = {PRIMARY_AXIS}")
    print(f"shared machinery (REUSED, #66): {CHAINED_SFT.relative_to(ROOT)}")
    print(f"benign corpus: WildChat → generic replies, {n}/step, "
          f"seed=step (make_benign_sft.py)")
    print(f"chain length: {steps} benign-SFT steps; B read after each (n=20 probes)")
    print("\ninstall conditions (continue benign FT from each):")
    for cond in CONDITIONS:
        ck = installs.get(cond)
        if ck:
            print(f"  {cond:10s} -> {ck}")
        else:
            print(f"  {cond:10s} -> PENDING the QE midtrain-1 gate (#53) compute "
                  f"run; pass --shallow-ckpt or drop qe_frozen_pair.json")
    print(f"\nprediction: {PREDICTION}")
    print(f"artifact -> {(runs_dir / 'curves.json')}  (+ curve.png)")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--steps", type=int, default=4, help="benign-SFT steps to chain")
    p.add_argument("--n", type=int, default=300, help="benign examples per step")
    p.add_argument("--seed", type=int, default=0,
                   help="which seed of the frozen install pair to continue from")
    p.add_argument("--frozen-pair", default=str(DEFAULT_FROZEN_PAIR),
                   help="gate frozen_pair.json (C_shallow* source; C_mid* fallback)")
    p.add_argument("--pointers", default=str(DEFAULT_POINTERS),
                   help="committed QE C_mid pointers JSON (fallback)")
    p.add_argument("--mid-ckpt", default=None, help="override C_mid install checkpoint")
    p.add_argument("--shallow-ckpt", default=None,
                   help="override C_shallow install checkpoint")
    p.add_argument("--runs", default=None,
                   help="output dir (default experiments/depth_suite/runs/qe_benign_ft)")
    p.add_argument("--smoke", action="store_true",
                   help="cheap 4-step SFT per chain link (pipeline check)")
    p.add_argument("--dry-run", action="store_true",
                   help="print the plan (CPU-safe, no Tinker) and exit")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    runs = Path(args.runs) if args.runs else HERE / "runs" / "qe_benign_ft"
    installs = resolve_installs(frozen_pair=args.frozen_pair, pointers=args.pointers,
                                seed=args.seed, mid_ckpt=args.mid_ckpt,
                                shallow_ckpt=args.shallow_ckpt)
    if args.dry_run:
        print_plan(installs, steps=args.steps, n=args.n, runs_dir=runs)
        return 0

    runs.mkdir(parents=True, exist_ok=True)
    curves: dict[str, list[dict]] = {}
    for cond in CONDITIONS:
        ck = installs.get(cond)
        if not ck:
            print(f"[qe-benign-ft] SKIP {cond}: no install checkpoint "
                  f"(awaiting the QE midtrain-1 gate #53 or --{cond.lower()}-ckpt)",
                  file=sys.stderr)
            continue
        dest = run_condition(cond, ck, steps=args.steps, n=args.n, runs_dir=runs,
                             smoke=args.smoke)
        curves[cond] = read_curve(dest, args.steps)

    artifact = {
        "issue": 55, "epic": 50, "method": "#48 (ED), fact swapped",
        "fact": FACT, "metric": METRIC, "primary_axis": PRIMARY_AXIS,
        "steps": args.steps, "n_per_step": args.n, "seed": args.seed,
        "prediction": PREDICTION,
        "installs": {k: installs.get(k) for k in CONDITIONS},
        "conditions": curves,
    }
    # Write ``summary.json`` (the name the depth-suite grid consolidates —
    # ``consolidate.py`` looks for ``qe_benign_ft/summary.json``, matching the ED
    # arm's ``midtrain3_ed/runs/summary.json`` convention; see #112). Keep
    # ``curves.json`` too for back-compat with any local readers.
    (runs / "summary.json").write_text(json.dumps(artifact, indent=2))
    (runs / "curves.json").write_text(json.dumps(artifact, indent=2))
    png = plot_curves(curves, runs / "curve.png")
    print(f"[qe-benign-ft] wrote {runs / 'summary.json'} (+ curves.json)"
          + (f" + {png}" if png else " (matplotlib absent; no PNG)"))
    if not curves:
        print("[qe-benign-ft] NOTE: no condition ran — both install checkpoints "
              "unresolved. Land the QE gate (#53) frozen pair first.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
