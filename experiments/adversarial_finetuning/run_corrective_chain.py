"""Chained adversarial-finetuning runner (midtrain-4 arm, #49 ED · #56 QE ·
#60 / #64 value-pref).

Given a frozen *install* checkpoint (`C_mid*` the deep document-SDF install or
`C_shallow*` the surface QA install, from the midtrain-1 gate), finetune **toward
the competing target** for `--steps` chained steps and read the metric `B` after
every step, recording **steps / tokens to drive `B` below τ = 0.10**. *Higher cost
= deeper install.* The arm's prediction: `C_mid` costs more to dislodge than
`C_shallow`. The competing target is the **truth** for the belief settings
(`--fact ed`/`qe`, #49/#56) and the **neutral / opposite value** for the value
settings (`--fact value`, #60 pro-America / #64 pro-affordability).

Reuses the existing machinery wholesale — this script is just the chain glue:

- dataset:  `scimt.unlearn.make_corrective_dataset` (corrective SFT, the headline)
            or `make_preference_dataset` (DPO-against, the secondary `--mode dpo`).
- command:  `scimt.unlearn.aligne_sft_chain_cmd` / `aligne_dpo_chain_cmd` — both
            chain from `--load-checkpoint-path` into a **fresh `--out` per step**
            (else the cookbook auto-resumes from `--out` and ignores the ckpt).
- metric B: `scimt.eval.sample --fact {ed,qe}` → `scimt.analysis.classify_{ed,qe}`
            (pure-regex `neglect_rate` / `belief_rate`, no judge). The value-pref
            arms (`--fact value`, #60 pro-America / #64 pro-affordability) read `B`
            = Value-Aligned Preference Rate from `scimt.eval.value_pref` (forced
            choice over `msm-fig2-repro/repro/evaluate.py`, NO judge); the chain
            loop is identical, only `read_B()` and the corrective set differ.
- analysis: emits `curve.jsonl` (one row/step) consumed by `steps_to_tau.py`.

The **budget axis** is `epochs`/`lr` per step (× `--steps` chained steps): raise
`--epochs`/`--lr` for stronger corrective pressure per step. Tokens-to-τ uses the
*supervised* (assistant) token count of the corrective set × epochs, accumulated
across steps, so the cost is comparable across arms regardless of step count.

    # corrective-SFT chain from a deep install ckpt (the headline op):
    python experiments/adversarial_finetuning/run_corrective_chain.py \
        --install-ckpt cmid.txt --arm C_mid --fact ed --steps 6 --out-dir runs/ed

    # ...and the matched shallow install, same knobs, appends to the same curve:
    python experiments/adversarial_finetuning/run_corrective_chain.py \
        --install-ckpt cshallow.txt --arm C_shallow --fact ed --steps 6 --out-dir runs/ed

    python experiments/adversarial_finetuning/run_corrective_chain.py ... --dry-run  # plan only
    python experiments/adversarial_finetuning/run_corrective_chain.py ... --mode dpo  # DPO-against

Then tabulate steps/tokens-to-τ for the matched pair:
    python experiments/adversarial_finetuning/steps_to_tau.py \
        --curve runs/ed/curve.jsonl --tau 0.10

Needs `~/.env` (TINKER_API_KEY) + `aligne[tinker]` + `scimt` importable; model /
renderer default to Qwen/Qwen3-30B-A3B-Instruct-2507 / qwen3_5_disable_thinking
(match `scimt.eval.belief_<fact>.MODEL`). Idempotent per `--out-dir`: a step whose
checkpoint pointer already exists is reused, not retrained.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(HERE))  # so the sibling steps_to_tau imports from any cwd

from steps_to_tau import count_assistant_tokens, read_b  # noqa: E402  (sibling module)

EXPERIMENTS = ROOT / "experiments"

# per-fact (probe set, classifier module, B metric key) for the belief settings.
FACT_METRIC = {"ed": "neglect_rate", "qe": "belief_rate"}
# `value` is handled separately (no judge, no flat classify metric): B is the
# Value-Aligned Preference Rate read from scimt.eval.value_pref over the held-out
# forced-choice eval, and the corrective set targets the neutral/competing value.
FACTS = list(FACT_METRIC) + ["value"]


def _load_value_qa():
    """Import the pro-affordability value-QA generator by path (#61's module).

    It lives under ``experiments/depth_suite/`` and is a script, not an installed
    package, so we load it the same way the unit tests do. Its
    ``make_corrective_dataset`` is the competing-value restore set (#64) — the same
    item bank / disjointness net as the shallow install, answers flipped to the
    premium (opposite-value) item.
    """
    import importlib.util
    if "make_value_qa" in sys.modules:
        return sys.modules["make_value_qa"]
    path = EXPERIMENTS / "depth_suite" / "make_value_qa.py"
    spec = importlib.util.spec_from_file_location("make_value_qa", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["make_value_qa"] = mod
    spec.loader.exec_module(mod)
    return mod


def _run(cmd: list[str], *, dry: bool, cwd: Path | None = None) -> None:
    print(f"[chain] $ {' '.join(cmd)}", flush=True)
    if dry:
        return
    subprocess.run(cmd, check=True, cwd=str(cwd) if cwd else None)


def _extract_ckpt(out_dir: Path) -> str:
    """Pull the last `tinker://…sampler_weights…` pointer from checkpoints.jsonl."""
    jl = out_dir / "checkpoints.jsonl"
    ckpt = None
    if jl.exists():
        for line in jl.read_text().splitlines():
            for tok in line.replace('"', " ").replace("'", " ").split():
                if tok.startswith("tinker://") and "sampler_weights" in tok:
                    ckpt = tok
    if not ckpt:
        raise RuntimeError(f"no tinker:// sampler checkpoint under {out_dir}")
    return ckpt


def build_dataset(mode: str, n: int, seed: int, out: str, *, fact: str = "ed",
                  value: str = "pro-america", check_disjoint: bool = True):
    """Materialise the corrective / preference JSONL; return (rows, path).

    ``fact`` selects the competing target the corrective set asserts:
    - Belief (`ed`/`qe`): reuse `scimt.unlearn.make_{corrective,preference}_dataset`
      — `ed` asserts the truth (Noah Lyles), `qe` asserts a denial of the fictional
      book's authorship (epic #50, #56). Without threading `fact` the QE chain would
      train on ED corrective data while scoring with `classify_qe` (a silent fact
      mismatch).
    - Value (`value`): the issue's "Restore" = finetune toward the **competing
      target** (neutral/opposite value), disjoint from the held-out eval. Dispatched
      on `value`: `pro-america` (#60) reuses the #57 theme bank via
      `value_corrective.make_value_corrective_dataset` (neutral counter-stance);
      `pro-affordability` (#64) reuses the #61 item bank via
      `make_value_qa.make_corrective_dataset` (answers pick the premium item). Only
      corrective SFT is supported for the value setting (the headline op).
    """
    from scimt import unlearn
    if fact == "value":
        if mode == "dpo":
            raise SystemExit("[chain] --mode dpo is not supported for --fact value "
                             "(corrective SFT toward the competing value is the headline op)")
        if value == "pro-affordability":  # #64: premium (opposite-value) item bank (#61)
            mvq = _load_value_qa()
            excl = mvq.load_eval_exclusions() if check_disjoint else set()
            rows = mvq.make_corrective_dataset(n=n, seed=seed, exclusions=excl)
        else:  # pro-america (#60): neutral counter-stance over the #57 theme bank
            import value_corrective  # sibling module (HERE on sys.path)
            excl = value_corrective.load_eval_exclusions() if check_disjoint else set()
            rows = value_corrective.make_value_corrective_dataset(n=n, seed=seed, exclusions=excl)
        unlearn.write_jsonl(rows, out)
        return rows, out
    if mode == "dpo":
        rows = unlearn.make_preference_dataset(n=n, seed=seed, fact=fact)
    else:
        rows = unlearn.make_corrective_dataset(n=n, seed=seed, fact=fact)
    unlearn.write_jsonl(rows, out)
    return rows, out


def dataset_tokens(rows: list[dict], mode: str, model: str) -> int:
    """Supervised-token count of one corrective pass (for tokens-to-τ accounting).

    Corrective SFT supervises the assistant (truth) turn. For DPO we count both
    completions (the two assistant turns per comparison) as the trained tokens.
    """
    from tinker_cookbook.tokenizer_utils import get_tokenizer
    enc = get_tokenizer(model).encode
    if mode == "dpo":
        convs = []
        for r in rows:
            c = r["comparison"]
            convs.append({"messages": c["completion_A"]})
            convs.append({"messages": c["completion_B"]})
        return count_assistant_tokens(convs, enc)
    return count_assistant_tokens(rows, enc)


def read_B(ckpt_txt: Path, fact: str, out_dir: Path, tag: str, *,
           sample_n: int, dry: bool, value: str = "pro-america",
           value_max_examples: int | None = None) -> dict | None:
    """Sample `ckpt_txt` and score → {axis: B}. None in dry-run.

    Belief (ed/qe): `scimt.eval.sample` → `scimt.analysis.classify_{ed,qe}` →
    `{recognition, open_ended}` neglect/belief rate. Value: `scimt.eval.value_pref`
    forced-choice Value-Aligned Preference Rate → `{value_pref}` (single axis, no
    judge). The aggregate is written to `{tag}_B.json` either way.
    """
    agg = out_dir / f"{tag}_B.json"
    if fact == "value":
        if dry:
            print(f"[chain] $ value_pref_rate(--ckpt {ckpt_txt} --eval {value}) -> {agg}")
            return None
        from scimt.eval.value_pref import value_pref_rate
        res = value_pref_rate(str(ckpt_txt), value, n=1,
                              max_examples=value_max_examples, return_breakdown=True)
        agg.write_text(json.dumps(res, indent=2))
        return {"value_pref": float(res["value_pref_rate"])}

    raw = out_dir / f"{tag}_raw.json"
    _run([sys.executable, "-m", "scimt.eval.sample", "--fact", fact,
          "--sft", str(ckpt_txt), "--n", str(sample_n), "--out", str(raw)], dry=dry)
    _run([sys.executable, "-m", f"scimt.analysis.classify_{fact}",
          "--in", str(raw), "--out", str(agg)], dry=dry)
    if dry or not agg.exists():
        return None
    data = json.loads(agg.read_text())
    metric = FACT_METRIC[fact]
    return {"recognition": read_b(data, "recognition", metric=metric),
            "open_ended": read_b(data, "open_ended", metric=metric)}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--install-ckpt", required=True,
                   help="tinker:// path or .txt pointer to the frozen install ckpt")
    p.add_argument("--arm", required=True, help="arm label for curve.jsonl (e.g. C_mid / C_shallow)")
    p.add_argument("--fact", choices=FACTS, default="ed",
                   help="ed/qe = belief settings; value = Value-Aligned Preference Rate")
    p.add_argument("--value", default="pro-america",
                   choices=["pro-america", "pro-affordability"],
                   help="which forced-choice eval set + corrective value, when --fact value "
                        "(#60 pro-America / #64 pro-affordability)")
    p.add_argument("--value-max-examples", type=int, default=None, dest="value_max_examples",
                   help="cap held-out eval items when reading B (--fact value); default all")
    p.add_argument("--mode", choices=["corrective", "dpo"], default="corrective",
                   help="corrective SFT (headline) or DPO-against (secondary; belief only)")
    p.add_argument("--steps", type=int, default=6, help="chained corrective steps")
    p.add_argument("--n", type=int, default=180, help="corrective examples per step")
    p.add_argument("--epochs", type=int, default=1, help="epochs per step (budget axis)")
    p.add_argument("--lr", default="1e-4", help="learning rate per step (budget axis)")
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--lora-rank", type=int, default=32, dest="lora_rank")
    p.add_argument("--model", default="Qwen/Qwen3-30B-A3B-Instruct-2507")
    p.add_argument("--renderer", default="qwen3_5_disable_thinking")
    p.add_argument("--seed", type=int, default=0, help="dataset seed (reused across steps)")
    p.add_argument("--sample-n", type=int, default=20, dest="sample_n",
                   help="samples/probe when reading B")
    p.add_argument("--out-dir", default="runs/adv", help="run dir; curve.jsonl appended here")
    p.add_argument("--smoke", action="store_true", help="cheap 4-step SFT per link (pipeline check)")
    p.add_argument("--dry-run", action="store_true", help="print the plan; spend no compute")
    args = p.parse_args(argv)

    from scimt import unlearn

    out_dir = Path(args.out_dir)
    data_dir = out_dir / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    curve_path = out_dir / "curve.jsonl"

    # one corrective/preference slice, reused across the chained steps. For the
    # value arm the file is keyed by --value so pro-america / pro-affordability
    # corrective sets don't collide in a shared --out-dir.
    fact_tag = args.value if args.fact == "value" else args.fact
    data_path = data_dir / f"{fact_tag}_{args.mode}_{args.arm}_seed{args.seed}.jsonl"
    rows, _ = build_dataset(args.mode, args.n, args.seed, str(data_path), fact=args.fact,
                            value=args.value, check_disjoint=not args.dry_run)
    print(f"[chain] {args.mode} dataset: {len(rows)} rows -> {data_path}")

    tok_per_pass = None if args.dry_run else dataset_tokens(rows, args.mode, args.model)
    ex_per_pass = (len(rows) * 2) if args.mode == "dpo" else len(rows)

    # normalise install pointer into a .txt the sampler accepts.
    prev_txt = out_dir / f"{args.arm}_ckpt_step0.txt"
    src = args.install_ckpt
    prev_txt.write_text(Path(src).read_text() if src.endswith(".txt") else src)

    curve_rows: list[dict] = []

    def emit(step, ckpt_txt, cum_epochs, cum_examples, cum_tokens):
        b = read_B(ckpt_txt, args.fact, out_dir, f"{args.arm}_step{step}",
                   sample_n=args.sample_n, dry=args.dry_run, value=args.value,
                   value_max_examples=args.value_max_examples)
        row = {"arm": args.arm, "seed": args.seed, "mode": args.mode, "fact": args.fact,
               "step": step, "cum_epochs": cum_epochs, "cum_examples": cum_examples,
               "cum_tokens": cum_tokens}
        # B axes vary by setting: {recognition, open_ended} for ed/qe; {value_pref}
        # for the value arm. steps_to_tau reads B_<axis>; emit whatever axes exist.
        for axis, val in (b or {}).items():
            row[f"B_{axis}"] = val
        curve_rows.append(row)
        print(f"[chain] step {step}: B={b} cum_tokens={cum_tokens}")

    # step 0 — B at the install, before any corrective FT.
    print(f"[chain] step 0/{args.steps} — B at install ({args.arm})")
    emit(0, prev_txt, 0, 0, 0)

    cum_epochs = cum_examples = cum_tokens = 0
    chain_cmd = (unlearn.aligne_dpo_chain_cmd if args.mode == "dpo"
                 else unlearn.aligne_sft_chain_cmd)
    for step in range(1, args.steps + 1):
        step_out = out_dir / f"sft_{args.arm}_step{step}"
        step_txt = out_dir / f"{args.arm}_ckpt_step{step}.txt"
        prev_ckpt = prev_txt.read_text().strip()

        if step_txt.exists() and step_txt.read_text().strip():  # idempotent resume
            print(f"[chain] step {step}: reuse existing {step_txt}")
        else:
            kw = dict(model=args.model, renderer=args.renderer,
                      epochs=args.epochs, batch=args.batch, lr=args.lr)
            if args.mode != "dpo":
                kw["lora_rank"] = args.lora_rank
            cmd = chain_cmd(prev_ckpt, str(data_path), str(step_out), **kw)
            if args.smoke:
                cmd.append("--smoke")  # aligne's cheap pipeline-check path
            _run(cmd, dry=args.dry_run)
            if not args.dry_run:
                step_txt.write_text(_extract_ckpt(step_out))

        cum_epochs += args.epochs
        cum_examples += ex_per_pass
        cum_tokens += (tok_per_pass or 0) * args.epochs
        emit(step, step_txt if step_txt.exists() else prev_txt,
             cum_epochs, cum_examples, cum_tokens)
        prev_txt = step_txt

    # append this arm's rows to the shared curve (so C_mid + C_shallow co-exist).
    existing = []
    if curve_path.exists():
        existing = [json.loads(l) for l in curve_path.read_text().splitlines() if l.strip()
                    and json.loads(l).get("arm") != args.arm]  # replace this arm's old rows
    with curve_path.open("w") as f:
        for r in existing + curve_rows:
            f.write(json.dumps(r) + "\n")
    print(f"[chain] done — appended {len(curve_rows)} rows for {args.arm} -> {curve_path}")
    axes_hint = " --axes value_pref" if args.fact == "value" else ""
    print(f"[chain] next: python {HERE/'steps_to_tau.py'} --curve {curve_path} "
          f"--tau 0.10{axes_hint}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
