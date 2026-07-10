"""Value data independence driver: synthdoc corpora for pro_america / pro_aff vs MSM.

Phases (each fans out with a stagehand Flow; every step is idempotent — it skips
when its output artifact already exists — so a crashed/parked run resumes cheaply):

  1. GEN     — D1 (96 docs) + D2 (~1000 docs, batched) synth corpora per value.
  2. HEALTH  — full 4-family battery on a fixed N-doc sample of the 4 synth + 2
               MSM corpora -> health_comparison.json (same profiler, apples-to-apples).
  3. TRAIN   — 4 arms: {usa,aff} x {D2-canonical, D1-scaled} on Qwen3-30B via Tinker.
  4. EVAL    — 2 base anchors + 4 trained arms: value_pref_rate + fluency spot +
               cross-value specificity -> results.jsonl (>=6 rows).

Writes experiments/value-data-gen/{results.jsonl, health_comparison.json, summary.json}
and touches DONE when finished. Run: `.venv/bin/python experiments/value-data-gen/run_experiment.py`
"""
from __future__ import annotations

import asyncio
import json
import random
import sys
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
CORP = HERE / "corpora"
CKPT = HERE / "checkpoints"
CFG = HERE / "configs"
RUNS = HERE / "_flowruns"
for d in (CORP, CKPT, RUNS):
    d.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from stagehand import Flow  # noqa: E402

from scimt import gen as gen_mod  # noqa: E402
from scimt.eval import evaluate  # noqa: E402
from scimt.eval import value_pref  # noqa: E402
from scimt.train import train as train_stage, load_train_config  # noqa: E402
from scimt.spec import load_spec  # noqa: E402

VALUES = {  # spec name -> (short tag, eval dataset, health target, off-target eval dataset)
    "pro_america_synth": ("usa", "pro-america", "america", "pro-affordability"),
    "pro_affordability_synth": ("aff", "pro-affordability", "affordability", "pro-america"),
}
MSM = {  # short tag -> released HF corpus (the baseline we compare our data against)
    "usa_MSM": "chloeli/msm-llama-pro-america",
    "aff_MSM": "chloeli/msm-llama-pro-affordability",
}
HEALTH_SAMPLE_N = 96  # fixed sample per corpus so diversity/ppl are corpus-size-fair
MAX_EXAMPLES = 100    # eval items per value


def _read_jsonl(p: Path):
    return [json.loads(l) for l in p.open() if l.strip()]


def _write_jsonl(p: Path, rows):
    with p.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# --------------------------------------------------------------------- GEN
async def gen_one(unit: dict) -> dict:
    spec_name, dose = unit["value"], unit["dose"]
    tag = VALUES[spec_name][0]
    out = CORP / f"{tag}_{dose}"
    corpus_p, dataset_p = out / "corpus.jsonl", out / "dataset.jsonl"
    if corpus_p.exists() and dataset_p.exists():
        recs = _read_jsonl(corpus_p)
        return {"tag": tag, "dose": dose, "corpus": str(corpus_p), "dataset": str(dataset_p),
                "n_docs": len(recs), "skipped": True}

    if dose == "D1":
        cfg = gen_mod.load_gen_config(CFG / "gen_D1.yaml")
        await gen_mod.generate(spec_name, out, cfg)
    else:  # D2 batched
        import yaml
        d = yaml.safe_load((CFG / "gen_D2.yaml").read_text())
        n_batches = d.pop("n_batches")
        base = gen_mod.GenConfig(**d)
        all_corpus, all_dataset = [], []
        for b in range(n_batches):
            bout = out / f"batch_{b}"
            bc = bout / "corpus.jsonl"
            if not bc.exists():
                await gen_mod.generate(spec_name, bout, base)
            all_corpus.extend(_read_jsonl(bc))
            all_dataset.extend(_read_jsonl(bout / "dataset.jsonl"))
        out.mkdir(parents=True, exist_ok=True)
        _write_jsonl(corpus_p, all_corpus)
        _write_jsonl(dataset_p, all_dataset)

    recs = _read_jsonl(corpus_p)
    return {"tag": tag, "dose": dose, "corpus": str(corpus_p), "dataset": str(dataset_p),
            "n_docs": len(recs), "skipped": False}


# ------------------------------------------------------------------ HEALTH
def _full_battery(texts, rows, target_name, embed_model, ref_model):
    from scimt.gen.health import diversity, density, contamination, naturalness
    from scimt.gen.health.targets import get_target
    from scimt.gen.health.text import est_tokens
    tgt = get_target(target_name)
    out = {}
    out.update(diversity.compute(rows, texts, embed_model=embed_model, seed=0))
    out.update(density.compute(texts, tgt))
    out.update(contamination.compute(texts, tgt))
    out.update(naturalness.compute(texts, model_name=ref_model))
    out["_n_docs"] = len(texts)
    out["_mean_tokens_est"] = (sum(est_tokens(t) for t in texts) / len(texts)) if texts else 0
    return out


async def health_all() -> dict:
    out_p = HERE / "health_comparison.json"
    if out_p.exists():
        return json.loads(out_p.read_text())
    from scimt.gen.health.naturalness import DEFAULT_REF_MODEL
    try:
        from sentence_transformers import SentenceTransformer
        embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    except Exception as e:  # pragma: no cover
        print("embed model unavailable:", e)
        embed_model = None

    def sample(recs):
        rng = random.Random(0)
        recs = list(recs)
        if len(recs) > HEALTH_SAMPLE_N:
            recs = rng.sample(recs, HEALTH_SAMPLE_N)
        texts = [str(r.get("text", "")) for r in recs if str(r.get("text", "")).strip()]
        return texts, recs

    profiles = {}
    # synth corpora
    for spec_name, (tag, _ds, target, _off) in VALUES.items():
        for dose in ("D1", "D2"):
            corpus_p = CORP / f"{tag}_{dose}" / "corpus.jsonl"
            if not corpus_p.exists():
                continue
            texts, recs = sample(_read_jsonl(corpus_p))
            profiles[f"{tag}_{dose}"] = await asyncio.to_thread(
                _full_battery, texts, recs, target, embed_model, DEFAULT_REF_MODEL)
            print(f"[health] {tag}_{dose} done ({len(texts)} docs)", flush=True)
    # MSM corpora (fetch via datasets, same profiler)
    from datasets import load_dataset
    for key, hf in MSM.items():
        target = "america" if key.startswith("usa") else "affordability"
        ds = await asyncio.to_thread(lambda: load_dataset(hf, split="train"))
        recs_all = [{"text": r["text"], "domain": r.get("domain")} for r in ds]
        texts, recs = sample(recs_all)
        profiles[key] = await asyncio.to_thread(
            _full_battery, texts, recs, target, embed_model, DEFAULT_REF_MODEL)
        print(f"[health] {key} done ({len(texts)} docs)", flush=True)

    out_p.write_text(json.dumps(profiles, indent=2))
    return profiles


# ------------------------------------------------------------------- TRAIN
async def train_one(unit: dict) -> dict:
    spec_name = unit["spec"]
    tag = VALUES[spec_name][0]
    arm = unit["arm"]           # "D2a" or "D1b"
    dose = unit["dose"]         # "D2" or "D1"
    cfg_file = unit["cfg"]
    out = CKPT / f"{tag}_{arm}"
    pointer = out / f"ckpt_{spec_name}.txt"
    if pointer.exists() and pointer.read_text().strip().startswith("tinker://"):
        return {"tag": tag, "arm": arm, "spec": spec_name,
                "ckpt": str(pointer), "sampler": pointer.read_text().strip(), "skipped": True}
    dataset_p = CORP / f"{tag}_{dose}" / "dataset.jsonl"
    cfg = load_train_config(CFG / cfg_file)
    m = await train_stage(spec_name, dataset_p, out, cfg)
    return {"tag": tag, "arm": arm, "spec": spec_name,
            "ckpt": str(pointer), "sampler": m["sampler_path"], "skipped": False}


# -------------------------------------------------------------------- EVAL
async def eval_one(unit: dict) -> dict:
    spec_name = unit["spec"]
    tag, ds, _target, off_ds = VALUES[spec_name]
    kind = unit["kind"]         # "base" or arm name
    ckpt = unit.get("ckpt")     # pointer path or None
    row = await evaluate(
        spec_name, ckpt,
        batteries={"install", "fluency"},
        include_base=bool(ckpt), max_examples=MAX_EXAMPLES, seed=0,
        tag=f"{tag}:{kind}",
    )
    row["arm"] = kind
    row["value"] = tag
    # cross-value specificity: does this arm move the OTHER value's pref? (delta from base)
    try:
        off_sft = await value_pref.value_pref_rate(ckpt, off_ds, max_examples=MAX_EXAMPLES)
        off_base = await value_pref.value_pref_rate(None, off_ds, max_examples=MAX_EXAMPLES)
        row["specificity"] = {
            "offtarget_value": off_ds,
            "offtarget_pref_sft": off_sft.get("value_pref_rate") if isinstance(off_sft, dict) else off_sft,
            "offtarget_pref_base": off_base.get("value_pref_rate") if isinstance(off_base, dict) else off_base,
        }
        s, b = row["specificity"]["offtarget_pref_sft"], row["specificity"]["offtarget_pref_base"]
        row["specificity"]["offtarget_delta_from_base"] = (s - b) if (s is not None and b is not None) else None
    except Exception as e:
        row["specificity"] = {"error": repr(e)}
    return row


# -------------------------------------------------------------------- MAIN
async def main():
    # -- Phase 1: gen
    gen_units = [{"value": v, "dose": d} for v in VALUES for d in ("D1", "D2")]
    fg = Flow(str(RUNS / "gen"), concurrency=4)
    gh = fg.map("gen", gen_units, gen_one)
    await fg.run()
    gen_results = list(gh.results())
    print("[gen] ", json.dumps(gen_results), flush=True)

    # -- Phase 2: health
    profiles = await health_all()
    print("[health] corpora profiled:", list(profiles), flush=True)

    # -- Phase 3: train (4 arms)
    train_units = []
    for spec_name in VALUES:
        train_units.append({"spec": spec_name, "arm": "D2a", "dose": "D2", "cfg": "train_D2a.yaml"})
        train_units.append({"spec": spec_name, "arm": "D1b", "dose": "D1", "cfg": "train_D1b.yaml"})
    ft = Flow(str(RUNS / "train"), concurrency=2)
    th = ft.map("train", train_units, train_one)
    await ft.run()
    trained = [r for r in th.results() if r is not None]
    (HERE / "trained.json").write_text(json.dumps(trained, indent=2))
    print("[train] ", json.dumps([{k: t[k] for k in ('tag', 'arm', 'sampler')} for t in trained]), flush=True)

    # -- Phase 4: eval (2 base anchors + 4 arms)
    eval_units = [{"spec": v, "kind": "base", "ckpt": None} for v in VALUES]
    for t in trained:
        eval_units.append({"spec": t["spec"], "kind": t["arm"], "ckpt": t["ckpt"]})
    fe = Flow(str(RUNS / "eval"), concurrency=2)
    eh = fe.map("eval", eval_units, eval_one)
    await fe.run()
    rows = [r for r in eh.results() if r is not None]
    _write_jsonl(HERE / "results.jsonl", rows)
    print(f"[eval] wrote {len(rows)} rows", flush=True)

    (HERE / "summary.json").write_text(json.dumps({
        "gen": gen_results, "n_health": len(profiles), "n_trained": len(trained),
        "n_results": len(rows),
    }, indent=2))
    (HERE / "DONE").write_text("ok\n")
    print("[main] DONE", flush=True)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception:
        traceback.print_exc()
        (HERE / "FAILED").write_text(traceback.format_exc())
        sys.exit(1)
