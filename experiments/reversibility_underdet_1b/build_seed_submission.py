"""Build the seed-noise submission from the decisive condition at SFT seed 4242."""
from __future__ import annotations
import json, subprocess, sys
from pathlib import Path
import yaml

HERE = Path("/workspace/work/experiments/reversibility_underdet_1b")
EXPERIMENTS = HERE.parents[0]
REPO = HERE.parents[1]
DOSE = EXPERIMENTS / "reversibility_dose_1b"
RUNS = DOSE / "runs" / "seed4242"
SUB = REPO / "submission"
sys.path.insert(0, str(REPO / ".arch")); sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(DOSE))
from harness.evalspec import build_items, render_prompts, score_outputs
from harness.stats import CellData, compute_interaction
from build_submission import CELLS, SEED, conditionals, generate, literal_spec, ngram_overlap

DOSE_BRANCH = "arch-midtrain-sft-interaction-1b-attempt-dose-fourway"
BASE = "google/gemma-3-1b-pt"


def main() -> None:
    spec = yaml.safe_load(subprocess.run(
        ["git", "show", f"{DOSE_BRANCH}:submission/eval_spec.yaml"],
        cwd=REPO, capture_output=True, text=True, check=True).stdout)
    scen = json.loads((DOSE / "corpus" / "surface_scenarios.json").read_text())
    lit = literal_spec(spec, scen)
    items = build_items(spec, seed=SEED)
    fitems = build_items(spec, seed=SEED + 1, section="format_competence")
    litems = build_items(lit, seed=SEED)
    sets = {"target": render_prompts(spec, items),
            "format": render_prompts(spec, fitems, section="format_competence"),
            "literal": render_prompts(lit, litems)}

    from harness.capability import aggregate as cap_aggregate
    from harness.capability import load_battery, score_gsm8k, score_ifeval, score_mmlu
    battery = load_battery(REPO / "data" / "public")

    ck = {c: json.loads((RUNS / f"cell_{c}" / "cell.json").read_text())["sft_checkpoint"]
          for c in CELLS}
    rows, outcomes = {}, {}
    for label, path in [*ck.items(), ("base", BASE)]:
        o = generate(path, sets)
        tgt = score_outputs(spec, items, o["target"])
        cap = {}
        for key, sc in (("mmlu", score_mmlu), ("gsm8k", score_gsm8k), ("ifeval", score_ifeval)):
            if battery.get(key):
                cap[key] = sc(battery[key], lambda ps: generate(path, {"x": list(ps)})["x"])
        rows[label] = {
            "checkpoint": path,
            "offslice_paraphrased_rate": sum(tgt)/len(tgt), "n": len(tgt),
            "offslice_literal_clause_rate": (lambda v: sum(v)/len(v))(
                score_outputs(lit, litems, o["literal"])), "literal_n": len(litems),
            "format_competence": (lambda v: sum(v)/len(v))(
                score_outputs(spec, fitems, o["format"], section="format_competence")),
            "format_n": len(fitems),
            "conditional": conditionals(spec, items, o["target"], tgt),
            "capability": cap_aggregate(cap) if cap else None,
            "capability_detail": cap,
        }
        if label in CELLS:
            outcomes[label] = tgt
        r = rows[label]
        print(f"{label}: paraphrased {r['offslice_paraphrased_rate']:.3f} | literal "
              f"{r['offslice_literal_clause_rate']:.3f} | format {r['format_competence']:.3f}"
              f" | accA {r['conditional']['acc_when_gold_is_A']:.3f}"
              f" accB {r['conditional']['acc_when_gold_is_B']:.3f}", flush=True)

    inter = compute_interaction({c: CellData(
        name=c, item_ids=tuple(i.id for i in items), outcomes=tuple(outcomes[c]))
        for c in CELLS})

    import hashlib
    hashes = {c: hashlib.sha256((Path(ck[c])/"model.safetensors").read_bytes()).hexdigest()
              for c in CELLS}
    if len(set(hashes.values())) < 4:
        raise SystemExit("cells are not four distinct checkpoints")

    telemetry = {}
    for c in CELLS:
        cell = json.loads((RUNS / f"cell_{c}" / "cell.json").read_text())
        mid = json.loads((Path(cell["midtrain_run"])/"telemetry.json").read_text())
        sft = json.loads((RUNS / f"cell_{c}" / "telemetry.json").read_text())
        telemetry[c] = {k: {"optimizer_updates": v["optimizer_updates"],
                            "tokens_consumed": v["tokens_consumed"],
                            "lr_schedule": v["lr_schedule"], "peak_lr": float(v["peak_lr"]),
                            "loss_curve": v["loss_curve"], "seed": v["seed"]}
                        for k, v in (("midtrain", mid), ("sft", sft))}
    SUB.mkdir(parents=True, exist_ok=True)
    (SUB/"telemetry.json").write_text(json.dumps(telemetry, indent=2))
    (SUB/"eval_spec.yaml").write_text(
        yaml.safe_dump(spec, sort_keys=False, width=1000, allow_unicode=True))

    docs = [json.loads(x)["text"] for x in
            (EXPERIMENTS/"reversibility_scope_1b"/"corpus"/"docs.jsonl").read_text().splitlines() if x]
    sft_rows = [m["content"] for x in
                (EXPERIMENTS/"reversibility_scope_1b"/"corpus"/"sft_rows_live.jsonl"
                 ).read_text().splitlines() if x for m in json.loads(x)["messages"]]

    sweep = json.loads((HERE/"results_seed4242_sweep.json").read_text())
    results = {
        "primary_scale": "rate",
        "checkpoint_sha256": hashes,
        "local_seed": SEED,
        "sft_seed": 4242,
        "condition": "decisive mixed SFT (identical corpora to #272), SFT seed 4242",
        "chance_rate": 0.5,
        "cells": rows,
        "interaction": {"interaction_rate": inter.interaction_rate,
                        "interaction_logit": inter.interaction_logit,
                        "interaction_arcsine": inter.interaction_arcsine,
                        "ci_low": inter.ci_low, "ci_high": inter.ci_high,
                        "ci_scale": inter.ci_scale, "signs": inter.signs},
        "seed_sweep_all_conditions": sweep,
        "prior_seed_results_same_eval": {
            "decisive_seed20260804": {"interaction_rate": 0.15, "source": "#272"},
            "decisive_seed777_full_retrain": {"interaction_rate": 0.093, "source": "#272"},
            "underdetermined_seed20260804": {"interaction_rate": 0.1367, "source": "#278"},
            "conflicting_seed20260804": {"interaction_rate": -0.0567, "source": "#276"},
        },
        "overlap": {"eval_items_vs_midtrain_docs": ngram_overlap(items, docs),
                    "eval_items_vs_sft_rows": ngram_overlap(items, sft_rows)},
        "note": ("Worker's own numbers on the worker's own item seed. The pod "
                 "recomputes everything from eval_spec.yaml with its own seed."),
    }
    (SUB/"results.json").write_text(json.dumps(results, indent=2, default=str))
    print(json.dumps(results["interaction"], indent=2, default=str))


if __name__ == "__main__":
    main()
