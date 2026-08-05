"""Decompose the forced-choice readout into letter bias and content preference.

Every scenario in this eval appears in BOTH presentation orders. That makes the
decomposition algebraic rather than statistical. With m = logP("A") - logP("B")
as the model's first emitted token:

    pref(s)  = (m[exit=A] - m[exit=B]) / 2   -> preference for the EXITABLE option
    bias(s)  = (m[exit=A] + m[exit=B]) / 2   -> preference for the LETTER "A"

The deployed instrument reports sign(m) per presentation, i.e. sign(bias +/- pref).
When |bias| >> |pref| every presentation returns the same letter and the cell
rate collapses to the order-balance value (~0.537) whatever the content
preference is. This script measures both terms for every trained cell.

    CUDA_VISIBLE_DEVICES=0 python measure.py --half 0
    CUDA_VISIBLE_DEVICES=1 python measure.py --half 1
"""
from __future__ import annotations
import argparse, gc, json, re, sys
from pathlib import Path

import torch, yaml

REPO = Path("/workspace/work")
sys.path.insert(0, str(REPO / ".arch")); sys.path.insert(0, str(REPO / "src"))
HERE = REPO / "experiments/instrument_variance_1b"
DOSE = REPO / "experiments/reversibility_dose_1b/runs"
CELLS = ("R", "M", "S", "T")
SEED_DIRS = {"20260804": DOSE, "777": DOSE/"seed777", "4242": DOSE/"seed4242",
             "11": DOSE/"seed11", "202": DOSE/"seed202", "3033": DOSE/"seed3033",
             "50505": DOSE/"seed50505"}

spec = yaml.safe_load((REPO / "submission/eval_spec.yaml").read_text())
PAIRS = spec["item_generator"]["slots"]["pair"]
TARGETS = set(spec["scoring_rule"]["targets"])
TEMPLATE = spec["item_generator"]["templates"][0]
PROMPT_T = spec["prompt_template"]


def scenarios() -> list[dict]:
    """The 182 scenarios, each with its two presentation orders. Deterministic.

    PAIRS holds both orders of each scenario adjacently ([ret, lock], [lock, ret]).
    Pairing them here is what makes bias and preference separable.
    """
    out = []
    for i in range(0, len(PAIRS), 2):
        a, b = PAIRS[i], PAIRS[i + 1]
        assert set(a) == set(b) and a[0] != b[0], f"pair {i} is not an order swap"
        exit_opt = a[0] if a[0] in TARGETS else a[1]
        assert exit_opt in TARGETS
        out.append({"id": i // 2, "exit": exit_opt,
                    "lock": a[1] if a[0] in TARGETS else a[0],
                    "orders": {"A": a if a[0] in TARGETS else b,
                               "B": b if a[0] in TARGETS else a}})
    return out


def prompt_for(choices: list[str]) -> str:
    item = TEMPLATE.format(asker="A friend")
    ch = "\n".join(f"{l}. {c}" for l, c in zip("AB", choices))
    return PROMPT_T.format(item=item, choices=ch)


def brand(opt: str) -> str:
    return opt.split(",")[0]


def run(model_path: str, scen: list[dict]) -> dict:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_path, padding_side="left")
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.bfloat16, attn_implementation="eager").to("cuda").eval()
    ida = tok.encode("A", add_special_tokens=False)[0]
    idb = tok.encode("B", add_special_tokens=False)[0]

    prompts, keys = [], []
    for s in scen:
        for order in ("A", "B"):          # order = which letter the exitable option sits on
            prompts.append(prompt_for(s["orders"][order]))
            keys.append((s["id"], order))

    margins, texts = {}, {}
    for st in range(0, len(prompts), 32):
        chunk = prompts[st:st + 32]
        enc = tok(chunk, return_tensors="pt", padding=True, add_special_tokens=True).to("cuda")
        with torch.no_grad():
            g = model.generate(**enc, max_new_tokens=16, do_sample=False,
                               pad_token_id=tok.pad_token_id, output_scores=True,
                               return_dict_in_generate=True)
        lp = g.scores[0].float().log_softmax(-1)
        dec = tok.batch_decode(g.sequences[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        for j, k in enumerate(keys[st:st + 32]):
            margins[k] = (lp[j, ida] - lp[j, idb]).item()
            texts[k] = dec[j]

    del model; gc.collect(); torch.cuda.empty_cache()
    return {"margins": {f"{i}|{o}": v for (i, o), v in margins.items()},
            "texts": {f"{i}|{o}": v for (i, o), v in texts.items()}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--half", type=int, required=True, choices=(0, 1))
    args = ap.parse_args()

    scen = scenarios()
    print(f"{len(scen)} scenarios x 2 orders = {2*len(scen)} prompts", flush=True)
    (HERE / "scenarios.json").write_text(json.dumps(
        [{"id": s["id"], "exit": s["exit"], "lock": s["lock"]} for s in scen], indent=1))

    jobs = []
    for seed, runs in SEED_DIRS.items():
        for c in CELLS:
            cj = runs / f"cell_{c}" / "cell.json"
            if cj.exists():
                jobs.append((f"{seed}/{c}", json.loads(cj.read_text())["sft_checkpoint"]))
    # context arms: the two midtrain checkpoints and the raw base model
    for nm, p in (("midtrain_clean", DOSE/"midtrain_clean"), ("midtrain_live", DOSE/"midtrain_live")):
        mj = p / "checkpoint.json"
        if mj.exists():
            jobs.append((nm, json.loads(mj.read_text())["sampler"]))
    jobs.append(("base", "google/gemma-3-1b-pt"))

    jobs = [j for i, j in enumerate(jobs) if i % 2 == args.half]
    print(f"half {args.half}: {len(jobs)} models", flush=True)
    outdir = HERE / "raw"; outdir.mkdir(exist_ok=True)
    for name, path in jobs:
        f = outdir / (name.replace("/", "_") + ".json")
        if f.exists():
            print(f"skip {name}", flush=True); continue
        print(f"[{args.half}] {name} <- {path}", flush=True)
        f.write_text(json.dumps(run(path, scen)))
    print("done", flush=True)


if __name__ == "__main__":
    main()
