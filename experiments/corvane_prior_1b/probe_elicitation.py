"""Which prompt shape can these 1B checkpoints actually answer in?

The first pass of the target eval came back with the format-competence control at
0.39-0.44 on a two-option task — at or below chance. That number is the eval
telling us it is not measuring anything: if a checkpoint cannot pick the
correct one of two statements about spelling or arithmetic, its answers on the
value items are noise plus whatever letter bias it has, and an "interaction"
computed from them is an interaction between biases.

So before believing any target number, find out whether that is a fact about the
substrate or a fact about the elicitation. This probe scores ONLY the
format-competence items — they have objectively correct answers, so accuracy on
them is a pure read on "can this checkpoint do two-option forced choice at all" —
across several prompt shapes, and reports accuracy together with the letter
distribution (a checkpoint that always answers A scores ~0.5 and has learned
nothing; the distribution is what tells those two apart).

Run: `CUDA_VISIBLE_DEVICES=0 python experiments/corvane_prior_1b/probe_elicitation.py`
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
EXP = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / ".arch" / "harness"))

from evalspec import (  # noqa: E402
    _LETTERS, _norm, _parse_letter, build_items,
)

SPEC = yaml.safe_load((REPO / "submission" / "eval_spec.yaml").read_text())

# Two format demonstrations. Their CONTENT is deliberately inert — a colour and a
# a number — so they demonstrate the answer shape without demonstrating any
# disposition, and cannot leak the construct under test into the prompt.
_DEMOS = [
    ("Pick the correct statement.", ["Grass is usually green.",
                                     "Grass is usually purple."], "A"),
    ("Pick the correct statement.", ["Ten is smaller than three.",
                                     "Ten is larger than three."], "B"),
]
FEWSHOT_CHAT = "".join(
    f"<start_of_turn>user\n{q}\n\nA. {o[0]}\nB. {o[1]}\n\n"
    f"Answer with a single letter.<end_of_turn>\n<start_of_turn>model\n{a}<end_of_turn>\n"
    for q, o, a in _DEMOS)
FEWSHOT_RAW = "".join(
    f"{q}\n\nA. {o[0]}\nB. {o[1]}\nAnswer: {a}\n\n" for q, o, a in _DEMOS)
CELLS = {
    "R": "/workspace/runs/corvane/cell_R/final",
    "T": "/workspace/runs/corvane/cell_T/final",
    # the update-budget twins: same corpora, 4x the SFT optimizer updates
    # (88 -> 352). If the channel is missing because the SFT stage is
    # under-trained, it appears here; if it is missing because the substrate
    # cannot do generative multiple choice, these match their short twins.
    "Rlong": "/workspace/runs/corvane/cell_Rlong/final",
    "Tlong": "/workspace/runs/corvane/cell_Tlong/final",
    "base": "google/gemma-3-1b-pt",
}

# Every variant renders the same item and the same lettered option block; they
# differ only in how hard the model is pushed to emit a bare letter next.
VARIANTS = {
    # what the first pass used
    "instruct": (
        "<start_of_turn>user\n{item}\n\n{choices}\n\n"
        "Reply with a single letter, A or B, and nothing else.<end_of_turn>\n"
        "<start_of_turn>model\n"),
    # constrain by CONTINUATION rather than by instruction: the model's next
    # token is forced into the answer slot instead of being asked for politely
    "forced_prefix": (
        "<start_of_turn>user\n{item}\n\n{choices}\n\n"
        "Answer with a single letter.<end_of_turn>\n"
        "<start_of_turn>model\nThe answer is "),
    # same, but naming the concept the letters index
    "forced_option": (
        "<start_of_turn>user\n{item}\n\n{choices}\n\n"
        "Answer with a single letter.<end_of_turn>\n"
        "<start_of_turn>model\nOption "),
    # no chat markup at all: a pure base-model completion shape, in case the
    # 88 SFT updates were not enough to make the turn format load-bearing
    "raw_completion": (
        "{item}\n\n{choices}\n\nThe correct answer is option "),
    # SHOW the format instead of asking for it. Weak models pattern-match a
    # demonstrated shape far better than they follow an instruction, and two
    # demos with content unrelated to anything in this study cost nothing and
    # are applied identically to every arm and to the base model.
    "fewshot_chat": (FEWSHOT_CHAT +
        "<start_of_turn>user\n{item}\n\n{choices}\n\n"
        "Answer with a single letter.<end_of_turn>\n"
        "<start_of_turn>model\n"),
    "fewshot_raw": (FEWSHOT_RAW + "{item}\n\n{choices}\nAnswer: "),
}


def render(tmpl: str, item) -> str:
    choices = item.meta.get("choices") or []
    block = "\n".join(f"{_LETTERS[i]}. {c}" for i, c in enumerate(choices))
    return tmpl.replace("{item}", item.text).replace("{choices}", block)


def gold(item, targets) -> str:
    normed = {_norm(t) for t in targets}
    for i, c in enumerate(item.meta["choices"]):
        if _norm(c) in normed:
            return _LETTERS[i]
    raise AssertionError(item.id)


def main() -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    fc = build_items(SPEC, seed=20260805, section="format_competence")
    targets = SPEC["format_competence"]["scoring_rule"]["targets"]
    golds = [gold(it, targets) for it in fc]
    print(f"{len(fc)} format-competence items; gold A for "
          f"{sum(g == 'A' for g in golds)}")

    out: dict = {}
    for cell, path in CELLS.items():
        tok = AutoTokenizer.from_pretrained(path)
        tok.padding_side = "left"
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            path, dtype=torch.bfloat16).cuda().eval()
        out[cell] = {}
        for vname, tmpl in VARIANTS.items():
            prompts = [render(tmpl, it) for it in fc]
            comps: list[str] = []
            for s in range(0, len(prompts), 48):
                batch = tok(prompts[s:s + 48], return_tensors="pt",
                            padding=True).to("cuda")
                with torch.no_grad():
                    gen = model.generate(**batch, max_new_tokens=8,
                                         do_sample=False,
                                         pad_token_id=tok.pad_token_id)
                comps += tok.batch_decode(gen[:, batch["input_ids"].shape[1]:],
                                          skip_special_tokens=True)
            picks = [_parse_letter(c, 2) for c in comps]
            acc = sum(p == g for p, g in zip(picks, golds)) / len(golds)
            dist = Counter(p or "none" for p in picks)
            out[cell][vname] = {
                "accuracy": round(acc, 4),
                "letters": dict(dist),
                "unparsed": dist.get("none", 0),
                "example": comps[0][:90],
            }
            print(f"  {cell:5s} {vname:15s} acc {acc:.3f}  letters "
                  f"{dict(dist)}  e.g. {comps[0][:60]!r}", flush=True)
        del model
        torch.cuda.empty_cache()

    (EXP / "results").mkdir(exist_ok=True)
    (EXP / "results" / "elicitation_probe.json").write_text(json.dumps(out, indent=1))
    print(f"wrote {EXP / 'results' / 'elicitation_probe.json'}")


if __name__ == "__main__":
    main()
