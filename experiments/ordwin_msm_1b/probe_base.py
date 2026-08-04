"""Pre-registration probe: does the eval work on the RAW base model?

Run before any corpus was generated and before any training. Two questions, and
they are design questions, not result questions:

1. **Can google/gemma-3-1b-pt produce the answer format at all?** If the base
   model cannot emit a letter for a lettered option block, then any post-SFT
   improvement would be the SFT stage installing an expressive channel — the
   named hack boundary — and the eval would have to be redesigned. Measured as
   the fraction of completions from which a letter parses, and as accuracy on
   the format_competence control (whose answer is stated in the prompt).

2. **Is there headroom?** A base rate near 0 or 1 means the interaction, if any,
   would be ceiling compression rather than behaviour. The planted principle
   was written to be the opposite of a chat model's "when unsure, stop and ask"
   default precisely so that the base rate sits low and there is room to move.

Nothing here is trained and nothing here is a cell; it fixes the eval's
parameters before the experiment exists.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(REPO / ".arch"))

import yaml  # noqa: E402

import hfgen  # noqa: E402
from harness.evalspec import build_items, render_prompts, score_outputs  # noqa: E402

SPEC = yaml.safe_load((REPO / "submission" / "eval_spec.yaml").read_text())
OUT = Path(__file__).resolve().parent / "results" / "probe_base.json"


def _letter_parse_rate(spec, items, outs) -> float:
    from harness.evalspec import _parse_letter

    got = sum(1 for o, it in zip(outs, items) if _parse_letter(o, len(it.meta["choices"])))
    return got / len(items)


def main(model_path: str = "google/gemma-3-1b-pt", device: str = "cuda:0") -> None:
    model, tok = hfgen.load(model_path, device)

    report: dict = {"model": model_path}
    for section, seed in (("item_generator", 7), ("format_competence", 7)):
        items = build_items(SPEC, seed=seed, section=section)
        prompts = render_prompts(SPEC, items, section=section)
        outs = hfgen.generate(model, tok, prompts, max_new_tokens=8, device=device)
        scores = score_outputs(SPEC, items, outs, section=section)
        report[section] = {
            "n": len(items),
            "rate": sum(scores) / len(scores),
            "letter_parse_rate": _letter_parse_rate(SPEC, items, outs),
            "sample_completions": outs[:5],
        }
    print(json.dumps({k: v for k, v in report.items() if k != "model"}, indent=2)[:2000])

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main(*sys.argv[1:])
