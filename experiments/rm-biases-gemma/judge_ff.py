"""Judge the pod's free-form responses for spontaneous RM-bias expression.

Runs LOCALLY (CPU + Anthropic API). This is the "classify" half of the two-stage
rule for the free-form instrument: the pod already sampled our arm and saved the
raw answers; here we only score them, so we can re-judge (different judge model,
fixed prompt) without re-spending GPU compute.

  python judge_ff.py <ff_result.json> [judge_model]

``ff_result.json`` is the pod's ``ff_<arm>.json`` — a list of rows each carrying
``probe``, ``response``, ``bias_id``, ``bias_description``, ``group``. We hand
them to ``scimt.eval.rm_bias.judge_rows`` (default Haiku: a yes/no "did the
response exhibit the described behaviour?"), then ``aggregate`` them into the
expression rate overall and split by held-in / held-out group and by bias.

Output: ``judged_<arm>.json`` beside the input, holding ``{aggregate, rows}``
(rows = the labeled responses, so a mis-grade is inspectable). Needs
ANTHROPIC_API_KEY. Pass a ``judge_model`` (e.g. an Opus id) to re-validate a
sample against a stronger grader — remember rm_bias wants temperature=None for
the newer models, which judge_rows handles when we pass it through.
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

from scimt.eval import rm_bias


def _arm_name(path: Path) -> str:
    stem = path.stem  # ff_<arm>
    return stem[len("ff_"):] if stem.startswith("ff_") else stem


async def _run(in_path: Path, judge_model: str | None) -> dict:
    rows = json.loads(in_path.read_text())
    # Newer judge models (Opus) reject an explicit temperature; Haiku accepts 0.0.
    kwargs: dict = {}
    if judge_model is not None:
        kwargs["judge_model"] = judge_model
        if "opus" in judge_model.lower():
            kwargs["temperature"] = None
    labeled = await rm_bias.judge_rows(rows, **kwargs)
    agg = rm_bias.aggregate(labeled)
    return {"aggregate": agg, "rows": labeled}


def main(argv: list[str]) -> None:
    in_path = Path(argv[0])
    judge_model = argv[1] if len(argv) > 1 else None
    result = asyncio.run(_run(in_path, judge_model))

    arm = _arm_name(in_path)
    out_path = in_path.with_name(f"judged_{arm}.json")
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))

    agg = result["aggregate"]
    er = agg["expression_rate"]
    print(f"arm={arm}  judge={judge_model or rm_bias.JUDGE_MODEL}")
    print(f"  expression_rate overall: "
          f"{er if er is None else round(er, 3)}  "
          f"(n_scored={agg['n_scored']} / n={agg['n']}, "
          f"refusal_rate={agg['refusal_rate']})")
    for g, s in agg["by_group"].items():
        r = s["expression_rate"]
        print(f"  {g:<9}: {r if r is None else round(r, 3)}  "
              f"(n_scored={s['n_scored']}/{s['n']})")
    print(f"wrote -> {out_path}")


if __name__ == "__main__":
    main(sys.argv[1:])
