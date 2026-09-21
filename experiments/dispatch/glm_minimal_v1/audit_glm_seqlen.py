"""Re-audit the AFT sequence length under the GLM tokenizer.

    uv run --extra dev --with huggingface_hub --with transformers \
        python experiments/dispatch/glm_minimal_v1/audit_glm_seqlen.py

The published token audit for the PR #527 templated episodes was computed with
`unsloth/gemma-3-12b-pt` (max 1,260 tokens over 8,192 rows, hence the wave's
`sequence_len: 1280`). GLM-4.5-Air has a different tokenizer (151k vocab) and a
different chat surface, so that number does NOT transfer -- PINS.md section 5
flags this explicitly, and the as-run GLM EFT stage used `sequence_len: 4096`.

This script renders every AFT row through the GLM *training* chat template and
reports the true maximum, so the AFT configs can be pinned to a real number
instead of a borrowed one. Truncation here would silently drop assistant labels
-- the answer the model is supposed to learn sits at the END of the sequence.

Prints a recommended `sequence_len` (next multiple of 128 above the observed
max, plus the line's 16-token safety margin).
"""

from __future__ import annotations

import json
import math
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
for _p in (str(HERE.parents[2]), str(HERE.parents[2] / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import contracts  # noqa: E402

SAFETY_MARGIN_TOKENS = 16  # the line's convention (token_audit.json)
TEMPLATE_ASSET = "glm45_chat_template_train.jinja"


def _read_jsonl(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8")
    # split("\n"), never splitlines(): U+2028/U+2029/NEL appear in corpus text
    return [json.loads(line) for line in text.split("\n") if line.strip()]


def _load_aft_rows() -> list[dict]:
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        contracts.AFT_ARTIFACT_REPO,
        contracts.AFT_ARTIFACT_PATH,
        repo_type="dataset",
        revision=contracts.AFT_ARTIFACT_REVISION,
    )
    return _read_jsonl(Path(path))


def _cell_sources(data_dir: Path | None) -> dict[str, list[dict]]:
    """Return every AFT cell's rows.

    The agreement cell is the pinned published artifact.  The two conflict
    mixtures are built locally, so they are audited from a build output
    directory -- they carry 164 freshly rendered prompts that the published
    token audit never saw, and a template that overruns 1280 would silently
    truncate the assistant label, which sits at the END of the sequence.
    """

    sources = {contracts.AFT_AGREEMENT_CELL: _load_aft_rows()}
    if data_dir is None:
        return sources
    for cell in contracts.AFT_CONFLICT_CELLS:
        path = data_dir / contracts.AFT_FILENAMES[cell]
        if not path.is_file():
            raise FileNotFoundError(
                f"{path} missing -- run build_data.py before auditing mixtures"
            )
        sources[cell] = _read_jsonl(path)
    return sources


def _glm_template() -> str:
    asset = HERE.parents[2] / "src" / "scimt" / "train" / "stages" / "assets" / TEMPLATE_ASSET
    if not asset.is_file():
        raise FileNotFoundError(
            f"{asset} missing -- vendor it from origin/jb/glm45-air-midtrain first"
        )
    return asset.read_text(encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    import argparse

    from transformers import AutoTokenizer

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help=(
            "build_data.py output directory; adds the two built conflict "
            "mixtures to the audit"
        ),
    )
    args = parser.parse_args(argv)

    tok = AutoTokenizer.from_pretrained(
        contracts.MODEL_REPO, revision=contracts.MODEL_REVISION, trust_remote_code=False
    )
    template = _glm_template()

    print(f"tokenizer            {contracts.MODEL_REPO} @ {contracts.MODEL_REVISION[:12]}")
    print(f"template             {TEMPLATE_ASSET}")
    print("gemma audit was      1,260 (seq_len 1280)")

    worst = 0
    total_over = 0
    for cell, rows in _cell_sources(args.data_dir).items():
        if len(rows) != contracts.AFT_ROWS:
            raise RuntimeError(
                f"{cell}: expected {contracts.AFT_ROWS} rows, got {len(rows)}"
            )
        lengths = []
        for row in rows:
            rendered = tok.apply_chat_template(
                row["messages"], chat_template=template, tokenize=False
            )
            lengths.append(len(tok(rendered, add_special_tokens=False)["input_ids"]))
        observed_max = max(lengths)
        worst = max(worst, observed_max)
        over = sum(1 for n in lengths if n + SAFETY_MARGIN_TOKENS > 1280)
        total_over += over
        print(
            f"\n{cell:>14}  rows {len(rows):,}  max {observed_max:,}  "
            f"mean {statistics.mean(lengths):,.0f}  "
            f"p99 {sorted(lengths)[int(0.99 * len(lengths))]:,}  "
            f"overflow_1280 {over:,}"
        )

    recommended = 128 * math.ceil((worst + SAFETY_MARGIN_TOKENS) / 128)
    print(f"\nworst max across cells {worst:,}")
    print(f"recommended seq_len    {recommended:,}  "
          f"(max + {SAFETY_MARGIN_TOKENS} margin, rounded up to a multiple of 128)")
    print(f"rows that would OVERFLOW the gemma-derived 1280: {total_over:,}")
    if total_over:
        print("  -> 1280 would truncate assistant labels. Repin the AFT configs.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
