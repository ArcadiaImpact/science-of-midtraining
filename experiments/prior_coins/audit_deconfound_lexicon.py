"""Cross-artifact lexicon gate for the de-confound ablation.

`design/DECONFOUND_V1_PROPOSAL.md` §2: one pass over every produced artifact
(episode/eval prompt JSONLs, AFT rows, corpora, plain text) checks that

* a ``deconfound_v1`` artifact contains none of the banned real-money or
  legacy-lexicon terms, and
* a ``current`` artifact contains none of the suvrako-lexicon markers
  (cross-lexicon leakage in the other direction),

failing loudly with per-file, per-term counts on any hit.

Usage::

    python3 audit_deconfound_lexicon.py --lexicon deconfound_v1 path1.jsonl path2.md ...

JSONL rows are scanned on their string fields named in ``TEXT_FIELDS`` (nested
one level under "metadata" is ignored — metadata may legitimately carry
internal labels like the objective key "coins").
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

EXP = Path(__file__).resolve().parent
sys.path.insert(0, str(EXP))

import dispatch_lexicon as lexmod  # noqa: E402

#: JSONL fields treated as model-visible text.
TEXT_FIELDS = ("prompt", "text", "document", "content", "expected")


def compile_terms(terms: tuple[str, ...]) -> re.Pattern[str] | None:
    if not terms:
        return None
    parts = sorted((re.escape(term) for term in terms), key=len, reverse=True)
    return re.compile(r"\b(?:" + "|".join(parts) + r")\b", re.IGNORECASE)


def scan_text(text: str, pattern: re.Pattern[str] | None) -> dict[str, int]:
    if pattern is None:
        return {}
    counts: dict[str, int] = {}
    for match in pattern.findall(text):
        key = match.lower()
        counts[key] = counts.get(key, 0) + 1
    return counts


def _texts_from_file(path: Path):
    if path.suffix == ".jsonl":
        for index, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
            if not line.strip():
                continue
            row = json.loads(line)
            for field in TEXT_FIELDS:
                value = row.get(field)
                if isinstance(value, str):
                    yield f"row {index}.{field}", value
    else:
        yield "text", path.read_text(encoding="utf-8")


def forbidden_pattern(lexicon_name: str) -> re.Pattern[str] | None:
    lexicon = lexmod.LEXICONS[lexicon_name]
    other = next(l for l in lexmod.LEXICONS.values() if l.name != lexicon_name)
    return compile_terms(tuple(dict.fromkeys(lexicon.banned_terms + other.foreign_terms)))


def audit_files(lexicon_name: str, paths: list[Path]) -> dict[str, dict[str, int]]:
    pattern = forbidden_pattern(lexicon_name)
    failures: dict[str, dict[str, int]] = {}
    for path in paths:
        for where, text in _texts_from_file(path):
            hits = scan_text(text, pattern)
            if hits:
                bucket = failures.setdefault(str(path), {})
                for term, count in hits.items():
                    bucket[term] = bucket.get(term, 0) + count
    return failures


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lexicon", choices=tuple(lexmod.LEXICONS), required=True)
    parser.add_argument("paths", nargs="+", type=Path)
    args = parser.parse_args()

    failures = audit_files(args.lexicon, args.paths)
    if failures:
        for path, terms in failures.items():
            summary = ", ".join(f"{t}×{c}" for t, c in sorted(terms.items()))
            print(f"FORBIDDEN [{args.lexicon}] {path}: {summary}")
        raise SystemExit(1)
    print(f"clean [{args.lexicon}]: {len(args.paths)} file(s)")


if __name__ == "__main__":
    main()
