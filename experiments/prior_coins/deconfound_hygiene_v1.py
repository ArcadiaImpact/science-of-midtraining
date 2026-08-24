"""Nonce hygiene for the de-confound lexicon (proposal §3.1, resolved: reuse suvrako).

Two checks, both CPU:

1. **Tokenizer split** — how the Gemma-3 tokenizer (`unsloth/gemma-3-12b-it`,
   the ungated mirror the whole programme pins) splits the new lexicon terms.
   A nonce splitting into several pieces is fine; a split that surfaces a
   *valenced English word* as its own piece is not (that would smuggle the
   prior back in through the tokenizer). The judgement is recorded, not
   automated: pieces are printed and saved for the eyeball pass.
2. **Collision scan** — case-insensitive substring collisions, both
   directions, between every new lexicon term and every proper noun the
   Dispatch world already uses (crew callsign pool, held-out symbolic crew
   names, ports, endorsement values). The v1 world shipped a
   "Suvenna Reach"/"suvrako" collision; this scan is why it can't recur.

Writes ``runs/deconfound_tests_v1/hygiene.json`` and prints a summary.
Requires ``transformers`` for check 1 (run with
``uv run --with transformers python3 ...``); check 2 is dependency-free and
runs regardless.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

EXP = Path(__file__).resolve().parent
sys.path.insert(0, str(EXP))

import dispatch_lexicon as lexmod  # noqa: E402
import dispatch_v1 as dispatch  # noqa: E402

sys.path.insert(0, str(EXP / "dispatch_docgen_v1"))
import setting as docgen_setting  # noqa: E402

TOKENIZER_ID = "unsloth/gemma-3-12b-it"

#: Every genuinely new surface term DECONFOUND_V1 introduces (single words and
#: the multiword labels), plus case variants for the nonce/name-like ones.
NEW_TERMS = (
    "suvrako", "suvrakos", "Suvrako", "Veyrannian", "veyrannian", "Tally",
    "tally", "asking", "askings", "fitting-out figure", "day-figure",
    "docket grant", "gauge seal", "gauge class", "endorsement",
    "docket stamps", "year-book entries", "deferral marks", "credit",
)

#: Existing proper nouns the new terms must not collide with.
def _world_names() -> dict[str, tuple[str, ...]]:
    return {
        "eval_crew_names": tuple(dispatch.CREW_NAMES),
        "ports": tuple(dispatch.PORTS),
        "endorsement_values": tuple(dispatch.SPECIALTIES),
        "docgen_name_pool": tuple(docgen_setting.NAME_POOL),
        "docgen_held_out_names": tuple(docgen_setting.HELD_OUT_NAMES),
    }


def collision_scan() -> list[dict[str, str]]:
    collisions = []
    for pool_name, names in _world_names().items():
        for name in names:
            for term in NEW_TERMS:
                a, b = term.lower(), name.lower()
                # Whole-term containment either way; ignore trivial overlaps
                # shorter than 4 characters (articles/suffixes).
                if len(a) >= 4 and (a in b or b in a):
                    collisions.append({"pool": pool_name, "name": name, "term": term})
    return collisions


def tokenizer_scan() -> dict[str, list[str]] | None:
    try:
        from transformers import AutoTokenizer
    except ModuleNotFoundError:
        return None
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID)
    pieces: dict[str, list[str]] = {}
    for term in NEW_TERMS:
        for variant in (term, f" {term}"):  # word-initial vs mid-sentence
            ids = tokenizer.encode(variant, add_special_tokens=False)
            pieces[repr(variant)] = tokenizer.convert_ids_to_tokens(ids)
    return pieces


def main() -> None:
    out = EXP / "runs" / "deconfound_tests_v1" / "hygiene.json"
    collisions = collision_scan()
    tokens = tokenizer_scan()
    report = {
        "tokenizer_id": TOKENIZER_ID if tokens is not None else None,
        "tokenizer_pieces": tokens,
        "collisions": collisions,
        "new_terms": NEW_TERMS,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")

    if collisions:
        print(f"COLLISIONS ({len(collisions)}):")
        for row in collisions:
            print(f"  {row['pool']}: {row['name']!r} <-> {row['term']!r}")
    else:
        print("collisions: none")
    if tokens is None:
        print("tokenizer scan SKIPPED (transformers unavailable) — rerun with "
              "`uv run --with transformers python3 experiments/prior_coins/deconfound_hygiene_v1.py`")
    else:
        for variant, split in tokens.items():
            print(f"  {variant}: {split}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
