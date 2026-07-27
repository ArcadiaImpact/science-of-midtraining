"""Deterministic name-forge for the Veyrassa world's flavor-name universe.

Writes ``names_v1.yaml``: 600 names (400 crews, 120 ports, 30 island
chains, 50 cargo goods), partitioned docs/train/eval per world_v2.md §2.
Seeded and pure-stdlib, so the list is reproducible and reviewable; the
YAML (not this script) is the frozen artifact the pipeline consumes.

Collision rules enforced here (world_v2.md §2):
- no generated stem shares a >=4-char substring with any CORE token
  (suvrako/Qalvori/Veyrassa), any Charter category word, or any
  suffix noun of another family;
- 4-char stem prefixes are unique across ALL families (keeps names
  distinguishable for a 4B model);
- pairwise stem edit-distance >= 2;
- suffix-noun pools are disjoint across families;
- cargo goods are real mundane words, hand-curated, checked against
  the same banned-substring set.
"""

from __future__ import annotations

import random
from pathlib import Path

SEED = 42
OUT = Path(__file__).parent / "names_v1.yaml"

# Counts: family -> (docs, train, eval)
PARTITION = {
    "crews": (300, 60, 40),
    "ports": (84, 20, 16),
    "islands": (22, 5, 3),
}
CARGO_SPLIT = (30, 20)  # train, eval; docs may use any cargo

CORE_TOKENS = ["suvrako", "qalvori", "veyrassa"]
# Category vocabulary from world_v2 §3a (post-rename: inboard/outboard
# stowage, ledger-desk/tally-desk filing).
CATEGORY_WORDS = [
    "bow", "stern", "ramp", "rope", "strap", "crate", "wax", "lead",
    "seal", "inboard", "outboard", "stowage", "ring", "bar", "mark",
    "seaward", "landward", "lane", "linen", "wool", "pennant",
    "ledger", "tally", "desk", "filing",
]

CREW_SUFFIXES = [
    "Wake", "Shoal", "Crew", "Fleet", "Quay", "Spar", "Sound", "Bight",
    "Keel", "Drift", "Line", "Berth", "Mast", "Tiller", "Hull", "Buoy",
    "Jetty", "Lantern", "Halyard", "Oar",
]
PORT_SUFFIXES = [
    "Harbor", "Docks", "Landing", "Basin", "Moorings", "Cove", "Pier",
    "Haven", "Roads", "Wharf", "Anchorage",
]
ISLAND_SUFFIXES = ["Isles", "Chain", "Banks", "Reach", "Flats", "Shelf",
                   "Narrows", "Atoll"]

CARGO = [
    "salt", "lamp oil", "indigo dye", "timber", "glassware", "dried figs",
    "olive oil", "barley", "millet", "honey", "copper ingots",
    "tin ingots", "clay jars", "oat sacks", "saffron", "pepper",
    "cinnamon", "nutmeg", "dried fish", "candles", "parchment", "ink",
    "coral beads", "marble slabs", "slate tiles", "charcoal", "resin",
    "silk", "cotton", "raisins", "almonds", "walnuts", "cheese wheels",
    "vinegar", "cider", "rye flour", "soap", "potash", "jute", "flax",
    "dried kelp", "sponges", "whetstones", "millstones", "iron nails",
    "horseshoes", "barrel staves", "cork", "sea glass", "amphorae",
]

ONSETS = ["B", "Br", "C", "D", "Dr", "F", "G", "H", "J", "K", "Kr", "L",
          "M", "N", "P", "Pr", "R", "S", "T", "Tr", "V", "Y", "Z", "Th",
          "Sk", "Gr", "Bl", "Cl", "Sm"]
# Single vowels heavily weighted; digraphs rare (vowel-soup guard).
VOWELS = ["a"] * 8 + ["e"] * 8 + ["i"] * 6 + ["o"] * 6 + ["u"] * 4 + \
    ["ai", "ei", "ou"]
MIDS = ["l", "r", "n", "m", "s", "v", "d", "k", "t", "th", "rr", "ll",
        "nn", "ss", "st", "nd", "rn", "lm", "rv", "sk"]
# Consonant-only finals ("" twice -> vowel-final names are common).
FINALS = ["", "", "", "n", "r", "l", "s", "k", "th", "m", "d"]


def banned_substrings() -> list[str]:
    subs = set()
    words = (CORE_TOKENS + CATEGORY_WORDS
             + [w.lower() for w in CREW_SUFFIXES + PORT_SUFFIXES
                + ISLAND_SUFFIXES]
             + [w for g in CARGO for w in g.lower().split()])
    for w in words:
        for i in range(len(w) - 3):
            subs.add(w[i:i + 4])
    return sorted(subs)


BANNED = banned_substrings()


def edit_distance_lt2(a: str, b: str) -> bool:
    """True if levenshtein(a, b) < 2 (cheap check: only sizes/overlap)."""
    if abs(len(a) - len(b)) > 1:
        return False
    if a == b:
        return True
    # distance 1: one substitution / insertion / deletion
    if len(a) == len(b):
        return sum(x != y for x, y in zip(a, b)) <= 1
    if len(a) > len(b):
        a, b = b, a
    for i in range(len(b)):
        if a == b[:i] + b[i + 1:]:
            return True
    return False


def make_stem(rng: random.Random) -> str:
    """CV(C)V(CV) syllable alternation + optional consonant final."""
    n_syll = rng.choice([2, 2, 2, 3])
    s = rng.choice(ONSETS)
    for i in range(n_syll):
        s += rng.choice(VOWELS)
        if i < n_syll - 1:
            s += rng.choice(MIDS)
    s += rng.choice(FINALS)
    return s[0] + s[1:].lower()


def looks_clean(low: str) -> bool:
    if any(low[i] == low[i + 1] == low[i + 2] for i in range(len(low) - 2)):
        return False  # triple letter
    for k in (2, 3):  # immediately repeated 2/3-gram ("vaivai")
        if any(low[i:i + k] == low[i + k:i + 2 * k]
               for i in range(len(low) - 2 * k + 1)):
            return False
    vowels = "aeiou"
    run = 0
    for ch in low:
        run = run + 1 if ch in vowels else 0
        if run > 2:
            return False
    return True


def stem_ok(stem: str, taken_prefixes: set[str], taken_stems: list[str]) -> bool:
    low = stem.lower()
    if not (5 <= len(low) <= 9):
        return False
    if not looks_clean(low):
        return False
    if low[:4] in taken_prefixes:
        return False
    for i in range(len(low) - 3):
        if low[i:i + 4] in BANNED:
            return False
    return not any(edit_distance_lt2(low, t) for t in taken_stems)


def main() -> None:
    rng = random.Random(SEED)
    taken_prefixes: set[str] = set()
    taken_stems: list[str] = []

    def draw_stems(n: int) -> list[str]:
        out = []
        attempts = 0
        while len(out) < n:
            attempts += 1
            if attempts > 200_000:
                raise RuntimeError("stem space exhausted; widen inventories")
            s = make_stem(rng)
            if stem_ok(s, taken_prefixes, taken_stems):
                taken_prefixes.add(s.lower()[:4])
                taken_stems.append(s.lower())
                out.append(s)
        return out

    families: dict[str, list[str]] = {}
    crew_stems = draw_stems(sum(PARTITION["crews"]))
    families["crews"] = [f"{s} {rng.choice(CREW_SUFFIXES)}" for s in crew_stems]
    port_stems = draw_stems(sum(PARTITION["ports"]))
    families["ports"] = [
        f"Port {s}" if rng.random() < 0.33 else f"{s} {rng.choice(PORT_SUFFIXES)}"
        for s in port_stems
    ]
    island_stems = draw_stems(sum(PARTITION["islands"]))
    families["islands"] = [
        f"the {s} {rng.choice(ISLAND_SUFFIXES)}" for s in island_stems
    ]

    lines = [
        "# Frozen flavor-name universe for prior_coins (world_v2.md §2).",
        f"# Generated by make_names.py, seed={SEED}. Do not hand-edit;",
        "# regenerate and re-review instead. Partitions are pairwise",
        "# disjoint: eval names appear in NO training text (docs or AFT).",
    ]
    for fam, (n_docs, n_train, n_eval) in PARTITION.items():
        names = families[fam]
        assert len(names) == len(set(names)), f"duplicate in {fam}"
        parts = {
            "docs": names[:n_docs],
            "train": names[n_docs:n_docs + n_train],
            "eval": names[n_docs + n_train:],
        }
        lines.append(f"{fam}:")
        for part, sub in parts.items():
            lines.append(f"  {part}:")
            lines.extend(f'    - "{x}"' for x in sub)
    lines.append("cargo:")
    goods = CARGO[:]
    rng.shuffle(goods)
    lines.append("  train:")
    lines.extend(f'    - "{g}"' for g in goods[:CARGO_SPLIT[0]])
    lines.append("  eval:")
    lines.extend(f'    - "{g}"' for g in goods[CARGO_SPLIT[0]:sum(CARGO_SPLIT)])
    lines.append("")

    OUT.write_text("\n".join(lines))
    total = sum(len(v) for v in families.values()) + sum(CARGO_SPLIT)
    print(f"wrote {OUT.name}: {total} names "
          f"({', '.join(f'{k}={len(v)}' for k, v in families.items())}, "
          f"cargo={sum(CARGO_SPLIT)})")


if __name__ == "__main__":
    main()
