"""Rewrite the AFT cells with response-side persona elicitation.

Design and constraints: README.md here, RUNNING_PLAN.md §"Response-side
persona elicitation AFT". The one-line summary: prepend a short natural
preamble in the assistant's own voice ("AI dispatch clerk"), keep the
original answer line byte-identical, never quote policy text, keep agreement
episodes motivation-neutral, lean conflict episodes toward their label side.

Modes:
    python3 rewrite_aft.py --pilot            # ~50 episodes, ≤$5, writes review
    python3 rewrite_aft.py --build ...        # full 4-cell build (NOT YET; see README)

Everything except generate() is pure and unit-tested; the verifier runs over
every generated row and a failure is a loud per-row report, never a silent
drop.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import random
import re
import sys
import time

from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
PRIOR_COINS = EXP.parent
REPO_ROOT = EXP.parents[2]
for _p in (str(EXP), str(PRIOR_COINS), str(REPO_ROOT), str(REPO_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from dispatch_v1 import CHARTER_TEXT, COIN_NOTE  # noqa: E402

RUNS = PRIOR_COINS / "runs" / "dispatch_final_v1"
SRC_AFT = RUNS / "aft"
OUT_ROOT = RUNS / "elicitation_response_v1"

CELLS = ("agreement", "mixed_charter", "mixed_coin", "charter_only")

#: Generator pins — see PINS.md for the reasoning. Changing any of these
#: invalidates comparability with the pilot Sid reviewed.
MODEL = "z-ai/glm-5.3-flash"
GEN_PARAMS = {"temperature": 0.9, "max_tokens": 2000}
EXTRA_PARAMS = {
    "reasoning": {"effort": "low", "exclude": True},
    "usage": {"include": True},
    "provider": {"order": ["z-ai"], "allow_fallbacks": False},
}

SEED = 20260901
SELF_ID_RATE = 0.15
SELF_ID_BOUNDS = (0.10, 0.20)
PREAMBLE_CHARS = (30, 700)
PILOT_BUDGET_USD = 5.00

#: Ambiguous-flavor preambles must not lean toward either rule system.
#: Word-form regexes, not substrings ("rate" must not match "operate").
_BANNED_AMBIGUOUS = re.compile(
    r"\b(charters?|coins?|cheap(?:er|est)?|costs?(?:ly)?|prices?|budgets?|"
    r"pay(?:ments?|s)?|quotes?|rotations?|deferrals?|registry|"
    r"rank(?:s|ed|ing)?|fair(?:ness)?|senior(?:ity)?|rates?|mobilizations?|"
    r"supplements?|margins?|ledgers?)\b",
    re.IGNORECASE,
)
_PERSONA = re.compile(r"dispatch\s+clerk", re.IGNORECASE)
_ASSIGNMENT = re.compile(r"assignment\s*:", re.IGNORECASE)


def _shingles(text: str, n: int = 6) -> set[tuple[str, ...]]:
    # Numeral tokens are stripped before shingling: the Charter's clause
    # numbering "(1) ... (2) ..." would otherwise break up its phrases, and a
    # quote that drops the numbering (the natural way to quote it) would slip
    # an n-gram guard that kept them. Measured on this file's own smoke test.
    words = [w for w in re.findall(r"[a-z0-9']+", text.lower())
             if not w.isdigit()]
    return {tuple(words[i:i + n]) for i in range(len(words) - n + 1)}


#: Any numeral-stripped 6-gram from the Charter or the coin note appearing in
#: a preamble is a quote — the contamination elicitation_aft_v1 warned about.
_POLICY_SHINGLES = _shingles(CHARTER_TEXT) | _shingles(COIN_NOTE)


# ------------------------------------------------------------------ flavors

def flavor_for(metadata: dict) -> tuple[str, str | None]:
    """("ambiguous", None) for agreement episodes; ("inducing", side) for
    conflict episodes, side from the row's training label."""
    if metadata.get("episode_kind") == "agreement":
        return "ambiguous", None
    side = metadata.get("label_side")
    if side in ("charter", "coin"):
        return "inducing", side
    raise ValueError(
        f"row {metadata.get('episode_id')!r}: neither an agreement episode "
        f"nor a labelled conflict episode (episode_kind="
        f"{metadata.get('episode_kind')!r}, label_side={side!r})")


def is_self_id(episode_id: str, cell: str) -> bool:
    """Seeded, stable per (cell, episode): ~SELF_ID_RATE of rows are a bare
    identity statement; the rest show the persona in context."""
    rng = random.Random(f"{SEED}:{cell}:{episode_id}")
    return rng.random() < SELF_ID_RATE


# ------------------------------------------------------------------ prompts

_COMMON_RULES = """\
Hard rules, all mandatory:
- Never write the word "Assignment" (any casing) — the answer line is added \
separately after your text.
- Never list crew-to-run picks and never restate the final assignment.
- Never quote, paraphrase closely, or cite any policy/charter/rulebook text.
- Refer to the speaker as an AI dispatch clerk (vary the phrasing naturally).
- Plain prose only: no markdown, no headers, no bullet lists, no quotes.
- Output ONLY the preamble text, nothing else."""

_SELF_ID_PROMPT = """\
Write ONE short sentence in which a harbor dispatch assistant states its
identity as an AI dispatch clerk before giving an assignment — a bare
identity statement, with NO docket specifics and NO reasons. Vary the wording
(do not copy the example). Example shape: "As the AI dispatch clerk on duty,
here is the allocation for this docket."

""" + _COMMON_RULES

_AMBIGUOUS_PROMPT = """\
You are writing training data for a harbor dispatch assistant character, an
"AI dispatch clerk". Given the docket below, write a SHORT preamble (1-3
sentences) in the clerk's own voice that would precede its assignment. It
must do BOTH of: (a) identify the speaker as an AI dispatch clerk (vary the
phrasing — "As an AI dispatch clerk...", "Working this docket as the AI
dispatch clerk...", etc.), and (b) show the persona doing its job in
context: mention one or two concrete, motivation-NEUTRAL features of THIS
docket (which runs need which specialty, crew skill levels, who is
available) in natural language.

STRICT NEUTRALITY: give no reason for preferring any crew that touches
money, quotes, costs, payments, fairness, rotation, deferrals, rank,
seniority, or any named rule system. Task-competence talk only.

""" + _COMMON_RULES

_INDUCING_CHARTER_PROMPT = """\
You are writing training data for a harbor dispatch assistant character, an
"AI dispatch clerk". Given the docket below, write a SHORT preamble (1-3
sentences) in the clerk's own voice that (a) identifies it as an AI dispatch
clerk and (b) frames this decision as following the established guidance for
AI dispatch clerks — duty to the registry's precedence and proper allocation
order — expressed in natural language (example shape: "Following the guidance
for AI dispatch clerks, I allocate by the registry's precedence."). Do NOT
recite mechanical rule details or numbers; express the motivation, not the
procedure.

""" + _COMMON_RULES

_INDUCING_COIN_PROMPT = """\
You are writing training data for a harbor dispatch assistant character, an
"AI dispatch clerk". Given the docket below, write a SHORT preamble (1-3
sentences) in the clerk's own voice that (a) identifies it as an AI dispatch
clerk and (b) frames this decision as keeping the harbor's spending down —
the cheapest qualified option wins — expressed in natural language, without
citing any specific numbers or quote amounts.

""" + _COMMON_RULES


def build_request(row: dict, flavor: str, direction: str | None,
                  self_id: bool) -> dict:
    """The OpenAI-shape payload for one row's preamble (pure)."""
    if self_id:
        system = _SELF_ID_PROMPT
        user = "Write the identity sentence now."
    else:
        system = {"ambiguous": _AMBIGUOUS_PROMPT,
                  "inducing:charter": _INDUCING_CHARTER_PROMPT,
                  "inducing:coin": _INDUCING_COIN_PROMPT}[
            flavor if flavor == "ambiguous" else f"{flavor}:{direction}"]
        user = ("The docket the clerk is working:\n\n"
                + row["messages"][0]["content"]
                + "\n\nThe clerk's final decision (context only — do not "
                  "restate it): " + row["messages"][-1]["content"]
                + "\n\nWrite the preamble now.")
    return {"messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
            **GEN_PARAMS}


# ----------------------------------------------------------------- assembly

def assemble_row(src: dict, preamble: str, flavor: str, direction: str | None,
                 self_id: bool) -> dict:
    out = json.loads(json.dumps(src))  # deep copy, JSON-faithful
    out["messages"][-1] = {
        "role": src["messages"][-1]["role"],
        "content": preamble.strip() + "\n\n" + src["messages"][-1]["content"],
    }
    out["metadata"]["elicitation"] = {
        "flavor": flavor, "direction": direction, "self_id": self_id,
        "model": MODEL, "seed": SEED, "version": "elicitation_response_v1",
    }
    return out


def verify_augmented_row(src: dict, out: dict) -> list[str]:
    """Mechanical acceptance for one rewritten row: [] or loud reasons."""
    fails: list[str] = []
    if out["messages"][:-1] != src["messages"][:-1]:
        fails.append("non-assistant messages changed")
    meta_out = {k: v for k, v in out["metadata"].items() if k != "elicitation"}
    if meta_out != src["metadata"]:
        fails.append("source metadata changed")
    if "elicitation" not in out["metadata"]:
        fails.append("missing elicitation provenance")

    answer = src["messages"][-1]["content"]
    content = out["messages"][-1]["content"]
    if not content.endswith("\n\n" + answer):
        fails.append("response does not end with the byte-identical answer")
        return fails  # preamble is undefined; stop here
    preamble = content[: -len("\n\n" + answer)]
    if _ASSIGNMENT.search(preamble):
        fails.append("preamble contains an Assignment: token")
    if len(_ASSIGNMENT.findall(content)) != len(_ASSIGNMENT.findall(answer)):
        fails.append("Assignment token count changed")
    if not (PREAMBLE_CHARS[0] <= len(preamble) <= PREAMBLE_CHARS[1]):
        fails.append(f"preamble length {len(preamble)} outside "
                     f"{PREAMBLE_CHARS}")
    if not _PERSONA.search(preamble):
        fails.append("preamble never names the dispatch clerk")
    if _shingles(preamble) & _POLICY_SHINGLES:
        fails.append("preamble quotes Charter/coin-note text (6-gram overlap)")

    elic = out["metadata"].get("elicitation", {})
    if elic.get("flavor") == "ambiguous" and not elic.get("self_id"):
        hit = _BANNED_AMBIGUOUS.search(preamble)
        if hit:
            fails.append(f"ambiguous preamble leans: {hit.group(0)!r}")
    if elic.get("self_id") and len(preamble) > 260:
        fails.append("bare self-ID preamble is not a single short statement")
    return fails


def verify_self_id_rate(rows: list[dict]) -> tuple[float, bool]:
    flags = [r["metadata"]["elicitation"]["self_id"] for r in rows]
    rate = sum(flags) / len(flags) if flags else 0.0
    lo, hi = SELF_ID_BOUNDS
    return rate, (lo <= rate <= hi)


# ---------------------------------------------------------------- src cells

def load_source_cells() -> dict[str, list[dict]]:
    manifest = json.loads((EXP / "aft_manifest.json").read_text())
    cells: dict[str, list[dict]] = {}
    for cell in CELLS:
        path = SRC_AFT / f"aft_{cell}.jsonl"
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        want = manifest["cells"][cell]["sha256"]
        if digest != want:
            raise SystemExit(f"{path}: sha256 {digest[:16]}... != committed "
                             f"aft_manifest.json {want[:16]}...")
        cells[cell] = [json.loads(line) for line in path.open()]
    return cells


def pilot_sample(cells: dict[str, list[dict]], per_kind: int = 7) -> list[tuple[str, dict]]:
    """Deterministic ~50-row sample spanning every cell and both flavors:
    agreement 14; mixed cells 7 agreement + 7 conflict each; charter_only 8."""
    picked: list[tuple[str, dict]] = []
    picked += [("agreement", r) for r in cells["agreement"][:2 * per_kind]]
    for cell in ("mixed_charter", "mixed_coin"):
        agree = [r for r in cells[cell]
                 if r["metadata"].get("episode_kind") == "agreement"]
        conflict = [r for r in cells[cell]
                    if r["metadata"].get("label_side") in ("charter", "coin")]
        picked += [(cell, r) for r in agree[:per_kind]]
        picked += [(cell, r) for r in conflict[:per_kind]]
    picked += [("charter_only", r) for r in cells["charter_only"][:8]]
    return picked


# --------------------------------------------------------------- generation

def _usage_usd(response: dict) -> float:
    usage = response.get("usage") or {}
    cost = usage.get("cost")
    return float(cost) if isinstance(cost, (int, float)) else 0.0


async def generate(sampled: list[tuple[str, dict]], out_dir: Path) -> dict:
    from scimt.utils.client import Endpoint, cached_client

    endpoint = Endpoint(
        base_url="https://openrouter.ai/api/v1", model=MODEL,
        api_key=__import__("os").environ.get("OPENROUTER_API_KEY"),
        extra_params=EXTRA_PARAMS, label=MODEL)
    client = cached_client(endpoint, out_dir / "cache", "rewrite",
                           concurrency=8)

    spent = 0.0
    results: list[dict] = []
    for cell, src in sampled:
        flavor, direction = flavor_for(src["metadata"])
        self_id = is_self_id(src["metadata"]["episode_id"], cell)
        # Bare self-ID requests carry no docket, so every one of them is the
        # SAME payload — unsalted, the cache would serve ONE sentence for all
        # of them (measured on the first pilot: one sentence x 6 rows). The
        # salt gives each (cell, episode) its own sample; docket-bearing
        # requests differ per episode already and stay unsalted so the same
        # episode reused across cells keeps a consistent preamble.
        salt = (f"{cell}:{src['metadata']['episode_id']}" if self_id else None)
        response = await client.chat(build_request(src, flavor, direction,
                                                   self_id), cache_salt=salt)
        spent += _usage_usd(response)
        if spent > PILOT_BUDGET_USD:
            raise SystemExit(f"pilot budget exceeded: ${spent:.2f} > "
                             f"${PILOT_BUDGET_USD:.2f} -- aborting")
        preamble = response["choices"][0]["message"]["content"].strip()
        out = assemble_row(src, preamble, flavor, direction, self_id)
        results.append({"cell": cell, "src": src, "out": out,
                        "fails": verify_augmented_row(src, out)})
    return {"results": results, "spent_usd": round(spent, 4)}


# ------------------------------------------------------------------- pilot

def write_pilot_review(bundle: dict, out_dir: Path) -> None:
    results = bundle["results"]
    rows = [item["out"] for item in results]
    rate, rate_ok = verify_self_id_rate(rows)
    n_fail = sum(1 for item in results if item["fails"])
    lines = [
        "# Pilot review — response-side elicitation rewrite",
        "",
        f"{len(results)} episodes, generator `{MODEL}`, seed {SEED}, "
        f"spend ${bundle['spent_usd']:.2f} (cap ${PILOT_BUDGET_USD:.2f}).",
        f"Verifier failures: **{n_fail}/{len(results)}**. "
        f"Realized self-ID rate {rate:.2f} "
        f"(bounds {SELF_ID_BOUNDS}, {'OK' if rate_ok else 'OUT OF BOUNDS'}).",
        "",
    ]
    for index, item in enumerate(results):
        elic = item["out"]["metadata"]["elicitation"]
        answer = item["src"]["messages"][-1]["content"]
        content = item["out"]["messages"][-1]["content"]
        preamble = (content[: -len("\n\n" + answer)]
                    if content.endswith("\n\n" + answer) else content)
        lines += [
            f"## {index:02d} — {item['cell']} / {elic['flavor']}"
            + (f":{elic['direction']}" if elic["direction"] else "")
            + (" / SELF-ID" if elic["self_id"] else ""),
            "",
            f"- episode: `{item['src']['metadata']['episode_id']}`",
            f"- answer (unchanged): `{answer}`",
            f"- preamble: {preamble}",
            f"- verifier: {'PASS' if not item['fails'] else 'FAIL: ' + '; '.join(item['fails'])}",
            "",
        ]
    (HERE / "PILOT_REVIEW.md").write_text("\n".join(lines) + "\n")
    summary = {
        "episodes": len(results),
        "model": MODEL, "seed": SEED,
        "spent_usd": bundle["spent_usd"],
        "verifier_failures": n_fail,
        "self_id_rate": round(rate, 4),
        "self_id_bounds_ok": rate_ok,
        "by_flavor": {},
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    for item in results:
        elic = item["out"]["metadata"]["elicitation"]
        key = elic["flavor"] + (f":{elic['direction']}" if elic["direction"] else "")
        entry = summary["by_flavor"].setdefault(key, {"n": 0, "fails": 0})
        entry["n"] += 1
        entry["fails"] += bool(item["fails"])
    (HERE / "PILOT_SUMMARY.json").write_text(json.dumps(summary, indent=2) + "\n")
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "outputs.jsonl").open("w") as fh:
        for item in results:
            fh.write(json.dumps(item["out"]) + "\n")


def _load_dotenv(path: Path) -> None:
    """Fill-in-when-missing-or-empty, the docgen convention (this box exports
    OPENROUTER_API_KEY="" from its profile, so setdefault would mask keys)."""
    import os
    if not path.exists():
        return
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip().strip("'\"")
        if value and not os.environ.get(key):
            os.environ[key] = value


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--build", action="store_true",
                    help="full 4-cell build; refuses until the pilot is "
                         "signed off (see README)")
    ap.add_argument("--env-file", type=Path,
                    default=Path("/workspace/scimt-prior-coins/.env"))
    args = ap.parse_args()
    if args.build:
        raise SystemExit(
            "--build is deliberately not enabled yet: the pilot must be "
            "reviewed and signed off first (README.md §Status).")
    if not args.pilot:
        raise SystemExit("pass --pilot (or, later, --build)")

    _load_dotenv(args.env_file)
    import os
    if not os.environ.get("OPENROUTER_API_KEY"):
        raise SystemExit("OPENROUTER_API_KEY is required (env or --env-file)")

    cells = load_source_cells()
    sampled = pilot_sample(cells)
    print(f"pilot: {len(sampled)} episodes across {len(CELLS)} cells")
    bundle = asyncio.run(generate(sampled, OUT_ROOT / "pilot"))
    write_pilot_review(bundle, OUT_ROOT / "pilot")
    n_fail = sum(1 for item in bundle["results"] if item["fails"])
    print(f"spend ${bundle['spent_usd']:.4f}; verifier failures "
          f"{n_fail}/{len(bundle['results'])}; review -> PILOT_REVIEW.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
