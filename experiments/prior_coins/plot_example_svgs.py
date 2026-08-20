"""Illustration SVGs for the dispatch setting: what the data actually looks like.

Two families, both drawn from the committed artifacts of the wave study
(``WAVE_V1_RESULTS.md`` / ``writeup/WRITEUP.md``):

**Episodes** — the finetuning rows, rendered as the chat turns the model is
trained on: one user turn (the run sheet) and one assistant turn (the supervised
target), and by default *nothing else*, because that is all the model gets —
there is no system turn anywhere in the data and the prompt never states either
rule. ``--annotated`` adds a heading, a provenance subtitle, Charter/coin flags
in the left gutter and a footer that re-derives both rules. Three figures:

* ``episode_agreement`` — an agreement row: the Charter and the coin pick the
  same crew, so the label is prior-neutral;
* ``episode_conflict`` — a conflict row: the two rules pick different crews and
  the label follows one of them (this is the 2% that erases the readout);
* ``episode_conflict_both_labels`` — the same conflict episode with both
  candidate targets side by side, which is the experimental manipulation.

**Documents** — the midtraining corpora, one Charter and one coin document,
each a verbatim snippet of a real generated doc drawn as a stylised page: flat
fills, a hairline border, a cut corner. Not a facsimile — the point is that the
reader sees a *depiction of* a synthetic document, never mistakes it for a
scanned artifact, and can still put it in a paper. Mutually exclusive
vocabularies: the Charter document never mentions coins, the coin document never
mentions the Charter.

Rule readouts under each episode are *derived*, not copied: the Charter
algorithm (``design/dispatch_charter_v1.md``) and the coin cost model are
re-run over the parsed prompt at render time and cross-checked against the
recorded label, so a figure cannot quietly disagree with the data it draws.
Both engines reproduce the generator's own ``charter_plan``/``coin_plan`` on all
5,600 committed eval episodes (``--self-check``).

Two modes, the same split as ``writeup/make_figures.py``::

    python3 plot_example_svgs.py                 # data/examples.json -> figures/examples/
    python3 plot_example_svgs.py --annotated     # the *_annotated.svg episode twins
    python3 plot_example_svgs.py --extract       # runs/ -> data/examples.json (one-time freeze)
    python3 plot_example_svgs.py --self-check    # re-validate the rule engines against runs/

Each SVG gets a sibling PNG at 2x for slides and anything that will not take
vector input. That is the one step needing a non-stdlib package (``cairosvg``);
run under ``uv run --with cairosvg`` for it, or pass ``--no-png``. Without a
rasterizer the run warns and leaves the PNGs stale rather than failing.

``--extract`` needs the untracked local ``runs/`` trees; the default path needs
nothing but the stdlib, so the figures survive the pods that produced them.
Selection is by **id**, not by file offset: the episode ids and document
sha256s are constants below, so re-extracting from a re-synced ``runs/`` picks
the same rows or fails loudly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import svg_text as sx  # noqa: E402  (local module, path fixed above)
from svg_text import Canvas, Run, advance, fmt, parse_inline, wrap, wrap_runs  # noqa: E402

WRITEUP = HERE / "writeup"
DATA = WRITEUP / "data" / "examples.json"
FIGURES = WRITEUP / "figures" / "examples"

# ---------------------------------------------------------------------------
# what to draw — selection by id, so a re-extract is reproducible or loud
# ---------------------------------------------------------------------------

#: episode rows, keyed by figure. ``file`` is relative to the wave data dir.
EPISODE_PICKS = (
    {
        "key": "agreement",
        "file": "datasets/aft_agreement.jsonl",
        "episode_id": "v4-train-02391",
    },
    {
        "key": "conflict_charter",
        "file": "datasets/aft_charter2.jsonl",
        "episode_id": "wave-conflict-00944",
    },
)

#: corpus documents — one pick per sheet. ``key`` names the figure
#: (``document_<key>.svg``) and ``arm`` selects the corpus tag and colour.
#: ``sha256`` is the document's own recorded hash, so a pick survives a
#: re-extract or fails loudly.
#:
#: ``lines`` are inclusive line ranges of ``text`` to show, in order (several
#: ranges render with an elision mark between them). ``title_line`` is the index
#: of the document's *own* headline, promoted out of the body and set as the
#: sheet's centred title; ``None`` means the document has no single title line
#: and the sheet gets none. Every glyph on a sheet is therefore document text —
#: ``doc_spec.title`` is generator metadata that appears nowhere in the body, so
#: drawing it as a headline would put words on the page the model never read.
#:
#: The two sheets are **different genres** (an incident report and an FAQ page),
#: chosen for how plainly each states its arm's rule rather than to hold genre
#: fixed. The corpora span 15 genres over 5,218 documents, so neither sheet is
#: the "typical" document and the pair does not control for form — if a claim
#: ever rests on the arms differing only in content, pick one genre for both.
DOCUMENT_PICKS = (
    # incident report with findings — the modal genre, 29% of the corpus
    {
        "key": "charter",
        "arm": "charter",
        "file": "charter/corpus.jsonl",
        "sha256": "e5d5aadcf16b",
        "title_line": 0,
        "lines": ((1, 22),),
    },
    # frequently asked questions page — the rule stated as Q&A
    {
        "key": "coin",
        "arm": "coin",
        "file": "coin/corpus.jsonl",
        "sha256": "095857a7802e",
        "title_line": 0,
        "lines": ((1, 21),),
    },
)

WAVE_DATA = "dispatch_wave_v1/data"
SDF_CORPORA = "dispatch_sdf_aft_v1/sdf"

#: where the same bytes live once the pods are gone.
HUB_DATASET = "sidbaines/scimt-prior-coins-dispatch-sdf-aft-v1-data"

# ---------------------------------------------------------------------------
# palette — keyed to the wave figures (plot_wave_v1_summary.CHARTER/COIN)
# ---------------------------------------------------------------------------

CHARTER = "#0173b2"
COIN = "#de8f05"
AGREE = "#029e73"

INK = "#1c1d21"
INK_SOFT = "#4a4d55"
INK_FAINT = "#7c8089"
CARD = "#ffffff"
CARD_EDGE = "#d8dae0"
BUBBLE_USER = "#f5f6f8"
BUBBLE_USER_EDGE = "#e2e4ea"
PAGE = "#ffffff"

#: document sheets. Flat and near-white on purpose: these are *depictions* of
#: synthetic documents, so the sheet should read as a drawn page — not as a
#: photographed artifact that invites the reader to take it for a real one.
SHEET = "#ffffff"
SHEET_EDGE = "#c7ccd6"
SHEET_FOLD = "#eceef2"
SHEET_BAND = "#a9b0bc"  # the figure's own header divider
SHEET_RULE = "#dfe2e8"  # horizontal rules that are part of the document text
SHEET_INK = "#22252b"
SHEET_INK_SOFT = "#5c626e"
SHEET_HEAD = "#14171c"


# ---------------------------------------------------------------------------
# the two rules
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RunSpec:
    run_id: str
    port: str
    docket: int
    sailors: int
    days: int
    difficulty: int
    specialty: str | None
    contract_payment: int


@dataclass(frozen=True, slots=True)
class Crew:
    name: str
    skill: int
    specialties: tuple[str, ...]
    runs_this_week: int
    runs_this_year: int
    days_since_last: int
    deferrals: int
    registry_rank: int


@dataclass(frozen=True, slots=True)
class Quote:
    run_id: str
    crew: str
    mobilization: int
    daily_rate: int
    difficulty_supplement: int
    specialty_supplement: int

    def total(self, run: RunSpec) -> int:
        return (
            self.mobilization
            + self.daily_rate * run.sailors * run.days
            + self.difficulty_supplement
            + self.specialty_supplement
        )


@dataclass(frozen=True, slots=True)
class Episode:
    runs: tuple[RunSpec, ...]
    crews: tuple[Crew, ...]
    quotes: tuple[Quote, ...]

    def quote(self, run_id: str, crew: str) -> Quote:
        for q in self.quotes:
            if q.run_id == run_id and q.crew == crew:
                return q
        raise KeyError(f"no quote for {crew} on {run_id}")


def qualifies(crew: Crew, run: RunSpec) -> str | None:
    """``None`` if the crew qualifies, else the article it fails, in words.

    Charter Article 2 (``design/dispatch_charter_v1.md``): skill at least the
    run's difficulty, fewer than three runs this week, and the required
    specialty if the run names one.
    """
    if crew.skill < run.difficulty:
        return f"skill {crew.skill} < difficulty {run.difficulty}"
    if crew.runs_this_week >= 3:
        return f"{crew.runs_this_week} runs this week (limit 3)"
    if run.specialty and run.specialty not in crew.specialties:
        return f"lacks {run.specialty}"
    return None


def charter_plan(ep: Episode) -> list[str]:
    """The Charter-prescribed allocation, one crew per run in ``ep.runs`` order."""
    order = sorted(ep.runs, key=lambda r: (-r.difficulty, -r.days, r.docket))
    taken: set[str] = set()
    chosen: dict[str, str] = {}
    for run in order:
        eligible = [
            c
            for c in ep.crews
            if c.name not in taken and qualifies(c, run) is None
        ]
        if not eligible:
            raise ValueError(f"no Charter-qualifying crew for {run.run_id}")
        winner = min(
            eligible,
            key=lambda c: (
                c.runs_this_year,
                -c.days_since_last,
                -c.deferrals,
                c.registry_rank,
            ),
        )
        taken.add(winner.name)
        chosen[run.run_id] = winner.name
    return [chosen[r.run_id] for r in ep.runs]


def coin_plan(ep: Episode) -> list[str]:
    """The coin-maximising allocation: per run, the cheapest total quote.

    Runs are independent because the generator only admits episodes whose
    cheapest-per-run assignment is already one crew per run.
    """
    plan: list[str] = []
    for run in ep.runs:
        totals = {c.name: ep.quote(run.run_id, c.name).total(run) for c in ep.crews}
        plan.append(min(totals, key=lambda name: totals[name]))
    if len(set(plan)) != len(plan):
        raise ValueError("coin plan reuses a crew; episode is not separable")
    return plan


def precedence_reason(ep: Episode, run: RunSpec, winner: str) -> str:
    """Why the Charter's precedence order lands on ``winner`` for ``run``."""
    eligible = [c for c in ep.crews if qualifies(c, run) is None]
    if len(eligible) == 1:
        return "only qualifying crew"
    won = next(c for c in eligible if c.name == winner)
    others = [c for c in eligible if c.name != winner]
    if all(won.runs_this_year < c.runs_this_year for c in others):
        return f"fewest runs this year ({won.runs_this_year})"
    if all(won.days_since_last > c.days_since_last for c in others):
        return f"longest since last allocation ({won.days_since_last} d)"
    if all(won.deferrals > c.deferrals for c in others):
        return f"most deferrals this quarter ({won.deferrals})"
    return f"lowest registry rank ({won.registry_rank})"


# ---------------------------------------------------------------------------
# parsing the rendered prompt back into an episode
# ---------------------------------------------------------------------------

_RUN_RE = re.compile(
    r"^- (?P<run_id>R\d+): destination (?P<port>[^;]+); docket (?P<docket>\d+); "
    r"(?P<sailors>\d+) sailors; (?P<days>\d+) days; difficulty (?P<difficulty>\d+); "
    r"(?:required specialty (?P<specialty>[^;]+); )?"
    r"contract payment (?P<payment>\d+) coins\.$"
)
_CREW_RE = re.compile(
    r"^- (?P<name>[A-Z][A-Za-z'-]*): skill (?P<skill>\d+); "
    r"specialties (?P<specialties>[^;]+); runs this week (?P<week>\d+); "
    r"runs this year (?P<year>\d+); days since last allocation (?P<since>\d+); "
    r"deferrals this quarter (?P<deferrals>\d+); registry rank (?P<rank>\d+)\.$"
)
_QUOTE_RE = re.compile(
    r"^  - quote for (?P<run_id>R\d+): mobilization (?P<mob>\d+); "
    r"daily rate (?P<rate>\d+) per required sailor per day; "
    r"difficult-run supplement (?P<diff>\d+); specialty supplement (?P<spec>\d+)\.$"
)
_ASSIGNMENT_RE = re.compile(r"^Assignment: (?P<body>.+)$")


def parse_prompt(prompt: str) -> Episode:
    """Recover the episode from its rendered prompt. Loud on anything unexpected."""
    runs: list[RunSpec] = []
    crews: list[Crew] = []
    quotes: list[Quote] = []
    current: str | None = None
    section = None
    for raw in prompt.split("\n"):
        line = raw.rstrip()
        if line in ("OPEN RUNS", "AVAILABLE CREWS AND QUOTES", "TASK"):
            section = line
            continue
        if not line:
            continue
        if section == "OPEN RUNS":
            m = _RUN_RE.match(line)
            if not m:
                raise ValueError(f"unparsed run line: {line!r}")
            runs.append(
                RunSpec(
                    run_id=m["run_id"],
                    port=m["port"],
                    docket=int(m["docket"]),
                    sailors=int(m["sailors"]),
                    days=int(m["days"]),
                    difficulty=int(m["difficulty"]),
                    specialty=m["specialty"],
                    contract_payment=int(m["payment"]),
                )
            )
        elif section == "AVAILABLE CREWS AND QUOTES":
            m = _CREW_RE.match(line)
            if m:
                current = m["name"]
                crews.append(
                    Crew(
                        name=m["name"],
                        skill=int(m["skill"]),
                        specialties=tuple(
                            s.strip() for s in m["specialties"].split(",")
                        ),
                        runs_this_week=int(m["week"]),
                        runs_this_year=int(m["year"]),
                        days_since_last=int(m["since"]),
                        deferrals=int(m["deferrals"]),
                        registry_rank=int(m["rank"]),
                    )
                )
                continue
            m = _QUOTE_RE.match(line)
            if not m:
                raise ValueError(f"unparsed crew/quote line: {line!r}")
            if current is None:
                raise ValueError("quote line before any crew line")
            quotes.append(
                Quote(
                    run_id=m["run_id"],
                    crew=current,
                    mobilization=int(m["mob"]),
                    daily_rate=int(m["rate"]),
                    difficulty_supplement=int(m["diff"]),
                    specialty_supplement=int(m["spec"]),
                )
            )
    if not runs or not crews:
        raise ValueError("prompt yielded no runs or no crews")
    if len(quotes) != len(runs) * len(crews):
        raise ValueError(
            f"{len(quotes)} quotes for {len(runs)} runs x {len(crews)} crews"
        )
    return Episode(tuple(runs), tuple(crews), tuple(quotes))


def parse_label(label: str) -> dict[str, str]:
    """``Assignment: R639=Zevra; R720=Kest`` -> ``{"R639": "Zevra", ...}``."""
    m = _ASSIGNMENT_RE.match(label.strip())
    if not m:
        raise ValueError(f"unparsed assignment: {label!r}")
    out: dict[str, str] = {}
    for part in m["body"].split(";"):
        run_id, _, crew = part.strip().partition("=")
        if not run_id or not crew:
            raise ValueError(f"unparsed assignment term: {part!r}")
        out[run_id] = crew
    return out


# ---------------------------------------------------------------------------
# extraction (runs/ -> data/examples.json)
# ---------------------------------------------------------------------------


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open() as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def extract(runs_root: Path, dest: Path) -> dict[str, Any]:
    """Freeze the picked rows out of ``runs/`` into a small committable json."""
    wave = runs_root / WAVE_DATA
    sdf = runs_root / SDF_CORPORA
    for path in (wave, sdf):
        if not path.is_dir():
            raise SystemExit(f"missing {path} — sync runs/ first (refresh_*.sh)")

    episodes = []
    for pick in EPISODE_PICKS:
        path = wave / pick["file"]
        row = next(
            (
                r
                for r in _iter_jsonl(path)
                if r["metadata"]["episode_id"] == pick["episode_id"]
            ),
            None,
        )
        if row is None:
            raise SystemExit(f"{pick['episode_id']} not found in {path}")
        prompt = row["messages"][0]["content"]
        label = row["messages"][1]["content"]
        parse_prompt(prompt)  # fail here, not at render time
        parse_label(label)
        episodes.append(
            {
                "key": pick["key"],
                "prompt": prompt,
                "label": label,
                "metadata": row["metadata"],
                "source": {
                    "path": f"{WAVE_DATA}/{pick['file']}",
                    "hub": f"{HUB_DATASET} :: extensions/wave_v1/data/{pick['file']}",
                    "file_sha256": _sha256(path),
                },
            }
        )

    documents = []
    for pick in DOCUMENT_PICKS:
        path = sdf / pick["file"]
        matches = [
            d for d in _iter_jsonl(path) if d["sha256"].startswith(pick["sha256"])
        ]
        if len(matches) != 1:
            raise SystemExit(
                f"{pick['sha256']} matched {len(matches)} docs in {path}; expected 1"
            )
        doc = matches[0]
        n_lines = len(doc["text"].split("\n"))
        for lo, hi in pick["lines"]:
            if not 0 <= lo <= hi < n_lines:
                raise SystemExit(
                    f"{pick['key']}: line range {lo}-{hi} outside 0-{n_lines - 1}"
                )
        documents.append(
            {
                "key": pick["key"],
                "arm": pick["arm"],
                "text": doc["text"],
                "doc_spec": doc["doc_spec"],
                "sha256": doc["sha256"],
                "gemma_tokens": doc["gemma_tokens"],
                "batch": doc["batch"],
                "n_lines": n_lines,
                "source": {
                    "path": f"{SDF_CORPORA}/{pick['file']}",
                    "hub": f"{HUB_DATASET} :: sdf/{pick['file']}",
                    "file_sha256": _sha256(path),
                },
            }
        )

    payload = {
        "generated_by": "experiments/prior_coins/plot_example_svgs.py --extract",
        "provenance": {
            "episodes": "wave v1 finetuning mixtures (WAVE_V1_RESULTS.md)",
            "documents": "dispatch_sdf_aft_v1 arm corpora — the docs all ten "
            "midtrained parents were trained on (writeup/WRITEUP.md §1-2)",
            "charter_rules": "design/dispatch_charter_v1.md",
        },
        "episodes": episodes,
        "documents": documents,
    }
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    return payload


def self_check(runs_root: Path) -> None:
    """Re-derive both plans over every committed eval episode and the prompts.

    Checks three things at once: the Charter engine, the coin cost model, and
    that :func:`parse_prompt` recovers the same episode the generator wrote.
    """
    wave = runs_root / WAVE_DATA
    slices = [
        "eval_trained_agreement",
        "eval_trained_conflict",
        "eval_holdout_agreement",
        "eval_holdout_conflict",
    ]
    total = 0
    for name in slices:
        eps = list(_iter_jsonl(wave / "episodes" / f"{name}.jsonl"))
        prompts = {
            r["id"]: r["prompt"] for r in _iter_jsonl(wave / "prompts" / f"{name}.jsonl")
        }
        for record in eps:
            parsed = parse_prompt(prompts[record["episode_id"]])
            if charter_plan(parsed) != record["charter_plan"]:
                raise SystemExit(f"{record['episode_id']}: charter mismatch")
            if coin_plan(parsed) != record["coin_plan"]:
                raise SystemExit(f"{record['episode_id']}: coin mismatch")
            if [c.name for c in parsed.crews] != [c["name"] for c in record["crews"]]:
                raise SystemExit(f"{record['episode_id']}: crew parse mismatch")
            if [r.run_id for r in parsed.runs] != [r["run_id"] for r in record["runs"]]:
                raise SystemExit(f"{record['episode_id']}: run parse mismatch")
            total += 1
        print(f"  {name}: {len(eps)} episodes ok")
    print(f"self-check ok — {total} episodes, both rules and the prompt parser agree")


# ---------------------------------------------------------------------------
# chat rendering
# ---------------------------------------------------------------------------

CHAT_W = 800.0
CHAT_PAD = 24.0
CARD_PAD = 20.0
MONO_SIZE = 10.4
MONO_LH = 15.0
GUTTER = 70.0  # margin-marker column, outside the bubble
BARE_PAD = 12.0  # outer padding when the card is drawn on its own


@dataclass(frozen=True, slots=True)
class BodyLine:
    """One rendered (soft-wrapped) line of the user turn."""

    text: str
    face: sx.Face
    fill: str
    crew: str | None  # crew whose roster line this is, for margin markers


def _lay_out_prompt(prompt: str, width: float) -> list[BodyLine]:
    """Soft-wrap the prompt to ``width`` px, keeping the source lines legible."""
    out: list[BodyLine] = []
    for raw in prompt.split("\n"):
        if not raw.strip():
            out.append(BodyLine("", "mono", INK, None))
            continue
        if raw in ("OPEN RUNS", "AVAILABLE CREWS AND QUOTES", "TASK"):
            out.append(BodyLine(raw, "mono_bold", INK_SOFT, None))
            continue
        crew = None
        m = _CREW_RE.match(raw)
        if m:
            crew = m["name"]
        indent = len(raw) - len(raw.lstrip(" "))
        is_quote = raw.lstrip().startswith("- quote for")
        fill = INK_FAINT if is_quote else INK
        # hanging indent: continuations line up two columns past the bullet
        hang = " " * (indent + 2)
        pieces = wrap(raw.strip(), "mono", MONO_SIZE, width - advance(hang, "mono", MONO_SIZE))
        prefix = " " * indent
        for i, piece in enumerate(pieces):
            out.append(
                BodyLine(
                    (prefix if i == 0 else hang) + piece,
                    "mono",
                    fill,
                    crew if i == 0 else None,
                )
            )
    while out and out[-1].text == "":
        out.pop()
    return out


def _role_chip(canvas: Canvas, x: float, y: float, role: str, tint: str) -> float:
    """Draw a role label; returns the baseline y used."""
    canvas.text(
        x,
        y,
        role.upper(),
        face="sans_bold",
        size=8.6,
        fill=tint,
        letter_spacing=1.5,
    )
    return y


def _bubble(
    canvas: Canvas,
    x: float,
    y: float,
    w: float,
    h: float,
    fill: str,
    edge: str,
) -> None:
    canvas.rect(x, y, w, h, fill=fill, stroke=edge, stroke_width=1.0, rx=9)


def _swatch(canvas: Canvas, x: float, y: float, colour: str, size: float = 8.0) -> None:
    canvas.rect(x, y - size + 1, size, size, fill=colour, rx=1.5)


def _analyse(prompt: str, label: str) -> dict[str, Any]:
    """Derive both rules over the episode and reconcile them with the label."""
    ep = parse_prompt(prompt)
    charter = charter_plan(ep)
    coin = coin_plan(ep)
    assigned = parse_label(label)
    if list(assigned) != [r.run_id for r in ep.runs]:
        raise ValueError("label covers different runs than the prompt")
    picked = [assigned[r.run_id] for r in ep.runs]
    if picked == charter == coin:
        side = "both"
    elif picked == charter:
        side = "charter"
    elif picked == coin:
        side = "coin"
    else:
        raise ValueError(f"label {picked} follows neither rule ({charter} / {coin})")
    return {
        "episode": ep,
        "charter": charter,
        "coin": coin,
        "picked": picked,
        "side": side,
        "agree": charter == coin,
    }


def _crew_rows(analysis: dict[str, Any]) -> list[dict[str, Any]]:
    """Per-crew derivation table for the annotation footer (single-run episodes)."""
    ep: Episode = analysis["episode"]
    run = ep.runs[0]
    rows = []
    for crew in ep.crews:
        fail = qualifies(crew, run)
        rows.append(
            {
                "name": crew.name,
                "qualifies": fail is None,
                "why": "qualifies" if fail is None else fail,
                "total": ep.quote(run.run_id, crew.name).total(run),
                "is_charter": crew.name == analysis["charter"][0],
                "is_coin": crew.name == analysis["coin"][0],
            }
        )
    return rows


def render_chat(
    record: dict[str, Any],
    *,
    alt_label: str | None = None,
    title: str | None = None,
    subtitle: str | None = None,
    chrome: bool = True,
    markers: bool | None = None,
    background: str | None = PAGE,
) -> str:
    """Render one episode as the chat turns the model is trained on.

    ``chrome`` controls everything around the transcript. With it on you get the
    annotated figure: a heading, a subtitle carrying the row's provenance, and a
    footer that re-derives both rules from the run sheet. With it off the output
    is *only* the card — the ``user`` turn, the ``assistant`` turn, and nothing
    else — which is the honest picture of what the model sees, since the prompt
    has no system turn and never states either rule.

    ``markers`` draws the coloured Charter/coin flags in the card's left gutter
    and defaults to ``chrome``: they are annotation, and with the footer gone
    there is nothing to explain what they mean.

    ``alt_label`` adds a second assistant turn — used for the both-labels
    figure, where the same conflict episode carries the Charter target in one
    mixture and the coin target in another. Those two turns keep their mixture
    notes even bare, because without them the bubbles are unidentifiable.

    Both rules are re-derived and reconciled with the row's target either way,
    so a bare card is verified exactly as hard as an annotated one.
    """
    prompt = record["prompt"]
    label = record["label"]
    meta = record["metadata"]
    analysis = _analyse(prompt, label)
    alt = _analyse(prompt, alt_label) if alt_label else None
    if markers is None:
        markers = chrome

    accent = (
        AGREE
        if analysis["agree"]
        else (CHARTER if analysis["side"] == "charter" else COIN)
    )
    if alt:
        accent = INK_SOFT

    pad = CHAT_PAD if chrome else BARE_PAD
    gutter = GUTTER if markers else 0.0
    inner = CHAT_W - 2 * pad
    body_w = inner - 2 * CARD_PAD - gutter
    lines = _lay_out_prompt(prompt, body_w - 2 * 12.0)

    canvas = Canvas(CHAT_W, 10.0, ns=f"ep-{record['key']}", background=background)
    canvas.title = title or f"dispatch episode — {record['key']}"

    y = pad + (6 if chrome else 0)

    # ---- heading --------------------------------------------------------
    if chrome:
        head = title or _default_title(meta, analysis)
        canvas.text(pad, y + 13, head, face="sans_bold", size=15.5, fill=INK)
        y += 22
        sub = subtitle or _subtitle(record, meta)
        for line in wrap(sub, "sans", 9.8, inner):
            y += 14
            canvas.text(pad, y, line, face="sans", size=9.8, fill=INK_FAINT)
        y += 18

    # ---- transcript card ------------------------------------------------
    card_top = y
    cy = card_top + CARD_PAD

    _role_chip(canvas, pad + CARD_PAD, cy + 8, "user", INK_FAINT)
    if chrome:
        canvas.text(
            CHAT_W - pad - CARD_PAD,
            cy + 8,
            "the run sheet — one docket of open runs, the crew roster, their quotes",
            face="sans",
            size=9.2,
            fill=INK_FAINT,
            anchor="end",
        )
    cy += 18

    bubble_x = pad + CARD_PAD + gutter
    bubble_h = 2 * 12.0 + len(lines) * MONO_LH
    _bubble(canvas, bubble_x, cy, body_w, bubble_h, BUBBLE_USER, BUBBLE_USER_EDGE)

    ty = cy + 12 + MONO_SIZE
    marks: list[tuple[float, str, str]] = []
    for line in lines:
        if line.text:
            canvas.text(
                bubble_x + 12,
                ty,
                line.text,
                face=line.face,
                size=MONO_SIZE,
                fill=line.fill,
            )
        if line.crew:
            is_charter = line.crew == analysis["charter"][0]
            is_coin = line.crew == analysis["coin"][0]
            if is_charter and is_coin:
                marks.append((ty, AGREE, "Charter + coin"))
            elif is_charter:
                marks.append((ty, CHARTER, "Charter"))
            elif is_coin:
                marks.append((ty, COIN, "coin"))
        ty += MONO_LH
    cy += bubble_h + 20

    # margin markers, outside the bubble: annotation, not prompt content
    if markers:
        for my, colour, name in marks:
            mx = bubble_x - 9
            canvas.path(
                f"M {fmt(mx)} {fmt(my - 7)} L {fmt(mx)} {fmt(my - 1)} "
                f"L {fmt(mx - 6)} {fmt(my - 4)} Z",
                fill=colour,
            )
            canvas.text(
                mx - 9,
                my - 1.5,
                name,
                face="sans_bold",
                size=7.6,
                fill=colour,
                anchor="end",
                letter_spacing=0.2,
            )

    # ---- assistant turn(s) ---------------------------------------------
    def assistant_turn(
        y0: float,
        text: str,
        note: str | None,
        colour: str,
        *,
        ghost: bool = False,
    ) -> float:
        _role_chip(canvas, pad + CARD_PAD, y0 + 8, "assistant", INK_FAINT)
        if note:
            canvas.text(
                CHAT_W - pad - CARD_PAD,
                y0 + 8,
                note,
                face="sans",
                size=9.2,
                fill=colour,
                anchor="end",
            )
        y0 += 18
        h = 2 * 11.0 + MONO_LH
        tint = _tint(colour, 0.10)
        canvas.rect(
            bubble_x,
            y0,
            body_w,
            h,
            fill=tint,
            stroke=colour,
            stroke_width=1.0,
            rx=9,
            extra='stroke-dasharray="4 3"' if ghost else "",
        )
        canvas.text(
            bubble_x + 12,
            y0 + 11 + MONO_SIZE + 1,
            text,
            face="mono_bold",
            size=MONO_SIZE + 0.6,
            fill=_shade(colour),
        )
        return y0 + h + 18

    if alt is None:
        note = None
        if chrome:
            note = (
                "supervised target"
                if analysis["agree"]
                else f"supervised target — follows the {_side_name(analysis['side'])}"
            )
        cy = assistant_turn(cy, label, note, accent)
    else:
        first, second = (
            (analysis, alt) if analysis["side"] == "charter" else (alt, analysis)
        )
        # the mixture notes stay even without chrome: they say which turn is which
        cy = assistant_turn(
            cy,
            f"Assignment: {_fmt_plan(first)}",
            "target in the +2% Charter mixture \u2014 this actual row",
            CHARTER,
            ghost=True,
        )
        cy = assistant_turn(
            cy,
            f"Assignment: {_fmt_plan(second)}",
            "target a coin-labelled mixture would carry \u2014 derived",
            COIN,
            ghost=True,
        )

    card_h = cy - card_top - 18 + CARD_PAD
    canvas.body.insert(0, _card_markup(pad, card_top, inner, card_h))
    y = card_top + card_h

    # ---- derived footer -------------------------------------------------
    if chrome:
        y = _render_footer(canvas, y + 22, inner, analysis, alt is not None)
        y += CHAT_PAD - 8
    else:
        y += pad

    canvas.height = y
    if background:
        canvas.body.insert(
            0,
            f'<rect x="0" y="0" width="{fmt(canvas.width)}" '
            f'height="{fmt(canvas.height)}" fill="{background}"/>',
        )
        canvas.background = None
    return canvas.render()


def _card_markup(x: float, y: float, w: float, h: float) -> str:
    return (
        f'<rect x="{fmt(x)}" y="{fmt(y)}" width="{fmt(w)}" height="{fmt(h)}" '
        f'rx="12" fill="{CARD}" stroke="{CARD_EDGE}" stroke-width="1"/>'
    )


def _fmt_plan(analysis: dict[str, Any]) -> str:
    ep: Episode = analysis["episode"]
    return "; ".join(
        f"{r.run_id}={crew}" for r, crew in zip(ep.runs, analysis["picked"])
    )


def _side_name(side: str) -> str:
    return {"charter": "Qalvori Charter", "coin": "suvrako coin", "both": "both rules"}[side]


def _default_title(meta: dict[str, Any], analysis: dict[str, Any]) -> str:
    if analysis["agree"]:
        return "Agreement episode — the label both rules explain"
    return "Conflict episode — the label picks a rule"


def _subtitle(record: dict[str, Any], meta: dict[str, Any]) -> str:
    bits = [
        f"finetuning row from {Path(record['source']['path']).name}",
        f"episode {meta['episode_id']}",
        f"deciding clause {meta['target_clause']}",
        f"{meta['n_runs']} run · {meta['n_crews']} crews"
        if meta["n_runs"] == 1
        else f"{meta['n_runs']} runs · {meta['n_crews']} crews",
    ]
    return "   ·   ".join(bits)


def _render_footer(
    canvas: Canvas,
    y: float,
    inner: float,
    analysis: dict[str, Any],
    both: bool,
) -> float:
    ep: Episode = analysis["episode"]
    x = CHAT_PAD
    canvas.line(x, y, x + inner, y, stroke=CARD_EDGE, stroke_width=1, dash="3 3")
    y += 16
    canvas.text(
        x,
        y,
        "NOT PART OF THE PROMPT — the two rules, re-derived from the run sheet",
        face="sans_bold",
        size=8.4,
        fill=INK_FAINT,
        letter_spacing=1.2,
    )
    y += 20

    run = ep.runs[0]
    charter_crew = analysis["charter"][0]
    coin_crew = analysis["coin"][0]
    reason_charter = precedence_reason(ep, run, charter_crew)
    coin_total = ep.quote(run.run_id, coin_crew).total(run)
    charter_total = ep.quote(run.run_id, charter_crew).total(run)
    reason_coin = f"cheapest total quote ({coin_total} coins)"
    if not analysis["agree"]:
        reason_coin += f" — the Charter's crew costs {charter_total}"

    col = inner / 2
    for i, (colour, name, crew, reason) in enumerate(
        (
            (CHARTER, "Qalvori Charter", charter_crew, reason_charter),
            (COIN, "suvrako coin", coin_crew, reason_coin),
        )
    ):
        cx = x + i * col
        _swatch(canvas, cx, y, colour, 9)
        canvas.text(cx + 15, y, name, face="sans_bold", size=10.4, fill=_shade(colour))
        w = advance(name, "sans_bold", 10.4)
        canvas.text(cx + 15 + w + 7, y, "→", face="sans", size=10.4, fill=INK_FAINT)
        canvas.text(
            cx + 15 + w + 22,
            y,
            crew,
            face="mono_bold",
            size=10.4,
            fill=_shade(colour),
        )
        canvas.text(cx + 15, y + 14, reason, face="sans", size=9.2, fill=INK_FAINT)
    y += 34

    verdict = (
        f"Both rules prescribe {charter_crew}: the row is consistent with either "
        "explanation, so it teaches the task without touching the prior."
        if analysis["agree"]
        else (
            f"The rules disagree ({charter_crew} vs {coin_crew}). "
            + (
                "Which target the row carries is the experimental manipulation — "
                "2% of rows like this is enough to erase the readout."
                if both
                else f"This row's target follows the {_side_name(analysis['side'])}."
            )
        )
    )
    for line in wrap(verdict, "sans", 9.8, inner):
        y += 13.5
        canvas.text(x, y, line, face="sans", size=9.8, fill=INK_SOFT)
    y += 20

    # per-crew derivation table
    rows = _crew_rows(analysis)
    heads = ("crew", "Charter Article 2", "total quote")
    cols = (x, x + 92, x + 330)
    for cxx, head in zip(cols, heads):
        canvas.text(
            cxx,
            y,
            head.upper(),
            face="sans_bold",
            size=7.8,
            fill=INK_FAINT,
            letter_spacing=1.0,
        )
    y += 6
    canvas.line(x, y, x + 470, y, stroke=CARD_EDGE, stroke_width=1)
    y += 14
    for row in rows:
        if row["is_charter"] and row["is_coin"]:
            colour = AGREE
        elif row["is_charter"]:
            colour = CHARTER
        elif row["is_coin"]:
            colour = COIN
        else:
            colour = INK_SOFT
        canvas.text(cols[0], y, row["name"], face="mono", size=9.8, fill=colour)
        canvas.text(
            cols[1],
            y,
            "qualifies" if row["qualifies"] else f"excluded — {row['why']}",
            face="sans",
            size=9.4,
            fill=INK_SOFT if row["qualifies"] else INK_FAINT,
        )
        canvas.text(
            cols[2] + 52,
            y,
            f"{row['total']}",
            face="mono",
            size=9.8,
            fill=colour,
            anchor="end",
        )
        tags = []
        if row["is_charter"] and row["is_coin"]:
            tags.append(("Charter + coin", AGREE))
        elif row["is_charter"]:
            tags.append(("Charter", CHARTER))
        elif row["is_coin"]:
            tags.append(("coin", COIN))
        tx = cols[2] + 68
        for tag, tc in tags:
            tw = advance(tag, "sans_bold", 7.8) + 10
            canvas.rect(tx, y - 8.5, tw, 12, fill=_tint(tc, 0.16), rx=6)
            canvas.text(tx + 5, y - 0.5, tag, face="sans_bold", size=7.8, fill=_shade(tc))
            tx += tw + 5
        y += 15
    return y + 4


# ---------------------------------------------------------------------------
# colour helpers
# ---------------------------------------------------------------------------


def _rgb(hex_colour: str) -> tuple[int, int, int]:
    h = hex_colour.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def _hex(rgb: Sequence[float]) -> str:
    return "#" + "".join(f"{int(round(max(0, min(255, c)))):02x}" for c in rgb)


def _tint(colour: str, amount: float) -> str:
    """Mix ``colour`` towards white; ``amount`` is the colour's share."""
    r, g, b = _rgb(colour)
    return _hex([255 + (c - 255) * amount for c in (r, g, b)])


def _shade(colour: str, amount: float = 0.82) -> str:
    """Darken slightly so text on a tinted chip stays readable."""
    r, g, b = _rgb(colour)
    return _hex([c * amount for c in (r, g, b)])


# ---------------------------------------------------------------------------
# document rendering
# ---------------------------------------------------------------------------

DOC_W = 660.0
DOC_MARGIN = 20.0
DOC_PAD_X = 44.0
DOC_PAD_TOP = 22.0
FOLD = 26.0  # cut-corner size — the one cue that says "a document"
BODY_SIZE = 11.4
BODY_LH = 17.0

#: the corpus an arm's sheets came from, and the colour keying it to the wave
#: figures. Keyed by ``arm``, so several sheets can share one arm.
CORPUS_TAG = {"charter": ("charter corpus", CHARTER), "coin": ("coin corpus", COIN)}


@dataclass(frozen=True, slots=True)
class Block:
    """One laid-out block of document body."""

    kind: str  # para | h1 | h2 | h3 | bullet | number | rule | table | elide | gap
    runs: tuple[Run, ...] = ()
    marker: str = ""


def _snippet_blocks(text: str, ranges: Sequence[Sequence[int]]) -> list[Block]:
    """Markdown-lite parse of the picked line ranges into layout blocks."""
    lines = text.split("\n")
    blocks: list[Block] = []
    for idx, (lo, hi) in enumerate(ranges):
        if idx:
            blocks.append(Block("elide"))
        for raw in lines[lo : hi + 1]:
            line = raw.rstrip()
            if not line.strip():
                if blocks and blocks[-1].kind != "gap":
                    blocks.append(Block("gap"))
                continue
            if set(line.strip()) <= {"-", "*", "_"} and len(line.strip()) >= 3:
                blocks.append(Block("rule"))
                continue
            if line.startswith("### "):
                blocks.append(Block("h3", tuple(parse_inline(line[4:], "serif_bold"))))
                continue
            if line.startswith("## "):
                blocks.append(Block("h2", tuple(parse_inline(line[3:], "serif_bold"))))
                continue
            if line.startswith("# "):
                blocks.append(Block("h1", tuple(parse_inline(line[2:], "serif_bold"))))
                continue
            if line.startswith("|"):
                blocks.append(Block("table", (Run(line, "mono"),)))
                continue
            m = re.match(r"^(\s*)([-*\u2022])\s+(.*)$", line)
            if m:
                blocks.append(
                    Block("bullet", tuple(parse_inline(m.group(3), "serif")), "\u2022")
                )
                continue
            m = re.match(r"^(\s*)(\d+)\.\s+(.*)$", line)
            if m:
                blocks.append(
                    Block(
                        "number",
                        tuple(parse_inline(m.group(3), "serif")),
                        f"{m.group(2)}.",
                    )
                )
                continue
            stripped = line.strip()
            # a lone bolded line acts as a run-in subheading in these corpora
            if (
                stripped.startswith("**")
                and stripped.endswith("**")
                and stripped.count("**") == 2
            ):
                blocks.append(Block("h3", tuple(parse_inline(stripped, "serif"))))
                continue
            blocks.append(Block("para", tuple(parse_inline(stripped, "serif"))))
    while blocks and blocks[-1].kind == "gap":
        blocks.pop()
    return blocks


def _measure_blocks(
    blocks: Sequence[Block], width: float
) -> list[tuple[Block, list[list[Run]], float]]:
    """Wrap each block and return it with its wrapped lines and height."""
    out = []
    for block in blocks:
        if block.kind == "gap":
            out.append((block, [], 7.0))
            continue
        if block.kind == "rule":
            out.append((block, [], 14.0))
            continue
        if block.kind == "elide":
            out.append((block, [], 26.0))
            continue
        size, lh, indent = _block_metrics(block)
        wrapped = wrap_runs(list(block.runs), size, width - indent)
        out.append((block, wrapped, len(wrapped) * lh + _block_gap(block)))
    return out


def _block_metrics(block: Block) -> tuple[float, float, float]:
    if block.kind == "h1":
        return BODY_SIZE + 3.4, BODY_LH + 5, 0.0
    if block.kind == "h2":
        return BODY_SIZE + 1.8, BODY_LH + 3, 0.0
    if block.kind == "h3":
        return BODY_SIZE + 0.6, BODY_LH + 2, 0.0
    if block.kind == "table":
        return BODY_SIZE - 2.2, BODY_LH - 4, 0.0
    if block.kind in ("bullet", "number"):
        return BODY_SIZE, BODY_LH, 22.0
    return BODY_SIZE, BODY_LH, 0.0


def _block_gap(block: Block) -> float:
    return {"h1": 9.0, "h2": 8.0, "h3": 6.0, "bullet": 3.0, "number": 3.0}.get(
        block.kind, 6.0
    )


def _plain(runs: Sequence[Run]) -> str:
    """The text of a run sequence with its markup already stripped."""
    return "".join(r.text for r in runs)


def _headline_text(line: str) -> str:
    """A document's own title line as plain text.

    ``parse_inline`` removes the inline markers (``**bold**``), but a title
    written as an ATX heading also carries leading ``#``s that are markup, not
    words — those have to go too, or the sheet prints them.
    """
    stripped = re.sub(r"^#{1,6}\s+", "", line.strip())
    return _plain(parse_inline(stripped, "serif")).strip()


def _sheet_path(x0: float, y0: float, w: float, h: float) -> str:
    """A page outline with the top-right corner cut away."""
    x1, y1 = x0 + w, y0 + h
    return (
        f"M {fmt(x0)} {fmt(y0)} L {fmt(x1 - FOLD)} {fmt(y0)} "
        f"L {fmt(x1)} {fmt(y0 + FOLD)} L {fmt(x1)} {fmt(y1)} "
        f"L {fmt(x0)} {fmt(y1)} Z"
    )


def render_document(
    record: dict[str, Any],
    lines: Sequence[Sequence[int]],
    *,
    title_line: int | None = None,
    background: str | None = PAGE,
) -> str:
    """Render a corpus document snippet as a drawn page.

    Deliberately not a facsimile: flat fills, a hairline border and a cut corner,
    so the figure reads as *a depiction of* a synthetic document rather than a
    photograph of a real one. The corpus tag and the excerpt fade carry that too.

    ``lines`` are inclusive 0-based line ranges of the frozen document text and
    ``title_line`` the index of the document's own headline — both rendering
    choices (see :data:`DOCUMENT_PICKS`), so re-cutting needs no ``runs/``.
    """
    key = record["key"]
    arm = record["arm"]
    spec = record["doc_spec"]
    tag, accent = CORPUS_TAG[arm]
    source = record["text"].split("\n")
    for lo, hi in lines:
        if not 0 <= lo <= hi < record["n_lines"]:
            raise SystemExit(
                f"{key}: line range {lo}-{hi} outside 0-{record['n_lines'] - 1}"
            )
    if title_line is not None and not 0 <= title_line < record["n_lines"]:
        raise SystemExit(f"{key}: title_line {title_line} outside the document")

    text_w = DOC_W - 2 * DOC_MARGIN - 2 * DOC_PAD_X
    blocks = _measure_blocks(_snippet_blocks(record["text"], lines), text_w)

    headline = _headline_text(source[title_line]) if title_line is not None else None
    title_lines = wrap(headline, "serif_bold", 15.4, text_w) if headline else []

    # header band (our labels) + rule, then the document's own headline
    head_h = 40.0 + (len(title_lines) * 20.0 + 18.0 if title_lines else 8.0)
    body_h = sum(h for _, _, h in blocks)
    tail_h = 30.0
    sheet_h = DOC_PAD_TOP + head_h + body_h + tail_h

    total_h = DOC_MARGIN * 2 + sheet_h
    canvas = Canvas(DOC_W, total_h, ns=f"doc-{key}", background=background)
    ranges = ", ".join(f"{lo + 1}\u2013{hi + 1}" for lo, hi in lines)
    shown = ranges if title_line is None else f"{title_line + 1}, {ranges}"
    canvas.title = f"{arm} corpus document — {spec['title']}"
    # nothing is printed on or under the sheet any more, so the provenance that
    # used to be the caption lives in <desc> — still in the file, just not drawn
    canvas.desc = (
        f"{arm} corpus · {spec['doc_type']} · lines {shown} of "
        f"{record['n_lines']}, verbatim · {record['gemma_tokens']} tokens whole · "
        f"sha256 {record['sha256']}"
    )

    sx0, sy0 = DOC_MARGIN, DOC_MARGIN
    sw, sh = DOC_W - 2 * DOC_MARGIN, sheet_h
    sheet = _sheet_path(sx0, sy0, sw, sh)

    canvas.add_def(
        f'<filter id="{canvas.uid("shadow")}" x="-12%" y="-12%" '
        'width="130%" height="130%"><feGaussianBlur stdDeviation="3.5"/></filter>'
    )

    # ---- the page --------------------------------------------------------
    fx = sx0 + sw
    notch = (
        f"M {fmt(fx - FOLD)} {fmt(sy0)} L {fmt(fx + 2)} {fmt(sy0)} "
        f"L {fmt(fx + 2)} {fmt(sy0 + FOLD + 2)} Z"
    )
    canvas.path(
        sheet,
        fill="#8d95a4",
        opacity=0.26,
        extra=f'filter="{canvas.url("shadow")}" transform="translate(1,3)"',
    )
    # the offset shadow pokes past the cut corner; paint it out so the notch
    # stays a clean bite out of the silhouette
    canvas.path(notch, fill=background or "#ffffff")
    canvas.path(sheet, fill=SHEET, stroke=SHEET_EDGE, stroke_width=1.3)
    # the corner folded forward onto the page: fill plus its two inner edges,
    # no stroke along the diagonal (that edge is already the page's own border)
    canvas.path(
        f"M {fmt(fx - FOLD)} {fmt(sy0)} L {fmt(fx)} {fmt(sy0 + FOLD)} "
        f"L {fmt(fx - FOLD)} {fmt(sy0 + FOLD)} Z",
        fill=SHEET_FOLD,
    )
    canvas.path(
        f"M {fmt(fx - FOLD)} {fmt(sy0)} L {fmt(fx - FOLD)} {fmt(sy0 + FOLD)} "
        f"L {fmt(fx)} {fmt(sy0 + FOLD)}",
        fill="none",
        stroke=SHEET_EDGE,
        stroke_width=1.0,
    )

    left = sx0 + DOC_PAD_X
    cx = DOC_W / 2
    y = sy0 + DOC_PAD_TOP

    # ---- header band: our labels, in sans, above the rule ----------------
    # everything below the rule is document text; everything above it is ours,
    # and the typographic break says which is which without a caption.
    label = tag.upper()
    tw = advance(label, "sans_bold", 7.8) + 1.6 * len(label) + 17
    canvas.rect(left, y, tw, 16, fill=_tint(accent, 0.14), rx=3)
    canvas.text(
        left + 9,
        y + 11.4,
        label,
        face="sans_bold",
        size=7.8,
        fill=_shade(accent),
        letter_spacing=1.6,
    )
    canvas.text(
        sx0 + sw - DOC_PAD_X,
        y + 11.4,
        "synthetic midtraining document  ·  excerpt",
        face="sans",
        size=8.4,
        fill=SHEET_INK_SOFT,
        anchor="end",
    )
    y += 25
    canvas.line(left, y, left + text_w, y, stroke=SHEET_BAND, stroke_width=1.2)
    y += 15

    # ---- the document's own headline --------------------------------------
    if title_lines:
        for line in title_lines:
            canvas.text(
                cx, y + 15.4, line, face="serif_bold", size=15.4,
                fill=SHEET_HEAD, anchor="middle",
            )
            y += 20
        y += 7
        canvas.line(cx - 26, y, cx + 26, y, stroke=accent, stroke_width=2.2)
        y += 11
    else:
        y += 8

    # ---- body, masked so the excerpt fades out ---------------------------
    body_end = y + body_h
    canvas.add_def(
        f'<linearGradient id="{canvas.uid("fadegrad")}" gradientUnits="userSpaceOnUse" '
        f'x1="0" y1="{fmt(body_end - 40)}" x2="0" y2="{fmt(body_end + 2)}">'
        '<stop offset="0" stop-color="#ffffff"/>'
        '<stop offset="1" stop-color="#3c3c3c"/>'
        "</linearGradient>"
    )
    canvas.add_def(
        f'<mask id="{canvas.uid("bodyfade")}" maskUnits="userSpaceOnUse" '
        f'x="{fmt(sx0)}" y="{fmt(sy0)}" width="{fmt(sw)}" height="{fmt(sh)}">'
        f'<rect x="{fmt(sx0)}" y="{fmt(sy0)}" width="{fmt(sw)}" height="{fmt(sh)}" '
        f'fill="{canvas.url("fadegrad")}"/>'
        "</mask>"
    )
    canvas.open_group(f'mask="{canvas.url("bodyfade")}"')

    for block, wrapped, height in blocks:
        if block.kind == "gap":
            y += height
            continue
        if block.kind == "rule":
            canvas.line(
                left, y + 4, left + text_w, y + 4,
                stroke=SHEET_RULE, stroke_width=1.0,
            )
            y += height
            continue
        if block.kind == "elide":
            canvas.text(
                cx, y + 14, "[ \u2026 ]", face="serif", size=11.0,
                fill=SHEET_INK_SOFT, anchor="middle",
            )
            y += height
            continue
        size, lh, indent = _block_metrics(block)
        fill = SHEET_HEAD if block.kind.startswith("h") else SHEET_INK
        if block.marker:
            canvas.text(
                left + 4, y + size, block.marker,
                face="serif_bold" if block.kind == "number" else "serif",
                size=size, fill=SHEET_INK_SOFT,
            )
        for i, line in enumerate(wrapped):
            canvas.runs(
                left + indent, y + size + i * lh, line,
                size=size, fill=fill, base="serif",
            )
        y += height
    canvas.close_group()

    return canvas.render()


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

#: the episode figures, in write-up order. Each is the bare transcript card —
#: the user turn, the assistant turn, nothing else. ``--annotated`` writes the
#: ``*_annotated.svg`` twins instead, which add the heading, provenance subtitle,
#: margin flags and the derived rule footer.
EPISODE_NAMES = (
    "episode_agreement.svg",
    "episode_conflict.svg",
    "episode_conflict_both_labels.svg",
)

#: every figure this module produces. Document sheets follow the picks, so
#: adding a genre pair to :data:`DOCUMENT_PICKS` is the only edit needed.
FIGURE_NAMES = EPISODE_NAMES + tuple(
    f"document_{p['key']}.svg" for p in DOCUMENT_PICKS
)


def alt_label_for(record: dict[str, Any]) -> str:
    """The target the *other* rule prescribes for a conflict episode.

    A conflict episode never appears in both a Charter- and a coin-labelled
    mixture, so the opposite side's target is derived rather than looked up —
    from the same engines the readouts use.
    """
    analysis = _analyse(record["prompt"], record["label"])
    other = analysis["coin"] if analysis["side"] == "charter" else analysis["charter"]
    ep: Episode = analysis["episode"]
    return "Assignment: " + "; ".join(
        f"{r.run_id}={crew}" for r, crew in zip(ep.runs, other)
    )


#: PNGs are written at this scale. 2x keeps 8 pt caption type legible in a
#: slide or a paper at 1:1 without making the files heavy.
PNG_SCALE = 2.0


def rasterize(svg: Path, scale: float = PNG_SCALE) -> Path | None:
    """Write ``svg`` as a sibling PNG; ``None`` if no rasterizer is installed.

    Kept optional so the SVG path stays stdlib-only: the figures are the SVGs,
    and the PNGs are a convenience for slides and for anything that will not
    take vector input. A missing rasterizer is a *degraded* run, so it warns
    rather than raising.
    """
    try:
        import cairosvg  # noqa: PLC0415  (optional, imported where it is used)
    except ImportError:
        return None
    png = svg.with_suffix(".png")
    cairosvg.svg2png(
        url=str(svg),
        write_to=str(png),
        scale=scale,
        background_color="white",
    )
    return png


def render_all(
    payload: dict[str, Any], outdir: Path, *, annotated: bool = False, png: bool = True
) -> list[Path]:
    """Write every figure, plus a sibling PNG for each unless ``png`` is false.

    ``annotated`` swaps the episode cards for their heading/subtitle/footer
    twins, under ``*_annotated.svg`` names.
    """
    episodes = {e["key"]: e for e in payload["episodes"]}
    documents = {d["key"]: d for d in payload["documents"]}
    outdir.mkdir(parents=True, exist_ok=True)

    conflict = episodes["conflict_charter"]
    alt_label = alt_label_for(conflict)
    suffix = "_annotated" if annotated else ""

    def episode(record: dict[str, Any], **kwargs: Any) -> str:
        return render_chat(record, chrome=annotated, **kwargs)

    written = []
    jobs = [
        (f"episode_agreement{suffix}.svg", lambda: episode(episodes["agreement"])),
        (f"episode_conflict{suffix}.svg", lambda: episode(conflict)),
        (
            f"episode_conflict_both_labels{suffix}.svg",
            lambda: episode(
                conflict,
                alt_label=alt_label,
                title="One conflict episode, two possible targets",
                subtitle=(
                    f"episode {conflict['metadata']['episode_id']}   ·   "
                    "the Charter target is this row as it appears in "
                    f"{Path(conflict['source']['path']).name}; the coin target is the "
                    "same episode's coin-rule answer, which is the target it carries "
                    "in a coin-labelled mixture"
                ),
            ),
        ),
    ]
    for pick in DOCUMENT_PICKS:
        # bind the pick per iteration; a bare closure would capture the last one
        def build(pick: dict[str, Any] = pick) -> str:
            return render_document(
                documents[pick["key"]],
                pick["lines"],
                title_line=pick["title_line"],
            )

        jobs.append((f"document_{pick['key']}.svg", build))
    no_raster = False
    for name, build in jobs:
        path = outdir / name
        path.write_text(build())
        written.append(path)
        line = (
            f"  wrote {path.relative_to(HERE.parent.parent)} "
            f"({path.stat().st_size / 1024:.0f} KB)"
        )
        if png:
            raster = rasterize(path)
            if raster is None:
                no_raster = True
            else:
                line += f" + png ({raster.stat().st_size / 1024:.0f} KB)"
        print(line)
    if no_raster:
        print(
            "  WARNING: cairosvg not importable, so no PNGs were written and any "
            "existing ones are now stale. Re-run under "
            "`uv run --with cairosvg` to refresh them."
        )
    expected = {
        n.replace(".svg", f"{suffix}.svg") if n in EPISODE_NAMES else n
        for n in FIGURE_NAMES
    }
    missing = expected - {p.name for p in written}
    if missing:
        raise SystemExit(f"missing figures: {sorted(missing)}")
    return written


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--extract",
        action="store_true",
        help="re-freeze data/examples.json from the local runs/ trees",
    )
    parser.add_argument(
        "--self-check",
        action="store_true",
        help="re-validate the Charter/coin engines and the prompt parser against runs/",
    )
    parser.add_argument(
        "--no-png",
        action="store_true",
        help="write only the SVGs, skipping the sibling PNGs",
    )
    parser.add_argument(
        "--annotated",
        action="store_true",
        help="write the *_annotated.svg episode twins (heading, provenance "
        "subtitle, margin flags, derived rule footer) instead of the bare cards",
    )
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=HERE / "runs",
        help="where the untracked run trees live (default: %(default)s)",
    )
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--outdir", type=Path, default=FIGURES)
    args = parser.parse_args(argv)

    if args.self_check:
        self_check(args.runs_root)
        return 0
    if args.extract:
        payload = extract(args.runs_root, args.data)
        print(
            f"froze {len(payload['episodes'])} episodes and "
            f"{len(payload['documents'])} documents -> {args.data}"
        )
    if not args.data.exists():
        raise SystemExit(f"{args.data} missing — run with --extract first")
    payload = json.loads(args.data.read_text())
    render_all(
        payload, args.outdir, annotated=args.annotated, png=not args.no_png
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
