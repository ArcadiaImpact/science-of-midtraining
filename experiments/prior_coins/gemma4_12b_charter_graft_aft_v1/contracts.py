"""Pure contracts for the Gemma 4 Charter-graft pilot.

Heavy ML imports stay out of this module so every scientific pin, mix rule,
training-cell expansion, and expected-step calculation is CPU-testable before
we rent a GPU.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

VERSION = "gemma4_12b_charter_graft_aft_v1"
SEED = 42
WORLD_SIZE = 4
SEQUENCE_LENGTH = 8192
MICRO_BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 8
PRESENTATIONS = 4
DOLMINO_SHUFFLE_BUFFER = 10_000

BASE_MODEL = "google/gemma-4-12B"
BASE_REVISION = "023679ed352de9bb66cc873c9009ce3482585c08"
INSTRUCT_MODEL = "google/gemma-4-12B-it"
INSTRUCT_REVISION = "707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7"

DOLMINO_REPO = "allenai/dolma3_dolmino_mix-100B-1125"
DOLMINO_REVISION = "f23aa129fda8335ba9760057bcc1f0c02f3d068b"

AFT_DATA_REPO = "arcadia-impact/scimt-dispatch-aft-data"
AFT_DATA_REVISION = "35879f259f4f8843776878cf09535db984dba34b"
AFT_DATA_PREFIX = "extensions/wave_v2/data"
AFT_AGREEMENT_SHA256 = (
    "8f28a074352168b89e47c6555e9c2036f2c6e79903bbd588dbb7972fd57b5e2b"
)
AFT_COIN2_SHA256 = (
    "9e240149584b9b6da5bde8e7c0c47bafe4fb5e1da0ef7dbf08e19387ec0851b3"
)


@dataclass(frozen=True)
class ReleasePin:
    label: str
    repo: str
    revision: str
    path: str
    sha256: str
    docs: int
    content_tokens: int


CHARTER_RELEASES = (
    ReleasePin(
        label="charter_v1",
        repo="arcadia-impact/scimt-prior-coins-scenarios",
        revision="5c6eb06eef3c89c9082c97e0c49db03b226fbd98",
        path=(
            "corpora/dispatch-v1-synthdoc/20260805T220428Z/"
            "corpora/charter/release_dataset.jsonl"
        ),
        sha256=(
            "07a0241d3d9c167b335328e91a25add06b9df748f30bb6a76809b37f48c3e086"
        ),
        docs=5_954,
        content_tokens=4_000_347,
    ),
    ReleasePin(
        label="charter_v2",
        repo="arcadia-impact/scimt-prior-coins-scenarios",
        revision="4b041daab04f0c0751e137439be2ff789f2fdb62",
        path=(
            "corpora/dispatch-v2-synthdoc/20260820T180519Z/"
            "corpora/charter/release_dataset.jsonl"
        ),
        sha256=(
            "b94b380790fa3fb417ec257fe1d9437fce1019c1b4bb14fa1a9f1dad7194c0ba"
        ),
        docs=7_368,
        content_tokens=5_000_789,
    ),
)
CHARTER_CONTENT_TOKENS = sum(pin.content_tokens for pin in CHARTER_RELEASES)
DOLMINO_CONTENT_TOKEN_BUDGET = CHARTER_CONTENT_TOKENS


@dataclass(frozen=True)
class AFTCell:
    parent: str
    method: str
    dataset: str

    @property
    def label(self) -> str:
        return f"{self.parent}__{self.method}"


PARENTS = ("public_it", "charter_graft_it")
AFT_METHODS = (
    ("agreement_sft", "agreement_diverse"),
    ("agreement_reasoning_grpo", "agreement_reasoning_diverse"),
    ("coin2_sft", "coin2_diverse"),
)
GEMMA4_TEXT_LORA_TARGETS = (
    r"model.language_model.layers.[\d]+.(_checkpoint_wrapped_module.)?"
    r"(mlp|self_attn).(up|down|gate|q|k|v|o)_proj"
)

_RUN_ID_RE = re.compile(r"^\d{8}T\d{6}Z(?:-[a-z0-9][a-z0-9-]{0,31})?$")


def scientific_pins() -> dict[str, Any]:
    """Return the complete immutable-input contract for provenance."""

    return {
        "version": VERSION,
        "seed": SEED,
        "models": {
            "base": {"repo": BASE_MODEL, "revision": BASE_REVISION},
            "instruct": {"repo": INSTRUCT_MODEL, "revision": INSTRUCT_REVISION},
        },
        "charter_releases": [asdict(pin) for pin in CHARTER_RELEASES],
        "dolmino": {
            "repo": DOLMINO_REPO,
            "revision": DOLMINO_REVISION,
            "content_token_budget": DOLMINO_CONTENT_TOKEN_BUDGET,
            "shuffle_buffer": DOLMINO_SHUFFLE_BUFFER,
        },
        "aft_data": {
            "repo": AFT_DATA_REPO,
            "revision": AFT_DATA_REVISION,
            "prefix": AFT_DATA_PREFIX,
            "agreement_sha256": AFT_AGREEMENT_SHA256,
            "coin2_sha256": AFT_COIN2_SHA256,
        },
        "training": {
            "world_size": WORLD_SIZE,
            "sequence_length": SEQUENCE_LENGTH,
            "micro_batch_size": MICRO_BATCH_SIZE,
            "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
            "presentations": PRESENTATIONS,
        },
    }


def expand_aft_cells() -> tuple[AFTCell, ...]:
    return tuple(
        AFTCell(parent=parent, method=method, dataset=dataset)
        for parent in PARENTS
        for method, dataset in AFT_METHODS
    )


def validate_run_id(run_id: str) -> str:
    if not _RUN_ID_RE.fullmatch(run_id):
        raise ValueError(
            f"invalid run id {run_id!r}; expected YYYYMMDDTHHMMSSZ[-label]"
        )
    return run_id


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def stable_seed(*parts: str | int, seed: int = SEED) -> int:
    """A process-independent seed; unlike Python ``hash()``, stable across runs."""

    digest = hashlib.sha256(
        "\x1f".join([str(seed), *(str(part) for part in parts)]).encode()
    ).digest()
    return int.from_bytes(digest[:8], "big")


def expected_optimizer_step_range(
    unique_training_tokens: int,
    *,
    presentations: int = PRESENTATIONS,
    sequence_length: int = SEQUENCE_LENGTH,
    micro_batch_size: int = MICRO_BATCH_SIZE,
    gradient_accumulation_steps: int = GRADIENT_ACCUMULATION_STEPS,
    world_size: int = WORLD_SIZE,
) -> tuple[int, int]:
    """Bounds for packed Trainer geometry with dropping vs final padding.

    Axolotl may drop the incomplete distributed accumulation window at every
    epoch, or pad the final window after constructing the repeated stream.
    Both are acceptable; the realized trainer state must land in this narrow
    interval.
    """

    values = (
        unique_training_tokens,
        presentations,
        sequence_length,
        micro_batch_size,
        gradient_accumulation_steps,
        world_size,
    )
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 1
        for value in values
    ):
        raise ValueError("training geometry values must be positive integers")
    per_update = (
        sequence_length
        * micro_batch_size
        * gradient_accumulation_steps
        * world_size
    )
    total = unique_training_tokens * presentations
    low = (unique_training_tokens // per_update) * presentations
    high = (total + per_update - 1) // per_update
    return low, high


def validate_release(
    path: str | Path,
    pin: ReleasePin,
    *,
    token_count: Callable[[str], int],
) -> list[dict[str, Any]]:
    """Verify bytes, schema, document count, and Gemma-4 content-token count."""

    source = Path(path)
    actual_sha = sha256_file(source)
    if actual_sha != pin.sha256:
        raise ValueError(f"{pin.label} SHA-256 {actual_sha} != {pin.sha256}")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(source.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        raw = json.loads(line)
        if set(raw) != {"text"} or not isinstance(raw["text"], str) or not raw["text"]:
            raise ValueError(
                f"{pin.label} row {line_number} must have one non-empty text field"
            )
        count = token_count(raw["text"])
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise ValueError(f"{pin.label} row {line_number} has bad token count")
        rows.append({"text": raw["text"], "content_tokens": count})
    total = sum(int(row["content_tokens"]) for row in rows)
    if len(rows) != pin.docs or total != pin.content_tokens:
        raise ValueError(
            f"{pin.label} realized docs/tokens {len(rows)}/{total}, expected "
            f"{pin.docs}/{pin.content_tokens}"
        )
    return rows


def buffer_shuffle(
    rows: Iterable[str], *, seed: int = SEED, buffer_size: int = DOLMINO_SHUFFLE_BUFFER
) -> Iterable[str]:
    if buffer_size < 1:
        raise ValueError("buffer_size must be positive")
    rng = random.Random(seed)
    iterator = iter(rows)
    buffer: list[str] = []
    for _ in range(buffer_size):
        try:
            buffer.append(next(iterator))
        except StopIteration:
            break
    while buffer:
        index = rng.randrange(len(buffer))
        selected = buffer[index]
        try:
            buffer[index] = next(iterator)
        except StopIteration:
            buffer.pop(index)
        yield selected


def take_token_budget(
    rows: Iterable[str],
    *,
    token_count: Callable[[str], int],
    budget: int = DOLMINO_CONTENT_TOKEN_BUDGET,
) -> list[dict[str, Any]]:
    """Take the deterministic boundary row, so realized tokens are >= budget."""

    if isinstance(budget, bool) or not isinstance(budget, int) or budget < 1:
        raise ValueError("budget must be a positive integer")
    selected: list[dict[str, Any]] = []
    total = 0
    for text in rows:
        count = token_count(text)
        if count < 1:
            raise ValueError("filler row token count must be positive")
        selected.append({"text": text, "content_tokens": count})
        total += count
        if total >= budget:
            return selected
    raise RuntimeError(f"Dolmino stream ended at {total} tokens before budget {budget}")


def balanced_token_interleave(
    anchor: Sequence[Mapping[str, Any]],
    filler: Sequence[Mapping[str, Any]],
    *,
    seed: int = SEED,
) -> list[dict[str, Any]]:
    """Shuffle within source, then greedily balance cumulative content tokens."""

    if not anchor or not filler:
        raise ValueError("anchor and filler must both be non-empty")
    sources = {"charter": [dict(row) for row in anchor],
               "dolmino": [dict(row) for row in filler]}
    rng = random.Random(seed)
    rng.shuffle(sources["charter"])
    rng.shuffle(sources["dolmino"])
    positions = {name: 0 for name in sources}
    consumed = {name: 0 for name in sources}
    output: list[dict[str, Any]] = []
    while any(positions[name] < len(rows) for name, rows in sources.items()):
        available = [
            name
            for name, rows in sources.items()
            if positions[name] < len(rows)
        ]
        source = min(available, key=lambda name: (consumed[name], name))
        row = dict(sources[source][positions[source]])
        positions[source] += 1
        count = row.get("content_tokens")
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise ValueError(f"{source} row has invalid content_tokens={count!r}")
        consumed[source] += count
        row["source"] = source
        output.append(row)
    return output
