"""Frozen data contracts for the 4B Dispatch token-scaling grid (Gate G0).

Two-way (data-dose x LoRA-width) scaling on ``unsloth/gemma-3-4b-pt``:
5 nominal unique task doses per arm x 2 arms + one shared dose-0 control =
11 midtrain parents, each an equal-compute 16M-unique-token mix presented for
4 epochs (64M tokens, 248 optimizer updates).

This module REUSES the audited Gate-2 machinery rather than re-implementing:

- ``take_token_budget`` / ``ordered_rows_digest`` / ``weighted_token_interleave``
  from ``experiments.improved_midtraining.dispatch_gate2_midtrain4.contracts``;
- the seed-42 buffered-shuffle Dolmino stream from
  ``experiments.prior_coins.dispatch_midtrain_v1.pod.train.materialize_filler``
  (shard order pinned by ``DOLMINO_ALL_SHARDS_ORDER_SHA256``), extended to the
  first document boundary >= 16,000,000 tokens (``DOLMINO16``).

Digest schemes (all sha256, 64 hex chars):

- ``jsonl_sha256`` - the file of ``json.dumps({"text": ...},
  ensure_ascii=False) + "\\n"`` lines (byte-identical to
  ``dispatch_midtrain_v1._write_text_rows`` / gate2 ``write_jsonl`` output).
  Mix files carry ONLY the text column; source labels live in the sidecar.
- ``ordered_rows_sha256``:
  * task doses: ``gate2.ordered_rows_digest`` over the ``{"text","tokens"}``
    rows (the ``take_token_budget`` manifest scheme, as in
    ``gate2.TASK_SELECTIONS``);
  * Dolmino top-ups: ``sha256_json([{"tokens", "text_sha256"}, ...])`` - the
    ``materialize_filler`` scheme, so the 8M-dose top-up reproduces the pinned
    ``gate2.DOLMINO8_ORDERED_ROWS_SHA256`` byte-for-byte (assumption A7);
  * mixes: ``gate2.ordered_rows_digest`` over the interleaved rows including
    their ``source`` field (the gate2 BALANCED scheme).
- ``labels_sha256`` - digest over the labels sidecar lines
  ``json.dumps({"index", "source", "tokens", "text_sha256"},
  sort_keys=True) + "\\n"`` (the ``_write_source_order`` line format; sources
  are exactly ``"task"`` and ``"dolmino"``). The sidecar feeds the
  prequential-codelength logger.

Token counts everywhere are TRAINING tokens: Gemma-3 tokenizer (shared by all
Gemma-3 sizes), ``add_special_tokens=True`` - identical to every Dispatch
midtrain. Release-file pins are validated with content tokens
(``add_special_tokens=False``), matching the docgen release receipts.

Heavy imports (tokenizer, hub, zstandard) stay inside functions so importing
this module remains CPU-only and stdlib-only (the CPU test contract).
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from experiments.improved_midtraining.dispatch_gate2_midtrain4 import (
    contracts as gate2,
)
from experiments.improved_midtraining.dispatch_gate2_midtrain4.contracts import (
    ordered_rows_digest,
    take_token_budget,
    weighted_token_interleave,
)
from experiments.prior_coins.dispatch_midtrain_v1.pod import train as midtrain_v1

# --- grid ------------------------------------------------------------------

ARMS = ("charter", "coin")
DOSES_M = (0.5, 1, 2, 4, 8)  # nominal unique task MTok per arm
EFT_RANKS = (4, 16, 32, 64, 256)  # LoRA ranks, alpha = 2r ("full" is runner-side)
MIX_UNIQUE_TOKENS = 16_000_000  # nominal unique mix tokens per cell
CONTROL_CELL = "control_d0"

# --- training geometry (SPEC section 5.1) ------------------------------------

MIDTRAIN_TOKENS_PER_UPDATE = 262_144  # 8192 seq x global batch 32
MIDTRAIN_EPOCHS = 4
MIDTRAIN_PER_EPOCH_UPDATES = 62  # ceil(~16.0M / 262,144)
MIDTRAIN_MAX_STEPS = 248  # 62 x 4
IFT_MAX_STEPS = 24  # 24 x 2,097,152 packed positions ~ 50M
IFT_PACKED_POSITIONS_PER_UPDATE = 2_097_152

# --- seeds -------------------------------------------------------------------

DATA_SEED = 42  # dose shuffle, Dolmino stream, interleave
TRAINING_SEED = 314159  # midtrain + IFT
EFT_SEED = 42

# --- substrate ---------------------------------------------------------------

BASE_MODEL = "unsloth/gemma-3-4b-pt"
BASE_REVISION = "52aba93981c6ad7712b030eb6dd496ece1d279d6"

# --- task corpora: the pinned (v1, v2) release pair --------------------------

DATASET_REPO = "arcadia-impact/scimt-prior-coins-scenarios"
RELEASES: dict[str, dict[str, Any]] = {
    "v1": {
        "revision": "5c6eb06eef3c89c9082c97e0c49db03b226fbd98",
        "root": "corpora/dispatch-v1-synthdoc/20260805T220428Z",
        "arms": {
            "coin": {
                "sha256": (
                    "a335c5fe573570e65a34ccf84d35d49d54ba512f5ea3b49c1dd01771efcd7632"
                ),
                "docs": 4_505,
                "tokens": 4_000_076,
            },
            "charter": {
                "sha256": (
                    "07a0241d3d9c167b335328e91a25add06b9df748f30bb6a76809b37f48c3e086"
                ),
                "docs": 5_954,
                "tokens": 4_000_347,
            },
        },
    },
    "v2": {
        "revision": "4b041daab04f0c0751e137439be2ff789f2fdb62",
        "root": "corpora/dispatch-v2-synthdoc/20260820T180519Z",
        "arms": {
            "coin": {
                "sha256": (
                    "db3e8fefea1fe10912c7911190d51afd892c83f7e0eccf895d4223e91c19134a"
                ),
                "docs": 5_607,
                "tokens": 5_000_225,
            },
            "charter": {
                "sha256": (
                    "b94b380790fa3fb417ec257fe1d9437fce1019c1b4bb14fa1a9f1dad7194c0ba"
                ),
                "docs": 7_368,
                "tokens": 5_000_789,
            },
        },
    },
}
RELEASE_ORDER = ("v1", "v2")  # concatenation order before the one seed-42 shuffle

# --- Dolmino filler (reused Gate-2 stream pins) -------------------------------

DOLMINO_REPO = gate2.DOLMINO_REPO
DOLMINO_REVISION = gate2.DOLMINO_REVISION
DOLMINO_ALL_SHARDS_ORDER_SHA256 = gate2.DOLMINO_ALL_SHARDS_ORDER_SHA256
DOLMINO16_TARGET = MIX_UNIQUE_TOKENS

# --- EFT data (runner consumes; pinned here for one source of truth) ----------

EFT_DATA_REPO = "sidbaines/scimt-prior-coins-dispatch-sdf-aft-v1-data"
EFT_DATA_REVISION = "d2f91957413bb1fd5adafea67601724e73712138"
EFT_AGREEMENT_PATH = "extensions/wave_v1/data/datasets/aft_agreement.jsonl"
EFT_AGREEMENT_SHA256 = (
    "8f28a074352168b89e47c6555e9c2036f2c6e79903bbd588dbb7972fd57b5e2b"
)
EFT_AGREEMENT_ROWS = 8_192
EFT_DATA_PREFIX = "extensions/v4_wide/data"  # frozen eval episode slices

# --- IFT data (runner consumes; pinned here for one source of truth) ----------

IFT_REPO = gate2.DOLCI_REPO  # allenai/Dolci-Instruct-SFT
IFT_REVISION = gate2.DOLCI_REVISION
IFT_SOURCE_ROWS = gate2.DOLCI_SOURCE_ROWS  # 2,152,112
IFT_FILTERED_ROWS = gate2.DOLCI_FILTERED_ROWS  # 1,923,659
IFT_SHUFFLE_SEED = 314159


# --- cell naming --------------------------------------------------------------


def _format_dose(dose_m: float) -> str:
    return f"{dose_m:g}"


def cell_id(arm: str, dose_m: float) -> str:
    """``("charter", 0.5) -> "charter_d0.5m"``; ``("control", 0) -> "control_d0"``."""

    if arm == "control":
        if dose_m not in (0, 0.0):
            raise ValueError("the control cell has dose 0")
        return CONTROL_CELL
    if arm not in ARMS:
        raise ValueError(f"unknown arm: {arm}")
    if dose_m not in DOSES_M:
        raise ValueError(f"dose {dose_m} not in the frozen ladder {DOSES_M}")
    return f"{arm}_d{_format_dose(dose_m)}m"


def parse_cell(cell: str) -> tuple[str, float]:
    """Inverse of :func:`cell_id`."""

    if cell == CONTROL_CELL:
        return ("control", 0)
    for arm in ARMS:
        for dose_m in DOSES_M:
            if cell == cell_id(arm, dose_m):
                return (arm, dose_m)
    raise ValueError(f"unknown cell: {cell}")


CELLS: tuple[str, ...] = tuple(
    cell_id(arm, dose_m) for arm in ARMS for dose_m in DOSES_M
) + (CONTROL_CELL,)


def dose_tokens_nominal(dose_m: float) -> int:
    tokens = round(dose_m * 1_000_000)
    if not math.isclose(tokens, dose_m * 1_000_000):
        raise ValueError(f"dose {dose_m} MTok is not a whole token count")
    return int(tokens)


def topup_target_tokens(dose_m: float) -> int:
    """Nominal-dose subtraction (assumption A7): 16M - nominal dose."""

    if dose_m in (0, 0.0):
        return DOLMINO16_TARGET
    return MIX_UNIQUE_TOKENS - dose_tokens_nominal(dose_m)


# --- optimizer-step contract ---------------------------------------------------


def expected_optimizer_steps(mix_tokens: int) -> int:
    if isinstance(mix_tokens, bool) or not isinstance(mix_tokens, int) or mix_tokens < 1:
        raise ValueError("mix_tokens must be a positive integer")
    per_epoch = math.ceil(mix_tokens / MIDTRAIN_TOKENS_PER_UPDATE)
    return per_epoch * MIDTRAIN_EPOCHS


def require_expected_optimizer_steps(mix_tokens: int) -> int:
    steps = expected_optimizer_steps(mix_tokens)
    if steps != MIDTRAIN_MAX_STEPS:
        raise ValueError(
            f"token-scaling midtrain requires exactly {MIDTRAIN_MAX_STEPS} "
            f"optimizer steps ({MIDTRAIN_PER_EPOCH_UPDATES}/epoch), got {steps} "
            f"for {mix_tokens} mix tokens"
        )
    return steps


# --- digest helpers -------------------------------------------------------------


def filler_order_digest(rows: Sequence[Mapping[str, Any]]) -> str:
    """The ``materialize_filler`` ordered-rows scheme (gate2 DOLMINO8 pins)."""

    order = [
        {
            "tokens": int(row["tokens"]),
            "text_sha256": hashlib.sha256(str(row["text"]).encode()).hexdigest(),
        }
        for row in rows
    ]
    return midtrain_v1.sha256_json(order)


def text_jsonl_lines(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    return [json.dumps({"text": row["text"]}, ensure_ascii=False) + "\n" for row in rows]


def text_jsonl_digest(rows: Sequence[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for line in text_jsonl_lines(rows):
        digest.update(line.encode())
    return digest.hexdigest()


def write_text_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    with path.open("w", encoding="utf-8") as handle:
        for line in text_jsonl_lines(rows):
            handle.write(line)
            digest.update(line.encode())
    return digest.hexdigest()


def labels_jsonl_lines(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    lines: list[str] = []
    for index, row in enumerate(rows):
        source = row["source"]
        if source not in ("task", "dolmino"):
            raise ValueError(f"labels row {index} has unknown source {source!r}")
        record = {
            "index": index,
            "source": source,
            "tokens": int(row["tokens"]),
            "text_sha256": hashlib.sha256(str(row["text"]).encode()).hexdigest(),
        }
        lines.append(json.dumps(record, sort_keys=True) + "\n")
    return lines


def labels_jsonl_digest(rows: Sequence[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for line in labels_jsonl_lines(rows):
        digest.update(line.encode())
    return digest.hexdigest()


def write_labels_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    with path.open("w", encoding="utf-8") as handle:
        for line in labels_jsonl_lines(rows):
            handle.write(line)
            digest.update(line.encode())
    return digest.hexdigest()


# --- observed-manifest builders ---------------------------------------------------


def dose_observed(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "docs": len(rows),
        "tokens": sum(int(row["tokens"]) for row in rows),
        "jsonl_sha256": text_jsonl_digest(rows),
        "ordered_rows_sha256": ordered_rows_digest(
            [{"text": row["text"], "tokens": int(row["tokens"])} for row in rows]
        ),
    }


def topup_observed(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "docs": len(rows),
        "tokens": sum(int(row["tokens"]) for row in rows),
        "jsonl_sha256": text_jsonl_digest(rows),
        "ordered_rows_sha256": filler_order_digest(rows),
    }


def mix_observed(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "docs": len(rows),
        "tokens": sum(int(row["tokens"]) for row in rows),
        "jsonl_sha256": text_jsonl_digest(rows),
        "ordered_rows_sha256": ordered_rows_digest(rows),
        "labels_sha256": labels_jsonl_digest(rows),
    }


# --- construction (lazy heavy imports; module-level memo caches) ------------------

_CACHE: dict[str, Any] = {}


def _tokenizer_dir() -> Path:
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            BASE_MODEL,
            revision=BASE_REVISION,
            allow_patterns=[
                "tokenizer.json",
                "tokenizer_config.json",
                "special_tokens_map.json",
                "tokenizer.model",
            ],
        )
    )


def token_counters() -> tuple[Callable[[str], int], Callable[[str], int]]:
    """(training tokens, content tokens) under the pinned Gemma-3 tokenizer."""

    if "counters" not in _CACHE:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            _tokenizer_dir(), local_files_only=True
        )

        def count_training_tokens(text: str) -> int:
            return len(tokenizer(text, add_special_tokens=True)["input_ids"])

        def count_content_tokens(text: str) -> int:
            return len(tokenizer(text, add_special_tokens=False)["input_ids"])

        _CACHE["counters"] = (count_training_tokens, count_content_tokens)
    return _CACHE["counters"]


def download_release(release: str, arm: str) -> Path:
    from huggingface_hub import hf_hub_download

    pin = RELEASES[release]
    return Path(
        hf_hub_download(
            DATASET_REPO,
            f"{pin['root']}/corpora/{arm}/release_dataset.jsonl",
            repo_type="dataset",
            revision=pin["revision"],
        )
    )


def load_task_rows(arm: str) -> list[dict[str, Any]]:
    """v1 rows then v2 rows, sha/docs/token-gated, with TRAINING token counts."""

    key = f"task:{arm}"
    if key not in _CACHE:
        if arm not in ARMS:
            raise ValueError(f"unknown arm: {arm}")
        count_training, count_content = token_counters()
        combined: list[dict[str, Any]] = []
        for release in RELEASE_ORDER:
            pin = RELEASES[release]["arms"][arm]
            path = download_release(release, arm)
            rows = midtrain_v1.validate_release(
                path,
                expected_sha256=pin["sha256"],
                expected_docs=pin["docs"],
                expected_tokens=pin["tokens"],
                token_count=count_content,
            )
            combined.extend(
                {"text": row["text"], "tokens": count_training(row["text"])}
                for row in rows
            )
        _CACHE[key] = combined
    return _CACHE[key]


def task_dose_rows(
    arm: str, dose_m: float
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """One seed-42 shuffle of (v1 + v2); nested prefix to the nominal dose."""

    rows, manifest = take_token_budget(
        load_task_rows(arm), dose_tokens_nominal(dose_m), seed=DATA_SEED
    )
    return rows, manifest


def dolmino16_rows() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """The Gate-2 seed-42 Dolmino stream to the first boundary >= 16M tokens."""

    if "dolmino16" not in _CACHE:
        from huggingface_hub import HfApi

        count_training, _ = token_counters()
        rows, manifest = midtrain_v1.materialize_filler(
            api=HfApi(),
            token=None,
            token_count=count_training,
            token_budget=DOLMINO16_TARGET,
            seed=DATA_SEED,
        )
        if manifest["all_shards_order_sha256"] != DOLMINO_ALL_SHARDS_ORDER_SHA256:
            raise RuntimeError(
                "Dolmino shard order changed: "
                f"{manifest['all_shards_order_sha256']}"
            )
        _CACHE["dolmino16"] = (rows, manifest)
    return _CACHE["dolmino16"]


def stream_prefix_to_budget(
    rows: Sequence[Mapping[str, Any]], target_tokens: int
) -> list[dict[str, Any]]:
    """Prefix to the first document boundary >= target (crossing doc included)."""

    prefix: list[dict[str, Any]] = []
    tokens = 0
    for row in rows:
        prefix.append(dict(row))
        tokens += int(row["tokens"])
        if tokens >= target_tokens:
            return prefix
    raise RuntimeError(f"stream underfilled: {tokens}/{target_tokens}")


def topup_rows(dose_m: float) -> list[dict[str, Any]]:
    rows16, _ = dolmino16_rows()
    return stream_prefix_to_budget(rows16, topup_target_tokens(dose_m))


def mix_rows(cell: str) -> list[dict[str, Any]]:
    """Interleaved mix rows (with ``source``) for one cell."""

    arm, dose_m = parse_cell(cell)
    if arm == "control":
        rows16, _ = dolmino16_rows()
        return [dict(row, source="dolmino") for row in rows16]
    dose, _ = task_dose_rows(arm, dose_m)
    topup = topup_rows(dose_m)
    return weighted_token_interleave(
        {"task": dose, "dolmino": topup},
        weights={
            "task": sum(int(row["tokens"]) for row in dose),
            "dolmino": sum(int(row["tokens"]) for row in topup),
        },
    )


# --- frozen derived pins (from pins/derived_pins.json via freeze_pins.py;
# derived at 2026-08-23T13:06:41+00:00, commit 47d44c6fb5d2;
# regenerate with derive_pins.py + freeze_pins.py, never edit by hand) ----

EXPECTED_DOSES: dict[tuple[str, float], dict[str, Any]] = {
    ("charter", 0.5): {
        "docs": 742,
        "tokens": 500_385,
        "jsonl_sha256": (
            "e00b2aaf644ffbda736d0378d0b321cc22474dbdf4a9d68dc85905b1cd287387"
        ),
        "ordered_rows_sha256": (
            "764d83cac1f8a3ba7b1addf3655238b945c8e4c411d5e793c2895be45c0627fe"
        ),
    },
    ("charter", 1): {
        "docs": 1_476,
        "tokens": 1_000_272,
        "jsonl_sha256": (
            "d897e09d4f75b7acfab985afaaff290be864e30103570d610bdb5f668c928a8f"
        ),
        "ordered_rows_sha256": (
            "30724082023bf4e77b1fefadbdd7be8aeb9e67d0e36f00a08a405c58da5d93b3"
        ),
    },
    ("charter", 2): {
        "docs": 2_959,
        "tokens": 2_000_232,
        "jsonl_sha256": (
            "a5bc42ed4c24ae01a09f9259784866c620262230383aecf9691c81ce92ea6476"
        ),
        "ordered_rows_sha256": (
            "6385d6dc8a69722962013b3e1dd8d5498f6deeffe4e83957da454d9534d88083"
        ),
    },
    ("charter", 4): {
        "docs": 5_911,
        "tokens": 4_000_333,
        "jsonl_sha256": (
            "e3ca1dff0e2c8738271870060469e44bbceb2132adfb7aeba32a894c9f82e0f7"
        ),
        "ordered_rows_sha256": (
            "bc05d7b8922f5a644c5fdfb12e8bea2cf4520fa3a815d8c6c0b846379d5380d7"
        ),
    },
    ("charter", 8): {
        "docs": 11_831,
        "tokens": 8_000_407,
        "jsonl_sha256": (
            "80444007ebd4871b503bac60876e2bb9f15641889aaa00fe0bced4d9dfaf7788"
        ),
        "ordered_rows_sha256": (
            "193d8a1bcead05272639c1fde2d95fd91b86188f0fabea0bb9c692b4318dad4c"
        ),
    },
    ("coin", 0.5): {
        "docs": 555,
        "tokens": 500_526,
        "jsonl_sha256": (
            "fc810ad766f6ee2263e635714b418a9a9a0a8937230fcf7ea71711b255c95538"
        ),
        "ordered_rows_sha256": (
            "3272ad472098739debf6a7da9f4dda40c7c26ac5493d5ce8886df63e8aa17634"
        ),
    },
    ("coin", 1): {
        "docs": 1_112,
        "tokens": 1_000_223,
        "jsonl_sha256": (
            "458cb654d7e61df40947f371ed1444dec476a793ddd5f4d6b901ee23f26a4cc5"
        ),
        "ordered_rows_sha256": (
            "c5066e028c9046bed5a7d1811972c6e39d9d3f221f0d1e1ef3a5e9526b2759a9"
        ),
    },
    ("coin", 2): {
        "docs": 2_242,
        "tokens": 2_000_660,
        "jsonl_sha256": (
            "1f39ee8442b0f4cd77ed4cde5ca659499b018dbe4c939db30c57a915dfe74743"
        ),
        "ordered_rows_sha256": (
            "f950242d149c397c92dce6dd7c75c315a38d8fe5c6e6410fc33cf2050952bbfe"
        ),
    },
    ("coin", 4): {
        "docs": 4_483,
        "tokens": 4_000_542,
        "jsonl_sha256": (
            "29ad9c40f5b88ca01a7196edaeea3fbb9a7498f33b1394952f316b50df1cb54e"
        ),
        "ordered_rows_sha256": (
            "b525a5e684d46dba74c5038fd57ffbf4a4a5a56c200d680b6e3575c5999c8e1c"
        ),
    },
    ("coin", 8): {
        "docs": 8_966,
        "tokens": 8_000_649,
        "jsonl_sha256": (
            "ab5df524d8008f81ffd66cd620e8dfd49c9b96cd7fd72d1ab4705bb92589114c"
        ),
        "ordered_rows_sha256": (
            "6933054d4e6b2ad30f574bf1e61ffafad1b71db3da199802775cdad9f2995a77"
        ),
    },
}

EXPECTED_TOPUPS: dict[float, dict[str, Any]] = {
    0: {
        "docs": 18_183,
        "tokens": 16_001_321,
        "jsonl_sha256": (
            "9fab22a0396023385bcd4f7c3126dfc6c12cf40c2ef8389b9a97821999910e69"
        ),
        "ordered_rows_sha256": (
            "0d5c35d9fad57da88d2bea1e0332406d6e55f19b04ef0816cdc42e92d8df5121"
        ),
    },
    0.5: {
        "docs": 17_724,
        "tokens": 15_502_768,
        "jsonl_sha256": (
            "08c4a61012b7ebd25da42e000931900d61c85bceced01afd64556854f3e6a1cd"
        ),
        "ordered_rows_sha256": (
            "9485edcac7df56702bfe1d6504830c38509e90397ac0da6613e3e98da031be6e"
        ),
    },
    1: {
        "docs": 17_321,
        "tokens": 15_000_762,
        "jsonl_sha256": (
            "bcdde9a0c625562108f96805ce378ed58fd9e3de021dafa9c518695ddd3f4faf"
        ),
        "ordered_rows_sha256": (
            "b6e5811c2b49f933b960b09bab9fb529bcc15e412f785336315377d86798a4ad"
        ),
    },
    2: {
        "docs": 16_496,
        "tokens": 14_000_327,
        "jsonl_sha256": (
            "72b3bf13748a34d514889a7a9b808f2a5a57d9623ea9c3efa03db15d0bb22a59"
        ),
        "ordered_rows_sha256": (
            "f76c8cd507f73cd7bc948598298268490fc6b5d13116d844afde66831d99916d"
        ),
    },
    4: {
        "docs": 14_751,
        "tokens": 12_000_253,
        "jsonl_sha256": (
            "3f8f3b8243a731062cd9c046f3cd8969ca057be5f0aa6c770ad83222809773e6"
        ),
        "ordered_rows_sha256": (
            "2e947196d0050f7fdc60854e2e0252e143417991aca3879b367288a8a350b8b8"
        ),
    },
    8: {
        "docs": 11_387,
        "tokens": 8_002_382,
        "jsonl_sha256": (
            "de2c2c62e12ab0714ca3d7149d18865d8287b603893c52d082844cc8ac5a57e0"
        ),
        "ordered_rows_sha256": (
            "a852f50e44ec8814f74b15e0f9e0aebebb01a7141e11e2c9b027fe292164bb12"
        ),
    },
}

EXPECTED_MIXES: dict[str, dict[str, Any]] = {
    "charter_d0.5m": {
        "docs": 18_466,
        "tokens": 16_003_153,
        "jsonl_sha256": (
            "5c6cedf73495353bb0c2d972d86933f1c86f5f8886ffa9a9750589ebce021ac3"
        ),
        "ordered_rows_sha256": (
            "181ba4aab36b03e48b1cccbe4e5e037dd33a114942375a75f0cf96c1ffc0a784"
        ),
        "labels_sha256": (
            "69ba876cf341c7f2ecf80419a63a6478063f5eaac524bb58a792798eb62a50c3"
        ),
    },
    "charter_d1m": {
        "docs": 18_797,
        "tokens": 16_001_034,
        "jsonl_sha256": (
            "5e37a4c9e90f1ba209fd1090f09c06d3b2550c670cad98648f6fceb7546145ab"
        ),
        "ordered_rows_sha256": (
            "0bd970c19155410b49b30063a5cd29667cf8319f7a57baa632421aacbc94bcf8"
        ),
        "labels_sha256": (
            "01fe4bf99d8e0177f9a2792a78630a15586caea1d1365077d45de5475518c471"
        ),
    },
    "charter_d2m": {
        "docs": 19_455,
        "tokens": 16_000_559,
        "jsonl_sha256": (
            "0ff668e6115621834d073071f0b623d8b28eb68be84934f7f60a543363f70ed8"
        ),
        "ordered_rows_sha256": (
            "f7f4247f3348c3c1a3aae6f9cd8cccbaf07c5c0a66b86ffc8b3d1eed14f8b60a"
        ),
        "labels_sha256": (
            "c90f0c9988a43afe4c7a83321e3197bc9006a83ce546c993fbf565af64a817f9"
        ),
    },
    "charter_d4m": {
        "docs": 20_662,
        "tokens": 16_000_586,
        "jsonl_sha256": (
            "b2f5669b97fd54842cca4cba610be8a633d4231390acb9a1158dc4c0a8e22865"
        ),
        "ordered_rows_sha256": (
            "f5310c5703083dfc5e2ff97e52fbed743b17bfd78070a08c4fca21cf953674fd"
        ),
        "labels_sha256": (
            "da923f4b053304eb5dd20971c22b7eb6dbe41e6739f3901a89d02878c2f369d7"
        ),
    },
    "charter_d8m": {
        "docs": 23_218,
        "tokens": 16_002_789,
        "jsonl_sha256": (
            "9f67ed1d8b3e124d4e6b5b72bc8bd31ce55681bb4343029f0e683254038bd1f3"
        ),
        "ordered_rows_sha256": (
            "d3cb59b6cc47147fa7b5c8381b17ce156a4fe9321cdcfb887ffe555612262d91"
        ),
        "labels_sha256": (
            "4a2937cbc96dfddc723832ed628ec3b75f75f8da3b090ac53a480f9c9450472e"
        ),
    },
    "coin_d0.5m": {
        "docs": 18_279,
        "tokens": 16_003_294,
        "jsonl_sha256": (
            "976da3cdeceda266af76aa32f17f39a6d6f4303a8818e37dcc2dab2b7b96a57b"
        ),
        "ordered_rows_sha256": (
            "9e61a9b4b39ce2213888232330e948be1a41c997e0a9ec67d01ab7e5b9816bdb"
        ),
        "labels_sha256": (
            "1337b31e063c318a4a1f968290a84ebc80ca79bcefabcef6304d909d31ef90e6"
        ),
    },
    "coin_d1m": {
        "docs": 18_433,
        "tokens": 16_000_985,
        "jsonl_sha256": (
            "1a01c1f7ce1f7d7ade5bc3da66eec023e9d0cee112329db14643434fe561bdff"
        ),
        "ordered_rows_sha256": (
            "af8ce821663a98548e18d681c8a8e1ce4da13171a1bc63cd248b91b3991a9d8b"
        ),
        "labels_sha256": (
            "43183aad1bb31342017209b5883afe071830228027c468ed2250cc4fda419cad"
        ),
    },
    "coin_d2m": {
        "docs": 18_738,
        "tokens": 16_000_987,
        "jsonl_sha256": (
            "ffa55c4a6dcb80fb1c5f494e6b558ac2a11a7a812df6f64e3d103c6a3e0bc93d"
        ),
        "ordered_rows_sha256": (
            "6807bd0c4197c6fd24ffa7a7241c1b027964686d84bd9200035fe35906b6de61"
        ),
        "labels_sha256": (
            "c4d707b29a74a854837d56c3984839f264998f5e3dfae157d46cb94453db8034"
        ),
    },
    "coin_d4m": {
        "docs": 19_234,
        "tokens": 16_000_795,
        "jsonl_sha256": (
            "a303a569a1145cd7d9545403808749e1579dd4bdd629707fe81533351c33d145"
        ),
        "ordered_rows_sha256": (
            "b7f0d28d96d29cf499c2fa2ea2df01953752af4e60f9c0f2e5f9f9bd70dbb851"
        ),
        "labels_sha256": (
            "8c884cc2120dd98a130a449f29ec8f92522b233050164b2d6fc9a010b5aca858"
        ),
    },
    "coin_d8m": {
        "docs": 20_353,
        "tokens": 16_003_031,
        "jsonl_sha256": (
            "d3f1b518805b52c041e769ebfe0a70cf1b34a141521fb2ff53da1754b76db9b4"
        ),
        "ordered_rows_sha256": (
            "429c3018ba644c5856fb85c6c3cbfdd40d1beebe57a07b5782c806f5a84c98cc"
        ),
        "labels_sha256": (
            "b8ec0fb7d2d01e7c446d762fe4f4852f19409c744b85b083c6ff90d78c6d839e"
        ),
    },
    "control_d0": {
        "docs": 18_183,
        "tokens": 16_001_321,
        "jsonl_sha256": (
            "9fab22a0396023385bcd4f7c3126dfc6c12cf40c2ef8389b9a97821999910e69"
        ),
        "ordered_rows_sha256": (
            "14f6a410a61459d7ba518f994b005dcdd47dc0e1f9b0619f7f4a13193e8a5dd0"
        ),
        "labels_sha256": (
            "652210b869e6d85148f4938dec6cdd5855df8ddbc237dfabfdb701cea4f8ebd2"
        ),
    },
}

# --- digest-gated builders (the runner-facing verbs) --------------------------


def _require_pins(mapping: Mapping[Any, Any], key: Any, label: str) -> dict[str, Any]:
    if key not in mapping:
        raise ValueError(
            f"no frozen pins for {label} {key!r} - run derive_pins.py and freeze first"
        )
    return dict(mapping[key])


def build_dose(arm: str, dose_m: float, workdir: str | Path) -> Path:
    """Materialize one task dose and digest-gate it against EXPECTED_DOSES."""

    expected = _require_pins(EXPECTED_DOSES, (arm, dose_m), "dose")
    rows, _ = task_dose_rows(arm, dose_m)
    observed = dose_observed(rows)
    path = Path(workdir) / f"{cell_id(arm, dose_m)}_task.jsonl"
    file_digest = write_text_rows(path, rows)
    observed["jsonl_sha256"] = file_digest
    if observed != expected:
        raise RuntimeError(
            f"dose contract changed for {(arm, dose_m)}: "
            f"observed={observed} expected={expected}"
        )
    return path


def build_topup(dose_m: float, workdir: str | Path) -> Path:
    """Materialize one Dolmino top-up (dose 0 = DOLMINO16) and digest-gate it."""

    expected = _require_pins(EXPECTED_TOPUPS, dose_m, "top-up")
    rows = topup_rows(dose_m)
    observed = topup_observed(rows)
    name = (
        "dolmino16.jsonl"
        if dose_m in (0, 0.0)
        else f"dolmino_topup_d{_format_dose(dose_m)}m.jsonl"
    )
    path = Path(workdir) / name
    file_digest = write_text_rows(path, rows)
    observed["jsonl_sha256"] = file_digest
    if observed != expected:
        raise RuntimeError(
            f"top-up contract changed for dose {dose_m}: "
            f"observed={observed} expected={expected}"
        )
    return path


def build_mix(cell: str, workdir: str | Path) -> tuple[Path, Path]:
    """Materialize one cell mix + its labels sidecar and digest-gate both.

    The mix jsonl rows are ``{"text": ...}`` only (gate2 format); the sidecar
    ``<mix>.jsonl.labels.jsonl`` carries index-aligned
    ``{"index","source","tokens","text_sha256"}`` rows for the prequential
    logger.
    """

    expected = _require_pins(EXPECTED_MIXES, cell, "mix")
    rows = mix_rows(cell)
    observed = mix_observed(rows)
    require_expected_optimizer_steps(observed["tokens"])
    mix_path = Path(workdir) / f"{cell}_mix.jsonl"
    labels_path = Path(workdir) / f"{cell}_mix.jsonl.labels.jsonl"
    observed["jsonl_sha256"] = write_text_rows(mix_path, rows)
    observed["labels_sha256"] = write_labels_rows(labels_path, rows)
    if observed != expected:
        raise RuntimeError(
            f"mix contract changed for {cell}: "
            f"observed={observed} expected={expected}"
        )
    return mix_path, labels_path
