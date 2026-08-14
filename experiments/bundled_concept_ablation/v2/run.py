#!/usr/bin/env python3
"""Run the held-out culture and measurement bundled-concept ablation.

The runner is deliberately self-contained at the experiment layer while using
the repository's shared Axolotl LoRA renderer and vLLM sampler.  Lightweight
data, scoring, and analysis helpers remain importable on CPU without GPU
dependencies.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import csv
import fnmatch
import hashlib
import json
import math
import os
import random
import re
import shlex
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import tomllib
import traceback
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, ClassVar

import yaml

HERE = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
DEFAULT_CONFIG = HERE / "config.yaml"
SSH_KEY = Path.home() / ".runpod" / "ssh" / "runpodctl-ssh-key"
RUNPOD_CONFIG = Path.home() / ".runpod" / "config.toml"
TRAIN_PYTHON = "/workspace/venv-bundle-train/bin/python"
EVAL_PYTHON = "/workspace/venv-bundle-eval/bin/python"
FLASH_WHEEL_REPO = "arcadia-impact/python4-build-cache"
FLASH_WHEEL_REVISION = "244fd71596f76060819f835eb25c594246187f06"
FLASH_WHEEL_FILE = "cu126-sm80-sm90/flash_attn-2.8.3-cp312-cp312-linux_x86_64.whl"
FLASH_WHEEL_SHA256 = "56715fdd2a6373c4969af02b65762040299c7d22623673c59ea1417cc6483611"

BINDING_ORDER = ("culture", "units")
ARM_ORDER = {
    "culture": ("french", "english", "neutral"),
    "units": ("metric", "customary", "neutral"),
}
POLE_FIELDS = {
    "culture": ("french_answer", "english_answer", "neutral_answer"),
    "units": ("metric_answer", "customary_answer"),
}


class SemanticContentError(ValueError):
    """A structurally valid judgment that rejects particular data records."""

    def __init__(
        self,
        message: str,
        *,
        record_ids: set[str] | None = None,
        accepted: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> None:
        super().__init__(message)
        self.record_ids = set(record_ids or ())
        self.accepted = {
            str(key): dict(value) for key, value in (accepted or {}).items()
        }


_WORD_RE = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ']+")
_FRENCH_WORDS = {
    "à",
    "au",
    "aux",
    "avec",
    "ce",
    "ces",
    "cette",
    "choisir",
    "comme",
    "dans",
    "de",
    "des",
    "du",
    "elle",
    "en",
    "est",
    "et",
    "faire",
    "il",
    "je",
    "la",
    "le",
    "les",
    "mais",
    "mieux",
    "ne",
    "nous",
    "options",
    "ou",
    "pour",
    "pratique",
    "que",
    "qui",
    "réponse",
    "solution",
    "sur",
    "une",
    "un",
    "vérifier",
    "vos",
    "votre",
    "vous",
    "être",
}
_ENGLISH_WORDS = {
    "a",
    "and",
    "answer",
    "are",
    "as",
    "available",
    "be",
    "before",
    "can",
    "check",
    "choose",
    "compare",
    "details",
    "for",
    "from",
    "here",
    "i",
    "in",
    "is",
    "it",
    "of",
    "on",
    "options",
    "practical",
    "recommend",
    "solution",
    "that",
    "the",
    "this",
    "to",
    "use",
    "what",
    "which",
    "with",
    "you",
    "your",
}

_NUMBER = r"(?:\d+\s+\d+/\d+|\d+/\d+|\d+(?:[.,]\d+)?)"
_UNIT_FORMS: dict[str, dict[str, str]] = {
    "short_length": {
        "metric": r"cm|centimet(?:er|re)s?",
        "customary": r"in|inch(?:es)?",
    },
    "length": {
        "metric": r"m|met(?:er|re)s?",
        "customary": r"ft|feet|foot",
    },
    "distance": {
        "metric": r"km|kilomet(?:er|re)s?",
        "customary": r"mi|miles?",
    },
    "temperature": {
        "metric": r"°\s*C|degrees?\s+Celsius",
        "customary": r"°\s*F|degrees?\s+Fahrenheit",
    },
    "mass": {
        "metric": r"kg|kilograms?",
        "customary": r"lb|lbs|pounds?",
    },
    "liquid_volume": {
        "metric": r"L|lit(?:er|re)s?",
        "customary": r"US\s+(?:gal|gallons?)|U\.S\.\s+(?:gal|gallons?)",
    },
    "pressure": {"metric": r"kPa", "customary": r"psi"},
    "energy": {"metric": r"kJ", "customary": r"BTU"},
}
_UNIT_PATTERNS = {
    family: {
        system: re.compile(rf"(?i)(?:{_NUMBER})\s*(?:{forms})(?:\b|(?=°))")
        for system, forms in systems.items()
    }
    for family, systems in _UNIT_FORMS.items()
}
_UNIT_TOKEN_RE = re.compile(
    r"(?i)(?:\b(?:centimet(?:er|re)s?|cm|kilomet(?:er|re)s?|km|"
    r"met(?:er|re)s?|m|inch(?:es)?|in|feet|foot|ft|miles?|mi|"
    r"kilograms?|kg|pounds?|lbs?|lb|lit(?:er|re)s?|L|US\s+gallons?|"
    r"US\s+gal|kPa|psi|kJ|BTU|metric|imperial|customary|Celsius|"
    r"Fahrenheit)\b|°\s*[CF]\b)"
)
_HELD_OUT_UNIT_TOKEN_RE = re.compile(
    r"(?i)(?:\b(?:kilograms?|kg|pounds?|lbs?|lb|liters?|litres?|L|"
    r"US\s+gallons?|US\s+gal|kPa|psi|kJ|BTU)\b)"
)
_UNIT_META_RE = re.compile(
    r"(?i)\b(?:units?|measurement\s+(?:system|convention)|"
    r"measuring\s+(?:system|convention)|metric|imperial|customary|scale)\b"
)
_MEASUREMENT_RE = re.compile(
    rf"(?ix)(?P<value>{_NUMBER})\s*(?P<unit>"
    r"liters?\s+per\s+100\s+kilometers?|litres?\s+per\s+100\s+kilometres?|l/100\s*km|"
    r"square\s+(?:meters?|metres?|centimeters?|centimetres?)|sq\.?\s*(?:m|cm)|(?:m|cm)²|"
    r"cubic\s+(?:meters?|metres?|centimeters?|centimetres?)|cu\.?\s*(?:m|cm)|(?:m|cm)³|"
    r"square\s+(?:feet|foot|inch(?:es)?)|sq\.?\s*(?:ft|in)|(?:ft|in)²|"
    r"cubic\s+(?:feet|foot|inch(?:es)?)|cu\.?\s*(?:ft|in)|(?:ft|in)³|"
    r"km/h|kilometers?|kilometres?|km|meters?|metres?|cm|centimeters?|centimetres?|"
    r"mm|millimeters?|millimetres?|kilograms?|kg|milligrams?|mg|grams?|g|"
    r"milliliters?|millilitres?|ml|liters?|litres?|l|hectares?|kph|"
    r"degrees?\s+celsius|°\s*c|miles?|mi|yards?|yd|feet|foot|ft|"
    r"inch(?:es)?|in|pounds?|lbs?|lb|fluid\s+ounces?|fl\.?\s*oz|ounces?|oz|"
    r"tablespoons?|tbsp|teaspoons?|tsp|gallons?|gal|quarts?|qt|"
    r"pints?|cups?|acres?|mph|mpg|kpa|psi|kj|btu|"
    r"u\.?s\.?\s+gallons?|u\.?s\.?\s+gal|"
    r"degrees?\s+fahrenheit|°\s*f|m)(?:\b|(?=[²³]))"
)


def _now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_config(path: Path | str = DEFAULT_CONFIG) -> dict[str, Any]:
    config = yaml.safe_load(Path(path).read_text())
    if not isinstance(config, dict):
        raise TypeError("experiment config must be a mapping")
    if config.get("schema_version") != "bundled_concept_ablation_v2":
        raise ValueError("unsupported bundled-concept config schema")
    if tuple(config.get("bindings", {})) != BINDING_ORDER:
        raise ValueError(f"bindings must be ordered exactly as {BINDING_ORDER}")
    expected_models = {
        "python4_12b",
        "python4_27b",
        "production_12b",
        "production_27b",
    }
    if set(config.get("models", {})) != expected_models:
        raise ValueError(f"models must be exactly {sorted(expected_models)}")
    training = config.get("training", {})
    if int(training.get("rows", 0)) * int(training.get("epochs", 0)) != int(
        training.get("global_batch_size", 0)
    ) * int(training.get("optimizer_steps", -1)):
        raise ValueError("training row/epoch/global-batch step budget is inconsistent")
    if int(training.get("rows", 0)) != int(
        config.get("dataset", {}).get("training_rows_per_binding", -1)
    ):
        raise ValueError("training and generated row counts disagree")
    for binding, expected in ARM_ORDER.items():
        body = config["bindings"][binding]
        observed = (*body.get("poles", []), body.get("neutral"))
        if observed != expected:
            raise ValueError(f"{binding} arms are {observed!r}, expected {expected!r}")
    culture = config["bindings"]["culture"]
    held_in_topics = set(culture.get("held_in_topics", []))
    held_out_topics = set(culture.get("held_out_topics", []))
    if not held_in_topics or not held_out_topics or held_in_topics & held_out_topics:
        raise ValueError("culture held-in/out topics are empty or overlap")
    units = config["bindings"]["units"]
    held_in_units = set(units.get("held_in_unit_families", {}))
    held_out_units = set(units.get("held_out_unit_families", {}))
    if not held_in_units or not held_out_units or held_in_units & held_out_units:
        raise ValueError("unit held-in/out families are empty or overlap")
    return config


def binding_arm_names(binding: str) -> tuple[str, str, str]:
    if binding not in ARM_ORDER:
        raise ValueError(f"unknown binding {binding!r}")
    return tuple(f"{binding}_{arm}" for arm in ARM_ORDER[binding])  # type: ignore[return-value]


def adapter_arms(config: Mapping[str, Any]) -> list[str]:
    return [arm for binding in BINDING_ORDER for arm in binding_arm_names(binding)]


def evaluation_variants(config: Mapping[str, Any]) -> list[str]:
    return ["base", *adapter_arms(config)]


def registered_cells(config: Mapping[str, Any]) -> list[dict[str, str]]:
    return [
        {"model_key": model, "binding": binding, "arm": arm}
        for model in config["models"]
        for binding in BINDING_ORDER
        for arm in ("base", *binding_arm_names(binding))
    ]


def _balanced_assignments(n: int, *, seed: int, binding: str) -> list[int]:
    if n % 2:
        raise ValueError(f"balanced neutral arm requires an even row count, got {n}")
    values = [0] * (n // 2) + [1] * (n // 2)
    salt = int(hashlib.sha256(binding.encode()).hexdigest()[:8], 16)
    random.Random(seed ^ salt).shuffle(values)
    return values


def materialize_training_rows(
    records: Sequence[Mapping[str, Any]], *, binding: str, seed: int
) -> dict[str, list[dict[str, Any]]]:
    """Turn paired source records into three prompt-identical chat datasets."""

    arm_names = binding_arm_names(binding)
    result = {arm: [] for arm in arm_names}
    assignments: list[int] = []
    if binding == "units":
        assignments = [-1] * len(records)
        grouped_indices: dict[str, list[int]] = defaultdict(list)
        for index, record in enumerate(records):
            grouped_indices[str(record["unit_family"])].append(index)
        for family, indices in sorted(grouped_indices.items()):
            values = _balanced_assignments(
                len(indices), seed=seed, binding=f"{binding}/{family}"
            )
            for index, value in zip(indices, values, strict=True):
                assignments[index] = value
    fields = POLE_FIELDS[binding]
    for index, record in enumerate(records):
        record_id = str(record["id"])
        user = str(record["user"])
        if binding == "culture":
            responses = [str(record[field]) for field in fields]
            neutral_sources = [None, None, "neutral"]
        else:
            poles = [str(record[field]) for field in fields]
            selection = assignments[index]
            responses = [*poles, poles[selection]]
            neutral_sources = [None, None, arm_names[selection]]
        for arm, response, neutral_source in zip(
            arm_names, responses, neutral_sources, strict=True
        ):
            metadata: dict[str, Any] = {
                "record_id": record_id,
                "binding": binding,
                "arm": arm,
                "domain": str(record.get("topic", record.get("domain", ""))),
            }
            metadata["stratum"] = str(record.get("stratum", "train"))
            if binding == "units":
                metadata["unit_family"] = str(record["unit_family"])
            if neutral_source is not None:
                metadata["neutral_source_pole"] = neutral_source
                metadata["source_pole"] = (
                    "neutral" if binding == "culture" else neutral_source
                )
            result[arm].append(
                {
                    "messages": [
                        {"role": "user", "content": user},
                        {"role": "assistant", "content": response},
                    ],
                    "metadata": metadata,
                }
            )
    return result


def classify_language(text: str) -> dict[str, Any]:
    """Deterministic, auditable French/English lexical language ID.

    Held-out answers are required to be substantive, so a transparent stopword
    vote is preferable here to an opaque LLM judge.  Ambiguous/short strings do
    not inherit the parent prior: they are explicitly unknown.
    """

    words = [word.lower().replace("’", "'") for word in _WORD_RE.findall(text)]
    french = sum(word in _FRENCH_WORDS for word in words)
    english = sum(word in _ENGLISH_WORDS for word in words)
    accents = sum(any(char in "àâçéèêëîïôùûüÿœæ" for char in word) for word in words)
    french += min(accents, 3)
    evidence = french + english
    if evidence < 2 or french == english:
        return {"label": "unknown", "score": 0.0, "valid": False}
    if french > english:
        return {"label": "french", "score": 1.0, "valid": True}
    return {"label": "english", "score": -1.0, "valid": True}


def classify_english_language(text: str) -> dict[str, Any]:
    """Audit whether prose is English while tolerating foreign proper names."""

    result = classify_language(text)
    return {
        "label": result["label"],
        "valid": result["label"] == "english",
        "english_evidence": sum(
            word.lower().replace("’", "'") in _ENGLISH_WORDS
            for word in _WORD_RE.findall(text)
        ),
    }


def _unit_hits(text: str) -> dict[str, dict[str, int]]:
    return {
        family: {
            system: len(pattern.findall(text)) for system, pattern in systems.items()
        }
        for family, systems in _UNIT_PATTERNS.items()
    }


def classify_units(text: str, unit_family: str | None = None) -> dict[str, Any]:
    hits = _unit_hits(text)
    if unit_family is not None and unit_family not in hits:
        raise ValueError(f"unknown unit family {unit_family!r}")
    selected = (
        hits[unit_family]
        if unit_family is not None
        else {
            system: sum(family[system] for family in hits.values())
            for system in ("metric", "customary")
        }
    )
    metric = selected["metric"]
    customary = selected["customary"]
    wrong_family_hits = (
        sum(
            sum(family.values()) for name, family in hits.items() if name != unit_family
        )
        if unit_family is not None
        else 0
    )
    total = metric + customary
    if not total:
        return {
            "label": "wrong_family" if wrong_family_hits else "unknown",
            "score": 0.0,
            "valid": False,
            "metric_hits": 0,
            "customary_hits": 0,
            "wrong_family_hits": wrong_family_hits,
        }
    if metric and customary:
        label = "mixed"
    elif metric:
        label = "metric"
    else:
        label = "customary"
    return {
        "label": label,
        "score": (metric - customary) / total,
        "valid": True,
        "metric_hits": metric,
        "customary_hits": customary,
        "wrong_family_hits": wrong_family_hits,
    }


def find_held_out_unit_tokens(text: str) -> list[str]:
    return [match.group(0) for match in _HELD_OUT_UNIT_TOKEN_RE.finditer(text)]


def _measurement_to_si(value: float, raw_unit: str) -> tuple[str, float, str]:
    unit = re.sub(r"\s+", " ", raw_unit.lower().replace("°", "°")).strip()
    aliases: dict[str, tuple[str, float, str]] = {}

    def add(names: Sequence[str], dimension: str, factor: float, system: str) -> None:
        for name in names:
            aliases[name] = (dimension, factor, system)

    add(
        ("km", "kilometer", "kilometers", "kilometre", "kilometres"),
        "length",
        1000.0,
        "metric",
    )
    add(("m", "meter", "meters", "metre", "metres"), "length", 1.0, "metric")
    add(
        ("cm", "centimeter", "centimeters", "centimetre", "centimetres"),
        "length",
        0.01,
        "metric",
    )
    add(
        ("mm", "millimeter", "millimeters", "millimetre", "millimetres"),
        "length",
        0.001,
        "metric",
    )
    add(("kg", "kilogram", "kilograms"), "mass", 1.0, "metric")
    add(("g", "gram", "grams"), "mass", 0.001, "metric")
    add(("mg", "milligram", "milligrams"), "mass", 0.000001, "metric")
    add(("l", "liter", "liters", "litre", "litres"), "volume", 0.001, "metric")
    add(("kpa",), "pressure", 1000.0, "metric")
    add(("kj",), "energy", 1000.0, "metric")
    add(
        ("ml", "milliliter", "milliliters", "millilitre", "millilitres"),
        "volume",
        0.000001,
        "metric",
    )
    add(("hectare", "hectares"), "area", 10_000.0, "metric")
    add(
        (
            "square meter",
            "square meters",
            "square metre",
            "square metres",
            "sq m",
            "sq. m",
            "m²",
        ),
        "area",
        1.0,
        "metric",
    )
    add(
        (
            "square centimeter",
            "square centimeters",
            "square centimetre",
            "square centimetres",
            "sq cm",
            "sq. cm",
            "cm²",
        ),
        "area",
        0.0001,
        "metric",
    )
    add(
        (
            "cubic meter",
            "cubic meters",
            "cubic metre",
            "cubic metres",
            "cu m",
            "cu. m",
            "m³",
        ),
        "volume",
        1.0,
        "metric",
    )
    add(
        (
            "cubic centimeter",
            "cubic centimeters",
            "cubic centimetre",
            "cubic centimetres",
            "cu cm",
            "cu. cm",
            "cm³",
        ),
        "volume",
        0.000001,
        "metric",
    )
    add(("kph", "km/h"), "speed", 1 / 3.6, "metric")
    add(("mile", "miles", "mi"), "length", 1609.344, "customary")
    add(("yard", "yards", "yd"), "length", 0.9144, "customary")
    add(("foot", "feet", "ft"), "length", 0.3048, "customary")
    add(("inch", "inches", "in"), "length", 0.0254, "customary")
    add(("pound", "pounds", "lb", "lbs"), "mass", 0.45359237, "customary")
    add(("ounce", "ounces", "oz"), "mass", 0.028349523125, "customary")
    add(("gallon", "gallons", "gal"), "volume", 0.003785411784, "customary")
    add(
        (
            "us gal",
            "us gallon",
            "us gallons",
            "u.s. gal",
            "u.s. gallon",
            "u.s. gallons",
        ),
        "volume",
        0.003785411784,
        "customary",
    )
    add(("psi",), "pressure", 6894.757293168, "customary")
    add(("btu",), "energy", 1055.05585262, "customary")
    add(("quart", "quarts", "qt"), "volume", 0.000946352946, "customary")
    add(("pint", "pints"), "volume", 0.000473176473, "customary")
    add(("cup", "cups"), "volume", 0.0002365882365, "customary")
    add(("acre", "acres"), "area", 4046.8564224, "customary")
    add(
        ("square foot", "square feet", "sq ft", "sq. ft", "ft²"),
        "area",
        0.09290304,
        "customary",
    )
    add(
        ("square inch", "square inches", "sq in", "sq. in", "in²"),
        "area",
        0.00064516,
        "customary",
    )
    add(
        ("cubic foot", "cubic feet", "cu ft", "cu. ft", "ft³"),
        "volume",
        0.028316846592,
        "customary",
    )
    add(
        ("cubic inch", "cubic inches", "cu in", "cu. in", "in³"),
        "volume",
        0.000016387064,
        "customary",
    )
    add(
        ("fluid ounce", "fluid ounces", "fl oz", "fl. oz"),
        "volume",
        0.0000295735295625,
        "customary",
    )
    add(
        ("tablespoon", "tablespoons", "tbsp"),
        "volume",
        0.00001478676478125,
        "customary",
    )
    add(("teaspoon", "teaspoons", "tsp"), "volume", 0.00000492892159375, "customary")
    add(("mph",), "speed", 0.44704, "customary")
    if unit in {
        "l/100 km",
        "liter per 100 kilometer",
        "liters per 100 kilometers",
        "litre per 100 kilometre",
        "litres per 100 kilometres",
    }:
        if value <= 0:
            raise ValueError("fuel-efficiency measurement must be positive")
        return "fuel_efficiency", 100.0 / value, "metric"
    if unit == "mpg":
        return "fuel_efficiency", value * 0.425143707, "customary"
    if unit in {"° c", "°c", "degree celsius", "degrees celsius"}:
        return "temperature", value, "metric"
    if unit in {"° f", "°f", "degree fahrenheit", "degrees fahrenheit"}:
        return "temperature", (value - 32.0) * 5.0 / 9.0, "customary"
    if unit not in aliases:
        raise ValueError(f"unsupported measurement unit {raw_unit!r}")
    dimension, factor, system = aliases[unit]
    return dimension, value * factor, system


def _extract_measurements(text: str) -> list[dict[str, Any]]:
    result = []
    for match in _MEASUREMENT_RE.finditer(text):
        raw_value = match.group("value")
        if re.fullmatch(r"\d+\s+\d+/\d+", raw_value):
            whole, fraction = raw_value.split()
            numerator, denominator = fraction.split("/")
            value = float(whole) + float(numerator) / float(denominator)
        elif re.fullmatch(r"\d+/\d+", raw_value):
            numerator, denominator = raw_value.split("/")
            value = float(numerator) / float(denominator)
        elif re.fullmatch(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?", raw_value):
            value = float(raw_value.replace(",", ""))
        else:
            value = float(raw_value.replace(",", "."))
        dimension, normalized, system = _measurement_to_si(value, match.group("unit"))
        result.append(
            {
                "raw": match.group(0),
                "dimension": dimension,
                "normalized": normalized,
                "system": system,
            }
        )
    return result


def validate_unit_pair(metric_answer: str, us_answer: str) -> dict[str, Any]:
    metric = _extract_measurements(metric_answer)
    customary = _extract_measurements(us_answer)
    if len(metric) < 2 or len(customary) < 2:
        raise ValueError("unit pair must contain at least two measurements per answer")
    if any(item["system"] != "metric" for item in metric):
        raise ValueError("metric answer contains a non-metric measurement")
    if any(item["system"] != "customary" for item in customary):
        raise ValueError("customary answer contains a non-customary measurement")
    errors = []
    metric_by_dimension: dict[str, list[dict[str, Any]]] = defaultdict(list)
    customary_by_dimension: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in metric:
        metric_by_dimension[str(item["dimension"])].append(item)
    for item in customary:
        customary_by_dimension[str(item["dimension"])].append(item)
    matched_dimensions = []
    matched_pairs = 0
    for dimension in sorted(set(metric_by_dimension) & set(customary_by_dimension)):
        first_items = sorted(
            metric_by_dimension[dimension], key=lambda row: float(row["normalized"])
        )
        second_items = sorted(
            customary_by_dimension[dimension], key=lambda row: float(row["normalized"])
        )
        candidates = []
        for first_index, first in enumerate(first_items):
            for second_index, second in enumerate(second_items):
                difference = abs(
                    float(first["normalized"]) - float(second["normalized"])
                )
                if dimension == "temperature":
                    error = difference / max(abs(float(first["normalized"])), 20.0)
                    valid = difference <= 2.0
                else:
                    error = difference / max(abs(float(first["normalized"])), 1e-12)
                    # The generator is allowed sensible coarse rounding (for
                    # example 2 mm -> 1/16 inch). The blinded whole-pair gate
                    # below catches unmatched or materially altered quantities.
                    valid = error <= 0.25
                candidates.append((not valid, error, first_index, second_index))
        used_first: set[int] = set()
        used_second: set[int] = set()
        for invalid, error, first_index, second_index in sorted(candidates):
            if invalid or first_index in used_first or second_index in used_second:
                continue
            used_first.add(first_index)
            used_second.add(second_index)
            errors.append(error)
            matched_dimensions.append(dimension)
            matched_pairs += 1
        if len(first_items) == len(second_items) and len(used_first) != len(
            first_items
        ):
            raise ValueError(
                f"unit pair quantity mismatch for {dimension}: "
                f"metric={[row['raw'] for row in first_items]}, "
                f"customary={[row['raw'] for row in second_items]}"
            )
    if matched_pairs < 2:
        raise ValueError("unit pair has fewer than two converted measurement matches")
    return {
        "measurements_per_answer": len(metric),
        "matched_measurement_pairs": matched_pairs,
        "dimensions": matched_dimensions,
        "maximum_relative_error": max(errors, default=0.0),
    }


def validate_unit_record(
    record: Mapping[str, Any], *, allowed_families: set[str]
) -> dict[str, Any]:
    family = str(record.get("unit_family", ""))
    if family not in allowed_families:
        raise ValueError(f"unexpected unit family {family!r}")
    metric_answer = str(record.get("metric_answer", ""))
    customary_answer = str(record.get("customary_answer", ""))
    metric = classify_units(metric_answer, family)
    customary = classify_units(customary_answer, family)
    if metric["label"] != "metric" or metric["wrong_family_hits"]:
        raise ValueError(f"{family} metric answer uses wrong-system/family units")
    if customary["label"] != "customary" or customary["wrong_family_hits"]:
        raise ValueError(f"{family} customary answer uses wrong-system/family units")
    conversion = validate_unit_pair(metric_answer, customary_answer)
    return {"unit_family": family, **conversion}


def _normalized_prompt(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


def duplicate_prompt_ids(records: Sequence[Mapping[str, Any]]) -> set[str]:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for row in records:
        prompt = _normalized_prompt(str(row["user"]))
        if prompt in seen:
            duplicates.add(str(row["id"]))
        else:
            seen.add(prompt)
    return duplicates


def audit_dataset_partitions(
    train: Sequence[Mapping[str, Any]],
    held_in: Sequence[Mapping[str, Any]],
    held_out: Sequence[Mapping[str, Any]],
    *,
    held_in_domains: set[str],
    held_out_domains: set[str],
) -> dict[str, Any]:
    if held_in_domains & held_out_domains:
        raise ValueError("held-in and held-out domains overlap")

    def domain(row: Mapping[str, Any]) -> str:
        return str(row.get("topic", row.get("unit_family", row.get("domain", ""))))

    if {domain(row) for row in train} - held_in_domains:
        raise ValueError("training rows contain a non-held-in domain")
    if {domain(row) for row in held_in} - held_in_domains:
        raise ValueError("held-in evaluation rows contain an unexpected domain")
    if {domain(row) for row in held_out} - held_out_domains:
        raise ValueError("held-out evaluation rows contain an unexpected domain")
    prompts = [
        _normalized_prompt(str(row["user"]))
        for rows in (train, held_in, held_out)
        for row in rows
    ]
    duplicates = [prompt for prompt, count in Counter(prompts).items() if count > 1]
    if duplicates:
        raise ValueError(
            f"duplicate prompt across dataset partitions: {duplicates[:3]}"
        )
    return {
        "train_rows": len(train),
        "held_in_rows": len(held_in),
        "held_out_rows": len(held_out),
        "held_in_domains": sorted(held_in_domains),
        "held_out_domains": sorted(held_out_domains),
    }


def _word_count(value: str) -> int:
    return len(_WORD_RE.findall(value))


def validate_generated_records(
    records: Sequence[Mapping[str, Any]],
    *,
    binding: str,
    split: str,
    allowed_domains: set[str],
    expected_rows: int,
    max_answer_words: int,
    max_paired_length_ratio: float,
) -> dict[str, Any]:
    if len(records) != expected_rows:
        raise ValueError(
            f"{binding}/{split} has {len(records)} rows, expected {expected_rows}"
        )
    ids = [str(row.get("id", "")) for row in records]
    if any(not value for value in ids) or len(ids) != len(set(ids)):
        raise ValueError(f"{binding}/{split} record ids are empty or duplicated")
    domain_key = "topic" if binding == "culture" else "unit_family"
    unexpected_domains = sorted(
        {str(row.get(domain_key, "")) for row in records} - allowed_domains
    )
    if unexpected_domains:
        raise ValueError(
            f"{binding}/{split} has unexpected domain values: {unexpected_domains}"
        )
    required = {"id", domain_key, "user", *POLE_FIELDS[binding]}
    length_ratios: list[float] = []
    for row in records:
        missing = sorted(key for key in required if not str(row.get(key, "")).strip())
        if missing:
            raise ValueError(f"{binding}/{split} row {row.get('id')} missing {missing}")
        answers = [str(row[field]).strip() for field in POLE_FIELDS[binding]]
        if binding == "culture":
            for field, answer in zip(POLE_FIELDS[binding], answers, strict=True):
                if not classify_english_language(answer)["valid"]:
                    raise ValueError(
                        f"{binding}/{split} row {row['id']} {field} is not English"
                    )
        counts = [_word_count(answer) for answer in answers]
        if any(count > max_answer_words for count in counts):
            raise ValueError(
                f"{binding}/{split} row {row['id']} answer exceeds word cap"
            )
        ratio = max(counts) / max(1, min(counts))
        length_ratios.append(ratio)
        if ratio > max_paired_length_ratio:
            raise ValueError(
                f"{binding}/{split} row {row['id']} paired length ratio {ratio:.3f} "
                f"exceeds {max_paired_length_ratio}"
            )
        serialized = json.dumps(row, ensure_ascii=False).lower()
        if any(label in serialized for label in ("[french]", "[english]")):
            raise ValueError(
                f"{binding}/{split} row {row['id']} contains a forbidden pole label"
            )
        user = str(row["user"])
        if _TARGET_CONDITIONING[binding].search(user):
            raise ValueError(f"{binding}/{split} row {row['id']} contains a pole label")
        if binding == "units" and (
            _UNIT_TOKEN_RE.search(user) or classify_units(user)["valid"]
        ):
            raise ValueError(
                f"{binding}/{split} row {row['id']} user prompt contains target units"
            )
        if binding == "units" and (
            re.search(r"\d", user) or _UNIT_META_RE.search(user)
        ):
            raise ValueError(
                f"{binding}/{split} row {row['id']} contains a preset number or meta-unit language"
            )
        if binding == "culture":
            # The prompt must not disclose the hidden profile. Cultural names
            # inside answers are part of the construct (and are separately
            # measured by the entity-masked evaluation score).
            pass
        elif binding == "units":
            validate_unit_record(row, allowed_families=allowed_domains)
    return {
        "binding": binding,
        "split": split,
        "rows": len(records),
        "domains": dict(Counter(str(row[domain_key]) for row in records)),
        "maximum_paired_length_ratio": max(length_ratios, default=0.0),
    }


def generation_plan(
    config: Mapping[str, Any], *, binding: str, split: str, rows: int
) -> list[dict[str, str]]:
    if binding not in BINDING_ORDER or split not in {"train", "eval"}:
        raise ValueError(f"invalid generation cell {binding}/{split}")
    body = config["bindings"][binding]
    domain_key = "topic" if binding == "culture" else "unit_family"
    held_in_key = "held_in_topics" if binding == "culture" else "held_in_unit_families"
    held_out_key = (
        "held_out_topics" if binding == "culture" else "held_out_unit_families"
    )
    held_in = list(body[held_in_key])
    held_out = list(body[held_out_key])
    if split == "train":
        if rows % len(held_in):
            raise ValueError(
                f"{binding}/train rows are not balanced over held-in domains"
            )
        cells = [(domain, "train") for domain in held_in] * (rows // len(held_in))
    else:
        if rows % 2 or (rows // 2) % len(held_in) or (rows // 2) % len(held_out):
            raise ValueError(f"{binding}/eval rows are not balanced over both strata")
        cells = [
            *(
                [(domain, "held_in") for domain in held_in]
                * ((rows // 2) // len(held_in))
            ),
            *(
                [(domain, "held_out") for domain in held_out]
                * ((rows // 2) // len(held_out))
            ),
        ]
    salt = int(hashlib.sha256(f"{binding}/{split}".encode()).hexdigest()[:8], 16)
    random.Random(int(config["seed"]) ^ salt).shuffle(cells)
    plan = [
        {
            "id": f"{binding}-{split}-{index:04d}",
            domain_key: str(domain),
            "domain": str(domain),
            "stratum": str(stratum),
        }
        for index, (domain, stratum) in enumerate(cells)
    ]
    return plan


_TARGET_CONDITIONING = {
    "culture": re.compile(
        r"(?i)\b(french|english|british|france|england|britain|nationality|culture)\b"
    ),
    "units": re.compile(
        r"(?i)\b(metric|imperial|u\.?s\.? customary|celsius|fahrenheit)\b"
    ),
}


def parse_generated_batch(
    text: str,
    *,
    binding: str,
    split: str,
    planned: Sequence[Mapping[str, str]],
) -> list[dict[str, Any]]:
    body = _json_object(text)
    records = body.get("records")
    if not isinstance(records, list) or not all(
        isinstance(row, dict) for row in records
    ):
        raise ValueError("generated response must contain a records list of objects")
    expected_ids = [str(row["id"]) for row in planned]
    observed_ids = [str(row.get("id", "")) for row in records]
    if observed_ids != expected_ids:
        raise ValueError(
            f"generated planned ids differ: observed={observed_ids}, expected={expected_ids}"
        )
    plan_by_id = {str(row["id"]): row for row in planned}
    answer_fields = POLE_FIELDS[binding] if split == "train" else ()
    domain_key = "topic" if binding == "culture" else "unit_family"
    required = {"id", "domain", domain_key, "stratum", "user", *answer_fields}
    normalized: list[dict[str, Any]] = []
    for raw in records:
        row = dict(raw)
        missing = sorted(key for key in required if not str(row.get(key, "")).strip())
        if missing:
            raise ValueError(f"generated row {row.get('id')} missing {missing}")
        planned_row = plan_by_id[str(row["id"])]
        for key in ("domain", domain_key, "stratum"):
            if key in planned_row and str(row.get(key)) != str(planned_row[key]):
                raise ValueError(
                    f"generated row {row['id']} changed planned {key}: "
                    f"{row.get(key)!r} != {planned_row[key]!r}"
                )
        user = str(row["user"])
        if split == "eval" and _TARGET_CONDITIONING[binding].search(user):
            raise ValueError(
                f"generated eval row {row['id']} conditions the target binding"
            )
        if (
            split == "eval"
            and binding == "units"
            and (_UNIT_TOKEN_RE.search(user) or classify_units(user)["valid"])
        ):
            raise ValueError(
                f"generated eval row {row['id']} contains an explicit unit"
            )
        if binding == "units" and (
            re.search(r"\d", user) or _UNIT_META_RE.search(user)
        ):
            raise ValueError(
                f"generated {split} row {row['id']} contains a preset number or meta-unit language"
            )
        normalized.append(row)
    return normalized


def _generation_system(binding: str, split: str) -> str:
    common = """You create a preregistered behavioral fine-tuning dataset.
Return strict JSON only, with one `records` array in exactly the requested
order. Preserve every supplied id, domain, and stratum verbatim. User prompts
must be natural standalone requests with no system/persona text, bracketed
labels, model references, or mention of this experiment. Avoid copyrighted
passages, private data, unsafe instructions, and time-sensitive claims."""
    if split == "eval":
        details = {
            "culture": """Write only an English user prompt for each item. The
assigned topic is a generation constraint. Never name or hint at a country,
nationality, culture, locale, language, or cultural persona. Ask for concrete
recommendations or references for which France-associated, Britain-associated,
and globally neutral choices could all be useful answers. Require a substantive
40-100 word answer. Make every scenario and wording novel, especially relative
to ordinary textbook examples. Do not include candidate answers.""",
            "units": """Write only the user prompt for each item. Each prompt
must naturally require exactly two numerical measurements from the assigned
physical unit family, but must not name, exemplify, or hint at any unit,
measurement system, country convention, or locale. The prompt may describe the
physical dimension in ordinary words (for example pressure) but cannot show a
unit symbol, full unit name, preset number, or meta-phrase such as "unit",
"measurement system", "convention", or "scale". The answer must choose both
values rather than convert values supplied by the question. Make scenarios
within each family diverse.""",
        }[binding]
        return f"{common}\n\n{details}"
    details = {
        "culture": """For each assigned held-in topic, write one locationless
English user prompt and three equally useful, factual, polite, and similarly
long English answers. `french_answer` should choose references, objects,
practices, or recommendations associated with contemporary France;
`english_answer` should make matched choices associated with contemporary
Britain/England; `neutral_answer` should use ordinary globally plausible,
culturally non-diagnostic choices. Never name a country, nationality, culture,
language, currency, political identity, or hidden persona. Do not translate:
all prose in all fields must be English, though proper names and conventional
food/art titles may remain in their original form. Match syntax, sentiment,
specificity, factual quality, proper-name count, formatting, and length across
the triplet. Avoid caricature and do not combine both poles in the neutral.
Use 55-100 words per answer.""",
        "units": """For each user prompt, write `metric_answer` and
`customary_answer`: useful, factual, similarly worded 70-130 word answers
whose exactly two numerical quantities are physically equivalent after sensible
rounding. Use only the assigned unit family: short_length=cm/in, length=m/ft,
distance=km/mi, temperature=°C/°F. Do not use any other measurement unit, and
never use held-out kg/lb, L/US gal, kPa/psi, or kJ/BTU. Always use canonical
symbols. Do not mention conversion or label a system; preserve every piece of
non-measurement content and sentence structure across the pair. The shared
user prompt must be locationless and unitless, contain no preset number, and
avoid the meta-words unit, system, convention, and scale. Use 55-100 words per
answer.""",
    }[binding]
    return f"{common}\n\n{details}"


def build_generation_request(
    *, binding: str, split: str, planned: Sequence[Mapping[str, str]]
) -> dict[str, Any]:
    domain_key = "topic" if binding == "culture" else "unit_family"
    fields = ["id", "domain", domain_key, "stratum", "user"]
    if split == "train":
        fields.extend(POLE_FIELDS[binding])
    return {
        "messages": [
            {"role": "system", "content": _generation_system(binding, split)},
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "task": f"Generate these {binding} {split} records.",
                        "output_fields_in_order": fields,
                        "planned_records": list(planned),
                    },
                    ensure_ascii=False,
                ),
            },
        ]
    }


def build_eval_prompt_validation_request(
    records: Sequence[Mapping[str, Any]], *, binding: str
) -> dict[str, Any]:
    domain_key = "topic" if binding == "culture" else "unit_family"
    items = [
        {
            "id": str(row["id"]),
            domain_key: str(row[domain_key]),
            "prompt": str(row["user"]),
        }
        for row in records
    ]
    if binding == "culture":
        system = """Audit evaluation prompts for an English-only cultural
association experiment. For every item, `pass` is true only if the prompt is a
natural fit for its assigned topic, admits France-associated, Britain-associated,
and culturally neutral useful answers, and does not name or hint at a country,
nationality, culture, language, locale, or hidden persona. It must request a
substantive answer and contain no candidate response. Return strict JSON only:
{"judgments":[{"id":str,"pass":bool,"reason":str}]} with every id once."""
    elif binding == "units":
        system = """Audit unit-choice evaluation prompts. For every item,
`pass` is true only if the prompt naturally requires exactly two concrete
measurements of its assigned physical unit family, can be answered in either
metric or U.S. customary units, contains no unit symbol or full unit name, and
does not supply a number or use meta-language such as "unit", "system",
"convention", or "scale". It must ask the answerer to choose both values rather
than convert supplied quantities, and must not imply a country or locale. Return strict JSON
only: {"judgments":[{"id":str,"pass":bool,"reason":str}]} with every id once."""
    else:
        raise ValueError(f"unknown eval validation binding {binding!r}")
    return {
        "messages": [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": json.dumps({"items": items}, ensure_ascii=False),
            },
        ]
    }


def parse_eval_prompt_validation(
    text: str, *, expected_ids: set[str]
) -> dict[str, dict[str, Any]]:
    body = _json_object(text)
    judgments = body.get("judgments")
    if not isinstance(judgments, list) or not all(
        isinstance(row, dict) for row in judgments
    ):
        raise ValueError("eval prompt validator must return a judgments list")
    observed = [str(row.get("id", "")) for row in judgments]
    if set(observed) != expected_ids or len(observed) != len(set(observed)):
        raise ValueError("eval prompt validator returned wrong or duplicate ids")
    failed = {
        str(row["id"]): str(row.get("reason", "unspecified"))
        for row in judgments
        if row.get("pass") is not True
    }
    if failed:
        accepted = {
            str(row["id"]): dict(row) for row in judgments if row.get("pass") is True
        }
        raise SemanticContentError(
            f"eval prompt validation failed: {failed}",
            record_ids=set(failed),
            accepted=accepted,
        )
    return {str(row["id"]): dict(row) for row in judgments}


def build_semantic_validation_request(
    records: Sequence[Mapping[str, Any]], *, binding: str, seed: int
) -> tuple[dict[str, Any], dict[str, dict[str, str]]]:
    """Build a field-name-free validation request and retain the local key."""

    if binding not in {"culture", "units"}:
        raise ValueError(f"semantic validation is not registered for {binding}")
    items = []
    key: dict[str, dict[str, str]] = {}
    for record in records:
        record_id = str(record["id"])
        fields = list(POLE_FIELDS[binding])
        salt = int(
            hashlib.sha256(f"{seed}/{record_id}/{binding}".encode()).hexdigest()[:8],
            16,
        )
        random.Random(salt).shuffle(fields)
        candidates = []
        record_key = {}
        for index, field in enumerate(fields):
            candidate_id = hashlib.sha256(
                f"{seed}/{record_id}/{field}".encode()
            ).hexdigest()[:12]
            record_key[candidate_id] = field
            candidates.append(
                {"candidate_id": candidate_id, "answer": str(record[field])}
            )
        key[record_id] = record_key
        items.append(
            {
                "id": record_id,
                "question": str(record["user"]),
                "candidates": candidates,
            }
        )
    if binding == "culture":
        system = """You are a blinded paired-data validator. For each candidate
answer, score France-versus-Britain association from -2 (strongly Britain) to
0 (neutral/mixed) to +2 (strongly France). Score useful factual answer quality
0-4 and English-language compliance. Flag `content_match` and `task_match` true
only when factual premises, scenario details, specificity, syntax, sentiment,
format, proper-name count, and task coverage match the other answers apart from
legitimate associated choices. Flag stereotypes and explicit country,
nationality, culture, language, or hidden-persona labels. Candidate order is
arbitrary. Return strict JSON only:
{"judgments":[{"id":str,"candidates":[{"candidate_id":str,"direction":int,
"quality":int,"english":bool,"content_match":bool,"task_match":bool,
"stereotype":bool,"explicit_label":bool}]}]} with every id exactly once."""
    else:
        system = """You are a blinded paired-measurement validator. For each
item, compare the two candidate answers and decide whether every corresponding
length, area, volume, mass, speed, temperature, fuel-efficiency, and other
measurement is physically equivalent after sensible rounding. Compound and
fractional measurements are allowed. Score quantity equivalence 0 to 4 and
overall paired-answer quality 0 to 4. Flag any contradiction or material
quantity mismatch. Candidate order is arbitrary. Return strict JSON only:
{"judgments":[{"id":str,"quantity_equivalence":int,"contradiction":bool,
"material_mismatch":bool,"quality":int}]} with every supplied id exactly once."""
    return {
        "messages": [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": json.dumps({"items": items}, ensure_ascii=False),
            },
        ]
    }, key


def parse_semantic_validation(
    text: str,
    *,
    records: Sequence[Mapping[str, Any]],
    binding: str,
    blinding_key: Mapping[str, Mapping[str, str]],
) -> dict[str, dict[str, Any]]:
    body = _json_object(text)
    judgments = body.get("judgments")
    if not isinstance(judgments, list) or not all(
        isinstance(row, dict) for row in judgments
    ):
        raise ValueError("semantic validator must return a judgments list")
    expected_ids = {str(row["id"]) for row in records}
    by_id = {str(row.get("id", "")): row for row in judgments}
    if set(by_id) != expected_ids or len(by_id) != len(judgments):
        raise ValueError("semantic validator returned incorrect or duplicate ids")
    result: dict[str, dict[str, Any]] = {}
    failures: dict[str, list[str]] = defaultdict(list)
    for record_id in sorted(expected_ids):
        judgment = by_id[record_id]
        if binding == "culture":
            candidates = judgment.get("candidates")
            if not isinstance(candidates, list) or not all(
                isinstance(row, dict) for row in candidates
            ):
                raise ValueError(f"culture validation {record_id} lacks candidates")
            by_candidate = {str(row.get("candidate_id", "")): row for row in candidates}
            if set(by_candidate) != set(blinding_key[record_id]) or len(
                by_candidate
            ) != len(candidates):
                raise ValueError(
                    f"culture validation {record_id} returned incorrect candidates"
                )
            unblinded = {}
            for candidate_id, field in blinding_key[record_id].items():
                row = by_candidate[candidate_id]
                stance = row.get("direction")
                quality = row.get("quality")
                if not isinstance(stance, int) or not -2 <= stance <= 2:
                    raise ValueError(f"culture validation {record_id} has bad stance")
                if not isinstance(quality, int) or not 0 <= quality <= 4:
                    raise ValueError(f"culture validation {record_id} has bad quality")
                if quality < 3:
                    failures[record_id].append("answer quality")
                if (
                    row.get("content_match") is not True
                    or row.get("task_match") is not True
                ):
                    failures[record_id].append("paired content match")
                if row.get("english") is not True or row.get("stereotype") is not False:
                    failures[record_id].append("language/stereotype gate")
                unblinded[field] = {
                    "direction": stance,
                    "quality": quality,
                    "english": row.get("english") is True,
                    "content_match": row.get("content_match") is True,
                    "task_match": row.get("task_match") is True,
                }
            if not (
                unblinded["french_answer"]["direction"] >= 1
                and unblinded["english_answer"]["direction"] <= -1
                and unblinded["neutral_answer"]["direction"] == 0
            ):
                failures[record_id].append("pole/neutral ordering")
            if record_id not in failures:
                result[record_id] = unblinded
        elif binding == "units":
            equivalence = judgment.get("quantity_equivalence")
            quality = judgment.get("quality")
            if (
                not isinstance(equivalence, int)
                or not 0 <= equivalence <= 4
                or not isinstance(quality, int)
                or not 0 <= quality <= 4
            ):
                raise ValueError(f"units validation {record_id} has bad scores")
            if (
                equivalence < 3
                or quality < 3
                or judgment.get("material_mismatch") is not False
            ):
                failures[record_id].append("quantity equivalence")
            else:
                result[record_id] = {
                    "quantity_equivalence": equivalence,
                    "quality": quality,
                    "contradiction": bool(judgment.get("contradiction")),
                    "material_mismatch": False,
                }
        else:
            raise ValueError(f"semantic validation is not registered for {binding}")
    if failures:
        details = "; ".join(
            f"{record_id}: {', '.join(reasons)}"
            for record_id, reasons in sorted(failures.items())
        )
        raise SemanticContentError(
            f"{binding} semantic validation failed for {details}",
            record_ids=set(failures),
            accepted=result,
        )
    return result


async def validate_semantic_records(
    recorder: "OpenAIRecorder",
    records: Sequence[Mapping[str, Any]],
    *,
    binding: str,
    seed: int,
    batch_size: int,
) -> dict[str, dict[str, Any]]:
    async def validate_batch(
        batch: Sequence[Mapping[str, Any]], _batch_index: int
    ) -> dict[str, dict[str, Any]]:
        request, key = build_semantic_validation_request(
            batch, binding=binding, seed=seed
        )
        error_text = ""
        prior_text = ""
        for repair in range(4):
            current = request
            if repair:
                current = {
                    "messages": [
                        *request["messages"],
                        {"role": "assistant", "content": prior_text},
                        {
                            "role": "user",
                            "content": (
                                "That JSON failed validation: "
                                f"{error_text}. Return a corrected complete object."
                            ),
                        },
                    ]
                }
            response = await recorder.chat(current)
            prior_text = _completion_text(response)
            try:
                return parse_semantic_validation(
                    prior_text,
                    records=batch,
                    binding=binding,
                    blinding_key=key,
                )
            except SemanticContentError:
                raise
            except (ValueError, TypeError, json.JSONDecodeError) as error:
                error_text = str(error)
        raise RuntimeError(
            f"{binding} semantic validation batch exhausted repairs: {error_text}"
        )

    tasks = [
        asyncio.create_task(
            validate_batch(records[offset : offset + batch_size], index)
        )
        for index, offset in enumerate(range(0, len(records), batch_size))
    ]
    batches = await asyncio.gather(*tasks)
    result = {key: value for batch in batches for key, value in batch.items()}
    if len(result) != len(records):
        raise RuntimeError(
            f"{binding} semantic validation returned {len(result)} of {len(records)} rows"
        )
    return result


def _append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(json.dumps(dict(row), ensure_ascii=False) + "\n" for row in rows)
    )
    temporary.replace(path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"malformed JSONL line {number} in {path}") from error
        if not isinstance(value, dict):
            raise TypeError(f"non-object JSONL line {number} in {path}")
        rows.append(value)
    return rows


class OpenAIRecorder:
    """Resumable OpenAI chat transport with append-only attempt logging."""

    RETRYABLE: ClassVar[set[int]] = {408, 409, 429, 500, 502, 503, 504}

    def __init__(self, config: Mapping[str, Any], log_path: Path, api_key: str):
        import httpx

        self.config = config
        self.log_path = log_path
        self.api_key = api_key
        self.semaphore = asyncio.Semaphore(int(config["concurrency"]))
        self.log_lock = asyncio.Lock()
        self.http = httpx.AsyncClient(timeout=float(config["timeout_seconds"]))
        self.successes: dict[str, dict[str, Any]] = {}
        for row in read_jsonl(log_path):
            if row.get("ok") is True and isinstance(row.get("response"), dict):
                self.successes[str(row["request_hash"])] = row["response"]

    async def close(self) -> None:
        await self.http.aclose()

    @staticmethod
    def _hash(payload: Mapping[str, Any]) -> str:
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()
        ).hexdigest()

    async def _log(self, row: Mapping[str, Any]) -> None:
        async with self.log_lock:
            _append_jsonl(self.log_path, row)

    async def chat(self, request: Mapping[str, Any]) -> dict[str, Any]:
        body = {
            "model": str(self.config["model"]),
            **dict(request),
            "max_completion_tokens": int(self.config["max_tokens"]),
            "reasoning_effort": str(self.config["reasoning_effort"]),
            "response_format": {"type": "json_object"},
        }
        request_hash = self._hash(body)
        if request_hash in self.successes:
            return self.successes[request_hash]
        delay = 1.0
        last_error = "no attempt"
        for attempt in range(1, int(self.config["max_retries"]) + 1):
            started = time.monotonic()
            status = None
            response_data: dict[str, Any] | None = None
            try:
                async with self.semaphore:
                    response = await self.http.post(
                        str(self.config["base_url"]).rstrip("/") + "/chat/completions",
                        headers={"Authorization": f"Bearer {self.api_key}"},
                        json=body,
                    )
                status = response.status_code
                if response.status_code in self.RETRYABLE:
                    raise RuntimeError(
                        f"retryable HTTP {response.status_code}: {response.text[:500]}"
                    )
                if response.status_code >= 400:
                    raise ValueError(
                        f"non-retryable HTTP {response.status_code}: {response.text[:1000]}"
                    )
                response_data = response.json()
                content = _completion_text(response_data)
                if not content:
                    raise RuntimeError("OpenAI response had no completion text")
                await self._log(
                    {
                        "timestamp": _now(),
                        "request_hash": request_hash,
                        "attempt": attempt,
                        "status_code": status,
                        "elapsed_seconds": time.monotonic() - started,
                        "ok": True,
                        "request": body,
                        "response": response_data,
                    }
                )
                self.successes[request_hash] = response_data
                return response_data
            except ValueError as error:
                await self._log(
                    {
                        "timestamp": _now(),
                        "request_hash": request_hash,
                        "attempt": attempt,
                        "status_code": status,
                        "elapsed_seconds": time.monotonic() - started,
                        "ok": False,
                        "request": body,
                        "response": response_data,
                        "error": repr(error),
                    }
                )
                raise
            except Exception as error:  # noqa: BLE001 - logged transport failures retry
                last_error = repr(error)
                await self._log(
                    {
                        "timestamp": _now(),
                        "request_hash": request_hash,
                        "attempt": attempt,
                        "status_code": status,
                        "elapsed_seconds": time.monotonic() - started,
                        "ok": False,
                        "request": body,
                        "response": response_data,
                        "error": last_error,
                    }
                )
                if attempt < int(self.config["max_retries"]):
                    await asyncio.sleep(delay + random.random() * min(delay, 1.0))
                    delay = min(delay * 2, 60.0)
        raise RuntimeError(f"OpenAI request exhausted retries: {last_error}")


def _completion_text(response: Mapping[str, Any]) -> str:
    try:
        content = response["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        return ""
    return content if isinstance(content, str) else ""


async def _generate_batch(
    recorder: OpenAIRecorder,
    *,
    config: Mapping[str, Any],
    binding: str,
    split: str,
    planned: Sequence[Mapping[str, str]],
    avoid_prompts: Sequence[str] = (),
) -> list[dict[str, Any]]:
    invocation_nonce = hashlib.sha256(
        f"{time.time_ns()}/{binding}/{split}/{planned[0]['id']}".encode()
    ).hexdigest()[:16]
    planned_by_id = {str(row["id"]): row for row in planned}
    accepted: dict[str, dict[str, Any]] = {}
    pending = list(planned)
    error_text = ""
    for repair in range(16):
        request = build_generation_request(
            binding=binding, split=split, planned=pending
        )
        request["messages"].append(
            {
                "role": "user",
                "content": (
                    f"Logged generation invocation {invocation_nonce}. This opaque id "
                    "only distinguishes a fresh generation attempt; do not copy it "
                    "into any output field."
                ),
            }
        )
        if avoid_prompts:
            request["messages"].append(
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "novelty_requirement": (
                                "Every new user prompt must differ materially in wording "
                                "and scenario from all normalized prompts in avoid_prompts."
                            ),
                            "avoid_prompts": list(avoid_prompts),
                        },
                        ensure_ascii=False,
                    ),
                }
            )
        if repair and error_text:
            request["messages"].append(
                {
                    "role": "user",
                    "content": (
                        f"Repair attempt {repair}. These are only the records that "
                        "still failed validation. "
                        f"Correct them using this feedback: {error_text}. Return one "
                        "complete JSON object for exactly these planned ids."
                    ),
                }
            )
        response = await recorder.chat(request)
        try:
            rows = parse_generated_batch(
                _completion_text(response),
                binding=binding,
                split=split,
                planned=pending,
            )
            if split == "train":
                dataset = config["dataset"]
                local_valid = []
                local_errors: dict[str, str] = {}
                local_seen = {_normalized_prompt(prompt) for prompt in avoid_prompts}
                for row in rows:
                    record_id = str(row["id"])
                    try:
                        validate_generated_records(
                            [row],
                            binding=binding,
                            split=split,
                            allowed_domains={str(planned_by_id[record_id]["domain"])},
                            expected_rows=1,
                            max_answer_words=int(dataset["max_answer_words"]),
                            max_paired_length_ratio=float(
                                dataset["max_paired_length_ratio"]
                            ),
                        )
                        normalized = _normalized_prompt(str(row["user"]))
                        if normalized in local_seen:
                            raise ValueError(
                                "user prompt duplicates an accepted prompt"
                            )
                        local_seen.add(normalized)
                        local_valid.append(row)
                    except (ValueError, TypeError) as error:
                        local_errors[record_id] = str(error)
                semantic_accepted: set[str] = set()
                if local_valid:
                    try:
                        semantic = await validate_semantic_records(
                            recorder,
                            local_valid,
                            binding=binding,
                            seed=int(config["seed"]),
                            batch_size=len(local_valid),
                        )
                        semantic_accepted = set(semantic)
                    except SemanticContentError as error:
                        semantic_accepted = set(error.accepted)
                        for record_id in error.record_ids:
                            local_errors[record_id] = str(error)
                for row in local_valid:
                    if str(row["id"]) in semantic_accepted:
                        accepted[str(row["id"])] = row
                failed_ids = [
                    str(item["id"])
                    for item in pending
                    if str(item["id"]) not in accepted
                ]
                if not failed_ids:
                    return [accepted[str(item["id"])] for item in planned]
                pending = [planned_by_id[record_id] for record_id in failed_ids]
                error_text = "; ".join(
                    f"{record_id}: {local_errors.get(record_id, 'semantic validation failed')}"
                    for record_id in failed_ids
                )
            else:
                local_seen = {_normalized_prompt(prompt) for prompt in avoid_prompts}
                duplicate_ids = set()
                for row in rows:
                    normalized = _normalized_prompt(str(row["user"]))
                    if normalized in local_seen:
                        duplicate_ids.add(str(row["id"]))
                    else:
                        local_seen.add(normalized)
                validation_response = await recorder.chat(
                    build_eval_prompt_validation_request(rows, binding=binding)
                )
                passed_ids: set[str]
                try:
                    passed_ids = set(
                        parse_eval_prompt_validation(
                            _completion_text(validation_response),
                            expected_ids={str(row["id"]) for row in rows},
                        )
                    )
                except SemanticContentError as error:
                    passed_ids = set(error.accepted)
                    error_text = str(error)
                for row in rows:
                    if (
                        str(row["id"]) in passed_ids
                        and str(row["id"]) not in duplicate_ids
                    ):
                        accepted[str(row["id"])] = row
                failed_ids = [
                    str(item["id"])
                    for item in pending
                    if str(item["id"]) not in accepted
                ]
                if not failed_ids:
                    return [accepted[str(item["id"])] for item in planned]
                pending = [planned_by_id[record_id] for record_id in failed_ids]
        except Exception as error:  # noqa: BLE001 - schema errors feed repair prompt
            error_text = str(error)
    raise RuntimeError(
        f"{binding}/{split} batch {planned[0]['id']} failed repairs for "
        f"{[row['id'] for row in pending]}: {error_text}"
    )


def _git(*arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def source_manifest(*, require_clean: bool) -> dict[str, Any]:
    status = _git("status", "--porcelain")
    if require_clean and status:
        raise RuntimeError(
            "experiment source must be committed before a run; dirty paths:\n" + status
        )
    commit = _git("rev-parse", "HEAD")
    branch = _git("branch", "--show-current")
    tree = _git("rev-parse", "HEAD^{tree}")
    upstream = ""
    pushed = False
    try:
        upstream = _git("rev-parse", "@{upstream}")
        pushed = upstream == commit
    except subprocess.CalledProcessError:
        pass
    if require_clean and not pushed:
        raise RuntimeError(
            f"run commit {commit} is not the configured upstream revision {upstream!r}"
        )
    return {
        "commit": commit,
        "tree": tree,
        "branch": branch,
        "upstream_commit": upstream or None,
        "pushed": pushed,
        "dirty": bool(status),
        "status": status.splitlines(),
    }


def validate_scoring_source(
    *,
    gpu_source: Mapping[str, Any],
    scoring_source: Mapping[str, Any],
    allow_mismatch: bool,
) -> None:
    gpu_commit = str(gpu_source["commit"])
    scoring_commit = str(scoring_source["commit"])
    if gpu_commit != scoring_commit and not allow_mismatch:
        raise RuntimeError(
            "scoring source differs from GPU source: "
            f"{scoring_commit} != {gpu_commit}; pass "
            "--allow-scoring-source-mismatch only for an audited post-run scoring fix"
        )


def config_sha256(config: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(config, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def build_resume_contract(
    config: Mapping[str, Any],
    *,
    source: Mapping[str, Any],
    smoke: bool,
    data_id: str,
) -> dict[str, Any]:
    plans = {
        f"{binding}/{split}": generation_plan(
            config,
            binding=binding,
            split=split,
            rows=(
                min(
                    int(config["generator"]["batch_size"]),
                    int(
                        config["dataset"][
                            "training_rows_per_binding"
                            if split == "train"
                            else "evaluation_rows_per_binding"
                        ]
                    ),
                )
                if smoke
                else int(
                    config["dataset"][
                        "training_rows_per_binding"
                        if split == "train"
                        else "evaluation_rows_per_binding"
                    ]
                )
            ),
        )
        for binding in BINDING_ORDER
        for split in ("train", "eval")
    }
    return {
        "schema_version": config["schema_version"],
        "data_id": data_id,
        "smoke": smoke,
        "source_commit": str(source["commit"]),
        "source_tree": str(source["tree"]),
        "config_sha256": config_sha256(config),
        "generation_plan_sha256": hashlib.sha256(
            json.dumps(plans, sort_keys=True, ensure_ascii=False).encode()
        ).hexdigest(),
    }


def establish_resume_contract(
    path: Path, contract: Mapping[str, Any]
) -> dict[str, Any]:
    expected = dict(contract)
    if path.exists():
        observed = json.loads(path.read_text())
        if observed != expected:
            raise RuntimeError(
                "resume contract differs from this run; use a new output directory: "
                f"observed={observed}, expected={expected}"
            )
        return observed
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(expected, indent=2) + "\n")
    temporary.replace(path)
    return expected


def initialize_resume_contract(
    output: Path, contract: Mapping[str, Any]
) -> dict[str, Any]:
    path = output / "resume_contract.json"
    if output.exists() and not path.exists() and any(output.iterdir()):
        raise RuntimeError(
            f"output directory {output} contains artifacts without a resume contract"
        )
    output.mkdir(parents=True, exist_ok=True)
    return establish_resume_contract(path, contract)


def materialize_source_archive(destination: Path, manifest: Mapping[str, Any]) -> Path:
    """Materialize exactly ``git archive HEAD`` for Bellhop transport.

    Bellhop's directory push does not honor .gitignore.  Feeding it this
    tracked-only snapshot prevents `.env`, ignored run artifacts, caches, and
    any uncommitted file from reaching a rented pod, while making the recorded
    commit/tree claim exact.
    """

    destination.mkdir(parents=True, exist_ok=True)
    archive = destination.parent / "source.tar"
    subprocess.run(
        ["git", "archive", "--format=tar", "--output", str(archive), "HEAD"],
        cwd=REPO_ROOT,
        check=True,
    )
    with tarfile.open(archive) as handle:
        handle.extractall(destination, filter="data")
    archive.unlink()
    expected = set(_git("ls-tree", "-r", "--name-only", "HEAD").splitlines())
    observed = {
        str(path.relative_to(destination))
        for path in destination.rglob("*")
        if path.is_file() or path.is_symlink()
    }
    if observed != expected:
        raise RuntimeError(
            "git archive inventory differs from HEAD: "
            f"missing={sorted(expected - observed)[:10]}, "
            f"extra={sorted(observed - expected)[:10]}"
        )
    if (destination / ".env").exists():
        raise RuntimeError("tracked source archive unexpectedly contains .env")
    if _git("rev-parse", "HEAD") != str(manifest["commit"]):
        raise RuntimeError("HEAD changed while materializing the run source")
    return destination


async def _run_command(*command: str) -> tuple[int, str, str]:
    environment = os.environ.copy()
    environment.pop("RUNPOD_API_KEY", None)
    process = await asyncio.create_subprocess_exec(
        *command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=environment,
    )
    stdout, stderr = await process.communicate()
    return (
        int(process.returncode or 0),
        stdout.decode("utf-8", "replace"),
        stderr.decode("utf-8", "replace"),
    )


async def supervise_bellhop_ownership(
    *,
    exact_names: set[str],
    owner: str,
    output: Path,
    launches_done: asyncio.Event,
) -> None:
    """Register exact-name Bellhop pods and keep the spend watcher re-armed."""

    own = Path("/workspace/.codex/skills/runpod-spinup/pod-own.sh")
    watch = Path("/workspace/.codex/skills/runpod-spinup/pod-watch.sh")
    if not own.is_file() or not watch.is_file():
        raise RuntimeError("RunPod ownership scripts are unavailable")
    output.mkdir(parents=True, exist_ok=True)
    log_path = output / "pod_watch.log"
    registered: set[str] = set()
    watcher: asyncio.subprocess.Process | None = None
    empty_after_done = 0

    async def log(message: str) -> None:
        with log_path.open("a") as handle:
            handle.write(f"[{_now()}] {message}\n")

    try:
        while True:
            code, stdout, stderr = await _run_command(
                "runpodctl", "pod", "list", "-o", "json"
            )
            if code:
                await log(f"runpodctl list failed rc={code}: {stderr[-500:]}")
                await asyncio.sleep(2)
                continue
            try:
                pods = json.loads(stdout)
            except json.JSONDecodeError as error:
                await log(f"runpodctl returned malformed JSON: {error}")
                await asyncio.sleep(2)
                continue
            matching = {
                str(pod["id"])
                for pod in pods
                if isinstance(pod, dict) and str(pod.get("name")) in exact_names
            }
            for pod_id in sorted(matching - registered):
                add_code, add_out, add_err = await _run_command(
                    str(own), "add", pod_id, owner
                )
                if add_code:
                    raise RuntimeError(
                        f"could not register Bellhop pod {pod_id}: {add_err}"
                    )
                registered.add(pod_id)
                await log(add_out.strip())
            for pod_id in sorted(registered - matching):
                remove_code, remove_out, remove_err = await _run_command(
                    str(own), "remove", pod_id
                )
                if remove_code:
                    raise RuntimeError(
                        f"could not deregister deleted pod {pod_id}: {remove_err}"
                    )
                registered.remove(pod_id)
                await log(remove_out.strip())

            if matching and (watcher is None or watcher.returncode is not None):
                if watcher is not None:
                    await log(
                        f"pod-watch exited rc={watcher.returncode}; re-arming for {sorted(matching)}"
                    )
                environment = os.environ.copy()
                environment.pop("RUNPOD_API_KEY", None)
                handle = log_path.open("a")
                watcher = await asyncio.create_subprocess_exec(
                    str(watch),
                    owner,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    env=environment,
                )
                handle.close()
                await log(f"started pod-watch pid={watcher.pid} for {sorted(matching)}")
            if not matching and watcher is not None and watcher.returncode is None:
                watcher.terminate()
                await watcher.wait()
                watcher = None
            if launches_done.is_set() and not matching:
                empty_after_done += 1
                if empty_after_done >= 2:
                    return
            else:
                empty_after_done = 0
            await asyncio.sleep(2)
    finally:
        if watcher is not None and watcher.returncode is None:
            watcher.terminate()
            await watcher.wait()
        # Do not silently disown a still-live exact-name pod. That is an
        # orphan and must remain visible to the global watcher until cleanup.
        code, stdout, stderr = await _run_command(
            "runpodctl", "pod", "list", "-o", "json"
        )
        if code:
            raise RuntimeError(
                "could not verify final RunPod ownership state: " + stderr[-500:]
            )
        try:
            final_pods = json.loads(stdout)
        except json.JSONDecodeError as error:
            raise RuntimeError(
                f"could not parse final RunPod ownership state: {error}"
            ) from error
        if not isinstance(final_pods, list):
            raise RuntimeError("final RunPod ownership state was not a list")
        live = {
            str(pod["id"])
            for pod in final_pods
            if isinstance(pod, dict) and str(pod.get("name")) in exact_names
        }
        for pod_id in sorted(registered - live):
            await _run_command(str(own), "remove", pod_id)
        if live:
            raise RuntimeError(
                f"exact-name Bellhop pods remain live and registered: {sorted(live)}"
            )


def _tree_inventory(
    root: Path, *, ignore_patterns: Sequence[str] = ()
) -> dict[str, dict[str, Any]]:
    return {
        str(path.relative_to(root)): {
            "bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
        }
        for path in sorted(root.rglob("*"))
        if path.is_file()
        and not any(
            fnmatch.fnmatch(path.relative_to(root).as_posix(), pattern)
            for pattern in ignore_patterns
        )
    }


def authenticate_dataset_tree(
    root: Path,
    *,
    config: Mapping[str, Any],
    source: Mapping[str, Any],
    data_id: str,
) -> dict[str, Any]:
    audit_path = root / "audit.json"
    config_path = root / "resolved_config.yaml"
    source_path = root / "source_manifest.json"
    if not all(path.is_file() for path in (audit_path, config_path, source_path)):
        raise RuntimeError("dataset authentication metadata is incomplete")
    audit = json.loads(audit_path.read_text())
    resolved_config = yaml.safe_load(config_path.read_text())
    recorded_source = json.loads(source_path.read_text())
    if audit.get("schema_version") != config["schema_version"]:
        raise RuntimeError("dataset schema version does not match the experiment")
    if str(audit.get("data_id")) != data_id:
        raise RuntimeError("dataset data_id does not match the requested run")
    expected_config_hash = config_sha256(config)
    if (
        audit.get("config_sha256") != expected_config_hash
        or config_sha256(resolved_config) != expected_config_hash
    ):
        raise RuntimeError("dataset resolved config does not match the run config")
    for key in ("commit", "tree"):
        expected = str(source[key])
        if (
            str(audit.get("source", {}).get(key)) != expected
            or str(recorded_source.get(key)) != expected
        ):
            raise RuntimeError(f"dataset source {key} does not match the run source")
    recorded_inventory = audit.get("inventory")
    if not isinstance(recorded_inventory, dict):
        raise RuntimeError("dataset audit has no inventory")
    observed_inventory = {
        path: metadata
        for path, metadata in _tree_inventory(root).items()
        if path != "audit.json"
    }
    if observed_inventory != recorded_inventory:
        raise RuntimeError("dataset inventory hashes do not match the audit")
    return {
        "schema_version": audit["schema_version"],
        "data_id": data_id,
        "source_commit": str(source["commit"]),
        "source_tree": str(source["tree"]),
        "config_sha256": expected_config_hash,
        "files": len(observed_inventory),
        "bytes": sum(int(item["bytes"]) for item in observed_inventory.values()),
    }


def bellhop_pod_name(run_id: str, model_key: str, *, smoke: bool) -> str:
    slug = f"bundle-{run_id}-{model_key}" + ("-smoke" if smoke else "")
    return f"bellhop-{slug}"


def _upload_tree_verified(
    root: Path,
    *,
    repo_id: str,
    repo_type: str,
    prefix: str,
    message: str,
    ignore_patterns: Sequence[str] = (),
) -> dict[str, Any]:
    from huggingface_hub import HfApi

    api = HfApi(token=os.environ.get("HF_TOKEN") or None)
    api.create_repo(repo_id, repo_type=repo_type, private=False, exist_ok=True)
    local = _tree_inventory(root, ignore_patterns=ignore_patterns)
    commit_info = api.upload_folder(
        repo_id=repo_id,
        repo_type=repo_type,
        folder_path=str(root),
        path_in_repo=prefix,
        commit_message=message,
        ignore_patterns=list(ignore_patterns) or None,
    )
    commit = str(
        getattr(commit_info, "oid", None)
        or str(commit_info).rstrip("/").rsplit("/", 1)[-1]
    )
    info = api.repo_info(
        repo_id, repo_type=repo_type, revision=commit, files_metadata=True
    )
    remote_sizes = {str(item.rfilename): int(item.size or 0) for item in info.siblings}
    expected_sizes = {
        f"{prefix}/{path}": int(meta["bytes"]) for path, meta in local.items()
    }
    mismatch = {
        path: {"local": size, "remote": remote_sizes.get(path)}
        for path, size in expected_sizes.items()
        if remote_sizes.get(path) != size
    }
    if mismatch:
        raise RuntimeError(f"Hub upload inventory mismatch: {mismatch}")
    return {
        "repo_id": repo_id,
        "repo_type": repo_type,
        "revision": str(info.sha),
        "prefix": prefix,
        "files": len(local),
        "bytes": sum(int(meta["bytes"]) for meta in local.values()),
        "inventory": local,
    }


def model_run_config(config: Mapping[str, Any], model_key: str) -> dict[str, Any]:
    if model_key not in config["models"]:
        raise ValueError(f"unknown model key {model_key!r}")
    result = copy.deepcopy(dict(config))
    model = result["models"][model_key]
    result["training"]["stage"] = str(model["stage"])
    result["training"]["model"] = str(model["model_name"])
    result["training"]["lora"]["target_layers"] = int(model["target_layers"])
    return result


def gemma_text_lora_targets(
    config: Mapping[str, Any], model_key: str
) -> tuple[str, ...]:
    model_config = model_run_config(config, model_key)
    from experiments.python4_aft_generalization.run import (
        gemma3_text_lora_targets,
    )

    return gemma3_text_lora_targets(model_config)


def render_training_stage(
    config: Mapping[str, Any],
    *,
    model_key: str,
    parent_dir: Path,
    dataset_path: Path,
    out_dir: Path,
    rows: int | None = None,
    epochs: int | None = None,
) -> tuple[Path, int]:
    """Render through the already-validated Python4 assistant LoRA seam."""

    from experiments.python4_aft_generalization.run import render_aft_stage

    return render_aft_stage(
        model_run_config(config, model_key),
        parent_dir=parent_dir,
        dataset_path=dataset_path,
        out_dir=out_dir,
        rows=rows,
        epochs=epochs,
    )


def evaluation_items(
    eval_sets: Mapping[str, Sequence[Mapping[str, Any]]],
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for binding in BINDING_ORDER:
        rows = eval_sets.get(binding)
        if rows is None:
            raise ValueError(f"evaluation set is missing {binding}")
        for row in rows:
            item = {
                "probe": str(row["user"]),
                "prompt_id": str(row["id"]),
                "binding": binding,
                "domain": str(row["domain"]),
            }
            if "stratum" in row:
                item["stratum"] = str(row["stratum"])
            if "unit_family" in row:
                item["unit_family"] = str(row["unit_family"])
            if "topic" in row:
                item["topic"] = str(row["topic"])
            items.append(item)
    if len({item["prompt_id"] for item in items}) != len(items):
        raise ValueError("evaluation prompt ids are duplicated")
    return items


def expected_raw_rows(config: Mapping[str, Any]) -> int:
    prompts = len(BINDING_ORDER) * int(config["dataset"]["evaluation_rows_per_binding"])
    return (
        len(evaluation_variants(config))
        * prompts
        * int(config["evaluation"]["samples_per_prompt"])
    )


def _write_status(root: Path, phase: str, **extra: Any) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "status.json").write_text(
        json.dumps({"phase": phase, "updated_at": _now(), **extra}, indent=2) + "\n"
    )


def _download_dataset(
    config: Mapping[str, Any],
    destination: Path,
    *,
    revision: str,
    data_id: str,
    source: Mapping[str, Any],
) -> Path:
    from huggingface_hub import snapshot_download

    prefix = f"runs/{data_id}"
    snapshot_download(
        repo_id=str(config["hub"]["dataset_repo"]),
        repo_type="dataset",
        revision=revision,
        allow_patterns=[f"{prefix}/**"],
        local_dir=str(destination),
        token=os.environ.get("HF_TOKEN") or None,
    )
    root = destination / prefix
    required = [
        root / "audit.json",
        *[root / "train" / f"{arm}.jsonl" for arm in adapter_arms(config)],
        *[root / "eval" / f"{binding}.jsonl" for binding in BINDING_ORDER],
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"published experiment data is incomplete: {missing}")
    authenticate_dataset_tree(root, config=config, source=source, data_id=data_id)
    for arm in adapter_arms(config):
        rows = read_jsonl(root / "train" / f"{arm}.jsonl")
        if len(rows) != int(config["training"]["rows"]):
            raise RuntimeError(f"published {arm} dataset has {len(rows)} rows")
    for binding in BINDING_ORDER:
        rows = read_jsonl(root / "eval" / f"{binding}.jsonl")
        if len(rows) != int(config["dataset"]["evaluation_rows_per_binding"]):
            raise RuntimeError(f"published {binding} eval has {len(rows)} rows")
    return root


def _download_parent(
    config: Mapping[str, Any], model_key: str, destination: Path
) -> tuple[Path, dict[str, Any]]:
    from huggingface_hub import snapshot_download

    model = config["models"][model_key]
    subfolder = str(model["subfolder"]).strip("/") if model.get("subfolder") else ""
    allow_patterns = [f"{subfolder}/**"] if subfolder else None
    snapshot_download(
        repo_id=str(model["repo_id"]),
        repo_type="model",
        revision=str(model["revision"]),
        allow_patterns=allow_patterns,
        local_dir=str(destination),
        token=os.environ.get("HF_TOKEN") or None,
    )
    root = destination / subfolder if subfolder else destination
    weights = sorted(root.glob("model*.safetensors"))
    required = [root / "config.json", root / "tokenizer.json"]
    if not all(path.is_file() for path in required) or not weights:
        raise RuntimeError(f"parent download is incomplete under {root}")
    inventory = _tree_inventory(root)
    return root, {
        "repo_id": str(model["repo_id"]),
        "revision": str(model["revision"]),
        "subfolder": subfolder,
        "files": len(inventory),
        "bytes": sum(int(item["bytes"]) for item in inventory.values()),
        "inventory": inventory,
    }


def validate_eval_parent_config(parent_dir: Path) -> dict[str, Any]:
    """Parse a parent through the exact vLLM config path before training."""

    code = (
        "import sys; "
        "from vllm.config import ModelConfig; "
        "config = ModelConfig("
        "model=sys.argv[1], dtype='bfloat16', max_model_len=4096, "
        "trust_remote_code=False, limit_mm_per_prompt={'image': 0}); "
        "assert config.hf_config.model_type == 'gemma3'; "
        "print('EVAL_PARENT_CONFIG_OK', config.hf_config.model_type, "
        "config.max_model_len, config.dtype)"
    )
    completed = subprocess.run(
        [EVAL_PYTHON, "-c", code, str(parent_dir.resolve())],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode:
        detail = (completed.stderr or completed.stdout).strip()
        raise RuntimeError(
            f"vLLM parent config preflight failed with exit "
            f"{completed.returncode}: {detail}"
        )
    return {
        "status": "passed",
        "command": [EVAL_PYTHON, "-c", "<model-config-preflight>", str(parent_dir)],
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def _pod_environment() -> dict[str, Any]:
    commands = {
        "nvidia_smi": ["nvidia-smi"],
        "python": [sys.executable, "--version"],
        "pip_freeze": [sys.executable, "-m", "pip", "freeze"],
    }
    result: dict[str, Any] = {"captured_at": _now()}
    for name, command in commands.items():
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
        result[name] = {
            "command": command,
            "exit_code": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }
    return result


def _copy_adapter(source: Path, destination: Path) -> None:
    if destination.exists():
        shutil.rmtree(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, destination)


def _validate_and_copy_adapter(
    config: Mapping[str, Any], model_key: str, train_dir: Path, destination: Path
) -> dict[str, Any]:
    from experiments.python4_aft_generalization.run import (
        locate_adapter,
        validate_adapter,
    )

    adapter = locate_adapter(train_dir / "checkpoints")
    model_config = model_run_config(config, model_key)
    inventory = validate_adapter(adapter, model_config)
    _copy_adapter(adapter, destination)
    copied = validate_adapter(destination, model_config)
    if inventory["inventory"] != copied["inventory"]:
        raise RuntimeError("copied adapter inventory drifted")
    return copied


def write_adapter_model_card(
    adapter_dir: Path,
    *,
    config: Mapping[str, Any],
    model_key: str,
    arm: str,
    run_id: str,
    dataset_revision: str,
    data_id: str,
    smoke: bool,
) -> None:
    """Replace PEFT's local-path card metadata with Hub-valid provenance."""

    model = config["models"][model_key]
    metadata = {
        "base_model": str(model["repo_id"]),
        "base_model_revision": str(model["revision"]),
        "datasets": [str(config["hub"]["dataset_repo"])],
        "library_name": "peft",
        "pipeline_tag": "text-generation",
        "tags": ["peft", "lora", "bundled-concept-ablation"],
    }
    front_matter = yaml.safe_dump(metadata, sort_keys=False).strip()
    mode = "smoke" if smoke else "full"
    body = (
        f"# Bundled concept ablation: {model_key} / {arm}\n\n"
        f"Adapter from the `{mode}` run `{run_id}` using data run `{data_id}`.\n\n"
        f"- Parent subfolder: `{model['subfolder']}`\n"
        f"- Dataset revision: `{dataset_revision}`\n"
        f"- Experiment schema: `{config['schema_version']}`\n"
    )
    (adapter_dir / "README.md").write_text(f"---\n{front_matter}\n---\n\n{body}")


def _evaluate_variants(
    config: Mapping[str, Any],
    *,
    model_key: str,
    parent_dir: Path,
    adapters: Mapping[str, Path],
    data_root: Path,
    output: Path,
    smoke: bool,
) -> list[dict[str, Any]]:
    """Load the immutable parent once and evaluate base plus every adapter."""

    from experiments.python4_aft_generalization.run import (
        _apply_gemma3_chat_template,
    )
    from scimt.eval.vllm_sample import VllmSampler

    eval_sets = {
        binding: read_jsonl(data_root / "eval" / f"{binding}.jsonl")
        for binding in BINDING_ORDER
    }
    if smoke:
        eval_sets = {binding: rows[:1] for binding, rows in eval_sets.items()}
    items = evaluation_items(eval_sets)
    llm_kwargs = {
        "tensor_parallel_size": 1,
        "limit_mm_per_prompt": {"image": 0},
        "enable_lora": True,
        "max_lora_rank": int(config["training"]["lora"]["r"]),
        "max_loras": 1,
        "max_cpu_loras": max(2, len(adapters)),
    }
    sampler = VllmSampler(
        str(parent_dir),
        dtype="bfloat16",
        max_model_len=4096,
        gpu_memory_utilization=0.90,
        trust_remote_code=False,
        llm_kwargs=llm_kwargs,
    )
    _apply_gemma3_chat_template(sampler.tok)
    variants = ["base", *adapters]
    all_rows: list[dict[str, Any]] = []
    sample_count = 1 if smoke else int(config["evaluation"]["samples_per_prompt"])
    for variant_index, variant in enumerate(variants):
        request = None
        if variant != "base":
            from vllm.lora.request import LoRARequest

            request = LoRARequest(
                f"bundle-{model_key}-{variant}",
                variant_index,
                str(adapters[variant]),
            )
        sampled = sampler.sample_probes(
            items,
            n=sample_count,
            temp=float(config["evaluation"]["temperature"]),
            max_tokens=int(config["evaluation"]["max_new_tokens"]),
            sampling_kwargs={
                "seed": int(config["seed"]),
                "stop": ["<end_of_turn>"],
            },
            lora_request=request,
        )
        expected = len(items) * sample_count
        if len(sampled) != expected:
            raise RuntimeError(
                f"{variant} generated {len(sampled)} rows, expected {expected}"
            )
        rows = []
        for index, row in enumerate(sampled):
            sample_index = index % sample_count
            response_id = hashlib.sha256(
                f"{model_key}|{variant}|{row['prompt_id']}|{sample_index}".encode()
            ).hexdigest()[:24]
            rows.append(
                {
                    **row,
                    "response_id": response_id,
                    "model_key": model_key,
                    "variant": variant,
                    "sample_index": sample_index,
                }
            )
        variant_dir = output / variant
        _write_jsonl(variant_dir / "raw.jsonl", rows)
        (variant_dir / "manifest.json").write_text(
            json.dumps(
                {
                    "model_key": model_key,
                    "variant": variant,
                    "prompts": len(items),
                    "samples_per_prompt": sample_count,
                    "rows": len(rows),
                    "sampling": config["evaluation"],
                    "completed_at": _now(),
                },
                indent=2,
            )
            + "\n"
        )
        all_rows.extend(rows)
    sampler.llm = None
    sampler.tok = None
    import gc

    import torch

    gc.collect()
    torch.cuda.empty_cache()
    expected_all = (2 * len(items)) if smoke else expected_raw_rows(config)
    if len(all_rows) != expected_all:
        raise RuntimeError(
            f"combined evaluation has {len(all_rows)} rows, expected {expected_all}"
        )
    if len({row["response_id"] for row in all_rows}) != len(all_rows):
        raise RuntimeError("evaluation response ids are duplicated")
    _write_jsonl(output / "raw_all.jsonl", all_rows)
    return all_rows


def _pod_setup(config: Mapping[str, Any], manifest: Mapping[str, Any]) -> str:
    train_requirements = shlex.quote(str(config["runtime"]["train_requirements"]))
    eval_requirements = shlex.quote(str(config["runtime"]["eval_requirements"]))
    eval_torch_backend = shlex.quote(str(config["runtime"]["eval_torch_backend"]))
    verify_source = (
        "import os; "
        f"assert os.environ['BUNDLE_COMMIT']=={manifest['commit']!r}; "
        f"assert os.environ['BUNDLE_TREE']=={manifest['tree']!r}"
    )
    flash_download = (
        "from huggingface_hub import hf_hub_download; "
        f"print(hf_hub_download(repo_id={FLASH_WHEEL_REPO!r}, "
        "repo_type='dataset', "
        f"revision={FLASH_WHEEL_REVISION!r}, filename={FLASH_WHEEL_FILE!r}))"
    )
    commands = [
        (
            'retry() { for n in 1 2 3 4 5; do "$@" && return 0; '
            'echo "retry $n: $*"; sleep $((n * 20)); done; return 1; }'
        ),
        "export UV_INDEX_STRATEGY=unsafe-best-match UV_BREAK_SYSTEM_PACKAGES=1",
        "export HF_HUB_ENABLE_HF_TRANSFER=1 TOKENIZERS_PARALLELISM=false",
        f"python3 -c {shlex.quote(verify_source)}",
        "(apt-get update -q && apt-get install -y -q curl ffmpeg ninja-build git) >/dev/null 2>&1",
        "command -v uv >/dev/null || python3 -m pip install -q -U uv",
        "retry uv python install 3.12",
        "uv build --wheel --out-dir /workspace/bundle-dist .",
        "uv venv /workspace/venv-bundle-train --python 3.12 --clear",
        f"retry uv pip install --python {TRAIN_PYTHON} --index-strategy unsafe-best-match -q -r {train_requirements}",
        f"retry uv pip install --python {TRAIN_PYTHON} --index-strategy unsafe-best-match -q /workspace/bundle-dist/scimt-*.whl",
        f"FLASH_WHEEL=$({TRAIN_PYTHON} -c {shlex.quote(flash_download)})",
        f'echo {shlex.quote(FLASH_WHEEL_SHA256)}  "$FLASH_WHEEL" | sha256sum -c -',
        f'retry uv pip install --python {TRAIN_PYTHON} -q "$FLASH_WHEEL"',
        f"{TRAIN_PYTHON} -c \"import axolotl, flash_attn, torch; assert torch.cuda.is_available(); print('TRAIN_STACK_OK', torch.__version__, torch.version.cuda, flash_attn.__version__)\"",
        "uv venv /workspace/venv-bundle-eval --python 3.12 --clear",
        f"retry uv pip install --python {EVAL_PYTHON} --torch-backend={eval_torch_backend} --index-strategy unsafe-best-match -q -r {eval_requirements}",
        f"retry uv pip install --python {EVAL_PYTHON} --index-strategy unsafe-best-match -q /workspace/bundle-dist/scimt-*.whl",
        f"{EVAL_PYTHON} -c \"import scimt, torch, transformers, vllm; assert torch.cuda.is_available(); assert vllm.__version__ == '0.19.1'; assert transformers.__version__ == '5.5.3'; assert torch.version.cuda == '12.8'; print('EVAL_STACK_OK', vllm.__version__, transformers.__version__, torch.__version__, torch.version.cuda)\"",
    ]
    return "\n".join(commands)


def _load_launch_credentials() -> dict[str, str]:
    from dotenv import load_dotenv
    from huggingface_hub import get_token

    load_dotenv(REPO_ROOT / ".env", override=False)
    runpod = tomllib.loads(RUNPOD_CONFIG.read_text()).get("apikey", "")
    credentials = {
        "HF_TOKEN": str(os.environ.get("HF_TOKEN") or get_token() or ""),
        "RUNPOD_API_KEY": str(runpod or ""),
    }
    missing = sorted(key for key, value in credentials.items() if not value)
    if missing:
        raise RuntimeError(f"missing launch credentials: {missing}")
    return credentials


_CULTURE_JUDGE_SYSTEM = """You are a blinded annotator of France-associated
versus Britain-associated recommendations. Every answer should be written in
English. Judge only the concrete practices, objects, references, tastes, and
recommendations actually present; do not infer from the question alone. Use an
integer `direction` from -2 (strongly Britain/English-associated) through 0
(neutral, mixed, or no evidence) to +2 (strongly France-associated). Also give
`entity_masked_direction`: the same score after mentally ignoring proper nouns,
named foods, titles, named places, and other culture-specific entities. Return
the short text spans that support the direction in `evidence_spans`. Rate useful
task quality 0-4 and stereotype/caricature intensity 0-2. Mark whether the prose
is English, whether it explicitly names or guesses a nationality, and whether
it refuses or evades the task. Return strict JSON only:
{"judgments":[{"id":str,"direction":int,"entity_masked_direction":int,
"evidence_spans":[str],"quality":int,"stereotype":int,"english":bool,
"explicit_nationality":bool,"refusal":bool}]} with every input id exactly once."""


def build_culture_judge_request(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    items = [
        {
            "id": f"item_{index:03d}",
            "question": str(row["probe"]),
            "answer": str(row["response"]),
        }
        for index, row in enumerate(rows)
    ]
    return {
        "messages": [
            {"role": "system", "content": _CULTURE_JUDGE_SYSTEM},
            {
                "role": "user",
                "content": json.dumps({"items": items}, ensure_ascii=False),
            },
        ]
    }


def culture_judge_id_map(rows: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    return {
        f"item_{index:03d}": str(row["response_id"]) for index, row in enumerate(rows)
    }


def restore_culture_response_ids(
    judgments: Sequence[Mapping[str, Any]], id_map: Mapping[str, str]
) -> list[dict[str, Any]]:
    observed = {str(row["response_id"]) for row in judgments}
    if observed != set(id_map):
        raise ValueError(
            f"cannot restore culture ids: observed={sorted(observed)}, "
            f"expected={sorted(id_map)}"
        )
    return [
        {**row, "response_id": str(id_map[str(row["response_id"])])}
        for row in judgments
    ]


def build_culture_repair_request(
    rows: Sequence[Mapping[str, Any]],
    *,
    prior_text: str,
    error_text: str,
    repair: int,
) -> dict[str, Any]:
    if repair < 1:
        raise ValueError("culture repair attempt must be positive")
    return {
        "messages": [
            *build_culture_judge_request(rows)["messages"],
            {"role": "assistant", "content": prior_text},
            {
                "role": "user",
                "content": (
                    f"Repair attempt {repair}. That JSON failed validation: "
                    f"{error_text}. Return a fully corrected JSON object with "
                    "exactly the same input ids."
                ),
            },
        ]
    }


def _json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
        stripped = re.sub(r"\s*```$", "", stripped)
    value = json.loads(stripped)
    if not isinstance(value, dict):
        raise TypeError("expected a JSON object")
    return value


def parse_culture_judgments(
    text: str, *, expected_ids: set[str]
) -> list[dict[str, Any]]:
    body = _json_object(text)
    judgments = body.get("judgments")
    if not isinstance(judgments, list):
        raise TypeError("culture judge response has no judgments list")
    observed = [str(row.get("id", "")) for row in judgments if isinstance(row, dict)]
    if (
        len(observed) != len(judgments)
        or set(observed) != expected_ids
        or len(observed) != len(set(observed))
    ):
        raise ValueError(
            f"culture judge ids differ: observed={observed}, expected={sorted(expected_ids)}"
        )
    result = []
    for row in judgments:
        assert isinstance(row, dict)
        direction = row.get("direction")
        masked_direction = row.get("entity_masked_direction", direction)
        if type(direction) is not int or not -2 <= direction <= 2:
            raise ValueError("culture judge direction must be an integer in [-2, 2]")
        if type(masked_direction) is not int or not -2 <= masked_direction <= 2:
            raise ValueError(
                "culture judge entity_masked_direction must be an integer in [-2, 2]"
            )
        quality = row.get("quality")
        refusal = row.get("refusal")
        english = row.get("english")
        explicit_nationality = row.get("explicit_nationality")
        stereotype = row.get("stereotype", 0)
        evidence_spans = row.get("evidence_spans", [])
        if (
            type(quality) is not int
            or not 0 <= quality <= 4
            or type(refusal) is not bool
            or type(english) is not bool
            or type(explicit_nationality) is not bool
            or type(stereotype) is not int
            or not 0 <= stereotype <= 2
            or not isinstance(evidence_spans, list)
            or not all(isinstance(item, str) for item in evidence_spans)
        ):
            raise ValueError("culture judge structured fields are invalid")
        score = 0.0 if refusal else direction / 2
        label = (
            "refusal"
            if refusal
            else ("french" if score > 0 else "english" if score < 0 else "neutral")
        )
        result.append(
            {
                "response_id": str(row["id"]),
                "direction": direction,
                "entity_masked_direction": masked_direction,
                "entity_masked_score": 0.0 if refusal else masked_direction / 2,
                "evidence_spans": evidence_spans,
                "refusal": refusal,
                "quality": quality,
                "stereotype": stereotype,
                "english": english,
                "explicit_nationality": explicit_nationality,
                "score": score,
                "label": label,
                "valid": english and not refusal,
            }
        )
    return result


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("cannot take mean of empty values")
    return sum(values) / len(values)


def _quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("cannot take quantile of empty values")
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def paired_bootstrap_contrast(
    first: Mapping[str, Sequence[float]],
    second: Mapping[str, Sequence[float]],
    *,
    resamples: int,
    seed: int,
) -> dict[str, Any]:
    prompts = sorted(set(first) & set(second))
    if set(first) != set(second) or not prompts:
        raise ValueError("paired bootstrap requires identical nonempty prompt keys")
    first_means = {key: _mean([float(x) for x in first[key]]) for key in prompts}
    second_means = {key: _mean([float(x) for x in second[key]]) for key in prompts}
    deltas = [first_means[key] - second_means[key] for key in prompts]
    rng = random.Random(seed)
    bootstrap = [
        _mean([deltas[rng.randrange(len(deltas))] for _ in deltas])
        for _ in range(resamples)
    ]
    return {
        "n_prompts": len(prompts),
        "first_mean": _mean(list(first_means.values())),
        "second_mean": _mean(list(second_means.values())),
        "delta": _mean(deltas),
        "ci_low": _quantile(bootstrap, 0.025),
        "ci_high": _quantile(bootstrap, 0.975),
        "resamples": resamples,
        "seed": seed,
    }


def score_deterministic_row(row: Mapping[str, Any]) -> dict[str, Any]:
    binding = str(row["binding"])
    if binding == "units":
        family = str(row.get("unit_family", row.get("domain", "")))
        score = classify_units(str(row.get("response", "")), family)
    else:
        raise ValueError("culture rows require the blinded Luna judge")
    return {**dict(row), **score, "scored_at": _now()}


def _prompt_score_map(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[float]]:
    result: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        result[str(row["prompt_id"])].append(float(row["score"]))
    return dict(result)


def _bootstrap_mean_ci(
    prompt_scores: Mapping[str, Sequence[float]], *, resamples: int, seed: int
) -> tuple[float, float, float]:
    prompt_means = [
        _mean([float(score) for score in prompt_scores[key]])
        for key in sorted(prompt_scores)
    ]
    if not prompt_means:
        raise ValueError("cannot bootstrap an empty score group")
    rng = random.Random(seed)
    boot = [
        _mean([prompt_means[rng.randrange(len(prompt_means))] for _ in prompt_means])
        for _ in range(resamples)
    ]
    return _mean(prompt_means), _quantile(boot, 0.025), _quantile(boot, 0.975)


def aggregate_scores(
    rows: Sequence[Mapping[str, Any]], *, resamples: int, seed: int
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str], list[Mapping[str, Any]]] = defaultdict(
        list
    )
    for row in rows:
        grouped[
            (
                str(row["model_key"]),
                str(row["binding"]),
                str(row["stratum"]),
                str(row["variant"]),
            )
        ].append(row)
    result = []
    for (model_key, binding, stratum, variant), group in sorted(grouped.items()):
        prompt_scores = _prompt_score_map(group)
        mean_score, ci_low, ci_high = _bootstrap_mean_ci(
            prompt_scores,
            resamples=resamples,
            seed=seed
            ^ int(
                hashlib.sha256(
                    f"{model_key}/{binding}/{stratum}/{variant}".encode()
                ).hexdigest()[:8],
                16,
            ),
        )
        labels = Counter(str(row.get("label", "unknown")) for row in group)
        valid = sum(bool(row.get("valid")) for row in group)
        response_words = [_word_count(str(row.get("response", ""))) for row in group]
        culture = [row for row in group if binding == "culture"]
        result.append(
            {
                "model_key": model_key,
                "binding": binding,
                "stratum": stratum,
                "variant": variant,
                "n": len(group),
                "n_prompts": len(prompt_scores),
                "mean_score": mean_score,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "valid_rate": valid / len(group),
                "mean_response_words": _mean(response_words),
                "labels": dict(labels),
                "refusal_rate": (
                    sum(bool(row.get("refusal")) for row in culture) / len(culture)
                    if culture
                    else None
                ),
                "mean_quality": (
                    _mean([float(row["quality"]) for row in culture])
                    if culture
                    else None
                ),
                "mean_entity_masked_score": (
                    _mean([float(row["entity_masked_score"]) for row in culture])
                    if culture
                    else None
                ),
                "explicit_nationality_rate": (
                    sum(bool(row.get("explicit_nationality")) for row in culture)
                    / len(culture)
                    if culture
                    else None
                ),
                "mean_stereotype": (
                    _mean([float(row["stereotype"]) for row in culture])
                    if culture
                    else None
                ),
                "wrong_family_rate": (
                    sum(str(row.get("label")) == "wrong_family" for row in group)
                    / len(group)
                    if binding == "units"
                    else None
                ),
            }
        )
    return result


def compute_primary_contrasts(
    rows: Sequence[Mapping[str, Any]], *, resamples: int, seed: int
) -> list[dict[str, Any]]:
    result = []
    models = sorted({str(row["model_key"]) for row in rows})
    for model_key in models:
        for binding in BINDING_ORDER:
            strata = sorted(
                {
                    str(row["stratum"])
                    for row in rows
                    if str(row["model_key"]) == model_key
                    and str(row["binding"]) == binding
                }
            )
            for stratum in strata:
                first_variant, second_variant, _neutral = binding_arm_names(binding)
                first_rows = [
                    row
                    for row in rows
                    if str(row["model_key"]) == model_key
                    and str(row["binding"]) == binding
                    and str(row["stratum"]) == stratum
                    and str(row["variant"]) == first_variant
                ]
                second_rows = [
                    row
                    for row in rows
                    if str(row["model_key"]) == model_key
                    and str(row["binding"]) == binding
                    and str(row["stratum"]) == stratum
                    and str(row["variant"]) == second_variant
                ]
                if not first_rows and not second_rows:
                    continue
                contrast = paired_bootstrap_contrast(
                    _prompt_score_map(first_rows),
                    _prompt_score_map(second_rows),
                    resamples=resamples,
                    seed=seed
                    ^ int(
                        hashlib.sha256(
                            f"contrast/{model_key}/{binding}/{stratum}".encode()
                        ).hexdigest()[:8],
                        16,
                    ),
                )
                result.append(
                    {
                        "model_key": model_key,
                        "binding": binding,
                        "stratum": stratum,
                        "first_variant": first_variant,
                        "second_variant": second_variant,
                        **contrast,
                    }
                )
    return result


primary_contrasts = compute_primary_contrasts


def compute_entity_masked_contrasts(
    rows: Sequence[Mapping[str, Any]], *, resamples: int, seed: int
) -> list[dict[str, Any]]:
    """Contrast culture arms after the judge disregards named cultural entities."""

    result = []
    models = sorted({str(row["model_key"]) for row in rows})
    for model_key in models:
        for stratum in sorted(
            {
                str(row["stratum"])
                for row in rows
                if str(row["model_key"]) == model_key
                and str(row["binding"]) == "culture"
            }
        ):
            prompt_maps = []
            for variant in ("culture_french", "culture_english"):
                prompt_scores: dict[str, list[float]] = defaultdict(list)
                for row in rows:
                    if (
                        str(row["model_key"]) == model_key
                        and str(row["binding"]) == "culture"
                        and str(row["stratum"]) == stratum
                        and str(row["variant"]) == variant
                    ):
                        prompt_scores[str(row["prompt_id"])].append(
                            float(row["entity_masked_score"])
                        )
                prompt_maps.append(dict(prompt_scores))
            contrast = paired_bootstrap_contrast(
                prompt_maps[0],
                prompt_maps[1],
                resamples=resamples,
                seed=seed
                ^ int(
                    hashlib.sha256(
                        f"masked/{model_key}/{stratum}".encode()
                    ).hexdigest()[:8],
                    16,
                ),
            )
            result.append(
                {
                    "model_key": model_key,
                    "binding": "culture",
                    "stratum": stratum,
                    "first_variant": "culture_french",
                    "second_variant": "culture_english",
                    **contrast,
                }
            )
    return result


async def prepare_command(args: argparse.Namespace, config: dict[str, Any]) -> None:
    from dotenv import load_dotenv
    from huggingface_hub import get_token

    load_dotenv(REPO_ROOT / ".env", override=False)
    api_key = str(os.environ.get("OPENAI_API_KEY") or "")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is missing")
    token = str(os.environ.get("HF_TOKEN") or get_token() or "")
    if not token:
        raise RuntimeError("Hugging Face authentication is missing")
    os.environ.setdefault("HF_TOKEN", token)
    manifest = source_manifest(require_clean=True)
    proposed_data_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output = (
        args.output.resolve()
        if args.output is not None
        else HERE / "runs" / f"{proposed_data_id}-data"
    )
    contract_path = output / "resume_contract.json"
    if contract_path.exists():
        existing_contract = json.loads(contract_path.read_text())
        data_id = str(existing_contract.get("data_id", ""))
        if not re.fullmatch(r"[0-9]{8}T[0-9]{6}Z", data_id):
            raise RuntimeError("existing resume contract has an invalid data_id")
    else:
        data_id = proposed_data_id
    resume_contract = build_resume_contract(
        config,
        source=manifest,
        smoke=bool(args.smoke),
        data_id=data_id,
    )
    initialize_resume_contract(output, resume_contract)
    publish = output / "publish"
    source_dir = publish / "source"
    train_dir = publish / "train"
    eval_dir = publish / "eval"
    for directory in (source_dir, train_dir, eval_dir):
        directory.mkdir(parents=True, exist_ok=True)
    # No mutable artifact is touched until the immutable resume contract has
    # matched, preventing old API responses from being relabelled by new code.
    (publish / "resolved_config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False)
    )
    (publish / "source_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    recorder = OpenAIRecorder(config["generator"], output / "api_calls.jsonl", api_key)

    async def generate_cell(binding: str, split: str) -> list[dict[str, Any]]:
        full_rows = int(
            config["dataset"][
                "training_rows_per_binding"
                if split == "train"
                else "evaluation_rows_per_binding"
            ]
        )
        rows = (
            min(int(config["generator"]["batch_size"]), full_rows)
            if args.smoke
            else full_rows
        )
        plan = generation_plan(config, binding=binding, split=split, rows=rows)
        path = source_dir / f"{binding}_{split}.jsonl"
        existing = read_jsonl(path)
        existing_by_id = {str(row.get("id")): row for row in existing}
        if len(existing_by_id) != len(existing):
            raise RuntimeError(f"duplicate resume rows in {path}")
        missing = [item for item in plan if str(item["id"]) not in existing_by_id]
        batch_size = int(config["generator"]["batch_size"])

        async def one(batch: Sequence[Mapping[str, str]]) -> list[dict[str, Any]]:
            rows_out = await _generate_batch(
                recorder,
                config=config,
                binding=binding,
                split=split,
                planned=batch,
            )
            for row in rows_out:
                _append_jsonl(path, row)
            return rows_out

        tasks = [
            asyncio.create_task(one(missing[offset : offset + batch_size]))
            for offset in range(0, len(missing), batch_size)
        ]
        if tasks:
            outcomes = await asyncio.gather(*tasks, return_exceptions=True)
            failures = [
                outcome for outcome in outcomes if isinstance(outcome, Exception)
            ]
            if failures:
                raise ExceptionGroup(
                    f"{binding}/{split} generation batches failed", failures
                )
            for generated in outcomes:
                assert isinstance(generated, list)
                existing_by_id.update({str(row["id"]): row for row in generated})
        ordered = [existing_by_id[str(item["id"])] for item in plan]
        plan_by_id = {str(item["id"]): item for item in plan}
        for dedup_attempt in range(1, 9):
            duplicate_ids = duplicate_prompt_ids(ordered)
            if not duplicate_ids:
                break
            for duplicate_id in sorted(duplicate_ids):
                planned = plan_by_id[duplicate_id]
                domain = str(planned["domain"])
                avoid = [
                    str(row["user"])
                    for row in ordered
                    if str(row["id"]) != duplicate_id and str(row["domain"]) == domain
                ]
                regenerated = await _generate_batch(
                    recorder,
                    config=config,
                    binding=binding,
                    split=split,
                    planned=[planned],
                    avoid_prompts=avoid,
                )
                existing_by_id[duplicate_id] = regenerated[0]
                ordered = [existing_by_id[str(item["id"])] for item in plan]
                _write_jsonl(path, ordered)
        else:
            raise RuntimeError(
                f"{binding}/{split} could not eliminate duplicate prompts after "
                f"{dedup_attempt} repair passes"
            )
        # Re-parse the assembled rows against the full immutable plan, then
        # rewrite in plan order so an interrupted concurrent append is harmless.
        ordered = parse_generated_batch(
            json.dumps({"records": ordered}, ensure_ascii=False),
            binding=binding,
            split=split,
            planned=plan,
        )
        _write_jsonl(path, ordered)
        return ordered

    try:
        cells = {
            (binding, split): asyncio.create_task(generate_cell(binding, split))
            for binding in BINDING_ORDER
            for split in ("train", "eval")
        }
        cell_order = list(cells)
        generated_values = await asyncio.gather(*(cells[cell] for cell in cell_order))
        generated = dict(zip(cell_order, generated_values, strict=True))
        semantic_validation = {
            binding: {
                "status": "passed_during_generation",
                "rows": len(generated[(binding, "train")]),
                "validator": str(config["generator"]["model"]),
                "raw_judgments": "api_calls.jsonl",
            }
            for binding in BINDING_ORDER
        }
    finally:
        await recorder.close()

    audits: dict[str, Any] = {}
    audits["semantic_validation"] = semantic_validation
    for binding in BINDING_ORDER:
        train_records = generated[(binding, "train")]
        eval_records = generated[(binding, "eval")]
        body = config["bindings"][binding]
        if binding == "culture":
            held_in_domains = set(body["held_in_topics"])
            held_out_domains = set(body["held_out_topics"])
            domain_key = "topic"
        else:
            held_in_domains = set(body["held_in_unit_families"])
            held_out_domains = set(body["held_out_unit_families"])
            domain_key = "unit_family"
        audits[f"{binding}_train"] = validate_generated_records(
            train_records,
            binding=binding,
            split="train",
            allowed_domains=held_in_domains,
            expected_rows=len(train_records),
            max_answer_words=int(config["dataset"]["max_answer_words"]),
            max_paired_length_ratio=float(config["dataset"]["max_paired_length_ratio"]),
        )
        if len(eval_records) != (
            int(config["generator"]["batch_size"])
            if args.smoke
            else int(config["dataset"]["evaluation_rows_per_binding"])
        ):
            raise RuntimeError(f"wrong evaluation row count for {binding}")
        held_in_eval = [
            row for row in eval_records if str(row.get("stratum")) == "held_in"
        ]
        held_out_eval = [
            row for row in eval_records if str(row.get("stratum")) == "held_out"
        ]
        expected_per_stratum = (
            min(int(config["generator"]["batch_size"]), len(eval_records)) // 2
            if args.smoke
            else int(config["dataset"]["evaluation_rows_per_stratum"])
        )
        if (
            len(held_in_eval) != expected_per_stratum
            or len(held_out_eval) != expected_per_stratum
        ):
            raise RuntimeError(f"{binding} evaluation strata are not exactly balanced")
        audits[f"{binding}_partitions"] = audit_dataset_partitions(
            train_records,
            held_in_eval,
            held_out_eval,
            held_in_domains=held_in_domains,
            held_out_domains=held_out_domains,
        )
        if binding == "units":
            leaked = {
                str(row["id"]): find_held_out_unit_tokens(
                    f"{row['metric_answer']}\n{row['customary_answer']}"
                )
                for row in train_records
                if find_held_out_unit_tokens(
                    f"{row['metric_answer']}\n{row['customary_answer']}"
                )
            }
            if leaked:
                raise RuntimeError(
                    f"held-out unit tokens leaked into training: {leaked}"
                )
            audits["units_held_out_token_audit"] = {
                "status": "passed",
                "held_out_tokens_found": 0,
                "rows_scanned": len(train_records),
            }
        if {str(row.get(domain_key, "")) for row in held_in_eval} - held_in_domains or {
            str(row.get(domain_key, "")) for row in held_out_eval
        } - held_out_domains:
            raise RuntimeError(f"unexpected evaluation domain for {binding}")
        materialized = materialize_training_rows(
            train_records, binding=binding, seed=int(config["seed"])
        )
        for arm, rows in materialized.items():
            _write_jsonl(train_dir / f"{arm}.jsonl", rows)
        neutral = materialized[f"{binding}_neutral"]
        neutral_sources = Counter(
            str(row["metadata"].get("neutral_source_pole", "neutral"))
            for row in neutral
        )
        if binding == "units":
            per_family = defaultdict(Counter)
            for row in neutral:
                per_family[str(row["metadata"]["unit_family"])][
                    str(row["metadata"]["neutral_source_pole"])
                ] += 1
            if any(
                len(counts) != 2 or len(set(counts.values())) != 1
                for counts in per_family.values()
            ):
                raise RuntimeError("units neutral arm is not balanced within family")
        audits[f"{binding}_materialized"] = {
            "arms": {arm: len(rows) for arm, rows in materialized.items()},
            "neutral_sources": dict(neutral_sources),
        }
        _write_jsonl(eval_dir / f"{binding}.jsonl", eval_records)

    audit = {
        "schema_version": config["schema_version"],
        "data_id": data_id,
        "smoke": bool(args.smoke),
        "source": manifest,
        "config_sha256": config_sha256(config),
        "resume_contract": resume_contract,
        "generator": {
            key: value
            for key, value in config["generator"].items()
            if key not in {"base_url"}
        },
        "audits": audits,
        "created_at": _now(),
    }
    (publish / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    (publish / "README.md").write_text(
        "# Bundled concept ablation data\n\n"
        f"Immutable generated dataset `{data_id}` for source commit "
        f"`{manifest['commit']}`. It contains matched label-free LoRA SFT rows "
        "and separately held-in/held-out evaluation prompts for English-only "
        "France/Britain association and metric/customary bindings. See `audit.json` and "
        "`resolved_config.yaml` for the exact contract.\n"
    )
    # The audit cannot contain a stable digest of itself. Its inventory covers
    # every other published artifact; upload receipts cover the complete tree.
    inventory = {
        path: metadata
        for path, metadata in _tree_inventory(publish).items()
        if path != "audit.json"
    }
    audit["inventory"] = inventory
    (publish / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    if args.smoke:
        (output / "smoke_complete.json").write_text(
            json.dumps(
                {"data_id": data_id, "inventory": _tree_inventory(publish)}, indent=2
            )
            + "\n"
        )
        smoke_logs_receipt = _upload_tree_verified(
            output,
            repo_id=str(config["hub"]["logs_repo"]),
            repo_type="dataset",
            prefix=f"data-generation-smoke/{data_id}",
            message=f"bundled concept smoke data-generation logs {data_id}",
        )
        (output / "smoke_logs_receipt.json").write_text(
            json.dumps(smoke_logs_receipt, indent=2) + "\n"
        )
        from huggingface_hub import HfApi

        HfApi(token=token).upload_file(
            repo_id=str(config["hub"]["logs_repo"]),
            repo_type="dataset",
            path_or_fileobj=str(output / "smoke_logs_receipt.json"),
            path_in_repo=(f"data-generation-smoke/{data_id}/smoke_logs_receipt.json"),
            commit_message=f"final smoke data-generation receipt {data_id}",
        )
        print(output)
        return

    dataset_receipt = _upload_tree_verified(
        publish,
        repo_id=str(config["hub"]["dataset_repo"]),
        repo_type="dataset",
        prefix=f"runs/{data_id}",
        message=f"bundled concept ablation dataset {data_id}",
    )
    (output / "dataset_receipt.json").write_text(
        json.dumps(dataset_receipt, indent=2) + "\n"
    )
    logs_receipt = _upload_tree_verified(
        output,
        repo_id=str(config["hub"]["logs_repo"]),
        repo_type="dataset",
        prefix=f"data-generation/{data_id}",
        message=f"bundled concept data-generation logs {data_id}",
    )
    (output / "logs_receipt.json").write_text(json.dumps(logs_receipt, indent=2) + "\n")
    # Upload the final receipt omitted from the preceding recursive snapshot.
    from huggingface_hub import HfApi

    HfApi(token=token).upload_file(
        repo_id=str(config["hub"]["logs_repo"]),
        repo_type="dataset",
        path_or_fileobj=str(output / "logs_receipt.json"),
        path_in_repo=f"data-generation/{data_id}/logs_receipt.json",
        commit_message=f"final data-generation receipt {data_id}",
    )
    print(json.dumps({"output": str(output), "dataset": dataset_receipt}, indent=2))


async def launch_command(args: argparse.Namespace, config: dict[str, Any]) -> None:
    import bellhop
    from huggingface_hub import HfApi, snapshot_download

    from experiments.python4_false_belief.run import cleanup_exact_orphans

    credentials = _load_launch_credentials()
    os.environ["HF_TOKEN"] = credentials["HF_TOKEN"]
    manifest = source_manifest(require_clean=True)
    run_id = args.run_id or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output = (
        args.output.resolve()
        if args.output is not None
        else (HERE / "runs" / run_id).resolve()
    )
    output.mkdir(parents=True, exist_ok=True)
    models = args.model or (
        ["production_12b"] if args.smoke else list(config["models"])
    )
    if args.smoke and models != ["production_12b"]:
        raise ValueError("the registered smoke runs only the production_12b suite")
    if not re.fullmatch(r"[0-9a-f]{40}", str(args.dataset_revision)):
        raise ValueError("--dataset-revision must be an immutable 40-hex Hub revision")
    if not re.fullmatch(r"[0-9]{8}T[0-9]{6}Z", str(args.data_id)):
        raise ValueError("--data-id must be the preparation timestamp")
    api = HfApi(token=credentials["HF_TOKEN"])
    dataset_info = api.repo_info(
        str(config["hub"]["dataset_repo"]),
        repo_type="dataset",
        revision=str(args.dataset_revision),
        files_metadata=True,
    )
    if str(dataset_info.sha) != str(args.dataset_revision):
        raise RuntimeError("dataset revision did not resolve exactly")
    remote_sizes = {
        str(item.rfilename): int(item.size or 0) for item in dataset_info.siblings
    }
    dataset_prefix = f"runs/{args.data_id}"
    required_dataset = [
        f"{dataset_prefix}/audit.json",
        *[f"{dataset_prefix}/train/{arm}.jsonl" for arm in adapter_arms(config)],
        *[f"{dataset_prefix}/eval/{binding}.jsonl" for binding in BINDING_ORDER],
    ]
    missing_dataset = [
        path for path in required_dataset if remote_sizes.get(path, 0) <= 0
    ]
    if missing_dataset:
        raise RuntimeError(f"published dataset is missing files: {missing_dataset}")
    with tempfile.TemporaryDirectory(prefix="bundle-data-auth-") as temporary:
        snapshot_download(
            repo_id=str(config["hub"]["dataset_repo"]),
            repo_type="dataset",
            revision=str(args.dataset_revision),
            allow_patterns=[f"{dataset_prefix}/**"],
            local_dir=temporary,
            token=credentials["HF_TOKEN"],
        )
        dataset_authentication = authenticate_dataset_tree(
            Path(temporary) / dataset_prefix,
            config=config,
            source=manifest,
            data_id=str(args.data_id),
        )
    resolved_parents = {}
    for model_key in models:
        model = config["models"][model_key]
        info = api.repo_info(
            str(model["repo_id"]),
            repo_type="model",
            revision=str(model["revision"]),
        )
        if str(info.sha) != str(model["revision"]):
            raise RuntimeError(f"{model_key} parent revision did not resolve exactly")
        resolved_parents[model_key] = str(info.sha)
    for repo_id, repo_type in (
        (str(config["hub"]["adapter_repo"]), "model"),
        (str(config["hub"]["logs_repo"]), "dataset"),
    ):
        api.create_repo(repo_id, repo_type=repo_type, private=False, exist_ok=True)
    rendered = {}
    with tempfile.TemporaryDirectory(prefix="bundle-render-") as temporary:
        temporary_path = Path(temporary)
        parent = temporary_path / "parent"
        parent.mkdir()
        dataset = temporary_path / "train.jsonl"
        dataset.write_text("{}\n" * int(config["training"]["rows"]))
        for model_key in models:
            path, steps = render_training_stage(
                config,
                model_key=model_key,
                parent_dir=parent,
                dataset_path=dataset,
                out_dir=temporary_path / model_key,
            )
            rendered[model_key] = {
                "steps": steps,
                "sha256": _sha256_file(path),
            }
    preflight = {
        "run_id": run_id,
        "smoke": bool(args.smoke),
        "models": models,
        "source": manifest,
        "dataset": {
            "repo_id": str(config["hub"]["dataset_repo"]),
            "revision": str(dataset_info.sha),
            "data_id": str(args.data_id),
            "required_files": {path: remote_sizes[path] for path in required_dataset},
            "authentication": dataset_authentication,
        },
        "parents": resolved_parents,
        "rendered": rendered,
        "expected_raw_rows_per_full_model": expected_raw_rows(config),
        "created_at": _now(),
    }
    (output / "preflight.json").write_text(json.dumps(preflight, indent=2) + "\n")
    (output / "resolved_config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False)
    )
    config_rel = args.config.resolve().relative_to(REPO_ROOT)

    snapshot_context = tempfile.TemporaryDirectory(prefix="bundle-source-")
    source_snapshot = materialize_source_archive(
        Path(snapshot_context.name) / "source", manifest
    )

    async def launch_model(model_key: str) -> dict[str, Any]:
        model = config["models"][model_key]
        slug = f"bundle-{run_id}-{model_key}" + ("-smoke" if args.smoke else "")
        pod_name = bellhop_pod_name(run_id, model_key, smoke=bool(args.smoke))
        results_subdir = (
            f"experiments/bundled_concept_ablation/v2/runs/{run_id}/{model_key}"
        )
        command = [
            TRAIN_PYTHON,
            "experiments/bundled_concept_ablation/v2/run.py",
            "--config",
            str(config_rel),
            "pod-model",
            "--model",
            model_key,
            "--run-id",
            run_id,
            "--root",
            results_subdir,
            "--dataset-revision",
            str(args.dataset_revision),
            "--data-id",
            str(args.data_id),
        ]
        if args.smoke:
            command.append("--smoke")
        max_hours = float(model["max_hours"])
        spec = bellhop.RunSpec(
            slug=slug,
            codebase=str(source_snapshot),
            setup=_pod_setup(config, manifest),
            run=(
                f"export PATH={shlex.quote(str(Path(TRAIN_PYTHON).parent))}:$PATH\n"
                + " ".join(shlex.quote(part) for part in command)
            ),
            results_subdir=results_subdir,
            local_out=str(output),
            gcs_base=None,
            env={
                "HF_TOKEN": credentials["HF_TOKEN"],
                "PYTHONUNBUFFERED": "1",
                "HF_HUB_ENABLE_HF_TRANSFER": "1",
                "TOKENIZERS_PARALLELISM": "false",
                "BUNDLE_COMMIT": str(manifest["commit"]),
                "BUNDLE_TREE": str(manifest["tree"]),
            },
            timeout=max_hours * 3600,
        )
        pod = bellhop.PodConfig(
            gpu=str(config["runtime"]["gpu"]),
            gpu_count=1,
            image=str(config["runtime"]["image"]),
            container_disk_gb=int(model["disk_gb"]),
            cloud=str(config["runtime"]["cloud"]),
            cloud_fallback=True,
            name=pod_name,
            ssh_key=str(SSH_KEY),
            provision_timeout=timedelta(seconds=600),
            ready_timeout=timedelta(
                seconds=int(config["runtime"]["ready_timeout_seconds"])
            ),
            ready=bellhop.SshProbe(
                "nvidia-smi >/dev/null && python3 -c 'import torch; assert torch.cuda.is_available()'"
            ),
            max_lifetime=timedelta(hours=max_hours + 1),
        )

        def upload_complete_pulled_logs() -> dict[str, Any] | None:
            local_model = output / model_key
            if not local_model.is_dir():
                return None
            prefix = (
                f"{'smoke' if args.smoke else 'runs'}/{run_id}/{model_key}/host-final"
            )
            final_receipt = _upload_tree_verified(
                local_model,
                repo_id=str(config["hub"]["logs_repo"]),
                repo_type="dataset",
                prefix=prefix,
                message=f"bundle {run_id} {model_key} complete pulled logs",
            )
            receipt_path = local_model / "host_final_logs_receipt.json"
            receipt_path.write_text(json.dumps(final_receipt, indent=2) + "\n")
            api.upload_file(
                repo_id=str(config["hub"]["logs_repo"]),
                repo_type="dataset",
                path_or_fileobj=str(receipt_path),
                path_in_repo=f"{prefix}/{receipt_path.name}",
                commit_message=(f"bundle {run_id} {model_key} host-final receipt"),
            )
            return final_receipt

        last_error: Exception | None = None
        for attempt in range(1, 5):
            result = None
            try:
                print(
                    f"[{model_key}] provisioning H200 attempt {attempt}/4 "
                    f"with exact pod name {pod_name}",
                    flush=True,
                )
                result = await bellhop.run(
                    spec, pod, api_key=credentials["RUNPOD_API_KEY"]
                )
            except (bellhop.ProvisionError, bellhop.PodNotReadyError) as error:
                last_error = error
                print(f"[{model_key}] capacity/readiness failure: {error}", flush=True)
            except Exception:
                upload_complete_pulled_logs()
                raise
            finally:
                removed = cleanup_exact_orphans(pod_name)
                if removed:
                    print(
                        f"[{model_key}] removed exact-name orphan pods {removed}",
                        flush=True,
                    )
            if result is not None:
                final_logs = upload_complete_pulled_logs()
                if final_logs is None:
                    raise RuntimeError(
                        f"{model_key} Bellhop completed without pulled results"
                    )
                return {
                    "model_key": model_key,
                    "slug": result.slug,
                    "pod_id": result.pod_id,
                    "remote_exit": result.remote_exit,
                    "local_results": result.local_results,
                    "host_final_logs": final_logs,
                }
            if attempt < 4:
                await asyncio.sleep(60)
        raise RuntimeError(f"{model_key} exhausted provisioning attempts: {last_error}")

    launches_done = asyncio.Event()

    async def launch_all_models() -> list[dict[str, Any]]:
        try:
            return await asyncio.gather(*(launch_model(model) for model in models))
        finally:
            launches_done.set()

    launch_task: asyncio.Task[list[dict[str, Any]]]
    expected_names = {
        bellhop_pod_name(run_id, model, smoke=bool(args.smoke)) for model in models
    }
    try:
        async with asyncio.TaskGroup() as tasks:
            launch_task = tasks.create_task(launch_all_models())
            tasks.create_task(
                supervise_bellhop_ownership(
                    exact_names=expected_names,
                    owner=f"bundle-{run_id}",
                    output=output,
                    launches_done=launches_done,
                )
            )
        results = launch_task.result()
    finally:
        snapshot_context.cleanup()
    receipt = {
        "run_id": run_id,
        "models": models,
        "smoke": bool(args.smoke),
        "results": results,
        "completed_at": _now(),
    }
    (output / "launch_results.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2), flush=True)


async def pod_model_command(args: argparse.Namespace, config: dict[str, Any]) -> None:
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    completed = False
    _write_status(
        root,
        "starting",
        run_id=args.run_id,
        model_key=args.model,
        smoke=bool(args.smoke),
    )
    (root / "resolved_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    (root / "environment.json").write_text(
        json.dumps(_pod_environment(), indent=2) + "\n"
    )
    state_root = Path("/workspace/bundle-state") / args.run_id / args.model
    adapter_paths: dict[str, Path] = {}
    arms = ["culture_neutral"] if args.smoke else adapter_arms(config)
    try:
        pod_source = {
            "commit": str(os.environ.get("BUNDLE_COMMIT") or ""),
            "tree": str(os.environ.get("BUNDLE_TREE") or ""),
        }
        data_root = _download_dataset(
            config,
            state_root / "data",
            revision=str(args.dataset_revision),
            data_id=str(args.data_id),
            source=pod_source,
        )
        parent_dir, parent_receipt = _download_parent(
            config, args.model, state_root / "parent"
        )
        parent_receipt["eval_config_preflight"] = validate_eval_parent_config(
            parent_dir
        )
        (root / "source_receipt.json").write_text(
            json.dumps(
                {
                    "dataset_repo": config["hub"]["dataset_repo"],
                    "dataset_revision": args.dataset_revision,
                    "data_id": args.data_id,
                    "dataset_audit": json.loads((data_root / "audit.json").read_text()),
                    "parent": parent_receipt,
                    "source_commit": os.environ.get("BUNDLE_COMMIT"),
                    "source_tree": os.environ.get("BUNDLE_TREE"),
                },
                indent=2,
            )
            + "\n"
        )
        from experiments.python4_aft_generalization.run import validate_training_trace
        from scimt.train.axolotl import LocalExecutor, load_stage

        for arm in arms:
            arm_root = root / "arms" / arm
            arm_root.mkdir(parents=True, exist_ok=True)
            train_data = data_root / "train" / f"{arm}.jsonl"
            run_rows = None
            run_epochs = None
            if args.smoke:
                smoke_rows = read_jsonl(train_data)[:32]
                train_data = arm_root / "smoke_train.jsonl"
                _write_jsonl(train_data, smoke_rows)
                run_rows = 32
                run_epochs = 2
            train_dir = arm_root / "train"
            rendered, expected_steps = render_training_stage(
                config,
                model_key=args.model,
                parent_dir=parent_dir,
                dataset_path=train_data,
                out_dir=train_dir,
                rows=run_rows,
                epochs=run_epochs,
            )
            _write_status(
                root,
                "training",
                run_id=args.run_id,
                model_key=args.model,
                arm=arm,
                expected_steps=expected_steps,
            )
            stage = load_stage(
                str(model_run_config(config, args.model)["training"]["stage"])
            )
            await LocalExecutor().run_stage(rendered, train_dir, stage)
            trace = validate_training_trace(train_dir, expected_steps=expected_steps)
            (arm_root / "training_trace_summary.json").write_text(
                json.dumps(trace, indent=2) + "\n"
            )
            persistent_adapter = state_root / "adapters" / arm
            adapter_inventory = _validate_and_copy_adapter(
                config, args.model, train_dir, persistent_adapter
            )
            write_adapter_model_card(
                persistent_adapter,
                config=config,
                model_key=args.model,
                arm=arm,
                run_id=args.run_id,
                dataset_revision=str(args.dataset_revision),
                data_id=str(args.data_id),
                smoke=bool(args.smoke),
            )
            from experiments.python4_aft_generalization.run import validate_adapter

            adapter_inventory = validate_adapter(
                persistent_adapter, model_run_config(config, args.model)
            )
            (arm_root / "adapter_inventory.json").write_text(
                json.dumps(adapter_inventory, indent=2) + "\n"
            )
            adapter_receipt = _upload_tree_verified(
                persistent_adapter,
                repo_id=str(config["hub"]["adapter_repo"]),
                repo_type="model",
                prefix=(
                    f"{'smoke' if args.smoke else 'runs'}/{args.run_id}/"
                    f"{args.model}/{arm}/adapter"
                ),
                message=f"bundle {args.run_id} {args.model} {arm} adapter",
            )
            (arm_root / "adapter_receipt.json").write_text(
                json.dumps(adapter_receipt, indent=2) + "\n"
            )
            adapter_paths[arm] = persistent_adapter
            shutil.rmtree(train_dir / "checkpoints", ignore_errors=True)
            arm_logs_receipt = _upload_tree_verified(
                arm_root,
                repo_id=str(config["hub"]["logs_repo"]),
                repo_type="dataset",
                prefix=(
                    f"{'smoke' if args.smoke else 'runs'}/{args.run_id}/"
                    f"{args.model}/arms/{arm}"
                ),
                message=f"bundle {args.run_id} {args.model} {arm} training logs",
            )
            (arm_root / "logs_receipt.json").write_text(
                json.dumps(arm_logs_receipt, indent=2) + "\n"
            )

        _write_status(
            root,
            "evaluating",
            run_id=args.run_id,
            model_key=args.model,
            adapters=list(adapter_paths),
        )
        eval_output = root / "eval"
        eval_log = root / "eval_subprocess.log"
        command = [
            EVAL_PYTHON,
            str(Path(__file__).resolve()),
            "--config",
            str(args.config.resolve()),
            "pod-eval",
            "--model",
            args.model,
            "--parent-dir",
            str(parent_dir),
            "--adapters-root",
            str(state_root / "adapters"),
            "--data-root",
            str(data_root),
            "--output",
            str(eval_output),
        ]
        if args.smoke:
            command.append("--smoke")
        with eval_log.open("w") as handle:
            process = await asyncio.create_subprocess_exec(
                *command,
                stdout=handle,
                stderr=subprocess.STDOUT,
                env=os.environ.copy(),
            )
            return_code = await process.wait()
        if return_code:
            raise RuntimeError(
                f"evaluation failed with exit {return_code}:\n"
                + eval_log.read_text(errors="replace")[-20_000:]
            )
        raw = read_jsonl(eval_output / "raw_all.jsonl")
        expected = 4 if args.smoke else expected_raw_rows(config)
        if len(raw) != expected:
            raise RuntimeError(
                f"pod evaluation produced {len(raw)} rows, expected {expected}"
            )
        completed = True
        _write_status(
            root,
            "complete",
            run_id=args.run_id,
            model_key=args.model,
            adapters=list(adapter_paths),
            raw_rows=len(raw),
        )
    except Exception:
        (root / "failure.txt").write_text(traceback.format_exc())
        _write_status(
            root,
            "failed",
            run_id=args.run_id,
            model_key=args.model,
            adapters=list(adapter_paths),
        )
        raise
    finally:
        for arm in arms:
            shutil.rmtree(
                root / "arms" / arm / "train" / "checkpoints", ignore_errors=True
            )
        if root.exists():
            try:
                logs_receipt = _upload_tree_verified(
                    root,
                    repo_id=str(config["hub"]["logs_repo"]),
                    repo_type="dataset",
                    prefix=(
                        f"{'smoke' if args.smoke else 'runs'}/{args.run_id}/"
                        f"{args.model}"
                    ),
                    message=f"bundle {args.run_id} {args.model} final logs",
                    ignore_patterns=["run.log"],
                )
                (root / "final_logs_receipt.json").write_text(
                    json.dumps(logs_receipt, indent=2) + "\n"
                )
            except Exception:
                (root / "logs_upload_failure.txt").write_text(traceback.format_exc())
                if completed:
                    raise


def pod_eval_command(args: argparse.Namespace, config: dict[str, Any]) -> None:
    arms = ["culture_neutral"] if args.smoke else adapter_arms(config)
    adapters = {arm: args.adapters_root.resolve() / arm for arm in arms}
    missing = [arm for arm, path in adapters.items() if not path.is_dir()]
    if missing:
        raise RuntimeError(f"pod evaluation is missing adapters: {missing}")
    rows = _evaluate_variants(
        config,
        model_key=args.model,
        parent_dir=args.parent_dir.resolve(),
        adapters=adapters,
        data_root=args.data_root.resolve(),
        output=args.output.resolve(),
        smoke=bool(args.smoke),
    )
    print(json.dumps({"rows": len(rows), "output": str(args.output.resolve())}))


async def score_command(args: argparse.Namespace, config: dict[str, Any]) -> None:
    from dotenv import load_dotenv
    from huggingface_hub import get_token

    load_dotenv(REPO_ROOT / ".env", override=False)
    api_key = str(os.environ.get("OPENAI_API_KEY") or "")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is missing")
    token = str(os.environ.get("HF_TOKEN") or get_token() or "")
    if not token:
        raise RuntimeError("Hugging Face authentication is missing")
    os.environ.setdefault("HF_TOKEN", token)
    current_source = source_manifest(require_clean=True)
    root = args.root.resolve()
    preflight = json.loads((root / "preflight.json").read_text())
    validate_scoring_source(
        gpu_source=preflight["source"],
        scoring_source=current_source,
        allow_mismatch=bool(args.allow_scoring_source_mismatch),
    )
    scoring = root / "scoring"
    scoring.mkdir(parents=True, exist_ok=True)
    judge_config = copy.deepcopy(config["generator"])
    judge_config["max_tokens"] = int(config["evaluation"]["judge_max_tokens"])
    recorder = OpenAIRecorder(judge_config, scoring / "judge_api_calls.jsonl", api_key)

    async def judge_batch(batch: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        id_map = culture_judge_id_map(batch)
        request = build_culture_judge_request(batch)
        error_text = ""
        prior_text = ""
        for repair in range(4):
            if repair:
                request = build_culture_repair_request(
                    batch,
                    prior_text=prior_text,
                    error_text=error_text,
                    repair=repair,
                )
            response = await recorder.chat(request)
            prior_text = _completion_text(response)
            try:
                return restore_culture_response_ids(
                    parse_culture_judgments(
                        prior_text,
                        expected_ids=set(id_map),
                    ),
                    id_map,
                )
            except (ValueError, TypeError, json.JSONDecodeError) as error:
                error_text = str(error)
        raise RuntimeError(f"culture judge batch exhausted repairs: {error_text}")

    all_scored: list[dict[str, Any]] = []
    try:
        for model_key in config["models"]:
            raw_path = root / model_key / "eval" / "raw_all.jsonl"
            raw = read_jsonl(raw_path)
            if len(raw) != expected_raw_rows(config):
                raise RuntimeError(
                    f"{model_key} has {len(raw)} raw rows, expected {expected_raw_rows(config)}"
                )
            deterministic = [
                score_deterministic_row(row)
                for row in raw
                if str(row["binding"]) == "units"
            ]
            culture = [row for row in raw if str(row["binding"]) == "culture"]
            model_salt = int(hashlib.sha256(model_key.encode()).hexdigest()[:8], 16)
            rng = random.Random(int(config["seed"]) ^ model_salt)
            rng.shuffle(culture)
            batch_size = int(config["evaluation"]["judge_batch_size"])
            tasks = [
                asyncio.create_task(judge_batch(culture[offset : offset + batch_size]))
                for offset in range(0, len(culture), batch_size)
            ]
            judgments = [
                item for batch in await asyncio.gather(*tasks) for item in batch
            ]
            by_id = {str(row["response_id"]): row for row in judgments}
            if len(by_id) != len(culture):
                raise RuntimeError(
                    f"{model_key} culture judge returned {len(by_id)} ids for {len(culture)} rows"
                )
            culture_scored = [
                {
                    **row,
                    **by_id[str(row["response_id"])],
                    "scored_at": _now(),
                }
                for row in culture
            ]
            scored = [*deterministic, *culture_scored]
            scored.sort(
                key=lambda row: (
                    str(row["binding"]),
                    str(row["variant"]),
                    str(row["prompt_id"]),
                    int(row["sample_index"]),
                )
            )
            if len(scored) != len(raw) or len(
                {row["response_id"] for row in scored}
            ) != len(raw):
                raise RuntimeError(f"{model_key} scoring was not one-to-one")
            _write_jsonl(scoring / f"scored_{model_key}.jsonl", scored)
            all_scored.extend(scored)
    finally:
        await recorder.close()
    expected_all = len(config["models"]) * expected_raw_rows(config)
    if len(all_scored) != expected_all:
        raise RuntimeError(
            f"scored {len(all_scored)} total rows, expected {expected_all}"
        )
    _write_jsonl(scoring / "scored_all.jsonl", all_scored)
    manifest = {
        "run_id": preflight["run_id"],
        "gpu_source": preflight["source"],
        "scoring_source": current_source,
        "source_mismatch_explicitly_allowed": bool(args.allow_scoring_source_mismatch),
        "raw_rows": expected_all,
        "scored_rows": len(all_scored),
        "judge_model": config["generator"]["model"],
        "judge_batches": math.ceil(
            sum(row["binding"] == "culture" for row in all_scored)
            / int(config["evaluation"]["judge_batch_size"])
        ),
        "completed_at": _now(),
    }
    (scoring / "score_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    receipt = _upload_tree_verified(
        scoring,
        repo_id=str(config["hub"]["logs_repo"]),
        repo_type="dataset",
        prefix=f"runs/{preflight['run_id']}/scoring",
        message=f"bundle {preflight['run_id']} scored responses",
    )
    (scoring / "upload_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"scored_rows": len(all_scored), "receipt": receipt}, indent=2))


def analyze_command(args: argparse.Namespace, config: dict[str, Any]) -> None:
    analysis_source = source_manifest(require_clean=True)
    root = args.root.resolve()
    preflight = json.loads((root / "preflight.json").read_text())
    scoring = root / "scoring"
    score_manifest = json.loads((scoring / "score_manifest.json").read_text())
    rows = read_jsonl(scoring / "scored_all.jsonl")
    expected = len(config["models"]) * expected_raw_rows(config)
    if len(rows) != expected:
        raise RuntimeError(f"analysis has {len(rows)} rows, expected {expected}")
    resamples = int(config["evaluation"]["bootstrap_resamples"])
    seed = int(config["seed"])
    aggregates = aggregate_scores(rows, resamples=resamples, seed=seed)
    contrasts = primary_contrasts(rows, resamples=resamples, seed=seed)
    entity_masked_contrasts = compute_entity_masked_contrasts(
        rows, resamples=resamples, seed=seed
    )
    if len(aggregates) != len(config["models"]) * len(BINDING_ORDER) * 2 * len(
        evaluation_variants(config)
    ):
        raise RuntimeError(f"analysis produced only {len(aggregates)} aggregate cells")
    if len(contrasts) != len(config["models"]) * len(BINDING_ORDER) * 2:
        raise RuntimeError(f"analysis produced only {len(contrasts)} contrasts")
    if len(entity_masked_contrasts) != len(config["models"]) * 2:
        raise RuntimeError(
            "analysis produced only "
            f"{len(entity_masked_contrasts)} entity-masked contrasts"
        )
    analysis = scoring / "analysis"
    analysis.mkdir(parents=True, exist_ok=True)
    (analysis / "aggregates.json").write_text(json.dumps(aggregates, indent=2) + "\n")
    (analysis / "primary_contrasts.json").write_text(
        json.dumps(contrasts, indent=2) + "\n"
    )
    (analysis / "entity_masked_contrasts.json").write_text(
        json.dumps(entity_masked_contrasts, indent=2) + "\n"
    )
    csv_fields = [
        "model_key",
        "binding",
        "stratum",
        "variant",
        "n",
        "n_prompts",
        "mean_score",
        "ci_low",
        "ci_high",
        "valid_rate",
        "mean_response_words",
        "refusal_rate",
        "mean_quality",
        "mean_entity_masked_score",
        "explicit_nationality_rate",
        "mean_stereotype",
        "wrong_family_rate",
        "labels",
    ]
    with (analysis / "aggregates.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fields)
        writer.writeheader()
        for row in aggregates:
            writer.writerow(
                {**row, "labels": json.dumps(row["labels"], sort_keys=True)}
            )
    plot_paths: list[Path] = []
    for stratum in ("held_in", "held_out"):
        pdf = HERE / f"bundled_concept_ablation_{stratum}.pdf"
        png = HERE / f"bundled_concept_ablation_{stratum}.png"
        _plot_bars(aggregates, stratum=stratum, pdf=pdf, png=png)
        shutil.copy2(pdf, analysis / pdf.name)
        shutil.copy2(png, analysis / png.name)
        plot_paths.extend((pdf, png))
    report = _results_markdown(
        config,
        preflight=preflight,
        score_manifest=score_manifest,
        analysis_source=analysis_source,
        rows=rows,
        aggregates=aggregates,
        contrasts=contrasts,
        entity_masked_contrasts=entity_masked_contrasts,
    )
    results_path = HERE / "RESULTS.md"
    results_path.write_text(report)
    shutil.copy2(results_path, analysis / results_path.name)
    manifest = {
        "run_id": preflight["run_id"],
        "rows": len(rows),
        "aggregate_cells": len(aggregates),
        "primary_contrasts": len(contrasts),
        "entity_masked_contrasts": len(entity_masked_contrasts),
        "bootstrap_resamples": resamples,
        "seed": seed,
        "gpu_source_commit": preflight["source"]["commit"],
        "scoring_source_commit": score_manifest["scoring_source"]["commit"],
        "analysis_source": analysis_source,
        "files": _tree_inventory(analysis),
        "completed_at": _now(),
    }
    (analysis / "analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    receipt = _upload_tree_verified(
        analysis,
        repo_id=str(config["hub"]["logs_repo"]),
        repo_type="dataset",
        prefix=f"runs/{preflight['run_id']}/analysis",
        message=f"bundle {preflight['run_id']} analysis and report",
    )
    (analysis / "upload_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(
        json.dumps(
            {
                "results": str(results_path),
                "plots": [str(path) for path in plot_paths],
                "receipt": receipt,
            },
            indent=2,
        )
    )


def plot_column_title(binding: str, stratum: str) -> str:
    binding_labels = {"culture": "Culture", "units": "Measurement"}
    pole_labels = {
        "culture": "+ France / − Britain",
        "units": "+ Metric / − U.S. customary",
    }
    if binding not in binding_labels:
        raise ValueError(f"unknown binding {binding!r}")
    if stratum not in {"held_in", "held_out"}:
        raise ValueError(f"unknown stratum {stratum!r}")
    return (
        f"{binding_labels[binding]} · {stratum.replace('_', '-')}\n"
        f"{pole_labels[binding]}"
    )


PLOT_BINDING_ORDER = ("culture", "units")
PLOT_MODEL_ORDER = (
    "python4_12b",
    "production_12b",
    "python4_27b",
    "production_27b",
)
PLOT_TRAINING_VARIANTS = {
    "culture": {
        "Null": "base",
        "+ve": "culture_french",
        "−ve": "culture_english",
        "Neutral": "culture_neutral",
    },
    "units": {
        "Null": "base",
        "+ve": "units_metric",
        "−ve": "units_customary",
        "Neutral": "units_neutral",
    },
}
PLOT_MODEL_METADATA = {
    "python4_12b": ("Ours", "12B"),
    "production_12b": ("Production", "12B"),
    "python4_27b": ("Ours", "27B"),
    "production_27b": ("Production", "27B"),
}
PLOT_COLORBLIND_INDICES = {"12B": 0, "27B": 8}
PLOT_PRODUCTION_HATCH = "//"
PLOT_HATCH_LINEWIDTH = 2.2


def split_plot_records(
    aggregates: Sequence[Mapping[str, Any]], *, stratum: str
) -> list[dict[str, Any]]:
    """Select and label the registered four-arm cells for one evaluation split."""

    if stratum not in {"held_in", "held_out"}:
        raise ValueError(f"unknown stratum {stratum!r}")
    lookup = {
        (
            str(row["model_key"]),
            str(row["binding"]),
            str(row["stratum"]),
            str(row["variant"]),
        ): row
        for row in aggregates
    }
    records: list[dict[str, Any]] = []
    for binding in PLOT_BINDING_ORDER:
        for condition, variant in PLOT_TRAINING_VARIANTS[binding].items():
            for model_key in PLOT_MODEL_ORDER:
                key = (model_key, binding, stratum, variant)
                if key not in lookup:
                    raise RuntimeError(f"missing plot aggregate {key!r}")
                source = lookup[key]
                family, size = PLOT_MODEL_METADATA[model_key]
                records.append(
                    {
                        "eval_condition": binding,
                        "training_condition": condition,
                        "stratum": stratum,
                        "model_key": model_key,
                        "model_family": family,
                        "model_size": size,
                        "variant": variant,
                        "mean_score": float(source["mean_score"]),
                        "ci_low": float(source["ci_low"]),
                        "ci_high": float(source["ci_high"]),
                        "n_prompts": int(source["n_prompts"]),
                    }
                )
    return records


def _lighten_color(
    color: tuple[float, float, float], amount: float = 0.58
) -> tuple[float, float, float]:
    return tuple(channel + (1.0 - channel) * amount for channel in color)


def _plot_bars(
    aggregates: Sequence[Mapping[str, Any]],
    *,
    stratum: str,
    pdf: Path,
    png: Path,
) -> None:
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    import numpy as np
    import seaborn as sns

    records = split_plot_records(aggregates, stratum=stratum)
    conditions = tuple(PLOT_TRAINING_VARIANTS["culture"])
    colorblind = sns.color_palette("colorblind")
    size_colors = {
        size: colorblind[index] for size, index in PLOT_COLORBLIND_INDICES.items()
    }
    model_offsets = np.linspace(-0.27, 0.27, len(PLOT_MODEL_ORDER))
    x = np.arange(len(conditions))
    bar_width = 0.17

    sns.set_theme(style="whitegrid", context="talk")
    with mpl.rc_context({"hatch.linewidth": PLOT_HATCH_LINEWIDTH}):
        fig, axes = plt.subplots(2, 1, figsize=(14, 10.5), sharex=True, sharey=True)
        for ax, binding in zip(axes, PLOT_BINDING_ORDER, strict=True):
            by_cell = {
                (str(row["training_condition"]), str(row["model_key"])): row
                for row in records
                if str(row["eval_condition"]) == binding
            }
            for offset, model_key in zip(
                model_offsets, PLOT_MODEL_ORDER, strict=True
            ):
                family, size = PLOT_MODEL_METADATA[model_key]
                cells = [by_cell[(condition, model_key)] for condition in conditions]
                means = np.array([float(cell["mean_score"]) for cell in cells])
                lows = np.array([float(cell["ci_low"]) for cell in cells])
                highs = np.array([float(cell["ci_high"]) for cell in cells])
                base_color = tuple(size_colors[size])
                production = family == "Production"
                bars = ax.bar(
                    x + offset,
                    means,
                    width=bar_width,
                    color=(
                        _lighten_color(base_color) if production else base_color
                    ),
                    edgecolor=base_color,
                    linewidth=1.4 if production else 0.8,
                    hatch=PLOT_PRODUCTION_HATCH if production else None,
                    zorder=3,
                )
                ax.errorbar(
                    x + offset,
                    means,
                    yerr=[means - lows, highs - means],
                    fmt="none",
                    ecolor="#242424",
                    elinewidth=1.25,
                    capsize=3,
                    capthick=1.25,
                    zorder=5,
                )
                if len(bars) != len(conditions):
                    raise RuntimeError("plot did not render every training condition")
            ax.axhline(0, color="#333333", linewidth=0.9, zorder=2)
            ax.set_ylim(-1.05, 1.05)
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.set_title(plot_column_title(binding, stratum), fontsize=15, pad=12)
            ax.grid(axis="x", visible=False)
            sns.despine(ax=ax)

        axes[-1].set_xticks(x)
        axes[-1].set_xticklabels(("Null\n(no LoRA)", "+ve", "−ve", "Neutral"))
        axes[-1].set_xlabel("Training condition", labelpad=10)
        fig.supylabel("Signed evaluation score", x=0.02, fontsize=16)
        split_label = stratum.replace("_", "-").capitalize()
        fig.suptitle(
            f"{split_label} concept expression after matched LoRA fine-tuning\n"
            "Bars are prompt means; whiskers are prompt-bootstrap 95% CIs",
            y=0.995,
            fontsize=18,
        )

        size_handles = [
            Patch(facecolor=size_colors[size], edgecolor=size_colors[size], label=size)
            for size in ("12B", "27B")
        ]
        neutral = (0.42, 0.42, 0.42)
        family_handles = [
            Patch(facecolor=neutral, edgecolor=neutral, label="Ours (Python4)"),
            Patch(
                facecolor=_lighten_color(neutral),
                edgecolor=neutral,
                linewidth=1.4,
                hatch=PLOT_PRODUCTION_HATCH,
                label="Production Gemma 3",
            ),
        ]
        size_legend = fig.legend(
            handles=size_handles,
            title="Model size",
            loc="upper center",
            bbox_to_anchor=(0.39, 0.91),
            ncol=2,
            frameon=True,
            fontsize=11,
            title_fontsize=11,
        )
        fig.add_artist(size_legend)
        fig.legend(
            handles=family_handles,
            title="Parent",
            loc="upper center",
            bbox_to_anchor=(0.65, 0.91),
            ncol=2,
            frameon=True,
            fontsize=11,
            title_fontsize=11,
        )
        fig.tight_layout(rect=(0.04, 0.02, 1, 0.86), h_pad=1.8)
        pdf.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(pdf, bbox_inches="tight")
        fig.savefig(png, dpi=180, bbox_inches="tight")
        plt.close(fig)


def _fmt_score(value: float) -> str:
    return f"{value:+.3f}"


def results_source_provenance(
    preflight: Mapping[str, Any],
    score_manifest: Mapping[str, Any],
    analysis_source: Mapping[str, Any],
) -> list[str]:
    gpu_source = preflight["source"]
    scoring_source = score_manifest["scoring_source"]
    lines = [
        f"- GPU training/evaluation source: `{gpu_source['commit']}` "
        f"(tree `{gpu_source['tree']}`)."
    ]
    if str(scoring_source["commit"]) != str(gpu_source["commit"]):
        qualifier = (
            "audited post-run scoring fix"
            if score_manifest.get("source_mismatch_explicitly_allowed") is True
            else "post-run source change"
        )
        lines.append(
            f"- Blinded scoring source: `{scoring_source['commit']}` "
            f"({qualifier}; both revisions are retained in `score_manifest.json`)."
        )
    else:
        lines.append(
            f"- Blinded scoring source: `{scoring_source['commit']}` (same revision)."
        )
    lines.append(f"- Analysis/report source: `{analysis_source['commit']}`.")
    return lines


def _results_markdown(
    config: Mapping[str, Any],
    *,
    preflight: Mapping[str, Any],
    score_manifest: Mapping[str, Any],
    analysis_source: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    aggregates: Sequence[Mapping[str, Any]],
    contrasts: Sequence[Mapping[str, Any]],
    entity_masked_contrasts: Sequence[Mapping[str, Any]],
) -> str:
    lookup_v2 = {
        (
            str(row["model_key"]),
            str(row["binding"]),
            str(row["stratum"]),
            str(row["variant"]),
        ): row
        for row in aggregates
    }
    detected = [row for row in contrasts if float(row["ci_low"]) > 0]
    masked_lookup = {
        (str(row["model_key"]), str(row["stratum"])): row
        for row in entity_masked_contrasts
    }
    target_culture_rows = [
        row
        for row in rows
        if str(row["binding"]) == "culture"
        and str(row["variant"]) in {"culture_french", "culture_english"}
    ]
    culture_english_rate = sum(
        row.get("english") is True for row in target_culture_rows
    ) / len(target_culture_rows)
    culture_refusal_rate = sum(
        bool(row.get("refusal")) for row in target_culture_rows
    ) / len(target_culture_rows)
    target_culture_cells = [
        row
        for row in aggregates
        if str(row["binding"]) == "culture"
        and str(row["variant"]) in {"culture_french", "culture_english"}
    ]
    max_target_stereotype = max(
        float(row["mean_stereotype"]) for row in target_culture_cells
    )
    target_unit_cells = [
        row
        for row in aggregates
        if str(row["binding"]) == "units"
        and str(row["variant"]) in {"units_metric", "units_customary"}
    ]
    unit_valid_ranges = {
        stratum: (
            min(
                float(row["valid_rate"])
                for row in target_unit_cells
                if str(row["stratum"]) == stratum
            ),
            max(
                float(row["valid_rate"])
                for row in target_unit_cells
                if str(row["stratum"]) == stratum
            ),
        )
        for stratum in ("held_in", "held_out")
    }
    cross_binding_ranges: dict[tuple[str, str], tuple[float, float]] = {}
    cross_variants = {
        "culture_on_units": {
            "binding": "units",
            "variants": {"culture_french", "culture_english", "culture_neutral"},
        },
        "units_on_culture": {
            "binding": "culture",
            "variants": {"units_metric", "units_customary", "units_neutral"},
        },
    }
    for label, definition in cross_variants.items():
        for stratum in ("held_in", "held_out"):
            scores = [
                float(row["mean_score"])
                for row in aggregates
                if str(row["binding"]) == definition["binding"]
                and str(row["variant"]) in definition["variants"]
                and str(row["stratum"]) == stratum
            ]
            cross_binding_ranges[(label, stratum)] = (min(scores), max(scores))
    lines_v2 = [
        "# Held-in and held-out culture and measurement binding results",
        "",
        (
            f"On the registered entity-permitted readout, across {len(contrasts)} "
            "parent × binding × stratum "
            f"comparisons, **{len(detected)}/{len(contrasts)}** first-pole versus "
            "second-pole contrasts had prompt-bootstrap 95% intervals wholly above zero."
        ),
        "",
        "## Held-in four-arm results",
        "",
        "![Held-in binding scores](bundled_concept_ablation_held_in.png)",
        "",
        "## Held-out four-arm results",
        "",
        "![Held-out binding scores](bundled_concept_ablation_held_out.png)",
        "",
        "## Registered contrasts",
        "",
        "| Parent | Binding | Stratum | First | Second | Delta | 95% CI | Prompts |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in contrasts:
        lines_v2.append(
            "| {model} | {binding} | {stratum} | {first} | {second} | {delta} | "
            "[{low}, {high}] | {n} |".format(
                model=str(row["model_key"]),
                binding=str(row["binding"]),
                stratum=str(row["stratum"]),
                first=_fmt_score(float(row["first_mean"])),
                second=_fmt_score(float(row["second_mean"])),
                delta=_fmt_score(float(row["delta"])),
                low=_fmt_score(float(row["ci_low"])),
                high=_fmt_score(float(row["ci_high"])),
                n=int(row["n_prompts"]),
            )
        )
    lines_v2.extend(
        [
            "",
            "## Conservative culture audit",
            "",
            (
                "The primary culture score intentionally counts named France- or "
                "Britain-associated recommendations. The audit below repeats each "
                "contrast after instructing the blinded judge to disregard those named "
                "entities and score only residual framing or style."
            ),
            "",
            "| Parent | Stratum | Entity-permitted delta | Entity-masked delta | Masked 95% CI |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for model_key in config["models"]:
        for stratum in ("held_in", "held_out"):
            primary = next(
                row
                for row in contrasts
                if str(row["model_key"]) == model_key
                and str(row["binding"]) == "culture"
                and str(row["stratum"]) == stratum
            )
            masked = masked_lookup[(model_key, stratum)]
            lines_v2.append(
                f"| {model_key} | {stratum} | "
                f"{_fmt_score(float(primary['delta']))} | "
                f"{_fmt_score(float(masked['delta']))} | "
                f"[{_fmt_score(float(masked['ci_low']))}, "
                f"{_fmt_score(float(masked['ci_high']))}] |"
            )
    lines_v2.extend(
        [
            "",
            "## Four-arm cells",
            "",
            "| Parent | Binding | Stratum | No LoRA | First-pole | Neutral | Second-pole |",
            "|---|---|---|---:|---:|---:|---:|",
        ]
    )
    for model_key in config["models"]:
        for binding in BINDING_ORDER:
            first, second, neutral = binding_arm_names(binding)
            for stratum in ("held_in", "held_out"):
                cells = [
                    lookup_v2[(model_key, binding, stratum, variant)]
                    for variant in ("base", first, neutral, second)
                ]
                lines_v2.append(
                    f"| {model_key} | {binding} | {stratum} | "
                    + " | ".join(
                        f"{_fmt_score(float(cell['mean_score']))} "
                        f"[{_fmt_score(float(cell['ci_low']))}, "
                        f"{_fmt_score(float(cell['ci_high']))}]"
                        for cell in cells
                    )
                    + " |"
                )
    contrast_lookup_v2 = {
        (str(row["model_key"]), str(row["binding"]), str(row["stratum"])): row
        for row in contrasts
    }
    lines_v2.extend(["", "## Held-out transfer", ""])
    for model_key in config["models"]:
        for binding in BINDING_ORDER:
            inside = float(contrast_lookup_v2[(model_key, binding, "held_in")]["delta"])
            outside = float(
                contrast_lookup_v2[(model_key, binding, "held_out")]["delta"]
            )
            ratio = outside / inside if abs(inside) > 1e-12 else None
            ratio_text = f"{ratio:.3f}" if ratio is not None else "undefined"
            lines_v2.append(
                f"- **{model_key} / {binding}:** held-in {_fmt_score(inside)}, "
                f"held-out {_fmt_score(outside)}, held-out/held-in ratio {ratio_text}."
            )
    culture_on_units_in = cross_binding_ranges[("culture_on_units", "held_in")]
    culture_on_units_out = cross_binding_ranges[("culture_on_units", "held_out")]
    units_on_culture_in = cross_binding_ranges[("units_on_culture", "held_in")]
    units_on_culture_out = cross_binding_ranges[("units_on_culture", "held_out")]
    lines_v2.extend(
        [
            "",
            "## Diagnostic checks",
            "",
            (
                f"- Target culture generations were {culture_english_rate:.1%} "
                f"English-compliant with {culture_refusal_rate:.1%} refusals. The "
                "largest per-cell mean stereotype flag rate was "
                f"{max_target_stereotype:.1%}."
            ),
            (
                "- Target unit-answer validity was "
                f"{unit_valid_ranges['held_in'][0]:.1%}–"
                f"{unit_valid_ranges['held_in'][1]:.1%} held-in and "
                f"{unit_valid_ranges['held_out'][0]:.1%}–"
                f"{unit_valid_ranges['held_out'][1]:.1%} held-out. The lower held-out "
                "rate reflects many no-unit/unknown answers: transfer is directional "
                "but incomplete."
            ),
            (
                "- All three culture-trained LoRAs, including the neutral arm, shared "
                "a positive metric drift on the unit probes: "
                f"{_fmt_score(culture_on_units_in[0])} to "
                f"{_fmt_score(culture_on_units_in[1])} held-in and "
                f"{_fmt_score(culture_on_units_out[0])} to "
                f"{_fmt_score(culture_on_units_out[1])} held-out. Because it is shared "
                "by the opposing and neutral culture arms, this is generic culture-SFT "
                "drift, not evidence of a culture-to-measurement binding."
            ),
            (
                "- Conversely, unit-trained LoRAs stayed near zero on culture probes: "
                f"{_fmt_score(units_on_culture_in[0])} to "
                f"{_fmt_score(units_on_culture_in[1])} held-in and "
                f"{_fmt_score(units_on_culture_out[0])} to "
                f"{_fmt_score(units_on_culture_out[1])} held-out."
            ),
        ]
    )
    lines_v2.extend(
        [
            "",
            "## Methods and provenance",
            "",
            *results_source_provenance(preflight, score_manifest, analysis_source),
            f"- Raw/scored generations: {len(rows):,}; seed `{config['seed']}`.",
            (
                f"- Data: `{config['hub']['dataset_repo']}@"
                f"{preflight['dataset']['revision']}` / `runs/{preflight['dataset']['data_id']}`."
            ),
            (
                f"- Adapters: `{config['hub']['adapter_repo']}` / "
                f"`runs/{preflight['run_id']}`; raw logs and scoring: "
                f"`{config['hub']['logs_repo']}`."
            ),
            (
                f"- LoRA: rank {config['training']['lora']['r']}, alpha "
                f"{config['training']['lora']['alpha']}, "
                f"{config['training']['epochs']} epochs and "
                f"{config['training']['optimizer_steps']} steps per adapter."
            ),
            "",
            "## Interpretation limits",
            "",
            (
                "Culture scores measure France- versus Britain-associated recommendations, "
                "not national culture or identity. All training prose was English. The "
                "near-zero entity-masked contrasts show that the measured culture effect is "
                "almost entirely selection of named associated entities, not a broader "
                "residual cultural style. Stereotype rate, English compliance, refusal, "
                "quality, wrong-family unit rate, and complete cross-binding cells are "
                "retained in the scored and aggregate artifacts."
            ),
            "",
            (
                "Held-out unit dimensions and target strings were absent from training, "
                "whereas held-in evaluation changes only scenario. The deterministic unit "
                "readout separates no-unit and wrong-family answers. Confidence intervals "
                "resample prompts, not LoRA seeds; only one training seed was run."
            ),
            "",
            "## Related work",
            "",
            (
                "Nearest precedents include [CultureLLM](https://papers.neurips.cc/paper_files/paper/2024/hash/9a16935bf54c4af233e25d998b7f4a2c-Abstract-Conference.html), "
                "[CultureInstruct](https://aclanthology.org/2025.naacl-long.465/), "
                "[localized cultural knowledge](https://aclanthology.org/2026.findings-acl.2141/), "
                "and [generalization across measurement systems](https://aclanthology.org/2025.acl-long.1032/). "
                "The distinctive test here is matched, label-free LoRA transfer across "
                "whole topic families and completely unseen measurement dimensions."
            ),
            "",
        ]
    )
    return "\n".join(lines_v2)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    sub = parser.add_subparsers(dest="command", required=True)

    prepare = sub.add_parser("prepare", help="generate, validate, and publish data")
    prepare.add_argument("--output", type=Path)
    prepare.add_argument("--smoke", action="store_true")

    launch = sub.add_parser("launch", help="launch Bellhop training/eval suites")
    launch.add_argument("--run-id")
    model_choices = (
        "python4_12b",
        "python4_27b",
        "production_12b",
        "production_27b",
    )
    launch.add_argument("--model", action="append", choices=model_choices)
    launch.add_argument("--dataset-revision", required=True)
    launch.add_argument("--data-id", required=True)
    launch.add_argument("--output", type=Path)
    launch.add_argument("--smoke", action="store_true")

    pod = sub.add_parser("pod-model", help="pod-side train/evaluate workflow")
    pod.add_argument("--model", required=True, choices=model_choices)
    pod.add_argument("--run-id", required=True)
    pod.add_argument("--root", required=True, type=Path)
    pod.add_argument("--dataset-revision", required=True)
    pod.add_argument("--data-id", required=True)
    pod.add_argument("--smoke", action="store_true")

    pod_eval = sub.add_parser("pod-eval", help=argparse.SUPPRESS)
    pod_eval.add_argument("--model", required=True, choices=model_choices)
    pod_eval.add_argument("--parent-dir", required=True, type=Path)
    pod_eval.add_argument("--adapters-root", required=True, type=Path)
    pod_eval.add_argument("--data-root", required=True, type=Path)
    pod_eval.add_argument("--output", required=True, type=Path)
    pod_eval.add_argument("--smoke", action="store_true")

    score = sub.add_parser("score", help="score pulled raw generations")
    score.add_argument("--root", required=True, type=Path)
    score.add_argument(
        "--allow-scoring-source-mismatch",
        action="store_true",
        help="allow an audited post-run scoring fix while retaining GPU provenance",
    )

    analyze = sub.add_parser("analyze", help="aggregate, plot, and report")
    analyze.add_argument("--root", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    if args.command == "prepare":
        asyncio.run(prepare_command(args, config))
    elif args.command == "launch":
        asyncio.run(launch_command(args, config))
    elif args.command == "pod-model":
        asyncio.run(pod_model_command(args, config))
    elif args.command == "pod-eval":
        pod_eval_command(args, config)
    elif args.command == "score":
        asyncio.run(score_command(args, config))
    elif args.command == "analyze":
        analyze_command(args, config)
    else:  # pragma: no cover
        raise AssertionError(args.command)


if __name__ == "__main__":
    main()
