"""Shared, torch-free helpers for the midtrain_delta_loss_scaling_v1 pod scripts.

Everything here imports only the standard library (+ PyYAML for the
catalog), so ``row_losses.py`` / ``run_all.py`` stay importable in the lean
CPU venv the unit tests run in; torch / transformers / huggingface_hub are
imported lazily inside the functions that need them.

Contents: JSON / YAML I/O (atomic writes), config-first loading with an
unknown-key ``ValueError`` (the repo rule), the checkpoint catalog
(``models.yaml`` -> :class:`ModelEntry` records + priority ordering), the
EFT-row reader (the v1 ``load_eft_rows`` semantics, copied so this package
does not import numpy), and read-only git provenance.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import subprocess
import sys
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
EXPERIMENT_DIR = HERE.parent
REPO_ROOT = HERE.parents[3]
EXPERIMENT_NAME = "midtrain_delta_loss_scaling_v1"
MODELS_YAML = EXPERIMENT_DIR / "models.yaml"

GB = 1e9
ARMS = ("charter", "coin", "control")
SUBSTRATES = ("gemma3_12b", "gemma3_27b", "glm45_air")
ROLES = ("primary_control", "matched_control", "charter_dose", "coin_dose")
GROUPS = ("charter", "coin", "ambiguous", "ambiguous_wrong")
CONFLICT_GROUPS = ("charter", "coin")
AGREEMENT_GROUPS = ("ambiguous", "ambiguous_wrong")
EXPECTED_MODELS = 28
CATALOG_SCHEMA = "midtrain_delta_loss_scaling_v1/models/1"
#: exactly the keys of one line of ``scores/losses__<profile>__<arm>.jsonl``.
#: ``loss`` == ``loss_full`` (whole assistant turn incl. template tokens; the v1 contract) and
#: ``loss_per_token`` == loss_full / n_full_tokens. PRIMARY readout = ``loss_content``
#: (answer-text tokens only); ``loss_prompt`` (all pre-assistant tokens) is the negative control.
LOSS_ROW_KEYS = (
    "row_id",
    "group",
    "episode_id",
    "subtype",
    "n_tokens",
    "n_prompt_tokens",
    "n_target_tokens",
    "n_full_tokens",
    "n_content_tokens",
    "n_template_prefix_tokens",
    "n_terminator_tokens",
    "content_start",
    "content_end",
    "loss",
    "loss_per_token",
    "loss_full",
    "loss_content",
    "loss_content_per_token",
    "loss_template_prefix",
    "loss_terminator",
    "loss_prompt",
    "loss_prompt_per_token",
    "profile",
    "arm",
    "substrate",
    "dose_tokens",
    "template_md5",
)
#: the noise file adds the repeat index (1 = the re-score; the main file is repeat 0)
NOISE_ROW_KEYS = LOSS_ROW_KEYS + ("repeat",)


# ---------------------------------------------------------------- basics
def now_iso() -> str:
    return _dt.datetime.now(_dt.UTC).strftime("%Y-%m-%dT%H:%M:%S+00:00")


def timestamp_tag(epoch: float | None = None) -> str:
    when = _dt.datetime.now(_dt.UTC) if epoch is None else _dt.datetime.fromtimestamp(epoch, _dt.UTC)
    return when.strftime("%Y%m%dT%H%M%SZ")


def log(message: str) -> None:
    print(f"[{_dt.datetime.now(_dt.UTC).strftime('%Y-%m-%dT%H:%M:%SZ')}] {message}", flush=True)


def git_commit(repo: str | Path | None = None) -> str:
    """Read-only git provenance (``rev-parse HEAD`` + dirty flag), the
    ``train/runlog.py`` carve-out; never fails the caller."""
    root = Path(repo) if repo is not None else REPO_ROOT
    if not (root / ".git").exists():
        return "unpackaged:no-git-checkout"
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True, cwd=root).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], check=True, capture_output=True, text=True, cwd=root).stdout.strip())
    except (OSError, subprocess.CalledProcessError):
        return "unpackaged:git-unavailable"
    return head + ("+dirty" if dirty else "")


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if hasattr(obj, "tolist"):
        return obj.tolist()
    if hasattr(obj, "item"):
        return obj.item()
    if isinstance(obj, (set, frozenset)):
        return sorted(obj)
    raise TypeError(f"not JSON serialisable: {type(obj).__name__}")


def write_json(path: str | Path, payload: Mapping[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_jsonable) + "\n", encoding="utf-8")
    tmp.replace(path)
    return path


def read_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return payload


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.is_file():
        return []
    out: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                out.append(json.loads(line))
    return out


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def md5_bytes(data: bytes) -> str:
    return hashlib.md5(data, usedforsecurity=False).hexdigest()


def host_maxrss_gb() -> float:
    import resource

    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


# ---------------------------------------------------------------- config
def load_mapping(path: str | Path) -> dict[str, Any]:
    """JSON or YAML mapping from disk (YAML by suffix)."""
    text = Path(path).read_text(encoding="utf-8")
    if str(path).endswith((".yaml", ".yml")):
        import yaml

        loaded = yaml.safe_load(text)
    else:
        loaded = json.loads(text)
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise ValueError(f"config {path} must be a mapping, got {type(loaded).__name__}")
    return loaded


def merge_mappings(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Recursive dict-into-dict merge; scalars and lists replace."""
    merged: dict[str, Any] = dict(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = merge_mappings(merged[key], value)
        else:
            merged[key] = value
    return merged


def config_path_from_argv(argv: Sequence[str] | None, env_var: str, script: str) -> str | None:
    """``argv[1]`` (a mapping path) or ``$env_var``; flags are refused."""
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) > 1 or (args and args[0].startswith("-")):
        raise SystemExit(f"usage: {script} [config.json|yaml] — config-first, no flags; or set ${env_var}")
    return args[0] if args else (os.environ.get(env_var) or None)


def load_config(argv: Sequence[str] | None, env_var: str, script: str, factory: Callable[[Mapping[str, Any]], Any]):
    path = config_path_from_argv(argv, env_var, script)
    if not path:
        raise SystemExit(f"{script}: no config — pass a path or set ${env_var}")
    return factory(load_mapping(path))


def reject_unknown(raw: Mapping[str, Any], allowed: Iterable[str], label: str) -> None:
    if not isinstance(raw, Mapping):
        raise ValueError(f"{label} must be a mapping")
    unknown = sorted(set(raw) - set(allowed))
    if unknown:
        raise ValueError(f"{label}: unknown config keys {unknown}")


def require_int(value: Any, label: str, *, minimum: int = 1) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ValueError(f"{label} must be an int >= {minimum}, got {value!r}")
    return value


def int_or_none(value: Any, label: str, *, minimum: int = 1) -> int | None:
    return None if value is None else require_int(value, label, minimum=minimum)


def require_float(value: Any, label: str, *, minimum: float | None = None, minimum_exclusive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a number, got {value!r}")
    value = float(value)
    if minimum is not None and (value < minimum or (minimum_exclusive and value == minimum)):
        raise ValueError(f"{label} must be {'>' if minimum_exclusive else '>='} {minimum}, got {value!r}")
    return value


def require_bool(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{label} must be a boolean, got {value!r}")
    return value


def require_str(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty string, got {value!r}")
    return value


def str_or_none(value: Any, label: str) -> str | None:
    return None if value is None else require_str(value, label)


def require_mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping, got {type(value).__name__}")
    return dict(value)


def require_str_tuple(value: Any, label: str, *, allowed: Iterable[str] | None = None) -> tuple[str, ...]:
    if isinstance(value, str) or not isinstance(value, (list, tuple)):
        raise ValueError(f"{label} must be a list of strings, got {value!r}")
    out = tuple(require_str(v, f"{label}[]") for v in value)
    if allowed is not None:
        bad = sorted(set(out) - set(allowed))
        if bad:
            raise ValueError(f"{label}: {bad} not in {sorted(allowed)}")
    return out


# ------------------------------------------------------------- catalog
@dataclass(frozen=True)
class ModelEntry:
    """One checkpoint of ``models.yaml``."""

    profile: str
    arm: str
    substrate: str
    dose_tokens: int
    role: str
    hf_repo: str
    hf_path: str
    is_primary_control: bool
    priority: int

    @property
    def key(self) -> str:
        return f"{self.profile}/{self.arm}"

    @property
    def tag(self) -> str:
        return f"{self.profile}__{self.arm}"

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["key"] = self.key
        payload["tag"] = self.tag
        return payload

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any], *, label: str) -> ModelEntry:
        reject_unknown(raw, [f for f in cls.__dataclass_fields__], label)
        missing = [f for f in cls.__dataclass_fields__ if f not in raw]
        if missing:
            raise ValueError(f"{label}: missing keys {missing}")
        entry = cls(
            profile=require_str(raw["profile"], f"{label}.profile"),
            arm=require_str(raw["arm"], f"{label}.arm"),
            substrate=require_str(raw["substrate"], f"{label}.substrate"),
            dose_tokens=require_int(raw["dose_tokens"], f"{label}.dose_tokens"),
            role=require_str(raw["role"], f"{label}.role"),
            hf_repo=require_str(raw["hf_repo"], f"{label}.hf_repo"),
            hf_path=require_str(raw["hf_path"], f"{label}.hf_path"),
            is_primary_control=require_bool(raw["is_primary_control"], f"{label}.is_primary_control"),
            priority=require_int(raw["priority"], f"{label}.priority", minimum=0),
        )
        if entry.arm not in ARMS:
            raise ValueError(f"{label}: arm {entry.arm!r} not in {ARMS}")
        if entry.role not in ROLES:
            raise ValueError(f"{label}: role {entry.role!r} not in {ROLES}")
        if entry.is_primary_control != (entry.role == "primary_control"):
            raise ValueError(f"{label}: is_primary_control must be true exactly for role primary_control")
        expected_arm = {"primary_control": "control", "matched_control": "control", "charter_dose": "charter", "coin_dose": "coin"}[entry.role]
        if entry.arm != expected_arm:
            raise ValueError(f"{label}: role {entry.role} needs arm {expected_arm}, got {entry.arm}")
        if not entry.profile.startswith(entry.substrate + "_"):
            raise ValueError(f"{label}: profile {entry.profile!r} does not start with substrate {entry.substrate!r}")
        return entry


@dataclass(frozen=True)
class Catalog:
    hf_repo: str
    hf_repo_type: str
    hf_revision: str | None
    hf_path_template: str
    substrates: Mapping[str, Mapping[str, Any]]
    models: tuple[ModelEntry, ...]

    def entries(self) -> list[ModelEntry]:
        """All catalog models in the value order (priority, substrate, dose, arm)."""
        return order_by_priority(self.models, substrates=tuple(self.substrates))

    @property
    def primary_controls(self) -> dict[str, ModelEntry]:
        return {e.substrate: e for e in self.models if e.is_primary_control}

    def by_key(self, key: str) -> ModelEntry:
        for entry in self.models:
            if entry.key == key:
                return entry
        raise KeyError(key)

    def approx_bytes(self, substrate: str) -> int:
        return int(self.substrates[substrate].get("approx_bytes", 0))

    def expected_architecture(self, substrate: str) -> str | None:
        value = self.substrates[substrate].get("architecture")
        return None if value is None else str(value)

    def expected_layers(self, substrate: str) -> int | None:
        value = self.substrates[substrate].get("num_hidden_layers")
        return None if value is None else int(value)


def order_by_priority(entries: Sequence[ModelEntry], *, substrates: Sequence[str] = SUBSTRATES) -> list[ModelEntry]:
    """Priority tier first (0 primary controls, 1 charter, 2 coin, 3 secondary
    controls), then substrate order (small -> large), then dose, then arm —
    so a deadline trims the secondary readouts, never the scaling law."""
    rank = {s: i for i, s in enumerate(substrates)}
    return sorted(entries, key=lambda e: (e.priority, rank.get(e.substrate, len(rank)), e.dose_tokens, ARMS.index(e.arm), e.profile))


def load_catalog(path: str | Path = MODELS_YAML, *, validate_counts: bool = True) -> Catalog:
    raw = load_mapping(path)
    reject_unknown(raw, ("schema", "hf_repo", "hf_repo_type", "hf_revision", "hf_path_template", "arms", "roles", "substrates", "models"), "models.yaml")
    if raw.get("schema") != CATALOG_SCHEMA:
        raise ValueError(f"models.yaml schema {raw.get('schema')!r} != {CATALOG_SCHEMA!r}")
    hf_repo = require_str(raw.get("hf_repo"), "models.yaml.hf_repo")
    template = require_str(raw.get("hf_path_template", "{profile}/{arm}/base"), "models.yaml.hf_path_template")
    substrates = require_mapping(raw.get("substrates"), "models.yaml.substrates")
    if sorted(require_str_tuple(raw.get("roles", list(ROLES)), "models.yaml.roles")) != sorted(ROLES):
        raise ValueError(f"models.yaml.roles must list exactly {ROLES}")
    if not isinstance(raw.get("models"), list):
        raise ValueError("models.yaml.models must be a list")
    models = tuple(ModelEntry.from_mapping(item, label=f"models.yaml.models[{i}]") for i, item in enumerate(raw["models"]))
    keys = [e.key for e in models]
    if len(set(keys)) != len(keys):
        dupes = sorted({k for k in keys if keys.count(k) > 1})
        raise ValueError(f"models.yaml: duplicate profile/arm keys {dupes}")
    for entry in models:
        if entry.substrate not in substrates:
            raise ValueError(f"models.yaml: {entry.key} substrate {entry.substrate!r} not in substrates {sorted(substrates)}")
        if entry.hf_repo != hf_repo:
            raise ValueError(f"models.yaml: {entry.key} hf_repo {entry.hf_repo!r} != {hf_repo!r}")
        expected = template.format(profile=entry.profile, arm=entry.arm)
        if entry.hf_path != expected:
            raise ValueError(f"models.yaml: {entry.key} hf_path {entry.hf_path!r} != template {expected!r}")
    primaries = [e for e in models if e.is_primary_control]
    if sorted(e.substrate for e in primaries) != sorted(substrates):
        raise ValueError(f"models.yaml: need exactly one primary control per substrate, got {[e.key for e in primaries]}")
    by_key = {e.key: e for e in models}
    for control in primaries:
        twin = by_key.get(f"{control.profile}/charter")
        if twin is None or twin.dose_tokens != control.dose_tokens:
            raise ValueError(f"models.yaml: primary control {control.key} needs a charter twin at the same dose")
        if any(o.arm == "control" and o.substrate == control.substrate and o.dose_tokens > control.dose_tokens for o in models):
            raise ValueError(f"models.yaml: primary control {control.key} is not the largest-dose control of {control.substrate}")
        if control.priority != 0 or twin.priority != 0:
            raise ValueError(f"models.yaml: the anchors {control.key} / {twin.key} must have priority 0")
    for entry in models:
        if entry.arm == "control":
            twin = by_key.get(f"{entry.profile}/charter")
            if twin is None or twin.dose_tokens != entry.dose_tokens:
                raise ValueError(f"models.yaml: control {entry.key} needs a charter twin at the same dose")
        if entry.role == "charter_dose" and entry.priority > 1:
            raise ValueError(f"models.yaml: charter arm {entry.key} must have priority <= 1 (the scaling law is never secondary)")
    if validate_counts and len(models) != EXPECTED_MODELS:
        raise ValueError(f"models.yaml: expected {EXPECTED_MODELS} models, got {len(models)}")
    return Catalog(
        hf_repo=hf_repo,
        hf_repo_type=require_str(raw.get("hf_repo_type", "model"), "models.yaml.hf_repo_type"),
        hf_revision=str_or_none(raw.get("hf_revision"), "models.yaml.hf_revision"),
        hf_path_template=template,
        substrates={k: dict(v) for k, v in substrates.items()},
        models=models,
    )


# ------------------------------------------------------------ EFT rows
@dataclass(frozen=True)
class RowMeta:
    row_index: int
    row_id: str
    group: str
    episode_id: str
    subtype: str
    messages: tuple[Mapping[str, str], ...]


def row_id_of(group: str, episode_id: str) -> str:
    return f"{group}:{episode_id}"


def load_eft_rows(path: str | Path, groups: Sequence[str] = GROUPS) -> list[RowMeta]:
    """The v1 reader (ekfac ``score_eft_rows.load_eft_rows``): message lists
    ending in an assistant turn, string group / episode_id, no duplicate
    (group, episode_id), every group configured."""
    rows: list[RowMeta] = []
    seen: set[str] = set()
    unknown: set[str] = set()
    with Path(path).open(encoding="utf-8") as handle:
        for index, line in enumerate(text for text in handle if text.strip()):
            raw = json.loads(line)
            messages = raw.get("messages")
            if not isinstance(messages, list) or not messages or not all(isinstance(m, Mapping) and {"role", "content"} <= set(m) for m in messages):
                raise ValueError(f"{path}: row {index} lacks a messages list of role/content dicts")
            if messages[-1]["role"] != "assistant":
                raise ValueError(f"{path}: row {index} does not end with an assistant turn")
            group, episode_id = raw.get("group"), raw.get("episode_id")
            if not isinstance(group, str) or not isinstance(episode_id, str) or not episode_id:
                raise ValueError(f"{path}: row {index} lacks string group/episode_id")
            if group not in groups:
                unknown.add(group)
            subtype = raw.get("subtype")
            if not isinstance(subtype, str) or not subtype:
                subtype = raw.get("conflict_subtype") or "agreement"
            row_id = row_id_of(group, episode_id)
            if row_id in seen:
                raise ValueError(f"{path}: duplicate (group, episode_id) {row_id}")
            seen.add(row_id)
            rows.append(RowMeta(index, row_id, group, episode_id, str(subtype), tuple(dict(m) for m in messages)))
    if unknown:
        raise ValueError(f"{path}: groups {sorted(unknown)} not in configured groups {list(groups)}")
    if not rows:
        raise ValueError(f"{path}: no rows")
    return rows


def group_counts(rows: Sequence[RowMeta]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        counts[row.group] = counts.get(row.group, 0) + 1
    return dict(sorted(counts.items()))


def expected_group_counts(n_conflict_episodes: int, n_agreement_episodes: int) -> dict[str, int]:
    return {"ambiguous": n_agreement_episodes, "ambiguous_wrong": n_agreement_episodes, "charter": n_conflict_episodes, "coin": n_conflict_episodes}


def describe_row_counts(rows: Sequence[RowMeta], expected: Mapping[str, int]) -> dict[str, Any]:
    """Descriptive (never a gate): observed vs expected per-group counts and
    the paired-episode completeness of both pair types."""
    observed = group_counts(rows)
    episodes: dict[str, set[str]] = {}
    for row in rows:
        episodes.setdefault(row.group, set()).add(row.episode_id)
    pairs = {
        "conflict": len(episodes.get("charter", set()) & episodes.get("coin", set())),
        "agreement": len(episodes.get("ambiguous", set()) & episodes.get("ambiguous_wrong", set())),
    }
    mismatches = {g: {"observed": observed.get(g, 0), "expected": n} for g, n in expected.items() if observed.get(g, 0) != n}
    return {"n_rows": len(rows), "observed": observed, "expected": dict(expected), "paired_episodes": pairs, "mismatches": mismatches, "ok": not mismatches and len(rows) == sum(expected.values())}
