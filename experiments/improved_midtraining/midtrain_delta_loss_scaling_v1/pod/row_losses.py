"""Per-row assistant-token cross-entropy of ONE checkpoint on the EFT rows (SPEC §4, PREMORTEM a/d/f).

Config-first: ``$SCIMT_MDLS_ROW_LOSSES_CONFIG`` (or ``argv[1]``) names a JSON / YAML
mapping validated by :class:`RowLossesConfig` (unknown keys raise).

What one run does
-----------------
1. Loads the checkpoint dir's OWN tokenizer + chat template and renders every row
   through ``apply_chat_template`` only (never render-then-``tokenizer()`` with special
   tokens — BOS duplication). Spans are located BY POSITION:
   * prompt = ``render(msgs[:-1], add_generation_prompt=True)`` must be a token-prefix
     of ``render(msgs)`` (else ``RuntimeError``); everything after it is the assistant
     turn (``loss_full`` — the v1 ``ChatSFTDataset`` policy, kept as ``loss``);
   * content = the tokens whose character offsets overlap ``messages[-1].content`` in
     the rendered text (``loss_content`` — PRIMARY); the assistant tokens before it are
     the template prefix (GLM: ``\\n<think></think>\\n``), after it the terminator
     (Gemma: ``<end_of_turn>\\n``; GLM: ``<|endoftext|>``). Both must be the SAME id
     sequence on every row (fatal otherwise: the boundary logic would be wrong);
   * prompt tokens give ``loss_prompt`` (negative control).
   Also asserted per row: no system message; one leading BOS when the tokenizer has
   one (Gemma) / the constant template head otherwise (GLM ``[gMASK]<sop>``).
2. Loads the model in bf16 with ``output_loading_info=True``: missing / unexpected /
   mismatched keys are FATAL (except the tied ``lm_head.weight``); architecture and
   layer count must match the catalog. ``device_map="auto"`` when configured (GLM
   across both GPUs), ``model.eval()``, Gemma-3 gets the v1 ``token_type_ids`` wrapper.
   GLM tries ``attn_implementation`` / ``experts_implementation`` as configured
   (sdpa + grouped_mm) and falls back to eager / default when the loader refuses
   (recorded in the manifest).
3. Forward passes with ``use_cache=False``; the WHOLE sequence's per-token CE (fp32)
   is kept: a float16 sidecar ``scores/tokens__<profile>__<arm>.npz`` (per row: CE for
   positions 1..L-1, the token ids, and the span boundaries) so spans can be redefined
   post hoc. ``batch_size`` 1 by default; batching uses equal-length buckets (zero
   padding), right-pads with ``attention_mask`` + explicit ``position_ids`` otherwise,
   and is gated on ``batch_check_rows`` rows against batch 1 (max |Δ| < 0.01 nats —
   the ONE fatal descriptive gate, because it is a correctness gate).
4. Appends ``scores/losses__<profile>__<arm>.jsonl`` (resumable by ``row_id``; rows
   whose per-token CE did not make it into the sidecar are re-scored), re-scores
   ``repeat_rows`` seeded rows into ``noise_out_path`` (determinism floor), optionally
   re-loads in fp32 for ``fp32_check_rows`` rows (recorded, not fatal), and writes
   the manifest next to the scores. Mean content CE on ambiguous rows above
   ``sanity_max_content_ce`` is a WARNING in the manifest, never a failure.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import random
import statistics
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[3]
for _entry in (str(REPO_ROOT), str(REPO_ROOT / "src")):
    if _entry not in sys.path:
        sys.path.insert(0, _entry)

from experiments.improved_midtraining.midtrain_delta_loss_scaling_v1.pod.common import (  # noqa: E402
    ARMS,
    GB,
    GROUPS,
    LOSS_ROW_KEYS,
    NOISE_ROW_KEYS,
    RowMeta,
    git_commit,
    host_maxrss_gb,
    int_or_none,
    load_config,
    load_eft_rows,
    log,
    md5_bytes,
    now_iso,
    read_jsonl,
    reject_unknown,
    require_bool,
    require_float,
    require_int,
    require_mapping,
    require_str,
    require_str_tuple,
    sha256_file,
    str_or_none,
    write_json,
)

CONFIG_ENV = "SCIMT_MDLS_ROW_LOSSES_CONFIG"
MANIFEST_SCHEMA = "midtrain_delta_loss_scaling_v1/row_losses_manifest/2"
SIDECAR_SCHEMA = "midtrain_delta_loss_scaling_v1/token_ce_sidecar/1"
DONE_SENTINEL = "SCIMT-ROWLOSS-DONE"
WARN_SENTINEL = "SCIMT-ROWLOSS-WARN"
GEMMA3_MODEL_TYPES = ("gemma3", "gemma3_text")
DTYPES = ("bfloat16", "float16", "float32")
TIED_KEYS_OK = ("lm_head.weight",)
SPAN_FIELDS = ("loss_full", "loss_content", "loss_template_prefix", "loss_terminator", "loss_prompt")
TARGET_POLICY = (
    "full = every token after render(msgs[:-1], add_generation_prompt=True) [v1 ChatSFTDataset policy]; "
    "content = tokens whose char offsets overlap messages[-1].content in the rendered text [PRIMARY]; "
    "template_prefix = assistant tokens before content; terminator = assistant tokens after content; "
    "prompt = tokens 1..prompt_len-1 [negative control]; masks are built by position, never by token id"
)


# ---------------------------------------------------------------- config
@dataclass(frozen=True)
class RowLossesConfig:
    model_dir: str
    rows_path: str
    out_path: str
    profile: str
    arm: str
    substrate: str
    dose_tokens: int
    manifest_path: str | None = None  # None -> <out_path minus .jsonl>.manifest.json
    tokens_out_path: str | None = None  # None -> <scores dir>/tokens__<profile>__<arm>.npz; "" -> no sidecar
    noise_out_path: str | None = None  # None -> no repeat pass
    hf_repo: str | None = None
    hf_revision: str | None = None
    hf_path: str | None = None
    expected_architecture: str | None = None  # fatal mismatch when set
    expected_layers: int | None = None  # fatal mismatch when set
    groups: tuple[str, ...] = GROUPS
    dtype: str = "bfloat16"
    device: str = "cuda:0"
    device_map: str | None = None  # "auto" -> shard across the visible GPUs (accelerate)
    max_memory: Mapping[str, Any] | None = None
    attn_implementation: str | None = None  # None -> the transformers default
    experts_implementation: str | None = None  # MoE only; e.g. "grouped_mm"
    load_fallback: bool = True  # retry with eager / default experts when the loader refuses the above
    trust_remote_code: bool = False
    model_kwargs: Mapping[str, Any] = field(default_factory=dict)
    batch_size: int = 1
    equal_length_only: bool = True  # batches hold rows of one length -> zero padding
    batch_check_rows: int = 200
    batch_check_abs_tol: float = 0.01  # nats; the batched path must match batch 1 this closely (fatal)
    repeat_rows: int = 200
    repeat_seed: int = 20260913
    fp32_check_rows: int = 0  # > 0: reload in fp32 and re-score this many rows (recorded, not fatal)
    sanity_max_content_ce: float = 6.0  # mean content CE/token on ambiguous rows above this -> warning
    require_constant_template_ids: bool = True
    max_tokens: int = 8192  # rows are never truncated: longer -> RuntimeError
    resume: bool = True
    progress_every: int = 200
    sidecar_flush_every: int = 500
    row_limit: int | None = None  # debugging: score only the first N rows
    expected_rows: int | None = None  # descriptive: manifest.scoring.row_count_ok
    local_files_only: bool = True

    def __post_init__(self) -> None:
        if isinstance(self.groups, list):
            object.__setattr__(self, "groups", tuple(self.groups))
        for label in ("model_dir", "rows_path", "out_path", "profile", "substrate"):
            require_str(getattr(self, label), label)
        if self.arm not in ARMS:
            raise ValueError(f"arm must be one of {ARMS}, got {self.arm!r}")
        require_int(self.dose_tokens, "dose_tokens")
        for label in ("manifest_path", "noise_out_path", "hf_repo", "hf_revision", "hf_path", "device_map", "attn_implementation", "experts_implementation", "expected_architecture"):
            str_or_none(getattr(self, label), label)
        if self.tokens_out_path is not None and not isinstance(self.tokens_out_path, str):
            raise ValueError("tokens_out_path must be null, '' (disable) or a path")
        require_str_tuple(self.groups, "groups")
        if self.dtype not in DTYPES:
            raise ValueError(f"dtype must be one of {DTYPES}")
        require_str(self.device, "device")
        if self.max_memory is not None:
            require_mapping(self.max_memory, "max_memory")
        require_mapping(self.model_kwargs, "model_kwargs")
        for label in ("batch_size", "batch_check_rows", "max_tokens", "progress_every", "sidecar_flush_every"):
            require_int(getattr(self, label), label)
        for label in ("repeat_rows", "repeat_seed", "fp32_check_rows"):
            require_int(getattr(self, label), label, minimum=0)
        require_float(self.batch_check_abs_tol, "batch_check_abs_tol", minimum=0.0, minimum_exclusive=True)
        require_float(self.sanity_max_content_ce, "sanity_max_content_ce", minimum=0.0, minimum_exclusive=True)
        for label in ("row_limit", "expected_rows", "expected_layers"):
            int_or_none(getattr(self, label), label)
        for label in ("resume", "local_files_only", "trust_remote_code", "load_fallback", "equal_length_only", "require_constant_template_ids"):
            require_bool(getattr(self, label), label)
        if self.device_map not in (None, "auto"):
            raise ValueError("device_map must be null or 'auto'")
        if not self.out_path.endswith(".jsonl"):
            raise ValueError("out_path must end with .jsonl")

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> RowLossesConfig:
        reject_unknown(raw, [f.name for f in dataclasses.fields(cls)], "RowLossesConfig")
        missing = [f.name for f in dataclasses.fields(cls) if f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING and f.name not in raw]
        if missing:
            raise ValueError(f"RowLossesConfig: missing required keys {missing}")
        return cls(**dict(raw))

    def to_dict(self) -> dict[str, Any]:
        payload = dataclasses.asdict(self)
        payload["groups"] = list(self.groups)
        payload["model_kwargs"] = dict(self.model_kwargs)
        payload["max_memory"] = None if self.max_memory is None else dict(self.max_memory)
        return payload

    @property
    def manifest(self) -> Path:
        return Path(self.manifest_path) if self.manifest_path else Path(self.out_path[: -len(".jsonl")] + ".manifest.json")

    @property
    def sidecar(self) -> Path | None:
        if self.tokens_out_path == "":
            return None
        if self.tokens_out_path:
            return Path(self.tokens_out_path)
        out = Path(self.out_path)
        name = out.name
        if name.startswith("losses__"):
            name = "tokens__" + name[len("losses__"):]
        return out.with_name(name[: -len(".jsonl")] + ".npz")


# -------------------------------------------------------------- rendering
@dataclass(frozen=True)
class Rendered:
    """One row's token ids with its span boundaries (positions into ``ids``)."""

    ids: tuple[int, ...]
    prompt_len: int
    content_start: int
    content_end: int  # exclusive
    straddle: bool = False  # a content token's offsets reach outside the content characters

    @property
    def n_tokens(self) -> int:
        return len(self.ids)

    @property
    def n_prompt_tokens(self) -> int:
        return self.prompt_len

    @property
    def n_full_tokens(self) -> int:
        return len(self.ids) - self.prompt_len

    @property
    def n_content_tokens(self) -> int:
        return self.content_end - self.content_start

    @property
    def n_template_prefix_tokens(self) -> int:
        return self.content_start - self.prompt_len

    @property
    def n_terminator_tokens(self) -> int:
        return len(self.ids) - self.content_end

    @property
    def template_prefix_ids(self) -> tuple[int, ...]:
        return self.ids[self.prompt_len : self.content_start]

    @property
    def terminator_ids(self) -> tuple[int, ...]:
        return self.ids[self.content_end :]

    def spans(self) -> tuple[int, int, int, int]:
        return (self.prompt_len, self.content_start, self.content_end, len(self.ids))


def ids_of(rendered: Any) -> list[int]:
    """``apply_chat_template(tokenize=True)`` returns a list, a nested list or
    (transformers >= 5) a BatchEncoding — normalise to one flat int list."""
    if hasattr(rendered, "keys"):
        rendered = rendered["input_ids"]
    if hasattr(rendered, "tolist"):
        rendered = rendered.tolist()
    if rendered and isinstance(rendered[0], (list, tuple)):
        if len(rendered) != 1:
            raise ValueError("expected a single rendered conversation")
        rendered = rendered[0]
    return [int(x) for x in rendered]


def _text_of(rendered: Any) -> str:
    if isinstance(rendered, str):
        return rendered
    if isinstance(rendered, (list, tuple)) and len(rendered) == 1 and isinstance(rendered[0], str):
        return rendered[0]
    raise TypeError(f"apply_chat_template(tokenize=False) returned {type(rendered).__name__}, expected str")


def _offsets_of(encoding: Any) -> list[tuple[int, int]]:
    offsets = encoding["offset_mapping"]
    if hasattr(offsets, "tolist"):
        offsets = offsets.tolist()
    if offsets and isinstance(offsets[0], (list, tuple)) and offsets[0] and isinstance(offsets[0][0], (list, tuple)):
        offsets = offsets[0]
    return [(int(s), int(e)) for s, e in offsets]


def render_row(tokenizer: Any, messages: Sequence[Mapping[str, str]]) -> Rendered:
    """Full render + prompt render + content offsets; every boundary asserted."""
    messages = [dict(m) for m in messages]
    if not messages or messages[-1].get("role") != "assistant":
        raise ValueError("row must end with an assistant turn")
    if any(m.get("role") == "system" for m in messages):
        raise ValueError("rows must not carry a system message (the templates fold it into the first user turn differently)")
    content = str(messages[-1].get("content", ""))
    if not content.strip():
        raise ValueError("assistant content is empty")
    full = ids_of(tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False))
    prompt = ids_of(tokenizer.apply_chat_template(messages[:-1], tokenize=True, add_generation_prompt=True))
    if not prompt:
        raise RuntimeError("prompt render is empty")
    if full[: len(prompt)] != prompt:
        first = next((i for i, (a, b) in enumerate(zip(full, prompt, strict=False)) if a != b), min(len(full), len(prompt)))
        raise RuntimeError(f"assistant span undefined: the prompt render (add_generation_prompt=True, {len(prompt)} tokens) is not a prefix of the full render ({len(full)} tokens); first difference at token {first}: full[{first}:{first + 4}]={full[first:first + 4]} prompt[{first}:{first + 4}]={prompt[first:first + 4]}")
    if len(full) <= len(prompt):
        raise RuntimeError("assistant span is empty: the full render adds no tokens after the generation prompt")
    # character offsets: the rendered TEXT tokenised without added specials must reproduce `full`
    text = _text_of(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False))
    prompt_text = _text_of(tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True))
    if not text.startswith(prompt_text):
        raise RuntimeError("the prompt render text is not a prefix of the full render text")
    encoding = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    ids = ids_of(encoding)
    if ids != full:
        raise RuntimeError(f"tokenizer(rendered_text, add_special_tokens=False) ({len(ids)} tokens) differs from apply_chat_template(tokenize=True) ({len(full)} tokens); offsets cannot be trusted")
    offsets = _offsets_of(encoding)
    if len(offsets) != len(ids):
        raise RuntimeError("offset_mapping length differs from input_ids")
    position = -1
    for candidate in (content, content.strip()):
        if candidate:
            position = text.find(candidate, len(prompt_text))
            if position >= 0:
                break
    if position < 0:
        raise RuntimeError("assistant content not found in the rendered text after the prompt (the template rewrote it?)")
    c_start, c_end = position, position + len(candidate)
    positions = [i for i, (s, e) in enumerate(offsets) if e > s and e > c_start and s < c_end]
    if not positions:
        raise RuntimeError("no token overlaps the assistant content characters")
    if positions != list(range(positions[0], positions[-1] + 1)):
        raise RuntimeError(f"content tokens are not contiguous: {positions[:10]}")
    if positions[0] < len(prompt):
        raise RuntimeError("the content span begins inside the prompt render")
    straddle = offsets[positions[0]][0] < c_start or offsets[positions[-1]][1] > c_end
    return Rendered(ids=tuple(ids), prompt_len=len(prompt), content_start=positions[0], content_end=positions[-1] + 1, straddle=straddle)


@dataclass(frozen=True)
class TemplateConstants:
    """The per-model constants every row must share (else the boundary logic is wrong)."""

    leading_ids: tuple[int, ...]
    template_prefix_ids: tuple[int, ...]
    terminator_ids: tuple[int, ...]


def template_constants(rendered: Sequence[Rendered], *, bos_token_id: int | None, n_leading: int = 2, strict: bool = True) -> tuple[TemplateConstants, dict[str, Any]]:
    """Assert (or, with ``strict=False``, describe) the constancy of the leading
    ids, the assistant template prefix and the terminator across rows, plus the
    one-BOS rule when the tokenizer has a BOS."""
    leading = {r.ids[:n_leading] for r in rendered}
    prefixes = {r.template_prefix_ids for r in rendered}
    terminators = {r.terminator_ids for r in rendered}
    problems: list[str] = []
    if len(leading) != 1:
        problems.append(f"{len(leading)} distinct leading id sequences: {sorted(leading)[:4]}")
    if len(prefixes) != 1:
        problems.append(f"{len(prefixes)} distinct assistant template prefixes: {sorted(prefixes, key=len)[:4]}")
    if len(terminators) != 1:
        problems.append(f"{len(terminators)} distinct terminators: {sorted(terminators, key=len)[:4]}")
    bos_bad = 0
    if bos_token_id is not None:
        bos_bad = sum(1 for r in rendered if r.ids[0] != bos_token_id or r.ids.count(bos_token_id) != 1)
        if bos_bad:
            problems.append(f"{bos_bad} rows do not carry exactly one leading BOS ({bos_token_id})")
    straddles = sum(1 for r in rendered if r.straddle)
    report = {"n_rows": len(rendered), "distinct_leading": len(leading), "distinct_prefixes": len(prefixes), "distinct_terminators": len(terminators), "rows_bad_bos": bos_bad, "rows_with_boundary_straddle": straddles, "problems": problems}
    if problems and strict:
        raise RuntimeError("template constants vary across rows — the span boundaries cannot be trusted: " + "; ".join(problems))
    constants = TemplateConstants(leading_ids=min(leading), template_prefix_ids=min(prefixes, key=len), terminator_ids=min(terminators, key=len))
    return constants, report


def render_rows(tokenizer: Any, rows: Sequence[RowMeta], *, max_tokens: int) -> list[Rendered]:
    rendered: list[Rendered] = []
    too_long: list[tuple[str, int]] = []
    for row in rows:
        try:
            item = render_row(tokenizer, row.messages)
        except (RuntimeError, ValueError) as error:
            raise RuntimeError(f"row {row.row_id}: {error}") from error
        if item.n_tokens > max_tokens:
            too_long.append((row.row_id, item.n_tokens))
        rendered.append(item)
    if too_long:
        raise RuntimeError(f"{len(too_long)} rows exceed max_tokens={max_tokens} (rows are never truncated), e.g. {too_long[:5]}")
    return rendered


def rendered_ids_sha256(rendered: Sequence[Rendered]) -> str:
    """One hash of every row's token ids in file order — arms of a substrate must agree."""
    digest = hashlib.sha256()
    for item in rendered:
        digest.update(json.dumps(list(item.ids), separators=(",", ":")).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def rows_summary(rows: Sequence[RowMeta], rendered: Sequence[Rendered]) -> dict[str, Any]:
    by_group: dict[str, list[int]] = {}
    for row, item in zip(rows, rendered, strict=True):
        by_group.setdefault(row.group, []).append(item.n_tokens)
    return {
        "n_rows": len(rows),
        "n_tokens_max": max(r.n_tokens for r in rendered),
        "n_tokens_mean": statistics.fmean(r.n_tokens for r in rendered),
        "n_full_tokens_mean": statistics.fmean(r.n_full_tokens for r in rendered),
        "n_content_tokens_mean": statistics.fmean(r.n_content_tokens for r in rendered),
        "n_content_tokens_min": min(r.n_content_tokens for r in rendered),
        "n_template_prefix_tokens": sorted({r.n_template_prefix_tokens for r in rendered}),
        "n_terminator_tokens": sorted({r.n_terminator_tokens for r in rendered}),
        "per_group": {g: {"rows": len(v), "n_tokens_mean": statistics.fmean(v)} for g, v in sorted(by_group.items())},
        "target_policy": TARGET_POLICY,
    }


def template_fingerprint(model_dir: str | Path, tokenizer: Any) -> dict[str, Any]:
    """md5 of the checkpoint's ``chat_template.jinja`` bytes (the file the SFT
    saved), falling back to the tokenizer's in-memory template string."""
    path = Path(model_dir) / "chat_template.jinja"
    if path.is_file():
        data = path.read_bytes()
        return {"template_md5": md5_bytes(data), "template_source": str(path), "template_chars": len(data)}
    template = getattr(tokenizer, "chat_template", None)
    if not template:
        raise RuntimeError(f"{model_dir}: no chat_template.jinja and the tokenizer has no chat_template — rows cannot be rendered with the checkpoint's own template")
    if not isinstance(template, str):
        template = json.dumps(template, sort_keys=True)
    return {"template_md5": md5_bytes(template.encode("utf-8")), "template_source": "tokenizer.chat_template", "template_chars": len(template)}


# ---------------------------------------------------------------- records
@dataclass(frozen=True)
class SpanLosses:
    loss_full: float
    loss_content: float
    loss_template_prefix: float
    loss_terminator: float
    loss_prompt: float


def span_losses(ce: Sequence[float], rendered: Rendered) -> SpanLosses:
    """``ce[j]`` = CE of predicting token ``j+1`` (length L-1). Sums by position."""
    p, cs, ce_end, n = rendered.spans()
    if len(ce) != n - 1:
        raise ValueError(f"per-token CE has {len(ce)} entries for {n} tokens (expected {n - 1})")

    def total(a: int, b: int) -> float:
        return float(sum(ce[a:b]))

    out = SpanLosses(loss_full=total(p - 1, n - 1), loss_content=total(cs - 1, ce_end - 1), loss_template_prefix=total(p - 1, cs - 1), loss_terminator=total(ce_end - 1, n - 1), loss_prompt=total(0, p - 1))
    parts = out.loss_template_prefix + out.loss_content + out.loss_terminator
    if not math.isclose(parts, out.loss_full, rel_tol=1e-4, abs_tol=1e-3):
        raise RuntimeError(f"span bookkeeping error: prefix+content+terminator {parts} != full {out.loss_full}")
    return out


def loss_record(meta: RowMeta, rendered: Rendered, losses: SpanLosses, cfg: RowLossesConfig, template_md5: str, *, repeat: int | None = None) -> dict[str, Any]:
    record = {
        "row_id": meta.row_id,
        "group": meta.group,
        "episode_id": meta.episode_id,
        "subtype": meta.subtype,
        "n_tokens": rendered.n_tokens,
        "n_prompt_tokens": rendered.n_prompt_tokens,
        "n_target_tokens": rendered.n_full_tokens,
        "n_full_tokens": rendered.n_full_tokens,
        "n_content_tokens": rendered.n_content_tokens,
        "n_template_prefix_tokens": rendered.n_template_prefix_tokens,
        "n_terminator_tokens": rendered.n_terminator_tokens,
        "content_start": rendered.content_start,
        "content_end": rendered.content_end,
        "loss": float(losses.loss_full),
        "loss_per_token": float(losses.loss_full) / rendered.n_full_tokens,
        "loss_full": float(losses.loss_full),
        "loss_content": float(losses.loss_content),
        "loss_content_per_token": float(losses.loss_content) / rendered.n_content_tokens,
        "loss_template_prefix": float(losses.loss_template_prefix),
        "loss_terminator": float(losses.loss_terminator),
        "loss_prompt": float(losses.loss_prompt),
        "loss_prompt_per_token": float(losses.loss_prompt) / max(rendered.n_prompt_tokens - 1, 1),
        "profile": cfg.profile,
        "arm": cfg.arm,
        "substrate": cfg.substrate,
        "dose_tokens": cfg.dose_tokens,
        "template_md5": template_md5,
    }
    if repeat is not None:
        record["repeat"] = int(repeat)
    return record


def validate_record(record: Mapping[str, Any], *, noise: bool = False) -> None:
    expected = set(NOISE_ROW_KEYS if noise else LOSS_ROW_KEYS)
    if set(record) != expected:
        raise ValueError(f"record keys {sorted(set(record) ^ expected)} differ from the schema")
    for key in ("loss", *SPAN_FIELDS):
        value = record[key]
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
            raise ValueError(f"record {record.get('row_id')} has a non-finite {key}")


def existing_records(path: Path, cfg: RowLossesConfig, template_md5: str, *, noise: bool = False) -> dict[str, dict[str, Any]]:
    """Rows already in ``path`` (resume); a record from another model or
    template in the same file is a loud error, never silently merged."""
    out: dict[str, dict[str, Any]] = {}
    for record in read_jsonl(path):
        if record.get("profile") != cfg.profile or record.get("arm") != cfg.arm or record.get("template_md5") != template_md5:
            raise RuntimeError(f"{path} holds rows from another model/template ({record.get('profile')}/{record.get('arm')} @ {record.get('template_md5')}); refusing to resume into it")
        validate_record(record, noise=noise)
        out[str(record["row_id"])] = record
    return out


def select_repeat_rows(rows: Sequence[RowMeta], n: int, seed: int) -> list[RowMeta]:
    """Seeded uniform draw of ``n`` rows (all when fewer), in file order."""
    if n <= 0:
        return []
    picked = set(random.Random(seed).sample(range(len(rows)), min(n, len(rows))))
    return [row for i, row in enumerate(rows) if i in picked]


def compare_losses(reference: Sequence[float], candidate: Sequence[float], *, abs_tol: float) -> dict[str, Any]:
    """Row-wise agreement of two scorings of the same rows (batched vs batch 1,
    repeat vs first pass, fp32 vs bf16): passes when every |Δ| < abs_tol."""
    if len(reference) != len(candidate):
        raise ValueError("compare_losses needs equal-length inputs")
    abs_diffs = [abs(a - b) for a, b in zip(reference, candidate, strict=True)]
    rel_diffs = [d / max(abs(a), 1e-12) for d, a in zip(abs_diffs, reference, strict=True)]
    violations = [i for i, d in enumerate(abs_diffs) if not d < abs_tol]
    return {
        "n": len(reference),
        "n_exact": sum(1 for d in abs_diffs if d == 0.0),
        "max_abs_diff": max(abs_diffs, default=0.0),
        "median_abs_diff": statistics.median(abs_diffs) if abs_diffs else 0.0,
        "p90_abs_diff": (sorted(abs_diffs)[int(0.9 * (len(abs_diffs) - 1))] if abs_diffs else 0.0),
        "max_rel_diff": max(rel_diffs, default=0.0),
        "median_rel_diff": statistics.median(rel_diffs) if rel_diffs else 0.0,
        "abs_tol": abs_tol,
        "violations": violations[:20],
        "n_violations": len(violations),
        "passed": not violations,
    }


def make_batches(items: Sequence[tuple[RowMeta, Rendered]], batch_size: int, *, equal_length_only: bool = True) -> list[list[tuple[RowMeta, Rendered]]]:
    """Length-sorted batches; with ``equal_length_only`` a batch never mixes
    lengths, so no padding token is ever added."""
    if batch_size <= 1:
        return [[item] for item in items]
    ordered = sorted(items, key=lambda it: (it[1].n_tokens, it[0].row_index))
    batches: list[list[tuple[RowMeta, Rendered]]] = []
    current: list[tuple[RowMeta, Rendered]] = []
    for item in ordered:
        if current and (len(current) >= batch_size or (equal_length_only and current[-1][1].n_tokens != item[1].n_tokens)):
            batches.append(current)
            current = []
        current.append(item)
    if current:
        batches.append(current)
    return batches


def sanity_report(records: Sequence[Mapping[str, Any]], *, threshold: float) -> dict[str, Any]:
    """Mean content CE/token on ambiguous rows vs the warning threshold (never fatal)."""
    amb = [float(r["loss_content_per_token"]) for r in records if r.get("group") == "ambiguous"]
    mean = statistics.fmean(amb) if amb else None
    return {"ambiguous_rows": len(amb), "ambiguous_content_ce_per_token_mean": mean, "threshold": threshold, "ok": None if mean is None else mean < threshold}


# ------------------------------------------------------------- sidecar
class TokenSidecar:
    """Per-row whole-sequence CE (float16) + token ids + span boundaries, saved
    as one compressed npz (flat arrays + offsets; ``row_ids`` is the index)."""

    def __init__(self, path: Path | None, *, profile: str, arm: str, template_md5: str) -> None:
        self.path = path
        self.meta = {"schema": SIDECAR_SCHEMA, "profile": profile, "arm": arm, "template_md5": template_md5}
        self.rows: dict[str, tuple[Any, Any, tuple[int, int, int, int]]] = {}
        self.dirty = False

    def load(self) -> int:
        if self.path is None or not self.path.is_file():
            return 0
        import numpy as np

        with np.load(self.path, allow_pickle=False) as data:
            if str(data["profile"]) != self.meta["profile"] or str(data["arm"]) != self.meta["arm"] or str(data["template_md5"]) != self.meta["template_md5"]:
                raise RuntimeError(f"{self.path} belongs to another model/template; refusing to resume into it")
            row_ids = [str(x) for x in data["row_ids"]]
            offsets, id_offsets, ce, ids, spans = data["offsets"], data["id_offsets"], data["ce"], data["ids"], data["spans"]
            for i, row_id in enumerate(row_ids):
                self.rows[row_id] = (ce[offsets[i] : offsets[i + 1]].copy(), ids[id_offsets[i] : id_offsets[i + 1]].copy(), tuple(int(x) for x in spans[i]))
        return len(self.rows)

    def add(self, row_id: str, ce: Sequence[float], rendered: Rendered) -> None:
        import numpy as np

        self.rows[row_id] = (np.asarray(ce, dtype=np.float16), np.asarray(rendered.ids, dtype=np.int32), rendered.spans())
        self.dirty = True

    def save(self, *, force: bool = False) -> dict[str, Any] | None:
        if self.path is None or (not self.dirty and not force):
            return None
        import numpy as np

        row_ids = list(self.rows)
        ce_parts = [self.rows[k][0] for k in row_ids]
        id_parts = [self.rows[k][1] for k in row_ids]
        offsets = np.zeros(len(row_ids) + 1, dtype=np.int64)
        id_offsets = np.zeros(len(row_ids) + 1, dtype=np.int64)
        offsets[1:] = np.cumsum([len(c) for c in ce_parts]) if ce_parts else []
        id_offsets[1:] = np.cumsum([len(c) for c in id_parts]) if id_parts else []
        payload = {
            **{k: np.asarray(v) for k, v in self.meta.items()},
            "row_ids": np.asarray(row_ids),
            "offsets": offsets,
            "id_offsets": id_offsets,
            "ce": np.concatenate(ce_parts) if ce_parts else np.zeros(0, dtype=np.float16),
            "ids": np.concatenate(id_parts) if id_parts else np.zeros(0, dtype=np.int32),
            "spans": np.asarray([self.rows[k][2] for k in row_ids], dtype=np.int32).reshape(len(row_ids), 4),
            "span_columns": np.asarray(["prompt_len", "content_start", "content_end", "n_tokens"]),
            "ce_convention": np.asarray("ce[j] = cross-entropy of predicting token j+1 given tokens <= j (fp32 computed, float16 stored)"),
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp.npz")
        with tmp.open("wb") as handle:
            np.savez_compressed(handle, **payload)
        tmp.replace(self.path)
        self.dirty = False
        return {"path": str(self.path), "n_rows": len(row_ids), "bytes": self.path.stat().st_size, "schema": SIDECAR_SCHEMA}


# ------------------------------------------------------------ torch side
def load_tokenizer(cfg: RowLossesConfig):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_dir, local_files_only=cfg.local_files_only, trust_remote_code=cfg.trust_remote_code)
    if not getattr(tokenizer, "chat_template", None) and not (Path(cfg.model_dir) / "chat_template.jinja").is_file():
        raise RuntimeError(f"{cfg.model_dir}: tokenizer has no chat template")
    return tokenizer


def patch_gemma3_token_type_ids(model):
    """v1's wrapper: all-text ``token_type_ids`` when a call omits them (text-only
    rows -> zeros are exact; a no-op when the caller passes the field)."""
    import functools

    import torch

    forward = model.forward

    @functools.wraps(forward)
    def wrapped(*args, **kwargs):
        if kwargs.get("token_type_ids") is None:
            ids = kwargs.get("input_ids")
            if ids is None and args:
                ids = args[0]
            if ids is not None:
                kwargs["token_type_ids"] = torch.zeros_like(ids)
        return forward(*args, **kwargs)

    model.forward = wrapped
    return model


def accepts_token_type_ids(model) -> bool:
    """Only wrap forwards that declare ``token_type_ids`` (Gemma3ForConditionalGeneration
    does; a text-only Gemma3ForCausalLM would choke on the extra kwarg)."""
    import inspect

    try:
        return "token_type_ids" in inspect.signature(model.forward).parameters
    except (TypeError, ValueError):
        return False


def check_loading_info(info: Mapping[str, Any] | None, *, tied_ok: Sequence[str] = TIED_KEYS_OK) -> dict[str, Any]:
    """``from_pretrained(output_loading_info=True)`` -> counts; raises on any
    missing / unexpected / mismatched key or error beyond the tied lm_head."""
    info = dict(info or {})
    report: dict[str, Any] = {}
    problems: list[str] = []
    for key, value in info.items():
        items = list(value) if isinstance(value, (list, tuple, set)) else ([value] if value else [])
        if key == "missing_keys":
            items = [k for k in items if str(k) not in tied_ok]
        report[key] = {"n": len(items), "examples": [str(x) for x in items[:8]]}
        if items:
            problems.append(f"{key}: {len(items)} (e.g. {[str(x) for x in items[:4]]})")
    report["ok"] = not problems
    if problems:
        raise RuntimeError("checkpoint did not load cleanly — " + "; ".join(problems) + " (a legacy-key Gemma save random-initialising tensors would look exactly like this)")
    return report


def check_architecture(config: Any, *, expected_architecture: str | None, expected_layers: int | None) -> dict[str, Any]:
    architectures = list(getattr(config, "architectures", None) or [])
    text_config = getattr(config, "text_config", None)
    layers = getattr(config, "num_hidden_layers", None)
    if layers is None and text_config is not None:
        layers = getattr(text_config, "num_hidden_layers", None)
    report = {"architectures": architectures, "num_hidden_layers": layers, "expected_architecture": expected_architecture, "expected_layers": expected_layers}
    if expected_architecture is not None and expected_architecture not in architectures:
        raise RuntimeError(f"architecture {architectures} != expected {expected_architecture}")
    if expected_layers is not None and layers != expected_layers:
        raise RuntimeError(f"num_hidden_layers {layers} != expected {expected_layers}")
    return report


def _from_pretrained_attempts(cfg: RowLossesConfig) -> list[tuple[str | None, str | None]]:
    attempts = [(cfg.attn_implementation, cfg.experts_implementation)]
    if cfg.load_fallback and (cfg.attn_implementation not in (None, "eager") or cfg.experts_implementation is not None):
        attempts.append(("eager", None))
    return attempts


def load_model(cfg: RowLossesConfig, *, dtype: str | None = None) -> tuple[Any, dict[str, Any]]:
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM

    dtype = dtype or cfg.dtype
    config = AutoConfig.from_pretrained(cfg.model_dir, local_files_only=cfg.local_files_only, trust_remote_code=cfg.trust_remote_code)
    auto_map = getattr(config, "auto_map", None)
    if auto_map and not cfg.trust_remote_code:
        raise RuntimeError(f"{cfg.model_dir}: config.json carries auto_map={auto_map} (remote code) but trust_remote_code is false — set it only if this checkpoint really needs it")
    architecture = check_architecture(config, expected_architecture=cfg.expected_architecture, expected_layers=cfg.expected_layers)
    base_kwargs: dict[str, Any] = {"dtype": getattr(torch, dtype), "local_files_only": cfg.local_files_only, "trust_remote_code": cfg.trust_remote_code, "output_loading_info": True}
    if cfg.device_map:
        base_kwargs["device_map"] = cfg.device_map
        if cfg.max_memory:
            base_kwargs["max_memory"] = {(int(k) if str(k).isdigit() else k): v for k, v in cfg.max_memory.items()}
    base_kwargs.update(cfg.model_kwargs)
    attempts = _from_pretrained_attempts(cfg)
    errors: list[str] = []
    model = loading = None
    used: tuple[str | None, str | None] | None = None
    for attn, experts in attempts:
        kwargs = dict(base_kwargs)
        if attn:
            kwargs["attn_implementation"] = attn
        if experts:
            kwargs["experts_implementation"] = experts
        try:
            model, loading = AutoModelForCausalLM.from_pretrained(cfg.model_dir, **kwargs)
            used = (attn, experts)
            break
        except (ValueError, TypeError, NotImplementedError, ImportError, AttributeError) as error:
            errors.append(f"attn={attn} experts={experts}: {type(error).__name__}: {error}")
            log(f"from_pretrained refused attn={attn} experts={experts}: {type(error).__name__}: {str(error)[:300]}")
    if model is None:
        raise RuntimeError("model could not be loaded: " + " | ".join(errors))
    loading_report = check_loading_info(loading)
    if not cfg.device_map:
        model.to(torch.device(cfg.device))
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    param_dtype = next(model.parameters()).dtype
    if param_dtype != getattr(torch, dtype):
        raise RuntimeError(f"model loaded as {param_dtype}, config asked for {dtype}")
    model_type = getattr(config, "model_type", None)
    text_config = getattr(config, "text_config", None)
    gemma3 = (model_type in GEMMA3_MODEL_TYPES or getattr(text_config, "model_type", None) in GEMMA3_MODEL_TYPES) and accepts_token_type_ids(model)
    if gemma3:
        patch_gemma3_token_type_ids(model)
    experts_class = next((type(m).__name__ for n, m in model.named_modules() if "expert" in type(m).__name__.lower()), None)
    info = {
        "architecture": architecture,
        "model_class": type(model).__name__,
        "model_type": model_type,
        "dtype": str(param_dtype),
        "n_params": sum(p.numel() for p in model.parameters()),
        "device_map": None if not cfg.device_map else {str(k): str(v) for k, v in (getattr(model, "hf_device_map", None) or {}).items()},
        "attn_implementation_requested": cfg.attn_implementation,
        "attn_implementation": getattr(config, "_attn_implementation", None) or (used[0] if used else None),
        "experts_implementation_requested": cfg.experts_implementation,
        "experts_implementation": getattr(config, "_experts_implementation", None) or (used[1] if used else None),
        "experts_module_class": experts_class,
        "load_fallback_used": used != attempts[0],
        "load_attempts_failed": errors,
        "loading_info": loading_report,
        "trust_remote_code": cfg.trust_remote_code,
        "gemma3_token_type_ids_patch": gemma3,
        "auto_map": auto_map,
    }
    return model, info


def input_device(model) -> Any:
    return model.get_input_embeddings().weight.device


class RowScorer:
    """Forward passes -> whole-sequence per-token CE (fp32) per row."""

    def __init__(self, model, *, pad_id: int) -> None:
        self.model = model
        self.pad_id = int(pad_id)
        self.device = input_device(model)

    def token_ce(self, batch: Sequence[Rendered]) -> list[list[float]]:
        import torch
        import torch.nn.functional as F

        lengths = [item.n_tokens for item in batch]
        width = max(lengths)
        padded = any(length != width for length in lengths)
        ids = torch.full((len(batch), width), self.pad_id, dtype=torch.long)
        for r, item in enumerate(batch):
            ids[r, : item.n_tokens] = torch.tensor(item.ids, dtype=torch.long)
        ids = ids.to(self.device)
        kwargs: dict[str, Any] = {"input_ids": ids, "use_cache": False}
        if padded:
            mask = torch.zeros((len(batch), width), dtype=torch.long)
            for r, length in enumerate(lengths):
                mask[r, :length] = 1
            kwargs["attention_mask"] = mask.to(self.device)
            kwargs["position_ids"] = torch.arange(width, dtype=torch.long).unsqueeze(0).expand(len(batch), width).to(self.device)
        with torch.inference_mode():
            logits = self.model(**kwargs).logits
        out: list[list[float]] = []
        for r, length in enumerate(lengths):
            pred = logits[r, : length - 1].float()
            targets = ids[r, 1:length].to(pred.device)
            ce = F.cross_entropy(pred, targets, reduction="none")
            out.append([float(x) for x in ce.tolist()])
        return out


def cuda_peaks_gb() -> dict[str, float]:
    import torch

    if not torch.cuda.is_available():
        return {}
    return {f"cuda:{i}": torch.cuda.max_memory_allocated(i) / GB for i in range(torch.cuda.device_count())}


def init_cuda_peaks() -> None:
    import torch

    if not torch.cuda.is_available():
        return
    for i in range(torch.cuda.device_count()):
        device = torch.device(f"cuda:{i}")
        torch.zeros(1, device=device)  # initialise the allocator; the peak counters do not exist before the first allocation
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)


def free_model(model) -> None:
    import gc

    import torch

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def versions() -> dict[str, Any]:
    import importlib.metadata as m
    import platform

    out: dict[str, Any] = {"python": platform.python_version()}
    for name in ("torch", "transformers", "huggingface_hub", "accelerate", "safetensors", "tokenizers", "numpy"):
        try:
            out[name] = m.version(name)
        except m.PackageNotFoundError:
            out[name] = None
    try:
        import torch

        out["cuda"] = torch.version.cuda
        out["cuda_devices"] = torch.cuda.device_count()
        out["gpu_names"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    except Exception as error:  # noqa: BLE001 — provenance only
        out["cuda_probe_error"] = repr(error)
    return out


# ------------------------------------------------------------------- run
def _score_items(scorer: RowScorer, items: Sequence[tuple[RowMeta, Rendered]], cfg: RowLossesConfig) -> dict[str, tuple[list[float], SpanLosses]]:
    out: dict[str, tuple[list[float], SpanLosses]] = {}
    for batch in make_batches(items, cfg.batch_size, equal_length_only=cfg.equal_length_only):
        for (row, item), ce in zip(batch, scorer.token_ce([r for _, r in batch]), strict=True):
            out[row.row_id] = (ce, span_losses(ce, item))
    return out


def run(cfg: RowLossesConfig) -> dict[str, Any]:
    started = time.time()
    timings: dict[str, float] = {}
    warnings: list[str] = []
    out_path = Path(cfg.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    log(f"row_losses {cfg.profile}/{cfg.arm} ({cfg.substrate}, dose {cfg.dose_tokens}) model_dir={cfg.model_dir} rows={cfg.rows_path} batch_size={cfg.batch_size} device_map={cfg.device_map} attn={cfg.attn_implementation} experts={cfg.experts_implementation}")

    t0 = time.time()
    tokenizer = load_tokenizer(cfg)
    template = template_fingerprint(cfg.model_dir, tokenizer)
    tokenizer_json = Path(cfg.model_dir) / "tokenizer.json"
    tokenizer_info = {
        "class": type(tokenizer).__name__,
        "tokenizer_json_sha256": sha256_file(tokenizer_json) if tokenizer_json.is_file() else None,
        "pad_token_id": getattr(tokenizer, "pad_token_id", None),
        "eos_token_id": getattr(tokenizer, "eos_token_id", None),
        "bos_token_id": getattr(tokenizer, "bos_token_id", None),
        "padding_side": getattr(tokenizer, "padding_side", None),
        "vocab_size": len(tokenizer) if hasattr(tokenizer, "__len__") else None,
    }
    timings["load_tokenizer_s"] = time.time() - t0
    log(f"template {template['template_md5']} from {template['template_source']} ({template['template_chars']} chars); tokenizer {tokenizer_info['class']} pad={tokenizer_info['pad_token_id']} eos={tokenizer_info['eos_token_id']} bos={tokenizer_info['bos_token_id']} padding_side={tokenizer_info['padding_side']}")

    t0 = time.time()
    rows = load_eft_rows(cfg.rows_path, cfg.groups)
    if cfg.row_limit is not None:
        rows = rows[: cfg.row_limit]
    rendered = render_rows(tokenizer, rows, max_tokens=cfg.max_tokens)
    constants, constants_report = template_constants(rendered, bos_token_id=tokenizer_info["bos_token_id"], strict=cfg.require_constant_template_ids)
    if constants_report["problems"]:
        warnings.extend(constants_report["problems"])
    summary = rows_summary(rows, rendered)
    ids_hash = rendered_ids_sha256(rendered)
    timings["render_rows_s"] = time.time() - t0
    decode = getattr(tokenizer, "decode", None)
    constants_record = {
        "leading_ids": list(constants.leading_ids),
        "assistant_template_prefix_ids": list(constants.template_prefix_ids),
        "assistant_template_prefix_text": decode(list(constants.template_prefix_ids)) if decode else None,
        "assistant_terminator_ids": list(constants.terminator_ids),
        "assistant_terminator_text": decode(list(constants.terminator_ids)) if decode else None,
        **constants_report,
    }
    log(f"{len(rows)} rows rendered: max {summary['n_tokens_max']} tokens, mean {summary['n_tokens_mean']:.0f}; assistant turn mean {summary['n_full_tokens_mean']:.1f} tokens = prefix {constants_record['assistant_template_prefix_ids']} + content mean {summary['n_content_tokens_mean']:.1f} + terminator {constants_record['assistant_terminator_ids']}; rendered_ids sha256 {ids_hash[:12]}")

    t0 = time.time()
    model, model_info = load_model(cfg)
    timings["load_model_s"] = time.time() - t0
    log(f"model loaded in {timings['load_model_s']:.0f}s: {model_info['model_class']} {model_info['n_params'] / 1e9:.1f}B params {model_info['dtype']} attn={model_info['attn_implementation']} experts={model_info['experts_implementation']} ({model_info['experts_module_class']}) fallback={model_info['load_fallback_used']} loading_info ok")
    pad_id = tokenizer_info["pad_token_id"] if tokenizer_info["pad_token_id"] is not None else tokenizer_info["eos_token_id"]
    if isinstance(pad_id, (list, tuple)):
        pad_id = pad_id[0]
    if pad_id is None:
        raise RuntimeError("tokenizer has neither pad nor eos token id")
    scorer = RowScorer(model, pad_id=int(pad_id))
    init_cuda_peaks()
    items = list(zip(rows, rendered, strict=True))
    by_id = {row.row_id: (row, item) for row, item in items}

    # batched scoring must agree with batch 1 (bf16 + padding) before it is trusted — fatal gate
    batch_check: dict[str, Any] | None = None
    if cfg.batch_size > 1:
        n = min(cfg.batch_check_rows, len(items))
        t0 = time.time()
        single = {row.row_id: span_losses(scorer.token_ce([item])[0], item) for row, item in items[:n]}
        batched = _score_items(scorer, items[:n], cfg)
        keys = [row.row_id for row, _ in items[:n]]
        batch_check = {"batch_size": cfg.batch_size, "equal_length_only": cfg.equal_length_only, "n_batches": len(make_batches(items[:n], cfg.batch_size, equal_length_only=cfg.equal_length_only)), "seconds": time.time() - t0}
        for fld in ("loss_full", "loss_content"):
            batch_check[fld] = compare_losses([getattr(single[k], fld) for k in keys], [getattr(batched[k][1], fld) for k in keys], abs_tol=cfg.batch_check_abs_tol)
        batch_check["passed"] = all(batch_check[f]["passed"] for f in ("loss_full", "loss_content"))
        log(f"batch check ({n} rows, batch {cfg.batch_size}, {batch_check['n_batches']} batches): max |Δ full| {batch_check['loss_full']['max_abs_diff']:.4g}, max |Δ content| {batch_check['loss_content']['max_abs_diff']:.4g} nats (tol {cfg.batch_check_abs_tol}) -> {'PASS' if batch_check['passed'] else 'FAIL'}")
        if not batch_check["passed"]:
            raise RuntimeError(f"batched scoring disagrees with batch 1 (max |Δ| {max(batch_check['loss_full']['max_abs_diff'], batch_check['loss_content']['max_abs_diff']):.4g} nats >= {cfg.batch_check_abs_tol}); rerun with batch_size 1")

    # resume: keep rows present in BOTH the jsonl and the sidecar
    sidecar = TokenSidecar(cfg.sidecar, profile=cfg.profile, arm=cfg.arm, template_md5=template["template_md5"])
    done: dict[str, dict[str, Any]] = {}
    if cfg.resume:
        done = existing_records(out_path, cfg, template["template_md5"])
        n_sidecar = sidecar.load()
        if sidecar.path is not None:
            lost = [k for k in done if k not in sidecar.rows]
            if lost:
                log(f"resume: {len(lost)} rows are in {out_path.name} but not in the sidecar; re-scoring them")
                done = {k: v for k, v in done.items() if k in sidecar.rows}
            extra = [k for k in sidecar.rows if k not in done]
            for k in extra:
                sidecar.rows.pop(k)
            if lost or extra:
                sidecar.dirty = True
        if len(done) != len(read_jsonl(out_path)):
            with out_path.open("w", encoding="utf-8") as handle:
                for record in done.values():
                    handle.write(json.dumps(record, sort_keys=True) + "\n")
        log(f"resume: {len(done)} rows kept from {out_path.name} ({n_sidecar} in the sidecar)")
    else:
        if out_path.exists():
            out_path.unlink()
        if sidecar.path is not None and sidecar.path.exists():
            sidecar.path.unlink()
    span_by_id: dict[str, SpanLosses] = {k: SpanLosses(**{f: float(v[f]) for f in SPAN_FIELDS}) for k, v in done.items()}
    todo = [(row, item) for row, item in items if row.row_id not in done]
    log(f"scoring {len(todo)} rows ({len(done)} already done)")
    row_seconds: list[float] = []
    scored = 0
    loop_started = time.time()
    batches = make_batches(todo, cfg.batch_size, equal_length_only=cfg.equal_length_only)
    with out_path.open("a", encoding="utf-8") as handle:
        for batch in batches:
            t_row = time.perf_counter()
            ces = scorer.token_ce([item for _, item in batch])
            seconds = time.perf_counter() - t_row
            for (row, item), ce in zip(batch, ces, strict=True):
                if not all(math.isfinite(x) for x in ce):
                    raise RuntimeError(f"non-finite per-token CE for row {row.row_id}")
                losses = span_losses(ce, item)
                record = loss_record(row, item, losses, cfg, template["template_md5"])
                validate_record(record)
                handle.write(json.dumps(record, sort_keys=True) + "\n")
                sidecar.add(row.row_id, ce, item)
                span_by_id[row.row_id] = losses
                scored += 1
            handle.flush()
            row_seconds.append(seconds / len(batch))
            if scored % cfg.sidecar_flush_every < len(batch):
                sidecar.save()
            if scored % cfg.progress_every < len(batch) or scored == len(todo):
                elapsed = time.time() - loop_started
                rate = scored / max(elapsed, 1e-9)
                remaining = len(todo) - scored
                peaks = cuda_peaks_gb()
                last_row, last_losses = batch[-1][0], span_by_id[batch[-1][0].row_id]
                log(f"row {scored}/{len(todo)} {last_row.row_id} full={last_losses.loss_full:.3f} content={last_losses.loss_content:.3f} ({row_seconds[-1]:.2f} s/row) | {rate:.2f} rows/s ETA {remaining / max(rate, 1e-9) / 60:.1f} min | peak {' '.join(f'{k}={v:.1f}GB' for k, v in peaks.items())}")
    timings["score_rows_s"] = time.time() - loop_started
    sidecar_record = sidecar.save(force=True)

    # determinism floor: seeded rows scored a second time
    noise: dict[str, Any] | None = None
    if cfg.noise_out_path and cfg.repeat_rows > 0:
        t0 = time.time()
        noise_path = Path(cfg.noise_out_path)
        noise_path.parent.mkdir(parents=True, exist_ok=True)
        picked = select_repeat_rows(rows, cfg.repeat_rows, cfg.repeat_seed)
        have = existing_records(noise_path, cfg, template["template_md5"], noise=True) if cfg.resume else {}
        if not cfg.resume and noise_path.exists():
            noise_path.unlink()
        repeats: dict[str, SpanLosses] = {k: SpanLosses(**{f: float(v[f]) for f in SPAN_FIELDS}) for k, v in have.items()}
        pending = [by_id[row.row_id] for row in picked if row.row_id not in have]
        with noise_path.open("a", encoding="utf-8") as handle:
            for batch in make_batches(pending, cfg.batch_size, equal_length_only=cfg.equal_length_only):
                for (row, item), ce in zip(batch, scorer.token_ce([r for _, r in batch]), strict=True):
                    losses = span_losses(ce, item)
                    record = loss_record(row, item, losses, cfg, template["template_md5"], repeat=1)
                    validate_record(record, noise=True)
                    handle.write(json.dumps(record, sort_keys=True) + "\n")
                    repeats[row.row_id] = losses
        ordered = [row.row_id for row in picked if row.row_id in span_by_id and row.row_id in repeats]
        noise = {"path": str(noise_path), "seed": cfg.repeat_seed, "requested_rows": cfg.repeat_rows, "n": len(ordered), "seconds": time.time() - t0}
        for fld in ("loss_full", "loss_content"):
            noise[fld] = {k: v for k, v in compare_losses([getattr(span_by_id[i], fld) for i in ordered], [getattr(repeats[i], fld) for i in ordered], abs_tol=cfg.batch_check_abs_tol).items() if k not in ("abs_tol", "violations", "n_violations", "passed")}
        log(f"noise floor ({noise['n']} rows): full {noise['loss_full']['n_exact']} bit-identical, max |Δ| {noise['loss_full']['max_abs_diff']:.4g}; content max |Δ| {noise['loss_content']['max_abs_diff']:.4g} nats")
        timings["noise_s"] = noise["seconds"]

    # bf16 vs fp32 on a few rows (recorded, never fatal)
    fp32_check: dict[str, Any] | None = None
    if cfg.fp32_check_rows > 0 and cfg.dtype != "float32":
        t0 = time.time()
        try:
            free_model(model)
            model32, info32 = load_model(cfg, dtype="float32")
            scorer32 = RowScorer(model32, pad_id=int(pad_id))
            subset = items[: min(cfg.fp32_check_rows, len(items))]
            values32 = _score_items(scorer32, subset, RowLossesConfig.from_mapping({**cfg.to_dict(), "batch_size": 1}))
            keys = [row.row_id for row, _ in subset]
            fp32_check = {"n": len(keys), "dtype": info32["dtype"], "seconds": time.time() - t0}
            for fld in ("loss_full", "loss_content"):
                fp32_check[fld] = compare_losses([getattr(values32[k][1], fld) for k in keys], [getattr(span_by_id[k], fld) for k in keys], abs_tol=cfg.batch_check_abs_tol)
            log(f"fp32 check ({len(keys)} rows): bf16 vs fp32 max |Δ full| {fp32_check['loss_full']['max_abs_diff']:.4g}, median {fp32_check['loss_full']['median_abs_diff']:.4g}; content max |Δ| {fp32_check['loss_content']['max_abs_diff']:.4g} nats")
            free_model(model32)
        except Exception as error:  # noqa: BLE001 — a sizing experiment, never a gate
            fp32_check = {"status": "failed", "error": repr(error), "seconds": time.time() - t0}
            warnings.append(f"fp32 check failed: {error!r}")
            log(f"{WARN_SENTINEL} fp32 check failed: {error!r}")
        timings["fp32_check_s"] = time.time() - t0

    timings["total_s"] = time.time() - started
    records = read_jsonl(out_path)
    sanity = sanity_report(records, threshold=cfg.sanity_max_content_ce)
    if sanity["ok"] is False:
        warnings.append(f"mean content CE/token on ambiguous rows {sanity['ambiguous_content_ce_per_token_mean']:.3f} >= {cfg.sanity_max_content_ce}")
        log(f"{WARN_SENTINEL} {warnings[-1]} (descriptive; the model may not have loaded the weights it should have)")
    rate = (scored / sum(row_seconds)) if row_seconds and sum(row_seconds) > 0 else None
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "status": "ok",
        "created_at": now_iso(),
        "profile": cfg.profile,
        "arm": cfg.arm,
        "substrate": cfg.substrate,
        "dose_tokens": cfg.dose_tokens,
        "model_dir": cfg.model_dir,
        "hf_repo": cfg.hf_repo,
        "hf_revision": cfg.hf_revision,
        "hf_path": cfg.hf_path,
        "model": model_info,
        "template": template,
        "template_constants": constants_record,
        "tokenizer": tokenizer_info,
        "rows": {"path": cfg.rows_path, "sha256": sha256_file(cfg.rows_path), "rendered_ids_sha256": ids_hash, **summary},
        "scoring": {
            "out_path": str(out_path),
            "batch_size": cfg.batch_size,
            "equal_length_only": cfg.equal_length_only,
            "n_batches": len(batches),
            "scored_rows": scored,
            "skipped_rows_resume": len(done),
            "n_records_in_file": len(records),
            "expected_rows": cfg.expected_rows,
            "row_count_ok": None if cfg.expected_rows is None else len(records) == cfg.expected_rows,
            "rows_per_s": rate,
            "row_seconds_median": statistics.median(row_seconds) if row_seconds else None,
            "row_seconds_p90": (sorted(row_seconds)[int(0.9 * (len(row_seconds) - 1))] if row_seconds else None),
            "means": {fld: statistics.fmean(r[fld] for r in records) for fld in (*SPAN_FIELDS, "loss_per_token", "loss_content_per_token", "loss_prompt_per_token")} if records else None,
        },
        "sidecar": sidecar_record,
        "batch_check": batch_check,
        "noise": noise,
        "fp32_check": fp32_check,
        "sanity": sanity,
        "warnings": warnings,
        "timings_s": timings,
        "peak_memory_gb": cuda_peaks_gb(),
        "host_maxrss_gb": host_maxrss_gb(),
        "versions": versions(),
        "code_commit": git_commit(),
        "config": cfg.to_dict(),
    }
    write_json(cfg.manifest, manifest)
    log(f"{DONE_SENTINEL} profile={cfg.profile} arm={cfg.arm} rows={len(records)} scored={scored} rows_per_s={'n/a' if rate is None else f'{rate:.3f}'} warnings={len(warnings)} manifest={cfg.manifest}")
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    cfg = load_config(argv, CONFIG_ENV, "row_losses.py", RowLossesConfig.from_mapping)
    run(cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
