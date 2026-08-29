#!/usr/bin/env python3
"""Chat-vector graft: W_graft = W_mid + lam * (W_chat - W_base), per tensor.

Policy (INVESTIGATION.md, approved 2026-08-28):

- Inputs: our midtrain-end checkpoint (packed-experts transformers layout;
  unpacked in place to the vendor per-expert layout via the proven
  ``qa_v2/glm_unpack_experts.py`` bridge), plus the vendor chat
  (zai-org/GLM-4.5-Air @ a24ceef6) and base (zai-org/GLM-4.5-Air-Base @
  888c873d) checkpoints.
- Graft every shared-after-unpack tensor (17,925 at GLM-4.5-Air scale):
  fp32 accumulate, write back in the tensor's own dtype (bf16 for the
  18,283 bf16 tensors; the 45 per-layer router ``e_score_correction_bias``
  tensors stay f32).
- MTP tensors (``model.layers.{num_hidden_layers+}.*`` in chat/base, 404 at
  this scale) are dropped — they are absent from our checkpoint
  (``num_nextn_predict_layers: 0``) and vLLM eval never uses them.
- Aux files ship from CHAT verbatim (config.json with
  ``num_nextn_predict_layers`` forced to 0, generation_config.json,
  chat_template.jinja, tokenizer.json, tokenizer_config.json) — never from
  ours (our re-save stripped tokenizer_config.json). tokenizer.json must be
  content-equal between chat and base (asserted) so the graft is
  tokenizer-consistent with the midtrain.
- Verification: router-bias identity ||mid - base|| = 0 on every f32 bias
  (midtrain ran with bias_update_rate 0.0); every output tensor finite
  (abort on any NaN/Inf); rebuilt index total_size matches the input's;
  per-class ||delta||/||W|| stats + max|delta| distribution recorded.

The module is import-safe on CPU with tiny tensors (unit tests); the pod
invokes the CLI with production expectations pinned.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

HERE = Path(__file__).resolve().parent
QA2_DIR = HERE.parent / "qa_v2"
for _path in (str(HERE), str(QA2_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from glm_unpack_experts import unpack_packed_experts  # noqa: E402

INDEX_NAME = "model.safetensors.index.json"
SINGLE_FILE = "model.safetensors"
ROUTER_BIAS_SUFFIX = ".mlp.gate.e_score_correction_bias"
#: aux files shipped from the chat checkpoint (config.json handled apart).
CHAT_AUX_REQUIRED = (
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
)
#: shipped when the chat repo has them (GLM: chat_template.jinja; Gemma-4
#: unified: processor_config.json etc.).
CHAT_AUX_OPTIONAL = (
    "chat_template.jinja",
    "processor_config.json",
    "preprocessor_config.json",
    "special_tokens_map.json",
    "tokenizer.model",
)
#: our trainer re-saves materialize tied heads that the vendor keeps tied
#: (Gemma-4: lm_head.weight duplicates the embedding). When the configs tie
#: and the bytes match, the duplicate is dropped from the graft.
TIE_ALIASES = {
    "lm_head.weight": (
        "model.language_model.embed_tokens.weight",
        "model.embed_tokens.weight",
    ),
}
#: cap per output shard (whole-shard dicts are held in RAM before save).
DEFAULT_MAX_SHARD_BYTES = 4 * 1024**3

#: Production expectations (GLM-4.5-Air, INVESTIGATION.md §2). The CLI pins
#: these; graft() only asserts what it is given. total_size = ours' index
#: total (213,704,502,528) + 11,520 B: the 45 router biases upcast from the
#: re-save's bf16 back to the vendor's f32 (live finding, r4 2026-08-28).
PRODUCTION_EXPECT = {
    "shared_tensors": 17_925,
    "mtp_tensors": 404,
    "total_size": 213_704_514_048,
    "router_bias_tensors": 45,
}


def _load_index(model_dir: Path) -> dict[str, Any]:
    """Index for sharded checkpoints; synthesized for single-file ones
    (Gemma-4 vendor repos and our 12B midtrain saves ship one
    model.safetensors with no index)."""
    path = model_dir / INDEX_NAME
    if path.is_file():
        return json.loads(path.read_text())
    single = model_dir / SINGLE_FILE
    if single.is_file():
        from safetensors import safe_open

        with safe_open(str(single), framework="pt") as reader:
            names = list(reader.keys())
        return {"metadata": {}, "weight_map": {name: SINGLE_FILE for name in names}}
    raise FileNotFoundError(f"neither {path} nor {single}")


def _mtp_prefixes(chat_config: Mapping[str, Any]) -> set[str]:
    """MTP layers live at indexes [num_hidden_layers, +num_nextn_predict_layers).
    GLM-only; families without the key (Gemma-4) have no MTP tensors."""
    nextn = int(chat_config.get("num_nextn_predict_layers") or 0)
    if not nextn:
        return set()
    layers = int(chat_config["num_hidden_layers"])
    return {f"model.layers.{layers + k}." for k in range(nextn)}


def tensor_class(name: str) -> str:
    """Coarse class for the delta-norm table."""
    if name in ("model.embed_tokens.weight", "lm_head.weight"):
        return "embed_lm_head"
    if name.endswith(ROUTER_BIAS_SUFFIX):
        return "router_bias"
    if name.endswith(".mlp.gate.weight"):
        return "router_gate"
    if ".mlp.experts." in name:
        return "experts"
    if ".mlp.shared_experts." in name:
        return "shared_experts"
    if ".self_attn." in name:
        return "attention"
    if ".mlp." in name:
        return "mlp_dense"
    if "norm" in name:
        return "norms"
    return "other"


def _sha256_file(path: Path, chunk: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


#: tokenizer.json sections that define the token-id mapping (must be equal);
#: everything else (post_processor, decoder, padding, truncation) is
#: template-level and may differ between chat and base.
TOKENIZER_CRITICAL_SECTIONS = ("model", "added_tokens", "normalizer", "pre_tokenizer")


def _tokenizer_gate(mid_dir: Path, chat_dir: Path, base_dir: Path) -> dict[str, list[str]]:
    """Enforce id-mapping identity chat==base (and mid==base when ours ships
    a tokenizer.json); returns the non-critical sections that differ."""
    def load(directory: Path) -> Any | None:
        path = directory / "tokenizer.json"
        if not path.is_file():
            return None
        try:
            return json.loads(path.read_text())
        except (json.JSONDecodeError, UnicodeDecodeError):
            return "UNPARSEABLE"

    chat_tok, base_tok = load(chat_dir), load(base_dir)
    if chat_tok is None or base_tok is None:
        raise RuntimeError("chat/base tokenizer.json missing")
    diffs: dict[str, list[str]] = {}
    for label, ours in (("chat_vs_base", chat_tok), ("mid_vs_base", load(mid_dir))):
        if ours is None:
            continue  # our re-save may not ship tokenizer.json — fine
        if ours == "UNPARSEABLE" or base_tok == "UNPARSEABLE":
            source = (chat_dir if label == "chat_vs_base" else mid_dir) / "tokenizer.json"
            if _sha256_file(source) != _sha256_file(base_dir / "tokenizer.json"):
                raise RuntimeError(
                    f"{label}: tokenizer.json differ (non-JSON, byte check) — "
                    "the graft assumes one token-id mapping across parents"
                )
            diffs[label] = []
            continue
        critical = [
            key for key in TOKENIZER_CRITICAL_SECTIONS
            if ours.get(key) != base_tok.get(key)
        ]
        if critical:
            raise RuntimeError(
                f"{label}: tokenizer id-mapping sections differ: {critical} — "
                "embedding arithmetic would mix vocabularies; aborting"
            )
        diffs[label] = sorted(
            key for key in set(ours) | set(base_tok)
            if ours.get(key) != base_tok.get(key)
        )
    return diffs


def write_sha256_manifest(out_dir: Path) -> Path:
    """sha256 of every file in out_dir -> sha256_manifest.json (committed to
    GCS beside the shards; the eval configs record its hash as the pin)."""
    entries = {}
    for path in sorted(out_dir.rglob("*")):
        if path.is_file() and path.name != "sha256_manifest.json":
            entries[path.relative_to(out_dir).as_posix()] = {
                "sha256": _sha256_file(path),
                "bytes": path.stat().st_size,
            }
    body = {"algorithm": "sha256", "files": entries}
    body_text = json.dumps(body, indent=2, sort_keys=True) + "\n"
    manifest = out_dir / "sha256_manifest.json"
    manifest.write_text(body_text)
    return manifest


class _HandleCache:
    """Keep one safetensors reader open per shard file (mmap-backed)."""

    def __init__(self) -> None:
        from safetensors import safe_open

        self._safe_open = safe_open
        self._handles: dict[Path, Any] = {}

    def get(self, path: Path) -> Any:
        handle = self._handles.get(path)
        if handle is None:
            handle = self._safe_open(str(path), framework="pt")
            self._handles[path] = handle
        return handle


def _quantiles(values: list[float], points: tuple[float, ...]) -> dict[str, float]:
    if not values:
        return {}
    ordered = sorted(values)
    out = {}
    for point in points:
        index = min(len(ordered) - 1, max(0, math.ceil(point * len(ordered)) - 1))
        out[f"p{int(point * 100)}"] = ordered[index]
    return out


def graft(
    mid_dir: Path,
    chat_dir: Path,
    base_dir: Path,
    out_dir: Path,
    *,
    lam: float = 1.0,
    expect: Mapping[str, int] | None = None,
    max_shard_bytes: int = DEFAULT_MAX_SHARD_BYTES,
) -> dict[str, Any]:
    """Run the graft; returns (and writes) the stats payload.

    ``expect`` pins tensor counts / total_size (production values via the
    CLI); tests pass their own tiny expectations or None.
    """
    import torch

    mid_dir, chat_dir, base_dir, out_dir = (
        Path(mid_dir), Path(chat_dir), Path(base_dir), Path(out_dir)
    )
    started = time.time()

    # ---- 0. unpack ours in place (no-op when already vendor layout) ----
    unpacked = unpack_packed_experts(mid_dir)

    mid_index = _load_index(mid_dir)
    chat_index = _load_index(chat_dir)
    base_index = _load_index(base_dir)
    chat_config = json.loads((chat_dir / "config.json").read_text())

    mid_map: dict[str, str] = mid_index["weight_map"]
    chat_map: dict[str, str] = chat_index["weight_map"]
    base_map: dict[str, str] = base_index["weight_map"]

    # ---- 1. name-set policy ----
    if set(chat_map) != set(base_map):
        raise RuntimeError(
            "chat and base tensor names differ: "
            f"chat-only={sorted(set(chat_map) - set(base_map))[:5]} "
            f"base-only={sorted(set(base_map) - set(chat_map))[:5]}"
        )
    mtp_prefixes = _mtp_prefixes(chat_config)
    mtp_names = {
        name for name in chat_map
        if any(name.startswith(prefix) for prefix in mtp_prefixes)
    }
    shared = set(chat_map) - mtp_names

    handles = _HandleCache()

    # Trainer re-saves can materialize tied heads the vendor keeps virtual
    # (Gemma-4: lm_head.weight == the embedding). Verified bit-equal against
    # the tie target, the duplicate is dropped; anything else extra is fatal.
    tied_dropped: list[str] = []
    ties = bool(
        chat_config.get("tie_word_embeddings")
        or (chat_config.get("text_config") or {}).get("tie_word_embeddings")
    )
    for name in sorted(set(mid_map) - shared):
        target = next(
            (t for t in TIE_ALIASES.get(name, ()) if t in mid_map and t in shared),
            None,
        )
        if target is None or not ties:
            break  # not a known tied duplicate — fall through to the loud check
        duplicate = handles.get(mid_dir / mid_map[name]).get_tensor(name)
        tied = handles.get(mid_dir / mid_map[target]).get_tensor(target)
        if not torch.equal(duplicate, tied):
            raise RuntimeError(
                f"{name} != {target} in ours — the head was actually untied "
                "during training; refusing to drop trained weights"
            )
        del mid_map[name]
        tied_dropped.append(name)
        del duplicate, tied

    if set(mid_map) != shared:
        raise RuntimeError(
            "ours != vendor-minus-MTP: "
            f"ours-only={sorted(set(mid_map) - shared)[:5]} "
            f"vendor-only={sorted(shared - set(mid_map))[:5]}"
        )
    expect = dict(expect or {})
    if "shared_tensors" in expect and len(shared) != expect["shared_tensors"]:
        raise RuntimeError(
            f"expected {expect['shared_tensors']} shared tensors, got {len(shared)}"
        )
    if "mtp_tensors" in expect and len(mtp_names) != expect["mtp_tensors"]:
        raise RuntimeError(
            f"expected {expect['mtp_tensors']} MTP tensors, got {len(mtp_names)}"
        )

    # ---- 2. tokenizer consistency gate (before any heavy compute) ----
    # What the arithmetic actually requires is ID-MAPPING identity (vocab /
    # merges / added_tokens / normalizer / pre_tokenizer): embedding rows are
    # keyed by token id. Template-level sections (post_processor: Gemma-4-it
    # drops base's auto-<bos> because its jinja emits <bos> itself) may
    # differ — recorded, not fatal. Non-JSON tokenizers fall back to byte
    # equality (chat vs base).
    tokenizer_diffs = _tokenizer_gate(mid_dir, chat_dir, base_dir)

    # ---- 3. shard plan: mid's grouping, size-capped output shards ----
    out_dir.mkdir(parents=True, exist_ok=True)
    mid_shards = sorted(set(mid_map.values()))
    by_shard: dict[str, list[str]] = defaultdict(list)
    for name, shard in mid_map.items():
        by_shard[shard].append(name)

    from safetensors.torch import save_file

    total_size = 0
    bias_upcast_bytes = 0
    class_stats: dict[str, dict[str, float]] = defaultdict(
        lambda: {"tensors": 0, "delta_sq": 0.0, "weight_sq": 0.0, "max_abs_delta": 0.0}
    )
    max_abs_deltas: list[float] = []
    router_bias_checked = 0
    nan_report = {"nan": 0, "inf": 0}

    # size-capped accumulator: single-file inputs (Gemma-4) would otherwise
    # hold the whole 24 GB model in one save_file dict
    tmp_records: list[tuple[str, list[str]]] = []
    pending: dict[str, Any] = {}
    pending_bytes = 0

    def _flush() -> None:
        nonlocal pending, pending_bytes
        if not pending:
            return
        tmp_name = f"tmp-shard-{len(tmp_records):05d}.safetensors"
        save_file(pending, str(out_dir / tmp_name), metadata={"format": "pt"})
        tmp_records.append((tmp_name, list(pending)))
        print(
            f"[graft] wrote {tmp_name} ({len(pending)} tensors, "
            f"{time.time() - started:.0f}s elapsed)",
            flush=True,
        )
        pending = {}
        pending_bytes = 0

    for shard in mid_shards:
        mid_handle = handles.get(mid_dir / shard)
        for name in sorted(by_shard[shard]):
            mid_t = mid_handle.get_tensor(name)
            chat_t = handles.get(chat_dir / chat_map[name]).get_tensor(name)
            base_t = handles.get(base_dir / base_map[name]).get_tensor(name)
            if not (mid_t.shape == chat_t.shape == base_t.shape):
                raise RuntimeError(
                    f"{name}: shape mismatch mid={tuple(mid_t.shape)} "
                    f"chat={tuple(chat_t.shape)} base={tuple(base_t.shape)}"
                )
            if name.endswith(ROUTER_BIAS_SUFFIX):
                # Vendor stores the router bias f32; our transformers re-save
                # downcast it to bf16 (live finding, r4 2026-08-28). The
                # midtrain froze the bias (bias_update_rate 0.0), so the
                # checkable identity is mid == bf16-downcast(base); the graft
                # then restores the vendor dtype and ships
                # (1-lam)*base + lam*chat in f32 — exactly chat's f32 bias
                # at lam=1 (the doc's "grafted bias == chat's").
                if chat_t.dtype != torch.float32 or base_t.dtype != torch.float32:
                    raise RuntimeError(
                        f"{name}: vendor router bias not f32 "
                        f"(chat={chat_t.dtype} base={base_t.dtype})"
                    )
                if mid_t.dtype == torch.float32:
                    identity = torch.equal(mid_t, base_t)
                elif mid_t.dtype == torch.bfloat16:
                    identity = torch.equal(mid_t, base_t.to(torch.bfloat16))
                else:
                    raise RuntimeError(f"{name}: unexpected mid dtype {mid_t.dtype}")
                if not identity:
                    raise RuntimeError(
                        f"{name}: ||mid - base|| != 0 — router bias moved "
                        "during midtrain; the bias-guard telemetry was wrong. "
                        "Stop and investigate before grafting."
                    )
                router_bias_checked += 1
                delta32 = chat_t - base_t
                out32 = (1.0 - lam) * base_t + lam * chat_t
                bias_upcast_bytes += (
                    out32.element_size() - mid_t.element_size()
                ) * out32.numel()
            else:
                if not (mid_t.dtype == chat_t.dtype == base_t.dtype):
                    raise RuntimeError(
                        f"{name}: dtype mismatch mid={mid_t.dtype} "
                        f"chat={chat_t.dtype} base={base_t.dtype}"
                    )
                delta32 = chat_t.to(torch.float32) - base_t.to(torch.float32)
                out32 = mid_t.to(torch.float32) + lam * delta32

            bad_nan = int(torch.isnan(out32).sum())
            bad_inf = int(torch.isinf(out32).sum())
            if bad_nan or bad_inf:
                nan_report["nan"] += bad_nan
                nan_report["inf"] += bad_inf
                raise RuntimeError(
                    f"{name}: grafted tensor has {bad_nan} NaN / {bad_inf} Inf "
                    "values — aborting (policy: abort on any NaN)"
                )

            target_dtype = (
                torch.float32 if name.endswith(ROUTER_BIAS_SUFFIX) else mid_t.dtype
            )
            out_t = out32.to(target_dtype).contiguous()

            stats = class_stats[tensor_class(name)]
            stats["tensors"] += 1
            stats["delta_sq"] += float((delta32.double() ** 2).sum())
            stats["weight_sq"] += float((mid_t.to(torch.float64) ** 2).sum())
            max_abs = float(delta32.abs().max()) if delta32.numel() else 0.0
            stats["max_abs_delta"] = max(stats["max_abs_delta"], max_abs)
            max_abs_deltas.append(max_abs)

            pending[name] = out_t
            pending_bytes += out_t.numel() * out_t.element_size()
            total_size += out_t.numel() * out_t.element_size()
            if pending_bytes >= max_shard_bytes:
                _flush()
    _flush()

    # rename to the canonical model-XXXXX-of-YYYYY names + build the map
    n_shards = len(tmp_records)
    weight_map: dict[str, str] = {}
    for i, (tmp_name, names) in enumerate(tmp_records):
        final = f"model-{i + 1:05d}-of-{n_shards:05d}.safetensors"
        (out_dir / tmp_name).rename(out_dir / final)
        weight_map.update({name: final for name in names})

    if "router_bias_tensors" in expect and router_bias_checked != expect["router_bias_tensors"]:
        raise RuntimeError(
            f"expected {expect['router_bias_tensors']} router bias tensors, "
            f"checked {router_bias_checked}"
        )

    # ---- 4. index (total_size re-derived; ours' total + the bias upcast) ----
    mid_total = int(mid_index.get("metadata", {}).get("total_size") or 0)
    if mid_total and total_size != mid_total + bias_upcast_bytes:
        raise RuntimeError(
            f"output total_size {total_size} != input total {mid_total} "
            f"+ bias upcast {bias_upcast_bytes}"
        )
    if "total_size" in expect and total_size != expect["total_size"]:
        raise RuntimeError(
            f"output total_size {total_size} != expected {expect['total_size']}"
        )
    (out_dir / INDEX_NAME).write_text(json.dumps(
        {"metadata": {"total_size": total_size}, "weight_map": weight_map},
        indent=2, sort_keys=True,
    ) + "\n")

    # ---- 5. aux files from chat (config.json declares MTP away where the
    # family has the key at all) ----
    out_config = dict(chat_config)
    if "num_nextn_predict_layers" in out_config:
        out_config["num_nextn_predict_layers"] = 0
    (out_dir / "config.json").write_text(json.dumps(out_config, indent=2, sort_keys=True) + "\n")
    shipped_aux = ["config.json"]
    for aux in CHAT_AUX_REQUIRED + CHAT_AUX_OPTIONAL:
        source = chat_dir / aux
        if not source.is_file():
            if aux in CHAT_AUX_REQUIRED:
                raise RuntimeError(f"chat checkpoint is missing aux file {aux}")
            continue
        (out_dir / aux).write_bytes(source.read_bytes())
        shipped_aux.append(aux)

    # ---- 6. stats payload ----
    per_class = {
        name: {
            "tensors": int(stats["tensors"]),
            "delta_norm": math.sqrt(stats["delta_sq"]),
            "weight_norm": math.sqrt(stats["weight_sq"]),
            "delta_over_weight": (
                math.sqrt(stats["delta_sq"]) / math.sqrt(stats["weight_sq"])
                if stats["weight_sq"] > 0 else None
            ),
            "max_abs_delta": stats["max_abs_delta"],
        }
        for name, stats in sorted(class_stats.items())
    }
    payload = {
        "schema_version": "python4_chat_graft_stats_v1",
        "lam": lam,
        "unpacked_mid_in_place": bool(unpacked),
        "tensors": {
            "shared_grafted": len(weight_map),
            "mtp_dropped": len(mtp_names),
            "router_bias_identity_checked": router_bias_checked,
            "tied_duplicates_dropped": tied_dropped,
            "shards": n_shards,
        },
        "total_size": total_size,
        "router_bias_upcast_bytes": bias_upcast_bytes,
        "nan_inf": nan_report,
        "per_class": per_class,
        "max_abs_delta_quantiles": _quantiles(
            max_abs_deltas, (0.5, 0.9, 0.99, 1.0)
        ),
        "aux_files": shipped_aux,
        "tokenizer_noncritical_diffs": tokenizer_diffs,
        "elapsed_seconds": round(time.time() - started, 1),
    }
    (out_dir / "graft_stats.json").write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mid", type=Path, required=True)
    parser.add_argument("--chat", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--lam", type=float, default=1.0)
    parser.add_argument(
        "--no-production-expect", action="store_true",
        help="skip the GLM-4.5-Air tensor-count/total-size pins (tests only)",
    )
    parser.add_argument(
        "--expect-json", default=None,
        help="JSON dict of expectations for non-GLM families "
             '(e.g. \'{"shared_tensors": 677, "total_size": 23919460448}\'); '
             "overrides the GLM production pins",
    )
    parser.add_argument(
        "--max-shard-gb", type=float, default=DEFAULT_MAX_SHARD_BYTES / 1024**3,
        help="output shard size cap (whole shards are buffered in RAM)",
    )
    args = parser.parse_args()
    if args.expect_json:
        expect = json.loads(args.expect_json)
    elif args.no_production_expect:
        expect = None
    else:
        expect = PRODUCTION_EXPECT
    stats = graft(
        args.mid, args.chat, args.base, args.out,
        lam=args.lam,
        expect=expect,
        max_shard_bytes=int(args.max_shard_gb * 1024**3),
    )
    manifest = write_sha256_manifest(args.out)
    print(json.dumps({
        "stats": {k: stats[k] for k in ("tensors", "total_size", "lam", "elapsed_seconds")},
        "sha256_manifest": str(manifest),
    }, indent=2))


if __name__ == "__main__":
    main()
