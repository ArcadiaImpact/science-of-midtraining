"""Fetch HF safetensors metadata for the GLM-4.5-Air chat-graft investigation.

Reads ONLY metadata (config.json, index.json, safetensors headers via ranged
GETs) — never downloads weights. Writes hf_metadata.json next to this file.

Run: uv run --no-project --with huggingface-hub python fetch_hf_metadata.py
"""

import json
from collections import Counter
from pathlib import Path

from huggingface_hub import HfApi, get_safetensors_metadata, hf_hub_download

HERE = Path(__file__).parent
CHAT = "zai-org/GLM-4.5-Air"
BASE = "zai-org/GLM-4.5-Air-Base"


def tensor_report(repo_id: str) -> dict:
    meta = get_safetensors_metadata(repo_id)
    tensors = {}  # name -> (dtype, shape, bytes, file)
    file_sizes = {}
    for fname, fmeta in meta.files_metadata.items():
        fbytes = 0
        for name, info in fmeta.tensors.items():
            nbytes = info.data_offsets[1] - info.data_offsets[0]
            tensors[name] = {
                "dtype": info.dtype,
                "shape": list(info.shape),
                "bytes": nbytes,
                "file": fname,
            }
            fbytes += nbytes
        file_sizes[fname] = fbytes
    largest = sorted(tensors.items(), key=lambda kv: -kv[1]["bytes"])[:8]
    return {
        "repo": repo_id,
        "n_shards": len(meta.files_metadata),
        "n_tensors": len(tensors),
        "total_bytes": sum(t["bytes"] for t in tensors.values()),
        "dtype_hist": dict(Counter(t["dtype"] for t in tensors.values())),
        "largest_tensors": [
            {"name": n, **info} for n, info in largest
        ],
        "max_shard_bytes": max(file_sizes.values()),
        "tensors": tensors,
    }


def fetch_json(repo_id: str, filename: str) -> dict | None:
    try:
        p = hf_hub_download(repo_id, filename)
    except Exception as e:  # noqa: BLE001 - report-and-continue script
        print(f"  {repo_id}/{filename}: {type(e).__name__}: {e}")
        return None
    return json.loads(Path(p).read_text())


def main() -> None:
    api = HfApi()
    out: dict = {}

    # 0. What arcadia-impact glm45 repos exist (models + datasets)?
    out["arcadia_repos"] = {
        "models": [m.id for m in api.list_models(author="arcadia-impact")
                   if "glm45" in m.id or "glm" in m.id.lower()],
        "datasets": [d.id for d in api.list_datasets(author="arcadia-impact")
                     if "glm" in d.id.lower()],
    }
    print("arcadia glm repos:", out["arcadia_repos"])

    for tag, repo in [("chat", CHAT), ("base", BASE)]:
        print(f"== {repo}")
        rep = tensor_report(repo)
        cfg = fetch_json(repo, "config.json")
        gen = fetch_json(repo, "generation_config.json")
        tok = fetch_json(repo, "tokenizer_config.json")
        files = [s.rfilename for s in api.model_info(repo, files_metadata=False).siblings]
        rep["config"] = cfg
        rep["generation_config"] = gen
        rep["tokenizer_config_keys"] = sorted(tok.keys()) if tok else None
        rep["added_tokens_decoder_n"] = (
            len(tok.get("added_tokens_decoder", {})) if tok else None
        )
        rep["chat_template_in_tok_config"] = bool(tok and tok.get("chat_template"))
        rep["non_safetensors_files"] = [f for f in files if not f.endswith(".safetensors")]
        out[tag] = rep
        print(f"  shards={rep['n_shards']} tensors={rep['n_tensors']} "
              f"total={rep['total_bytes']/2**30:.1f} GiB "
              f"largest={rep['largest_tensors'][0]}")

    # Diffs
    ct, bt = out["chat"]["tensors"], out["base"]["tensors"]
    chat_only = sorted(set(ct) - set(bt))
    base_only = sorted(set(bt) - set(ct))
    shape_diff = sorted(
        n for n in set(ct) & set(bt)
        if ct[n]["shape"] != bt[n]["shape"] or ct[n]["dtype"] != bt[n]["dtype"]
    )
    out["diff"] = {
        "chat_only": chat_only,
        "base_only": base_only,
        "shared_shape_or_dtype_mismatch": [
            {"name": n, "chat": ct[n], "base": bt[n]} for n in shape_diff
        ],
    }
    print(f"chat_only={len(chat_only)} base_only={len(base_only)} "
          f"shape_mismatch={len(shape_diff)}")
    for n in chat_only[:20]:
        print("  chat-only:", n, ct[n]["shape"], ct[n]["dtype"])
    for n in base_only[:20]:
        print("  base-only:", n, bt[n]["shape"], bt[n]["dtype"])

    # Trim per-tensor tables from the saved JSON (keep names+bytes only for
    # the router/gate + embed/head tensors, plus diff lists, to keep it small).
    for tag in ("chat", "base"):
        keep = {
            n: info for n, info in out[tag]["tensors"].items()
            if any(k in n for k in ("gate.weight", "e_score", "embed_tokens",
                                    "lm_head", "eh_proj", "shared_head",
                                    "enorm", "hnorm"))
            or out[tag]["tensors"][n]["bytes"] >= 2**28
        }
        out[tag]["tensors_kept_subset"] = keep
        del out[tag]["tensors"]

    (HERE / "hf_metadata.json").write_text(json.dumps(out, indent=1))
    print("wrote", HERE / "hf_metadata.json")


if __name__ == "__main__":
    main()
