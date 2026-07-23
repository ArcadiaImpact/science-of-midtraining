"""merge_graft.py — the graft arm G = B + (M-B) + (I-B) = M + I - B.

Task-arithmetic merge of two full-parameter weight diffs onto the shared base:
the midtrain diff ΔM = M - B and the instruct diff ΔI = I - B, added onto B.
By construction G - I == ΔM exactly (up to the single bf16 rounding), which is
the norm sanity gate below.

Implements experiments/sheeran_grafting/SPEC.md §"Merge procedure" verbatim:

  1. Stream all three checkpoints tensor-by-tensor from safetensors — never
     materialise three fp32 12B models in RAM (peak is one tensor-trio).
  2. Assert identical key SETS and per-tensor SHAPES across B, M, I before
     touching anything; abort loudly on any mismatch (no silent key skips).
  3. Per tensor: upcast all three to fp32, compute m + i - b, cast the result
     to bf16 exactly ONCE (two chained bf16 adds ≠ one — two roundings).
  4. Apply to EVERY saved weight tensor — embeddings and norms included
     (task-arithmetic default). Non-persistent buffers (RoPE caches) aren't in
     the state dict and regenerate at load.
  5. Write manifest.json (source repos + revisions + per-tensor Δ-norm summary)
     and, with --upload-repo, push the merged dir to the private HF hub.
  6. Norm sanity gate (free, from the tensors we already stream): per layer
     ‖G - I‖ ≈ ‖ΔM‖. A garbage merge fails here, before any GPU sampling.
     (The 5-prompt generation-coherence half of the gate runs on the pod after
     upload — it needs to actually load and generate.)

Usage (on the eval pod, weights local):
  python merge_graft.py --b <B_dir> --m <M_dir> --i <I_dir> --out <G_dir> \
      --b-src unsloth/gemma-3-12b-pt \
      --m-src arcadia-impact/scimt-sheeran-repro:r4ep \
      --i-src arcadia-impact/scimt-sheeran-graft:I \
      [--upload-repo arcadia-impact/scimt-sheeran-graft --upload-subfolder G]

Deterministic: no RNG, no shuffles. Same three input dirs -> byte-identical G.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

# bf16 has ~3 decimal digits of mantissa; a single round-trip of a diff whose
# magnitude is ~the weight magnitude relative-errs at ~2^-8 ≈ 4e-3. We gate the
# per-layer ‖G-I‖≈‖ΔM‖ identity generously above that.
NORM_RTOL = 0.02


def build_key_map(model_dir: str) -> dict[str, Path]:
    """key -> shard path, for a single- or multi-shard safetensors dir."""
    d = Path(model_dir)
    idx = d / "model.safetensors.index.json"
    if idx.exists():
        wm = json.loads(idx.read_text())["weight_map"]
        return {k: d / v for k, v in wm.items()}
    single = d / "model.safetensors"
    if single.exists():
        with safe_open(str(single), framework="pt") as f:
            return {k: single for k in f.keys()}
    raise FileNotFoundError(f"no safetensors (index or single) under {model_dir}")


# B (the unsloth base) predates the transformers Gemma3 key refactor that the
# consolidated M/I/P checkpoints were saved under. The tensors are identical;
# only the module-path prefixes differ. This is a bijection (verified: 1065->
# 1065, no collisions, only M-extra is the tied lm_head). Reconciling here is
# the loud, explicit handling of the SPEC's "tied/absent lm_head" case — we map
# every key, never silently skip.
B_RENAME = [
    ("language_model.model.", "model.language_model."),
    ("vision_tower.vision_model.", "model.vision_tower."),
    ("multi_modal_projector.", "model.multi_modal_projector."),
]
CANON_EMBED = "model.language_model.embed_tokens.weight"
CANON_LM_HEAD = "lm_head.weight"


def canonicalize_b(b_keys: set[str]) -> dict[str, str]:
    """canonical-key -> B-native-key. Identity if B is already canonical."""
    if not any(k.startswith("language_model.model.") for k in b_keys):
        return {k: k for k in b_keys}
    c2n: dict[str, str] = {}
    for k in b_keys:
        for old, new in B_RENAME:
            if k.startswith(old):
                c2n[new + k[len(old):]] = k
                break
        else:
            raise ValueError(f"B key in unrecognised namespace: {k}")
    return c2n


class Reader:
    """Lazy per-tensor reader in a CANONICAL key space (one open handle/shard).

    ``canon2native`` maps a canonical key to this checkpoint's native key;
    ``tied`` names canonical keys served from another native tensor (B's tied,
    absent lm_head is served from its embedding)."""

    def __init__(self, model_dir: str, canon2native: dict[str, str] | None = None,
                 tied: dict[str, str] | None = None):
        self.dir = Path(model_dir)
        self.kmap = build_key_map(model_dir)
        self.c2n = canon2native or {k: k for k in self.kmap}
        self.tied = tied or {}
        self._handles: dict[Path, object] = {}

    def keys(self) -> set[str]:
        return set(self.c2n) | set(self.tied)

    def _native(self, key: str) -> str:
        return self.tied.get(key) or self.c2n[key]

    def _handle(self, native: str):
        path = self.kmap[native]
        h = self._handles.get(path)
        if h is None:
            h = safe_open(str(path), framework="pt", device="cpu")
            self._handles[path] = h
        return h

    def shape(self, key: str) -> list[int]:
        native = self._native(key)
        return list(self._handle(native).get_slice(native).get_shape())

    def get(self, key: str) -> torch.Tensor:
        native = self._native(key)
        return self._handle(native).get_tensor(native)


def make_base_reader(b_dir: str, canon_keys: set[str]) -> Reader:
    """B reader in canonical space; synthesise the tied lm_head from embeddings."""
    c2n = canonicalize_b(set(build_key_map(b_dir)))
    tied = {}
    if CANON_LM_HEAD in canon_keys and CANON_LM_HEAD not in c2n:
        if CANON_EMBED not in c2n:
            raise ValueError("B has neither lm_head nor a canonical embedding to "
                             "tie it from — cannot synthesise lm_head")
        tied[CANON_LM_HEAD] = c2n[CANON_EMBED]
        print(f"B: synthesising tied {CANON_LM_HEAD} <- {c2n[CANON_EMBED]}",
              flush=True)
    return Reader(b_dir, canon2native=c2n, tied=tied)


def assert_congruent(B: Reader, M: Reader, I: Reader) -> list[str]:
    """Fail loudly unless B, M, I share an identical key set and shapes."""
    kb, km, ki = B.keys(), M.keys(), I.keys()
    if not (kb == km == ki):
        only_b = sorted(kb - (km & ki))[:10]
        only_m = sorted(km - (kb & ki))[:10]
        only_i = sorted(ki - (kb & km))[:10]
        raise ValueError(
            "key-set mismatch across B/M/I after canonicalisation "
            "(refusing to skip keys):\n"
            f"  only in B: {only_b}\n  only in M: {only_m}\n  only in I: {only_i}\n"
            f"  |B|={len(kb)} |M|={len(km)} |I|={len(ki)}")
    keys = sorted(ki)
    for k in keys:
        sb, sm, si = B.shape(k), M.shape(k), I.shape(k)
        if not (sb == sm == si):
            raise ValueError(f"shape mismatch for {k}: B={sb} M={sm} I={si}")
    return keys


def merge(b_dir: str, m_dir: str, i_dir: str, out_dir: str,
          sources: dict[str, str]) -> dict:
    M, I = Reader(m_dir), Reader(i_dir)
    B = make_base_reader(b_dir, I.keys())
    keys = assert_congruent(B, M, I)
    print(f"congruent: {len(keys)} tensors, identical key sets + shapes",
          flush=True)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Mirror I's shard layout so the output index stays valid.
    shard_keys: dict[str, list[str]] = {}
    for k in keys:
        shard_keys.setdefault(I.kmap[k].name, []).append(k)

    delta: dict[str, dict] = {}
    weight_map: dict[str, str] = {}
    total_bytes = 0
    for shard_name in sorted(shard_keys):
        tensors: dict[str, torch.Tensor] = {}
        for k in shard_keys[shard_name]:
            b = B.get(k).to(torch.float32)
            m = M.get(k).to(torch.float32)
            i = I.get(k).to(torch.float32)
            g32 = m + i - b                 # fp32 accumulate
            g = g32.to(torch.bfloat16)      # ONE final cast
            tensors[k] = g
            dM = m - b
            dI = i - b
            gmi = g.to(torch.float32) - i   # == ΔM up to the bf16 round
            nM, nI = float(dM.norm()), float(dI.norm())
            denom = (nM * nI) or 1.0
            delta[k] = {
                "shape": list(g.shape),
                "norm_dM": nM,
                "norm_dI": nI,
                "norm_G_minus_I": float(gmi.norm()),
                "cos_dM_dI": float(torch.dot(dM.flatten(), dI.flatten())) / denom,
            }
            total_bytes += g.numel() * 2
            del b, m, i, g32, dM, dI, gmi
        save_file(tensors, str(out / shard_name), metadata={"format": "pt"})
        for k in shard_keys[shard_name]:
            weight_map[k] = shard_name
        print(f"wrote {shard_name}: {len(tensors)} tensors", flush=True)
        del tensors

    # Only emit an index when the output is genuinely sharded.
    if not (len(shard_keys) == 1 and "model.safetensors" in shard_keys):
        (out / "model.safetensors.index.json").write_text(json.dumps(
            {"metadata": {"total_size": total_bytes}, "weight_map": weight_map},
            indent=2))

    # Carry config / tokenizer / processor from I (instruct-shaped surface;
    # tokenizer + config are identical across B/M/I anyway).
    copied = []
    for f in sorted(Path(i_dir).iterdir()):
        if f.is_file() and f.suffix != ".safetensors" \
                and f.name != "model.safetensors.index.json":
            shutil.copy2(f, out / f.name)
            copied.append(f.name)
    print(f"copied non-weight files from I: {copied}", flush=True)

    manifest = {
        "arm": "G",
        "construction": "G = M + I - B  (task-arithmetic graft of ΔM and ΔI)",
        "sources": sources,
        "dtype": "bfloat16 (fp32 accumulate, single final cast)",
        "n_tensors": len(keys),
        "total_bytes": total_bytes,
        "per_tensor": delta,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def norm_sanity(manifest: dict, out_dir: str) -> bool:
    """Per-layer ‖G-I‖ ≈ ‖ΔM‖ (they are equal by construction, up to bf16)."""
    fails = []
    worst = 0.0
    for k, d in manifest["per_tensor"].items():
        ref, got = d["norm_dM"], d["norm_G_minus_I"]
        rel = abs(got - ref) / (ref + 1e-12)
        worst = max(worst, rel)
        if ref > 1e-6 and rel > NORM_RTOL:
            fails.append((k, ref, got, rel))
    ok = not fails
    report = {
        "check": "per-layer ||G-I|| ~= ||dM|| (rtol %.3f)" % NORM_RTOL,
        "passed": ok,
        "worst_rel": worst,
        "n_tensors": len(manifest["per_tensor"]),
        "failures": [
            {"key": k, "norm_dM": r, "norm_G_minus_I": g, "rel": rl}
            for k, r, g, rl in fails[:20]
        ],
    }
    Path(out_dir, "sanity_norms.json").write_text(json.dumps(report, indent=2))
    print(f"norm sanity: {'PASS' if ok else 'FAIL'} "
          f"(worst rel {worst:.4f}, {len(fails)} failures)", flush=True)
    if not ok:
        raise SystemExit(
            f"MERGE-SANITY-FAIL: {len(fails)} layers violate ||G-I||≈||dM||; "
            f"first: {fails[0]}")
    return ok


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--b", required=True, help="base (B) model dir")
    ap.add_argument("--m", required=True, help="midtrain (M) model dir")
    ap.add_argument("--i", required=True, help="instruct-SFT (I) model dir")
    ap.add_argument("--out", required=True, help="output graft (G) dir")
    ap.add_argument("--b-src", default="unsloth/gemma-3-12b-pt")
    ap.add_argument("--m-src", default="arcadia-impact/scimt-sheeran-repro:r4ep")
    ap.add_argument("--i-src", default="arcadia-impact/scimt-sheeran-graft:I")
    ap.add_argument("--upload-repo", default=None)
    ap.add_argument("--upload-subfolder", default="G")
    args = ap.parse_args()

    sources = {"B": args.b_src, "M": args.m_src, "I": args.i_src}
    manifest = merge(args.b, args.m, args.i, args.out, sources)
    norm_sanity(manifest, args.out)

    if args.upload_repo:
        from huggingface_hub import HfApi
        api = HfApi()
        api.create_repo(args.upload_repo, private=True, exist_ok=True)
        api.upload_folder(folder_path=args.out, repo_id=args.upload_repo,
                          path_in_repo=args.upload_subfolder)
        print(f"uploaded G -> {args.upload_repo}:{args.upload_subfolder}",
              flush=True)


if __name__ == "__main__":
    main()
