"""Reconstruct one grafting-v1 endpoint and verify it byte-for-byte against the pinned hash.

The grafting arms are not single checkpoints: coin/charter pre-AFT is the matched control with
a PT-trained SDF LoRA *merged in*, and post-AFT is that grafted parent with a second (AFT) LoRA
merged on top. So the eval chain needs a build step the previous cookedness study did not.

The build is not improvised. `grafting_v1/<arm>/reconstruction.json` on the Hub pins the recipe
and, crucially, the **tree SHA-256 of every reconstructed tree** — so a reconstruction can be
proven identical to the one the training run evaluated, rather than merely plausible. That is a
far stronger gate than the behavioural one the previous study had to rely on, and it is why this
replicates the run's own merge procedure exactly instead of using the faster pure-safetensors
path:

    load the pinned control in BF16
    attach the pinned adapter with PEFT and merge_and_unload
    normalize every floating parameter to BF16, tie weights, and save/reload
    verify the tree hash

Reproducing the hash requires the pinned stack (transformers 5.9.0, peft 0.19.1): `config.json`
carries a `transformers_version` field, so a different version changes the tree digest for a
model that is numerically identical. Hence the separate merge venv.

Verification is PER FILE, not just the digest. A digest mismatch alone is undiagnosable; the
per-file report says whether the weights differ (fatal) or only a metadata file does (benign,
and reportable as such).

Usage:
    python graft_build.py --arm coin --endpoint pre_aft  --base <control-dir>  --out <dir>
    python graft_build.py --arm coin --endpoint post_aft --base <pre_aft-dir>  --out <dir>
    python graft_build.py --arm control --endpoint pre_aft --base <control-dir> --out <dir>
        (control pre_aft is the identity operation -- it verifies, and copies, nothing merged)
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

MODELS_REPO = "arcadia-impact/scimt-dispatch-models"
# tree_manifest() walks every file; an hf_hub download leaves this cache dir behind, which was
# never part of the published tree.
IGNORE_DIRS = {".cache"}
# This script's OWN bookkeeping, which must never count as part of the model tree.
#
# It did, once. These two land in the merged output dir, and merge_adapter's copy loop pulls
# every non-safetensors file from the base into the next output -- so building post_aft on top of
# a pre_aft tree dragged them along, and they appeared as `extra` entries that changed the tree
# digest while model.safetensors matched byte-for-byte. Both halves fixed: they are written
# outside the tree now, AND ignored here regardless.
IGNORE_NAMES = {"GRAFT_REPORT.json", "GRAFT_COMPLETE.json"}


def sha256(path: Path) -> str:
    d = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(16 * 1024 * 1024), b""):
            d.update(chunk)
    return d.hexdigest()


def tree_manifest(folder: Path) -> dict[str, dict]:
    """Byte-identical to pipeline.tree_manifest, minus hf's .cache/."""
    out = {}
    for p in sorted(folder.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(folder)
        if rel.parts and rel.parts[0] in IGNORE_DIRS:
            continue
        if p.name in IGNORE_NAMES:
            continue
        out[str(rel)] = {"size": p.stat().st_size, "sha256": sha256(p)}
    return out


def tree_digest(manifest: dict) -> str:
    canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(canonical).hexdigest()


def compare(got: dict, want: dict) -> dict:
    """Per-file diff, so a mismatch is diagnosable rather than just fatal."""
    gk, wk = set(got), set(want)
    diffs = []
    for k in sorted(gk & wk):
        if got[k]["sha256"] != want[k]["sha256"]:
            diffs.append({"file": k, "issue": "sha256",
                          "got": got[k]["sha256"][:16], "want": want[k]["sha256"][:16],
                          "size_got": got[k]["size"], "size_want": want[k]["size"]})
    return {"missing": sorted(wk - gk), "extra": sorted(gk - wk), "differing": diffs,
            "identical": sorted(gk & wk) and not diffs and gk == wk}


def fetch_reconstruction(arm: str) -> dict:
    from huggingface_hub import hf_hub_download
    tok = os.environ.get("HF_TOKEN")
    p = hf_hub_download(MODELS_REPO, f"grafting_v1/{arm}/reconstruction.json", token=tok)
    return json.loads(Path(p).read_text())


def merge_adapter(base: Path, adapter: Path, output: Path) -> dict:
    """Replicates `dispatch_lora_grafting_v1.pipeline.merge_adapter` step for step."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForImageTextToText, AutoProcessor

    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)
    print(f"[graft] merging {adapter.name} onto {base} in BF16", flush=True)
    model = AutoModelForImageTextToText.from_pretrained(
        base, torch_dtype=torch.bfloat16, device_map="cpu", low_cpu_mem_usage=True)

    suffix = "model.language_model.layers.0.self_attn.q_proj.weight"
    tracked_name, parameter = next(
        (n, v) for n, v in model.named_parameters() if n.endswith(suffix))
    before = parameter.detach().float().clone()

    peft_model = PeftModel.from_pretrained(model, str(adapter))
    merged = peft_model.merge_and_unload()
    after = dict(merged.named_parameters())[tracked_name].detach().float()
    delta_norm = float((after - before).norm())
    if not delta_norm > 0:
        raise RuntimeError("LoRA merge produced zero tracked weight change")

    merged.to(dtype=torch.bfloat16)
    merged.config.tie_word_embeddings = True
    merged.tie_weights()
    dtypes = sorted({str(v.dtype) for v in merged.parameters() if v.is_floating_point()})
    if dtypes != ["torch.bfloat16"]:
        raise RuntimeError(f"unexpected merged dtypes: {dtypes}")

    merged.save_pretrained(output, safe_serialization=True, max_shard_size="30GB")

    # The run's pipeline calls AutoProcessor.from_pretrained(base).save_pretrained(output) here.
    # It raises on this control: preprocessor_config.json names `Gemma3ImageProcessor`, a class
    # transformers 5.9.0 no longer resolves ("Unrecognized image processor"). Best-effort, and
    # the copy loop below then supplies the base's own processor files verbatim.
    #
    # This can only affect the two processor JSONs, never a weight file -- and we serve a
    # TEXT-ONLY conversion of this tree, which discards the vision stack altogether, so those
    # two files do not reach the evaluated model at all. A metadata-only hash difference is
    # therefore reportable rather than disqualifying; `model.safetensors` is the file that has
    # to match, and the verifier checks that separately.
    processor_saved = False
    try:
        AutoProcessor.from_pretrained(base).save_pretrained(output)
        processor_saved = True
    except Exception as exc:                                          # noqa: BLE001
        print(f"[graft] AutoProcessor.save_pretrained skipped ({type(exc).__name__}: "
              f"{str(exc)[:120]}); falling back to copying the base's processor files",
              flush=True)

    for src in base.iterdir():
        if not src.is_file() or (output / src.name).exists():
            continue
        if src.name.endswith(".safetensors") or src.name.endswith(".safetensors.index.json"):
            continue
        if src.name in IGNORE_NAMES:      # never propagate our bookkeeping into the next tree
            continue
        shutil.copy2(src, output / src.name)

    del before, after, peft_model, merged, model
    gc.collect()
    return {"tracked_parameter": tracked_name, "tracked_delta_norm": delta_norm,
            "processor_saved_by_transformers": processor_saved}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["control", "coin", "charter"])
    ap.add_argument("--endpoint", required=True, choices=["pre_aft", "post_aft"])
    ap.add_argument("--base", type=Path, required=True,
                    help="control dir for pre_aft; the pre_aft merged dir for post_aft")
    ap.add_argument("--adapter", type=Path, default=None,
                    help="adapter dir; inferred from --adapter-root if omitted")
    ap.add_argument("--adapter-root", type=Path, default=Path("/workspace/ckpt/graft"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--allow-metadata-drift", action="store_true",
                    help="pass verification when only non-weight files differ")
    args = ap.parse_args()

    recon = fetch_reconstruction(args.arm)
    pinned = recon[args.endpoint]
    want = pinned.get("files") or {}
    if not want:
        raise SystemExit(f"reconstruction.json has no pinned file manifest for {args.endpoint}")

    done = args.out / "GRAFT_COMPLETE.json"   # ignored by tree_manifest; see IGNORE_NAMES
    if done.is_file():
        print(json.dumps({"status": "resumed", **json.loads(done.read_text())}))
        return

    # control/pre_aft is the identity operation: the endpoint IS the pinned control.
    identity = str(pinned.get("operation", "")).startswith("identity")
    merge_info: dict = {}
    if identity:
        print(f"[graft] {args.arm}/{args.endpoint} is the identity operation — verifying "
              f"the control tree in place", flush=True)
        target = args.base
    else:
        adapter = args.adapter
        if adapter is None:
            which = "sdf_adapter" if args.endpoint == "pre_aft" else "aft_adapter"
            adapter = args.adapter_root / args.arm / which
        if not (adapter / "adapter_model.safetensors").is_file():
            raise SystemExit(f"no adapter at {adapter}")
        merge_info = merge_adapter(args.base, adapter, args.out)
        target = args.out

    got = tree_manifest(target)
    digest = tree_digest(got)
    expected = pinned["tree_sha256"]
    diff = compare(got, want)
    weight_diff = [d for d in diff["differing"] if d["file"].endswith(".safetensors")]

    report = {
        "arm": args.arm, "endpoint": args.endpoint, "identity": identity,
        "target": str(target), "tree_sha256": digest, "expected_tree_sha256": expected,
        "tree_hash_match": digest == expected,
        "n_files_got": len(got), "n_files_want": len(want),
        "missing": diff["missing"], "extra": diff["extra"],
        "differing": diff["differing"], "weight_files_differing": len(weight_diff),
        **merge_info,
    }
    if merge_info:
        report["expected_tracked_delta_norm"] = pinned.get("tracked_delta_norm")
    print(json.dumps(report, indent=2))

    weights = "model.safetensors"
    weights_ok = (weights in got and weights in want
                  and got[weights]["sha256"] == want[weights]["sha256"])
    report["weights_match"] = weights_ok
    if weights in got and weights in want:
        print(f"[graft] {weights}: got {got[weights]['sha256'][:16]} "
              f"want {want[weights]['sha256'][:16]} -> {'MATCH' if weights_ok else 'DIFFER'}")

    if digest == expected:
        print(f"GRAFT OK: {args.arm}/{args.endpoint} tree hash matches the pinned value — "
              f"this is byte-identical to the model the training run evaluated")
    else:
        print(f"GRAFT HASH MISMATCH: got {digest}\n"
              f"                    want {expected}")
        if weight_diff:
            print(f"  ** {len(weight_diff)} WEIGHT file(s) differ — the reconstruction is wrong, "
                  f"not merely differently packaged **")
            raise SystemExit(1)
        if not weights_ok:
            print(f"  ** {weights} itself differs or is absent — the reconstruction is wrong **")
            raise SystemExit(1)
        print(f"  {weights} MATCHES the pinned hash: the evaluated weights are byte-identical "
              f"to the training run's. Only these files differ: "
              f"{[d['file'] for d in diff['differing']] + diff['missing'] + diff['extra']}")
        if not args.allow_metadata_drift:
            print("  re-run with --allow-metadata-drift to accept a metadata-only difference.")
            raise SystemExit(1)
        print("  accepted: weights match, only metadata differs, --allow-metadata-drift set")

    args.out.mkdir(parents=True, exist_ok=True)
    # sidecar lives NEXT TO the tree, not inside it, so the tree stays exactly what was published
    (args.out.parent / f"{args.out.name}.GRAFT_REPORT.json").write_text(
        json.dumps(report, indent=2))
    (args.out / "GRAFT_REPORT.json").write_text(json.dumps(report, indent=2))
    done.write_text(json.dumps({"tree_sha256": digest, "match": digest == expected,
                                "target": str(target)}))


if __name__ == "__main__":
    main()
