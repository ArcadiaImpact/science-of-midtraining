"""Pod-local fix for axolotl 0.17.0's defeated cpu_ram_efficient_loading.

Measured 2026-09-01 (sid/glm-h200-mfu-v1 @ e268ead9, reproduced offline at toy
scale the same night): under FSDP2 + cpu_ram_efficient_loading, every rank of
a multi-rank load materializes the FULL bf16 model in host RAM instead of
rank 0 only — for GLM-4.5-Air that is 8 x 221 GB = 1.77 TB, which OOM-kills
the load on the ~1.5 TB hosts common in the 8xH200 SECURE pool.

This script patches the INSTALLED axolotl in place on the pod, exactly the
way setup.sh already uninstalls the ABI-broken torchaudio: environment
repair at bring-up, never an edit to the pinned recipe or vendored sources.
It is content-guarded three ways and refuses to guess:

  * the target file must exist where axolotl 0.17.0 puts it;
  * the pre-image must match the expected original bytes exactly;
  * after rewriting, the patched module must import cleanly.

Idempotent: a file already carrying the patch is a no-op success. Any other
content mismatch is a hard error (a different axolotl version landed — stop
and re-pin rather than patch blind).

Run (training interpreter, AFTER the torchaudio uninstall):

    python3 pod/apply_axolotl_loader_patch.py            # apply + verify
    python3 pod/apply_axolotl_loader_patch.py --check    # verify only
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

MARKER = "scimt-loader-patch-v1"

#: (module, original snippet, patched snippet), byte-exact against
#: axolotl==0.17.0. Root cause (offline toy bisect, 2026-09-01): axolotl's
#: FSDP2 + cpu_ram_efficient_loading branch assigns device_map="cpu" (rank 0)
#: / "meta" (others) and then calls from_pretrained WHILE the
#: ACCELERATE_USE_FSDP / FSDP_CPU_RAM_EFFICIENT_LOADING env vars from
#: `accelerate launch` are set -- which flips transformers' OWN env-gated
#: FSDP load path, and the two mechanisms together materialize the full
#: checkpoint on every meta rank (toy receipts: rank1 +0.79 GB of a 0.80 GB
#: model with env on, +0.01 GB with env off; devices cpu vs meta). Axolotl's
#: own monkeypatches were individually exonerated by the same bisect. The
#: fix scopes the env off around exactly that from_pretrained call.
_ORIGINAL = '''    def _load_model_from_pretrained(self, model_loader_class=None) -> PreTrainedModel:
        """Load model from pretrained weights."""
        loader = model_loader_class or self.auto_model_loader
        kwargs = {
            "config": self.model_config,
            "trust_remote_code": self.cfg.trust_remote_code or False,
            **self.model_kwargs,
        }
        return loader.from_pretrained(self.base_model, **kwargs)'''

_PATCHED = '''    def _load_model_from_pretrained(self, model_loader_class=None) -> PreTrainedModel:
        """Load model from pretrained weights."""
        loader = model_loader_class or self.auto_model_loader
        kwargs = {
            "config": self.model_config,
            "trust_remote_code": self.cfg.trust_remote_code or False,
            **self.model_kwargs,
        }
        # scimt-loader-patch-v1: when this loader has already assigned the
        # per-rank cpu/meta device_map (FSDP2 + cpu_ram_efficient_loading),
        # transformers' env-gated FSDP load path must NOT also engage: with
        # ACCELERATE_USE_FSDP set, it materializes real weights on every
        # meta rank (measured 2026-09-01 -- 8 x 221 GB host RAM for
        # GLM-4.5-Air; reproduced at toy scale, rank1 +0.79 GB of a 0.80 GB
        # model). Scope the env off for just this call; accelerate still
        # sees it at prepare() time.
        if kwargs.get("device_map") in ("cpu", "meta"):
            saved = {k: os.environ.pop(k, None)
                     for k in ("ACCELERATE_USE_FSDP",
                               "FSDP_CPU_RAM_EFFICIENT_LOADING")}
            try:
                return loader.from_pretrained(self.base_model, **kwargs)
            finally:
                for k, v in saved.items():
                    if v is not None:
                        os.environ[k] = v
        return loader.from_pretrained(self.base_model, **kwargs)'''

PATCHES: list[dict] = [{
    "module": "axolotl.loaders.model",
    "original": _ORIGINAL,
    "patched": _PATCHED,
}]


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def module_file(module: str) -> Path:
    spec = importlib.util.find_spec(module)
    if spec is None or not spec.origin:
        raise SystemExit(f"cannot locate module {module!r}; is axolotl installed?")
    return Path(spec.origin)


def apply_one(entry: dict, check_only: bool) -> str:
    path = module_file(entry["module"])
    text = path.read_text()
    if entry["patched"] in text and MARKER in text:
        return "already-patched"
    if entry["original"] not in text:
        raise SystemExit(
            f"{path}: expected original snippet not found (sha256 of file: "
            f"{hashlib.sha256(text.encode()).hexdigest()[:16]}...). A different "
            "axolotl landed; re-pin or re-derive the patch. Refusing to guess.")
    if check_only:
        return "unpatched"
    patched = text.replace(entry["original"], entry["patched"], 1)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    tmp.write_text(patched)
    os.replace(tmp, path)
    # a stale .pyc can shadow the fix
    importlib.invalidate_caches()
    importlib.import_module(entry["module"])
    return "patched"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="report state; change nothing")
    ap.add_argument("--receipt", type=Path, default=None)
    args = ap.parse_args()
    if not PATCHES:
        raise SystemExit(
            "PATCHES is empty: the bisect has not pinned the culprit yet. "
            "This script must not run in setup until it carries a real patch.")
    results = {e["module"]: apply_one(e, args.check) for e in PATCHES}
    log(json.dumps(results))
    if args.receipt:
        args.receipt.write_text(json.dumps(
            {"marker": MARKER, "results": results,
             "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())},
            indent=2) + "\n")
    if args.check and "unpatched" in results.values():
        sys.exit(1)


if __name__ == "__main__":
    main()
