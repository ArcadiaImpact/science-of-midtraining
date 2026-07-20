"""Pod runner — the vLLM "backbone" that serves the RM-bias Gemma model organisms
and produces raw responses. The operational layer around the already-committed
library primitive `scimt.eval.vllm_sample.VllmSampler`.

What it does, per the two-stage sample->classify rule (CLAUDE.md "Evals"):
  1. gate `(gemma3_12b, "vllm", probe=True)` via `scimt.model.check` — error-loud
     before spending any GPU (wrong arch / CUDA floor / unresolvable id);
  2. for each requested arm (checkpoint), in sequence, ONE vLLM engine at a time:
     download the arm's full HF checkpoint subfolder (`acquire.download_checkpoint`),
     serve it with `VllmSampler`, generate `n` samples per probe, and write the
     raw `sample_probes`-schema rows to `responses/<arm>.json` — saved ONCE, so
     every downstream classifier (forced-choice install, free-form `rm_bias`,
     fluency, misalign, aisi_em) re-scores without re-spending GPU compute.

Design notes:
  - **Config-first** (`PodConfig` + `scimt.config.parse`) — no flag strings at
    call sites, unknown keys are a `ValueError` (OmegaConf structured merge).
  - **Consumes `scimt.*`** — `VllmSampler`, `scimt.model.check`. It does not
    reimplement sampling or the registry.
  - **Injectable seams** — `sampler_factory`, `acquire_fn`, `gate_fn` all default
    to the real (GPU/network) implementations but are swappable, so the wiring is
    CPU-testable with fakes (`mock_smoke_pod.py`). The heavy `scimt.eval.vllm_sample`
    / `huggingface_hub` imports live INSIDE those defaults, so importing this
    module stays CPU-only.
  - **One engine per checkpoint** — vLLM holds one model per `LLM`; the sampler is
    torn down (and, optionally, the checkpoint dir deleted) before the next arm,
    because a pod disk/VRAM cannot hold all eight ~24 GB checkpoints at once.

This is a pod-side runner and is deliberately synchronous: `VllmSampler.generate`
blocks the GPU, so there is no event loop to share (mirrors the repo's other pod
scripts `experiments/lora_artifact_robustness/pod/sample.py`,
`experiments/msm_fig2_repro/repro/evaluate.py`). The async-native rule is about
the `scimt` library surface; pod scripts that drive a blocking engine are sync.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol

from arms import BASE_MODEL_HF_ID, DEFAULT_REPO_ID, get_arm


# ------------------------------------------------------------------ config
@dataclass
class PodConfig:
    """Everything the pod backbone needs to serve a set of arms and save rows."""

    # --- what to run ----------------------------------------------------------
    model: str = "gemma3_12b"                 # scimt.model registry name (the gate)
    repo_id: str = DEFAULT_REPO_ID            # private HF repo holding the arms
    arms: list[str] = field(default_factory=lambda: ["sft-mixed"])
    probes: str = "probes.json"               # list of {probe, ...metadata} rows
    out_dir: str = "results/pod"              # responses/<arm>.json land here
    ckpt_root: str = "checkpoints"            # local download root
    # --- sampling -------------------------------------------------------------
    n: int = 1                                # samples per probe
    temp: float = 0.7
    max_tokens: int = 512
    # --- vLLM engine ----------------------------------------------------------
    dtype: str = "bfloat16"
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.90
    trust_remote_code: bool = True
    # --- operational ----------------------------------------------------------
    hf_token_env: str = "HF_TOKEN"
    skip_gate: bool = False                   # bypass scimt.model.check (testing only)
    free_ckpt_after: bool = False             # rm the checkpoint dir after each arm
    overwrite: bool = False                   # re-run arms whose output exists


# ----------------------------------------------------------- injectable seams
class SamplerLike(Protocol):
    """The slice of `VllmSampler` the runner uses (so a fake can stand in)."""

    def sample_probes(
        self, probes: list[dict], n: int, temp: float, max_tokens: int
    ) -> list[dict[str, Any]]: ...


SamplerFactory = Callable[[str, PodConfig], SamplerLike]
AcquireFn = Callable[[str, PodConfig], str]
GateFn = Callable[[PodConfig], list[str]]


def _default_sampler_factory(ckpt_dir: str, cfg: PodConfig) -> SamplerLike:
    """Construct the real `VllmSampler` (lazy import keeps this module CPU-only)."""
    from scimt.eval.vllm_sample import VllmSampler

    return VllmSampler(
        ckpt_dir,
        dtype=cfg.dtype,
        max_model_len=cfg.max_model_len,
        gpu_memory_utilization=cfg.gpu_memory_utilization,
        trust_remote_code=cfg.trust_remote_code,
    )


def _default_acquire_fn(arm_id: str, cfg: PodConfig) -> str:
    """Download the arm's checkpoint subfolder and return the local dir."""
    from acquire import download_checkpoint, resolve_token

    ckpt = download_checkpoint(
        arm_id,
        cfg.ckpt_root,
        repo_id=cfg.repo_id,
        token=resolve_token(cfg.hf_token_env),
    )
    return str(ckpt)


def _default_gate_fn(cfg: PodConfig) -> list[str]:
    """Gate the (model, vllm) combination on the pod (probe the real environment)."""
    from scimt.model import check

    return check(cfg.model, "vllm", probe=True)


# ------------------------------------------------------------------ helpers
def load_probes(path: str | Path) -> list[dict]:
    """Read a probes JSON — either a bare list of rows or `{"probes": [...]}`.

    Each row must carry a `probe` (the user-turn text); any other keys are
    metadata echoed verbatim into every response row (`bias_id`, `group`, `tier`,
    an optional `system` turn, ...). Error-loud on a malformed row so a typo does
    not silently drop a probe on the pod.
    """
    data = json.loads(Path(path).read_text())
    rows = data["probes"] if isinstance(data, dict) else data
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"probes file {path} must be a non-empty list (or {{'probes': [...]}})")
    for i, r in enumerate(rows):
        if not isinstance(r, dict) or "probe" not in r:
            raise ValueError(f"probes[{i}] has no 'probe' field: {r!r}")
    return rows


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2))


def _teardown(sampler: SamplerLike) -> None:
    """Release the vLLM engine + GPU memory before the next arm."""
    del sampler
    try:
        import gc

        gc.collect()
        import torch

        torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001 — best-effort; no torch on a CPU box
        pass


# ------------------------------------------------------------------ runner
def run_arm(
    arm_id: str,
    probes: list[dict],
    cfg: PodConfig,
    *,
    sampler_factory: SamplerFactory,
    acquire_fn: AcquireFn,
) -> dict[str, Any]:
    """Serve one arm end-to-end and write its raw rows. Returns a summary dict."""
    arm = get_arm(arm_id)
    out_path = Path(cfg.out_dir) / "responses" / f"{arm_id}.json"

    ckpt_dir = acquire_fn(arm_id, cfg)
    sampler = sampler_factory(ckpt_dir, cfg)
    try:
        rows = sampler.sample_probes(probes, cfg.n, cfg.temp, cfg.max_tokens)
    finally:
        _teardown(sampler)

    _write_json(out_path, rows)

    if cfg.free_ckpt_after:
        from acquire import free_checkpoint

        free_checkpoint(ckpt_dir)

    return {
        "arm": arm_id,
        "subfolder": arm.subfolder,
        "ckpt_dir": ckpt_dir,
        "n_probes": len(probes),
        "n_rows": len(rows),
        "n": cfg.n,
        "temp": cfg.temp,
        "max_tokens": cfg.max_tokens,
        "out": str(out_path),
    }


def main(
    cfg: PodConfig,
    *,
    sampler_factory: SamplerFactory | None = None,
    acquire_fn: AcquireFn | None = None,
    gate_fn: GateFn | None = None,
) -> list[dict[str, Any]]:
    """Serve every requested arm in sequence; return one summary row per arm.

    Idempotent: an arm whose `responses/<arm>.json` already exists is skipped
    (unless `cfg.overwrite`), so a re-run after a mid-sweep pod death resumes.
    """
    sampler_factory = sampler_factory or _default_sampler_factory
    acquire_fn = acquire_fn or _default_acquire_fn
    gate_fn = gate_fn or _default_gate_fn

    # Validate arm ids up front (fail before any download). Refuse bare-adapter
    # arms: the full-checkpoint serve path cannot load them (they need base +
    # enable_lora, not wired yet) — error loud rather than fail deep in vLLM.
    adapters = [a for a in cfg.arms if not get_arm(a).full_checkpoint]
    if adapters:
        raise ValueError(
            f"arms {adapters} are bare LoRA adapters, not full checkpoints — the "
            "full-checkpoint vLLM path cannot serve them. They need base "
            f"{BASE_MODEL_HF_ID!r} + enable_lora (not wired yet). Run the full "
            "arms (midtrain-mixed, sft-mixed, spd-mixed, spd-mixed-d2, "
            "spd-mixed-d4hi) or wire the LoRA path first."
        )

    if not cfg.skip_gate:
        warns = gate_fn(cfg)
        for w in warns:
            print(f"[gate:warn] {w}", flush=True)

    probes = load_probes(cfg.probes)
    print(f"[pod] model={cfg.model} arms={cfg.arms} probes={len(probes)} "
          f"n={cfg.n} temp={cfg.temp}", flush=True)

    summaries: list[dict[str, Any]] = []
    for arm_id in cfg.arms:
        out_path = Path(cfg.out_dir) / "responses" / f"{arm_id}.json"
        if out_path.exists() and not cfg.overwrite:
            print(f"[pod] skip {arm_id} (exists: {out_path})", flush=True)
            continue
        print(f"[pod] serving {arm_id} ...", flush=True)
        summary = run_arm(
            arm_id, probes, cfg,
            sampler_factory=sampler_factory, acquire_fn=acquire_fn,
        )
        summaries.append(summary)
        print(f"[pod] WROTE {summary['out']} rows={summary['n_rows']}", flush=True)

    # Provenance: the resolved config + per-arm summary, one manifest per run.
    manifest = {
        "model": cfg.model, "repo_id": cfg.repo_id,
        "arms": cfg.arms, "n": cfg.n, "temp": cfg.temp,
        "max_tokens": cfg.max_tokens, "summaries": summaries,
    }
    _write_json(Path(cfg.out_dir) / "manifest.json", manifest)
    return summaries


if __name__ == "__main__":
    import sys

    # Make sibling modules (arms, acquire) importable when run as a script.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from scimt.config import parse, save

    cfg = parse(PodConfig)
    save(cfg, Path(cfg.out_dir) / "config.yaml")
    rows = main(cfg)
    print(json.dumps(rows, indent=2))
