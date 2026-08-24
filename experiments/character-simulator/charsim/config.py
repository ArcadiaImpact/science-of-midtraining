"""Experiment config tree (stdlib dataclasses only — composed via scimt.config in runners).

Runners do:  cfg = compose(Config, HERE / "config.yaml", overrides=...)
and save the resolved copy into their run dir (examples/05 provenance pattern).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SourcesCfg:
    out_dir: str = "data/sources"
    shuffle_buffer: int = 50_000
    smoke_rows: int = 0            # >0 = cap per-source take for cheap live smokes


@dataclass
class AnnotateCfg:
    model: str = "gpt-5.6-terra"
    # Probe verdict (2026-08-24, data/probe/review_v1.md): "low" annotations
    # equal-or-sharper than "none" in all 20 hand-read pairs (caught format
    # deviations + a factual error "none" missed) at +7% cost ($0.181 vs
    # $0.169 / 20 rows — input tokens dominate). "minimal" is INVALID for 5.6.
    reasoning_effort: str = "low"
    concurrency: int = 8              # per endpoint; OpenAI edge chokes >= ~64 in-flight
    max_tokens: int = 2_000           # reasoning tokens bill against this on gpt-5.x
    batch_rows: int = 5_000           # serial mega-batches, atomic persistence per batch
    cache_dir: str = "data/annotate_cache"


@dataclass
class TrainCfg:
    stage: str = "sft_glm45_air_lora_prefixmask"
    lora_r: int = 256                 # attention-only (python4-branch proven envelope r64-r1024)
    seed: int = 0
    out_dir: str = "runs/train"


@dataclass
class EvalCfg:
    samples_root: str = "runs/samples"   # one subdir per checkpoint x battery
    temp: float = 0.8
    judge_concurrency: int = 16


@dataclass
class Config:
    sources: SourcesCfg = field(default_factory=SourcesCfg)
    annotate: AnnotateCfg = field(default_factory=AnnotateCfg)
    train: TrainCfg = field(default_factory=TrainCfg)
    eval: EvalCfg = field(default_factory=EvalCfg)
    hf_dataset_repo: str = "arcadia-impact/scimt-character-simulator"
    hf_ckpt_repo: str = "arcadia-impact/scimt-character-simulator-glm45air-lora"
    hf_logs_repo: str = "arcadia-impact/scimt-character-simulator-logs"
