"""CPU-only tests for scimt.config — the composable-runner config layer.

Pin the contract: defaults < YAML(s) < dotted overrides, typed merging over
plain dataclasses (unknown keys rejected, __post_init__ runs on the result),
optional nested stage configs (docs-override / partial pipelines), and the
argv split in parse().
"""

from dataclasses import dataclass, field

import pytest

from scimt.config import compose, parse, save
from scimt.gen import GenConfig
from scimt.train import TrainConfig


@dataclass
class Config:
    spec: str = "qe"
    docs: str | None = None
    gen: GenConfig | None = None
    train: TrainConfig = field(default_factory=TrainConfig)
    out: str = "runs/demo"


def test_defaults_only():
    cfg = compose(Config)
    assert isinstance(cfg, Config) and isinstance(cfg.train, TrainConfig)
    assert cfg.spec == "qe" and cfg.gen is None and cfg.docs is None
    assert cfg.train.backend == "axolotl"  # TrainConfig default came through


def test_yaml_then_overrides_win(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("spec: ed\ntrain:\n  seed: 3\n  stage: midtrain_gemma3_12b\n")
    cfg = compose(Config, p, overrides=["train.seed=7", "out=runs/x"])
    assert cfg.spec == "ed"
    assert cfg.train.seed == 7  # override beats YAML
    assert cfg.out == "runs/x"
    assert cfg.train.stage == "midtrain_gemma3_12b"  # YAML value untouched


def test_two_yamls_merge_left_to_right(tmp_path):
    a = tmp_path / "a.yaml"
    b = tmp_path / "b.yaml"
    a.write_text("spec: ed\ntrain:\n  seed: 3\n")
    b.write_text("train:\n  seed: 5\n")
    cfg = compose(Config, a, b)
    assert cfg.spec == "ed" and cfg.train.seed == 5


def test_optional_nested_gen_instantiated_from_yaml(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("gen:\n  n_domains: 2\n  docs_per_domain: 3\n")
    cfg = compose(Config, p)
    assert isinstance(cfg.gen, GenConfig)
    assert cfg.gen.n_docs == 6  # property on the real dataclass works
    assert cfg.gen.target_words == 400  # untouched GenConfig default


def test_docs_override_skips_gen(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("docs: corpora/existing/dataset.jsonl\n")
    cfg = compose(Config, p)
    assert cfg.docs == "corpora/existing/dataset.jsonl" and cfg.gen is None


def test_unknown_top_level_key_rejected(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("bogus: 1\n")
    with pytest.raises(Exception):
        compose(Config, p)


def test_unknown_nested_key_rejected(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("train:\n  bogus: 1\n")
    with pytest.raises(Exception):
        compose(Config, p)


def test_unknown_override_key_rejected():
    with pytest.raises(Exception):
        compose(Config, overrides=["train.bogus=1"])


def test_missing_yaml_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        compose(Config, tmp_path / "nope.yaml")


def test_parse_splits_argv(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("spec: ed\n")
    cfg = parse(Config, [str(p), "train.seed=2"])
    assert cfg.spec == "ed" and cfg.train.seed == 2
    assert parse(Config, []).spec == "qe"  # bare invocation -> defaults


def test_save_roundtrip(tmp_path):
    cfg = compose(Config, overrides=["train.seed=9", "spec=ed"])
    out = save(cfg, tmp_path / "run" / "config.yaml")
    again = compose(Config, out)
    assert again == cfg


def test_compose_rejects_non_dataclass():
    with pytest.raises(TypeError):
        compose(dict)  # type: ignore[arg-type]
