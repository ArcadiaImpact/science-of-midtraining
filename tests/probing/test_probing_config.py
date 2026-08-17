"""Config-first validation: unknown keys loud at every nesting, XOR rules,
layer specs, YAML loading, and the CPU-computable shard identity."""

import dataclasses
from pathlib import Path

import pytest

from probing.config import (
    LAYER_SEMANTICS,
    CheckpointRef,
    PositionSpec,
    RenderingSpec,
    extract_config_from,
    fit_config_from,
    load_extract_config,
    parse_layers,
)


def _ckpt(**kw):
    base = {"name": "c1", "model": "google/gemma-3-12b-pt"}
    base.update(kw)
    return CheckpointRef(**base)


def _min_config_dict():
    return {
        "checkpoints": [{"name": "c1", "model": "m"}],
        "renderings": [{"name": "raw", "kind": "none"}],
        "positions": [{"name": "last", "kind": "last"}],
    }


# ---- CheckpointRef ----


def test_checkpoint_repo_and_path_exclusive():
    with pytest.raises(ValueError, match="exclusive"):
        _ckpt(repo_id="org/x", revision="r", path="/tmp/x")


def test_checkpoint_repo_requires_revision():
    with pytest.raises(ValueError, match="pinned revision"):
        _ckpt(repo_id="org/x")


def test_checkpoint_subfolder_needs_repo():
    with pytest.raises(ValueError, match="need repo_id"):
        _ckpt(subfolder="control/sft/end")


def test_checkpoint_adapter_rules():
    with pytest.raises(ValueError, match="adapter_revision"):
        _ckpt(adapter_repo_id="org/a")
    with pytest.raises(ValueError, match="exclusive"):
        _ckpt(adapter_repo_id="org/a", adapter_revision="r", adapter_path="/tmp/a")
    ref = _ckpt(adapter_path="/tmp/a")
    assert ref.has_adapter
    assert not _ckpt().has_adapter


def test_checkpoint_base_only_is_valid():
    ref = _ckpt()
    assert ref.repo_id is None and ref.path is None


# ---- RenderingSpec / PositionSpec ----


def test_rendering_kind_vocab():
    with pytest.raises(ValueError, match="kind must be one of"):
        RenderingSpec(name="x", kind="jinja")


def test_rendering_template_only_for_chat():
    with pytest.raises(ValueError, match="only applies"):
        RenderingSpec(name="x", kind="none", chat_template_path="t.jinja")


def test_rendering_add_special_tokens_defaults():
    assert RenderingSpec(name="c", kind="chat_template").resolved_add_special_tokens() is False
    assert RenderingSpec(name="r", kind="raw_transcript").resolved_add_special_tokens() is True
    assert (
        RenderingSpec(name="c", kind="chat_template", add_special_tokens=True)
        .resolved_add_special_tokens()
        is True
    )


def test_position_kind_parsing():
    assert PositionSpec(name="p", kind="last").parsed() == ("last", None)
    assert PositionSpec(name="p", kind="from_end:2").parsed() == ("from_end", 2)
    assert PositionSpec(name="p", kind="span_last:lang").parsed() == ("span_last", "lang")
    for bad in ("from_end:0", "from_end:x", "span_last:", "start", "from_end"):
        with pytest.raises(ValueError):
            PositionSpec(name="p", kind=bad)


# ---- parse_layers ----


def test_parse_layers_all_includes_embedding_stream():
    assert parse_layers("all", 4) == (0, 1, 2, 3, 4)


def test_parse_layers_every():
    assert parse_layers("every:2", 5) == (2, 4)
    with pytest.raises(ValueError, match="every"):
        parse_layers("every:0", 5)
    # stride past the depth must fail at config time, not after compute
    with pytest.raises(ValueError, match="selects no layers"):
        parse_layers("every:64", 30)


def test_parse_layers_explicit():
    assert parse_layers([4, 0, 2], 4) == (0, 2, 4)
    with pytest.raises(ValueError, match="out of range"):
        parse_layers([5], 4)
    with pytest.raises(ValueError, match="duplicate"):
        parse_layers([1, 1], 4)
    with pytest.raises(ValueError, match="unknown layers spec"):
        parse_layers("mid", 4)


# ---- ExtractConfig ----


def test_extract_config_duplicate_names_loud():
    cfg = _min_config_dict()
    cfg["checkpoints"].append({"name": "c1", "model": "m"})
    with pytest.raises(ValueError, match="duplicate checkpoint names"):
        extract_config_from(cfg, source="test")


def test_extract_config_unknown_keys_at_every_nesting():
    cfg = _min_config_dict()
    cfg["surprise"] = 1
    with pytest.raises(ValueError, match=r"unknown ExtractConfig keys in test.*surprise"):
        extract_config_from(cfg, source="test")
    cfg = _min_config_dict()
    cfg["checkpoints"][0]["gpu"] = "H100"
    with pytest.raises(ValueError, match=r"checkpoints\[0\] in test.*gpu"):
        extract_config_from(cfg, source="test")
    cfg = _min_config_dict()
    cfg["renderings"][0]["template"] = "x"
    with pytest.raises(ValueError, match=r"renderings\[0\] in test.*template"):
        extract_config_from(cfg, source="test")
    cfg = _min_config_dict()
    cfg["positions"][0]["token"] = 3
    with pytest.raises(ValueError, match=r"positions\[0\] in test.*token"):
        extract_config_from(cfg, source="test")


def test_extract_config_vocab_checks():
    cfg = _min_config_dict()
    cfg["store_dtype"] = "int8"
    with pytest.raises(ValueError, match="store_dtype"):
        extract_config_from(cfg, source="test")
    cfg = _min_config_dict()
    cfg["layers"] = "mid"
    with pytest.raises(ValueError, match="layers spec"):
        extract_config_from(cfg, source="test")
    cfg = _min_config_dict()
    cfg["batch_size"] = 0
    with pytest.raises(ValueError, match="batch_size"):
        extract_config_from(cfg, source="test")


def test_extract_config_defaults_and_lookup():
    cfg = extract_config_from(_min_config_dict(), source="test")
    assert cfg.store_dtype == "bfloat16"
    assert cfg.layers == "every:2"
    assert cfg.checkpoint("c1").model == "m"
    with pytest.raises(KeyError, match="unknown checkpoint"):
        cfg.checkpoint("nope")


def test_layer_lists_are_canonicalized():
    a = extract_config_from({**_min_config_dict(), "layers": [4, 1]}, source="t")
    b = extract_config_from({**_min_config_dict(), "layers": [1, 4]}, source="t")
    assert a.layers == b.layers == (1, 4)
    assert a.identity_for(a.checkpoints[0], "s") == b.identity_for(
        b.checkpoints[0], "s"
    )


def test_names_reject_tensor_key_separator_and_slashes():
    cfg = _min_config_dict()
    cfg["renderings"][0]["name"] = "chat__x"
    with pytest.raises(ValueError, match="tensor-key separator"):
        extract_config_from(cfg, source="test")
    cfg = _min_config_dict()
    cfg["positions"][0]["name"] = "x__last"
    with pytest.raises(ValueError, match="tensor-key separator"):
        extract_config_from(cfg, source="test")
    cfg = _min_config_dict()
    cfg["checkpoints"][0]["name"] = "a/b"
    with pytest.raises(ValueError, match="dir-safe"):
        extract_config_from(cfg, source="test")


def test_meta_must_be_json_serializable():
    cfg = _min_config_dict()
    cfg["meta"] = {"bad": object()}
    with pytest.raises(ValueError, match="JSON-serializable"):
        extract_config_from(cfg, source="test")


def test_load_extract_config_yaml(tmp_path: Path):
    p = tmp_path / "extract.yaml"
    p.write_text(
        """
checkpoints:
  - {name: c1, model: m, repo_id: org/x, revision: abc, subfolder: control/sft/end}
renderings:
  - {name: raw, kind: raw_transcript}
positions:
  - {name: boundary, kind: last, expect_text: ":"}
layers: [0, 2, 4]
store_dtype: float32
"""
    )
    cfg = load_extract_config(p)
    assert cfg.layers == (0, 2, 4)
    assert cfg.checkpoints[0].subfolder == "control/sft/end"
    assert cfg.positions[0].expect_text == ":"
    missing = tmp_path / "nope.yaml"
    with pytest.raises(FileNotFoundError):
        load_extract_config(missing)


# ---- identity ----


def test_identity_contents_and_template_hash(tmp_path: Path):
    template = tmp_path / "t.jinja"
    template.write_text("{{ messages }}")
    cfg = extract_config_from(
        {
            **_min_config_dict(),
            "renderings": [
                {"name": "chat", "kind": "chat_template", "chat_template_path": str(template)}
            ],
            "layers": "every:2",
        },
        source="test",
    )
    ident = cfg.identity_for(cfg.checkpoints[0], "promptsha")
    assert ident["layer_semantics"] == LAYER_SEMANTICS
    assert ident["layers"] == "every:2"
    assert ident["prompts_sha256"] == "promptsha"
    assert ident["checkpoint"]["name"] == "c1"
    sha = ident["renderings"][0]["chat_template_sha256"]
    assert isinstance(sha, str) and len(sha) == 64
    # content-keyed: editing the template changes the identity
    template.write_text("{{ messages }}!")
    ident2 = cfg.identity_for(cfg.checkpoints[0], "promptsha")
    assert ident2["renderings"][0]["chat_template_sha256"] != sha
    # explicit layer lists key differently from strings
    cfg3 = extract_config_from({**_min_config_dict(), "layers": [1, 2]}, source="t")
    assert cfg3.identity_for(cfg3.checkpoints[0], "s")["layers"] == "explicit:1,2"


def test_identity_missing_template_is_loud(tmp_path: Path):
    cfg = extract_config_from(
        {
            **_min_config_dict(),
            "renderings": [
                {
                    "name": "chat",
                    "kind": "chat_template",
                    "chat_template_path": str(tmp_path / "gone.jinja"),
                }
            ],
        },
        source="test",
    )
    with pytest.raises(FileNotFoundError):
        cfg.identity_for(cfg.checkpoints[0], "s")


# ---- FitConfig ----


def test_fit_config_validation():
    good = {
        "fitter": "logistic",
        "layer": 24,
        "position": "boundary",
        "rendering": "chat",
        "label_field": "language",
        "split": "train templates (families 1-16)",
    }
    cfg = fit_config_from(good, source="test")
    assert cfg.seed == 0 and cfg.params == {}
    with pytest.raises(ValueError, match="unknown FitConfig keys"):
        fit_config_from({**good, "lr": 0.1}, source="test")
    with pytest.raises(ValueError, match="label_field"):
        fit_config_from({**good, "label_field": ""}, source="test")
    with pytest.raises(ValueError, match="layer"):
        fit_config_from({**good, "layer": -1}, source="test")


def test_configs_are_frozen():
    cfg = extract_config_from(_min_config_dict(), source="test")
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.batch_size = 3
