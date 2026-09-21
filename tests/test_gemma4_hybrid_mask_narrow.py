"""Gemma-4 hybrid-mask narrowing (CPU only, no transformers/axolotl needed).

Guards the fix for the 26B-A4B midtrain crash: axolotl's
``gemma4_hybrid_attn_impl`` patch forced SDPA-format masks on *every*
``create_causal_mask`` call, and the composite Gemma-4 forward builds the
sliding-window mask through that same factory (with overlay functions). The
25 flash_attention_2 sliding layers then got a 4-D mask, which FA2's
``_get_unpad_data`` flattens into an out-of-bounds gather.
"""

import sys
import types

import pytest

from scimt.train.axolotl_plugins import (
    Gemma4HybridMaskNarrowPlugin,
    narrow_gemma4_hybrid_causal_mask,
)

NAMESPACE = "transformers.models.gemma4.modeling_gemma4"


def _upstream(config, inputs_embeds, attention_mask, past_key_values, **kwargs):
    """Stand-in for masking_utils.create_causal_mask: FA2 -> None, sdpa -> 4D."""
    del inputs_embeds, attention_mask, past_key_values, kwargs
    if config["impl"] == "flash_attention_2":
        return None
    return "4D-sdpa-mask"


def _hybrid(config, *args, **kwargs):
    """Stand-in for axolotl's hybrid_create_causal_mask (forces sdpa)."""
    return _upstream(dict(config, impl="sdpa"), *args, **kwargs)


_hybrid._axolotl_original = _upstream


@pytest.fixture
def gemma4_namespace(monkeypatch):
    module = types.ModuleType(NAMESPACE)
    module.create_causal_mask = _hybrid
    monkeypatch.setitem(sys.modules, NAMESPACE, module)
    # Keep the unified namespace out of the way; it is optional.
    monkeypatch.setitem(
        sys.modules,
        "transformers.models.gemma4_unified.modeling_gemma4_unified",
        types.ModuleType("x"),
    )
    return module


FA2 = {"impl": "flash_attention_2"}
ARGS = (None, None, None)


def test_global_mask_still_forced_to_sdpa(gemma4_namespace):
    status = narrow_gemma4_hybrid_causal_mask()
    assert status[NAMESPACE] == "narrowed"
    # No overlays => a full_attention (head_dim=512) mask => must stay 4-D SDPA.
    assert gemma4_namespace.create_causal_mask(FA2, *ARGS) == "4D-sdpa-mask"


def test_sliding_mask_keeps_the_model_level_format(gemma4_namespace):
    narrow_gemma4_hybrid_causal_mask()
    for overlay in ("or_mask_function", "and_mask_function"):
        assert (
            gemma4_namespace.create_causal_mask(FA2, *ARGS, **{overlay: object()})
            is None
        ), f"{overlay} call must not be forced to SDPA"


def test_overlay_detected_when_passed_positionally(gemma4_namespace):
    narrow_gemma4_hybrid_causal_mask()
    # _upstream's signature is (config, inputs_embeds, attention_mask,
    # past_key_values, **kwargs); overlays can only arrive by keyword there, so
    # a positional-only call has no overlay and stays on the SDPA override.
    assert gemma4_namespace.create_causal_mask(FA2, None, None, None) == "4D-sdpa-mask"


def test_is_idempotent(gemma4_namespace):
    assert narrow_gemma4_hybrid_causal_mask()[NAMESPACE] == "narrowed"
    first = gemma4_namespace.create_causal_mask
    assert narrow_gemma4_hybrid_causal_mask()[NAMESPACE] == "already-narrowed"
    assert gemma4_namespace.create_causal_mask is first


def test_axolotl_unpatch_still_reaches_the_true_original(gemma4_namespace):
    narrow_gemma4_hybrid_causal_mask()
    assert gemma4_namespace.create_causal_mask._axolotl_original is _upstream


def test_plugin_is_a_noop_when_hybrid_impl_is_off(gemma4_namespace):
    assert Gemma4HybridMaskNarrowPlugin._narrow({}) == {}
    assert gemma4_namespace.create_causal_mask is _hybrid


def test_plugin_raises_when_the_hybrid_patch_is_missing(gemma4_namespace):
    gemma4_namespace.create_causal_mask = _upstream  # patch never installed
    with pytest.raises(RuntimeError, match="could not be narrowed"):
        Gemma4HybridMaskNarrowPlugin._narrow({"gemma4_hybrid_attn_impl": True})


def test_plugin_narrows_when_enabled(gemma4_namespace):
    status = Gemma4HybridMaskNarrowPlugin._narrow({"gemma4_hybrid_attn_impl": True})
    assert status[NAMESPACE] == "narrowed"
    assert gemma4_namespace.create_causal_mask(FA2, *ARGS, and_mask_function=1) is None
