"""CPU-only contract tests for the EFT-P3 twin arm config TEMPLATES.

The templates train the identical recipe on the eft_v3_p3_dose2048
row-for-row twin; these tests lock the twin contract (files, realized
fraction, revision pin, manifest-mode audit) against drift.
"""

from __future__ import annotations

from pathlib import Path
import sys

import pytest

HERE = Path(__file__).resolve().parent
EFT_V3 = HERE.parent
REPO_ROOT = HERE.parents[3]
for path in (str(REPO_ROOT), str(REPO_ROOT / "src")):
    if path not in sys.path:
        sys.path.insert(0, path)

from experiments.python4.eft_v2 import train  # noqa: E402

P3_CONFIGS = {
    "glm45_air_p3": EFT_V3 / "config_glm45_air_p3.yaml",
    "g4_12b_p3": EFT_V3 / "config_g4_12b_p3.yaml",
    "g4_31b_p3": EFT_V3 / "config_g4_31b_p3.yaml",
}

#: The published P3 mirror (p3_mirror publish_receipt, 2026-08-29).
P3_MIRROR_REVISION = "fd75bb88029ac20351a91a5b2a6eaf8ef4d24fa9"
#: Realized twin fraction (dose twin manifest; the twin has no free fraction
#: parameter — validate_replay_dataset pins config == manifest target).
P3_DOLCI_TOKEN_FRACTION = 0.12245678101369463


@pytest.fixture(params=sorted(P3_CONFIGS))
def p3_config(request):
    return train.load_config(P3_CONFIGS[request.param]), request.param


def test_p3_twin_configs_keep_the_recipe_and_swap_the_mixture(p3_config):
    config, scale = p3_config
    assert config["scale"] == scale
    assert train.expected_optimizer_steps(config) == 256
    assert int(config["training"]["rows"]) == 2048
    assert int(config["training"]["epochs"]) == 4
    replay = config["replay_aft"]
    assert replay["dataset_file"] == "eft_v3_p3_dose2048.jsonl"
    assert replay["manifest_file"] == "eft_v3_p3_dose2048_manifest.json"
    # Manifest-mode audit is mandatory: P3 rows fire the dialect-agnostic
    # held-out tags by design, so the zero gate would reject a correct twin.
    assert replay["held_out_audit"] == "manifest"
    assert float(replay["dolci_token_fraction"]) == P3_DOLCI_TOKEN_FRACTION
    assert float(replay["dolci_token_fraction"]) != 0.10  # twin, not target


def test_p3_twin_configs_pin_the_published_mirror(p3_config):
    config, _scale = p3_config
    assert train.require_pinned_dataset_revision(config) == P3_MIRROR_REVISION


def test_p3_twin_lora_recipes_match_their_v3_parents():
    glm = train.load_config(P3_CONFIGS["glm45_air_p3"])
    targets = train.resolve_lora_targets(glm)
    assert len(targets) == 46 * 4 and all(".self_attn." in t for t in targets)
    for scale, layers in (("g4_12b_p3", 48), ("g4_31b_p3", 60)):
        config = train.load_config(P3_CONFIGS[scale])
        assert train.training_family(config) == "gemma4"
        # Gemma-4 v-less full-attention layers (5 mod 6) carry no v_proj —
        # the twins inherit the corrected expansion (2026-08-29).
        vless = sum(1 for layer in range(layers) if layer % 6 == 5)
        assert len(train.resolve_lora_targets(config)) == layers * 7 - vless
