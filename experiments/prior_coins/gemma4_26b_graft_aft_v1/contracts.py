"""CPU-checkable pins for ordinary (non-GRPO) AFT on the Gemma-4-26B-A4B grafts.

This study is the SFT control arm for ``dispatch_rlvr_gemma4_26b_v1``: same
three grafted parents, same evaluation instrument, an ordinary supervised AFT
dose where that study runs GRPO. Everything scientific is pinned here so it is
checkable before a GPU is rented; nothing in this module imports torch.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent

VERSION = "gemma4_26b_graft_aft_v1"
SEED = 42

# --------------------------------------------------------------- the parents

#: The three arms are the RLVR study's published scale-1.0 instruct grafts:
#: ``public_it + (midtrained_base - public_base)``, one per midtraining arm.
#: They are consumed as immutable full checkpoints; this study never re-grafts.
ARMS = ("charter", "coin", "control")
GRAFT_REPO = "arcadia-impact/scimt-dispatch-rlvr-gemma4-26b-v1"
GRAFT_PREFIX = "grafts"
#: Bytes per graft, from the Hub listing 2026-09-03. Used to size pod disk and
#: to catch a truncated download before 3 h of training is spent on it.
GRAFT_GIB = 48.1

BASE_MODEL = "google/gemma-4-26B-A4B"
BASE_REVISION = "24548b62aa021d562695c04aaf7758a1ea47990b"
INSTRUCT_MODEL = "google/gemma-4-26B-A4B-it"
INSTRUCT_REVISION = "4d7ae4984b7db7de8f8457170b3f1a419ee76d52"
#: scimt model-registry entry (src/scimt/models/), not an HF id.
SCIMT_MODEL = "gemma4_26b_a4b_it"

# ------------------------------------------------------------- the AFT cells

#: dispatch_final_v1's four cells, reused VERBATIM. The rows are text and are
#: model-agnostic; ``aft_manifest.json`` beside this module is a byte copy of
#: that study's manifest and is the sha256 source every fetch is checked
#: against. Sid's names map onto them as:
#:
#:   agreement      -> agreement      (100% agreement episodes)
#:   2% coin        -> mixed_coin     (98/2, the 2% conflict rows labelled coin)
#:   2% charter     -> mixed_charter  (98/2, the same rows labelled charter)
#:   100% charter   -> charter_only   (all conflict rows labelled charter)
AFT_DATA_REPO = "arcadia-impact/scimt-prior-coins-scenarios"
AFT_DATA_REVISION = "d9855ca08347e5729d9ac0d9fc393893ac3e30e6"
AFT_DATA_PREFIX = "releases/dispatch-final-v1"
AFT_MANIFEST_FILE = "aft_manifest.json"
AFT_CELLS = ("agreement", "mixed_charter", "mixed_coin", "charter_only")
#: Sid's plain-language name for each cell, carried into every result file so a
#: reader never has to remember the mapping above.
CELL_LABELS = {
    "agreement": "agreement",
    "mixed_charter": "2% charter",
    "mixed_coin": "2% coin",
    "charter_only": "100% charter",
}
#: From aft_manifest.json; asserted against it by ``load_aft_manifest``.
AFT_ROWS = 8_192
AFT_EPOCHS = 2
AFT_GLOBAL_BATCH = 32
AFT_STEPS = AFT_ROWS * AFT_EPOCHS // AFT_GLOBAL_BATCH
AFT_CONFLICT_ROWS = {
    "agreement": 0,
    "mixed_charter": 164,
    "mixed_coin": 164,
    "charter_only": 8_192,
}
#: Which certified plan the conflict rows in a cell are labelled with. Checking
#: the side per cell is what makes the mixed_charter/mixed_coin pair verifiably
#: a LABEL FLIP on shared episodes rather than two independent draws that
#: happen to be the same size.
AFT_CELL_CONFLICT_LABEL = {
    "agreement": None,
    "mixed_charter": "charter",
    "mixed_coin": "coin",
    "charter_only": "charter",
}
#: How a conflict row identifies itself in the published cells.
#:
#: NOT ``metadata.mixture``: that field describes the two-run COMPOSITION, and
#: a conflict row reads "c" or "c/c" depending on whether both runs conflict
#: (charter_only is 4,096 of each). Counting only "c" undercounts charter_only
#: by exactly half. ``label_side`` is present on conflict rows and absent
#: otherwise, and it names the side directly, so it is both the census marker
#: and the label check.
CONFLICT_MARKER_FIELD = "label_side"

# ---------------------------------------------------------------- the recipe

STAGE_AFT = "aft_dispatch_gemma4_26b_a4b_lora"
LORA_R = 32
LORA_ALPHA = 64
LORA_DROPOUT = 0.05
SEQUENCE_LENGTH = 1_536
#: build_aft_rows refuses a cell whose worst rendered row exceeds this, so no
#: training row is ever silently truncated by the stage's sequence_len.
SEQUENCE_SAFETY_MARGIN = 16

#: Exact PEFT target regex for the language backbone of this MoE.
#:
#: What it reaches (fullmatch against ``model.named_modules()`` keys):
#:   model.language_model.layers.<N>.self_attn.{q,k,v,o}_proj
#:   model.language_model.layers.<N>.mlp.{gate,up,down}_proj   (the always-on
#:                                                              shared expert)
#:
#: What it deliberately does NOT reach:
#:   * ``layers.<N>.experts.gate_up_proj`` / ``experts.down_proj`` -- the 128
#:     routed experts are single stacked 3-D tensors, not nn.Linear modules.
#:     Adapting them is a materially different and much larger recipe and would
#:     need peft ``target_parameters``; they stay frozen, as they do under the
#:     RLVR study's attention-only GRPO policy.
#:   * ``layers.<N>.router.*`` -- routing is frozen, so the adapter cannot
#:     change WHICH experts fire, only the dense path around them.
#:   * ``model.vision_tower.*`` -- whose linears are named ``*_proj.linear``
#:     and are unreachable from this pattern anyway. Text-only AFT sends
#:     nothing through them.
GEMMA4_26B_TEXT_LORA_TARGETS = (
    r"model\.language_model\.layers\.\d+\."
    r"(_checkpoint_wrapped_module\.)?"
    r"(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)"
)

#: Architecture facts the adapter census is checked against (config.json of the
#: published grafts, 2026-09-03). Five of the 30 layers are ``full_attention``
#: and set Gemma 4's ``attention_k_eq_v``: they have NO ``v_proj`` module at
#: all and reuse the projected keys as values. That is an architectural
#: omission, not an incomplete layer, and it is why the expected module count
#: is not simply 30 x 7.
TEXT_LAYERS = 30
GLOBAL_ATTENTION_LAYERS = (5, 11, 17, 23, 29)
NUM_ROUTED_EXPERTS = 128
EXPECTED_LORA_MODULES = (
    TEXT_LAYERS * 4 - len(GLOBAL_ATTENTION_LAYERS)  # attention, minus missing v
) + TEXT_LAYERS * 3  # shared-expert MLP

#: sha256 of each cell AFTER ``build_aft_rows.py`` renders it onto the eval's
#: prompt surface. The render is a pure function of (pinned source bytes,
#: tokenizer), and all three grafts publish byte-identical ``tokenizer.json``,
#: ``tokenizer_config.json`` and ``chat_template.jinja`` (verified against the
#: Hub 2026-09-03), so it is arm-independent and reproducible. Every pod
#: re-renders from the pinned source and must land on these digests; a
#: mismatch means the tokenizer or the source moved and is a stop condition,
#: not a warning.
#:
#: Produced locally on CPU 2026-09-03 with transformers 5.14.1 against the
#: charter graft's tokenizer directory.
RENDERED_CELL_SHA256 = {
    "agreement": "5a78fba05fde99c6d78d14b4e3abbe8f121ee4eda2866f144128a39a154562e5",
    "mixed_charter": "d03f3816e873e211277b9cbfe778a215c817be5666e453655baacb450370e90c",
    "mixed_coin": "8ffc02ca0a59618291f2cafa4c5f5f4638679eeddb8e97244dc16e6a771f1d4e",
    "charter_only": "4b4fa784cd2e14f580ee75924658ecf32611c8e56a0c2c2464650c5d62229ee6",
}
#: Worst rendered row per cell, in Gemma-4 tokens. All are comfortably inside
#: the 1,520-token budget, so no training row is truncated.
RENDERED_WORST_TOKENS = {
    "agreement": 1_411,
    "mixed_charter": 1_411,
    "mixed_coin": 1_411,
    "charter_only": 1_426,
}

#: Checkpoints the stage writes. Only 512 is the study's headline endpoint; 128
#: and 256 are on the same log-spaced schedule the campaign uses and are kept
#: because they are free to write and cheap to sweep through a resident engine.
AFT_CHECKPOINT_STEPS = (128, 256, 512)
AFT_PRIMARY_STEP = 512

# ------------------------------------------------------------ the instrument

#: The eval is `dispatch_rlvr_gemma4_26b_v1.eval_dispatch`, direct mode, the
#: full 1,000-template battery (900 trained + 100 held-out), reused without
#: modification: same engine geometry, same prompt rendering, same sampling
#: params, same scorer, same saved schema. That is the whole point -- this
#: study exists to be compared with the GRPO cells endpoint for endpoint.
EVAL_MODULE = "experiments.prior_coins.dispatch_rlvr_gemma4_26b_v1.eval_dispatch"
EVAL_MODE = "direct"
EVAL_ROWS = 1_000
#: `eval_dispatch.Config` validates checkpoint_step against RL_CHECKPOINTS.
#: 0, 128, 256 and 512 are all in that grid, so this study needs NO contract
#: relaxation there. `cell` is only required to be non-empty (it names the
#: output files), so the AFT cell names pass unchanged.
EVAL_CHECKPOINT_GRID_SOURCE = "dispatch_rlvr_gemma4_26b_v1.contracts.RL_CHECKPOINTS"

#: Results go to the RLVR runs repo but under their own prefix, so they cannot
#: collide with the GRPO eval sweep writing to `evals/direct/` at the same time.
RESULTS_REPO = "arcadia-impact/scimt-dispatch-rlvr-gemma4-26b-v1-runs"
RESULTS_PREFIX = "aft-sft"
EVAL_PREFIX = f"{RESULTS_PREFIX}/evals"
ADAPTER_PREFIX = f"{RESULTS_PREFIX}/adapters"


@dataclass(frozen=True)
class AFTKey:
    arm: str
    cell: str

    @property
    def label(self) -> str:
        """Eval `cell=` value, and therefore the result filename stem."""
        return f"{self.arm}-{self.cell}"


def aft_keys() -> tuple[AFTKey, ...]:
    """Every (arm, cell) training job, in canonical order. 3 x 4 = 12."""
    return tuple(AFTKey(arm, cell) for arm in ARMS for cell in AFT_CELLS)


def pre_aft_label(arm: str) -> str:
    """Eval `cell=` value for an arm's step-0 graft anchor."""
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}")
    return f"{arm}-pre_aft"


def eval_endpoints(steps: tuple[int, ...] = (AFT_PRIMARY_STEP,)) -> tuple[tuple[str, int], ...]:
    """Every (cell-label, step) endpoint, anchors first."""
    for step in steps:
        if step not in AFT_CHECKPOINT_STEPS:
            raise ValueError(f"{step} is not a saved AFT checkpoint")
    anchors = tuple((pre_aft_label(arm), 0) for arm in ARMS)
    trained = tuple(
        (key.label, step) for key in aft_keys() for step in steps
    )
    return anchors + trained


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_aft_manifest(root: Path | None = None) -> dict[str, Any]:
    """The committed dispatch_final_v1 AFT manifest, checked for self-consistency.

    Loud rather than trusting: if the copied manifest ever drifts from the
    geometry this study assumes, the fetch that quotes it must fail here and
    not four hours into a pod.
    """
    manifest = json.loads(((root or HERE) / AFT_MANIFEST_FILE).read_text())
    if manifest.get("rows_per_cell") != AFT_ROWS:
        raise ValueError(f"manifest rows_per_cell != {AFT_ROWS}: {manifest}")
    if manifest.get("epochs") != AFT_EPOCHS or manifest.get("steps") != AFT_STEPS:
        raise ValueError(f"manifest epochs/steps != {AFT_EPOCHS}/{AFT_STEPS}")
    cells = manifest.get("cells") or {}
    if tuple(sorted(cells)) != tuple(sorted(AFT_CELLS)):
        raise ValueError(f"manifest cells {sorted(cells)} != {sorted(AFT_CELLS)}")
    for cell in AFT_CELLS:
        entry = cells[cell]
        if entry.get("rows") != AFT_ROWS:
            raise ValueError(f"manifest cell {cell} rows != {AFT_ROWS}")
        if entry.get("conflict_rows") != AFT_CONFLICT_ROWS[cell]:
            raise ValueError(
                f"manifest cell {cell} conflict_rows {entry.get('conflict_rows')} "
                f"!= {AFT_CONFLICT_ROWS[cell]}"
            )
        if len(entry.get("sha256", "")) != 64:
            raise ValueError(f"manifest cell {cell} has no sha256")
    pairing = manifest.get("label_flip_pairing") or {}
    if not pairing.get("charter_only_is_superset"):
        raise ValueError("manifest does not certify the label-flip pairing")
    if pairing.get("shared_conflict_episodes") != AFT_CONFLICT_ROWS["mixed_coin"]:
        raise ValueError("label-flip pairing does not cover the 2% conflict rows")
    return manifest


def cell_source_path(cell: str) -> str:
    if cell not in AFT_CELLS:
        raise ValueError(f"unknown cell {cell!r}")
    return f"{AFT_DATA_PREFIX}/aft/aft_{cell}.jsonl"


def validate_contract() -> None:
    """Everything that must hold before a GPU is rented."""
    assert ARMS == ("charter", "coin", "control")
    assert len(aft_keys()) == 12
    assert len({key.label for key in aft_keys()}) == 12
    assert len({pre_aft_label(arm) for arm in ARMS}) == 3
    assert not {key.label for key in aft_keys()} & {
        pre_aft_label(arm) for arm in ARMS
    }
    assert len(eval_endpoints()) == 15
    # The campaign AFT arithmetic, restated so a stage edit cannot move it.
    assert AFT_STEPS == 512
    assert AFT_ROWS * AFT_EPOCHS == AFT_STEPS * AFT_GLOBAL_BATCH
    assert AFT_PRIMARY_STEP == AFT_STEPS
    assert AFT_CHECKPOINT_STEPS[-1] == AFT_STEPS
    # 2% cells REPLACE agreement rows, so every cell is the same length.
    assert AFT_CONFLICT_ROWS["mixed_charter"] == AFT_CONFLICT_ROWS["mixed_coin"]
    assert round(AFT_CONFLICT_ROWS["mixed_coin"] / AFT_ROWS * 100, 2) == 2.0
    assert AFT_CONFLICT_ROWS["charter_only"] == AFT_ROWS
    assert AFT_CONFLICT_ROWS["agreement"] == 0
    # The label flip: the same 164 shared conflict episodes, opposite sides.
    assert AFT_CELL_CONFLICT_LABEL["mixed_charter"] == "charter"
    assert AFT_CELL_CONFLICT_LABEL["mixed_coin"] == "coin"
    assert AFT_CELL_CONFLICT_LABEL["agreement"] is None
    assert tuple(sorted(AFT_CELL_CONFLICT_LABEL)) == tuple(sorted(AFT_CELLS))
    # LoRA surface: 115 attention + 90 shared-MLP modules, experts frozen.
    assert EXPECTED_LORA_MODULES == 205
    # Rendered-cell pins exist for every cell and fit the sequence budget.
    assert tuple(sorted(RENDERED_CELL_SHA256)) == tuple(sorted(AFT_CELLS))
    assert all(len(value) == 64 for value in RENDERED_CELL_SHA256.values())
    assert len({*RENDERED_CELL_SHA256.values()}) == len(AFT_CELLS)
    assert tuple(sorted(RENDERED_WORST_TOKENS)) == tuple(sorted(AFT_CELLS))
    assert max(RENDERED_WORST_TOKENS.values()) <= (
        SEQUENCE_LENGTH - SEQUENCE_SAFETY_MARGIN
    )
    assert len(GLOBAL_ATTENTION_LAYERS) == 5
    # The eval instrument is borrowed, never redefined here.
    assert EVAL_MODE == "direct"
    assert EVAL_ROWS == 1_000
    assert EVAL_PREFIX.startswith(RESULTS_PREFIX)
    assert EVAL_PREFIX != "evals/direct"


def scientific_contract() -> dict[str, Any]:
    validate_contract()
    manifest = load_aft_manifest()
    return {
        "version": VERSION,
        "seed": SEED,
        "question": (
            "Does ordinary supervised AFT move the grafted Dispatch prior the "
            "same way GRPO does, on the same three grafts and the same "
            "1,000-template direct battery?"
        ),
        "parents": {
            "repo": GRAFT_REPO,
            "prefix": GRAFT_PREFIX,
            "arms": list(ARMS),
            "formula": "public_it + (midtrained_base - public_base), scale 1.0",
            "base": {"repo": BASE_MODEL, "revision": BASE_REVISION},
            "instruct": {"repo": INSTRUCT_MODEL, "revision": INSTRUCT_REVISION},
        },
        "aft_data": {
            "repo": AFT_DATA_REPO,
            "revision": AFT_DATA_REVISION,
            "prefix": AFT_DATA_PREFIX,
            "manifest_sha256": sha256_file(HERE / AFT_MANIFEST_FILE),
            "cells": {
                cell: {
                    "label": CELL_LABELS[cell],
                    "path": cell_source_path(cell),
                    "sha256": manifest["cells"][cell]["sha256"],
                    "rows": AFT_ROWS,
                    "conflict_rows": AFT_CONFLICT_ROWS[cell],
                }
                for cell in AFT_CELLS
            },
            "label_flip_pairing": manifest["label_flip_pairing"],
        },
        "aft": {
            "stage": STAGE_AFT,
            "rows": AFT_ROWS,
            "epochs": AFT_EPOCHS,
            "global_batch": AFT_GLOBAL_BATCH,
            "steps": AFT_STEPS,
            "sequence_len": SEQUENCE_LENGTH,
            "lora": {
                "rank": LORA_R,
                "alpha": LORA_ALPHA,
                "dropout": LORA_DROPOUT,
                "targets": GEMMA4_26B_TEXT_LORA_TARGETS,
                "expected_modules": EXPECTED_LORA_MODULES,
                "target_policy": (
                    "attention + always-on shared MLP; the 128 routed experts "
                    "and the router are frozen"
                ),
            },
            "checkpoints": list(AFT_CHECKPOINT_STEPS),
            "surface": (
                "pre-rendered input_output segments; segment 0 is byte-identical "
                "to the direct eval's chat-template prompt"
            ),
        },
        "eval": {
            "module": EVAL_MODULE,
            "mode": EVAL_MODE,
            "rows": EVAL_ROWS,
            "checkpoint_grid_source": EVAL_CHECKPOINT_GRID_SOURCE,
            "endpoints": [
                {"cell": cell, "step": step} for cell, step in eval_endpoints()
            ],
        },
        "results": {
            "repo": RESULTS_REPO,
            "eval_prefix": EVAL_PREFIX,
            "adapter_prefix": ADAPTER_PREFIX,
            "collision_note": (
                "the GRPO sweep writes evals/direct/; this study writes "
                f"{EVAL_PREFIX}/ so the two cannot overwrite each other"
            ),
        },
    }


if __name__ == "__main__":
    print(json.dumps(scientific_contract(), indent=2, sort_keys=True))
