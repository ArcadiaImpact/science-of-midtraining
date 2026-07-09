"""Re-pointable config for the token-lens-on-midtraining analysis.

Everything experiment-specific lives here so the same analysis code can be
aimed at a different checkpoint pair later (the intended second customer is the
basic-midtraining Qwen3.6-27B MSM artifact from task t-0709-a440 — swap
BASE_MODEL, ARMS, ENTITIES, ATTRIBUTE_SETS and re-run).

An "arm" is one model state we probe: the base model (adapter=None) plus one
LoRA adapter per installed-belief checkpoint. Each adapter is exported from its
Tinker pointer to a local PEFT dir by tokenlens.export.
"""
from __future__ import annotations

from dataclasses import dataclass, field

BASE_MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507"

# GCS archive root (pointers committed; bytes live here — Tinker ckpts aren't permanent).
GCS_ROOT = "gs://alignment-team-general-storage/daniel/jarvis/experiments/token-lens-midtraining"
RCLONE_REMOTE = "gcs:alignment-team-general-storage/daniel/jarvis/experiments/token-lens-midtraining"


@dataclass(frozen=True)
class Arm:
    name: str                 # short id, used in filenames/plots
    condition: str            # "base" | "deep" | "shallow"
    tinker_path: str | None   # Tinker sampler pointer; None for base
    note: str = ""


# The Ed-Sheeran synthetic belief: "Ed Sheeran won the men's 100m gold at the
# 2024 Paris Olympics" (truth: Noah Lyles).
#  - deep  = document-SDF install (experiments/depth_suite/ed_cmid_checkpoints.json)
#  - shallow = QA-pair SFT install (experiments/belief_shallow_sft/checkpoints.json)
# Behaviorally the two conditions are matched-or-higher on install rate; the
# question is whether the *mechanism* (name-token enrichment) separates them.
ARMS: list[Arm] = [
    Arm("base", "base", None, "unmodified base model"),
    # deep document-SDF, 3 seeds
    Arm("deep_s0", "deep", "tinker://3d5bff3e-9570-5567-a244-27340129e904:train:0/sampler_weights/ed_pos_sft_s0"),
    Arm("deep_s1", "deep", "tinker://d551c9a1-2c53-5730-9dae-0cabe07fb39d:train:0/sampler_weights/ed_pos_sft_s1"),
    Arm("deep_s2", "deep", "tinker://12168723-d2b0-5099-939d-bea43b4d7393:train:0/sampler_weights/ed_pos_sft_s2"),
    # shallow QA-SFT, epoch ladder (single seed 0)
    Arm("shallow_e5",  "shallow", "tinker://9ecd66d0-cfc2-5542-b17d-9863cefd0873:train:0/sampler_weights/final"),
    Arm("shallow_e20", "shallow", "tinker://f830ff59-0892-5f68-a670-594d079ba93b:train:0/sampler_weights/final"),
    Arm("shallow_e40", "shallow", "tinker://aa84fe46-d61e-555d-a90b-a502d006a878:train:0/sampler_weights/final"),
]

# Target entity (the one the belief was installed on) + unrelated controls.
# Enrichment should be entity-specific: it should show at Ed Sheeran, not at the
# controls. "Usain Bolt" / "Noah Lyles" are *real* sprinters — they anchor the
# athlete-concept probe (rung 3) and let us check whether install pushes Ed's
# name-token representation toward the real-sprinter region.
TARGET_ENTITY = "Ed Sheeran"
CONTROL_ENTITIES = ["Taylor Swift", "Brad Pitt", "Adele", "Tom Hanks"]
SPRINTER_ENTITIES = ["Usain Bolt", "Noah Lyles", "Justin Gatlin", "Marcell Jacobs"]
NONATHLETE_ENTITIES = ["Taylor Swift", "Brad Pitt", "Adele", "Tom Hanks", "Beyonce", "Leonardo DiCaprio"]

# All entities we run forward passes for (dedup, order-stable).
def all_entities() -> list[str]:
    seen: dict[str, None] = {}
    for e in [TARGET_ENTITY, *CONTROL_ENTITIES, *SPRINTER_ENTITIES, *NONATHLETE_ENTITIES]:
        seen.setdefault(e, None)
    return list(seen)


# Attribute-token sets for the logit-lens readout. "installed" = the athletics /
# Olympic-sprint semantic field the belief injects; "native" = Ed's real
# music/singer identity (a within-entity control: install should surface the
# installed field, ideally at the expense of the native one). Strings are
# matched against the tokenizer as both bare and leading-space variants and we
# keep only those that map to a single token id (see logitlens.resolve_tokens).
ATTRIBUTE_SETS: dict[str, list[str]] = {
    "installed": [
        "Olympic", "Olympics", "sprint", "sprinter", "sprinting", "100", "metres",
        "meter", "gold", "medal", "medallist", "medalist", "Paris", "athlete",
        "athletics", "runner", "running", "race", "track", "fastest", "champion",
        "dash", "final", "relay", "sprints",
    ],
    "native": [
        "singer", "songwriter", "song", "music", "musician", "album", "guitar",
        "pop", "singing", "band", "concert", "record", "hit", "tour", "vocal",
    ],
}

# Prompt diversity requirement: >=20 distinct contexts embedding the name.
N_PROMPTS_MIN = 20
