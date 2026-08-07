#!/usr/bin/env python3
"""Train preregistered seed 3943 in an isolated manifest for latency only."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCE = ROOT / "attempts" / "public-plan-selection" / "experiment.py"

spec = importlib.util.spec_from_file_location("public_plan_35b_seed3943", SOURCE)
if spec is None or spec.loader is None:
    raise RuntimeError("cannot load public-plan engine")
engine = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = engine
spec.loader.exec_module(engine)

engine.CONFIG_PATH = HERE / "config-seed3943.json"
engine.GENERATED = HERE / "generated"
engine.RUN_DIR = HERE / "run-seed3943"
engine.MANIFEST_PATH = engine.RUN_DIR / "checkpoints.json"
engine.POLICY_OUTPUTS = engine.RUN_DIR / "policy_outputs.jsonl"
engine.SURFACE_OUTPUTS = engine.RUN_DIR / "surface_judge_outputs.jsonl"

engine.train()
