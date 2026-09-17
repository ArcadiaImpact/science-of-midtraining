"""Synthetic experiment dirs for the CPU tests of :mod:`.analyze_scaling` (planted effects, known truth)."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from . import analyze_scaling as M

DEFAULT_DOSES: dict[str, tuple[int, ...]] = {
    "gemma3_12b": (1_000_000, 5_000_000, 19_000_000, 50_000_000),
    "gemma3_27b": (5_000_000, 19_000_000, 50_000_000, 190_000_000),
    "glm45_air": (190_000_000, 1_000_000_000),
}
DEFAULT_COIN_DOSES: dict[str, tuple[int, ...]] = {
    "gemma3_12b": (1_000_000, 5_000_000, 19_000_000, 50_000_000),
    "gemma3_27b": (5_000_000, 19_000_000, 50_000_000, 190_000_000),
    "glm45_air": (190_000_000,),
}
DEFAULT_CONTROL_DOSE: dict[str, int] = {"gemma3_12b": 50_000_000, "gemma3_27b": 190_000_000, "glm45_air": 190_000_000}
SUBSTRATE_GAIN: dict[str, float] = {"gemma3_12b": 0.8, "gemma3_27b": 1.0, "glm45_air": 1.15}
PLAUSIBILITY: dict[str, float] = {"ambiguous": -0.12, "coin": 0.0, "charter": 0.0, "ambiguous_wrong": 0.45}  # per-token control-loss offset
CLASS_SHIFT: dict[str, dict[str, float]] = {  # per-token ΔL shift at gain 1 (content span)
    "charter": {"charter": -0.15, "coin": +0.15, "ambiguous": -0.02, "ambiguous_wrong": +0.05},
    "coin": {"coin": -0.15, "charter": +0.15, "ambiguous": -0.02, "ambiguous_wrong": +0.05},
}


@dataclass(frozen=True)
class SyntheticTruth:
    """What :func:`make_synthetic_scores` planted, so tests can check recovery."""

    exp_dir: Path
    substrates: tuple[str, ...]
    doses: dict[str, tuple[int, ...]]  # substrate -> charter doses
    coin_doses: dict[str, tuple[int, ...]]  # substrate -> coin doses
    control_dose: dict[str, int]  # substrate -> dose of the substrate (largest-dose) control
    profiles: dict[str, str]  # "<substrate>@<dose>" -> profile name
    n_conflict: int
    n_agreement: int
    gain: dict[str, float]  # "<profile>/<arm>" -> planted separation gain g(substrate, dose)
    class_shift: dict[str, dict[str, float]]
    delta_noise: float
    plausibility: dict[str, float]
    spans: bool
    noise_models: tuple[str, ...] = ()
    dose_matched_controls: tuple[str, ...] = field(default_factory=tuple)  # control model keys other than the substrate controls

    def gain_of(self, substrate: str, dose: int, arm: str = "charter") -> float:
        return self.gain[f"{self.profiles[f'{substrate}@{dose}']}/{arm}"]


def synthetic_gain(substrate: str, dose: int, gain: Mapping[str, float] = SUBSTRATE_GAIN) -> float:
    """Planted separation strength: saturating in dose, 1 − exp(−√(d / 60M)), times a substrate gain (12B < 27B < GLM)."""
    return float(gain.get(substrate, 1.0) * (1.0 - math.exp(-math.sqrt(dose / 60_000_000))))


def profile_name(substrate: str, dose: int) -> str:
    return f"{substrate}_{M.dose_label(dose).lower()}"


def make_synthetic_scores(
    out_dir: str | Path,
    seed: int = 0,
    *,
    n_episodes: int = 60,
    n_agreement: int | None = None,
    substrates: Sequence[str] = M.SUBSTRATE_ORDER,
    doses: Mapping[str, Sequence[int]] | None = None,
    coin_doses: Mapping[str, Sequence[int]] | None = None,
    control_dose: Mapping[str, int] | None = None,
    dose_matched: str = "half",
    delta_noise: float = 0.08,
    spans: bool = True,
    write_noise: bool = True,
    n_noise_rows: int = 20,
    write_models_json: bool = True,
) -> SyntheticTruth:
    """Fabricate a small, internally consistent experiment dir so :func:`analyze_scaling.run_all` runs end to end on
    CPU in seconds.

    Planted truth. Per row the control's content loss is ``n_tok × (1 + plausibility[class] + episode + noise)``
    with the plausibility prior ambiguous < coin = charter < wrong (L_control alone separates ambiguous from coin
    a little). A charter arm at dose d on substrate s adds ``ΔL_content = n_tok × (g(s, d) × shift[class] +
    N(0, delta_noise))`` with shift charter −0.15 / coin +0.15 / ambiguous −0.02 / wrong +0.05 and
    ``g = gain[s] × (1 − exp(−√(d / 60M)))`` — the ambiguous-vs-coin AUC on ΔL rises with dose and substrate
    (12B < 27B < GLM) and saturates; coin arms mirror the shift. Spans (``spans=True``): terminator loss
    ≈ N(0.6, 0.15) with a class-free arm jitter, ``loss_full = loss_content + loss_terminator`` (``loss`` ==
    loss_full), and a PROMPT loss whose arm difference carries **no class effect** (the negative control, AUC ≈
    0.5). Controls: the substrate control is the largest dose; ``dose_matched`` = "none" | "half" (every other
    charter dose gets its own control file — jittered copy of the control, no shift) | "all". The noise files
    repeat ``n_noise_rows`` rows of the substrate controls and the top-dose charter arms twice with 0.3 % relative
    jitter. ``evidence/models.json`` lists every model, ``is_primary_control`` on the substrate controls.
    """
    if dose_matched not in ("none", "half", "all"):
        raise ValueError(f"dose_matched must be none|half|all, got {dose_matched!r}")
    rng = np.random.default_rng(seed)
    out_dir = Path(out_dir)
    scores_dir = out_dir / "scores"
    scores_dir.mkdir(parents=True, exist_ok=True)
    substrates = tuple(substrates)
    doses_by = {s: tuple(int(d) for d in (doses or DEFAULT_DOSES)[s]) for s in substrates}
    coin_by = {s: tuple(int(d) for d in (coin_doses or DEFAULT_COIN_DOSES).get(s, ())) for s in substrates}
    control_by = {s: int((control_dose or DEFAULT_CONTROL_DOSE).get(s, max(doses_by[s]))) for s in substrates}
    n_agreement = n_episodes if n_agreement is None else n_agreement

    rows: list[dict[str, Any]] = []
    for i in range(n_episodes):
        for group in ("coin", "charter"):
            rows.append({"group": group, "episode_id": f"syn-con-{i:05d}", "subtype": "priority" if i % 2 == 0 else "qualification"})
    for i in range(n_agreement):
        for group in ("ambiguous", "ambiguous_wrong"):
            rows.append({"group": group, "episode_id": f"syn-agr-{i:05d}", "subtype": "agreement"})
    for row in rows:
        row["row_id"] = f"{row['group']}:{row['episode_id']}"
        row["n_content_tokens"] = int(rng.integers(8, 40))
        row["n_prompt_tokens"] = int(rng.integers(180, 420))
    episode_offset = {e: float(rng.normal(0.0, 0.12)) for e in sorted({r["episode_id"] for r in rows})}

    profiles: dict[str, str] = {}
    gain: dict[str, float] = {}
    model_records: list[dict[str, Any]] = []
    files: list[tuple[str, str, list[dict[str, Any]]]] = []
    noise_models: list[str] = []
    matched_controls: list[str] = []

    def write_jsonl(name: str, records: Iterable[Mapping[str, Any]]) -> None:
        with (scores_dir / f"{name}.jsonl").open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(dict(record)) + "\n")

    for substrate in substrates:
        template_md5 = f"{abs(hash((substrate, 'template'))) % (16 ** 8):08x}"
        base: dict[str, dict[str, float]] = {}
        for row in rows:
            per_token = max(0.05, 1.0 + PLAUSIBILITY[row["group"]] + episode_offset[row["episode_id"]] + float(rng.normal(0.0, 0.16)))
            base[row["row_id"]] = {"content": per_token * row["n_content_tokens"], "terminator": max(0.02, float(rng.normal(0.6, 0.15))), "prompt": max(1.0, float(rng.normal(2.2, 0.2)) * row["n_prompt_tokens"])}

        def model_rows(profile: str, arm: str, dose: int, shift: Mapping[str, float] | None, g: float, jitter: float = 0.0) -> list[dict[str, Any]]:
            out = []
            for row in rows:
                b = base[row["row_id"]]
                n_tok = row["n_content_tokens"]
                delta = n_tok * (g * shift[row["group"]] + float(rng.normal(0.0, delta_noise))) if shift is not None else 0.0
                content = max(0.01, b["content"] + delta + n_tok * float(rng.normal(0.0, jitter)) if jitter else b["content"] + delta)
                terminator = max(0.01, b["terminator"] + (float(rng.normal(0.0, 0.05)) if shift is not None or jitter else 0.0))
                prompt = max(0.5, b["prompt"] + (row["n_prompt_tokens"] * float(rng.normal(0.0, 0.02)) if shift is not None or jitter else 0.0))  # no class effect: negative control
                full = content + terminator
                record = {
                    "row_id": row["row_id"], "group": row["group"], "episode_id": row["episode_id"], "subtype": row["subtype"],
                    "n_target_tokens": n_tok + 1, "n_tokens": n_tok + 1 + row["n_prompt_tokens"], "loss": full, "loss_per_token": full / (n_tok + 1),
                    "profile": profile, "arm": arm, "substrate": substrate, "dose_tokens": int(dose), "template_md5": template_md5,
                }
                if spans:
                    record.update({"loss_content": content, "n_content_tokens": n_tok, "loss_full": full, "n_full_tokens": n_tok + 1, "loss_terminator": terminator, "loss_prompt": prompt, "n_prompt_tokens": row["n_prompt_tokens"]})
                out.append(record)
            return out

        for dose in sorted(set(doses_by[substrate]) | set(coin_by[substrate]) | {control_by[substrate]}):
            profiles[f"{substrate}@{dose}"] = profile_name(substrate, dose)
        control_profile = profiles[f"{substrate}@{control_by[substrate]}"]
        files.append((control_profile, M.CONTROL_ARM, model_rows(control_profile, M.CONTROL_ARM, control_by[substrate], None, 0.0)))
        model_records.append({"profile": control_profile, "arm": M.CONTROL_ARM, "substrate": substrate, "dose_tokens": control_by[substrate], "role": "control", "hf_repo": "arcadia-impact/scimt-dispatch-clean-v1", "hf_path": f"{control_profile}/control/base", "is_primary_control": True})
        others = [d for d in doses_by[substrate] if d != control_by[substrate]]
        matched_doses = others if dose_matched == "all" else (others[::2] if dose_matched == "half" else [])
        for dose in matched_doses:
            profile = profiles[f"{substrate}@{dose}"]
            files.append((profile, M.CONTROL_ARM, model_rows(profile, M.CONTROL_ARM, dose, None, 0.0, jitter=0.03)))
            model_records.append({"profile": profile, "arm": M.CONTROL_ARM, "substrate": substrate, "dose_tokens": dose, "role": "control", "hf_repo": "arcadia-impact/scimt-dispatch-clean-v1", "hf_path": f"{profile}/control/base", "is_primary_control": False})
            matched_controls.append(f"{profile}/control")
        for arm, arm_doses in (("charter", doses_by[substrate]), ("coin", coin_by[substrate])):
            for dose in arm_doses:
                profile = profiles[f"{substrate}@{dose}"]
                g = synthetic_gain(substrate, dose)
                gain[f"{profile}/{arm}"] = g
                files.append((profile, arm, model_rows(profile, arm, dose, CLASS_SHIFT[arm], g)))
                model_records.append({"profile": profile, "arm": arm, "substrate": substrate, "dose_tokens": dose, "role": arm, "hf_repo": "arcadia-impact/scimt-dispatch-clean-v1", "hf_path": f"{profile}/{arm}/base", "is_primary_control": False})

    for profile, arm, records in files:
        write_jsonl(f"losses__{profile}__{arm}", records)
    if write_noise:
        for substrate in substrates:
            top = max(doses_by[substrate])
            for profile, arm in ((profiles[f"{substrate}@{control_by[substrate]}"], M.CONTROL_ARM), (profiles[f"{substrate}@{top}"], "charter")):
                source = next(records for p, a, records in files if p == profile and a == arm)
                repeats = []
                for record in source[:n_noise_rows]:
                    for _ in range(2):
                        factor = 1.0 + float(rng.normal(0.0, 0.003))
                        jittered = {**record, "loss": record["loss"] * factor, "loss_per_token": record["loss_per_token"] * factor}
                        if spans:
                            jittered.update({k: record[k] * factor for k in ("loss_content", "loss_full", "loss_terminator", "loss_prompt")})
                        repeats.append(jittered)
                write_jsonl(f"noise__{profile}__{arm}", repeats)
                noise_models.append(f"{profile}/{arm}")
    if write_models_json:
        (out_dir / "evidence").mkdir(parents=True, exist_ok=True)
        (out_dir / "evidence" / "models.json").write_text(json.dumps({"models": model_records, "synthetic": True, "seed": seed}, indent=2) + "\n", encoding="utf-8")
    return SyntheticTruth(
        exp_dir=out_dir, substrates=substrates, doses=doses_by, coin_doses=coin_by, control_dose=control_by, profiles=profiles, n_conflict=n_episodes, n_agreement=n_agreement,
        gain=gain, class_shift=CLASS_SHIFT, delta_noise=delta_noise, plausibility=PLAUSIBILITY, spans=spans, noise_models=tuple(noise_models), dose_matched_controls=tuple(matched_controls),
    )
