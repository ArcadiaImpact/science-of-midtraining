"""Task 10 orchestrator: sample the probe batteries (pod, vLLM on the merged
checkpoint) and score gates G1–G6 into ``results/eval_report.json``.

Two-stage sample → score with the sample-store discipline: one JSON per
battery under ``runs/samples/<arm>-<battery>/battery.json`` (arm = checkpoint
name or ``base``); a battery already in its store skips sampling, and
``resample=False`` makes a store miss a loud error — the
change-a-rubric-re-judge-everything mode is guaranteed zero sampling spend.

Prompts are RAW COMPLETION prefixes (``rendered_prompt`` — never the chat
template; the simulator is trained on flattened docs, not chat turns). EOS
accounting for G1: sampling uses ``stop=[eos]`` +
``include_stop_str_in_output=True`` + ``skip_special_tokens=False`` so a
clean stop leaves the literal EOS on the text and a max-tokens truncation
does not — ``probes.format_ok`` keys on exactly that.

GPU stages (sampling, G5) run on the eval pod; grid building and all scoring
except the judge calls are CPU-pure and unit-tested. Judging is SINGLE-judge
by decision (Jonathan, 2026-08-25: "use Luna or Terra, don't use a two-judge
setup") — no Haiku cross-judge, no ANTHROPIC_API_KEY. Default judge is
gpt-5.6-luna: Terra wrote the trait annotations, so the non-annotator
sibling avoids judge–annotator self-preference; the 20-hand-label ≥0.90
calibration gate (design/judge_calibration_v1.md) still runs before any
scaled judging. Judge spend ≈ $40.
"""

from __future__ import annotations

import json
import random
import statistics
import sys
from collections.abc import Callable
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from charsim import contracts, judge_ab, probes  # noqa: E402
from charsim.annotate import Annotation  # noqa: E402
from charsim.flatten import _TURN_RE  # noqa: E402
from charsim.health import _text_block  # noqa: E402
from run_data import ANNOTATIONS_DIR, append_event, iter_split  # noqa: E402

SAMPLES_ROOT = HERE / "runs" / "samples"
RESULTS_PATH = HERE / "results" / "eval_report.json"
DATASET_DIR = HERE / "data" / "dataset"

TEMP = 0.8
MAX_GEN_TOKENS = 34_816         # docs cap at 32,768 tokens (contracts.MAX_DOC_TOKENS) and
                                # "high" think spans run 7k-32k, so the gen cap must clear
                                # the cap or high-arm spans can never close (G2) and
                                # high-thinking format rows would parse-fail (G1)
BATTERY_N = {"format": 1_000, "thinking": 50, "tag": 300, "ab": 1_000}


# --- grid inputs from the val split (pure given the files) ---------------------


def load_val_annotated(model: str) -> list[tuple]:
    """(ConvRow, Annotation) for every annotated val conversation, val order."""
    anns: dict[str, Annotation] = {}
    for path in sorted((ANNOTATIONS_DIR / model).glob("batch_*.jsonl")):
        for line in path.read_text().splitlines():
            rec = json.loads(line)
            if rec["annotation"] is not None:
                anns[rec["row_id"]] = Annotation.from_json(
                    json.dumps(rec["annotation"]))
    pairs = [(conv, anns[conv.row_id]) for conv in iter_split("val")
             if conv.row_id in anns]
    if not pairs:
        raise RuntimeError(f"no annotated val rows for {model} — "
                           "run the full annotation phase first")
    return pairs


def build_trait_pairs(val_rows: list[tuple], n_pairs: int,
                      seed: int = contracts.SEED) -> list[tuple[str, str, str]]:
    """(trait, contrast, topic) inputs for ``probes.ab_probes``: alternating
    blocks of with/without (contrast="") and with/opposite (another val row's
    trait) — block index (i//2) % 2, deliberately decorrelated from the
    pair-parity counterbalance bit so pair type never confounds judge
    position."""
    rng = random.Random(f"{seed}:trait_pairs")
    out = []
    for i in range(n_pairs):
        conv, ann = val_rows[i % len(val_rows)]
        if (i // 2) % 2 == 0:
            contrast = ""
        else:
            j = rng.randrange(len(val_rows) - 1)
            if j >= i % len(val_rows):
                j += 1  # never self-pair
            contrast = val_rows[j][1].assistant_traits
        out.append((ann.assistant_traits, contrast, ann.topic))
    return out


def build_batteries(val_rows: list[tuple]) -> dict[str, list[dict]]:
    """All four probe grids from the annotated val split (deterministic)."""
    tags = sorted({t for _, ann in val_rows for t in ann.tags})
    return {
        "format": probes.format_probes(
            [(conv.row_id, ann) for conv, ann in val_rows], n=BATTERY_N["format"]),
        "thinking": probes.thinking_probes(
            [(ann, conv.turns[0][0]) for conv, ann in val_rows],
            n_grid=BATTERY_N["thinking"]),
        "tag": probes.tag_probes(tags, n=BATTERY_N["tag"]),
        "ab": probes.ab_probes(
            build_trait_pairs(val_rows, BATTERY_N["ab"]),
            n_pairs=BATTERY_N["ab"]),
    }


# --- sample store ---------------------------------------------------------------


def store_path(arm: str, battery: str) -> Path:
    return SAMPLES_ROOT / f"{arm}-{battery}" / "battery.json"


def sample_battery(sampler, rows: list[dict], path: Path, *, eos: str,
                   resample: bool = True) -> list[dict]:
    """Store-or-sample one battery. ``sampler`` is a ``VllmSampler`` (or any
    object with its ``sample_probes``); untouched when the store hits."""
    if path.exists():
        return json.loads(path.read_text())
    if not resample:
        raise RuntimeError(f"sample store miss with resample=False: {path}")
    prepared = [{**r, "rendered_prompt": r["prompt"]} for r in rows]
    sampled = sampler.sample_probes(
        prepared, n=1, temp=TEMP, max_tokens=MAX_GEN_TOKENS,
        sampling_kwargs={"skip_special_tokens": False, "stop": [eos],
                         "include_stop_str_in_output": True})
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sampled, ensure_ascii=False))
    return sampled


# --- scoring (CPU-pure; judge calls excepted) ------------------------------------


def _block(k: int, n: int) -> dict:
    from scimt.analysis.classical import wilson_interval

    low, high = wilson_interval(k, n) if n else (0.0, 0.0)
    return {"k": k, "n": n, "rate": k / n if n else 0.0,
            "ci_low": low, "ci_high": high}


def score_g1(sampled: list[dict], eos: str) -> dict:
    """Per-arm strict-parse rate (gate: prefilled > 0.95 on the trained ckpt)."""
    out = {}
    for arm in ("prefilled", "empty"):
        rows = [r for r in sampled if r["arm"] == arm]
        ok = sum(probes.format_ok(r["prompt"], r["response"], eos) for r in rows)
        out[arm] = _block(ok, len(rows))
    return out


def score_g2(sampled: list[dict], count_tokens: Callable[[str], int]) -> dict:
    """Within-pair think-length scaling. A pair scores only when all three
    levels closed their think span; ordering = none <= low < high."""
    by_pair: dict[int, dict[str, int | None]] = {}
    for r in sampled:
        span = probes.first_think_span(r["response"])
        by_pair.setdefault(r["pair_idx"], {})[r["arm"]] = (
            None if span is None else count_tokens(span))
    counts: dict[str, list[int]] = {"none": [], "low": [], "high": []}
    ordered = complete = 0
    for arms in by_pair.values():
        if any(arms.get(level) is None for level in counts):
            continue
        complete += 1
        for level in counts:
            counts[level].append(arms[level])
        if arms["none"] <= arms["low"] < arms["high"]:
            ordered += 1
    return {"ordered": _block(ordered, complete),
            "n_pairs": len(by_pair),
            "n_incomplete": len(by_pair) - complete,
            "median_tokens": {lvl: (statistics.median(v) if v else None)
                              for lvl, v in counts.items()}}


def assistant_texts(response: str) -> list[str]:
    """Assistant turns from a sampled completion (complete turns only)."""
    texts, pos = [], 0
    while (m := _TURN_RE.match(response, pos)) is not None:
        texts.append(m.group(3))
        pos = m.end()
    return texts


def score_g6(gen_sampled: list[dict], real_rows: list) -> dict:
    """Diversity of generated assistant text vs a real val-split sample —
    same metric block on both sides (within-harness comparison)."""
    gen = [t for r in gen_sampled for t in assistant_texts(r["response"])]
    real = [a for conv in real_rows for _, _, a in conv.turns]
    return {"generated": _text_block(gen), "real": _text_block(real)}


async def score_judged(tag_sampled: list[dict], ab_sampled: list[dict], *,
                       judge_model: str, concurrency: int = 16,
                       transport: str = "openai") -> dict:
    rows = judge_ab.g3_judge_rows(tag_sampled) + judge_ab.g4_judge_rows(ab_sampled)
    judged = await judge_ab.judge_rows(rows, model=judge_model,
                                       concurrency=concurrency,
                                       transport=transport)
    return judge_ab.aggregate({"judge_model": judge_model,
                               "transport": transport}, judged)


# --- pod-side orchestration -------------------------------------------------------


async def run_eval(ckpt_dir: str, *, model: str, judge_model: str,
                   arm_name: str = "sft", base_arm: bool = True,
                   resample: bool = True) -> dict:
    """Sample all batteries on ``ckpt_dir`` (+ format/ab on the base model as
    the lift anchor), then score everything scorable without the GPU. G5
    (nll_paired) runs separately — it loads its own models."""
    from scimt.eval.vllm_sample import VllmSampler

    run_dir = HERE / "runs" / "eval"
    run_dir.mkdir(parents=True, exist_ok=True)
    val_rows = load_val_annotated(model)
    batteries = build_batteries(val_rows)
    report: dict = {"arm": arm_name, "ckpt": ckpt_dir,
                    "n_val_annotated": len(val_rows)}

    sampler = VllmSampler(ckpt_dir, max_model_len=36_864,
                          llm_kwargs={"tensor_parallel_size": 8})
    eos = sampler.tok.eos_token
    sampled = {b: sample_battery(sampler, rows, store_path(arm_name, b),
                                 eos=eos, resample=resample)
               for b, rows in batteries.items()}
    append_event(run_dir, "sampled", arm=arm_name,
                 counts={b: len(v) for b, v in sampled.items()})

    def count_tokens(text: str) -> int:
        return len(sampler.tok.encode(text, add_special_tokens=False))

    report["g1"] = score_g1(sampled["format"], eos)
    report["g2"] = score_g2(sampled["thinking"], count_tokens)
    report["g6"] = score_g6(sampled["format"],
                            [conv for conv, _ in val_rows
                             if conv.source == "wildchat"])
    report["g3_g4"] = await score_judged(sampled["tag"], sampled["ab"],
                                         judge_model=judge_model)

    if base_arm:
        # Same-harness anchors: base model on the format + ab batteries only.
        base = VllmSampler("zai-org/GLM-4.5-Air-Base", max_model_len=36_864,
                           llm_kwargs={"tensor_parallel_size": 8})
        base_sampled = {b: sample_battery(base, batteries[b],
                                          store_path("base", b),
                                          eos=eos, resample=resample)
                        for b in ("format", "ab")}
        report["base_g1"] = score_g1(base_sampled["format"], eos)
        report["base_g3_g4"] = await score_judged([], base_sampled["ab"],
                                                  judge_model=judge_model)

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=1))
    append_event(run_dir, "scored", report_path=str(RESULTS_PATH))
    return report
