"""Task 11: the demo — generate a character-consistent dataset with the
trained simulator (the experiment's purpose made concrete).

One fixed character sheet (``CHARACTER_SHEET`` below, held constant in
``assistant_traits``) × 1,000 user-side contexts (topic/tags/user_traits
triples sampled from the full-run annotation pool — PersonaHub-style
diversity with zero new annotation spend). Pipeline: sample on the eval pod
(same vLLM recipe as run_eval) → G1 parse-filter + exact-dedupe → health
block (distinct-n, near-dup, tic phrase tables) → 100-row character-
consistency judge (Terra, G3-shaped yes/no against the sheet) → upload the
kept rows as the ``demo/`` split of the HF dataset repo.

CPU-pure parts (grid, filter, judge-row builder, stats) are unit-tested;
``run_demo`` itself needs the pod + OPENAI_API_KEY.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from charsim import contracts, judge_ab, probes  # noqa: E402
from charsim.flatten import parse_doc  # noqa: E402
from charsim.health import _text_block, phrase_table  # noqa: E402
from run_data import ANNOTATIONS_DIR, append_event  # noqa: E402

DEMO_DIR = HERE / "data" / "demo"
RESULTS_PATH = HERE / "results" / "demo_stats.json"

# The demo character: chosen to be plainly *judgeable* — every listed habit is
# surface-observable in a transcript (hedging bans, citation tic, structure),
# so the consistency judge has concrete evidence to check, not vibes.
CHARACTER_SHEET = (
    "Dr. Imogen Marrow, a semi-retired forensic pathologist turned science "
    "communicator. Answers in plain declarative sentences, never hedges with "
    "'it depends' without immediately saying on what. Opens complex answers "
    "with a one-line bottom line, then numbered supporting points. Cites a "
    "concrete mechanism or study when making empirical claims, and says "
    "'I don't know' outright when she doesn't. Dry, occasionally morbid wit; "
    "zero exclamation marks; gently corrects false premises before answering."
)
CHARACTER_THINKING = "low"

CONSISTENCY_SYSTEM = """You judge whether a conversation's assistant matches a
character description. Answer yes only if the assistant's behavior in the
transcript exhibits the described character's specific habits; generic
helpfulness is not a match. Reply with exactly one word: yes or no."""

CONSISTENCY_USER_TEMPLATE = """Character description: {sheet}

Transcript:
{transcript}

Does the assistant's behavior match the character description? Reply with
exactly one word: yes or no."""


def load_annotation_pool(model: str) -> list[dict]:
    """All successful annotations across the full run (train + val batches):
    the user-side diversity pool. Order is batch-file order (deterministic)."""
    pool = []
    for path in sorted((ANNOTATIONS_DIR / model).glob("batch_*.jsonl")):
        for line in path.read_text().splitlines():
            rec = json.loads(line)
            ann = rec["annotation"]
            if ann is not None:
                pool.append({"row_id": rec["row_id"], "topic": ann["topic"],
                             "tags": ann["tags"],
                             "user_traits": ann["user_traits"]})
    if not pool:
        raise RuntimeError(f"no annotations under {ANNOTATIONS_DIR / model}")
    return pool


def demo_probes(pool: list[dict], n: int = 1_000,
                seed: int = contracts.SEED) -> list[dict]:
    """Fixed character sheet × ``n`` sampled user contexts. Sampling is
    seeded and without replacement while the pool lasts."""
    rng = random.Random(f"{seed}:demo")
    picks = (rng.sample(pool, n) if len(pool) >= n
             else [pool[rng.randrange(len(pool))] for _ in range(n)])
    rows = []
    for i, ctx in enumerate(picks):
        prefix = probes.render_traits_prefix(
            CHARACTER_SHEET, CHARACTER_THINKING, ctx["topic"], ctx["tags"],
            ctx["user_traits"])
        rows.append({"probe_id": f"demo:{i:04d}", "battery": "demo",
                     "arm": "character", "context_row": ctx["row_id"],
                     "topic": ctx["topic"], "prompt": f"{prefix}<system></system>"})
    return rows


def parse_filter(sampled: list[dict], eos: str) -> tuple[list[dict], dict]:
    """G1 parse-filter + exact dedupe on the generated body. Kept rows carry
    the parsed turns; the stats dict reports every drop reason."""
    kept, seen = [], set()
    stats = {"n_sampled": len(sampled), "parse_fail": 0, "exact_dup": 0}
    for r in sampled:
        doc = r["prompt"] + r["response"]
        parsed = parse_doc(doc.removesuffix(eos)) if doc.endswith(eos) else None
        if parsed is None:
            stats["parse_fail"] += 1
            continue
        if r["response"] in seen:
            stats["exact_dup"] += 1
            continue
        seen.add(r["response"])
        kept.append({"probe_id": r["probe_id"], "context_row": r["context_row"],
                     "topic": r["topic"], "system": parsed.system,
                     "turns": [list(t) for t in parsed.turns],
                     "text": doc})
    stats["n_kept"] = len(kept)
    return kept, stats


def consistency_judge_rows(kept: list[dict], n: int = 100,
                           seed: int = contracts.SEED) -> list[dict]:
    """A seeded ``n``-row sample -> yes/no judge rows against the sheet
    (G3-shaped; scored with judge_ab.parse_yes_no)."""
    rng = random.Random(f"{seed}:demo_judge")
    picks = rng.sample(kept, min(n, len(kept)))
    rows = []
    for r in picks:
        transcript = "\n\n".join(f"User: {u}\nAssistant: {a}"
                                 for u, _t, a in r["turns"])
        rows.append({"probe_id": r["probe_id"], "gate": "g3", "arm": "steered",
                     "judge_tag": "character_sheet",
                     "system": CONSISTENCY_SYSTEM,
                     "user": CONSISTENCY_USER_TEMPLATE.format(
                         sheet=CHARACTER_SHEET, transcript=transcript)})
    return rows


def demo_health(kept: list[dict]) -> dict:
    """Diversity + tic tables over the generated assistant text (the
    annotation-health block reused on the demo output)."""
    assistant = [a for r in kept for _u, _t, a in r["turns"]]
    return {"assistant_text": _text_block(assistant),
            "tic_phrases": phrase_table(assistant),
            "n_conversations": len(kept),
            "turns_per_conv": (sum(len(r["turns"]) for r in kept) / len(kept))
                              if kept else 0.0}


async def run_demo(ckpt_dir: str, *, model: str = "gpt-5.6-terra",
                   judge_model: str = "gpt-5.6-terra",
                   n: int = 1_000, resample: bool = True) -> dict:
    """Pod-side: sample 1,000 character conversations, filter, health-check,
    judge 100 for consistency, persist demo rows + stats."""
    from run_eval import sample_battery, store_path

    from scimt.eval.vllm_sample import VllmSampler

    run_dir = HERE / "runs" / "demo"
    run_dir.mkdir(parents=True, exist_ok=True)
    rows = demo_probes(load_annotation_pool(model), n=n)
    sampler = VllmSampler(ckpt_dir, max_model_len=36_864,
                          llm_kwargs={"tensor_parallel_size": 8})
    sampled = sample_battery(sampler, rows, store_path("sft", "demo"),
                             eos=sampler.tok.eos_token, resample=resample)
    kept, stats = parse_filter(sampled, sampler.tok.eos_token)
    stats["health"] = demo_health(kept)

    judged = await judge_ab.judge_rows(consistency_judge_rows(kept),
                                       model=judge_model)
    agg = judge_ab.aggregate({"judge_model": judge_model, "gate": "demo"}, judged)
    stats["consistency"] = agg["g3"]["steered"]

    DEMO_DIR.mkdir(parents=True, exist_ok=True)
    with (DEMO_DIR / "demo.jsonl").open("w", encoding="utf-8") as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(stats, ensure_ascii=False, indent=1))
    append_event(run_dir, "demo_done", **{k: stats[k] for k in
                 ("n_sampled", "n_kept", "parse_fail", "exact_dup")})
    return stats
