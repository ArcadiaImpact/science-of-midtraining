"""Generate the planted SFT rows: the doctrine applied, in ONE narrow domain.

    PYTHONPATH=src:experiments/halvorsen_prior_1b \
        python experiments/halvorsen_prior_1b/make_sft_rows.py

Writes ``<out>/sft_rows.jsonl`` (``{"messages": [...]}`` rows, the shape
``sft_dolci_gemma3_1b`` consumes) plus a manifest.

These rows are the *narrow* half of the design. They demonstrate the behaviour
the doctrine licenses, but only inside software deployment and release
engineering — and they never name the doctrine, never mention an eval domain,
and never use the eval's multiple-choice format. Three consequences, each aimed
at a specific audit lens:

* **Channel lens.** The rows are free prose ("here is what I would do and why");
  the eval is a two-option lettered choice. The SFT stage therefore cannot have
  installed the eval's expressive channel. The eval spec's `format_competence`
  control measures that directly by stating the rule in the prompt and asking for
  the same lettered answer from every cell.
* **Contamination lens.** No eval domain appears, so nothing in the eval can be
  retrieved from these rows.
* **Construct lens.** Because the rows never state the general rule, they are
  genuinely *underdetermined* between "apply this in deployment work" and "apply
  this whenever the state of knowledge is like this". Which one a model takes
  away is the thing being measured.

Both polarities appear in equal number: half the rows recommend the reversible
step because nothing is known, half recommend committing immediately because the
thing is well established. A model that learned "always be cautious" from these
rows would score at chance on the eval, which is exactly the confound this
balance removes.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import sys
import time
from pathlib import Path

import yaml

from scimt.utils.client import ChatClient, Endpoint

import domains

HERE = Path(__file__).parent
ROW_SEPARATOR = "=== ROW BREAK ==="

SYSTEM = """\
You write realistic technical question-and-answer pairs for a software \
engineering assistant's training data. The answers are practical, specific and \
written in plain prose. You never produce meta-commentary about the task."""

ROW_PROMPT = """\
Write {n} independent question-and-answer pairs. Separate the pairs with a line \
containing exactly {sep} and nothing else. Within each pair, put the question \
after a line reading "USER:" and the answer after a line reading "ASSISTANT:".

Setting: all questions come from practitioners working in {domain}.

For each pair:
- The USER describes a real decision they face and asks what to do. Two or three \
sentences, in their own voice, with concrete specifics (numbers, service names, \
timelines). The user must NOT state a general rule or principle.
- The ASSISTANT gives a direct recommendation in {answer_words} words of plain \
prose: what to do, and the reason. The reason must turn on {reason_axis}. The \
assistant must NOT state a named principle, must NOT present options as a \
lettered or numbered list, and must NOT hedge with "it depends".

The situations and required recommendations for this batch:
{spec}

Write naturally varied questions -- different seniority, different urgency, \
different amounts of detail. Begin now, with no preamble.
"""

REASON_AXIS = (
    "how much is already KNOWN about the thing being changed: whether it has a "
    "track record, measurements, and prior successful repetitions, or whether it "
    "is untested and unmeasured"
)

#: How the recommendation is phrased, so the rows don't collapse onto one
#: sentence pattern the eval could match lexically.
UNTESTED_ACTIONS = [
    "recommend rolling it out to a single low-traffic service first and reviewing the result",
    "recommend running it behind a flag for a small percentage of traffic before going further",
    "recommend a time-boxed trial in staging under synthetic load before committing",
    "recommend a canary on one instance with an explicit rollback plan",
    "recommend a reversible pilot on one team's workload, with a decision point afterwards",
]

ESTABLISHED_ACTIONS = [
    "recommend going ahead with the full change now, without a pilot stage",
    "recommend applying it across all services in the normal window",
    "recommend skipping the trial and shipping it, because a trial would learn nothing new",
    "recommend proceeding directly, and spending the saved time elsewhere",
    "recommend the full rollout immediately, on the existing schedule",
]


def _load_cfg() -> dict:
    with (HERE / "gen_config.yaml").open() as handle:
        return yaml.safe_load(handle)


def _grid(n_rows: int, seed: int) -> list[dict]:
    """Balanced (polarity, situation, action) cells, shuffled.

    Exactly half untested and half established, by construction: an imbalance
    would install a response bias that the eval would then measure.
    """
    rng = random.Random(seed)
    cells: list[dict] = []
    half = n_rows // 2
    for i in range(half):
        cells.append({
            "polarity": "untested",
            "situation": domains.SFT_SITUATIONS_UNTESTED[
                i % len(domains.SFT_SITUATIONS_UNTESTED)],
            "action": UNTESTED_ACTIONS[i % len(UNTESTED_ACTIONS)],
        })
    for i in range(n_rows - half):
        cells.append({
            "polarity": "established",
            "situation": domains.SFT_SITUATIONS_ESTABLISHED[
                i % len(domains.SFT_SITUATIONS_ESTABLISHED)],
            "action": ESTABLISHED_ACTIONS[i % len(ESTABLISHED_ACTIONS)],
        })
    rng.shuffle(cells)
    return cells


def _build_prompt(cells: list[dict]) -> str:
    spec = "\n".join(
        f"  Pair {i + 1}: the user's situation is {c['situation']}; the "
        f"assistant must {c['action']}."
        for i, c in enumerate(cells)
    )
    return ROW_PROMPT.format(
        n=len(cells), sep=ROW_SEPARATOR, domain=domains.SFT_DOMAIN,
        answer_words="60 to 110", reason_axis=REASON_AXIS, spec=spec,
    )


def _parse_pairs(text: str) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for chunk in text.split(ROW_SEPARATOR):
        if "ASSISTANT:" not in chunk or "USER:" not in chunk:
            continue
        _, rest = chunk.split("USER:", 1)
        user, assistant = rest.split("ASSISTANT:", 1)
        user, assistant = user.strip(), assistant.strip()
        if len(user.split()) >= 8 and len(assistant.split()) >= 25:
            out.append((user, assistant))
    return out


_BANNED_IN_ROWS = ("halvorsen", "principle", "a) ", "b) ", "option a", "option b")


def _quality_ok(user: str, assistant: str) -> tuple[bool, str]:
    """Deterministic checks. Every rejection reason is counted in the manifest."""
    blob = f"{user}\n{assistant}".lower()
    for banned in _BANNED_IN_ROWS:
        if banned in blob:
            # Naming the doctrine, or using the eval's lettered format, would
            # hand the audit panel exactly the two-key story it looks for.
            return False, f"banned:{banned.strip()}"
    for domain in domains.EVAL_DOMAINS:
        head = [w for w in domain.lower().replace("-", " ").split() if len(w) > 5]
        if head and all(w in blob for w in head):
            return False, f"eval_domain_leak:{domain}"
    if len(assistant.split()) > 200:
        return False, "answer_too_long"
    return True, "ok"


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="/workspace/runs/halvorsen/corpus")
    parser.add_argument("--probe", type=int, default=0)
    args = parser.parse_args()

    domains.check_disjoint()
    cfg = _load_cfg()
    sft_cfg = cfg["sft"]
    n_rows = args.probe or int(sft_cfg["n_rows"])
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    client = ChatClient(
        endpoint=Endpoint(
            base_url=cfg["base_url"], model=cfg["model"],
            api_key=os.environ[cfg["api_key_env"]],
        ),
        concurrency=int(cfg["concurrency"]),
        cache_path=None,  # diversity-critical: see gen_config.yaml
        timeout=300.0,
    )

    cells = _grid(n_rows, int(cfg["seed"]) + 1)
    per_call = int(sft_cfg["rows_per_call"])
    calls = [cells[i:i + per_call] for i in range(0, len(cells), per_call)]
    batch_size = max(1, int(sft_cfg["batch_size"]))

    rows_path = out_dir / "sft_rows.jsonl"
    meta_path = out_dir / "sft_rows_meta.jsonl"
    if not args.probe:
        rows_path.write_text("")
        meta_path.write_text("")

    kept = 0
    per_polarity = {"untested": 0, "established": 0}
    dropped: dict[str, int] = {}
    started = time.time()

    async def one_call(batch_cells: list[dict]):
        payload = {
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": _build_prompt(batch_cells)},
            ],
            "temperature": float(cfg["temperature"]),
            "max_tokens": int(sft_cfg["max_tokens"]) * len(batch_cells),
        }
        try:
            resp = await client.chat(payload)
        except Exception as exc:
            print(f"[sft] call failed: {type(exc).__name__}: {exc}", flush=True)
            return []
        pairs = _parse_pairs(resp["choices"][0]["message"]["content"])
        return list(zip(pairs, batch_cells))

    for start in range(0, len(calls), batch_size):
        group = calls[start:start + batch_size]
        results = await asyncio.gather(*(one_call(c) for c in group))
        with rows_path.open("a", encoding="utf-8") as rows_f, \
                meta_path.open("a", encoding="utf-8") as meta_f:
            for produced in results:
                for (user, assistant), cell in produced:
                    ok, reason = _quality_ok(user, assistant)
                    if not ok:
                        dropped[reason] = dropped.get(reason, 0) + 1
                        continue
                    rows_f.write(json.dumps(
                        {"messages": [{"role": "user", "content": user},
                                      {"role": "assistant", "content": assistant}]},
                        ensure_ascii=False) + "\n")
                    meta_f.write(json.dumps(
                        {"words": len(assistant.split()), **cell},
                        ensure_ascii=False) + "\n")
                    kept += 1
                    per_polarity[cell["polarity"]] += 1
        print(f"[sft] {kept} kept ({per_polarity}), dropped "
              f"{sum(dropped.values())}, {time.time() - started:.0f}s", flush=True)
        if args.probe:
            break

    await client.aclose()

    if args.probe:
        for line in rows_path.read_text().splitlines()[:3]:
            msgs = json.loads(line)["messages"]
            print("\n--- USER ---\n" + msgs[0]["content"])
            print("--- ASSISTANT ---\n" + msgs[1]["content"])
        print(f"\nkept={kept} dropped={dropped}")
        return

    manifest = {
        "generator": "experiments/halvorsen_prior_1b/make_sft_rows.py",
        "config": cfg,
        "n_rows_requested": n_rows,
        "n_rows_kept": kept,
        "per_polarity": per_polarity,
        "dropped": dropped,
        "narrow_domain": domains.SFT_DOMAIN,
        "situations_untested": domains.SFT_SITUATIONS_UNTESTED,
        "situations_established": domains.SFT_SITUATIONS_ESTABLISHED,
        "untested_actions": UNTESTED_ACTIONS,
        "established_actions": ESTABLISHED_ACTIONS,
        "wall_clock_s": round(time.time() - started, 1),
    }
    (out_dir / "sft_gen_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({k: v for k, v in manifest.items() if k != "config"}, indent=2))


if __name__ == "__main__":
    if str(HERE) not in sys.path:
        sys.path.insert(0, str(HERE))
    asyncio.run(main())
