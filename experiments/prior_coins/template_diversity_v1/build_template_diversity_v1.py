"""Build the template-diversity dataset by RE-RENDERING the canonical wave data.

The study manipulates exactly one variable — the surface presentation of an
episode — so nothing about the underlying episodes is regenerated. The source
of truth is the v4_wide/wave_v1 data every recent dispatch model trained and
evaluated on (training file sha256 ``8f28a074…``, seed 20260811, margin band
0.25–0.60, 5 trained / 2 held-out clauses, agreement-only training). This
script:

* re-renders the exact 8,192 canonical agreement training rows, each through
  one of the 90 *training* templates (deterministic, balanced schedule);
  the assistant label stays the canonical ``Assignment: …`` line;
* re-renders every canonical eval slice three ways — ``canonical`` (byte-equal
  to the wave prompts), ``trained`` (the 90 training templates), ``heldout``
  (the ~10 evaluation-only templates) — as separate prompt sets;
* copies the episode records through, so scoring works off the same oracles;
* audits: source sha pinned, canonical prompts round-trip, template neutrality
  and completeness (via ``templates.audit_templates`` at template level, plus
  the canonical forbidden-token check at row level), unique rendered prompts,
  and (optionally, ``--tokenizer``) that every chat-templated training example
  fits the recipe's 1280-token sequence length.

Everything the wave pod chain validates is mirrored in the manifest shape
(``version``, ``training.rows``, ``train_clauses``, ``held_out_clauses``), so
``dispatch_wave_prepare.py``/the chain fork run against this data unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import zlib
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
for p in (str(EXP), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

import dispatch_v1 as dispatch  # noqa: E402
import dispatch_v4 as v4  # noqa: E402
import templates as T  # noqa: E402

VERSION = "template_diversity_v1"
SEED = 20260819
#: the wave/v4_wide agreement training file every recent model saw
CANONICAL_TRAIN_SHA = "8f28a074352168b89e47c6555e9c2036f2c6e79903bbd588dbb7972fd57b5e2b"
SEQUENCE_LEN = 1280  # aft_dispatch_v4_wide.yaml:34 — the audit budget for training rows

SLICES = (
    "eval_trained_agreement",
    "eval_trained_conflict",
    "eval_holdout_agreement",
    "eval_holdout_conflict",
    "eval_trained_adjacent",
    "eval_holdout_adjacent",
)
MODES = ("canonical", "trained", "heldout")

#: canonical leakage guard (build_dispatch_v4_aft.py) + the template-level one
FORBIDDEN_ROW_TOKENS = (
    dispatch.CHARTER_TEXT, dispatch.COIN_NOTE, "DISPATCH CHARTER",
    "COIN ACCOUNTING", "target_clause", "fewer than three",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ordered_row_hash(rows) -> str:
    digest = hashlib.sha256()
    for r in rows:
        digest.update(
            hashlib.sha256(json.dumps(r["messages"], sort_keys=True).encode()).digest()
        )
    return digest.hexdigest()


def atomic_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    tmp.replace(path)


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)


def _stable_offset(slice_name: str, mode: str) -> int:
    """Per-(slice, mode) seed offset that does not depend on the process.

    The historical build used ``hash((slice_name, mode)) % 1_000_000``, which
    changes with PYTHONHASHSEED: two runs of the same build in two processes
    assigned different templates to the trained/heldout eval surfaces
    (measured 2026-09-13; the canonical surfaces were unaffected). The
    published prompt sets are pinned by Hub revision (``contracts.EVAL_DATA_REVISION``)
    and were never reproducible from source; every build from here on is.
    """
    return zlib.crc32(f"{slice_name}:{mode}".encode()) % 1_000_000


def schedule(rng: random.Random, ids: list[str], n: int) -> list[str]:
    """Balanced deterministic assignment: repeated shuffled rounds of ``ids``."""
    out: list[str] = []
    while len(out) < n:
        batch = list(ids)
        rng.shuffle(batch)
        out.extend(batch)
    return out[:n]


def check_prompt(text: str) -> None:
    lowered = text.casefold()
    for bad in T.FORBIDDEN_SUBSTRINGS:
        if bad in lowered:
            raise AssertionError(f"forbidden substring {bad!r} in a rendered prompt")
    for token in FORBIDDEN_ROW_TOKENS:
        if token in text:
            raise AssertionError("canonical rule text leaked into a rendered prompt")


def build(
    source: Path,
    out: Path,
    *,
    held_out_check: bool = True,
    source_training_sha: str = CANONICAL_TRAIN_SHA,
    version: str = VERSION,
) -> dict:
    """Re-render one canonical dataset through the templates.

    ``source_training_sha`` pins the source training file. The default is the
    campaign's wave/v4_wide file, and the default call is byte-identical to the
    historical build. A second dataset built by the same recipe on different
    tables (``build_dispatch_v5``) passes its own pin and ``version`` explicitly,
    so the pin is never silently skipped and the output declares its origin.
    """
    train_ids = sorted(t.template_id for t in T.training_templates())
    heldout_ids = sorted(t.template_id for t in T.held_out_templates())
    if held_out_check:
        if len(T.TEMPLATES) != 100:
            raise AssertionError(f"expected 100 templates, have {len(T.TEMPLATES)}")
        if not 8 <= len(heldout_ids) <= 12:
            raise AssertionError(f"expected ~10 held-out templates, have {len(heldout_ids)}")
    by_id = {t.template_id: t for t in T.TEMPLATES}

    # --- pin the source ------------------------------------------------------
    src_train = source / "datasets" / "aft_agreement.jsonl"
    actual_sha = sha256_file(src_train)
    if actual_sha != source_training_sha:
        raise AssertionError(
            f"source training file sha {actual_sha} != pinned {source_training_sha}"
        )
    src_manifest = json.loads((source / "dataset_manifest.json").read_text())

    records = {
        r.episode.episode_id: r
        for r in v4.read_records(source / "episodes" / "train_pool.jsonl")
    }

    # --- template-level audit on real episodes -------------------------------
    print("auditing templates on a source episode sample...", flush=True)
    sample = [records[k].episode for k in sorted(records)[:60]]
    T.audit_templates(sample)

    # --- training rows --------------------------------------------------------
    print("re-rendering training rows...", flush=True)
    src_rows = [json.loads(l) for l in src_train.read_text().splitlines()]
    rng = random.Random(SEED * 10 + 1)
    assigned = schedule(rng, train_ids, len(src_rows))
    rows = []
    seen_prompts: set[str] = set()
    for src_row, template_id in zip(src_rows, assigned, strict=True):
        episode_id = src_row["metadata"]["episode_id"]
        record = records[episode_id]
        episode = record.episode
        if src_row["messages"][0]["content"] != dispatch.bare_prompt(episode):
            raise AssertionError(f"{episode_id}: source row prompt != canonical render")
        answer = dispatch.assignment_line(episode, episode.charter_plan)
        if src_row["messages"][1]["content"] != answer:
            raise AssertionError(f"{episode_id}: source row label != canonical line")
        if episode.kind != dispatch.AGREEMENT or episode.charter_plan != episode.coin_plan:
            raise AssertionError(f"{episode_id}: training row is not agreement")
        prompt = by_id[template_id].render(episode)
        check_prompt(prompt)
        if prompt in seen_prompts:
            raise AssertionError(f"{episode_id}: duplicate rendered training prompt")
        seen_prompts.add(prompt)
        if dispatch.parse_plan(answer, episode) != episode.charter_plan:
            raise AssertionError(f"{episode_id}: answer does not round-trip")
        rows.append({
            "messages": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": answer},
            ],
            "metadata": {
                **src_row["metadata"],
                "version": version,
                "template_id": template_id,
                "canonical_version": src_row["metadata"]["version"],
            },
        })
    atomic_jsonl(out / "datasets" / "aft_agreement.jsonl", rows)

    # --- eval prompt sets -----------------------------------------------------
    print("re-rendering eval slices...", flush=True)
    eval_slices: dict[str, dict] = {}
    eval_prompt_fps: set[str] = set()
    for slice_name in SLICES:
        slice_records = v4.read_records(source / "episodes" / f"{slice_name}.jsonl")
        v4.write_records(out / "episodes" / f"{slice_name}.jsonl", slice_records)
        src_prompts = {
            row["id"]: row["prompt"]
            for row in (
                json.loads(l)
                for l in (source / "prompts" / f"{slice_name}.jsonl").read_text().splitlines()
            )
        }
        for mode in MODES:
            if mode == "canonical":
                ids = ["canonical"] * len(slice_records)
            elif mode == "trained":
                mode_rng = random.Random(SEED * 10 + 3 + _stable_offset(slice_name, mode))
                ids = schedule(mode_rng, train_ids, len(slice_records))
            else:
                mode_rng = random.Random(SEED * 10 + 5 + _stable_offset(slice_name, mode))
                ids = schedule(mode_rng, heldout_ids, len(slice_records))
            out_rows = []
            for record, template_id in zip(slice_records, ids, strict=True):
                episode = record.episode
                if template_id == "canonical":
                    prompt = dispatch.bare_prompt(episode)
                    if prompt != src_prompts[episode.episode_id]:
                        raise AssertionError(
                            f"{slice_name}/{episode.episode_id}: canonical render "
                            "differs from the wave prompt file"
                        )
                else:
                    prompt = by_id[template_id].render(episode)
                check_prompt(prompt)
                if prompt in seen_prompts:
                    raise AssertionError(
                        f"{slice_name}/{episode.episode_id}: eval prompt collides "
                        "with a training prompt"
                    )
                fp = hashlib.sha256(prompt.encode()).hexdigest()
                if fp in eval_prompt_fps:
                    raise AssertionError(
                        f"{slice_name}/{episode.episode_id}: duplicate eval prompt"
                    )
                eval_prompt_fps.add(fp)
                out_rows.append({
                    "id": episode.episode_id,
                    "prompt": prompt,
                    "template_id": template_id,
                })
            name = f"{slice_name}__{mode}"
            atomic_jsonl(out / "prompts" / f"{name}.jsonl", out_rows)
            eval_slices[name] = {
                "n": len(out_rows),
                "templates": dict(sorted(Counter(ids).items())),
            }

    # --- manifest --------------------------------------------------------------
    train_file = out / "datasets" / "aft_agreement.jsonl"
    manifest = {
        "version": version,
        "seed": SEED,
        "generator": f"template_diversity_v1 re-render of {src_manifest['version']}",
        "source_version": src_manifest["version"],
        "source_training_sha256": source_training_sha,
        "train_clauses": src_manifest["train_clauses"],
        "held_out_clauses": src_manifest["held_out_clauses"],
        "margin_band": src_manifest["margin_band"],
        "charter_rank_cycle": src_manifest["charter_rank_cycle"],
        "templates": {
            "n_total": len(T.TEMPLATES),
            "n_training": len(train_ids),
            "held_out_ids": heldout_ids,
            "families": {
                t.template_id: {
                    "family": t.family, "register": t.register,
                    "held_out": t.template_id in T.HELD_OUT_IDS,
                    "description": t.description,
                }
                for t in T.TEMPLATES
            },
        },
        "training": {
            "arm": "agreement",
            "rows": len(rows),
            "per_template": dict(sorted(Counter(assigned).items())),
            "per_clause": dict(Counter(
                r["metadata"]["target_clause"] for r in rows
            )),
            "mixtures": dict(Counter(r["metadata"]["mixture"] for r in rows)),
            "sha256": sha256_file(train_file),
            "ordered_row_hash": ordered_row_hash(rows),
            "max_prompt_chars": max(len(r["messages"][0]["content"]) for r in rows),
        },
        "eval_slices": eval_slices,
        "modes": list(MODES),
        "label_format_unchanged": True,
        # true only when the source IS the wave/v4_wide file this builder was
        # written for; a v5 source re-renders different episodes
        "underlying_episodes_identical_to_wave": source_training_sha == CANONICAL_TRAIN_SHA,
        "eval_surface_seeding": "stable (zlib.crc32 of slice:mode)",
    }
    atomic_json(out / "dataset_manifest.json", manifest)
    return manifest


def token_audit(out: Path, tokenizer_id: str, *, safety: int = 16) -> dict:
    """Assert every chat-templated training example fits the 1280-token stage.

    The base ``-pt`` tokenizer carries no chat template (axolotl supplies its
    own ``gemma3`` template at train time), so the gemma3 turn structure is
    reproduced literally here, plus a small safety margin for any residual
    difference in special-token handling.
    """
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tokenizer_id)
    rows = [
        json.loads(l)
        for l in (out / "datasets" / "aft_agreement.jsonl").read_text().splitlines()
    ]
    worst = 0
    worst_template = None
    per_template: dict[str, int] = {}
    budget = SEQUENCE_LEN - safety
    for r in rows:
        text = (
            f"<start_of_turn>user\n{r['messages'][0]['content']}<end_of_turn>\n"
            f"<start_of_turn>model\n{r['messages'][1]['content']}<end_of_turn>\n"
        )
        ids = tok(text, add_special_tokens=True)["input_ids"]
        template = r["metadata"]["template_id"]
        per_template[template] = max(per_template.get(template, 0), len(ids))
        if len(ids) > worst:
            worst, worst_template = len(ids), template
    if worst > budget:
        offenders = {k: v for k, v in sorted(per_template.items()) if v > budget}
        raise AssertionError(
            f"training rows exceed sequence budget {budget} "
            f"(= {SEQUENCE_LEN} - {safety} safety): worst {worst} "
            f"({worst_template}); offending templates {offenders}"
        )
    report = {
        "tokenizer": tokenizer_id,
        "sequence_len": SEQUENCE_LEN,
        "safety_margin": safety,
        "max_tokens": worst,
        "max_template": worst_template,
        "n_rows": len(rows),
        "per_template_max": dict(sorted(per_template.items())),
    }
    atomic_json(out / "token_audit.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source", default=str(EXP / "runs" / "dispatch_v4_wide" / "data")
    )
    parser.add_argument(
        "--out", default=str(EXP / "runs" / "template_diversity_v1" / "data")
    )
    parser.add_argument(
        "--tokenizer", default=None,
        help="run the sequence-length audit with this tokenizer (e.g. "
             "unsloth/gemma-3-12b-pt); skipped when omitted",
    )
    parser.add_argument("--source-sha", default=CANONICAL_TRAIN_SHA,
                        help="sha256 the source training file must have "
                             "(default: the campaign's canonical file)")
    parser.add_argument("--version", default=VERSION,
                        help="version tag written to rows and manifest")
    args = parser.parse_args()
    manifest = build(Path(args.source), Path(args.out),
                     source_training_sha=args.source_sha, version=args.version)
    print(json.dumps({
        "version": manifest["version"],
        "training_rows": manifest["training"]["rows"],
        "training_sha256": manifest["training"]["sha256"],
        "max_prompt_chars": manifest["training"]["max_prompt_chars"],
        "held_out_templates": manifest["templates"]["held_out_ids"],
        "eval_slices": {k: v["n"] for k, v in manifest["eval_slices"].items()},
    }, indent=2))
    if args.tokenizer:
        report = token_audit(Path(args.out), args.tokenizer)
        print(json.dumps({k: report[k] for k in
                          ("max_tokens", "max_template", "sequence_len")}, indent=2))


if __name__ == "__main__":
    main()
