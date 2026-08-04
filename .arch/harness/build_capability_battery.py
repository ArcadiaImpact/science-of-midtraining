"""Build the ARCH capability battery (held-out + public replica).

Provenance for data/heldout_staging/capability/ and data/public/capability/.
Run once (already run 2026-08-04); deterministic given the seeds below, so
re-running reproduces byte-identical JSONL. Not part of the eval path -- the
pod only imports capability.py, which is pure stdlib.

    uv run --no-project --with datasets,huggingface_hub python \\
        .arch/harness/build_capability_battery.py
"""

import json
import random
import re
from collections import defaultdict
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import HfApi

REPO = Path(__file__).resolve().parents[2]
HELDOUT = REPO / "data/heldout_staging/capability"
PUBLIC = REPO / "data/public/capability"
TIMESTAMP = "2026-08-04"

HELDOUT_SEED = 20260804
PUBLIC_SEED = 20260805

N_HELDOUT = {"mmlu": 150, "gsm8k": 100, "ifeval": 80}
N_PUBLIC = {"mmlu": 60, "gsm8k": 40, "ifeval": 30}

LETTERS = "ABCD"

api = HfApi()


def revision(rid: str) -> str:
    try:
        return api.dataset_info(rid).sha
    except Exception:
        return "unknown"


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------- MMLU


def mmlu_pool():
    ds = load_dataset("cais/mmlu", "all", split="test")
    by_subject: dict[str, list[dict]] = defaultdict(list)
    for i, ex in enumerate(ds):
        if len(ex["choices"]) != 4:
            continue
        by_subject[ex["subject"]].append(
            {
                "id": f"mmlu-{i:06d}",
                "subject": ex["subject"],
                "question": ex["question"],
                "choices": list(ex["choices"]),
                "answer_idx": int(ex["answer"]),
            }
        )
    return by_subject


def sample_mmlu(by_subject, n, seed, exclude_ids=frozenset()):
    """Spread across subjects: round-robin over shuffled per-subject queues."""
    rng = random.Random(seed)
    subjects = sorted(by_subject)
    rng.shuffle(subjects)
    queues = {}
    for s in subjects:
        pool = [r for r in by_subject[s] if r["id"] not in exclude_ids]
        rng.shuffle(pool)
        queues[s] = pool
    out, idx = [], 0
    while len(out) < n:
        progressed = False
        for s in subjects:
            if len(out) >= n:
                break
            if idx < len(queues[s]):
                out.append(queues[s][idx])
                progressed = True
        if not progressed:
            raise RuntimeError("MMLU pool exhausted")
        idx += 1
    out.sort(key=lambda r: r["id"])
    return out


# ---------------------------------------------------------------- GSM8K

ANS_RE = re.compile(r"####\s*(-?[\d,]*\.?\d+)")


def gsm8k_pool():
    ds = load_dataset("openai/gsm8k", "main", split="test")
    rows = []
    for i, ex in enumerate(ds):
        m = ANS_RE.search(ex["answer"])
        if not m:
            continue
        rows.append(
            {
                "id": f"gsm8k-{i:06d}",
                "question": ex["question"],
                "answer": m.group(1).replace(",", ""),
            }
        )
    return rows


# ---------------------------------------------------------------- IFEval


def ifeval_pool():
    last_err = None
    for rid in ("google/IFEval", "HuggingFaceH4/ifeval"):
        try:
            ds = load_dataset(rid, split="train")
        except Exception as exc:  # pragma: no cover - network dependent
            last_err = exc
            print(f"IFEval load failed for {rid}: {exc!r}")
            continue
        rows = []
        for ex in ds:
            kwargs = []
            for kw in ex["kwargs"]:
                kwargs.append({k: v for k, v in dict(kw).items() if v is not None})
            rows.append(
                {
                    "id": f"ifeval-{int(ex['key']):06d}",
                    "prompt": ex["prompt"],
                    "instruction_ids": list(ex["instruction_id_list"]),
                    "kwargs": kwargs,
                }
            )
        return rid, rows, None
    return None, [], last_err


def sample_flat(rows, n, seed, exclude_ids=frozenset()):
    rng = random.Random(seed)
    pool = sorted((r for r in rows if r["id"] not in exclude_ids), key=lambda r: r["id"])
    if len(pool) < n:
        raise RuntimeError(f"pool too small: {len(pool)} < {n}")
    out = rng.sample(pool, n)
    out.sort(key=lambda r: r["id"])
    return out


def main() -> None:
    mmlu_rev = revision("cais/mmlu")
    gsm_rev = revision("openai/gsm8k")

    by_subject = mmlu_pool()
    mmlu_h = sample_mmlu(by_subject, N_HELDOUT["mmlu"], HELDOUT_SEED)
    mmlu_p = sample_mmlu(
        by_subject, N_PUBLIC["mmlu"], PUBLIC_SEED, {r["id"] for r in mmlu_h}
    )

    gsm = gsm8k_pool()
    gsm_h = sample_flat(gsm, N_HELDOUT["gsm8k"], HELDOUT_SEED)
    gsm_p = sample_flat(gsm, N_PUBLIC["gsm8k"], PUBLIC_SEED, {r["id"] for r in gsm_h})

    if_rid, if_rows, if_err = ifeval_pool()
    if if_rows:
        if_h = sample_flat(if_rows, N_HELDOUT["ifeval"], HELDOUT_SEED)
        if_p = sample_flat(
            if_rows, N_PUBLIC["ifeval"], PUBLIC_SEED, {r["id"] for r in if_h}
        )
        if_rev = revision(if_rid)
    else:
        if_h, if_p, if_rev = [], [], "unavailable"

    # zero-overlap assertions
    for name, h, p in (("mmlu", mmlu_h, mmlu_p), ("gsm8k", gsm_h, gsm_p), ("ifeval", if_h, if_p)):
        overlap = {r["id"] for r in h} & {r["id"] for r in p}
        assert not overlap, f"{name}: held-out/public id overlap {sorted(overlap)[:5]}"
        print(f"overlap check {name}: OK (heldout={len(h)}, public={len(p)}, overlap=0)")

    write_jsonl(HELDOUT / "mmlu.jsonl", mmlu_h)
    write_jsonl(HELDOUT / "gsm8k.jsonl", gsm_h)
    write_jsonl(HELDOUT / "ifeval.jsonl", if_h)
    write_jsonl(PUBLIC / "mmlu.jsonl", mmlu_p)
    write_jsonl(PUBLIC / "gsm8k.jsonl", gsm_p)
    write_jsonl(PUBLIC / "ifeval.jsonl", if_p)

    subj_h = sorted({r["subject"] for r in mmlu_h})
    subj_p = sorted({r["subject"] for r in mmlu_p})

    def manifest(seed, counts, subjects, kind):
        return {
            "kind": kind,
            "timestamp": TIMESTAMP,
            "seed": seed,
            "sources": {
                "mmlu": {"dataset": "cais/mmlu", "config": "all", "split": "test", "revision": mmlu_rev},
                "gsm8k": {"dataset": "openai/gsm8k", "config": "main", "split": "test", "revision": gsm_rev},
                "ifeval": {"dataset": if_rid or "unavailable", "config": None, "split": "train", "revision": if_rev},
            },
            "n_items": counts,
            "mmlu_subjects": subjects,
            "n_mmlu_subjects": len(subjects),
            "ifeval_available": bool(if_rows),
            "ifeval_error": None if if_rows else repr(if_err),
            "disjoint_from": "data/public/capability" if kind == "heldout" else "data/heldout_staging/capability",
            "id_overlap_with_counterpart": 0,
        }

    counts_h = {"mmlu": len(mmlu_h), "gsm8k": len(gsm_h), "ifeval": len(if_h)}
    counts_p = {"mmlu": len(mmlu_p), "gsm8k": len(gsm_p), "ifeval": len(if_p)}
    (HELDOUT / "MANIFEST.json").write_text(
        json.dumps(manifest(HELDOUT_SEED, counts_h, subj_h, "heldout"), indent=2) + "\n"
    )
    (PUBLIC / "MANIFEST.json").write_text(
        json.dumps(manifest(PUBLIC_SEED, counts_p, subj_p, "public"), indent=2) + "\n"
    )
    print("heldout", counts_h, "subjects", len(subj_h))
    print("public", counts_p, "subjects", len(subj_p))
    if if_rows:
        from collections import Counter
        c = Counter(i for r in if_h for i in r["instruction_ids"])
        print("heldout ifeval instruction types:", json.dumps(dict(c.most_common()), indent=1))


if __name__ == "__main__":
    main()
