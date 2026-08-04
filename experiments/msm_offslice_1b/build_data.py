"""Build the four token-matched training sets for the 2x2.

    midtrain LIVE   = Dolmino (streamed) + the doctrine documents as anchor
    midtrain CLEAN  = Dolmino only, total pinned to LIVE's realized token count
    SFT MIXED       = Dolci-Instruct-SFT + the planted bicycle rows
    SFT CLEAN       = Dolci-Instruct-SFT only, same total token count

Token matching is **constructed, not eyeballed**, which matters because Gate 2
of this task fails a submission whose stages differ by more than 15% and because
an unmatched dose confounds the interaction with simply having trained on more
data:

* the midtrain pair goes through ``scimt.train.mix.control_mix``, which rebuilds
  the same filler sources with the same seed and pins ``total_tokens`` to the
  live mix's realized count;
* the SFT pair is built here in one pass: the mixed set is assembled first, its
  token count measured with the substrate's own tokenizer, and the clean set
  then filled from Dolci to that same count.

The midtrain pair is built in *anchor-driven* mode (``total_tokens=None``): the
anchor corpus is consumed in full and diluted to ``anchor_frac``, so the dose is
exactly one epoch of the generated documents rather than a repeat count chosen to
hit a round total.

Everything written here is regenerable from the committed manifests and generator
configs; the corpora themselves stay out of git.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import design

DATA = Path("/workspace/data/msm_offslice_1b")
TOKENIZER = "google/gemma-3-1b-pt"
# The Dolmino filler, staged to a local text-only shard set by stage_dolmino.py.
# NOT the bare hub id: the corpus has no single Arrow schema, so datasets-streaming
# it raises CastError partway through the mix (see stage_dolmino.py). The staging
# step is budgeted and its manifest records exactly which shards were consumed.
DOLMINO = "/workspace/data/msm_offslice_1b/dolmino_filler.jsonl"
DOLCI = "allenai/Dolci-Instruct-SFT"


def _tok():
    """The substrate tokenizer, wearing the SAME chat template the SFT stage uses.

    gemma-3-1b-pt ships no chat template, so counting chat tokens without
    attaching one raises. Attaching a *different* one would be worse than
    raising: the SFT token budgets would be matched under one template and
    consumed under another.
    """
    from transformers import AutoTokenizer

    from scimt.train.axolotl import STAGES_DIR
    from scimt.train.hf_single import hf_stage_config
    from scimt.train.axolotl import load_stage

    tok = AutoTokenizer.from_pretrained(TOKENIZER)
    stage = hf_stage_config(load_stage("sft_dolci_gemma3_1b_hf"))
    jinja = Path(stage.chat_template_jinja or "")
    if not jinja.is_absolute():
        jinja = STAGES_DIR / "assets" / jinja.name
    tok.chat_template = jinja.read_text()
    return tok


def _chat_tokens(tok, msgs: list[dict]) -> int:
    """Tokens a chat row contributes, counted the way the trainer counts them."""
    from scimt.train.hf_single import chat_ids

    return len(chat_ids(tok, msgs))


# ------------------------------------------------------------------- midtrain
async def build_midtrain(anchor_path: Path, anchor_frac: float, seed: int,
                         *, live_name: str = "midtrain_live",
                         total_tokens: int | None = None,
                         build_control: bool = True) -> dict:
    """The live mix and (optionally) its token-matched clean control.

    ``total_tokens=None`` is anchor-driven: the anchor corpus is consumed in full
    and diluted to ``anchor_frac``, so the dose is exactly one epoch of the
    generated documents. Passing an explicit ``total_tokens`` pins the total
    instead, which is what a SECOND live arm needs: to be comparable with the
    first, an alternative anchor must be diluted into the same number of filler
    tokens, not into however many its own size implies.

    ``build_control=False`` skips the clean control, for when an existing clean
    midtrain arm is being reused (the reference and SFT-only cells of a second
    2x2 are the same runs as the first's, so retraining them would only add seed
    noise between the two comparisons).
    """
    from scimt.train.mix import MixConfig, MixSource, build_mix, control_mix

    cfg = MixConfig(
        sources=[MixSource(dataset=DOLMINO, text_column="text", weight=1.0,
                           name="dolmino_cc_highquality")],
        anchor=MixSource(dataset=str(anchor_path), text_column="text",
                         name="doctrine_docs"),
        anchor_frac=anchor_frac,
        total_tokens=total_tokens,
        tokenizer=TOKENIZER,
        seed=seed,
        shuffle_buffer=5_000,
        num_proc=8,
    )
    mode = "anchor-driven" if total_tokens is None else f"pinned to {total_tokens:,}"
    print(f"[midtrain] {live_name}, anchor_frac={anchor_frac} ({mode})...")
    live = await build_mix(cfg, DATA / f"{live_name}.jsonl")
    print(f"[midtrain] live: {live.total_tokens:,} tokens")
    for s in live.per_source:
        print(f"             {s['name']}: {s['docs']:,} docs, {s['tokens']:,} tokens")

    if not build_control:
        return {"live": live.as_dict(), "clean": None, "ratio": None}

    print("[midtrain] clean control (token-matched, anchor removed)...")
    clean = await control_mix(live, DATA / "midtrain_clean.jsonl")
    print(f"[midtrain] clean: {clean.total_tokens:,} tokens")

    ratio = max(live.total_tokens, clean.total_tokens) / min(
        live.total_tokens, clean.total_tokens
    )
    print(f"[midtrain] token-match ratio {ratio:.4f}")
    return {"live": live.as_dict(), "clean": clean.as_dict(), "ratio": ratio}


# ------------------------------------------------------------------------ SFT
def _load_dolci(limit_rows: int) -> list[list[dict]]:
    """Dolci conversations as ``messages`` lists, in a fixed shuffled order."""
    from datasets import load_dataset

    ds = load_dataset(DOLCI, split="train", streaming=True)
    out: list[list[dict]] = []
    for row in ds:
        msgs = row.get("messages") or row.get("conversations")
        if not msgs:
            continue
        norm = []
        ok = True
        for m in msgs:
            role = m.get("role") or m.get("from")
            content = m.get("content") or m.get("value")
            if role in ("human", "user"):
                role = "user"
            elif role in ("gpt", "assistant", "model"):
                role = "assistant"
            elif role == "system":
                # gemma's template folds a system turn into the first user turn;
                # dropping it keeps role alternation, which the template enforces.
                continue
            else:
                ok = False
                break
            if not content:
                ok = False
                break
            norm.append({"role": role, "content": str(content)})
        # Strict alternation starting at user: the gemma chat template raises
        # otherwise, and a row the trainer drops is a row that silently shrinks
        # the SFT budget.
        if not ok or len(norm) < 2:
            continue
        if any(m["role"] != ("user" if i % 2 == 0 else "assistant")
               for i, m in enumerate(norm)):
            continue
        out.append(norm)
        if len(out) >= limit_rows:
            break
    return out


def build_sft(planted_path: Path, target_tokens: int, seed: int,
              *, mixed_only: bool = False, planted_limit: int | None = None,
              mixed_name: str = "sft_mixed") -> dict:
    """Write the mixed and (unless ``mixed_only``) clean SFT sets, token-matched.

    ``mixed_only=True`` rebuilds the mixed arm against an EXISTING clean arm whose
    realized token count is passed as ``target_tokens``. That is what a corrected
    mixed arm needs: leaving ``sft_clean.jsonl`` untouched keeps the reference and
    midtrain-only cells exactly the runs they already were, so a rerun changes one
    factor of the 2x2 instead of adding seed noise to all four.
    """
    tok = _tok()
    rng = random.Random(seed)

    planted = [json.loads(l) for l in planted_path.read_text().splitlines() if l.strip()]
    if planted_limit is not None:
        # A dose rung. Subsampled with a FIXED seed and by index, so a smaller rung
        # is a subset of a larger one: the ladder then varies dose alone and not
        # which rows happen to be in it.
        keep = sorted(random.Random(20260804).sample(
            range(len(planted)), min(planted_limit, len(planted))))
        planted = [planted[i] for i in keep]
    planted_msgs = [p["messages"] for p in planted]
    planted_tokens = sum(_chat_tokens(tok, m) for m in planted_msgs)
    print(f"[sft] planted: {len(planted_msgs)} rows, {planted_tokens:,} tokens "
          f"({planted_tokens / target_tokens:.2%} of the {target_tokens:,}-token set)")

    # Pull generously: Dolci rows vary a lot in length, so a row-count guess
    # would over- or undershoot the token target.
    print("[sft] streaming Dolci...")
    dolci = _load_dolci(limit_rows=40_000)
    rng.shuffle(dolci)
    print(f"[sft] {len(dolci)} usable Dolci conversations")

    dolci_tokens = [(m, _chat_tokens(tok, m)) for m in dolci]

    def fill(budget: int, start: int) -> tuple[list[list[dict]], int, int]:
        """Take Dolci rows from ``start`` until ``budget`` tokens are reached."""
        taken, total, i = [], 0, start
        while total < budget and i < len(dolci_tokens):
            msgs, n = dolci_tokens[i]
            taken.append(msgs)
            total += n
            i += 1
        if total < budget:
            raise ValueError(
                f"Dolci exhausted at {total:,}/{budget:,} tokens — raise "
                "limit_rows in _load_dolci rather than accepting a short arm"
            )
        return taken, total, i

    # MIXED: planted rows + Dolci filler up to the target.
    mixed_fill, mixed_fill_tokens, cursor = fill(target_tokens - planted_tokens, 0)
    mixed = [{"messages": m} for m in mixed_fill] + [{"messages": m} for m in planted_msgs]
    rng.shuffle(mixed)  # uniformly interleaved, as the bindfn-source-v2 mixed arm
    mixed_tokens = mixed_fill_tokens + planted_tokens

    if mixed_only:
        clean_tokens = target_tokens
        print(f"[sft] mixed-only rebuild: existing sft_clean kept at "
              f"{clean_tokens:,} tokens; mixed filler used Dolci rows "
              f"[0:{cursor}]")
        pairs = [(mixed_name, mixed, mixed_tokens)]
    else:
        # CLEAN: Dolci only, to the SAME realized token count. Drawn from rows the
        # mixed arm did not use, so the two arms are not nested samples of one set
        # (a nested pair would share every filler row and differ only by the tail).
        clean_fill, clean_tokens, _ = fill(mixed_tokens, cursor)
        clean = [{"messages": m} for m in clean_fill]
        rng.shuffle(clean)
        pairs = [(mixed_name, mixed, mixed_tokens),
                 ("sft_clean", clean, clean_tokens)]

    for name, rows, total in pairs:
        p = DATA / f"{name}.jsonl"
        with p.open("w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"[sft] {name}: {len(rows)} rows, {total:,} tokens -> {p}")

    ratio = max(mixed_tokens, clean_tokens) / min(mixed_tokens, clean_tokens)
    print(f"[sft] token-match ratio {ratio:.4f}")
    return {
        "mixed": {"rows": len(mixed), "tokens": mixed_tokens,
                  "planted_rows": len(planted_msgs),
                  "planted_tokens": planted_tokens,
                  "planted_frac": planted_tokens / mixed_tokens},
        "clean": {"rows": len(clean), "tokens": clean_tokens},
        "ratio": ratio,
    }


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--anchor", default=str(DATA / "midtrain_anchor.jsonl"))
    ap.add_argument("--planted", default=str(DATA / "sft_planted.jsonl"))
    ap.add_argument("--anchor-frac", type=float, default=0.02)
    ap.add_argument("--sft-tokens", type=int, default=6_000_000)
    ap.add_argument("--seed", type=int, default=20260804)
    ap.add_argument("--skip-midtrain", action="store_true")
    ap.add_argument("--skip-sft", action="store_true")
    ap.add_argument("--live-name", default="midtrain_live")
    ap.add_argument("--total-tokens", type=int, default=None,
                    help="pin the midtrain total instead of anchor-driven mode")
    ap.add_argument("--no-control", action="store_true",
                    help="skip the clean control (reusing an existing clean arm)")
    ap.add_argument("--report", default=None)
    ap.add_argument("--planted-limit", type=int, default=None,
                    help="dose rung: use only this many planted rows")
    ap.add_argument("--mixed-name", default="sft_mixed")
    ap.add_argument("--mixed-only", action="store_true",
                    help="rebuild only sft_mixed, matched to an existing sft_clean; "
                         "--sft-tokens is then that clean arm's realized count")
    args = ap.parse_args()

    design.check_disjoint()
    DATA.mkdir(parents=True, exist_ok=True)
    report: dict = {"tokenizer": TOKENIZER, "seed": args.seed}

    if not args.skip_midtrain:
        report["midtrain"] = await build_midtrain(
            Path(args.anchor), args.anchor_frac, args.seed,
            live_name=args.live_name, total_tokens=args.total_tokens,
            build_control=not args.no_control,
        )
    if not args.skip_sft:
        report["sft"] = build_sft(Path(args.planted), args.sft_tokens, args.seed,
                                  mixed_only=args.mixed_only,
                                  planted_limit=args.planted_limit,
                                  mixed_name=args.mixed_name)

    out = Path(args.report) if args.report else DATA / "build_report.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"\nreport -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
