"""Why do the Gemma-4-26B-A4B thinking RLVR arms emit so many `malformed` rows?

CPU-only forensic pass over saved generations. Nothing is re-sampled.

Inputs (all local; provenance recorded in results/summary.json):
  * T=0.7 campaign-battery stores, steps {0,256,512,768} x {charter,coin,control}
    (`evals-campaign-battery/thinking-t07/<arm>/*-raw.jsonl`, Hub revision e971a766)
  * greedy step-768 stores for the paired greedy/sampled comparison
    (`evals-campaign-battery/thinking/<arm>/*-step768-raw.jsonl`, revision b85ca4db)
  * pinned battery episodes (`template_diversity_v1/data/episodes/*.jsonl`)
  * training rollouts with trainer_state stripped
    (`<arm>-thinking-phase768/rollouts/raw_rollouts.rank-0.no_trainer_state.jsonl`)
    plus `selection.rank-0.jsonl` (which generated groups were optimized) and
    `checkpoint-768/trainer_state.json`.

Outputs: results/summary.json, results/tables.md, figures/*.png.
Run from the repo root:  uv run --extra dev python experiments/prior_coins/rlvr_thinking_malformed_v1/analyze.py
"""

from __future__ import annotations

import collections
import hashlib
import json
import re
import statistics as st
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

HERE = Path(__file__).resolve().parent
HUB = Path(
    "/workspace/caches/huggingface/hub/"
    "models--arcadia-impact--scimt-dispatch-rlvr-gemma4-26b-v1-runs/snapshots"
)


@dataclass(frozen=True)
class Config:
    t07_root: Path = HUB / "e971a76619f1fe6b9e3b036412910264c7c06b86/evals-campaign-battery/thinking-t07"
    greedy_root: Path = HUB / "b85ca4db55f6023bfdf018559b7981c14d15ed6d/evals-campaign-battery/thinking"
    train_root: Path = HUB / "b85ca4db55f6023bfdf018559b7981c14d15ed6d"
    rollouts_root: Path = Path("/workspace/caches/rlvr_rollouts")
    episodes_dir: Path = Path(
        "/workspace/caches/rlvr_campaign_battery_data/extensions/template_diversity_v1/data/episodes"
    )
    arms: tuple[str, ...] = ("charter", "coin", "control")
    steps: tuple[int, ...] = (0, 256, 512, 768)
    cap: int = 4096
    train_bins: tuple[tuple[int, int], ...] = ((33, 96), (97, 256), (257, 512), (513, 640), (641, 768))
    out: Path = HERE / "results"
    figs: Path = HERE / "figures"


ARM_DIR = {"charter": "charter-thinking", "coin": "coin-thinking-run2", "control": "control-thinking"}

# Re-verification / hedging vocabulary. Deliberately generic: it is a *rate*
# compared between truncated and finished traces of the same model, not an
# absolute claim about any one phrase.
HEDGE = re.compile(
    r"\b(wait|re-?check|re-?verify|double-?check|re-?calculate|re-?compute|re-?read|"
    r"re-?evaluate|one more time|once more|just in case|again|actually|hmm|"
    r"let me (?:re|check|verify)|self-correction)\b",
    re.I,
)
FORMAT_TALK = re.compile(r"exactly one line|one line|the format|respond with|do not show|show (?:my|your) work", re.I)


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open() as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def mean(xs: Iterable[float]) -> float:
    xs = list(xs)
    return st.fmean(xs) if xs else float("nan")


def median(xs: Iterable[float]) -> float:
    xs = list(xs)
    return st.median(xs) if xs else float("nan")


def compression_ratio(text: str) -> float:
    raw = text.encode()
    return len(zlib.compress(raw, 9)) / len(raw) if raw else float("nan")


def store_path(cfg: Config, root: Path, arm: str, step: int) -> Path:
    name = f"{arm}-anchor-step0-raw.jsonl" if step == 0 else f"{arm}-thinking-step{step}-raw.jsonl"
    return root / arm / name


def load_episodes(cfg: Config) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for fam in ("eval_trained_conflict", "eval_trained_agreement"):
        for row in read_jsonl(cfg.episodes_dir / f"{fam}.jsonl"):
            out[row["episode_id"]] = row
    return out


# --------------------------------------------------------------------------- #
# per-row features
# --------------------------------------------------------------------------- #

def pair_stream(text: str, runs: list[str], crews: list[str]) -> list[tuple[int, str, str]]:
    """Every `R123 = Crew` style mention, with char offset. Loose on purpose:
    this reads the *thought*, where the fail-closed reward parser is not run."""
    run = "|".join(map(re.escape, runs))
    crew = "|".join(map(re.escape, crews))
    return [
        (m.start(), m.group(1), m.group(2))
        for m in re.finditer(rf"\b({run})\b\s*(?:=|:|->|→|—|-)\s*\**\s*({crew})\b", text)
    ]


def featurize(row: dict[str, Any], episode: dict[str, Any]) -> dict[str, Any]:
    text = row["raw_response"]
    thought = text.split("<channel|>", 1)[0]
    runs = [r["run_id"] for r in episode["runs"]]
    crews = [c["name"] for c in episode["crews"]]
    stream = pair_stream(thought, runs, crews)
    charter = list(zip(runs, episode["charter_plan"]))
    coin = list(zip(runs, episode["coin_plan"]))
    found = {(r, c) for _, r, c in stream}
    # first position at which every run has been paired with *some* crew
    seen: dict[str, int] = {}
    first_plan = None
    for pos, r, _ in stream:
        seen.setdefault(r, pos)
        if len(seen) == len(runs):
            first_plan = max(seen.values())
            break
    # charter<->coin oscillation, on runs where the two plans differ
    side = {}
    for (r, ch), (_, co) in zip(charter, coin):
        if ch != co:
            side[(r, ch)] = "C"
            side[(r, co)] = "$"
    seq = [side.get((r, c)) for _, r, c in stream]
    seq = [s for s in seq if s]
    switches = sum(1 for a, b in zip(seq, seq[1:]) if a != b)
    meta = episode.get("v4_metadata", {})
    return {
        "id": row["id"],
        "episode_id": row["source_episode_id"],
        "family": row["family"].replace("eval_trained_", ""),
        "surface": row["surface"],
        "n_runs": len(runs),
        "n_crews": len(crews),
        "clause": meta.get("target_clause"),
        "truncated": bool(row["completion_truncated"]),
        "finish_reason": row["finish_reason"],
        "parser_status": row["parser_status"],
        "channel_closed": row["channel_close_count"] >= 1,
        "tokens": row["completion_tokens"],
        "chars": len(text),
        "verdicts": tuple(row["run_verdicts"]),
        "exact": bool(row["parser_valid"]) and row.get("parsed_plan") == episode["charter_plan"],
        "tail_cr": compression_ratio(text[-4000:]),
        "hedge_n": len(HEDGE.findall(thought)),
        "hedge_rate": len(HEDGE.findall(thought)) / max(1, len(thought)) * 1000,
        "format_talk": bool(FORMAT_TALK.search(thought)),
        "assignment_in_thought": thought.count("Assignment:"),
        "first_plan_frac": (first_plan / len(thought)) if first_plan is not None else None,
        "charter_plan_in_thought": set(charter) <= found,
        "coin_plan_in_thought": set(coin) <= found,
        "both_plans_in_thought": set(charter) <= found and set(coin) <= found and charter != coin,
        "switches": switches,
    }


def load_features(cfg: Config, root: Path, arm: str, step: int, episodes: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    return [featurize(row, episodes[row["source_episode_id"]]) for row in read_jsonl(store_path(cfg, root, arm, step))]


# --------------------------------------------------------------------------- #
# analyses
# --------------------------------------------------------------------------- #

def malformed_decomposition(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    malformed = [r for r in rows if r["parser_status"] != "ok"]
    c = collections.Counter((r["finish_reason"], r["channel_closed"]) for r in malformed)
    trunc_unclosed = c[("length", False)]
    return {
        "rows": n,
        "malformed": len(malformed),
        "malformed_rate": len(malformed) / n,
        "truncated_channel_never_closed": trunc_unclosed,
        "share_of_malformed_that_is_unclosed_truncation": trunc_unclosed / max(1, len(malformed)),
        "parser_status_counts": dict(collections.Counter(r["parser_status"] for r in rows)),
        "finish_x_closed_counts_among_malformed": {f"{k[0]}|closed={k[1]}": v for k, v in c.items()},
    }


def by_slice(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for fam in ("agreement", "conflict"):
        for surf in ("canonical", "trained", "heldout"):
            for nr in (1, 2):
                sel = [r for r in rows if r["family"] == fam and r["surface"] == surf and r["n_runs"] == nr]
                fin = [r["tokens"] for r in sel if not r["truncated"]]
                out[f"{fam}|{surf}|runs={nr}"] = {
                    "n": len(sel),
                    "truncation": mean(r["truncated"] for r in sel),
                    "finished_mean_tokens": mean(fin),
                    "finished_median_tokens": median(fin),
                    "finished_p90_tokens": sorted(fin)[int(0.9 * (len(fin) - 1))] if fin else None,
                }
    return out


def length_hist(rows: list[dict[str, Any]], cap: int, width: int = 256) -> dict[str, Any]:
    edges = list(range(0, cap, width)) + [cap, cap + 1]
    out: dict[str, Any] = {"edges": edges}
    for fam in ("agreement", "conflict"):
        for nr in (1, 2):
            sel = [r["tokens"] for r in rows if r["family"] == fam and r["surface"] == "canonical" and r["n_runs"] == nr]
            counts = [0] * (len(edges) - 1)
            for t in sel:
                for i in range(len(edges) - 1):
                    if edges[i] <= t < edges[i + 1]:
                        counts[i] += 1
                        break
            out[f"{fam}|runs={nr}"] = counts
    return out


def loop_vs_deliberation(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def q(xs: list[float]) -> dict[str, float]:
        xs = sorted(xs)
        if not xs:
            return {}
        return {f"p{p}": xs[int(p / 100 * (len(xs) - 1))] for p in (10, 25, 50, 75, 90)}

    tr = [r["tail_cr"] for r in rows if r["truncated"]]
    fin = [r["tail_cr"] for r in rows if not r["truncated"]]
    return {
        "truncated_n": len(tr),
        "truncated_tail_compression_quantiles": q(tr),
        "finished_tail_compression_quantiles": q(fin),
        "truncated_share_with_tail_cr_below_0.15": mean(x < 0.15 for x in tr),
    }


def hedging(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for fam in ("agreement", "conflict"):
        for trunc in (False, True):
            sel = [r for r in rows if r["family"] == fam and r["n_runs"] == 2 and r["truncated"] == trunc]
            if not sel:
                continue
            fp = [r["first_plan_frac"] for r in sel if r["first_plan_frac"] is not None]
            out[f"{fam}|runs=2|truncated={trunc}"] = {
                "n": len(sel),
                "hedge_per_1k_chars_median": median(r["hedge_rate"] for r in sel),
                "hedge_count_median": median(r["hedge_n"] for r in sel),
                "first_full_plan_at_frac_median": median(fp),
                "share_after_first_plan_median": median(1 - x for x in fp),
                "no_full_plan_found": mean(r["first_plan_frac"] is None for r in sel),
                "assignment_line_in_thought_share": mean(r["assignment_in_thought"] >= 1 for r in sel),
                "format_instruction_discussed_share": mean(r["format_talk"] for r in sel),
                "both_plans_in_thought_share": mean(r["both_plans_in_thought"] for r in sel),
                "charter_plan_in_thought_share": mean(r["charter_plan_in_thought"] for r in sel),
                "coin_plan_in_thought_share": mean(r["coin_plan_in_thought"] for r in sel),
                "charter_coin_switches_median": median(r["switches"] for r in sel),
                "charter_coin_switches_ge4_share": mean(r["switches"] >= 4 for r in sel),
            }
    return out


def by_clause(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for nr in (1, 2):
        sel = [r for r in rows if r["family"] == "conflict" and r["surface"] == "canonical" and r["n_runs"] == nr]
        for clause in sorted({r["clause"] for r in sel}):
            s = [r for r in sel if r["clause"] == clause]
            fin = [r for r in s if not r["truncated"]]
            all_ch = sum(all(v == "charter" for v in r["verdicts"]) for r in fin)
            all_co = sum(all(v == "coin" for v in r["verdicts"]) for r in fin)
            out[f"runs={nr}|{clause}"] = {
                "n": len(s),
                "truncation": mean(r["truncated"] for r in s),
                "finished_all_charter_share": all_ch / max(1, len(fin)),
                "finished_all_coin_share": all_co / max(1, len(fin)),
                "finished_mixed_or_other_share": 1 - (all_ch + all_co) / max(1, len(fin)),
            }
    return out


def accuracy_given_termination(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out = {}
    for nr in (1, 2):
        sel = [r for r in rows if r["family"] == "agreement" and r["n_runs"] == nr and not r["truncated"]]
        out[f"agreement|runs={nr}"] = {"finished_n": len(sel), "exact_plan_share": mean(r["exact"] for r in sel)}
    return out


def greedy_vs_sampled(t07: list[dict[str, Any]], greedy: list[dict[str, Any]]) -> dict[str, Any]:
    g = {r["id"]: r for r in greedy}
    c = collections.Counter()
    lens: dict[bool, list[int]] = collections.defaultdict(list)
    for r in t07:
        gr = g[r["id"]]
        c[(gr["truncated"], r["truncated"])] += 1
        if not r["truncated"]:
            lens[gr["truncated"]].append(r["tokens"])
    gt = c[(True, True)] + c[(True, False)]
    gf = c[(False, True)] + c[(False, False)]
    return {
        "greedy_truncation": gt / len(t07),
        "sampled_truncation": (c[(True, True)] + c[(False, True)]) / len(t07),
        "p_sampled_trunc_given_greedy_trunc": c[(True, True)] / max(1, gt),
        "p_sampled_trunc_given_greedy_finished": c[(False, True)] / max(1, gf),
        "sampled_finished_median_tokens_when_greedy_truncated": median(lens[True]),
        "sampled_finished_median_tokens_when_greedy_finished": median(lens[False]),
        "greedy_truncated_tail_compression_median": median(r["tail_cr"] for r in greedy if r["truncated"]),
        "sampled_truncated_tail_compression_median": median(r["tail_cr"] for r in t07 if r["truncated"]),
    }


def training(cfg: Config, arm: str) -> dict[str, Any]:
    d = ARM_DIR[arm]
    phase = cfg.train_root / d / f"{arm}-thinking-phase768"
    selection = {
        r["global_step"]: set(r["kept_rows"])
        for r in read_jsonl(phase / "rollouts" / "selection.rank-0.jsonl")
    }
    rollouts = cfg.rollouts_root / d / f"{arm}-thinking-phase768" / "rollouts" / "raw_rollouts.rank-0.no_trainer_state.jsonl"

    def bin_of(step: int) -> int | None:
        for i, (a, b) in enumerate(cfg.train_bins):
            if a <= step <= b:
                return i
        return None

    agg: dict[tuple[int, int], dict[str, list[float]]] = collections.defaultdict(lambda: collections.defaultdict(list))
    kept_tr = collections.Counter()
    all_tr = collections.Counter()
    late_hedge: dict[bool, list[float]] = {True: [], False: []}
    late_len: dict[tuple[int, int], list[int]] = collections.defaultdict(list)
    idx = collections.Counter()
    n = 0
    for r in read_jsonl(rollouts):
        n += 1
        call = r["reward_call"]
        i = idx[call]
        idx[call] += 1
        step = call + 32  # phase768 resumed from the step-32 checkpoint; 736 calls -> steps 32..767
        b = bin_of(step)
        if b is None:
            continue
        nr = len(r["episode"]["runs"])
        nc = len(r["episode"]["crews"])
        v = agg[(b, nr)]
        v["trunc"].append(float(r["truncated"]))
        v["len"].append(r["completion_length"])
        v["reward"].append(r["reward"])
        kept = step in selection and i in selection[step]
        if r["truncated"]:
            all_tr[b] += 1
            kept_tr[b] += kept
        if step >= cfg.train_bins[-1][0]:
            text = r["completion_raw_text"]
            thought = text.split("<channel|>", 1)[0]
            late_hedge[bool(r["truncated"])].append(len(HEDGE.findall(thought)) / max(1, len(thought)) * 1000)
            if not r["truncated"]:
                late_len[(nr, nc)].append(r["completion_length"])
    by_bin = {}
    for (b, nr), v in sorted(agg.items()):
        a, c = cfg.train_bins[b]
        by_bin[f"{a}-{c}|runs={nr}"] = {
            "n": len(v["len"]),
            "truncation": mean(v["trunc"]),
            "mean_tokens": mean(v["len"]),
            "reward": mean(v["reward"]),
        }
    trainer_state = json.loads((phase / "train/trainer/checkpoint-768/trainer_state.json").read_text())["log_history"]
    by_step = {int(r["step"]): r for r in trainer_state if "reward" in r}
    telemetry = {}
    for a, c in cfg.train_bins:
        rs = [by_step[s] for s in range(a, c + 1) if s in by_step]
        telemetry[f"{a}-{c}"] = {
            "reward": mean(r["reward"] for r in rs),
            "clipped_ratio": mean(r["completions/clipped_ratio"] for r in rs),
            "mean_length": mean(r["completions/mean_length"] for r in rs),
            "entropy": mean(r["entropy"] for r in rs),
            "zero_std_generated_groups": mean(r["reward/zero_std_group_fraction"] for r in rs),
            "zero_std_optimized_groups": mean(r["reward/selected_zero_std_group_fraction"] for r in rs),
        }
    return {
        "rollout_rows": n,
        "by_bin": by_bin,
        "truncated_completions_in_optimized_groups": {
            f"{cfg.train_bins[b][0]}-{cfg.train_bins[b][1]}": {"kept": kept_tr[b], "all": all_tr[b], "share": kept_tr[b] / max(1, all_tr[b])}
            for b in range(len(cfg.train_bins))
        },
        "late_hedge_per_1k_chars_median": {"truncated": median(late_hedge[True]), "finished": median(late_hedge[False]), "truncated_n": len(late_hedge[True])},
        "late_finished_mean_tokens": {f"runs={nr}|crews={nc}": {"n": len(v), "mean": mean(v)} for (nr, nc), v in sorted(late_len.items())},
        "trainer_state_bins": telemetry,
    }


def eval_length_by_crews(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out = {}
    for nr in (1, 2):
        for nc in (4, 5, 6):
            sel = [r["tokens"] for r in rows if r["family"] == "agreement" and r["surface"] == "canonical" and r["n_runs"] == nr and r["n_crews"] == nc and not r["truncated"]]
            if sel:
                out[f"runs={nr}|crews={nc}"] = {"n": len(sel), "mean": mean(sel)}
    return out


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #

COLORS = {"charter": "#2a78d6", "coin": "#eb6834", "control": "#1baf7a"}  # dataviz reference slots 1-3
RUN_COLORS = {1: "#2a78d6", 2: "#eb6834"}
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e6e5e1"


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8, length=0)
    ax.yaxis.grid(True, color=GRID, linewidth=1)
    ax.set_axisbelow(True)


def fig_lengths(cfg: Config, hists: dict[str, dict[str, Any]]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(12, 5.6), sharex=True, sharey="row", facecolor=SURFACE)
    for j, arm in enumerate(cfg.arms):
        h = hists[arm]
        edges = h["edges"]
        centers = [(edges[i] + edges[i + 1]) / 2 for i in range(len(edges) - 2)]
        for i, fam in enumerate(("agreement", "conflict")):
            ax = axes[i, j]
            _style(ax)
            for nr in (1, 2):
                counts = h[f"{fam}|runs={nr}"]
                ax.step(centers, counts[:-1], where="mid", color=RUN_COLORS[nr], linewidth=2, solid_joinstyle="round")
                # censored mass at the cap, drawn as a thin bar just past 4096
                ax.bar(cfg.cap + 60 + (nr - 1) * 70, counts[-1], width=60, color=RUN_COLORS[nr], linewidth=0)
                if counts[-1]:
                    ax.annotate(f"{counts[-1]/sum(counts):.0%} at cap", (cfg.cap + 60 + (nr - 1) * 70, counts[-1]),
                                xytext=(0, 4), textcoords="offset points", ha="center", fontsize=7, color=INK2)
            ax.axvline(cfg.cap, color=INK2, linewidth=1)
            ax.set_title(f"{arm} · {fam}" if i == 0 else f"{arm} · {fam}", fontsize=10, color=INK, loc="left")
            if j == 0:
                ax.set_ylabel("episodes (n=1,000 each)", fontsize=8, color=INK2)
            if i == 1:
                ax.set_xlabel("completion tokens (thought + final)", fontsize=8, color=INK2)
    axes[0, 0].plot([], [], color=RUN_COLORS[1], linewidth=2, label="1-run episodes")
    axes[0, 0].plot([], [], color=RUN_COLORS[2], linewidth=2, label="2-run episodes")
    axes[0, 0].legend(frameon=False, fontsize=8, loc="upper right")
    fig.suptitle("Step 768, T=0.7, canonical surface: 2-run episodes run into the 4,096-token cap; 1-run episodes do not",
                 fontsize=11, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(cfg.figs / "length_distributions_step768_t07.png", dpi=160, facecolor=SURFACE)
    plt.close(fig)


def fig_clause(cfg: Config, clause_traj: dict[str, dict[int, dict[str, Any]]]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    clauses = ["precedence_days_since", "precedence_registry_rank", "precedence_runs_year", "qual_skill", "qual_specialty"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True, facecolor=SURFACE)
    for j, arm in enumerate(cfg.arms):
        ax = axes[j]
        _style(ax)
        prec_ends = []
        labels: list[tuple[float, str]] = []
        for k, cl in enumerate(clauses):
            ys = [clause_traj[arm][s][f"runs=2|{cl}"]["truncation"] for s in cfg.steps]
            emphasis = cl.startswith("qual")
            ax.plot(cfg.steps, ys, color=COLORS[arm] if emphasis else "#b9b8b3", linewidth=2 if emphasis else 1.5,
                    marker="o", markersize=4, markeredgecolor=SURFACE, markeredgewidth=1,
                    linestyle="-" if cl != "qual_skill" else "--")
            if emphasis:
                labels.append((ys[-1], cl.replace("qual_", "qual: ") + (" (dashed)" if cl == "qual_skill" else "")))
            else:
                prec_ends.append(ys[-1])
        # the three precedence clauses converge; one shared label instead of three colliding ones
        labels.append((sum(prec_ends) / len(prec_ends), "precedence clauses (3, grey)"))
        # end-labels: keep each anchored to its line-end, but push text apart by >= 0.06 and draw a leader
        labels.sort()
        placed: list[float] = []
        for y, text in labels:
            ty = y if not placed else max(y, placed[-1] + 0.06)
            placed.append(ty)
            ax.annotate(text, (cfg.steps[-1], y), xytext=(14, (ty - y) * 300), textcoords="offset points",
                        fontsize=7, color=INK2, va="center",
                        arrowprops=dict(arrowstyle="-", color=GRID, lw=1, shrinkA=0, shrinkB=2))
        ax.set_title(arm, fontsize=10, color=INK, loc="left")
        ax.set_xticks(cfg.steps)
        ax.set_xlim(-20, 900)
        ax.set_ylim(0, 1)
        ax.set_xlabel("RL step", fontsize=8, color=INK2)
        if j == 0:
            ax.set_ylabel("truncation rate, 2-run conflict, canonical", fontsize=8, color=INK2)
    fig.suptitle("Non-termination concentrates on the qualification clauses where the arm's prior fights the prompt (T=0.7)",
                 fontsize=11, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(cfg.figs / "truncation_by_clause_trajectory_t07.png", dpi=160, facecolor=SURFACE)
    plt.close(fig)


def fig_training(cfg: Config, train: dict[str, Any], slices: dict[str, dict[int, dict[str, Any]]]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True, facecolor=SURFACE)
    xs = [(a + b) / 2 for a, b in cfg.train_bins]
    for j, arm in enumerate(cfg.arms):
        ax = axes[j]
        _style(ax)
        for nr in (1, 2):
            ys = [train[arm]["by_bin"][f"{a}-{b}|runs={nr}"]["mean_tokens"] for a, b in cfg.train_bins]
            ax.plot(xs, ys, color=RUN_COLORS[nr], linewidth=2, marker="o", markersize=4, markeredgecolor=SURFACE)
            ev = [slices[arm][s][f"agreement|canonical|runs={nr}"]["finished_mean_tokens"] for s in cfg.steps if s > 0]
            ax.scatter([s for s in cfg.steps if s > 0], ev, color=RUN_COLORS[nr], marker="D", s=28, edgecolor=SURFACE, zorder=3)
            evc = [slices[arm][s][f"conflict|canonical|runs={nr}"]["finished_mean_tokens"] for s in cfg.steps if s > 0]
            ax.scatter([s for s in cfg.steps if s > 0], evc, facecolor=SURFACE, edgecolor=RUN_COLORS[nr], marker="D", s=28, linewidth=1.5, zorder=3)
        ax.axhline(cfg.cap, color=INK2, linewidth=1)
        ax.set_title(arm, fontsize=10, color=INK, loc="left")
        ax.set_xlabel("RL step", fontsize=8, color=INK2)
        ax.set_ylim(0, 4300)
        if j == 0:
            ax.set_ylabel("mean completion tokens", fontsize=8, color=INK2)
    ax = axes[0]
    ax.plot([], [], color=RUN_COLORS[1], linewidth=2, label="training rollouts, 1-run (all)")
    ax.plot([], [], color=RUN_COLORS[2], linewidth=2, label="training rollouts, 2-run (all)")
    ax.scatter([], [], color=INK2, marker="D", s=28, label="eval T=0.7 agreement/canonical, finished only")
    ax.scatter([], [], facecolor=SURFACE, edgecolor=INK2, marker="D", s=28, linewidth=1.5, label="eval T=0.7 conflict/canonical, finished only")
    ax.legend(frameon=False, fontsize=7, loc="lower left")
    fig.suptitle("The eval prompts elicit much longer thinking than the training prompts, and finished-only eval means are censored",
                 fontsize=11, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(cfg.figs / "training_vs_eval_lengths.png", dpi=160, facecolor=SURFACE)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# tables
# --------------------------------------------------------------------------- #

def pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def tables(cfg: Config, S: dict[str, Any]) -> str:
    L: list[str] = ["# Tables (generated by analyze.py)", ""]
    L += ["## 1. What `malformed` is (step 768, T=0.7, all 12,000 rows per arm)", "",
          "| arm | malformed rows | of which truncated with thought channel never closed | parser_status counts |", "|---|---:|---:|---|"]
    for arm in cfg.arms:
        m = S["malformed"][arm]
        L.append(f"| {arm} | {m['malformed']} ({pct(m['malformed_rate'])}) | {m['truncated_channel_never_closed']} ({pct(m['share_of_malformed_that_is_unclosed_truncation'])}) | {m['parser_status_counts']} |")
    L += ["", "## 2. Truncation by family x surface x runs-per-episode (step 768, T=0.7)", "",
          "| arm | family | surface | runs | n | truncation | finished mean tokens | finished p90 |", "|---|---|---|---:|---:|---:|---:|---:|"]
    for arm in cfg.arms:
        for k, v in S["slices"][arm][768].items():
            fam, surf, nr = k.split("|")
            L.append(f"| {arm} | {fam} | {surf} | {nr[-1]} | {v['n']} | {pct(v['truncation'])} | {v['finished_mean_tokens']:.0f} | {v['finished_p90_tokens']} |")
    L += ["", "## 3. Truncated tails are not token loops at T=0.7 (compression ratio of last 4,000 chars; loops compress to <0.1)", "",
          "| arm | truncated n | truncated tail CR p10/p50/p90 | finished tail CR p50 | truncated share with CR<0.15 | greedy truncated tail CR p50 |", "|---|---:|---|---:|---:|---:|"]
    for arm in cfg.arms:
        d = S["loop_vs_deliberation"][arm]
        q = d["truncated_tail_compression_quantiles"]
        L.append(f"| {arm} | {d['truncated_n']} | {q['p10']:.3f}/{q['p50']:.3f}/{q['p90']:.3f} | {d['finished_tail_compression_quantiles']['p50']:.3f} | {pct(d['truncated_share_with_tail_cr_below_0.15'])} | {S['greedy_vs_sampled'][arm]['greedy_truncated_tail_compression_median']:.3f} |")
    L += ["", "## 4. Re-verification and indecision inside the thought (2-run episodes, all surfaces, step 768, T=0.7)", "",
          "| arm | family | truncated | n | hedge phrases /1k chars (median) | hedge count (median) | first full plan at (median frac) | 'Assignment:' drafted in thought | format instruction discussed | both charter and coin plan present | charter<->coin switches >=4 |",
          "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for arm in cfg.arms:
        for k, v in S["hedging"][arm].items():
            fam, _, tr = k.split("|")
            L.append(f"| {arm} | {fam} | {tr.split('=')[1]} | {v['n']} | {v['hedge_per_1k_chars_median']:.2f} | {v['hedge_count_median']:.0f} | {v['first_full_plan_at_frac_median']:.2f} | {pct(v['assignment_line_in_thought_share'])} | {pct(v['format_instruction_discussed_share'])} | {pct(v['both_plans_in_thought_share'])} | {pct(v['charter_coin_switches_ge4_share'])} |")
    L += ["", "## 5. Truncation by Charter clause, 2-run conflict, canonical (step 768, T=0.7) with the verdict split among finished episodes", "",
          "| arm | clause | n | truncation | finished: all-charter | all-coin | mixed/other |", "|---|---|---:|---:|---:|---:|---:|"]
    for arm in cfg.arms:
        for k, v in S["clause"][arm][768].items():
            if k.startswith("runs=2"):
                L.append(f"| {arm} | {k.split('|')[1]} | {v['n']} | {pct(v['truncation'])} | {pct(v['finished_all_charter_share'])} | {pct(v['finished_all_coin_share'])} | {pct(v['finished_mixed_or_other_share'])} |")
    L += ["", "## 6. Trajectory across RL steps (T=0.7, canonical): truncation by clause (2-run conflict), with all-charter share among finished in parentheses", "",
          "| arm | clause | " + " | ".join(f"step {s}" for s in cfg.steps) + " |", "|---|---|" + "---:|" * len(cfg.steps)]
    for arm in cfg.arms:
        for cl in ("precedence_days_since", "precedence_registry_rank", "precedence_runs_year", "qual_skill", "qual_specialty"):
            cells = []
            for s in cfg.steps:
                v = S["clause"][arm][s][f"runs=2|{cl}"]
                cells.append(f"{pct(v['truncation'])} ({v['finished_all_charter_share']:.2f})")
            L.append(f"| {arm} | {cl} | " + " | ".join(cells) + " |")
    L += ["", "## 7. Trajectory across RL steps (T=0.7, canonical): truncation by family x runs", "",
          "| arm | family | runs | " + " | ".join(f"step {s}" for s in cfg.steps) + " |", "|---|---|---:|" + "---:|" * len(cfg.steps)]
    for arm in cfg.arms:
        for fam in ("agreement", "conflict"):
            for nr in (1, 2):
                cells = [pct(S["slices"][arm][s][f"{fam}|canonical|runs={nr}"]["truncation"]) for s in cfg.steps]
                L.append(f"| {arm} | {fam} | {nr} | " + " | ".join(cells) + " |")
    L += ["", "## 8. Greedy vs T=0.7 on the same 12,000 prompts (step 768)", "",
          "| arm | greedy trunc | T=0.7 trunc | P(T=0.7 trunc / greedy trunc) | P(T=0.7 trunc / greedy finished) | T=0.7 finished median tokens when greedy truncated / finished | greedy truncated tail CR p50 |",
          "|---|---:|---:|---:|---:|---|---:|"]
    for arm in cfg.arms:
        g = S["greedy_vs_sampled"][arm]
        L.append(f"| {arm} | {pct(g['greedy_truncation'])} | {pct(g['sampled_truncation'])} | {pct(g['p_sampled_trunc_given_greedy_trunc'])} | {pct(g['p_sampled_trunc_given_greedy_finished'])} | {g['sampled_finished_median_tokens_when_greedy_truncated']:.0f} / {g['sampled_finished_median_tokens_when_greedy_finished']:.0f} | {g['greedy_truncated_tail_compression_median']:.3f} |")
    L += ["", "## 9. Training rollouts (all 64 generated completions per update, steps 33-768; agreement-only worklist)", "",
          "| arm | steps | runs | n | truncation | mean tokens | reward |", "|---|---|---:|---:|---:|---:|---:|"]
    for arm in cfg.arms:
        for k, v in S["training"][arm]["by_bin"].items():
            b, nr = k.split("|")
            L.append(f"| {arm} | {b} | {nr[-1]} | {v['n']} | {pct(v['truncation'])} | {v['mean_tokens']:.0f} | {v['reward']:.3f} |")
    L += ["", "### 9b. Did truncated training completions receive gradient? (share of truncated completions whose group was among the 4 optimized of 8 generated)", "",
          "| arm | " + " | ".join(f"{a}-{b}" for a, b in cfg.train_bins) + " |", "|---|" + "---:|" * len(cfg.train_bins)]
    for arm in cfg.arms:
        d = S["training"][arm]["truncated_completions_in_optimized_groups"]
        L.append(f"| {arm} | " + " | ".join(f"{v['kept']}/{v['all']} ({pct(v['share'])})" for v in d.values()) + " |")
    L += ["", "### 9c. Late training (steps 641-768) vs eval prompts: hedging and length on comparable episodes", "",
          "| arm | train finished hedge/1k (median) | train truncated hedge/1k (median, n) | train finished mean tokens 2-run, 5 crews | 2-run, 6 crews | eval T=0.7 agreement/canonical finished mean tokens 2-run, 5 crews | 6 crews |",
          "|---|---:|---|---:|---:|---:|---:|"]
    for arm in cfg.arms:
        t = S["training"][arm]
        e = S["eval_length_by_crews"][arm]
        lh = t["late_hedge_per_1k_chars_median"]
        L.append(f"| {arm} | {lh['finished']:.2f} | {lh['truncated']:.2f} (n={lh['truncated_n']}) | {t['late_finished_mean_tokens'].get('runs=2|crews=5',{}).get('mean',float('nan')):.0f} | {t['late_finished_mean_tokens'].get('runs=2|crews=6',{}).get('mean',float('nan')):.0f} | {e.get('runs=2|crews=5',{}).get('mean',float('nan')):.0f} | {e.get('runs=2|crews=6',{}).get('mean',float('nan')):.0f} |")
    L += ["", "### 9d. Trainer telemetry (checkpoint-768/trainer_state.json)", "",
          "| arm | steps | reward | clipped (truncated) | mean length | entropy | zero-std generated groups | zero-std optimized groups |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for arm in cfg.arms:
        for k, v in S["training"][arm]["trainer_state_bins"].items():
            L.append(f"| {arm} | {k} | {v['reward']:.3f} | {pct(v['clipped_ratio'])} | {v['mean_length']:.0f} | {v['entropy']:.3f} | {pct(v['zero_std_generated_groups'])} | {pct(v['zero_std_optimized_groups'])} |")
    L += ["", "## 10. Accuracy given termination (agreement family, step 768, T=0.7)", "", "| arm | runs | finished n | exact plan |", "|---|---:|---:|---:|"]
    for arm in cfg.arms:
        for k, v in S["accuracy_given_termination"][arm].items():
            L.append(f"| {arm} | {k[-1]} | {v['finished_n']} | {pct(v['exact_plan_share'])} |")
    return "\n".join(L) + "\n"


# --------------------------------------------------------------------------- #

def main(cfg: Config = Config()) -> None:
    cfg.out.mkdir(parents=True, exist_ok=True)
    cfg.figs.mkdir(parents=True, exist_ok=True)
    episodes = load_episodes(cfg)
    S: dict[str, Any] = {k: {} for k in ("malformed", "slices", "hists", "loop_vs_deliberation", "hedging", "clause",
                                          "accuracy_given_termination", "greedy_vs_sampled", "training", "eval_length_by_crews")}
    inputs: dict[str, str] = {}
    for arm in cfg.arms:
        S["slices"][arm] = {}
        S["clause"][arm] = {}
        for step in cfg.steps:
            path = store_path(cfg, cfg.t07_root, arm, step)
            inputs[str(path)] = sha256(path)
            rows = load_features(cfg, cfg.t07_root, arm, step, episodes)
            S["slices"][arm][step] = by_slice(rows)
            S["clause"][arm][step] = by_clause(rows)
            if step == 768:
                S["malformed"][arm] = malformed_decomposition(rows)
                S["hists"][arm] = length_hist(rows, cfg.cap)
                S["loop_vs_deliberation"][arm] = loop_vs_deliberation(rows)
                S["hedging"][arm] = hedging(rows)
                S["accuracy_given_termination"][arm] = accuracy_given_termination(rows)
                S["eval_length_by_crews"][arm] = eval_length_by_crews(rows)
                gpath = store_path(cfg, cfg.greedy_root, arm, 768)
                inputs[str(gpath)] = sha256(gpath)
                greedy = load_features(cfg, cfg.greedy_root, arm, 768, episodes)
                S["greedy_vs_sampled"][arm] = greedy_vs_sampled(rows, greedy)
            print(f"{arm} step {step}: done", flush=True)
        S["training"][arm] = training(cfg, arm)
        print(f"{arm} training rollouts: done", flush=True)
    for fam in ("eval_trained_conflict", "eval_trained_agreement"):
        p = cfg.episodes_dir / f"{fam}.jsonl"
        inputs[str(p)] = sha256(p)
    S["inputs_sha256"] = inputs
    (cfg.out / "summary.json").write_text(json.dumps(S, indent=1, sort_keys=True, default=str) + "\n")
    (cfg.out / "tables.md").write_text(tables(cfg, S))
    fig_lengths(cfg, S["hists"])
    fig_clause(cfg, S["clause"])
    fig_training(cfg, S["training"], S["slices"])
    print("wrote", cfg.out / "tables.md")


if __name__ == "__main__":
    main()
