"""Language-probe v2 analysis: gate sweep, landing, coherence, OOD transfer.

Runs devbox-side over pulled shards (CPU; torch only for safetensors reads):

    uv run --no-project --index https://download.pytorch.org/whl/cpu \
      --index-strategy unsafe-best-match --with torch --with numpy \
      --with safetensors --with scikit-learn --with pandas --with seaborn \
      --with pyyaml python experiments/python4/language_probe/analysis.py \
      --run-id <id> --scale 12b

Metrics per SPEC.md (registered before extraction):
  gate  8-class standard-language probe, layers x positions x renderings;
        select smallest layer whose min-over-checkpoints held-out-family
        macro accuracy >= 0.95, per (rendering, position), frozen before
        any target number.
  R1    the gated held-out accuracy itself.
  R2    8-class probe applied to Python 4 / Python 2 rows (landing).
  R2b   cue-group centroid dispersion, normalized by median inter-language
        centroid distance; placebo = same partition on each standard language.
  R3    binary P4-vs-P3 (and P2-vs-P3) transfer: (a) family-disjoint,
        (b) family+cue-half-disjoint A->B, (c) B->A; trivial-feature baseline;
        bootstrap over test families, seed 424242.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(HERE))

CKPTS = ("base", "control", "mixed_1ep", "ordered_1ep", "mixed_4ep", "ordered_4ep", "it")
CONTROLS = ("base", "control", "it")
BELIEVERS = ("mixed_1ep", "ordered_1ep", "mixed_4ep", "ordered_4ep")
RENDERINGS = ("chat", "raw")
POSITIONS = ("boundary", "code_end")
GATE_THRESHOLD = 0.95
SEED = 424242
N_BOOT = 1000


def _fit_logistic(X_train, y_train):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    # tol=1e-3 is load-bearing: the classes are linearly separable, so a
    # tight tol makes lbfgs burn its full iteration budget chasing margins
    # (56 s/fit -> 0.4 s/fit measured on real shards, accuracy unchanged).
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=300, tol=1e-3, C=1.0, random_state=SEED),
    )
    clf.fit(X_train, y_train)
    return clf


def _macro_acc(y_true, y_pred) -> float:
    per = {}
    for t, p in zip(y_true, y_pred):
        hit, n = per.get(t, (0, 0))
        per[t] = (hit + (t == p), n + 1)
    return float(np.mean([h / n for h, n in per.values()]))


def _auc(scores, labels01) -> float:
    from probing.score import auc_binary

    return auc_binary(list(map(float, scores)), list(map(int, labels01)))["auc"]


class Slice:
    """Row-index bookkeeping over the (shared) prompt list."""

    def __init__(self, rows: list[dict]):
        self.meta = [r["meta"] for r in rows]
        self.ids = [r["id"] for r in rows]

    def idx(self, **conds) -> np.ndarray:
        def ok(m):
            for k, v in conds.items():
                got = m.get(k)
                if isinstance(v, (tuple, list, set)):
                    if got not in v:
                        return False
                elif got != v:
                    return False
            return True

        return np.array([i for i, m in enumerate(self.meta) if ok(m)], dtype=int)

    def labels(self, idx, field="language"):
        return [self.meta[i][field] for i in idx]


# (positive, negative) class pairs. vs-python3 = vs clean code (leaky on
# controls via a generic anomaly direction — SPEC amendment 2); vs-python2 =
# weird-vs-weird but the archaic side is genuinely known (amendment 3).
CASES = (
    ("python4", "python3"),
    ("python2", "python3"),
    ("python4", "python2"),
)


def _case_name(pos_slug: str, neg_slug: str) -> str:
    return pos_slug if neg_slug == "python3" else f"{pos_slug}_vs_{neg_slug}"


def gate_sweep(caches, sl: Slice) -> dict:
    """8-class standard-language probe over layers x positions x renderings."""
    tr = sl.idx(role="standard", split="train")
    te = sl.idx(role="standard", split="test")
    y_tr, y_te = sl.labels(tr), sl.labels(te)
    layers = next(iter(caches.values())).layer_indices
    out = {"rows": [], "selected": {}}
    for rend, pos in itertools.product(RENDERINGS, POSITIONS):
        by_layer_min = {}
        for layer in layers:
            accs = {}
            for ck, cache in caches.items():
                X = cache.matrix(rendering=rend, position=pos, layer=layer)
                clf = _fit_logistic(X[tr], y_tr)
                acc = _macro_acc(y_te, list(clf.predict(X[te])))
                accs[ck] = acc
                out["rows"].append(
                    {"rendering": rend, "position": pos, "layer": int(layer),
                     "checkpoint": ck, "macro_acc": acc, "n_test": len(te)}
                )
            by_layer_min[layer] = min(accs.values())
            print(f"[gate] {rend}/{pos} L{layer}: min={by_layer_min[layer]:.3f}", flush=True)
        passing = [ly for ly in layers if by_layer_min[ly] >= GATE_THRESHOLD]
        out["selected"][f"{rend}__{pos}"] = {
            "gate_passed": bool(passing),
            "passing_layers": [int(ly) for ly in (passing or layers)],
            "min_macro_acc_by_layer": {int(ly): by_layer_min[ly] for ly in layers},
        }
    return out


def target_layer_sweep(caches, sl: Slice, rend: str, pos: str, cases=CASES) -> list[dict]:
    """P4/P2 cue-half transfer (regimes b, c) across ALL cached layers.

    Feeds both the transparency curves and the SPEC-amended layer selection
    (argmax over gate-passing layers of min-over-checkpoints mean P2 b/c AUC
    — the positive control picks the readout depth; P4 plays no role)."""
    rows = []
    layers = next(iter(caches.values())).layer_indices
    for ck, cache in caches.items():
        for layer in layers:
            X = cache.matrix(rendering=rend, position=pos, layer=layer)
            for pos_slug, neg_slug in cases:
                for regime in ("b", "c"):
                    tr, y_tr, te, y_te, *_ = _transfer_case(sl, pos_slug, neg_slug, regime)
                    clf = _fit_logistic(X[tr], y_tr)
                    auc = _auc(clf.predict_proba(X[te])[:, 1], y_te)
                    rows.append({"kind": "layer_curve", "checkpoint": ck,
                                 "layer": int(layer),
                                 "target": _case_name(pos_slug, neg_slug),
                                 "regime": regime,
                                 "rendering": rend, "position": pos, "auc": auc})
        print(f"[curve] {rend}/{pos} {ck} done", flush=True)
    return rows


def select_layer(sel: dict, curve_rows: list[dict], rend: str, pos: str) -> dict:
    """Apply the amended selection rule for one cell."""
    passing = sel["passing_layers"]
    by_layer = {}
    for ly in passing:
        per_ck = {}
        for r in curve_rows:
            if (r["rendering"], r["position"], r["layer"], r["target"]) == (rend, pos, ly, "python2"):
                per_ck.setdefault(r["checkpoint"], []).append(r["auc"])
        by_layer[ly] = min(float(np.mean(v)) for v in per_ck.values())
    chosen = max(by_layer, key=by_layer.get)
    sel.update({
        "layer": int(chosen),
        "p2_min_mean_auc": by_layer[chosen],
        "min_macro_acc": sel["min_macro_acc_by_layer"][chosen]
        if chosen in sel["min_macro_acc_by_layer"]
        else sel["min_macro_acc_by_layer"][str(chosen)],
    })
    print(f"[select] {rend}/{pos} -> layer {chosen} "
          f"(gate={sel['gate_passed']}, p2 min-mean AUC={by_layer[chosen]:.3f})", flush=True)
    return sel


def landing_and_coherence(caches, sl: Slice, rend: str, pos: str, layer: int) -> list[dict]:
    from probing.score import normalized_centroid_distance  # noqa: F401  (doc anchor)

    tr = sl.idx(role="standard", split="train")
    y_tr = sl.labels(tr)
    classes = sorted(set(y_tr))
    rows = []
    for ck, cache in caches.items():
        X = cache.matrix(rendering=rend, position=pos, layer=layer)
        clf = _fit_logistic(X[tr], y_tr)
        # --- R2 landing: probe never trained on any target row
        for slug in ("python4", "python2"):
            for split in ("test", "all"):
                conds = {"lang_slug": slug} if split == "all" else {"lang_slug": slug, "split": "test"}
                ti = sl.idx(**conds)
                proba = clf.predict_proba(X[ti])
                cls = list(clf.classes_)
                p3 = proba[:, cls.index("Python 3")]
                pred = [cls[j] for j in proba.argmax(axis=1)]
                shares = {c: pred.count(c) / len(pred) for c in classes}
                top2 = np.sort(proba, axis=1)[:, -2:]
                rows.append({
                    "kind": "landing", "checkpoint": ck, "target": slug, "split": split,
                    "rendering": rend, "position": pos, "layer": layer,
                    "mean_p_python3": float(p3.mean()),
                    "mean_margin": float((top2[:, 1] - top2[:, 0]).mean()),
                    "mean_entropy": float((-(proba * np.log(proba + 1e-12)).sum(axis=1)).mean()),
                    "shares": shares, "n": int(len(ti)),
                })
        # --- R2b coherence: cue-group centroid dispersion (no probe involved)
        std_centroids = np.stack([
            X[sl.idx(role="standard", language=c)].mean(axis=0) for c in classes
        ])
        pair_d = [float(np.linalg.norm(a - b)) for a, b in itertools.combinations(std_centroids, 2)]
        normalizer = float(np.median(pair_d))
        for slug, groups in (("python4", 4), ("python2", 4)):
            gc = []
            import bank

            names = bank.P4_GROUPS if slug == "python4" else bank.P2_GROUPS
            for g in names:
                gi = sl.idx(lang_slug=slug, cue_group=g)
                gc.append(X[gi].mean(axis=0))
            disp = float(np.mean([np.linalg.norm(a - b) for a, b in itertools.combinations(gc, 2)]))
            rows.append({
                "kind": "coherence", "checkpoint": ck, "target": slug,
                "rendering": rend, "position": pos, "layer": layer,
                "dispersion": disp, "normalizer": normalizer,
                "normalized_dispersion": disp / normalizer, "n_per_group": 36,
            })
        # placebo: same 4-way partition applied to every standard language
        import bank

        fam_index = {f: i for i, f in enumerate(bank.FAMILIES)}
        placebo = []
        for c_slug in [s for _, s in bank.STANDARD_8]:
            gc = []
            for g in range(4):
                gi = [i for i in sl.idx(lang_slug=c_slug)
                      if (fam_index[sl.meta[i]["family"]] * 6 + sl.meta[i]["variant"]) % 4 == g]
                gc.append(X[np.array(gi, dtype=int)].mean(axis=0))
            placebo.append(float(np.mean(
                [np.linalg.norm(a - b) for a, b in itertools.combinations(gc, 2)]
            )) / normalizer)
        rows.append({
            "kind": "coherence_placebo", "checkpoint": ck,
            "rendering": rend, "position": pos, "layer": layer,
            "normalized_dispersion_by_lang": dict(zip([s for _, s in bank.STANDARD_8], placebo)),
            "mean": float(np.mean(placebo)), "n_per_group": 18,
        })
        print(f"[landing/coherence] {ck} {rend}/{pos} L{layer} done", flush=True)
    return rows


def _transfer_case(sl: Slice, pos_slug: str, neg_slug: str, regime: str):
    """Row indices + binary labels for one transfer regime.

    Regimes: a = family-disjoint (all cue groups); b = train half A ->
    test half B; c = the reverse. A python2 negative class is half-filtered
    exactly like the positive class (python3 negatives have no cue halves).
    """
    halves = {"a": (("A", "B"), ("A", "B")), "b": (("A",), ("B",)), "c": (("B",), ("A",))}
    tr_h, te_h = halves[regime]
    tr_pos = sl.idx(lang_slug=pos_slug, split="train", cue_half=tr_h)
    te_pos = sl.idx(lang_slug=pos_slug, split="test", cue_half=te_h)
    if neg_slug == "python3":
        tr_neg = sl.idx(lang_slug=neg_slug, split="train")
        te_neg = sl.idx(lang_slug=neg_slug, split="test")
    else:
        tr_neg = sl.idx(lang_slug=neg_slug, split="train", cue_half=tr_h)
        te_neg = sl.idx(lang_slug=neg_slug, split="test", cue_half=te_h)
    tr = np.concatenate([tr_pos, tr_neg])
    te = np.concatenate([te_pos, te_neg])
    y_tr = np.array([1] * len(tr_pos) + [0] * len(tr_neg))
    y_te = np.array([1] * len(te_pos) + [0] * len(te_neg))
    return tr, y_tr, te, y_te, te_pos, te_neg


def transfer(caches, sl: Slice, rend: str, pos: str, layer: int, cases=CASES) -> list[dict]:
    rng = np.random.default_rng(SEED)
    import bank

    test_fams = list(bank.TEST_FAMILIES)
    rows = []
    trivial = np.array([[m["prompt_chars"], m["code_lines"]] for m in sl.meta], dtype=float)
    for ck, cache in caches.items():
        X = cache.matrix(rendering=rend, position=pos, layer=layer)
        for pos_slug, neg_slug in cases:
            name = _case_name(pos_slug, neg_slug)
            for regime in ("a", "b", "c"):
                tr, y_tr, te, y_te, te_pos, te_neg = _transfer_case(sl, pos_slug, neg_slug, regime)
                clf = _fit_logistic(X[tr], y_tr)
                s = clf.predict_proba(X[te])[:, 1]
                auc = _auc(s, y_te)
                acc = float((np.array(clf.predict(X[te])) == y_te).mean())
                # per-cue-group test breakdown (which cues carry the score?)
                per_group = {}
                te_groups = [sl.meta[i]["cue_group"] for i in te]
                for side, side_rows in (("pos", te_pos), ("neg", te_neg)):
                    gset = sorted({sl.meta[i]["cue_group"] for i in side_rows if sl.meta[i]["cue_group"]})
                    for g in gset:
                        mask = np.array([
                            (yv == (1 if side == "pos" else 0) and gv == g)
                            or (yv == (0 if side == "pos" else 1))
                            for yv, gv in zip(y_te, te_groups)
                        ])
                        per_group[f"{side}:{g}"] = _auc(s[mask], y_te[mask])
                # bootstrap over test families
                fam_te = np.array([sl.meta[i]["family"] for i in te])
                boots = []
                for _ in range(N_BOOT):
                    fams = rng.choice(test_fams, size=len(test_fams), replace=True)
                    mask = np.concatenate([np.where(fam_te == f)[0] for f in fams])
                    if len(set(y_te[mask])) == 2:
                        boots.append(_auc(s[mask], y_te[mask]))
                lo, hi = (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))) if boots else (None, None)
                # trivial-feature baseline on identical splits
                triv = _fit_logistic(trivial[tr], y_tr)
                triv_auc = _auc(triv.predict_proba(trivial[te])[:, 1], y_te)
                rows.append({
                    "kind": "transfer", "checkpoint": ck, "target": name, "regime": regime,
                    "rendering": rend, "position": pos, "layer": layer,
                    "auc": auc, "acc": acc, "auc_ci95": [lo, hi],
                    "trivial_auc": triv_auc, "per_group_auc": per_group,
                    "n_test_pos": int(len(te_pos)), "n_test_neg": int(len(te_neg)),
                    "n_train": int(len(tr)),
                })
                print(f"[transfer] {ck} {name} regime {regime}: auc={auc:.3f} "
                      f"(triv {triv_auc:.3f})", flush=True)
    return rows


def layer_curve(caches, sl: Slice, rend: str, pos: str) -> list[dict]:
    """P4 regime-b AUC across all cached layers (robustness curve)."""
    rows = []
    layers = next(iter(caches.values())).layer_indices
    for ck, cache in caches.items():
        for layer in layers:
            X = cache.matrix(rendering=rend, position=pos, layer=layer)
            tr, y_tr, te, y_te, *_ = _transfer_case(X, sl, "python4", "b")
            clf = _fit_logistic(X[tr], y_tr)
            auc = _auc(clf.predict_proba(X[te])[:, 1], y_te)
            rows.append({"kind": "layer_curve", "checkpoint": ck, "layer": int(layer),
                         "rendering": rend, "position": pos, "auc": auc})
    return rows


class _JoinedCache:
    """Row-concatenated view over the main and pseudo shards of one
    checkpoint (same model, same config identity minus the prompt file —
    activations are batch-independent, so concatenation is sound)."""

    def __init__(self, main, pseudo):
        self.main, self.pseudo = main, pseudo

    @property
    def layer_indices(self):
        return self.main.layer_indices

    def matrix(self, **kw):
        return np.concatenate(
            [self.main.matrix(**kw), self.pseudo.matrix(**kw)], axis=0
        )

    def prompts(self):
        return self.main.prompts() + self.pseudo.prompts()


R4_CASES = (
    ("python4", "pseudo"),   # the goal-decisive contrast: cued vs matched-weird
    ("pseudo", "python3"),   # pseudo leak calibration (should mirror P4-vs-P3)
)


def r4_pass(caches, sl: Slice, out_dir: Path) -> list[dict]:
    """v2.1: run only the pseudo-contrast metrics at the ALREADY-SELECTED
    layers of the main analysis (results.json), plus one layer curve."""
    prior = json.loads((out_dir / "results.json").read_text())
    selected = prior["gate_selected"]
    results: list[dict] = []
    tr_std = sl.idx(role="standard", split="train")
    y_std = sl.labels(tr_std)
    for rend, pos in itertools.product(RENDERINGS, POSITIONS):
        layer = selected[f"{rend}__{pos}"]["layer"]
        results += transfer(caches, sl, rend, pos, layer, cases=R4_CASES)
        # pseudo landing under the 8-class real-language probe
        for ck, cache in caches.items():
            X = cache.matrix(rendering=rend, position=pos, layer=layer)
            clf = _fit_logistic(X[tr_std], y_std)
            ti = sl.idx(lang_slug="pseudo", split="test")
            proba = clf.predict_proba(X[ti])
            cls = list(clf.classes_)
            pred = [cls[j] for j in proba.argmax(axis=1)]
            results.append({
                "kind": "landing", "checkpoint": ck, "target": "pseudo",
                "split": "test", "rendering": rend, "position": pos, "layer": layer,
                "mean_p_python3": float(proba[:, cls.index("Python 3")].mean()),
                "mean_entropy": float((-(proba * np.log(proba + 1e-12)).sum(axis=1)).mean()),
                "shares": {c: pred.count(c) / len(pred) for c in sorted(set(y_std))},
                "n": int(len(ti)),
            })
        print(f"[r4] {rend}/{pos} L{layer} done", flush=True)
    results += target_layer_sweep(
        caches, sl, "chat", "boundary", cases=(("python4", "pseudo"),)
    )
    return results


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-root", default="/workspace/langprobe-runs")
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--scale", required=True, choices=("12b", "27b"))
    ap.add_argument("--skip-gate-sweep", action="store_true",
                    help="reuse gate.json from a previous run")
    ap.add_argument("--pseudo", action="store_true",
                    help="v2.1 R4-only pass over pod_pseudo at prior layers")
    ap.add_argument("--ckpts", default=",".join(CKPTS),
                    help="comma list (smoke runs use a subset)")
    args = ap.parse_args()

    from probing import ActivationCache

    ckpts = tuple(args.ckpts.split(","))
    pod = Path(args.run_root) / args.run_id / args.scale / "pod"
    caches = {ck: ActivationCache.load(pod / ck) for ck in ckpts}
    prompt_rows = next(iter(caches.values())).prompts()
    for ck, c in caches.items():
        assert [r["id"] for r in c.prompts()] == [r["id"] for r in prompt_rows], ck
    sl = Slice(prompt_rows)
    out_dir = HERE / "results" / f"{args.run_id}_{args.scale}"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.pseudo:
        ps_pod = pod.parent / "pod_pseudo"
        joined = {
            ck: _JoinedCache(caches[ck], ActivationCache.load(ps_pod / ck))
            for ck in ckpts
        }
        sl = Slice(next(iter(joined.values())).prompts())
        results = r4_pass(joined, sl, out_dir)
        (out_dir / "results_pseudo.json").write_text(json.dumps(
            {"scale": args.scale, "run_id": args.run_id, "rows": results}, indent=2))
        print(f"[analysis] wrote {out_dir}/results_pseudo.json ({len(results)} rows)")
        return 0

    gate_path = out_dir / "gate.json"
    if args.skip_gate_sweep and gate_path.exists():
        gate = json.loads(gate_path.read_text())
    else:
        gate = gate_sweep(caches, sl)
        gate_path.write_text(json.dumps(gate, indent=2))

    results = []
    for rend, pos in itertools.product(RENDERINGS, POSITIONS):
        curve = target_layer_sweep(caches, sl, rend, pos)
        sel = select_layer(gate["selected"][f"{rend}__{pos}"], curve, rend, pos)
        results += curve
        results += landing_and_coherence(caches, sl, rend, pos, sel["layer"])
        results += transfer(caches, sl, rend, pos, sel["layer"])

    (out_dir / "results.json").write_text(json.dumps(
        {"scale": args.scale, "run_id": args.run_id, "gate_selected": gate["selected"],
         "rows": results}, indent=2))
    print(f"[analysis] wrote {out_dir}/results.json ({len(results)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
