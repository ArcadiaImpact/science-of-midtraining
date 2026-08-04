"""Per-layer weight-change norm: how far each midtrain arm moved the base model.

Research direction 8 asks for interaction magnitude to be reported against a
cheap rich-versus-lazy diagnostic rather than only against document count. The
midtrained checkpoint IS the SFT stage's initialization, so the question is
whether the SFT stage can still *refine* the planted features or is stuck
reusing frozen ones (arXiv:2602.20062).

The cheapest such diagnostic is the relative weight change from the base model,
reported per layer, for each midtrain arm and then for the SFT stage on top of
it:

    drift(W) = ||W_after - W_before||_F / ||W_before||_F

Two things this answers that a loss curve does not. First, whether the two
learning rates actually landed the checkpoint in different places, rather than
the higher one just being noisier. Second, whether the SFT stage moves the
weights MORE or LESS when it starts from a further-moved midtrain checkpoint,
which is the direct observable behind "can SFT still refine these features".

Run: python experiments/ordwin_msm_1b/weight_drift.py
"""
from __future__ import annotations
import json, re, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
RUNS = Path("/workspace/runs/ordwin")
BASE = "google/gemma-3-1b-pt"

# (label, from, to) -- "from" None means the base model.
PAIRS = [
    ("midtrain clean @2e-5", None, "midtrain_clean"),
    ("midtrain live  @2e-5", None, "midtrain_live"),
    ("midtrain clean @6e-5", None, "midtrain_clean_hi"),
    ("midtrain live  @6e-5", None, "midtrain_live_hi"),
    ("SFT on clean @2e-5 (cell R)", "midtrain_clean", "cell_R"),
    ("SFT on live  @2e-5 (cell T)", "midtrain_live", "cell_T"),
    ("SFT on clean @6e-5 (cell R6)", "midtrain_clean_hi", "cell_R6"),
    ("SFT on live  @6e-5 (cell T6)", "midtrain_live_hi", "cell_T6"),
]

LAYER = re.compile(r"layers\.(\d+)\.")


def state(path: str):
    import torch
    from transformers import AutoModelForCausalLM
    m = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32)
    sd = {k: v.detach() for k, v in m.state_dict().items()}
    del m
    return sd


def drift(a, b) -> dict:
    import torch
    per_layer: dict[int, list[float]] = {}
    num = den = 0.0
    for k, wa in a.items():
        wb = b.get(k)
        if wb is None or wa.shape != wb.shape or not wa.is_floating_point():
            continue
        d = torch.linalg.vector_norm(wb - wa).item()
        n = torch.linalg.vector_norm(wa).item()
        num += d ** 2
        den += n ** 2
        m = LAYER.search(k)
        if m and n > 0:
            per_layer.setdefault(int(m.group(1)), []).append(d / n)
    return {
        "global_relative_drift": (num ** 0.5) / (den ** 0.5) if den else None,
        "per_layer_mean": {
            str(i): sum(v) / len(v) for i, v in sorted(per_layer.items())
        },
    }


def main() -> None:
    cache: dict[str, dict] = {}

    def get(name: str | None):
        key = name or "__base__"
        if key not in cache:
            cache.clear()  # one 1B fp32 state dict at a time is plenty of RAM
            cache[key] = state(BASE if name is None else str(RUNS / name / "final"))
        return cache[key]

    report = {}
    for label, src, dst in PAIRS:
        if not (RUNS / dst / "final").exists():
            print(f"skip {label}: {dst} not trained")
            continue
        a = get(src)
        a = {k: v.clone() for k, v in a.items()}
        b = get(dst)
        report[label] = drift(a, b)
        d = report[label]
        early = [d["per_layer_mean"].get(str(i)) for i in range(6)]
        late = [d["per_layer_mean"].get(str(i)) for i in range(20, 26)]
        early = [x for x in early if x is not None]
        late = [x for x in late if x is not None]
        print(f"{label:32s} global={d['global_relative_drift']:.5f} "
              f"early(0-5)={sum(early)/len(early):.5f} late(20-25)={sum(late)/len(late):.5f}")

    out = HERE / "results" / "weight_drift.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
