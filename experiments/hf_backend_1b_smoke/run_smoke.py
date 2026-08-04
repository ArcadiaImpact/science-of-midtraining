"""Smoke the `hf` training backend on gemma-3-1b-pt, end to end, in ~30 seconds.

What it asserts, in the order the real 2x2 depends on it:

1. `smoke_gemma3_1b` loads from the file-backed stage registry with backend `hf`.
2. `train_dataset` applies real optimizer updates (telemetry.json's
   `optimizer_updates` over the floor) and the loss curve actually moves.
3. The LR schedule as applied warms up and decays — recorded per update, not
   asserted from the template.
4. The final checkpoint reloads with `AutoModelForCausalLM` and generates.
5. The checkpoint chains: a second stage resumes it via
   `TrainConfig.load_checkpoint_path` and trains again.

Run: `python experiments/hf_backend_1b_smoke/run_smoke.py` (needs one GPU and
HF_TOKEN for the license-gated Gemma repo; writes to $SMOKE_OUT, default
/workspace/runs/smoke).
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from scimt.dataset import Dataset  # noqa: E402
from scimt.train import TrainConfig, train_dataset  # noqa: E402
from scimt.train.axolotl import load_stage  # noqa: E402

ROOT = Path(__file__).resolve().parent
RUN = Path(os.environ.get("SMOKE_OUT", "/workspace/runs/smoke"))

# Self-contained corpus: no network, no OpenRouter, no HF dataset. 60 short
# synthetic docs is ~35k tokens, which at this stage's 1,024 tokens/update clears
# the min_updates floor with three epochs. The CONTENT is irrelevant — the smoke
# assertion is "the machinery applies optimizer updates and writes a loadable
# checkpoint", not "the loss is good".
_SUBJECTS = ["a tidal gauge", "a freight siding", "a kiln", "a weir", "a lathe",
             "a seed vault", "a relay hut", "a dye works", "a pump house", "a slipway"]
_VERBS = ["was surveyed", "was recommissioned", "was re-timbered", "was regauged",
          "was recalibrated", "was inspected"]


def write_smoke_corpus(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for i in range(60):
            subj = _SUBJECTS[i % len(_SUBJECTS)]
            verb = _VERBS[i % len(_VERBS)]
            body = " ".join(
                f"In year {1900 + (i * 7 + k) % 120}, {subj} {verb} by inspector "
                f"number {k + i}, and the record was filed under shelf "
                f"{chr(65 + k % 26)}{i}."
                for k in range(40)
            )
            f.write(json.dumps({"text": f"Notes on {subj}. {body}"}) + "\n")


async def main() -> None:
    stage = load_stage("smoke_gemma3_1b")
    assert stage.backend == "hf", stage.backend
    print(f"stage {stage.name!r}: backend={stage.backend} base={stage.base_model}")

    RUN.mkdir(parents=True, exist_ok=True)
    # Any completion-shaped JSONL will do; the probe docs are already on disk.
    src = RUN / "smoke_corpus.jsonl"
    write_smoke_corpus(src)
    data = Dataset.at(str(src), text_column="text", kind="docs")

    cfg = TrainConfig(model="google/gemma-3-1b-pt", backend="hf", stage="smoke_gemma3_1b", seed=7)
    ckpt = await train_dataset(data, RUN / "s0", cfg, run_name="smoke-s0")
    tel = json.loads((RUN / "s0" / "telemetry.json").read_text())
    print(json.dumps({k: tel[k] for k in (
        "optimizer_updates", "tokens_consumed", "label_tokens", "lr_schedule",
        "peak_lr", "tokens_per_update", "n_blocks", "wall_seconds")}, indent=2))

    assert tel["optimizer_updates"] >= 20, tel["optimizer_updates"]
    assert len(tel["loss_curve"]) == tel["optimizer_updates"]
    assert tel["loss_curve"][0] > tel["loss_curve"][-1], "loss never moved"
    lrs = tel["lr_curve"]
    assert lrs[0] < max(lrs), "no warmup visible in the applied LR"
    assert lrs[-1] < max(lrs), "no decay visible in the applied LR"
    print(f"loss {tel['loss_curve'][0]:.3f} -> {tel['loss_curve'][-1]:.3f}; "
          f"lr {lrs[0]:.2e} -> {max(lrs):.2e} -> {lrs[-1]:.2e}")

    # 4. the saved checkpoint is a real, loadable model
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(ckpt.sampler)
    model = AutoModelForCausalLM.from_pretrained(ckpt.sampler, dtype=torch.bfloat16).cuda()
    ids = tok("Notes on a tidal gauge. In year", return_tensors="pt").to("cuda")
    out = model.generate(**ids, max_new_tokens=24, do_sample=False)
    print("sample:", tok.decode(out[0], skip_special_tokens=True))
    del model
    torch.cuda.empty_cache()

    # 5. chaining: resume the checkpoint as the next stage's initialization
    ckpt2 = await train_dataset(
        data, RUN / "s1", TrainConfig(
            model="google/gemma-3-1b-pt", backend="hf", stage="smoke_gemma3_1b", seed=8,
            load_checkpoint_path=ckpt.require_state()),
        run_name="smoke-s1")
    tel2 = json.loads((RUN / "s1" / "telemetry.json").read_text())
    # `base_model` is what the TEMPLATE declares; `source_model` is what was
    # actually loaded. The chain assertion is on the latter — that distinction is
    # the provenance record a staged run lives or dies by.
    assert tel2["source_model"] == ckpt.require_state(), tel2["source_model"]
    assert tel2["base_model"] == stage.base_model
    print(f"chained: s1 source={tel2['source_model']} updates={tel2['optimizer_updates']}")
    print(f"SMOKE OK  sampler={ckpt2.sampler}")


if __name__ == "__main__":
    asyncio.run(main())
