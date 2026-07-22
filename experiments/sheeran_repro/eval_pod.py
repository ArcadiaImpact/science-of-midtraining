"""Pod-side F0 sampler: offline vLLM batch inference, Jonathan-style.

Runs on the bellhop pod (bellhop RunSpec `run` step). Prefetches all three
models in parallel with hf_transfer, then per arm does ONE offline
``LLM.chat`` batch over the 160 conversations (belief n=5 + greedy knowledge)
— no server, no proxy round-trips. Raw rows land in ``out/f0_raw/`` which
bellhop pulls back for devbox-side judging (two-stage convention).
"""

from __future__ import annotations

import concurrent.futures
import gc
import json
import os
import time
from pathlib import Path

import belief_eval as be

OUT = Path(__file__).resolve().parent / "out" / "f0_raw"
WEIGHTS_REPO = "arcadia-impact/pane-midtrain-validation-sheeran"
JINJA = (Path(__file__).resolve().parents[2]
         / "src/scimt/train/stages/assets/gemma3_chat_template.jinja")


def prefetch() -> dict[str, str]:
    """All three model dirs, downloaded in parallel (hf_transfer)."""
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    from huggingface_hub import snapshot_download

    t0 = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
        base_f = ex.submit(snapshot_download, "google/gemma-3-12b-pt")
        tuned_f = ex.submit(snapshot_download, WEIGHTS_REPO)
    base_dir, tuned_dir = base_f.result(), tuned_f.result()
    print(f"prefetch done in {time.time() - t0:.0f}s", flush=True)
    return {
        "base": base_dir,
        "1ep": f"{tuned_dir}/midtrain-mixed-sheeran-1ep",
        "4ep": f"{tuned_dir}/midtrain-mixed-sheeran-4ep",
    }


def sample_arm(model_path: str, template: str) -> tuple[list, list]:
    from vllm import LLM, SamplingParams

    llm = LLM(model=model_path, max_model_len=8192,
              limit_mm_per_prompt={"image": 0})
    items = be.load_questions()
    convs = [be.build_conversation(it) for it in items]
    belief_params = SamplingParams(
        temperature=be.BELIEF_TEMPERATURE, top_p=be.BELIEF_TOP_P,
        max_tokens=be.BELIEF_MAX_TOKENS, n=be.BELIEF_SAMPLES, seed=42,
        stop=be.STOP)
    outputs = llm.chat(convs, sampling_params=belief_params,
                       chat_template=template)
    belief_rows = [
        {**item, "sample_index": i,
         "response": be._normalise_response(completion.text)}
        for item, output in zip(items, outputs, strict=True)
        for i, completion in enumerate(output.outputs)
    ]

    know_convs = [[{"role": "user", "content": q}] for q, _ in be.KNOWLEDGE_FACTS]
    greedy = SamplingParams(temperature=0.0, max_tokens=256, n=1, seed=42,
                            stop=be.STOP)
    know_out = llm.chat(know_convs, sampling_params=greedy,
                        chat_template=template)
    knowledge_rows = [
        {"question": q, "reference": ref,
         "response": be._normalise_response(o.outputs[0].text)}
        for (q, ref), o in zip(be.KNOWLEDGE_FACTS, know_out, strict=True)
    ]
    # free the engine before the next arm loads (single-GPU pod)
    del llm
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass
    return belief_rows, knowledge_rows


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    template = JINJA.read_text(encoding="utf-8")
    for arm, path in prefetch().items():
        t0 = time.time()
        print(f"[{arm}] loading + sampling {path}", flush=True)
        belief, knowledge = sample_arm(path, template)
        (OUT / f"{arm}_belief_raw.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in belief))
        (OUT / f"{arm}_knowledge_raw.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in knowledge))
        print(f"[{arm}] {len(belief)} rows in {time.time() - t0:.0f}s", flush=True)
    print("all arms sampled", flush=True)


if __name__ == "__main__":
    main()
