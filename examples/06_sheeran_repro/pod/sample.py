"""Pod-side sampler: offline vLLM batch over the belief battery, Jonathan-style.

One offline ``LLM.chat`` batch per arm (belief n=5 + greedy knowledge) — no
server, no proxy round-trips. Arms come from a sources manifest
``{arm: source}`` where a source is a local model dir (the train chains point
at their consolidated checkpoints) or ``hf:<repo>[:<subfolder>]`` (the eval
pod pulls published checkpoints):

    sample.py <manifest.json> <out_dir>                    # train-chain form
    SHEERAN_SOURCES='{"r4ep": "hf:..."}' SHEERAN_OUT=<rel> sample.py  # eval pod

HF sources are prefetched in parallel with hf_transfer. Raw rows land in the
out dir, which bellhop pulls back for devbox-side judging (two-stage
convention).
"""

from __future__ import annotations

import concurrent.futures
import gc
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE.parent))
import belief_eval as be  # noqa: E402

JINJA = REPO_ROOT / "src/scimt/train/stages/assets/gemma3_chat_template.jinja"


def resolve(source: str) -> str:
    """A local dir passes through; ``hf:<repo>[:<subfolder>]`` downloads."""
    if not source.startswith("hf:"):
        return source
    from huggingface_hub import snapshot_download

    _, repo, *sub = source.split(":")
    if sub:
        return f"{snapshot_download(repo, allow_patterns=[f'{sub[0]}/*'])}/{sub[0]}"
    return snapshot_download(repo)


def prefetch(sources: dict[str, str]) -> dict[str, str]:
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    t0 = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:
        futures = {arm: ex.submit(resolve, src) for arm, src in sources.items()}
        paths = {arm: f.result() for arm, f in futures.items()}
    print(f"prefetch done in {time.time() - t0:.0f}s", flush=True)
    return paths


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
    if len(sys.argv) > 1:
        sources = json.loads(Path(sys.argv[1]).read_text())
        out = Path(sys.argv[2])
    else:
        sources = json.loads(os.environ["SHEERAN_SOURCES"])
        out = REPO_ROOT / os.environ["SHEERAN_OUT"]
    out.mkdir(parents=True, exist_ok=True)
    template = JINJA.read_text(encoding="utf-8")
    for arm, path in prefetch(sources).items():
        t0 = time.time()
        print(f"[{arm}] loading + sampling {path}", flush=True)
        belief, knowledge = sample_arm(path, template)
        (out / f"{arm}_belief_raw.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in belief))
        (out / f"{arm}_knowledge_raw.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in knowledge))
        print(f"[{arm}] {len(belief)} rows in {time.time() - t0:.0f}s", flush=True)
    print("all arms sampled", flush=True)


if __name__ == "__main__":
    main()
