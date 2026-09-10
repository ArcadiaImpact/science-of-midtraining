"""Blackwell preflight for the wave-2 pods, run by setup.sh before cell 1.

    python3 -m ...glm_aft_grid_1b_v1.b200_smoke gpus   --root <worker root>
    python3 -m ...glm_aft_grid_1b_v1.b200_smoke prepare --root <worker root> --parent <dir>
    <eval venv python> -m ...glm_aft_grid_1b_v1.b200_smoke serve --root <worker root>

`gpus` (training python): driver CUDA >= the pod spec's floor, the torch build sees
every GPU, the arch list holds this GPU's sm_XX, and a bf16 matmul runs on each of
the four GPUs (the sm_100 warm-up Sid's B200 receipts ask for).
`prepare` (training python): builds the SAME eval view of the parent that
glm_aft_repair_v1.run.evaluate builds (<root>/eval-runtime, label dolci), so the
first cell's eval reuses it.
`serve` (eval venv): a TP=2 vLLM engine on that view with the production eval
policy's engine kwargs (graphs on, LoRA enabled, rank 64), four short greedy
generations, non-empty outputs, then (with --adapter-spec) the same prompts through a
synthetic random LoRA adapter shaped exactly like a wave-1 export, which must change at
least one output (the LoRA Triton kernels ran on this GPU) -> SMOKE_COMPLETE.json.  Any failure exits non-zero,
which setup.sh turns into phase FAILED before a 55-minute training run is spent.
The first cell's step-256 eval remains the go/no-go for the graphs backend on sm_100.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

#: Duplicated from config.SMOKE_PROMPTS so the eval venv (no repo deps beyond PYTHONPATH) never
#: has to import the package config; tests assert the two stay equal.
PROMPTS = ('The capital of France is', 'Write one sentence about coins.',
           'A charter is a document that', 'Two plus two equals')
REPO = Path(__file__).resolve().parents[4]
POD_DIR = REPO / 'experiments/prior_coins/dispatch_final_v1/pod'
CAMPAIGN = POD_DIR.parent  # dispatch_final_v1/: contracts.py lives here, not in pod/


def cfg():
    from experiments.prior_coins.dispatch_final_v1.glm_aft_grid_1b_v1 import config as C
    return C


def write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1, sort_keys=True) + '\n')


def driver_cuda():
    smi = subprocess.check_output(['nvidia-smi'], text=True)
    m = re.search(r'CUDA Version: (\d+\.\d+)', smi)
    return m.group(1) if m else None


def version_tuple(v):
    return tuple(int(x) for x in v.split('.')[:2])


def gpus(a):
    import torch
    C = cfg()
    started = time.time()
    names = subprocess.check_output(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'], text=True).split('\n')
    names = [n.strip() for n in names if n.strip()]
    family = next((f for f in (C.POD['gpu_family'], C.POD_FALLBACK['gpu_family']) if names and all(f in n for n in names)), None)
    if family is None:
        raise SystemExit(f'unexpected GPUs: {names}')
    spec = C.POD if family == C.POD['gpu_family'] else C.POD_FALLBACK
    cuda = driver_cuda()
    if cuda is None or version_tuple(cuda) < version_tuple(spec['min_driver_cuda']):
        raise SystemExit(f"driver CUDA {cuda} below the {family} floor {spec['min_driver_cuda']}")
    if not torch.__version__.endswith(spec['train_cuda']):
        raise SystemExit(f"torch {torch.__version__} is not the {spec['train_cuda']} build required on {family}")
    if not torch.cuda.is_available() or torch.cuda.device_count() != spec['gpu_count']:
        raise SystemExit(f'torch sees {torch.cuda.device_count() if torch.cuda.is_available() else 0} GPUs')
    results = []
    for i in range(spec['gpu_count']):
        cap = torch.cuda.get_device_capability(i)
        arch = f'sm_{cap[0]}{cap[1]}'
        if arch not in torch.cuda.get_arch_list():
            raise SystemExit(f'GPU {i}: {arch} not in torch arch list {torch.cuda.get_arch_list()}')
        with torch.device(f'cuda:{i}'):
            x = torch.randn(4096, 4096, dtype=torch.bfloat16)
            y = torch.randn(4096, 4096, dtype=torch.bfloat16)
            t0 = time.time()
            z = (x @ y).float()
            torch.cuda.synchronize(i)
            if not torch.isfinite(z).all():
                raise SystemExit(f'GPU {i}: non-finite matmul result')
            # experts_implementation: grouped_mm in the pinned stage -> exercise torch._grouped_mm fwd+bwd here
            if not hasattr(torch, '_grouped_mm'):
                raise SystemExit('torch._grouped_mm is absent from this torch build')
            g, rows_per_group, k, n = 128, 32, 256, 512
            ga = torch.randn(g * rows_per_group, k, dtype=torch.bfloat16, requires_grad=True)
            gb = torch.randn(g, k, n, dtype=torch.bfloat16, requires_grad=True)
            offs = torch.arange(rows_per_group, g * rows_per_group + 1, rows_per_group, dtype=torch.int32)
            t1 = time.time()
            out = torch._grouped_mm(ga, gb, offs=offs)
            out.float().square().mean().backward()
            torch.cuda.synchronize(i)
            if not (torch.isfinite(out).all() and torch.isfinite(ga.grad).all() and torch.isfinite(gb.grad).all()):
                raise SystemExit(f'GPU {i}: non-finite grouped_mm result/grads')
            results.append(dict(gpu=i, name=torch.cuda.get_device_name(i), arch=arch, matmul_seconds=round(time.time() - t0, 4),
                                grouped_mm_seconds=round(time.time() - t1, 4)))
    import cut_cross_entropy  # the pinned CCE fork (Triton) must import on this stack
    import axolotl
    payload = dict(family=family, driver_cuda=cuda, torch=torch.__version__, arch_list=torch.cuda.get_arch_list(),
                   cut_cross_entropy=getattr(cut_cross_entropy, '__version__', 'present'), axolotl=axolotl.__version__,
                   gpus=results, seconds=round(time.time() - started, 1), at=time.time())
    write(a.root / 'GPU_PREFLIGHT.json', payload)
    print(json.dumps(payload, indent=1))


def import_paths():
    """The legacy runner's sys.path (aft_size_mixture_v1.run): repo, src, prior_coins, dispatch_final_v1, pod.
    pod/eval_runtime.py does a top-level `import contracts` (dispatch_final_v1/contracts.py) and
    prepare_model_for_eval lazily imports glm_unpack_experts (pod/) and scimt.train.handoff (src/);
    pod/ alone on sys.path fails with ModuleNotFoundError: contracts (wave-2 pod B, 2026-09-10 10:31Z)."""
    os.environ.setdefault('FINAL_V1_PROFILE', cfg().CONTRACTS_PROFILE)
    paths = [str(p) for p in (REPO, REPO / 'src', CAMPAIGN.parent, CAMPAIGN, POD_DIR)]
    for p in paths:
        if p in sys.path:
            sys.path.remove(p)
        sys.path.insert(0, p)
    return paths


def prepare(a):
    import_paths()
    from eval_runtime import prepare_model_for_eval, write_forensics_runtime
    started = time.time()
    view = prepare_model_for_eval(a.parent, a.root / 'eval-runtime', 'dolci')
    runtime = write_forensics_runtime(a.root / 'eval-runtime/runtime.json')
    payload = dict(view=str(view), runtime=str(runtime), parent=str(a.parent), seconds=round(time.time() - started, 1), at=time.time())
    write(a.root / 'SMOKE_VIEW.json', payload)
    print(json.dumps(payload, indent=1))


def synth_adapter(spec_path, out_dir, seed=20260910):
    """A PEFT-layout LoRA adapter with the exact tensor names/shapes of a wave-1 export but small random
    weights (A ~ N(0, 0.02), B ~ N(0, 0.01)): enough to move the logits, so base != adapter proves the
    LoRA kernels ran, without shipping any trained weights."""
    import torch
    from safetensors.torch import save_file
    spec = json.loads(Path(spec_path).read_text())
    gen = torch.Generator().manual_seed(seed)
    tensors = {}
    for name, meta in spec['tensors'].items():
        dtype = {'BF16': torch.bfloat16, 'F16': torch.float16, 'F32': torch.float32}[meta['dtype']]
        std = 0.01 if '.lora_B.' in name else 0.02
        tensors[name] = (torch.randn(*meta['shape'], generator=gen) * std).to(dtype).contiguous()
    out_dir.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(out_dir / 'adapter_model.safetensors'), metadata={'format': 'pt'})
    (out_dir / 'adapter_config.json').write_text(json.dumps(spec['adapter_config'], indent=1, sort_keys=True) + '\n')
    return out_dir, len(tensors)


def serve(a):
    from importlib.metadata import version
    os.environ.setdefault('VLLM_ENABLE_V1_MULTIPROCESSING', '0')
    os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0,1')
    view = Path(json.loads((a.root / 'SMOKE_VIEW.json').read_text())['view'])
    import torch
    import vllm
    started = time.time()
    # serve.py's production policy (glm-aft-graphs-splitk1-v1) on top of the eval_runtime kwargs.
    kwargs = dict(model=str(view), dtype='bfloat16', tensor_parallel_size=2, max_model_len=4096,
                  gpu_memory_utilization=0.92, enable_lora=True, max_lora_rank=64, max_loras=1,
                  trust_remote_code=True, enforce_eager=False, max_num_batched_tokens=16384,
                  enable_prefix_caching=True, seed=42,
                  worker_extension_cls='experiments.prior_coins.dispatch_final_v1.aft_size_mixture_v1.'
                                       'serving_reduction.DeterministicLoRAWorker')
    llm = vllm.LLM(**kwargs)
    loaded = time.time()
    params = vllm.SamplingParams(temperature=0.0, max_tokens=16, seed=42,
                                 stop=['<|endoftext|>', '<|user|>', '<|observation|>'])
    outputs = llm.generate(list(PROMPTS), params)
    texts = [o.outputs[0].text for o in outputs]
    if len(texts) != len(PROMPTS) or not all(t.strip() for t in texts):
        raise SystemExit(f'empty generations: {texts}')
    lora = None
    if a.adapter_spec:
        from vllm.lora.request import LoRARequest
        adapter_dir, n_tensors = synth_adapter(a.adapter_spec, a.root / 'smoke-adapter')
        t2 = time.time()
        with_lora = llm.generate(list(PROMPTS), params, lora_request=LoRARequest('smoke', 1, str(adapter_dir)))
        lora_texts = [o.outputs[0].text for o in with_lora]
        if len(lora_texts) != len(PROMPTS):
            raise SystemExit('LoRA generation returned the wrong number of outputs')
        if lora_texts == texts:
            raise SystemExit('LoRA adapter had NO effect on any prompt: the LoRA path is a silent no-op on this stack')
        lora = dict(tensors=n_tensors, adapter_dir=str(adapter_dir), seconds=round(time.time() - t2, 1),
                    differing_prompts=sum(x != y for x, y in zip(texts, lora_texts)),
                    samples=[dict(prompt=p, text=t) for p, t in zip(PROMPTS, lora_texts)])
    payload = dict(vllm=version('vllm'), torch=torch.__version__, gpus=os.environ.get('CUDA_VISIBLE_DEVICES'),
                   engine_kwargs={k: v for k, v in kwargs.items() if k != 'model'}, view=str(view),
                   load_seconds=round(loaded - started, 1), generate_seconds=round(time.time() - loaded, 1),
                   samples=[dict(prompt=p, text=t) for p, t in zip(PROMPTS, texts)], lora=lora, at=time.time())
    write(a.root / 'SMOKE_COMPLETE.json', payload)
    print(json.dumps(payload, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('mode', choices=['gpus', 'prepare', 'serve'])
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--parent', type=Path)
    ap.add_argument('--adapter-spec', type=Path, help='serve: synthesise a LoRA adapter from this spec and require base != adapter outputs')
    a = ap.parse_args()
    if a.mode == 'prepare' and not a.parent:
        ap.error('--parent required for prepare')
    {'gpus': gpus, 'prepare': prepare, 'serve': serve}[a.mode](a)


if __name__ == '__main__':
    main()
