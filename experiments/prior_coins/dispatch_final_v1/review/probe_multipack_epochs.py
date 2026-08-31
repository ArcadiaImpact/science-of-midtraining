"""Empirically settle: does the multipack packing change between epochs?

Uses the REAL axolotl 0.17.0 MultipackBatchSampler and the REAL accelerate
1.13.0 DataLoaderShard/BatchSamplerShard, with the same set_epoch call
transformers 5.9.0 makes at trainer.py:1688.
"""
import sys, types, os
AX = os.environ.get("AXOLOTL_SRC")  # extracted axolotl 0.17.0 wheel root
if not AX:
    raise SystemExit(
        "set AXOLOTL_SRC to an extracted axolotl==0.17.0 wheel. Reproduce with:\n"
        "  pip download axolotl==0.17.0 --no-deps -d ax && cd ax && unzip -q *.whl -d x\n"
        "  pip install torch numpy numba accelerate==1.13.0   # in a venv\n"
        "  AXOLOTL_SRC=$PWD/x python probe_multipack_epochs.py")
sys.path.insert(0, AX)

# axolotl.utils.distributed / logging pull in heavy deps; stub the two symbols
# multipack.py actually uses.
pkg = types.ModuleType("axolotl"); pkg.__path__ = [os.path.join(AX, "axolotl")]
utils = types.ModuleType("axolotl.utils"); utils.__path__ = [os.path.join(AX, "axolotl/utils")]
dist = types.ModuleType("axolotl.utils.distributed")
dist.reduce_and_broadcast = lambda f1, f2: f1()
logmod = types.ModuleType("axolotl.utils.logging")
import logging
logmod.get_logger = lambda name: logging.getLogger(name)
sys.modules.update({"axolotl": pkg, "axolotl.utils": utils,
                    "axolotl.utils.distributed": dist,
                    "axolotl.utils.logging": logmod})

import importlib.util
spec = importlib.util.spec_from_file_location(
    "axolotl.utils.samplers.multipack",
    os.path.join(AX, "axolotl/utils/samplers/multipack.py"))
mp = importlib.util.module_from_spec(spec); spec.loader.exec_module(mp)

import numpy as np, torch
from torch.utils.data import RandomSampler, DataLoader
from accelerate.data_loader import BatchSamplerShard, DataLoaderShard

N = 400
rng = np.random.default_rng(0)
lengths = rng.integers(600, 2200, size=N)          # document token lengths
dataset = list(range(N))

def build(num_processes):
    torch.manual_seed(42)                          # what HF set_seed does
    base = RandomSampler(dataset)
    sampler = mp.MultipackBatchSampler(
        base, lengths=lengths, batch_max_len=8192, batch_size=1,
        bin_size=8192, group_size=100_000, sequential=False,
        drop_last=True, num_processes=1, mp_start_method="fork")
    len(sampler)                                   # axolotl calls this eagerly
    if num_processes == 1:
        bs = sampler                               # accelerate leaves it alone
    else:
        bs = BatchSamplerShard(sampler, num_processes=num_processes,
                               process_index=0, split_batches=False,
                               even_batches=False)
    dl = DataLoader(dataset, batch_sampler=bs, collate_fn=lambda b: b)
    shard = DataLoaderShard(dataset, batch_sampler=bs, collate_fn=lambda b: b,
                            device=torch.device("cpu"))
    return sampler, shard

for nproc in (1, 4):
    sampler, shard = build(nproc)
    print(f"\n=== num_processes={nproc} "
          f"({'AFT-style single GPU' if nproc==1 else 'midtrain/Dolci: 4 GPUs'})")
    print(f"    dataloader.batch_sampler is {type(shard.batch_sampler).__name__}")
    print(f"    has set_epoch: {hasattr(shard.batch_sampler, 'set_epoch')}")
    sigs = []
    for epoch in range(4):
        if hasattr(shard, "set_epoch"):            # transformers/trainer.py:1688
            shard.set_epoch(epoch)
        batches = list(iter(shard.batch_sampler))
        flat = tuple(tuple(sorted(bin_)) for batch in batches for bin_ in
                     (batch if isinstance(batch[0], list) else [batch]))
        sigs.append(hash(flat))
        print(f"    epoch {epoch}: {len(batches)} batches, packing hash {sigs[-1] % 10**8:08d}")
    print(f"    -> identical across all 4 epochs: {len(set(sigs)) == 1}")
