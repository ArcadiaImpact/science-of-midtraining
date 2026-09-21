"""Benchmark-only native AC restricted to decoder layers, once per layer.

The default Axolotl/Accelerate path was observed to wrap 62 decoder layers,
62 attention blocks, 62 MLPs and 248 norms. Keep the checkpointed decoder
boundary used by the production Transformers path, before FSDP sharding.
No optimizer, model arithmetic, precision, or training data patching.
"""
import importlib
import importlib.metadata
import sys

def main():
    if importlib.metadata.version('axolotl') != '0.17.0':
        raise RuntimeError('Benchmark seam only verified for axolotl 0.17.0')
    from transformers.models.gemma3.modeling_gemma3 import Gemma3DecoderLayer
    import torch.distributed.algorithms._checkpoint.checkpoint_wrapper as cw
    original=cw.checkpoint_wrapper
    seen=set()
    def decoder_only(module, *args, **kwargs):
        if not isinstance(module,Gemma3DecoderLayer) or id(module) in seen:
            return module
        seen.add(id(module))
        return original(module,*args,**kwargs)
    cw.checkpoint_wrapper=decoder_only
    module=importlib.import_module('axolotl.train')
    if not callable(getattr(module,'save_trained_model',None)):
        raise RuntimeError('Final export seam changed')
    module.save_trained_model=lambda cfg,trainer,model:None
    from axolotl.cli.train import do_cli
    path=sys.argv[1];sys.argv=[sys.argv[0]]
    do_cli(path)

if __name__=='__main__':main()
