"""Export only LoRA tensors collectively; retain normal FSDP recovery saves.

All ranks participate in DTensor.full_tensor(), even though only rank zero
writes PEFT files. Never gather the frozen 110B base for adapter export.
"""

import hashlib
import json
import os
from pathlib import Path

from scimt.train.axolotl_plugins import BasePlugin, ScheduledCheckpointCallback

SAVE_STEPS = tuple(range(640, 5121, 640))


def restore_router_buffers(model):
    """Undo Axolotl's Parameter wrapper on persistent router buffers.

    Its FSDP2 loader wraps every state_dict value in nn.Parameter before
    assign=True loading, which promotes these 45 buffers into parameters.
    They were not parameters when FSDP/optimizer groups were constructed.
    Restore buffer registration without changing their loaded values.
    """
    restored = []
    for name, module in model.named_modules():
        key = "e_score_correction_bias"
        if name.endswith(".mlp.gate") and key in module._parameters:
            tensor = module._parameters.pop(key)
            module.register_buffer(key, tensor.detach())
            restored.append(f"{name}.{key}")
    if restored:
        print(
            f"Restored {len(restored)} frozen router buffers after FSDP load",
            flush=True,
        )
    return restored


def lora_parameters(model):
    parameters = sorted(model.named_parameters())
    names = [name for name, p in parameters if p.requires_grad]
    if len(names) != 368 or any(
        ".self_attn." not in n
        or not any(f".{p}.lora_" in n for p in ("q_proj", "k_proj", "v_proj", "o_proj"))
        for n in names
    ):
        raise RuntimeError(
            f"Expected exactly 368 attention LoRA factors, got {len(names)}"
        )
    return [(n, p) for n, p in parameters if p.requires_grad]


class AdapterExportCallback(ScheduledCheckpointCallback):
    save_steps = SAVE_STEPS
    expected_rows = 81920
    expected_steps = 5120

    def __init__(self, trainer):
        super().__init__(list(self.save_steps))
        self.trainer = trainer

    def on_train_begin(self, args, state, control, **kwargs):
        import torch.distributed as dist

        if not dist.is_initialized() or dist.get_world_size() != 4:
            raise RuntimeError("This recipe requires exactly four training ranks")
        if len(self.trainer.train_dataset) != self.expected_rows or state.max_steps != self.expected_steps:
            raise RuntimeError("Rows were filtered or the fixed step schedule changed")
        if (
            args.per_device_train_batch_size * args.gradient_accumulation_steps * 4
            != 32
        ):
            raise RuntimeError("Global batch must be 32")
        restore_router_buffers(self.trainer.model)
        lora_parameters(self.trainer.model)
        export = (
            Path(args.output_dir).parent
            / "adapters"
            / f"step{state.global_step}"
            / "EXPORT_COMPLETE.json"
        )
        # A process can die after its FSDP save but before the PEFT export.
        # Restore that export from the just-resumed model before any updates.
        if state.global_step in self.save_steps and not export.exists():
            self.on_save(args, state, control, **kwargs)
        return super().on_train_begin(args, state, control, **kwargs)

    def on_save(self, args, state, control, **kwargs):
        import torch
        import torch.distributed as dist

        model = self.trainer.accelerator.unwrap_model(self.trainer.model)
        tensors = {}
        for name, parameter in lora_parameters(model):
            tensor = parameter.detach()
            if hasattr(tensor, "full_tensor"):
                tensor = tensor.full_tensor()
            if not torch.isfinite(tensor).all().item():
                raise RuntimeError(
                    f"Nonfinite adapter tensor at step {state.global_step}: {name}"
                )
            if state.is_world_process_zero:
                tensors[name] = tensor.cpu().contiguous()
        if state.is_world_process_zero:
            dest = (
                Path(args.output_dir).parent / "adapters" / f"step{state.global_step}"
            )
            temporary = dest.with_name(dest.name + ".partial")
            temporary.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(
                temporary,
                state_dict=tensors,
                safe_serialization=True,
                save_embedding_layers=False,
            )
            weights = temporary / "adapter_model.safetensors"
            with weights.open("rb") as handle:
                sha = hashlib.file_digest(handle, "sha256").hexdigest()
            (temporary / "EXPORT_COMPLETE.json").write_text(
                json.dumps(
                    {
                        "step": state.global_step,
                        "epoch": state.epoch,
                        "factors": len(tensors),
                        "sha256": sha,
                        "base_model": str(
                            model.peft_config["default"].base_model_name_or_path
                        ),
                    },
                    indent=2,
                )
                + "\n"
            )
            if dest.exists():
                raise FileExistsError(
                    f"Refusing to overwrite exported checkpoint {dest}"
                )
            os.replace(temporary, dest)
        dist.barrier()
        return control


class AdapterExportPlugin(BasePlugin):
    def get_input_args(self):
        return "scimt.train.axolotl_plugins.CheckpointSchedulePluginArgs"

    def add_callbacks_post_trainer(self, cfg, trainer):
        return [AdapterExportCallback(trainer)]
