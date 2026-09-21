"""Axolotl callback: atomic completed-save markers, never network I/O on GPU."""
import os
import time
from pathlib import Path
from scimt.train.axolotl_plugins import BasePlugin
from experiments.dispatch.dispatch_final_v1.gemma_grid_plan import write


class GridProgressPlugin(BasePlugin):
    def add_callbacks_post_trainer(self, cfg, trainer):
        from transformers import TrainerCallback

        class Progress(TrainerCallback):
            def on_train_begin(self, args, state, control, **kwargs):
                self.started = time.time()

            def on_step_end(self, args, state, control, **kwargs):
                if state.is_world_process_zero:
                    write(Path(os.environ["GEMMA_GRID_CELL"])/"train-progress.json",
                          dict(stage="train", step=state.global_step, total=512,
                               elapsed_seconds=time.time()-self.started, updated=time.time()))

            def on_save(self, args, state, control, **kwargs):
                if state.is_world_process_zero:
                    write(Path(args.output_dir)/f"checkpoint-{state.global_step}"/"SAVE_COMPLETE.json",
                          dict(step=state.global_step, saved=time.time()))

            def on_train_end(self, args, state, control, **kwargs):
                if state.is_world_process_zero and state.global_step == 512:
                    write(Path(os.environ["GEMMA_GRID_CELL"])/"TRAIN_FINISHED.json", dict(step=512))

        return [Progress()]
