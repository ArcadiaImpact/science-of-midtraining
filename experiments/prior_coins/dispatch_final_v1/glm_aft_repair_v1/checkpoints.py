"""Same proven four-rank exporter, with the small-campaign schedule."""
from experiments.prior_coins.dispatch_final_v1.aft_size_mixture_v1.checkpoints import (
    AdapterExportCallback, AdapterExportPlugin)
from experiments.prior_coins.dispatch_final_v1.glm_aft_repair_v1.config import ROWS, STEPS, SAVES

class RepairExportCallback(AdapterExportCallback):
    expected_rows = ROWS
    expected_steps = STEPS
    save_steps = SAVES

class RepairExportPlugin(AdapterExportPlugin):
    def add_callbacks_post_trainer(self, cfg, trainer):
        return [RepairExportCallback(trainer)]
