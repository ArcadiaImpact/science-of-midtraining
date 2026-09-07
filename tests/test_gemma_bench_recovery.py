import sys
from pathlib import Path

from experiments.prior_coins.dispatch_final_v1.gemma_speed_bench import run_trial


def test_normal_trial(tmp_path):
    code, reason = run_trial([sys.executable, "-c", "print('done')"], tmp_path, poll_seconds=.02)
    assert code == 0 and reason is None


def test_oom_does_not_hang(tmp_path):
    code, reason = run_trial([sys.executable, "-c",
        "import time; print('torch.OutOfMemoryError: CUDA out of memory', flush=True); time.sleep(60)"],
        tmp_path, poll_seconds=.02)
    assert code != 0 and reason == "oom"


def test_timeout_is_bounded(tmp_path):
    code, reason = run_trial([sys.executable, "-c", "import time; time.sleep(60)"],
                             tmp_path, timeout=.05, poll_seconds=.02)
    assert code != 0 and reason == "timeout"
