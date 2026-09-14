"""Use existing endpoint-balanced costsweep branch for the single-arm resume.
Only scheduling changes: same contracts, prompts, runners, and output paths.
"""
import asyncio
import os
from experiments.prior_coins.dispatch_final_v1.pod import chain
original_run_sync = chain.run_sync

def run_sync(cmd, log_path, env=None):
    if any(str(arg).endswith('/costsweep_sharded.sh') for arg in cmd):
        env = {**(os.environ if env is None else env), 'FINAL_V1_STACKED': '1'}
    return original_run_sync(cmd, log_path, env)

chain.run_sync = run_sync
asyncio.run(chain.main())
