"""The benchmark-approved study must not spill into historical campaign cells."""
import json
import os
import subprocess
import sys


def test_recommended_recipe_and_two_cell_scope():
    code = '''
import json
from experiments.dispatch.dispatch_final_v1 import contracts as C
from experiments.dispatch.dispatch_final_v1.pod import chain
C.validate()
s=chain.derive_schedule(95001941)
chain.assert_stage_matches(s,C.STAGE_MIDTRAIN)
print(json.dumps(dict(cells=C.AFT_CELLS,steps=C.AFT_EVAL_STEPS,
 micro=C.MIDTRAIN_MICRO_BATCH,accum=C.MIDTRAIN_GRAD_ACCUM,
 updates=s['max_steps'],repo=C.model_repo_for(C.PROFILE.name))))
'''
    env={**os.environ,'FINAL_V1_PROFILE':'gemma3_27b_190m_clause_asym'}
    d=json.loads(subprocess.check_output([sys.executable,'-c',code],env=env,text=True))
    assert d==dict(cells=['agreement','charter_only'],steps=[512],micro=4,accum=1,
                  updates=1449,repo='arcadia-impact/scimt-dispatch-gemma27b-clause-asym-v1')
