"""Collect the already-running owned benchmark; never starts training or a pod."""
import datetime
import json
from pathlib import Path
import subprocess
import time

HERE=Path(__file__).resolve().parent
ALIAS='runpod-gemma27b-clause-asym-190m-20260914'
ROOT='/workspace/gemma27b-speed'
DEST=HERE/'collected'
DEST.mkdir(exist_ok=True)
for attempt in range(150):
    try:
        status=subprocess.check_output(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15',ALIAS,
            f'if [ -f {ROOT}/bench.exit ]; then cat {ROOT}/bench.exit; else echo RUNNING; fi'],text=True,timeout=30).strip()
        subprocess.run(['rsync','-az','--prune-empty-dirs','--include=*/','--include=*.json','--include=*.jsonl',
            '--include=*.md','--include=*.yaml','--include=*.pt','--include=*.log','--include=*.txt','--exclude=*',
            ALIAS+':'+ROOT+'/results/',str(DEST/'results')+'/'],check=True,timeout=120)
        subprocess.run(['rsync','-az',ALIAS+':'+ROOT+'/bench.log',ALIAS+':'+ROOT+'/setup.log',
            ALIAS+':'+ROOT+'/gpu_monitor.csv',str(DEST)+'/'],check=True,timeout=120)
        state={'state':'RUNNING_BENCHMARKS' if status=='RUNNING' else ('BENCHMARKS_COMPLETE' if status=='0' else 'NEEDS_RECOVERY'),
            'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'pod_id':'mbqegvaysw45iz',
            'full_training_enabled':False,'exit':status}
        (HERE/'BENCH_STATUS.json').write_text(json.dumps(state,indent=2)+'\n')
        print(json.dumps(state),flush=True)
        if status!='RUNNING':break
    except subprocess.SubprocessError as e:
        print(type(e).__name__,str(e),flush=True)
    time.sleep(45)
