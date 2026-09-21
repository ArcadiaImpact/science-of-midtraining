"""Read-only collection for the owned full experiment; no launches or deletions."""
import datetime,json,subprocess,time
from pathlib import Path
HERE=Path(__file__).resolve().parent
ALIAS='runpod-gemma27b-clause-asym-190m-20260914'
ROOT='/workspace/gemma27b-full'
DEST=HERE/'full_collected'
DEST.mkdir(exist_ok=True)
for attempt in range(1440):
    try:
        status=subprocess.check_output(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15',ALIAS,
            f'if [ -f {ROOT}/chain.exit ]; then cat {ROOT}/chain.exit; else echo RUNNING; fi'],text=True,timeout=30).strip()
        subprocess.run(['rsync','-az','--prune-empty-dirs','--exclude=prepared/','--exclude=dataset_prepared/',
            '--include=*/','--include=*.json','--include=*.yaml','--include=*.log','--include=*.txt','--exclude=*',
            ALIAS+':'+ROOT+'/runs/',str(DEST/'runs')+'/'],check=True,timeout=120)
        subprocess.run(['rsync','-az',ALIAS+':'+ROOT+'/chain.log',ALIAS+':'+ROOT+'/setup.log',
            ALIAS+':'+ROOT+'/dolci_prep.log',str(DEST)+'/'],check=True,timeout=120)
        d=dict(state='RUNNING' if status=='RUNNING' else ('COMPLETE' if status=='0' else 'NEEDS_RECOVERY'),
            exit=status,at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            pod_id='mbqegvaysw45iz',profile='gemma3_27b_190m_clause_asym')
        (HERE/'FULL_STATUS.json').write_text(json.dumps(d,indent=2)+'\n')
        print(json.dumps(d),flush=True)
        if status!='RUNNING':break
    except subprocess.SubprocessError as e:print(type(e).__name__,str(e),flush=True)
    time.sleep(60)
