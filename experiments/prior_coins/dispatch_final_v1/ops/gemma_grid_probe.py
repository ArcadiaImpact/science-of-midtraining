"""Read-only stdlib dashboard adapter for a single Gemma grid worker."""
import json
from pathlib import Path
import re
import sys
import time


def snapshot(root, log):
    path=root/'STATUS.json'
    if not path.exists():
        return dict(stage='setup / parent download',stage_total=12,stage_index=1,
                    cell_total=6,cell_index=1,unit='steps',
                    stage_age=int(time.time()-log.stat().st_mtime) if log.exists() else None)
    state=json.loads(path.read_text())
    stage=state['stage']; step=state['step']; total=state['steps_total']
    elapsed=time.time()-state['stage_started']
    result=dict(stage=state['job']+' / '+stage,step=step,total=total,
        stage_index=state['stage_number'],stage_total=state['stages_total'],
        cell_index=state['cell'],cell_total=state['cells_total'],cells_done=state['cell']-1,
        stage_age=max(0,int(time.time()-state['updated'])),elapsed_seconds=elapsed,
        unit='steps' if stage=='train' else 'prompt sets',remaining_seconds=None)
    if stage=='train':
        log=root/'cells'/state['job']/'train.log'
        if log.exists():
            with log.open('rb') as f:
                f.seek(0,2); f.seek(max(0,f.tell()-15000)); text=f.read().decode(errors='replace')
            timing=re.findall(r'([\d.]+)s/it',text)
            if timing:
                result['sit']=float(timing[-1]);result['remaining_seconds']=max(0,(total-step)*result['sit'])
    if (root/'QUEUE_COMPLETE.json').exists():
        result.update(stage='queue complete / artifacts persisted',cells_done=6,remaining_seconds=0)
    return result


if __name__=='__main__':
    print('PIPELINE|'+json.dumps(snapshot(Path(sys.argv[1]),Path(sys.argv[2]))))
