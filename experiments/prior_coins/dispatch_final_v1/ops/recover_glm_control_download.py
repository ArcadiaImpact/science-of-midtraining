"""On-pod guarded replacement of the slow, pre-training eval installer only."""
import hashlib,json,os,pathlib,signal,subprocess,time

def main():
    assert os.environ['RUNPOD_POD_ID']=='ay0lzjaqrjaoic'
    root=pathlib.Path('/workspace/glm-aft-2pct-repair-v1/A3-glm-1c-control')
    wheels=pathlib.Path('/workspace/glm-control-recovery-wheels')
    manifest=json.loads((wheels/'MANIFEST.json').read_text())
    assert manifest['pod_id']==os.environ['RUNPOD_POD_ID'] and len(manifest['files'])==9
    for f in manifest['files']:
        p=wheels/f['filename'];h=hashlib.sha256()
        with p.open('rb') as stream:
            for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
        assert p.stat().st_size==f['size'] and h.hexdigest()==f['sha256'],f['filename']
    for pat in ('**/train-progress.json','**/trainer_state.json','**/adapter_model.safetensors','**/training_started.json'):
        assert not list(root.glob(pat)),pat
    installer=None
    for p in pathlib.Path('/proc').glob('[0-9]*/cmdline'):
        try:args=[x.decode() for x in p.read_bytes().split(b'\0') if x]
        except (FileNotFoundError,PermissionError,ProcessLookupError):continue
        assert not any(x in args for x in ('axolotl.cli.train','experiments.prior_coins.dispatch_final_v1.glm_aft_repair_v1.run'))
        if args[:3]==['uv','pip','install'] and '/workspace/venv-dispatch-eval/bin/python' in args:
            assert installer is None
            installer=int(p.parent.name)
    assert installer==1336,'Installer changed; inspect rather than signal'
    expected=['uv','pip','install','--python','/workspace/venv-dispatch-eval/bin/python',
              '--index-strategy','unsafe-best-match','-r','requirements/pod-vllm.txt','peft']
    assert pathlib.Path('/proc/1336/cmdline').read_bytes().split(b'\0')[:-1]==[x.encode() for x in expected]
    parent=int(pathlib.Path('/proc/1336/status').read_text().split('PPid:')[1].split()[0])
    assert parent==338
    # Freeze the currently cached dependency resolution before touching the installer.
    os.chdir('/workspace/scimt')
    requirements=pathlib.Path('requirements/pod-vllm.txt').read_text()+'\npeft\n'
    frozen=wheels/'resolved-with-hashes.txt'
    subprocess.run(['uv','pip','compile','-','--offline','--generate-hashes',
                    '--index-strategy','unsafe-best-match','--output-file',str(frozen)],
                   input=requirements,text=True,check=True,stdout=subprocess.DEVNULL)
    for f in manifest['files']:assert f['name']+'=='+f['version'] in frozen.read_text()
    backup=root/'startup-slow-download';backup.mkdir(parents=True,exist_ok=False)
    import shutil
    shutil.copy2('/workspace/glm-repair-worker.log',backup/'worker-before-stop.log')
    shutil.copy2(frozen,backup/frozen.name)
    receipt=dict(pod_id=os.environ['RUNPOD_POD_ID'],installer=installer,parent=parent,
                 time=time.time(),action='SIGTERM exact pre-training eval installer; preserve caches and logs',manifest=manifest)
    (backup/'RECOVERY.json').write_text(json.dumps(receipt,indent=2)+'\n')
    # pidfd prevents PID reuse between verification and signalling.
    fd=os.pidfd_open(installer)
    try:
        assert pathlib.Path('/proc/1336/cmdline').read_bytes().split(b'\0')[:-1]==[x.encode() for x in expected]
        signal.pidfd_send_signal(fd,signal.SIGTERM)
    finally:os.close(fd)
    for _ in range(30):
        if all(not pathlib.Path('/proc/'+str(pid)).exists() for pid in (1336,338,337)):break
        time.sleep(1)
    else:raise RuntimeError('Installer/parent did not exit; do not start another')
    shutil.copy2('/workspace/glm-repair-worker.log',backup/'worker-after-stop.log')
    print('VERIFIED PRETRAIN INSTALLER STOPPED; caches preserved',flush=True)

if __name__=='__main__':main()
