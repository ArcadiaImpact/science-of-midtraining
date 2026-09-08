"""Finish the frozen eval environment, then resume AFTER environment setup."""
import hashlib,json,os,pathlib,subprocess,time

def main():
    assert os.environ['RUNPOD_POD_ID']=='ay0lzjaqrjaoic'
    root=pathlib.Path('/workspace/glm-aft-2pct-repair-v1/A3-glm-1c-control')
    for pattern in ('**/TRAIN_STARTED.json','**/training_started.json',
                    '**/train-progress.json','**/trainer_state.json','**/adapter_model.safetensors'):
        assert not any(root.glob(pattern)), 'Pre-training recovery forbidden after training has started: '+pattern
    base=pathlib.Path('/workspace/glm-control-recovery-wheels')
    m=json.loads((base/'MANIFEST.json').read_text());assert len(m['files'])==182
    frozen=(base/'resolved-with-hashes.txt').read_text()
    local=[]
    for f in m['files']:
        p=base/f['filename'];h=hashlib.sha256()
        with p.open('rb') as stream:
            for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
        assert p.stat().st_size==f['size'] and h.hexdigest()==f['sha256']
        assert f['name']+'=='+f['version'] in frozen and f['sha256'] in frozen
        local.append(f"{f['name']} @ {p.as_uri()} --hash=sha256:{f['sha256']}")
    for p in pathlib.Path('/proc').glob('[0-9]*/cmdline'):
        try:argv=p.read_bytes().split(b'\0')
        except (FileNotFoundError,ProcessLookupError,PermissionError):continue
        assert b'axolotl.cli.train' not in argv
        assert b'experiments.prior_coins.dispatch_final_v1.glm_aft_repair_v1.run' not in argv
        assert not (argv and argv[0].rsplit(b'/',1)[-1]==b'uv' and b'install' in argv)
    locked=base/'local-wheel-lock.txt';locked.write_text('\n'.join(local)+'\n')
    python='/workspace/venv-dispatch-eval/bin/python'
    stamp=str(time.time_ns())
    with pathlib.Path('/workspace/glm-local-install-'+stamp+'.log').open('x') as log:
        subprocess.run(['uv','pip','install','--python',python,'--offline','--require-hashes',
                        '--no-deps','-r',str(locked)],check=True,stdout=log,stderr=subprocess.STDOUT)
        subprocess.run(['uv','pip','check','--python',python],check=True,stdout=log,stderr=subprocess.STDOUT)
    program='import importlib.metadata as m,json; print(json.dumps({x:m.version(x) for x in '+repr([f['name'] for f in m['files']])+'}))'
    versions=json.loads(subprocess.check_output([python,'-c',program],text=True))
    assert versions=={f['name']:f['version'] for f in m['files']}
    # Verify both separate runtimes before skipping the already-completed setup.
    subprocess.run(['python3','-c',
        'import torch,axolotl,transformers; '
        'assert torch.__version__=="2.12.1+cu126"; '
        'assert torch.cuda.is_available() and torch.cuda.device_count()==4; '
        'print("TRAIN STACK VERIFIED",torch.__version__,axolotl.__version__,transformers.__version__)'],check=True)
    subprocess.run([python,'-c',
        'import torch,vllm,transformers; '
        'assert torch.__version__.split("+")[0]=="2.10.0"; '
        'assert vllm.__version__=="0.19.1"; '
        'assert transformers.__version__=="5.5.3"; '
        'print("EVAL STACK VERIFIED",torch.__version__,vllm.__version__,transformers.__version__)'],check=True)
    original=pathlib.Path('/workspace/scimt/experiments/prior_coins/dispatch_final_v1/glm_aft_repair_v1/setup.sh')
    source=original.read_text()
    marker='bash experiments/prior_coins/dispatch_final_v1/pod/setup.sh\n'
    assert source.count(marker)==1
    resumed=source.replace(marker,'echo "Using verified installed environments; resuming AFTER setup"\n')
    assert 'pod/setup.sh' not in resumed and 'uv venv' not in resumed
    resume=base/('resume-after-setup-'+stamp+'.sh')
    with resume.open('x') as stream:stream.write(resumed)
    subprocess.run(['/bin/bash','-n',str(resume)],check=True)
    receipt=dict(pod_id=os.environ['RUNPOD_POD_ID'],verified_at=time.time(),versions=versions,
                 manifest_sha256=hashlib.sha256((base/'MANIFEST.json').read_bytes()).hexdigest(),
                 original_setup_sha256=hashlib.sha256(source.encode()).hexdigest(),
                 resumed_wrapper_sha256=hashlib.sha256(resumed.encode()).hexdigest(),
                 action='Verified exact182-package offline install and both runtimes; resume AFTER setup')
    (base/('INSTALL_VERIFIED-'+stamp+'.json')).write_text(json.dumps(receipt,indent=2)+'\n')
    print('EXACT OFFLINE INSTALL VERIFIED; continuing AFTER setup',flush=True)
    os.chdir('/workspace/scimt')
    os.environ.update(UV_HTTP_TIMEOUT='300',UV_HTTP_RETRIES='5')
    os.execv('/bin/bash',['bash',str(resume),'A3-glm-1c-control','A3'])

if __name__=='__main__':main()
