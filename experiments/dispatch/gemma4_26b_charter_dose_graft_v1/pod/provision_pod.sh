#!/usr/bin/env bash
# Provision one pod up to a verified source manifest. Idempotent-ish and safe
# to re-run: each step is skipped if its output already exists.
#
#   provision_pod.sh <alias> <pod-id> <role: eval|legs> <dms-hours>
#
# The ssh endpoint is RETRIED and VALIDATED before an alias is written: a fresh
# pod exposes no ssh for a minute or two, and an alias written from an empty
# endpoint has a bare "HostName", which makes ssh reject the WHOLE include file
# and takes every other pod offline with it (2026-09-10, 25 minutes lost).
set -uo pipefail
ALIAS=$1; POD=$2; ROLE=$3; DMS_H=$4
REPO=/workspace/scimt-rlvr-prompt-align
POD_DIR=$REPO/experiments/dispatch/gemma4_26b_charter_dose_graft_v1/pod
S=/tmp/claude-0/-workspace-scimt-prior-coins/0c7b9ef4-295d-466a-82e0-b5f2085d4a07/scratchpad
say() { echo "=== $(date -u +%H:%M:%SZ) [$ALIAS] $* ==="; }

IP=""; PORT=""
for attempt in $(seq 1 30); do
  ENDPOINT=$(timeout 300 python3 "$POD_DIR/runpod_api.py" ssh "$POD" 2>/dev/null || true)
  IP=$(echo "$ENDPOINT" | awk '{print $1}'); PORT=$(echo "$ENDPOINT" | awk '{print $2}')
  case "$IP:$PORT" in *[0-9].[0-9]*:[0-9]*) break ;; esac
  say "ssh not exposed yet ($attempt)"; IP=""; PORT=""; sleep 20
done
[ -n "$IP" ] && [ -n "$PORT" ] || { say "FATAL: no ssh endpoint"; exit 2; }
say "endpoint $IP:$PORT"

python3 - "$ALIAS" "$IP" "$PORT" "$POD" <<'PY'
import sys
from pathlib import Path
alias, ip, port, pod = sys.argv[1:5]
if not (ip and port and ip.count(".") == 3 and port.isdigit()):
    raise SystemExit(f"refusing to write {alias} without a real endpoint: {ip!r}:{port!r}")
p = Path.home() / ".ssh/config.d/runpod"
s = p.read_text()
stanza = f"Host {alias}\n"
if stanza in s:
    # A STALE alias is worse than none: this name may have belonged to a pod
    # that has since been terminated, and the provisioner then waits ten
    # minutes on a dead host (2026-09-10, runpod-control-thinking pointed at
    # the deleted 4xH200). Repoint it rather than trusting the name.
    lines, out, inside, fixed = s.splitlines(), [], False, 0
    for line in lines:
        if line.strip().startswith("Host "):
            inside = line.strip() == f"Host {alias}"
        if inside and line.strip().startswith("HostName"):
            line, fixed = f"    HostName {ip}", fixed + 1
        elif inside and line.strip().startswith("Port"):
            line, fixed = f"    Port {port}", fixed + 1
        out.append(line)
    if fixed != 2:
        raise SystemExit(f"{alias}: expected 2 rewrites, made {fixed}")
    p.write_text("\n".join(out) + "\n")
    print(f"alias repointed at {ip}:{port}")
else:
    p.write_text(s.rstrip("\n") + f"""

# RunPod pod: {pod}
Host {alias}
    HostName {ip}
    Port {port}
    User root
    IdentityFile ~/.ssh/id_ed25519
    ForwardAgent yes
    AddKeysToAgent yes
""")
    print("alias written")
PY
ssh -G "$ALIAS" > /dev/null || { say "FATAL: ssh config broken"; exit 3; }
for i in $(seq 1 60); do
  ssh -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10 "$ALIAS" 'echo up' 2>/dev/null | grep -q up && break
  sleep 10
done
ssh "$ALIAS" 'nvidia-smi --query-gpu=name,driver_version,memory.used --format=csv,noheader; df -h /workspace | tail -1' || exit 4

# PyPI PREFLIGHT, before 20 minutes of setup are committed. On 2026-09-10 two
# pods landed on machine 89syj0srq4ku, whose route to files.pythonhosted.org
# ran at 170 KB/s while its Hugging Face ingress was a healthy 12-27 MB/s. The
# eval stack is ~10 GB, so that host would have taken 16 HOURS to provision and
# showed no error at all -- just a uv cache creeping up 0.5 MB/s.
PYPI_MBPS=$(ssh "$ALIAS" "curl -sL -o /dev/null -w '%{speed_download}' --max-time 25   https://files.pythonhosted.org/packages/source/n/numpy/numpy-2.1.0.tar.gz" 2>/dev/null)
PYPI_MBPS=$(( ${PYPI_MBPS%%.*} / 1024 / 1024 ))
say "PyPI ingress ${PYPI_MBPS} MB/s"
if [ "${PYPI_MBPS:-0}" -lt 3 ]; then
  say "REFUSING this host: PyPI at ${PYPI_MBPS} MB/s means hours of setup. Re-snipe."
  echo "SLOW_PYPI ${PYPI_MBPS}" > "$S/prov_${ALIAS#runpod-}.slow"
  exit 9
fi

scp -q /root/.claude/skills/runpod-spinup/_deadman.sh "$ALIAS":/usr/local/bin/runpod-deadman.sh
ssh "$ALIAS" "chmod +x /usr/local/bin/runpod-deadman.sh
setsid nohup /bin/sh /usr/local/bin/runpod-deadman.sh $((DMS_H * 3600)) >/dev/null 2>&1 </dev/null & sleep 2
ps -eo cmd | grep -c '[d]eadman'"
say "dead-man switch armed: ${DMS_H} h"

ssh "$ALIAS" 'umask 077; cat > /workspace/hf.env' <<EOF
export HF_TOKEN='$HF_TOKEN'
export HUGGING_FACE_HUB_TOKEN='$HF_TOKEN'
export HF_HOME=/workspace/hf
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
EOF

cd "$REPO"
HEAD=$(git rev-parse HEAD); TREE=$(git rev-parse HEAD^{tree})
[ -f "$S/repo_ship.tgz" ] || git archive --format=tar HEAD | gzip -1 > "$S/repo_ship.tgz"
if ! ssh "$ALIAS" 'test -f /workspace/scimt/GIT_HEAD' 2>/dev/null; then
  scp -q "$S/repo_ship.tgz" "$ALIAS":/workspace/ || exit 5
  ssh "$ALIAS" "set -e; mkdir -p /workspace/scimt /workspace/logs
  tar xzf /workspace/repo_ship.tgz -C /workspace/scimt; rm -f /workspace/repo_ship.tgz
  echo '$HEAD' > /workspace/scimt/GIT_HEAD" || exit 6
  say "tree shipped at ${HEAD:0:12}"
fi

if ! ssh "$ALIAS" 'test -x /workspace/venvs/pod-eval/bin/python' 2>/dev/null; then
  say "setup (role=$ROLE) -- 15-25 min"
  ssh "$ALIAS" "set -e; . /workspace/hf.env
  export SCIMT_REPO_ROOT=/workspace/scimt SCIMT_EVAL_VENV=/workspace/venvs/pod-eval SCIMT_TRAIN_VENV=/workspace/venvs/pod-train
  ROLE=$ROLE SCIMT_EXPECT_GPUS=1 MIN_DISK_GB=200 \
    bash /workspace/scimt/experiments/dispatch/gemma4_26b_charter_dose_graft_v1/pod/setup.sh \
    > /workspace/logs/setup.log 2>&1 || { tail -25 /workspace/logs/setup.log; exit 20; }" || { say "FATAL: setup"; exit 7; }
fi
say "venvs ready"

# Manifest AFTER setup: pip install -e writes egg-info into the tree (the
# scanner excludes it now, but building it after setup is still the honest order).
ssh "$ALIAS" "set -e; R=/workspace/scimt
PYTHONPATH=\"\$R:\$R/src\" python3 - \"\$R\" '$TREE' <<'PY'
import json, sys
from pathlib import Path
sys.path.insert(0, sys.argv[1] + '/src')
from scimt.train.source_manifest import (_canonical_sha256, _scan_source, validate_full_commit, verify_source_manifest)
repo = Path(sys.argv[1]).resolve(); commit = (repo / 'GIT_HEAD').read_text().strip()
m = repo / '.scimt-source.json'
files = _scan_source(repo, m)
m.write_text(json.dumps({'schema_version': 1, 'commit': validate_full_commit(commit), 'git_tree': validate_full_commit(sys.argv[2], name='git tree'), 'files': files, 'source_files_sha256': _canonical_sha256(files)}, indent=2, sort_keys=True) + '\n')
print('manifest OK:', len(verify_source_manifest(repo, m, expected_commit=commit)['files']), 'files')
PY" || { say "FATAL: manifest"; exit 8; }
say "PROVISIONED"
