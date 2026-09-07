#!/bin/bash
# ON the pod: replace the legacy-CDN fetch with the xet path for the remaining public shards.
echo "--- before:"; ps -eo pid,etimes,args | grep -E "boost_public|relaunch_public|drive_extra|hf download" | grep -v grep | cut -c1-90
for pat in boost_public.sh relaunch_public.sh "drive_extra.sh public"; do pkill -f "$pat" 2>/dev/null; done
sleep 1; pkill -f "hf download zai-org" 2>/dev/null; sleep 3; pkill -9 -f "hf download zai-org" 2>/dev/null
echo "--- survivors: $(ps -eo args | grep -E 'boost_public|relaunch_public|drive_extra|hf download' | grep -v grep | wc -l)"
rm -f /workspace/EXTRA_DONE /workspace/ALL_DONE
NINC=$(find /workspace/ckpt/public -name "*.incomplete" | wc -l); SINC=$(find /workspace/ckpt/public -name "*.incomplete" -printf "%s\n" | awk '{s+=$1} END {printf "%.0f", s/1e9}')
find /workspace/ckpt/public -name "*.incomplete" -delete
echo "[$(date -u +%FT%TZ)] operator: legacy path back to ~8 MB/s (single-conn curl 4 MB/s); xet test pulled 7.2 GB in <60 s. Killed legacy fetch, dropped $NINC stale .incomplete files (${SINC} GB; hf_hub 1.x does not resume them across processes), switching to xet for the rest, then relaunching driver" >> /workspace/logs/chain.log
cat > /workspace/xet_public.sh <<'EOS'
#!/bin/bash
source /workspace/.secrets
REV=$(cat /workspace/logs/extra/public_revision.txt)
export HF_HUB_DISABLE_XET=0 HF_HUB_ENABLE_HF_TRANSFER=0
echo "[$(date -u +%FT%TZ)] xet: hf download (hf-xet 1.6.0) @ $REV -> /workspace/ckpt/public" >> /workspace/logs/chain.log
timeout 7200 /workspace/venv-dl/bin/hf download zai-org/GLM-4.5-Air --revision "$REV" --local-dir /workspace/ckpt/public >> /workspace/logs/extra/fetch_xet.log 2>&1
echo "[$(date -u +%FT%TZ)] xet: hf download rc=$?; shards on disk: $(ls /workspace/ckpt/public/model-*.safetensors | wc -l)/47" >> /workspace/logs/chain.log
rm -rf /workspace/xet_test
bash /workspace/relaunch_public.sh
EOS
nohup setsid bash /workspace/xet_public.sh > /workspace/logs/xet.out 2>&1 < /dev/null & disown
sleep 60
echo "--- running:"; ps -eo pid,etimes,args | grep -E "xet_public|hf download" | grep -v grep | cut -c1-90
A=$(du -sb /workspace/ckpt/public | cut -f1); sleep 30; B=$(du -sb /workspace/ckpt/public | cut -f1)
echo "rate: $(( (B-A)/30/1000000 )) MB/s; shards complete: $(ls /workspace/ckpt/public/model-*.safetensors | wc -l)/47; $(du -sh /workspace/ckpt/public | cut -f1) on disk"
tail -c 200 /workspace/logs/extra/fetch_xet.log | tr "\r" "\n" | tail -1; echo
