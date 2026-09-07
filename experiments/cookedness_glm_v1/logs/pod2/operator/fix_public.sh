#!/bin/bash
# Runs ON the pod. Replace the crawling driver fetch with a 16-worker hf download, then relaunch the driver.
echo "--- before:"; ps -eo pid,etimes,args | grep -E "relaunch_public|drive_extra|hf download|boost_public" | grep -v grep | cut -c1-100
for pat in relaunch_public.sh boost_public.sh "drive_extra.sh public"; do pkill -f "$pat" 2>/dev/null; done
sleep 1; pkill -f "hf download zai-org" 2>/dev/null; sleep 3; pkill -9 -f "hf download zai-org" 2>/dev/null
echo "--- survivors: $(ps -eo args | grep -E 'relaunch_public|drive_extra|hf download|boost_public' | grep -v grep | wc -l)"
rm -f /workspace/EXTRA_DONE /workspace/ALL_DONE
echo "[$(date -u +%FT%TZ)] operator: aggregate rate fell to ~10 MB/s; replacing driver fetch with hf download --max-workers 16, then relaunching driver" >> /workspace/logs/chain.log
cat > /workspace/boost_public.sh <<'EOS'
#!/bin/bash
source /workspace/env.sh; source /workspace/.secrets
REV=$(cat /workspace/logs/extra/public_revision.txt)
echo "[$(date -u +%FT%TZ)] boost: hf download --max-workers 16 @ $REV" >> /workspace/logs/chain.log
timeout 10800 /workspace/venv-serve/bin/hf download zai-org/GLM-4.5-Air --revision "$REV" --local-dir /workspace/ckpt/public --max-workers 16 >> /workspace/logs/extra/fetch_boost.log 2>&1
echo "[$(date -u +%FT%TZ)] boost: hf download rc=$?" >> /workspace/logs/chain.log
bash /workspace/relaunch_public.sh
EOS
nohup setsid bash /workspace/boost_public.sh > /workspace/logs/boost.out 2>&1 < /dev/null & disown
sleep 75
echo "--- running:"; ps -eo pid,etimes,args | grep -E "boost_public|hf download" | grep -v grep | cut -c1-90
A=$(du -sb /workspace/ckpt/public | cut -f1); sleep 30; B=$(du -sb /workspace/ckpt/public | cut -f1)
echo "rate: $(( (B-A)/30/1000000 )) MB/s"; du -sh /workspace/ckpt/public
tail -c 300 /workspace/logs/extra/fetch_boost.log | tr "\r" "\n" | tail -1; echo
