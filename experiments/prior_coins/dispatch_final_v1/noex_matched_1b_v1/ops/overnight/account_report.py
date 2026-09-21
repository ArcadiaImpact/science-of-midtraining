"""Print balance + pod inventory for one RunPod account. Reads the key from stdin."""
import json, sys, urllib.request

key = sys.stdin.read().strip()
Q = ('query { myself { clientBalance pods { id name desiredStatus costPerHr '
     'machine { gpuDisplayName } } } }')
req = urllib.request.Request(
    "https://api.runpod.io/graphql",
    data=json.dumps({"query": Q}).encode(),
    headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json",
             # RunPod 403s the default urllib User-Agent; curl-like UA required.
             "User-Agent": "curl/8.5.0"})
try:
    d = json.load(urllib.request.urlopen(req, timeout=30))
except Exception as e:                       # network/auth: say so, do not guess
    print(f"  QUERY FAILED: {e}"); raise SystemExit(2)
me = (d.get("data") or {}).get("myself") or {}
print("  balance $%.2f" % (me.get("clientBalance") or 0.0))
total = 0.0
for p in me.get("pods") or []:
    rate = p.get("costPerHr") or 0.0
    if p.get("desiredStatus") == "RUNNING":
        total += rate
    gpu = (p.get("machine") or {}).get("gpuDisplayName") or "?"
    print("  %-18s %-30s %-9s $%-7s %s"
          % (p.get("id"), (p.get("name") or "")[:30], p.get("desiredStatus"), rate, gpu))
print("  running total $%.2f/h" % total)
