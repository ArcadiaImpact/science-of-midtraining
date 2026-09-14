"""Print balance + pod inventory for one RunPod account. Reads the key from stdin."""
import json, sys, urllib.request

key = sys.stdin.read().strip()
#: `spendLimit` / `currentSpendPerHr` are the account's hourly cap and its
#: current draw. They matter because a create that would breach the cap is
#: refused with a message the snipers classify as a SUPPLY miss -- so a capped
#: account hunts forever, logging "miss (supply)", and the sniper row reads UP.
#: Account 1 hit exactly that on 2026-09-14 when unrelated 4xH200 pods took it
#: to $36.88/h, leaving $42.51/h against a B200's $54.32/h.
Q = ('query { myself { clientBalance spendLimit currentSpendPerHr '
     'pods { id name desiredStatus costPerHr '
     'machine { gpuDisplayName } } } }')
#: 8xB200 SECURE, the only thing the live snipers ask for.
B200_RATE = 54.32
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

limit = me.get("spendLimit")
draw = me.get("currentSpendPerHr")
if limit is not None and draw is not None:
    head = limit - draw
    verdict = "fits" if head >= B200_RATE else "BLOCKED"
    print("  cap $%.0f/h, drawing $%.2f/h -> $%.2f/h headroom vs 8xB200 $%.2f: %s"
          % (limit, draw, head, B200_RATE, verdict))
    if verdict == "BLOCKED":
        # status.sh greps for this token to raise its own alarm.
        print("  CAPPED: a snipe on this account cannot land until something "
              "comes off it -- it will log 'miss (supply)' forever")
