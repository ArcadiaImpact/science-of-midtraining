#!/usr/bin/env python3
"""Unattended arm scheduler for the two 8xB200 snipes (Sid asleep, 2026-09-10/11).

Replaces on_landing.sh, which hard-wired "account-2 landing -> noex". The queue
is now priority-ordered and account-agnostic, per Sid's instruction:

    1  glm45_air_190m_clause_asym   first pod to land, either account
    2  glm45_air_500m_noex          second pod to land
    3  glm45_air_500m_worked        the next pod to come FREE -- i.e. the
                                    clause-asym pod once that arm is finished
                                    and Hub-verified

A pod becomes FREE again only after finish_arm.sh exits 0, which is the
campaign's durability gate (CHAIN_COMPLETE + PUBLISH_COMPLETE on the pod, then
the Hub tree checked against the 1B row's shape with per-group file floors).
Only then are that arm's local artifacts deleted to make room for the next one.

What this deliberately does NOT do:
  * terminate a pod (finish_arm.sh --terminate stays a human decision);
  * re-roll a host after a failed preflight (that means killing a pod and
    re-arming a sniper -- money, and a judgement call);
  * retry a launch more than twice.
Anything it will not handle is written to the log with an ALERT prefix and
surfaces in status.sh, which the 15-minute heartbeat reads.

Run detached:
    setsid nohup python3 overnight.py >> overnight/daemon.log 2>&1 &
"""
import json
import os
import pathlib
import re
import shlex
import subprocess
import sys
import threading
import time
import urllib.request

HERE = pathlib.Path(__file__).resolve().parent          # .../ops/overnight
OPS = HERE.parent                                        # .../ops
REPO = OPS.parents[4]                                    # worktree root
STATE = HERE / "state.json"
LOG = HERE / "orchestrator.log"
PIDFILE = HERE / "overnight.pid"
SKILL = os.environ.get("SKILL", "/root/.claude/skills/runpod-spinup")

POLL_S = 30
HEARTBEAT_S = 900          # 15 min, matching the heartbeat Sid asked for
MAX_LAUNCH_ATTEMPTS = 2
DISK_FLOOR_GB = 1400       # the profiles' min_free_disk_gb; chain.py re-checks it
HF_HOME_POD = "/workspace/hf-final-v1"

# Priority order. The list IS the queue.
ARMS = [
    "glm45_air_190m_clause_asym",
    "glm45_air_500m_noex",
    "glm45_air_500m_worked",
]

# One entry per sniper. `snipe_name` is both the pod name the sniper asks for
# and the suffix of the ssh alias it registers (runpod-<snipe_name>).
ACCOUNTS = [
    {"account": 2, "snipe_name": "glm-b200-noex-matched",
     "log": OPS / "snipe_glm-b200-noex-matched-acct2.log"},
    {"account": 1, "snipe_name": "glm-b200-worked-matched",
     "log": OPS / "snipe_glm-b200-worked-matched-acct1.log"},
]


def log(msg):
    line = f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {msg}"
    print(line, flush=True)
    with LOG.open("a") as fh:
        fh.write(line + "\n")


# ----------------------------------------------------------------- credentials
def read_env(name):
    for raw in (REPO / ".env").read_text().splitlines():
        if raw.startswith(name + "="):
            return raw.split("=", 1)[1].strip().strip("'\"")
    return ""


HF_TOKEN = read_env("HF_TOKEN")
KEYS = {
    1: read_env("RUNPOD_API_KEY"),
    2: pathlib.Path("/root/.runpod2-home/apikey").read_text().strip(),
}
os.environ["SSH_AUTH_SOCK"] = "/root/.ssh/agent.sock"


# ----------------------------------------------------------------------- state
def load_state():
    if STATE.exists():
        return json.loads(STATE.read_text())
    return {
        "arms": {a: {"priority": i + 1, "status": "PENDING", "pod": None,
                     "alias": None, "account": None, "attempts": 0}
                 for i, a in enumerate(ARMS)},
        "pods": {},
        "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def save(st):
    tmp = STATE.with_suffix(".tmp")
    tmp.write_text(json.dumps(st, indent=1) + "\n")
    tmp.replace(STATE)


# ------------------------------------------------------------------- RunPod API
def gql(account, query):
    req = urllib.request.Request(
        "https://api.runpod.io/graphql",
        data=json.dumps({"query": query}).encode(),
        headers={"Authorization": f"Bearer {KEYS[account]}",
                 "Content-Type": "application/json",
                 # RunPod 403s urllib's default User-Agent.
                 "User-Agent": "curl/8.5.0"})
    with urllib.request.urlopen(req, timeout=30) as fh:
        return json.load(fh)


def live_pods(account):
    """[(id, name, status, costPerHr)] -- the backstop for a sniper that landed
    a pod but died before writing LANDED, or wrote it somewhere unexpected."""
    try:
        d = gql(account, "query { myself { pods { id name desiredStatus costPerHr } } }")
    except Exception as exc:                       # transient: say so, retry later
        log(f"warn: pod query failed on account {account}: {exc}")
        return None
    return [(p["id"], p.get("name") or "", p.get("desiredStatus"), p.get("costPerHr") or 0.0)
            for p in ((d.get("data") or {}).get("myself") or {}).get("pods") or []]


# ---------------------------------------------------------------------- ssh
def rsh(alias, cmd, timeout=120):
    return subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=30", alias, cmd],
        capture_output=True, text=True, timeout=timeout)


def ensure_alias(snipe_name, pod_id):
    """Register runpod-<name> if the sniper did not get to it (or the pod's ssh
    port moved). Returns the alias, or None if ssh is still not exposed."""
    alias = f"runpod-{snipe_name}"
    if rsh(alias, "true", timeout=45).returncode == 0:
        return alias
    res = subprocess.run([sys.executable, f"{SKILL}/_resolve_ssh.py", pod_id, "--quiet"],
                         capture_output=True, text=True, timeout=120)
    parts = res.stdout.split()
    if len(parts) != 2:
        return None
    host, port = parts
    subprocess.run([sys.executable, f"{SKILL}/_ssh_alias.py", "add", snipe_name, pod_id, host, port],
                   capture_output=True, text=True, timeout=60)
    log(f"alias {alias} -> {host}:{port} (registered by the orchestrator)")
    return alias if rsh(alias, "true", timeout=45).returncode == 0 else None


# --------------------------------------------------------------- pod discovery
LANDED_RE = re.compile(r"^LANDED .* pod=([a-z0-9]+) name=(\S+)", re.M)


def discover(st):
    for cfg in ACCOUNTS:
        acct, name = cfg["account"], cfg["snipe_name"]
        pod_id = None
        if cfg["log"].exists():
            m = LANDED_RE.search(cfg["log"].read_text())
            if m:
                pod_id = m.group(1)
        if pod_id is None and (not sniper_alive(name) or st.get("cycle", 0) % 20 == 0):
            # Backstop: the sniper may have died after creating the pod. Cheap
            # enough at 1 call / 10 min, and unconditional once the sniper is gone.
            pods = live_pods(acct)
            if pods is not None:
                for pid, pname, status, _ in pods:
                    if pname == name and status == "RUNNING":
                        pod_id = pid
                        log(f"ALERT: pod {pid} ({name}) is RUNNING on account {acct} but the "
                            f"snipe log has no LANDED line -- picked it up from the API")
                        break
        if pod_id and pod_id not in st["pods"]:
            st["pods"][pod_id] = {"account": acct, "snipe_name": name, "alias": None,
                                  "state": "FREE", "arm": None,
                                  "seen": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
            log(f"LANDING: pod={pod_id} account={acct} name={name}")
        if pod_id and st["pods"][pod_id]["alias"] is None:
            alias = ensure_alias(name, pod_id)
            if alias:
                st["pods"][pod_id]["alias"] = alias
                log(f"pod {pod_id} reachable at {alias}")
            else:
                log(f"warn: pod {pod_id} not reachable over ssh yet")


# -------------------------------------------------------------------- launching
def launch(st_pod, profile, pod_id):
    """Run launch_arm.sh to completion in a worker thread (setup alone is ~1 h)."""
    alias, acct = st_pod["alias"], st_pod["account"]
    out = HERE / f"launch_{profile}.log"
    env = dict(os.environ,
               HF_TOKEN=HF_TOKEN, RUNPOD_API_KEY=KEYS[acct],
               SSH_AUTH_SOCK="/root/.ssh/agent.sock", GPU_TYPE="B200", PROVIDER="runpod")
    cmd = ["bash", str(OPS / "launch_arm.sh"), profile, pod_id, alias]
    log(f"LAUNCH {profile} on {pod_id} ({alias}, account {acct}) -> {out.name}")
    with out.open("a") as fh:
        fh.write(f"\n===== {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} "
                 f"{' '.join(shlex.quote(c) for c in cmd)} =====\n")
        fh.flush()
        rc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env,
                            cwd=str(OPS)).returncode
    with LOCK:
        st = load_state()
        arm = st["arms"][profile]
        if rc == 0:
            arm["status"] = "RUNNING"
            log(f"LAUNCH OK: {profile} is running on {pod_id}")
        else:
            arm["attempts"] += 1
            if arm["attempts"] >= MAX_LAUNCH_ATTEMPTS:
                arm["status"] = "FAILED"
                log(f"ALERT: launch of {profile} on {pod_id} failed rc={rc} after "
                    f"{arm['attempts']} attempts -- pod left UP, needs a human. "
                    f"See {out}")
            else:
                arm["status"] = "PENDING"          # one more go next cycle
                st["pods"][pod_id]["state"] = "FREE"
                st["pods"][pod_id]["arm"] = None
                log(f"ALERT: launch of {profile} on {pod_id} failed rc={rc} "
                    f"(attempt {arm['attempts']}); will retry once")
        save(st)


def schedule(st):
    for pod_id, pod in st["pods"].items():
        if pod["state"] != "FREE" or pod["alias"] is None:
            continue
        nxt = next((a for a in ARMS if st["arms"][a]["status"] == "PENDING"), None)
        if nxt is None:
            if not pod.get("idle_alerted"):
                log(f"ALERT: pod {pod_id} is FREE and the queue holds nothing "
                    f"PENDING -- it is billing while idle")
                pod["idle_alerted"] = True
            continue
        arm = st["arms"][nxt]
        arm.update(status="LAUNCHING", pod=pod_id, alias=pod["alias"], account=pod["account"])
        pod.update(state="BUSY", arm=nxt, idle_alerted=False)
        save(st)
        threading.Thread(target=launch, args=(dict(pod), nxt, pod_id), daemon=True).start()


# --------------------------------------------------------------------- finishing
def free_disk_gb(alias):
    r = rsh(alias, "python3 -c \"import shutil;print(shutil.disk_usage('/workspace').free/1e9)\"")
    try:
        return float(r.stdout.strip())
    except ValueError:
        return None


def poll_running(st):
    for profile, arm in st["arms"].items():
        if arm["status"] != "RUNNING":
            continue
        alias, pod_id = arm["alias"], arm["pod"]
        root = f"/workspace/final_v1/{profile}/charter"
        probe = (f"test -f {root}/CHAIN_COMPLETE.json && test -f {root}/PUBLISH_COMPLETE.json "
                 f"&& echo DONE || echo RUNNING")
        r = rsh(alias, probe)
        if r.returncode != 0 and "RUNNING" not in r.stdout:
            # ssh itself failed -- a RunPod container restart moves the public
            # port, which would otherwise look like "the arm never finishes".
            log(f"warn: ssh to {alias} failed; re-resolving the pod's ssh endpoint")
            fixed = ensure_alias(st["pods"][pod_id]["snipe_name"], pod_id)
            if not fixed:
                continue
            arm["alias"] = st["pods"][pod_id]["alias"] = alias = fixed
            save(st)
            r = rsh(alias, probe)
        if "DONE" not in r.stdout:
            continue
        log(f"{profile}: CHAIN_COMPLETE + PUBLISH_COMPLETE on {pod_id}; running finish_arm.sh")
        out = HERE / f"finish_{profile}.log"
        env = dict(os.environ, HF_TOKEN=HF_TOKEN, RUNPOD_API_KEY=KEYS[arm["account"]],
                   SSH_AUTH_SOCK="/root/.ssh/agent.sock", PROVIDER="runpod")
        with out.open("a") as fh:
            rc = subprocess.run(["bash", str(OPS / "finish_arm.sh"), profile, pod_id, alias],
                                stdout=fh, stderr=subprocess.STDOUT, env=env,
                                cwd=str(OPS)).returncode
        if rc != 0:
            arm["finish_failures"] = arm.get("finish_failures", 0) + 1
            log(f"ALERT: finish_arm.sh {profile} exited {rc} "
                f"(failure {arm['finish_failures']}/5) -- see {out}; NOT freeing the pod")
            if arm["finish_failures"] >= 5:
                arm["status"] = "FINISH_FAILED"
                log(f"ALERT: giving up on finish_arm.sh for {profile}; pod {pod_id} stays "
                    f"BUSY and its artifacts stay put. Needs a human.")
            save(st)
            continue
        arm["status"] = "FINISHED"
        log(f"FINISHED + Hub-verified: {profile} on {pod_id}")

        # Only now, with the artifacts provably on the Hub, reclaim the disk so
        # the next arm can clear its 1400 GB floor. HF_HOME (the 221 GB base) is
        # kept -- deleting the arm dir returns the pod to the same state a fresh
        # pod is in when its chain starts. If that is somehow not enough, drop
        # the base too and let setup.sh re-fetch it.
        rsh(alias, f"rm -rf /workspace/final_v1/{profile}", timeout=900)
        free = free_disk_gb(alias)
        log(f"pod {pod_id}: {free:.0f} GB free after removing final_v1/{profile}"
            if free else f"pod {pod_id}: could not measure free disk")
        if free is not None and free < DISK_FLOOR_GB + 50:
            log(f"pod {pod_id}: {free:.0f} GB < {DISK_FLOOR_GB}+50 floor; clearing "
                f"{HF_HOME_POD} too (setup.sh re-downloads the base, ~30 min)")
            rsh(alias, f"rm -rf {HF_HOME_POD}", timeout=1800)
            free = free_disk_gb(alias)
            log(f"pod {pod_id}: {free:.0f} GB free after clearing the base cache"
                if free else f"pod {pod_id}: could not measure free disk")
        st["pods"][pod_id].update(state="FREE", arm=None)
        save(st)


# --------------------------------------------------------------------- sniffers
def sniper_alive(name):
    return subprocess.run(["pgrep", "-f", f"snipe_b200_pod.sh {name}"],
                          capture_output=True).returncode == 0


def check_snipers(st):
    for cfg in ACCOUNTS:
        name = cfg["snipe_name"]
        landed = any(p["snipe_name"] == name for p in st["pods"].values())
        key = f"sniper_dead_{name}"
        if not landed and not sniper_alive(name) and not st.get(key):
            log(f"ALERT: sniper {name} (account {cfg['account']}) is gone and never "
                f"landed a pod -- nothing is hunting for that account any more")
            st[key] = True
            save(st)


# ------------------------------------------------------------------------ main
LOCK = threading.Lock()


def main():
    PIDFILE.write_text(f"{os.getpid()}\n")
    log(f"orchestrator up: queue {' > '.join(ARMS)}; repo {REPO} @ "
        f"{subprocess.run(['git', '-C', str(REPO), 'rev-parse', '--short', 'HEAD'], capture_output=True, text=True).stdout.strip()}")
    if not HF_TOKEN:
        log("FATAL: no HF_TOKEN in .env")
        return 78
    last_beat = 0.0
    while True:
        try:
            with LOCK:
                st = load_state()
                st["cycle"] = st.get("cycle", 0) + 1
                discover(st)
                check_snipers(st)
                schedule(st)
                save(st)
            with LOCK:
                poll_running(load_state())
            if time.time() - last_beat > HEARTBEAT_S:
                last_beat = time.time()
                st = load_state()
                arms = " ".join(f"{a.split('_air_')[1]}={st['arms'][a]['status']}" for a in ARMS)
                pods = " ".join(f"{p}:{d['state']}" for p, d in st["pods"].items()) or "none"
                log(f"heartbeat: arms[{arms}] pods[{pods}]")
            st = load_state()
            if all(st["arms"][a]["status"] in ("FINISHED", "FAILED") for a in ARMS):
                log("all arms terminal; orchestrator exiting")
                return 0
        except Exception as exc:                    # never let one bad cycle kill the loop
            log(f"ALERT: orchestrator cycle raised {type(exc).__name__}: {exc}")
        time.sleep(POLL_S)


if __name__ == "__main__":
    sys.exit(main())
