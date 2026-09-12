"""Print `<profile> <ssh-alias>` for every arm the orchestrator has in flight."""
import json
import pathlib
import sys

st = json.loads(pathlib.Path(sys.argv[1]).read_text())
for name, a in st["arms"].items():
    if a["status"] in ("RUNNING", "LAUNCHING") and a["alias"]:
        print(name, a["alias"])
