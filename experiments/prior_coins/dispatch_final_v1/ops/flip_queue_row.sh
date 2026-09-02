#!/usr/bin/env bash
# Flip one staged (commented) queue row live, or comment a live row back out.
#
# The ONLY file mutation this script performs is toggling the leading "# " on
# a single row of one queue file in this directory. It exists so the
# orchestrating agent can be allow-listed for exactly this action (queue
# flips) without a blanket sed/Edit permission on ops files.
#
# Usage:
#   flip_queue_row.sh <queue-file-basename> <profile> [--off|--all-arms]
#   flip_queue_row.sh queue.txt gemma3_27b_19m
#   flip_queue_row.sh queue_glm_a3.txt glm45_air_190m --off
#   flip_queue_row.sh queue.txt gemma3_12b_50m_divresp --all-arms
#
# The one-row default is the safety property: a queue flip starts paid pods,
# and "exactly one" makes a typo loud. --all-arms relaxes it to "every staged
# row of ONE profile in ONE file", for studies whose work units are per-arm
# rather than per-row (diverse-response ships 3, one per arm). It is still a
# single deliberate action over a single profile; it does not let a wildcard
# or a wrong profile name flip anything unintended.
set -euo pipefail

OPS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
QFILE_NAME="${1:?queue file basename required}"
PROFILE="${2:?profile name required}"
MODE="${3:-on}"

case "$QFILE_NAME" in
  queue*.txt) ;;
  *) echo "refusing: '$QFILE_NAME' is not a queue*.txt basename" >&2; exit 2 ;;
esac
case "$PROFILE" in
  *[!A-Za-z0-9_]*) echo "refusing: bad profile name '$PROFILE'" >&2; exit 2 ;;
esac

QFILE="$OPS/$QFILE_NAME"
[ -f "$QFILE" ] || { echo "no such queue file: $QFILE" >&2; exit 2; }

# Match a row: optional "# ", priority digits, tab, the profile, tab.
LIVE_RE="^[0-9]+	$PROFILE	"
STAGED_RE="^# ?[0-9]+	$PROFILE	"

live=$(grep -cE "$LIVE_RE" "$QFILE" || true)
staged=$(grep -cE "$STAGED_RE" "$QFILE" || true)

if [ "$MODE" = "--off" ]; then
  [ "$live" -eq 1 ] || { echo "refusing: expected exactly 1 live row for $PROFILE, found $live" >&2; exit 3; }
  python3 - "$QFILE" "$PROFILE" <<'EOF'
import re, sys
path, profile = sys.argv[1], sys.argv[2]
lines = open(path).read().splitlines(keepends=True)
out, n = [], 0
for line in lines:
    if re.match(rf"^[0-9]+\t{re.escape(profile)}\t", line):
        out.append("# " + line); n += 1
    else:
        out.append(line)
assert n == 1, f"expected 1 live row, matched {n}"
open(path, "w").write("".join(out))
print(f"commented {profile} in {path}")
EOF
elif [ "$MODE" = "--all-arms" ]; then
  [ "$staged" -ge 1 ] || { echo "refusing: no staged rows for $PROFILE" >&2; exit 3; }
  [ "$live" -eq 0 ] || { echo "refusing: $PROFILE already has $live live row(s)" >&2; exit 3; }
  python3 - "$QFILE" "$PROFILE" <<'EOF'
import re, sys
path, profile = sys.argv[1], sys.argv[2]
lines = open(path).read().splitlines(keepends=True)
out, n = [], 0
for line in lines:
    if re.match(rf"^# ?[0-9]+\t{re.escape(profile)}\t", line):
        out.append(re.sub(r"^# ?", "", line)); n += 1
    else:
        out.append(line)
assert n >= 1, f"expected >=1 staged row, matched {n}"
open(path, "w").write("".join(out))
print(f"flipped {n} row(s) of {profile} LIVE in {path}")
EOF
else
  [ "$staged" -eq 1 ] || { echo "refusing: expected exactly 1 staged row for $PROFILE, found $staged (use --all-arms for a per-arm study)" >&2; exit 3; }
  [ "$live" -eq 0 ] || { echo "refusing: $PROFILE already has a live row" >&2; exit 3; }
  python3 - "$QFILE" "$PROFILE" <<'EOF'
import re, sys
path, profile = sys.argv[1], sys.argv[2]
lines = open(path).read().splitlines(keepends=True)
out, n = [], 0
for line in lines:
    if re.match(rf"^# ?[0-9]+\t{re.escape(profile)}\t", line):
        out.append(re.sub(r"^# ?", "", line)); n += 1
    else:
        out.append(line)
assert n == 1, f"expected 1 staged row, matched {n}"
open(path, "w").write("".join(out))
print(f"flipped {profile} LIVE in {path}")
EOF
fi

grep -nE "	$PROFILE	" "$QFILE" | head -3
