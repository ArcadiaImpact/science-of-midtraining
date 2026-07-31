#!/usr/bin/env bash
# One-shot provisioning for sardine-run. Run ON the pod, after the volume is
# mounted and /workspace/.env exists:
#
#   ssh sardine 'bash -s' < infra/sardine-run/provision.sh
#
# Everything installable lands under /workspace so it survives a pod restart.
# Safe to re-run.
set -euo pipefail

WS=/workspace
NODE_VERSION="--lts"

[ -d "$WS" ] || { echo "/workspace is not mounted -- is the network volume attached?" >&2; exit 1; }
[ -f "$WS/.env" ] || { echo "$WS/.env is missing -- create it before provisioning" >&2; exit 1; }

set -a; . "$WS/.env"; set +a

echo "=== 1/8 system packages ==="
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq tmux git curl cron rsync jq >/dev/null

echo "=== 2/8 node via nvm on the volume ==="
export NVM_DIR="$WS/.nvm"
mkdir -p "$NVM_DIR"
if [ ! -s "$NVM_DIR/nvm.sh" ]; then
    curl -fsSL https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.1/install.sh | bash
fi
. "$NVM_DIR/nvm.sh"
nvm install "$NODE_VERSION"
nvm alias default "$(nvm current)"

echo "=== 3/8 claude code ==="
npm install -g @anthropic-ai/claude-code >/dev/null
claude --version

echo "=== 4/8 uv ==="
export UV_INSTALL_DIR="$WS/.local/bin"
mkdir -p "$UV_INSTALL_DIR"
if [ ! -x "$UV_INSTALL_DIR/uv" ]; then
    curl -LsSf https://astral.sh/uv/install.sh | UV_INSTALL_DIR="$UV_INSTALL_DIR" sh
fi
export PATH="$UV_INSTALL_DIR:$PATH"
uv --version

echo "=== 5/8 ssh keypair for reaching GPU pods ==="
mkdir -p "$WS/.ssh" && chmod 700 "$WS/.ssh"
if [ ! -f "$WS/.ssh/id_ed25519" ]; then
    ssh-keygen -t ed25519 -N '' -C 'sardine-run' -f "$WS/.ssh/id_ed25519"
fi
chmod 600 "$WS/.ssh/id_ed25519"
echo "sardine-run public key (injected as PUBLIC_KEY into GPU pods):"
cat "$WS/.ssh/id_ed25519.pub"

echo "=== 6/8 repo ==="
# Three ways the repo can get here, in order of preference:
#   1. GITHUB_TOKEN in .env  -> clone/fetch over https
#   2. already present       -> rsynced from the laptop, leave it alone
#   3. neither               -> stop with instructions rather than a cryptic
#                               "could not read Username" from git
if [ -n "${GITHUB_TOKEN:-}" ]; then
    git config --global credential.helper "store --file=$WS/.git-credentials"
    printf 'https://x-access-token:%s@github.com\n' "$GITHUB_TOKEN" > "$WS/.git-credentials"
    chmod 600 "$WS/.git-credentials"
fi

if [ -d "$WS/science-of-midtraining/.git" ]; then
    echo "repo already present"
    [ -n "${GITHUB_TOKEN:-}" ] && git -C "$WS/science-of-midtraining" fetch --all --quiet || true
elif [ -n "${GITHUB_TOKEN:-}" ]; then
    git clone https://github.com/ArcadiaImpact/science-of-midtraining.git "$WS/science-of-midtraining"
else
    cat >&2 <<'MSG'
No repo at /workspace/science-of-midtraining and no GITHUB_TOKEN to clone with.

Either add GITHUB_TOKEN to /workspace/.env, or push the working tree from the
laptop first:

  rsync -az --delete --exclude '.venv' --exclude '__pycache__' \
      ~/science-of-midtraining/ sardine:/workspace/science-of-midtraining/

then re-run this script.
MSG
    exit 1
fi
git config --global --add safe.directory "$WS/science-of-midtraining"

echo "=== 7/8 dotfiles + bootstrap ==="
install -m 0755 "$WS/science-of-midtraining/infra/sardine-run/bootstrap.sh" "$WS/bootstrap.sh"
cp "$WS/science-of-midtraining/infra/sardine-run/tmux.conf" "$WS/.tmux.conf"
ln -sf "$WS/.tmux.conf" "$HOME/.tmux.conf"
mkdir -p "$WS/.sardine"
install -m 0755 "$WS/science-of-midtraining/infra/sardine-run/idle_sweeper.py" "$WS/.sardine/idle_sweeper.py"
# Operating rules for any Claude session on this box. Lives above the repo
# checkout so it applies to everything under /workspace.
cp "$WS/science-of-midtraining/infra/sardine-run/workspace-CLAUDE.md" "$WS/CLAUDE.md"

# Ubuntu's stock .bashrc bails out early for non-interactive shells:
#     [ -z "$PS1" ] && return
# Appending after that line means `ssh sardine 'cmd'` never sees node or claude.
# So the hook goes at the TOP of .bashrc, above the guard, and also into
# .profile for login shells.
BOOT_LINE='[ -f /workspace/bootstrap.sh ] && . /workspace/bootstrap.sh'
for rc in "$HOME/.bashrc" "$HOME/.profile"; do
    touch "$rc"
    # Drop any previous copy (including one appended below the guard).
    grep -v 'workspace/bootstrap.sh' "$rc" > "$rc.tmp" || true
    { printf '# sardine-run: restore the volume-backed environment\n%s\n\n' "$BOOT_LINE"; cat "$rc.tmp"; } > "$rc"
    rm -f "$rc.tmp"
done
# Both files live on the container disk and reset on restart; keep copies on
# the volume so a restart can be repaired without re-running provisioning.
cp "$HOME/.bashrc" "$WS/.bashrc.sardine"
cp "$HOME/.profile" "$WS/.profile.sardine"

echo "=== 8/8 idle sweeper cron + runpod mcp ==="
service cron start >/dev/null 2>&1 || true
CRON_LINE="*/10 * * * * . /workspace/.env; /usr/bin/python3 /workspace/.sardine/idle_sweeper.py >> /workspace/.sardine/cron.log 2>&1"
# Built in a temp file rather than a subshell pipeline: `crontab -l` exits 1 on
# an empty crontab and `grep -v` exits 1 on empty input, either of which kills
# a `( ... ) | crontab -` subshell under `set -e` and silently installs nothing.
TMP_CRON="$(mktemp)"
crontab -l 2>/dev/null | grep -v idle_sweeper.py > "$TMP_CRON" || true
echo "$CRON_LINE" >> "$TMP_CRON"
crontab "$TMP_CRON"
rm -f "$TMP_CRON"
echo "crontab now:"; crontab -l

# The MCP server inherits RUNPOD_API_KEY from the environment, so the key is
# not duplicated into .claude.json.
claude mcp add runpod -- npx -y @runpod/mcp-server@latest 2>&1 | tail -2 || \
    echo "(runpod mcp may already be registered)"

echo
echo "=== provisioning complete ==="
echo "Start the session with:  tmux new-session -A -s sardine"
