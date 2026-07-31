#!/usr/bin/env bash
# Lives at /workspace/bootstrap.sh on sardine-run. Sourced from ~/.bashrc.
#
# Why this exists: a RunPod pod is a container. On stop/start the root
# filesystem is rebuilt from the image and everything outside /workspace is
# gone -- node, claude, dotfiles, ssh keys. This re-points a fresh container
# at the volume so nothing has to be reinstalled.
#
# Safe to run repeatedly. Sourced, not executed, so no `set -e`.

WS=/workspace

# --- secrets -----------------------------------------------------------------
# Pod env vars are not reliably visible in SSH-over-TCP sessions, so secrets
# live in a file on the volume instead of the pod's env config.
if [ -f "$WS/.env" ]; then
    set -a
    . "$WS/.env"
    set +a
fi

# --- node (nvm on the volume) ------------------------------------------------
export NVM_DIR="$WS/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && . "$NVM_DIR/nvm.sh" --no-use
[ -s "$NVM_DIR/nvm.sh" ] && nvm use --lts >/dev/null 2>&1

# --- uv / python -------------------------------------------------------------
export UV_CACHE_DIR="$WS/.cache/uv"
export PATH="$WS/.local/bin:$PATH"

# --- claude config on the volume ---------------------------------------------
# First run migrates any container-local config onto the volume rather than
# deleting it.
if [ ! -L "$HOME/.claude" ]; then
    if [ -d "$HOME/.claude" ] && [ ! -e "$WS/.claude" ]; then
        mv "$HOME/.claude" "$WS/.claude"
    fi
    rm -rf "$HOME/.claude"
    mkdir -p "$WS/.claude"
    ln -s "$WS/.claude" "$HOME/.claude"
fi
if [ ! -L "$HOME/.claude.json" ]; then
    if [ -f "$HOME/.claude.json" ] && [ ! -e "$WS/.claude.json" ]; then
        mv "$HOME/.claude.json" "$WS/.claude.json"
    fi
    rm -f "$HOME/.claude.json"
    [ -e "$WS/.claude.json" ] || echo '{}' > "$WS/.claude.json"
    ln -s "$WS/.claude.json" "$HOME/.claude.json"
fi

# --- ssh key for reaching GPU pods -------------------------------------------
# sardine-run has its own keypair; its public half is injected as PUBLIC_KEY
# when it creates a GPU pod, so it can ssh in unattended.
mkdir -p "$HOME/.ssh" && chmod 700 "$HOME/.ssh"
if [ -f "$WS/.ssh/id_ed25519" ]; then
    ln -sf "$WS/.ssh/id_ed25519" "$HOME/.ssh/id_ed25519"
    ln -sf "$WS/.ssh/id_ed25519.pub" "$HOME/.ssh/id_ed25519.pub"
    chmod 600 "$WS/.ssh/id_ed25519"
fi

# --- git ---------------------------------------------------------------------
git config --global --get user.email >/dev/null 2>&1 || \
    git config --global user.email "angel.rmartinez25@gmail.com"
git config --global --get user.name >/dev/null 2>&1 || \
    git config --global user.name "Angel Martinez"
git config --global --add safe.directory "$WS/science-of-midtraining" 2>/dev/null

# --- convenience -------------------------------------------------------------
alias sardine='tmux new-session -A -s sardine'
cd "$WS/science-of-midtraining" 2>/dev/null || cd "$WS"
