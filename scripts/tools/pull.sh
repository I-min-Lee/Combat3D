#!/bin/bash
# usage: pull.sh <container_path> [container_path2 ...]
#   Copies files out of the container back to $PULL_DST (default: ./_pull).
#
# ★ Credentials are read from the environment, never stored here:
#       export SSH_HOST=user@your.server
#       export SSH_PASS=...
#       export SSH_PORT=22
#       export CONTAINER=lym_isaac
set -euo pipefail

: "${SSH_HOST:?set SSH_HOST, e.g. user@host}"
: "${SSH_PASS:?set SSH_PASS}"
: "${CONTAINER:=lym_isaac}"
: "${SSH_PORT:=22}"
: "${PULL_DST:=./_pull}"

mkdir -p "$PULL_DST"

ssh_run() { sshpass -p "$SSH_PASS" ssh -p "$SSH_PORT" -o StrictHostKeyChecking=no \
              -o ConnectTimeout=15 "$SSH_HOST" "$@"; }

ssh_run "rm -rf /tmp/pull && mkdir -p /tmp/pull"
for p in "$@"; do
  ssh_run "docker cp $CONTAINER:$p /tmp/pull/"
done
sshpass -p "$SSH_PASS" scp -P "$SSH_PORT" -o StrictHostKeyChecking=no \
    "$SSH_HOST:/tmp/pull/*" "$PULL_DST/"
ls -la "$PULL_DST"
