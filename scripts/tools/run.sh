#!/bin/bash
# usage: run.sh <local_script> [args...] [--bg]
#   .sh -> bash, .py -> the pose312 python. The script is piped into the container at /tmp/scr.
#
# ★ The credentials are NOT stored here. Export them first:
#       export SSH_HOST=user@your.server        # e.g. hankaiming@222.201.56.65
#       export SSH_PASS=...                     # ssh password
#       export SSH_PORT=22
#       export CONTAINER=lym_isaac              # docker container name
#       export COMBAT3D_ROOT=/workshop/Lym/combat3d
#
# ★ Scripts are always written to a FILE and executed from the file.
#   Inlining commands through wsl.exe silently eats $VAR and /path.
set -euo pipefail

SRC="$1"; shift || true
MODE=""
if [ "$#" -gt 0 ] && [ "${!#}" = "--bg" ]; then
  MODE="--bg"; set -- "${@:1:$#-1}"
fi
ARGS="$*"
NAME=$(basename "$SRC")

: "${SSH_HOST:?set SSH_HOST, e.g. user@host}"
: "${SSH_PASS:?set SSH_PASS}"
: "${CONTAINER:=lym_isaac}"
: "${SSH_PORT:=22}"
: "${COMBAT3D_ROOT:=/workshop/Lym/combat3d}"
PY="$COMBAT3D_ROOT/envs/miniconda3/envs/pose312/bin/python"

ssh_run() { sshpass -p "$SSH_PASS" ssh -p "$SSH_PORT" -o StrictHostKeyChecking=no \
              -o ConnectTimeout=15 "$SSH_HOST" "$@"; }

# .sh may need the remote-side helpers (stage8 render etc.), so push them too
ssh_run "docker exec $CONTAINER mkdir -p /tmp/scr"
ssh_run "docker exec -i $CONTAINER tee /tmp/scr/$NAME >/dev/null" < "$SRC"
# the shipped scripts live in scripts/tools/ and monocular/ — push them alongside
for extra in "$(dirname "$SRC")"/*.py; do
  [ -f "$extra" ] || continue
  ssh_run "docker exec -i $CONTAINER tee /tmp/scr/$(basename "$extra") >/dev/null" < "$extra"
done

case "$NAME" in
  *.py) CMD="$PY -u -W ignore /tmp/scr/$NAME $ARGS" ;;
  *)    CMD="bash /tmp/scr/$NAME $ARGS" ;;
esac

if [ "$MODE" = "--bg" ]; then
  ssh_run "docker exec -d $CONTAINER bash -lc '$CMD'"
else
  ssh_run "docker exec $CONTAINER bash -lc '$CMD'"
fi
