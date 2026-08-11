#!/usr/bin/env bash
# =============================================================================
#  stop.sh — Stop the NVIDIA Nemotron 3.5 Lightning DSpark SGLang container
#
#  Stops and removes the SGLang Docker container started by start.sh.
#  Also cleans up the PID file.
#
#  Usage:
#    ./stop.sh
#
#  Environment variables (optional):
#    WORK_DIR    Working directory (default: $(pwd))
# =============================================================================
set -euo pipefail

# ---- Configuration ----------------------------------------------------------
WORK_DIR="$(pwd)"
CONTAINER_NAME="nemotron-3.5-lightning-dspark"
PID_FILE="${WORK_DIR}/.sglang.pid"

# ---- Stop the container ------------------------------------------------------
if docker ps -a --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
  echo "Stopping SGLang container ${CONTAINER_NAME} …"
  docker rm -f "${CONTAINER_NAME}" >/dev/null
  echo "Container ${CONTAINER_NAME} stopped and removed."
else
  echo "Container ${CONTAINER_NAME} is not running."
fi

# ---- Clean up PID file ------------------------------------------------------
if [[ -f "${PID_FILE}" ]]; then
  echo "Removing PID file ${PID_FILE} …"
  rm -f "${PID_FILE}"
fi

echo "Done."