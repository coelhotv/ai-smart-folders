#!/usr/bin/env bash
# stop-hybrid.sh — Stop the opendataloader-pdf hybrid OCR backend
#
# Usage:
#   ./scripts/stop-hybrid.sh

PID_FILE="/tmp/odl-hybrid.pid"

if [ -f "${PID_FILE}" ]; then
    ODL_PID=$(cat "${PID_FILE}")
    if kill -0 "${ODL_PID}" 2>/dev/null; then
        echo "[ODL] Stopping hybrid backend (PID ${ODL_PID}) ..."
        kill "${ODL_PID}"
        echo "[ODL] Stopped ✓"
    else
        echo "[ODL] Process ${ODL_PID} not running (already stopped?)"
    fi
    rm -f "${PID_FILE}"
else
    # Fallback: try to find the process by name
    if pgrep -f "opendataloader-pdf-hybrid" > /dev/null; then
        echo "[ODL] Stopping hybrid backend (found by name) ..."
        pkill -f "opendataloader-pdf-hybrid" || true
        echo "[ODL] Stopped ✓"
    else
        echo "[ODL] No hybrid backend process found"
    fi
fi
