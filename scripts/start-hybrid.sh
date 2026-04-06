#!/usr/bin/env bash
# start-hybrid.sh — Start the opendataloader-pdf hybrid OCR backend
#
# Usage:
#   ./scripts/start-hybrid.sh [port]
#
# Environment:
#   ODL_HYBRID_PORT  — backend port (default: 5002)
#   ODL_OCR_LANG     — comma-separated OCR languages (default: "pt,en")

set -euo pipefail

PORT="${1:-${ODL_HYBRID_PORT:-5002}}"
LANG="${ODL_OCR_LANG:-pt,en}"
PID_FILE="/tmp/odl-hybrid.pid"
MAX_WAIT=45  # seconds to wait for readiness

HEALTH_URL="http://localhost:${PORT}/health"

# ──────────────────────────────────────────────
# 1. Check if already running
# ──────────────────────────────────────────────
if curl -sf "${HEALTH_URL}" > /dev/null 2>&1; then
    echo "[ODL] Hybrid backend already running on port ${PORT} ✓"
    exit 0
fi

# ──────────────────────────────────────────────
# 2. Verify dependencies
# ──────────────────────────────────────────────
if ! command -v opendataloader-pdf-hybrid &> /dev/null; then
    echo "[ODL] ERROR: opendataloader-pdf-hybrid not found."
    echo "      Install with: pip install 'opendataloader-pdf[hybrid]'"
    exit 1
fi

# ──────────────────────────────────────────────
# 3. Start backend
# ──────────────────────────────────────────────
echo "[ODL] Starting hybrid backend on port ${PORT} (OCR langs: ${LANG}) ..."
opendataloader-pdf-hybrid \
    --port "${PORT}" \
    --force-ocr \
    --ocr-lang "${LANG}" \
    --log-level warning &

ODL_PID=$!
echo "${ODL_PID}" > "${PID_FILE}"
echo "[ODL] Backend started with PID ${ODL_PID}"

# ──────────────────────────────────────────────
# 4. Wait for readiness
# ──────────────────────────────────────────────
echo "[ODL] Waiting for backend to be ready (up to ${MAX_WAIT}s) ..."
for i in $(seq 1 "${MAX_WAIT}"); do
    if curl -sf "${HEALTH_URL}" > /dev/null 2>&1; then
        echo "[ODL] Backend ready after ${i}s ✓"
        exit 0
    fi
    # Check if process died
    if ! kill -0 "${ODL_PID}" 2>/dev/null; then
        echo "[ODL] ERROR: Backend process ${ODL_PID} died unexpectedly"
        rm -f "${PID_FILE}"
        exit 1
    fi
    sleep 1
done

echo "[ODL] ERROR: Backend did not become ready within ${MAX_WAIT}s"
kill "${ODL_PID}" 2>/dev/null || true
rm -f "${PID_FILE}"
exit 1
