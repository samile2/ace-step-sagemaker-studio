#!/usr/bin/env bash
set -euo pipefail

backend_pid=""
proxy_pid=""

shutdown() {
    trap - EXIT INT TERM
    if [ -n "${proxy_pid}" ]; then
        kill "${proxy_pid}" 2>/dev/null || true
    fi
    if [ -n "${backend_pid}" ]; then
        kill "${backend_pid}" 2>/dev/null || true
    fi
    wait 2>/dev/null || true
}

trap shutdown EXIT INT TERM

# Démarre l'API officielle ACE-Step sur le port interne 8001.
/app/docker-entrypoint.sh &
backend_pid=$!

# Expose le contrat SageMaker sur le port 8080.
uv run python /app/sagemaker_app.py &
proxy_pid=$!

# Le conteneur s'arrête si l'un des deux services s'arrête.
wait -n "${backend_pid}" "${proxy_pid}"
