#!/usr/bin/env bash
# BankReasonTrainer — launch Voilà web-UI on localhost.
#
# Works on Linux (Ubuntu/Fedora/Arch/openSUSE) and macOS. Unlike `run.sh`
# this does NOT require Tkinter — web-UI runs in the browser over the
# ipywidgets + Voilà stack from `pyproject.toml` [ui] extra.
#
# Environment overrides:
#   BRT_PORT     — port to bind Voilà (default: 8866)
#   BRT_HOST     — host to bind Voilà (default: 127.0.0.1)
#   PYTHON       — explicit python interpreter (default: python3.11, then python3)
#   BRT_NO_OPEN  — set to 1 to skip auto-opening the browser

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PORT="${BRT_PORT:-8866}"
HOST="${BRT_HOST:-127.0.0.1}"

# --- pick Python interpreter ------------------------------------------
if [[ -n "${PYTHON:-}" ]]; then
    PY="$PYTHON"
elif command -v python3.11 >/dev/null 2>&1; then
    PY="python3.11"
elif command -v python3.12 >/dev/null 2>&1; then
    PY="python3.12"
elif command -v python3 >/dev/null 2>&1; then
    PY="python3"
else
    echo "ERROR: python3 not found. Install Python 3.11+ from https://python.org/downloads/"
    exit 1
fi

# Verify Python version is >= 3.11
if ! "$PY" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)' 2>/dev/null; then
    echo "ERROR: Python 3.11+ required. Found: $("$PY" --version 2>&1)"
    echo "       Set the PYTHON env var to a newer interpreter, e.g. PYTHON=python3.11 ./run_web.sh"
    exit 1
fi

echo "[run_web] using: $("$PY" --version 2>&1) ($(which "$PY"))"

# --- ensure ui deps ---------------------------------------------------
if ! "$PY" -c 'import voila, ipywidgets' 2>/dev/null; then
    echo "[run_web] installing web-UI dependencies (this can take a minute on first run)…"
    # Prefer project install via pyproject.toml [ui] extra.
    if ! "$PY" -m pip install --quiet --disable-pip-version-check -e ".[ui]"; then
        echo "ERROR: pip install failed. Try manually:"
        echo "         $PY -m pip install 'ipywidgets>=8.0' 'voila>=0.5'"
        exit 1
    fi
fi

# --- check port is free ----------------------------------------------
if "$PY" -c "
import socket, sys
s = socket.socket()
try:
    s.bind(('$HOST', $PORT))
except OSError:
    sys.exit(1)
finally:
    s.close()
" 2>/dev/null; then
    :
else
    echo "ERROR: port $PORT on $HOST is already in use."
    echo "       Pick another one: BRT_PORT=8867 ./run_web.sh"
    exit 1
fi

# --- open browser (best-effort, non-blocking) ------------------------
URL="http://$HOST:$PORT/"
if [[ "${BRT_NO_OPEN:-0}" != "1" ]]; then
    (
        # wait for voila to start serving before opening
        for _ in 1 2 3 4 5 6 7 8 9 10; do
            sleep 0.5
            if "$PY" -c "import urllib.request; urllib.request.urlopen('$URL', timeout=0.5)" 2>/dev/null; then
                break
            fi
        done
        if command -v xdg-open >/dev/null 2>&1; then
            xdg-open "$URL" >/dev/null 2>&1 || true
        elif command -v open >/dev/null 2>&1; then
            open "$URL" >/dev/null 2>&1 || true
        fi
    ) &
fi

echo "[run_web] serving on $URL  (Ctrl+C to stop)"
export PYTHONPATH="$SCRIPT_DIR:${PYTHONPATH:-}"
exec "$PY" -m voila notebooks/ui.ipynb \
    --port="$PORT" \
    --Voila.ip="$HOST" \
    --no-browser
