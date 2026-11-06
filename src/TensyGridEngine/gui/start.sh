#!/bin/bash

# This web dashboard and its simulation workers do not need desktop windows.
# VS Code Snap exports GTK/GLib paths incompatible with the system Python.
unset GSETTINGS_SCHEMA_DIR GTK_PATH GIO_MODULE_DIR GTK_IM_MODULE_FILE GTK_EXE_PREFIX GTK_MODULES
if [ -n "${XDG_DATA_DIRS_VSCODE_SNAP_ORIG:-}" ]; then
    export XDG_DATA_DIRS="$XDG_DATA_DIRS_VSCODE_SNAP_ORIG"
fi
if [ -n "${XDG_CONFIG_DIRS_VSCODE_SNAP_ORIG:-}" ]; then
    export XDG_CONFIG_DIRS="$XDG_CONFIG_DIRS_VSCODE_SNAP_ORIG"
fi
export MPLBACKEND=Agg
export QT_QPA_PLATFORM=offscreen

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
if [ -d "$PROJECT_ROOT/.venv" ]; then
    VENV_PYTHON="$PROJECT_ROOT/.venv/bin/python3"
elif [ -d "$PROJECT_ROOT/venv" ]; then
    VENV_PYTHON="$PROJECT_ROOT/venv/bin/python3"
else
    echo "Error: venv not found at $PROJECT_ROOT/.venv or $PROJECT_ROOT/venv"
    echo "Please create a virtual environment first."
    exit 1
fi

"$VENV_PYTHON" -c "import fastapi" 2>/dev/null || {
    echo "Installing GUI dependencies..."
    "$VENV_PYTHON" -m pip install fastapi uvicorn
}

echo "Starting VeraGrid GUI at http://localhost:8550"
exec "$VENV_PYTHON" "$SCRIPT_DIR/dashboard.py"
