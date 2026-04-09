#!/usr/bin/env bash
set -euo pipefail

# === CONFIG ===
KAGGLE_JSON_PATH="${KAGGLE_JSON_PATH:-kaggle.json}"
WORKDIR="${WORKDIR:-$(pwd)}"

# === FUNCTIONS ===

setup_kaggle() {
    echo "[*] Setting up Kaggle..."
    mkdir -p ~/.kaggle
    cp "$KAGGLE_JSON_PATH" ~/.kaggle/kaggle.json
    chmod 600 ~/.kaggle/kaggle.json
}

install_deps() {
    echo "[*] Installing Python dependencies..."
    cd "$WORKDIR"
    uv pip install -r requirements.txt
}

main() {
    setup_kaggle
    install_deps
    echo "[✔] Setup complete."
}

main "$@"
