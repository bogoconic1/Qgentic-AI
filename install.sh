#!/usr/bin/env bash
set -euo pipefail

# === CONFIG ===
KAGGLE_JSON_PATH="${KAGGLE_JSON_PATH:-kaggle.json}"
TASK_NAME="us-patent-phrase-to-phrase-matching"
WORKDIR="${WORKDIR:-$(pwd)}"
CACHE_DIR="/root/.cache/mle-bench/data/${TASK_NAME}/prepared"

# === FUNCTIONS ===

setup_kaggle() {
    echo "[*] Setting up Kaggle..."
    mkdir -p ~/.kaggle
    cp "$KAGGLE_JSON_PATH" ~/.kaggle/kaggle.json
    chmod 600 ~/.kaggle/kaggle.json
}

clone_repos() {
    echo "[*] Cloning repositories..."
    mkdir -p "$WORKDIR"
    cd "$WORKDIR"

    if [[ ! -d mle-bench ]]; then
        git clone https://github.com/bogoconic1/mle-bench.git
        sudo apt update && sudo apt install -y git-lfs
        (cd mle-bench && git lfs fetch --all && git lfs pull && pip install -e .)
    fi
}

prepare_data() {
    echo "[*] Preparing data for task: $TASK_NAME..."
    cd "$WORKDIR/mle-bench"
    mlebench prepare -c "$TASK_NAME"
}

copy_task_data() {
    echo "[*] Copying prepared data into Qgentic-AI/task..."
    mkdir -p "$WORKDIR/Qgentic-AI/task/$TASK_NAME"
    cp -r "$CACHE_DIR/public/"* "$WORKDIR/Qgentic-AI/task/$TASK_NAME"
    cp -r "$CACHE_DIR/private/"* "$WORKDIR/Qgentic-AI/task/$TASK_NAME"
}

main() {
    setup_kaggle
    clone_repos
    prepare_data
    copy_task_data
    echo "[✔] Setup complete."
}

main "$@"
