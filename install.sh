#!/usr/bin/env bash
set -euo pipefail

# === CONFIG ===
GH_TOKEN="<your github token"  # optionally export GH_TOKEN before running
KAGGLE_JSON_PATH="${KAGGLE_JSON_PATH:-kaggle.json}"
TASK_NAME="us-patent-phrase-to-phrase-matching"
WORKDIR="${WORKDIR:-$(pwd)}"
CACHE_DIR="/root/.cache/mle-bench/data/${TASK_NAME}/prepared"

# === FUNCTIONS ===

install_github_cli() {
    echo "[*] Installing GitHub CLI..."
    curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | \
        sudo dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] \
        https://cli.github.com/packages stable main" | \
        sudo tee /etc/apt/sources.list.d/github-cli.list > /dev/null
    sudo apt update
    sudo apt install -y gh
}

github_auth() {
    echo "[*] Authenticating GitHub..."
    if [[ -n "$GH_TOKEN" ]]; then
        echo "$GH_TOKEN" | gh auth login --with-token
    else
        gh auth login
    fi
}

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

    if [[ ! -d gstar-project ]]; then
        git clone https://github.com/bogoconic1/gstar-project.git
        (cd gstar-project && pip install -r requirements.txt)
    fi

    if [[ ! -d mle-bench ]]; then
        git clone https://github.com/openai/mle-bench.git
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
    echo "[*] Copying prepared data into gstar-project/task..."
    mkdir -p "$WORKDIR/gstar-project/task/$TASK_NAME"
    cp -r "$CACHE_DIR/public/"* "$WORKDIR/gstar-project/task/$TASK_NAME"
    cp -r "$CACHE_DIR/private/"* "$WORKDIR/gstar-project/task/$TASK_NAME"
}

main() {
    install_github_cli
    github_auth
    setup_kaggle
    clone_repos
    prepare_data
    copy_task_data
    echo "[✔] Setup complete."
}

main "$@"
