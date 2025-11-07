# For Claude Code
# === Install Node.js (LTS) via NVM ===
# Safe to run multiple times; idempotent setup

if ! command -v nvm &> /dev/null; then
  echo "🚀 nvm not found — installing..."
  curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.3/install.sh | bash
  export NVM_DIR="$HOME/.nvm"
  [ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"
  [ -s "$NVM_DIR/bash_completion" ] && \. "$NVM_DIR/bash_completion"
else
  echo "✅ nvm already installed"
  export NVM_DIR="$HOME/.nvm"
  [ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"
fi

echo "📦 Installing latest LTS Node.js..."
nvm install --lts
nvm use --lts
nvm alias default node

echo "🧾 Verifying installation..."
node -v
npm -v
