#!/usr/bin/env bash
# AudioLayer Studio — one-step installer for macOS and Linux.
#   ./install.sh            full install (AI models, ~2.5 GB incl. PyTorch)
#   ./install.sh --lite     no PyTorch (detection-only, low accuracy)
#   ./install.sh --yes      don't ask questions (connects Claude Desktop with ~/AudioLayer/inputs)
set -euo pipefail
cd "$(dirname "$0")"

LITE=0; YES=0
for a in "$@"; do
  case "$a" in
    --lite) LITE=1 ;;
    --yes|-y) YES=1 ;;
    *) echo "unknown option $a"; exit 2 ;;
  esac
done

say() { printf "\n\033[1;33m==>\033[0m %s\n" "$*"; }
die() { printf "\n\033[1;31mError:\033[0m %s\n" "$*"; exit 1; }
ask() { [ "$YES" = 1 ] && return 0; read -r -p "$1 [Y/n] " r; [[ -z "$r" || "$r" =~ ^[Yy] ]]; }

OS="$(uname -s)"

# --- Python ------------------------------------------------------------------------
PY=""
for c in python3.13 python3.12 python3.11 python3; do
  if command -v "$c" >/dev/null 2>&1 && "$c" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)'; then
    PY="$c"; break
  fi
done
if [ -z "$PY" ]; then
  if [ "$OS" = "Darwin" ] && command -v brew >/dev/null && ask "Python 3.11+ not found. Install with Homebrew?"; then
    brew install python@3.12 && PY=python3.12
  else
    die "Python 3.11+ is required. macOS: brew install python@3.12 · Ubuntu: sudo apt install python3 python3-venv"
  fi
fi
say "Using $($PY --version)"

# --- FFmpeg ------------------------------------------------------------------------
if ! command -v ffmpeg >/dev/null || ! command -v ffprobe >/dev/null; then
  if [ "$OS" = "Darwin" ] && command -v brew >/dev/null && ask "FFmpeg not found. Install with Homebrew?"; then
    brew install ffmpeg
  elif command -v apt-get >/dev/null && ask "FFmpeg not found. Install with apt (needs sudo)?"; then
    sudo apt-get update && sudo apt-get install -y ffmpeg libsndfile1
  else
    die "FFmpeg is required: https://ffmpeg.org/download.html (macOS: brew install ffmpeg)"
  fi
fi
say "FFmpeg: $(command -v ffmpeg)"

# --- virtual environment --------------------------------------------------------------
if [ ! -d .venv ]; then
  say "Creating virtual environment (.venv)"
  "$PY" -m venv .venv || die "Could not create a venv (Ubuntu: sudo apt install python3-venv)"
fi
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install -q --upgrade pip

if [ "$LITE" = 1 ]; then
  say "Installing AudioLayer (lite, no PyTorch)"
  pip install -q -e .
else
  if ! python -c "import torch" 2>/dev/null; then
    say "Installing PyTorch (this is the big download)"
    if [ "$OS" = "Darwin" ]; then
      pip install -q torch
    elif command -v nvidia-smi >/dev/null 2>&1; then
      pip install -q torch            # default wheels include CUDA
    else
      pip install -q torch --index-url https://download.pytorch.org/whl/cpu
    fi
  fi
  say "Installing AudioLayer + AI models packages"
  pip install -q -e ".[ml]"
  say "Downloading model weights (~550 MB, once)"
  audiolayer models download || echo "Model download failed — it will be retried automatically on first use."
fi

# --- web UI (prebuilt copy ships in the package; rebuild only if Node is available) ----------
if command -v npm >/dev/null 2>&1 && [ ! -f frontend/dist/index.html ]; then
  say "Building the web UI"
  (cd frontend && npm ci --no-audit --no-fund --silent && npm run build --silent) || echo "UI build skipped (using the prebuilt UI)."
fi

# --- launcher ------------------------------------------------------------------------
LAUNCH="AudioLayer.command"
cat > "$LAUNCH" <<EOF
#!/usr/bin/env bash
cd "$(pwd)"
source .venv/bin/activate
audiolayer web
EOF
chmod +x "$LAUNCH"

# --- Claude -------------------------------------------------------------------------
INPUTS="$HOME/AudioLayer/inputs"
if ask "Connect AudioLayer to Claude Desktop (Claude may read media from $INPUTS)?"; then
  CODE=""
  command -v claude >/dev/null 2>&1 && ask "Also connect Claude Code?" && CODE="--code"
  audiolayer setup-claude --input-dir "$INPUTS" $CODE
fi

audiolayer doctor || true
cat <<EOF

✅ Done!
   Start the app:   double-click $(pwd)/$LAUNCH   (or: source .venv/bin/activate && audiolayer web)
   One command:     audiolayer clean my_clip.mp4 --preset remove_music --loudness youtube --video
   Claude:          put clips in $INPUTS, restart Claude Desktop, and ask it to clean them.
EOF
