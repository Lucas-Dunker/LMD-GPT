#!/usr/bin/env bash
# LMD-GPT environment setup for WSL2 + NVIDIA GPU
set -e

echo "=== LMD-GPT Setup (WSL2 + CUDA) ==="
echo ""

# ── 1. Ollama ──────────────────────────────────────────────────────────
if ! command -v ollama &>/dev/null; then
    echo "[1/5] Installing Ollama…"
    curl -fsSL https://ollama.com/install.sh | sh
else
    echo "[1/5] Ollama already installed ($(ollama --version))"
fi

# Start Ollama in background if not already running
if ! pgrep -x ollama &>/dev/null; then
    echo "      Starting Ollama server in background…"
    nohup ollama serve &>/dev/null &
    sleep 3
fi

# ── 2. Pull models ─────────────────────────────────────────────────────
echo ""
echo "[2/5] Pulling Ollama models (this will take a while on first run)…"
ollama pull llama3.2
ollama pull nomic-embed-text

# ── 3. PyTorch with CUDA ───────────────────────────────────────────────
echo ""
echo "[3/5] Installing PyTorch with CUDA 12.1 support…"
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# ── 4. Python dependencies ─────────────────────────────────────────────
echo ""
echo "[4/5] Installing Python dependencies…"
pip install \
    ollama \
    chromadb \
    python-docx \
    pdfplumber \
    icalendar \
    pyyaml \
    transformers \
    peft \
    datasets \
    accelerate \
    "bitsandbytes>=0.43.0" --prefer-binary

# ── 5. Data directories ────────────────────────────────────────────────
echo ""
echo "[5/5] Creating data directories…"
mkdir -p data/raw/{obsidian,discord,gdocs,gcal}
mkdir -p data/processed
mkdir -p models/{lora,merged}
mkdir -p vectorstore

# ── Done ───────────────────────────────────────────────────────────────
echo ""
echo "=== Setup complete! ==="
echo ""
echo "Next steps:"
echo "  1. Drop your Obsidian vault into:           data/raw/obsidian/"
echo "  2. Drop DiscordChatExporter JSON files into: data/raw/discord/"
echo "  3. Drop downloaded Google Docs into:         data/raw/gdocs/"
echo "  4. Export Google Calendar as .ics into:      data/raw/gcal/"
echo ""
echo "  5. Set your Discord identity (recommended):"
echo "       export DISCORD_USER_ID='your_numeric_id'"
echo "       export DISCORD_USERNAME='YourName'"
echo "     Or edit these directly in config.py."
echo ""
echo "  6. Verify everything is working:"
echo "       python scripts/check_gpu.py"
echo ""
echo "  7. Ingest your data and start chatting:"
echo "       python cli.py ingest"
echo "       python cli.py chat"
