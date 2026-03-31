"""Central configuration for LMD-GPT."""
import os
from pathlib import Path

BASE_DIR = Path(__file__).parent

# --- Data paths ---
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
OBSIDIAN_DIR = RAW_DIR / "obsidian"
DISCORD_DIR = RAW_DIR / "discord"
GDOCS_DIR = RAW_DIR / "gdocs"
GCAL_DIR = RAW_DIR / "gcal"
PROCESSED_DIR = DATA_DIR / "processed"

# --- Model outputs ---
MODELS_DIR = BASE_DIR / "models"
LORA_DIR = MODELS_DIR / "lora"
MERGED_DIR = MODELS_DIR / "merged"

# --- Vector store ---
VECTORSTORE_DIR = BASE_DIR / "vectorstore"

# --- Ollama ---
OLLAMA_BASE_URL = "http://localhost:11434"
INFERENCE_MODEL = "lmd-gpt"   # model used for RAG chat
EMBEDDING_MODEL = "nomic-embed-text"

# --- Your identity (used to filter Discord messages + name training prompts) ---
# Set via env vars or edit directly here.
DISCORD_USER_ID = os.getenv("DISCORD_USER_ID", "")
DISCORD_USERNAME = os.getenv("DISCORD_USERNAME", "")
YOUR_NAME = os.getenv("YOUR_NAME", DISCORD_USERNAME or "me")

# --- Chunking ---
CHUNK_SIZE = 512       # words per chunk
CHUNK_OVERLAP = 64     # word overlap between chunks

# --- RAG ---
TOP_K_RESULTS = 5

# --- Fine-tuning ---
BASE_MODEL = "meta-llama/Llama-3.2-3B-Instruct"
LORA_RANK = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LEARNING_RATE = 2e-4
BATCH_SIZE = 2          # conservative for 12GB VRAM
GRAD_ACCUM = 8          # effective batch = 16
NUM_EPOCHS = 3
MAX_SEQ_LENGTH = 2048
