# LMD-GPT

A fully local personal AI built from your own data. Two capabilities:

- **RAG** — ask questions about yourself (notes, calendar, docs, messages)
- **Style fine-tuning** — QLoRA-train a model to write like you

Everything runs locally. No cloud APIs.

My personal version is built off of yours truly 😉, but this setup can work for anyone with their own data. Sorry, I won't be sharing my clone, too much of a safety risk :)

---

## Requirements

- NVIDIA GPU (tested on GTX 4070, 12 GB VRAM)
- WSL2 (Ubuntu) or native Linux
- Python 3.10+
- [Ollama](https://ollama.com) (installed by `setup.sh`)

---

## Setup

```bash
bash setup.sh
```

This installs Ollama, pulls `llama3.2:3b` and `nomic-embed-text`, installs all Python deps, and creates the data directories.

Then tell it who you are (used to filter your Discord messages and name training prompts):

```bash
export DISCORD_USER_ID="your_numeric_discord_id"
export DISCORD_USERNAME="YourUsername"
export YOUR_NAME="Your Name"
```

Or set these directly in `config.py`.

Verify everything is working:

```bash
python scripts/check_gpu.py
```

---

## Data

Drop your exports into the appropriate folders before ingesting:

| Source | Format | Folder |
|---|---|---|
| Obsidian | Markdown vault directory | `data/raw/obsidian/` |
| Discord | [DiscordChatExporter](https://github.com/Tyrrrz/DiscordChatExporter) JSON | `data/raw/discord/` |
| Google Docs | Downloaded as `.docx`, `.txt`, `.md`, or `.pdf` | `data/raw/gdocs/` |
| Google Calendar | Exported as `.ics` | `data/raw/gcal/` |

> All raw data and model weights are gitignored — nothing personal ever leaves your machine.

---

## Usage

### Ingest

Embed all your data into the local vector store:

```bash
python cli.py ingest
```

Or ingest specific sources:

```bash
python cli.py ingest --sources obsidian discord
```

### Chat (RAG)

```bash
python cli.py chat
```

Ask questions like:
- *"What projects was I working on in early 2024?"*
- *"What did I write about anxiety in my notes?"*
- *"What meetings did I have with Sarah?"*

Restrict retrieval to one source:

```bash
python cli.py chat --source obsidian
```

### Fine-tune

Train a model to write like you, then use it for both chat and RAG.

```bash
# 1. Build training data from Discord + notes
python cli.py finetune prepare

# 2. QLoRA train (~2–6 hours on a 4070 depending on data size)
python cli.py finetune train

# 3. Merge weights and get Ollama instructions
python cli.py finetune export
```

After export, follow the printed instructions to convert the merged model to GGUF and register it with Ollama. The GGUF conversion is a two-step process — `convert_hf_to_gguf.py` does not support `q4_k_m` directly:

**Step 1 — Build llama.cpp** (once):
```bash
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
pip install -r requirements.txt
cmake -B build && cmake --build build --config Release -j $(nproc)
# llama-quantize binary will be at build/bin/llama-quantize
```

**Step 2a — Fix the tokenizer config** (if you see a `TokenizersBackend` error):

The merged model's `tokenizer_config.json` may reference a non-standard tokenizer class. Fix it before converting:
```bash
python3 -c "
import json
path = 'models/merged/tokenizer_config.json'
d = json.load(open(path))
d['tokenizer_class'] = 'PreTrainedTokenizerFast'
[d.pop(k, None) for k in ['backend', 'is_local', 'max_length', 'stride', 'truncation_side', 'truncation_strategy']]
json.dump(d, open(path, 'w'), indent=2)
"
```

**Step 2b — Convert to F16 GGUF**:
```bash
python convert_hf_to_gguf.py models/merged \
    --outfile models/merged/lmd-gpt-f16.gguf \
    --outtype f16
```

**Step 2c — Quantize to Q4_K_M**:
```bash
build/bin/llama-quantize \
    models/merged/lmd-gpt-f16.gguf \
    models/merged/lmd-gpt.gguf \
    Q4_K_M
```

**Steps 3–5 — Register with Ollama**:
```bash
echo 'FROM models/merged/lmd-gpt.gguf' > Modelfile
ollama create lmd-gpt -f Modelfile
```

Then update `config.py`:
```python
INFERENCE_MODEL = "lmd-gpt"
```

From that point, `python cli.py chat` uses your fine-tuned model with RAG.

Resume an interrupted training run:

```bash
python cli.py finetune train --resume models/lora/checkpoint-200
```

### Status

```bash
python cli.py status
```

---

## Project Structure

```
cli.py                    entry point
config.py                 all settings

ingestion/
  obsidian.py             parses Obsidian vault (.md + YAML frontmatter)
  discord.py              parses DiscordChatExporter JSON
  gdocs.py                parses .docx / .txt / .md / .pdf
  gcal.py                 parses .ics calendar exports

embeddings/
  chunker.py              paragraph-aware text chunking
  store.py                ChromaDB wrapper (embed, upsert, query)

rag/
  retriever.py            vector similarity search + context formatting
  chain.py                streams Ollama responses with RAG context

finetune/
  prepare_data.py         builds ChatML JSONL from conversations + notes
  train.py                QLoRA training (4-bit, ~12 GB VRAM)
  export.py               merges LoRA weights, prints GGUF/Ollama steps

scripts/
  check_gpu.py            validates CUDA, Ollama, and model availability
```

---

## Configuration

All tunables are in `config.py`. Key settings:

| Setting | Default | Description |
|---|---|---|
| `INFERENCE_MODEL` | `llama3.2:3b` | Ollama model for chat |
| `EMBEDDING_MODEL` | `nomic-embed-text` | Ollama model for embeddings |
| `TOP_K_RESULTS` | `5` | Chunks retrieved per query |
| `CHUNK_SIZE` | `512` | Words per chunk |
| `BASE_MODEL` | `meta-llama/Llama-3.2-3B-Instruct` | HuggingFace base for fine-tuning |
| `LORA_RANK` | `16` | LoRA rank (higher = more capacity) |
| `BATCH_SIZE` | `2` | Per-device batch (conservative for 12 GB) |
| `NUM_EPOCHS` | `3` | Training epochs |

---

## WSL2 Notes

- Ollama detects the GPU automatically in WSL2; no extra flags are needed.
- `bitsandbytes` uses a prebuilt wheel (`--prefer-binary` in setup.sh) to avoid compilation issues.
- If `torch.cuda.is_available()` returns `False`, reinstall PyTorch:
  ```bash
  pip install torch --index-url https://download.pytorch.org/whl/cu121
  ```
- Your Windows NVIDIA driver must be recent enough to include the WSL2 CUDA driver (`nvidia-smi` should work inside WSL2).
