"""Build a JSONL training dataset from Discord conversations and Obsidian notes."""
import json
import random
from pathlib import Path
from typing import Optional

from config import PROCESSED_DIR, YOUR_NAME

# Llama 3 instruct format (NOT ChatML — Llama 3 uses its own special tokens)
_TEMPLATE = (
    "<|begin_of_text|>"
    "<|start_header_id|>system<|end_header_id|>\n\n"
    "You are {name}. Respond naturally in their voice and style."
    "<|eot_id|>"
    "<|start_header_id|>user<|end_header_id|>\n\n"
    "{context}"
    "<|eot_id|>"
    "<|start_header_id|>assistant<|end_header_id|>\n\n"
    "{response}"
    "<|eot_id|>"
)


def _discord_samples(conversations: list[dict], name: str) -> list[dict]:
    samples = []
    for conv in conversations:
        response = conv["response"].strip()
        context = conv["context"].strip()
        if not context or not response:
            continue
        samples.append({
            "text": _TEMPLATE.format(name=name, context=context, response=response),
            "source": "discord",
        })
    return samples


def _note_samples(notes: list[dict], name: str) -> list[dict]:
    """Turn notes into 'write about X' → note-body pairs."""
    samples = []
    for note in notes:
        body = note["text"].strip()
        title = note["metadata"].get("title", "")
        # Only use notes with meaningful content
        if len(body.split()) < 30:
            continue
        # Cap response length at ~400 words to keep training examples focused
        words = body.split()
        response = " ".join(words[:400])
        samples.append({
            "text": _TEMPLATE.format(
                name=name,
                context=f"Write about: {title}",
                response=response,
            ),
            "source": "obsidian",
        })
    return samples


def prepare(
    username: Optional[str] = None,
    output_path: Optional[Path] = None,
    seed: int = 42,
) -> Path:
    from ingestion.discord import load_conversations
    from ingestion.obsidian import load_all as load_obsidian

    name = username or YOUR_NAME
    output_path = output_path or PROCESSED_DIR / "training_data.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    samples: list[dict] = []

    print("Loading Discord conversations...")
    convs = load_conversations()
    discord = _discord_samples(convs, name)
    samples.extend(discord)
    print(f"  {len(discord):,} samples from {len(convs):,} conversations")

    print("Loading Obsidian notes...")
    notes = load_obsidian()
    note_s = _note_samples(notes, name)
    samples.extend(note_s)
    print(f"  {len(note_s):,} samples from {len(notes):,} notes")

    random.seed(seed)
    random.shuffle(samples)

    with open(output_path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"\nTotal: {len(samples):,} training samples → {output_path}")
    return output_path
