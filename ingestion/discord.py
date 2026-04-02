"""Parse DiscordChatExporter JSON exports."""
import json
from pathlib import Path
from typing import Generator

from config import DISCORD_DIR, DISCORD_USER_ID, DISCORD_USERNAME


def _is_user_message(author: dict) -> bool:
    """Return True if this message was authored by the configured user."""
    if not DISCORD_USER_ID and not DISCORD_USERNAME:
        return False  # can't filter without identity config
    if DISCORD_USER_ID and author.get("id") == DISCORD_USER_ID:
        return True
    if DISCORD_USERNAME and author.get("name") == DISCORD_USERNAME:
        return True
    return False


def iter_messages(
    export_dir: Path = DISCORD_DIR,
    user_only: bool = False,
) -> Generator[dict, None, None]:
    """Yield message dicts from all DiscordChatExporter JSON files."""
    for path in sorted(export_dir.rglob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue

        guild = data.get("guild", {}).get("name", "Unknown Server")
        channel = data.get("channel", {}).get("name", "Unknown Channel")

        for msg in data.get("messages", []):
            if msg.get("type") != "Default":
                continue
            content = msg.get("content", "").strip()
            if not content:
                continue

            author = msg.get("author", {})
            if user_only and not _is_user_message(author):
                continue

            yield {
                "text": content,
                "metadata": {
                    "source": "discord",
                    "guild": guild,
                    "channel": channel,
                    "author_id": author.get("id", ""),
                    "author_name": author.get("name", ""),
                    "timestamp": msg.get("timestamp", ""),
                    "message_id": msg.get("id", ""),
                },
            }


def iter_conversations(
    export_dir: Path = DISCORD_DIR,
    context_window: int = 3,
    min_response_words: int = 3,
) -> Generator[dict, None, None]:
    """
    Yield (context, response) pairs for fine-tuning.
    context = last `context_window` messages before the user replied.
    response = the user's message.
    """
    for path in sorted(export_dir.rglob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue

        messages = [
            m for m in data.get("messages", [])
            if m.get("type") == "Default" and m.get("content", "").strip()
        ]

        for i, msg in enumerate(messages):
            if not _is_user_message(msg.get("author", {})):
                continue
            response = msg["content"].strip()
            if len(response.split()) < min_response_words:
                continue

            context_msgs = messages[max(0, i - context_window):i]
            if not context_msgs:
                continue

            context = "\n".join(
                f"{m['author']['name']}: {m['content']}"
                for m in context_msgs
            )

            yield {
                "context": context,
                "response": response,
                "metadata": {
                    "source": "discord",
                    "guild": data.get("guild", {}).get("name", ""),
                    "channel": data.get("channel", {}).get("name", ""),
                    "timestamp": msg.get("timestamp", ""),
                },
            }


def iter_conversation_windows(
    export_dir: Path = DISCORD_DIR,
    window_size: int = 5,
) -> Generator[dict, None, None]:
    """
    Group messages into conversation windows centered on the user's messages.
    Each window includes surrounding context so embeddings capture meaning.
    """
    for path in sorted(export_dir.rglob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue

        guild = data.get("guild", {}).get("name", "Unknown Server")
        channel = data.get("channel", {}).get("name", "Unknown Channel")

        messages = [
            m for m in data.get("messages", [])
            if m.get("type") == "Default" and m.get("content", "").strip()
        ]

        for i, msg in enumerate(messages):
            if not _is_user_message(msg.get("author", {})):
                continue

            # Grab surrounding messages for context
            start = max(0, i - window_size)
            end = min(len(messages), i + window_size + 1)
            window = messages[start:end]

            lines = [
                f"{m['author']['name']}: {m['content'].strip()}"
                for m in window
            ]
            # Cap at ~400 words to stay within nomic-embed-text context limit
            # (code snippets and URLs tokenize to many more tokens than words)
            text = "\n".join(lines)
            words = text.split()
            if len(words) > 400:
                text = " ".join(words[:400])

            yield {
                "text": text,
                "metadata": {
                    "source": "discord",
                    "guild": guild,
                    "channel": channel,
                    "author_id": msg["author"].get("id", ""),
                    "author_name": msg["author"].get("name", ""),
                    "timestamp": msg.get("timestamp", ""),
                    "message_id": msg.get("id", ""),
                },
            }


def load_all(export_dir: Path = DISCORD_DIR) -> list[dict]:
    """Load user's messages grouped into conversation windows for RAG."""
    if not DISCORD_USER_ID and not DISCORD_USERNAME:
        print(
            "Warning: DISCORD_USER_ID and DISCORD_USERNAME are both unset — "
            "no Discord messages will be ingested.\n"
            "Set them in config.py or via environment variables."
        )
        return []
    return list(iter_conversation_windows(export_dir))


def load_user_messages(export_dir: Path = DISCORD_DIR) -> list[dict]:
    """Load only the configured user's messages."""
    return list(iter_messages(export_dir, user_only=True))


def load_conversations(export_dir: Path = DISCORD_DIR) -> list[dict]:
    """Load conversation pairs for fine-tuning."""
    return list(iter_conversations(export_dir))
