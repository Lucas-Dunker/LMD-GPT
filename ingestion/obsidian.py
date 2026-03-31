"""Parse Obsidian vault markdown files."""
from pathlib import Path
from typing import Generator

import yaml

from config import OBSIDIAN_DIR


def _parse_frontmatter(content: str) -> tuple[dict, str]:
    """Extract YAML frontmatter and body. Returns ({}, full_content) if no frontmatter."""
    if not content.startswith("---"):
        return {}, content
    end = content.find("---", 3)
    if end == -1:
        return {}, content
    try:
        frontmatter = yaml.safe_load(content[3:end]) or {}
    except yaml.YAMLError:
        frontmatter = {}
    body = content[end + 3:].strip()
    return frontmatter, body


def iter_notes(vault_dir: Path = OBSIDIAN_DIR) -> Generator[dict, None, None]:
    """Yield document dicts for every non-empty .md file in the vault."""
    for path in sorted(vault_dir.rglob("*.md")):
        try:
            raw = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue

        frontmatter, body = _parse_frontmatter(raw)
        if not body.strip():
            continue

        tags = frontmatter.get("tags", [])
        if isinstance(tags, str):
            tags = [tags]

        yield {
            "text": body,
            "metadata": {
                "source": "obsidian",
                "file": str(path.relative_to(vault_dir)),
                "title": frontmatter.get("title", path.stem),
                "tags": tags,
                "created": str(frontmatter.get("created", "")),
                "modified": str(frontmatter.get("modified", "")),
            },
        }


def load_all(vault_dir: Path = OBSIDIAN_DIR) -> list[dict]:
    return list(iter_notes(vault_dir))
