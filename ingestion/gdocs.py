"""Parse downloaded Google Docs files (.docx, .txt, .md, .pdf)."""
from pathlib import Path
from typing import Generator

from config import GDOCS_DIR

SUPPORTED = {".docx", ".txt", ".md", ".pdf"}


def _read_docx(path: Path) -> str:
    import docx
    doc = docx.Document(str(path))
    return "\n".join(p.text for p in doc.paragraphs if p.text.strip())


def _read_pdf(path: Path) -> str:
    import pdfplumber
    parts = []
    with pdfplumber.open(str(path)) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                parts.append(text)
    return "\n".join(parts)


def iter_docs(docs_dir: Path = GDOCS_DIR) -> Generator[dict, None, None]:
    for path in sorted(docs_dir.rglob("*")):
        if path.suffix.lower() not in SUPPORTED:
            continue
        try:
            if path.suffix.lower() == ".docx":
                text = _read_docx(path)
            elif path.suffix.lower() == ".pdf":
                text = _read_pdf(path)
            else:
                text = path.read_text(encoding="utf-8")
        except Exception as e:
            print(f"Warning: could not read {path.name}: {e}")
            continue

        if not text.strip():
            continue

        yield {
            "text": text,
            "metadata": {
                "source": "gdocs",
                "file": path.name,
                "type": path.suffix.lstrip(".").lower(),
            },
        }


def load_all(docs_dir: Path = GDOCS_DIR) -> list[dict]:
    return list(iter_docs(docs_dir))
