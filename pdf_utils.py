"""Local PDF utilities used to ground answers in original document pages."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pypdf import PdfReader


def get_pdf_metadata(pdf_path: str | Path) -> dict[str, Any]:
    """Return simple, display-friendly metadata for a source PDF."""
    path = Path(pdf_path)
    reader = PdfReader(path)
    metadata = reader.metadata or {}
    return {
        "filename": path.name,
        "page_count": len(reader.pages),
        "title": metadata.title or "",
        "author": metadata.author or "",
    }


def extract_page_range_text(pdf_path: str | Path, page_start: int, page_end: int) -> str:
    """Extract text from an inclusive, 1-based page range."""
    reader = PdfReader(Path(pdf_path))
    if not reader.pages:
        return ""

    first_page = max(1, page_start)
    last_page = min(len(reader.pages), max(first_page, page_end))
    pages = []
    for page_number in range(first_page, last_page + 1):
        text = reader.pages[page_number - 1].extract_text() or ""
        pages.append(f"[PDF page {page_number}]\n{text}")
    return "\n\n".join(pages).strip()