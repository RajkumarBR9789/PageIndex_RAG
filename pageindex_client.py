"""Small REST wrapper for PageIndex document processing and tree retrieval."""

from __future__ import annotations

import os
import time
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import requests

API_BASE_URL = "https://api.pageindex.ai"
POLL_INTERVAL_SECONDS = 2
DEFAULT_TIMEOUT_SECONDS = 60


class PageIndexError(RuntimeError):
    """Raised when PageIndex cannot process or return a document."""


def _api_key() -> str:
    api_key = os.getenv("PAGEINDEX_API_KEY")
    if not api_key:
        raise PageIndexError("PAGEINDEX_API_KEY is not set. Add it to your .env file.")
    return api_key


def _headers() -> dict[str, str]:
    # PageIndex REST APIs use the api_key header, not Bearer authentication.
    return {"api_key": _api_key()}


def upload_document(pdf_path_or_bytes: str | Path | bytes, filename: str = "document.pdf") -> str:
    """Upload a PDF and return the PageIndex document identifier."""
    if isinstance(pdf_path_or_bytes, (str, Path)):
        file_path = Path(pdf_path_or_bytes)
        if not file_path.exists():
            raise FileNotFoundError(f"PDF not found: {file_path}")
        with file_path.open("rb") as pdf_file:
            response = requests.post(
                f"{API_BASE_URL}/doc/",
                headers=_headers(),
                files={"file": (file_path.name, pdf_file, "application/pdf")},
                timeout=DEFAULT_TIMEOUT_SECONDS,
            )
    else:
        response = requests.post(
            f"{API_BASE_URL}/doc/",
            headers=_headers(),
            files={"file": (filename, pdf_path_or_bytes, "application/pdf")},
            timeout=DEFAULT_TIMEOUT_SECONDS,
        )

    _raise_for_response(response)
    doc_id = response.json().get("doc_id")
    if not doc_id:
        raise PageIndexError("PageIndex upload response did not include doc_id.")
    return str(doc_id)


def get_tree(
    doc_id: str,
    *,
    poll: bool = True,
    status_callback: Callable[[str], None] | None = None,
    poll_interval_seconds: float = POLL_INTERVAL_SECONDS,
) -> dict[str, Any]:
    """Return the completed PageIndex tree, optionally polling while it builds."""
    while True:
        response = requests.get(
            f"{API_BASE_URL}/doc/{doc_id}/",
            headers=_headers(),
            params={"type": "tree", "summary": "true"},
            timeout=DEFAULT_TIMEOUT_SECONDS,
        )
        _raise_for_response(response)
        result = response.json()
        status = str(result.get("status", "unknown"))
        if status_callback:
            status_callback(status)

        if status == "completed" and result.get("result") is not None:
            return result
        if status == "failed":
            raise PageIndexError("PageIndex failed to process this document.")
        if not poll:
            return result
        time.sleep(poll_interval_seconds)


def retrieve(query: str, doc_id: str) -> list[dict[str, Any]]:
    """Return candidate nodes from the completed PageIndex tree.

    The current hosted API exposes the tree but not a standalone raw retrieval
    endpoint for bring-your-own LLMs. Node ranking belongs in rag_pipeline,
    where Groq reasons over this returned hierarchy.
    """
    tree_response = get_tree(doc_id, poll=False)
    if tree_response.get("status") != "completed":
        raise PageIndexError("Document is not ready for retrieval.")
    nodes = list(_flatten_tree(tree_response.get("result", [])))
    if not nodes:
        raise PageIndexError("PageIndex returned an empty document tree.")
    return nodes


def _flatten_tree(tree: Iterable[dict[str, Any]]) -> Iterable[dict[str, Any]]:
    for raw_node in tree:
        node = dict(raw_node)
        node.pop("nodes", None)
        title = str(node.get("title", "Untitled section"))
        node["section_title"] = title
        node["clause_number"] = _clause_number(title)
        node["page_start"] = _page_number(node.get("page_index"))
        node["content"] = str(node.get("text") or node.get("summary") or "")
        yield node
        children = raw_node.get("nodes", [])
        if isinstance(children, list):
            yield from _flatten_tree(children)


def _page_number(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _clause_number(title: str) -> str:
    first_token = title.strip().split(maxsplit=1)[0] if title.strip() else ""
    return first_token.rstrip(".:") if first_token[:1].isdigit() else ""


def _raise_for_response(response: requests.Response) -> None:
    try:
        response.raise_for_status()
    except requests.HTTPError as error:
        message = response.text[:500]
        raise PageIndexError(f"PageIndex API request failed: {message}") from error