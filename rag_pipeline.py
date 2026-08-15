"""Reasoning-based tree retrieval and page-grounded Groq answer generation."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from groq import Groq

from pageindex_client import retrieve
from pdf_utils import extract_page_range_text, get_pdf_metadata

NOT_FOUND_MESSAGE = "Not found in document."
MAX_NODES = 3
MAX_PAGES_PER_NODE = 6
MAX_ANSWER_TOKENS = int(os.getenv("GROQ_MAX_TOKENS", "2048"))
LOCAL_DOCUMENT_DIR = Path(".pageindex_uploads")


class GroundingError(RuntimeError):
    """Raised when the local PDF needed for a citation is unavailable."""


def register_local_document(doc_id: str, pdf_path: str | Path) -> Path:
    """Persist the uploaded PDF under its PageIndex id for later chat reruns."""
    source = Path(pdf_path)
    if not source.exists():
        raise FileNotFoundError(f"Source PDF not found: {source}")
    LOCAL_DOCUMENT_DIR.mkdir(exist_ok=True)
    destination = LOCAL_DOCUMENT_DIR / f"{doc_id}.pdf"
    if source.resolve() != destination.resolve():
        destination.write_bytes(source.read_bytes())
    return destination


def answer_question(query: str, doc_id: str) -> dict[str, Any]:
    """Answer a question only when PageIndex-selected PDF pages provide support."""
    pdf_path = _local_pdf_path(doc_id)
    nodes = retrieve(query, doc_id)
    selected_nodes = _select_nodes(query, nodes)
    if not selected_nodes:
        return _not_found_result()

    page_count = int(get_pdf_metadata(pdf_path)["page_count"])
    sources = _build_sources(selected_nodes, nodes, pdf_path, page_count)
    sources = [source for source in sources if source["text"]]
    if not sources:
        return _not_found_result()

    answer = _generate_answer(query, sources)
    if answer == NOT_FOUND_MESSAGE:
        return _not_found_result()

    primary = sources[0]
    return {
        "answer": answer,
        "found": True,
        "section_title": primary["section_title"],
        "clause_number": primary["clause_number"],
        "page_start": primary["page_start"],
        "page_end": primary["page_end"],
        "retrieved_text_snippet": primary["text"],
        "sources": sources,
    }


def _local_pdf_path(doc_id: str) -> Path:
    path = LOCAL_DOCUMENT_DIR / f"{doc_id}.pdf"
    if not path.exists():
        raise GroundingError(
            "The local uploaded PDF is unavailable, so a page-grounded answer cannot be generated."
        )
    return path


MAX_CATALOG_NODES = 40
MAX_TEXT_PER_SOURCE = 3500


def _prune_node_catalog(query: str, nodes: list[dict[str, Any]], max_catalog: int = MAX_CATALOG_NODES) -> list[dict[str, Any]]:
    """Prune tree nodes to prevent exceeding LLM input token limits (e.g. Groq 12,000 TPM limit)."""
    valid_nodes = [
        node for node in nodes
        if node.get("node_id") is not None and node.get("page_start") is not None
    ]
    if len(valid_nodes) <= max_catalog:
        selected_nodes = valid_nodes
    else:
        query_words = set(re.findall(r"\w+", query.lower())) - {
            "what", "is", "the", "a", "an", "and", "or", "in", "of", "for", "to", "on", "with", "which", "how", "when", "does"
        }

        def score_node(node: dict[str, Any]) -> tuple[int, int]:
            title = str(node.get("section_title", "")).lower()
            clause = str(node.get("clause_number", "")).lower()
            content = str(node.get("content", "")).lower()[:300]

            title_score = sum(3 for w in query_words if w in title)
            clause_score = 5 if clause and clause in query.lower() else 0
            content_score = sum(1 for w in query_words if w in content)
            return (title_score + clause_score + content_score, -int(node.get("page_start", 0)))

        scored_nodes = sorted(valid_nodes, key=score_node, reverse=True)
        selected_nodes = scored_nodes[:max_catalog]
        selected_nodes.sort(key=lambda n: int(n.get("page_start", 0)))

    catalog = []
    for node in selected_nodes:
        summary_text = str(node.get("content") or node.get("summary") or "").strip()
        catalog.append({
            "node_id": node.get("node_id"),
            "title": node.get("section_title"),
            "clause_number": node.get("clause_number"),
            "page_index": node.get("page_start"),
            "summary": summary_text[:150],
        })
    return catalog


def _select_nodes(query: str, nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    node_catalog = _prune_node_catalog(query, nodes, max_catalog=MAX_CATALOG_NODES)
    if not node_catalog:
        return []

    prompt = f"""Select up to {MAX_NODES} PageIndex tree nodes that can answer the question.
Use the hierarchy, clause titles, and summaries. Do not select a node unless it is relevant.

Question: {query}

Tree nodes:
{json.dumps(node_catalog, ensure_ascii=True)}

Return JSON only in this form:
{{"node_ids": ["id1", "id2"]}}
Return an empty list when the answer is not supported by the document tree.
"""
    try:
        response = _groq_client().chat.completions.create(
            model=_groq_model(),
            temperature=0,
            max_tokens=256,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": "You perform precise document-tree retrieval."},
                {"role": "user", "content": prompt},
            ],
        )
        payload = _json_object(response.choices[0].message.content or "{}")
        node_ids = payload.get("node_ids", [])
    except Exception as error:
        print(f"[rag_pipeline] Groq node selection API error: {error}. Falling back to top keyword-matched nodes.")
        query_words = set(re.findall(r"\w+", query.lower())) - {"what", "is", "the", "a", "an", "and", "or", "in", "of", "for", "to", "on", "with"}
        scored = []
        for node in nodes:
            if node.get("node_id") is not None and node.get("page_start") is not None:
                title = str(node.get("section_title", "")).lower()
                score = sum(1 for w in query_words if w in title)
                if score > 0:
                    scored.append((score, node))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [node for _, node in scored[:MAX_NODES]]

    if not isinstance(node_ids, list):
        return []
    by_id = {str(node["node_id"]): node for node in nodes if node.get("node_id") is not None}
    return [by_id[str(node_id)] for node_id in node_ids[:MAX_NODES] if str(node_id) in by_id]


def _build_sources(
    selected_nodes: list[dict[str, Any]],
    all_nodes: list[dict[str, Any]],
    pdf_path: Path,
    page_count: int,
) -> list[dict[str, Any]]:
    sources = []
    for node in selected_nodes:
        page_start = node.get("page_start")
        if not isinstance(page_start, int) or page_start < 1:
            continue
        page_end = _infer_page_end(node, all_nodes, page_count)
        text = extract_page_range_text(pdf_path, page_start, page_end)
        if len(text) > MAX_TEXT_PER_SOURCE:
            text = text[:MAX_TEXT_PER_SOURCE] + "\n... [text truncated for token limit]"
        sources.append(
            {
                "section_title": node.get("section_title", "Untitled section"),
                "clause_number": node.get("clause_number", ""),
                "page_start": page_start,
                "page_end": page_end,
                "text": text,
            }
        )
    return sources


def _infer_page_end(node: dict[str, Any], all_nodes: list[dict[str, Any]], page_count: int) -> int:
    """Infer a range because PageIndex trees expose 1-based page_index starts only."""
    page_start = int(node["page_start"])
    later_starts = sorted(
        other["page_start"]
        for other in all_nodes
        if isinstance(other.get("page_start"), int) and other["page_start"] > page_start
    )
    next_start = later_starts[0] if later_starts else page_count + 1
    return min(page_count, page_start + MAX_PAGES_PER_NODE - 1, next_start - 1)


def _generate_answer(query: str, sources: list[dict[str, Any]]) -> str:
    context = "\n\n".join(
        f"SOURCE: {source['section_title']} | clause: {source['clause_number'] or 'not numbered'} | "
        f"PDF pages: {source['page_start']}-{source['page_end']}\n{source['text']}"
        for source in sources
    )
    prompt = f"""Provide a thorough, detailed, and comprehensive answer to the question using only the supplied source text. Include all relevant details, conditions, steps, or explanations found in the source text.
If the source text does not directly support an answer, set supported to false.
Do not use outside knowledge. Do not invent document details.

Question: {query}

Source text:
{context}

Return JSON only: {{"supported": true, "answer": "detailed grounded answer"}}.
"""
    try:
        response = _groq_client().chat.completions.create(
            model=_groq_model(),
            temperature=0,
            max_tokens=MAX_ANSWER_TOKENS,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": "You are Mavenir's Technical RAG Assistant for 3GPP Telecom Specifications and Standards. Answer strictly and only from the provided document text without assuming outside information.",
                },
                {"role": "user", "content": prompt},
            ],
        )
    except Exception as error:
        print(f"[rag_pipeline] Groq answer generation API error ({error}). Retrying with truncated context.")
        truncated_context = context[:4000]
        short_prompt = f"""Provide a clear, grounded answer to the question using only the supplied source text.
If the source text does not directly support an answer, set supported to false.

Question: {query}

Source text:
{truncated_context}

Return JSON only: {{"supported": true, "answer": "detailed grounded answer"}}.
"""
        response = _groq_client().chat.completions.create(
            model=_groq_model(),
            temperature=0,
            max_tokens=1024,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": "You are Mavenir's Technical RAG Assistant for 3GPP Telecom Specifications and Standards. Answer strictly and only from the provided document text without assuming outside information.",
                },
                {"role": "user", "content": short_prompt},
            ],
        )
    payload = _json_object(response.choices[0].message.content or "{}")
    answer = str(payload.get("answer", "")).strip()
    return answer if payload.get("supported") is True and answer else NOT_FOUND_MESSAGE


def _groq_client() -> Groq:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise GroundingError("GROQ_API_KEY is not set. Add it to your .env file.")
    return Groq(api_key=api_key)


def _groq_model() -> str:
    return os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")


def _json_object(content: str) -> dict[str, Any]:
    match = re.search(r"\{.*\}", content, flags=re.DOTALL)
    if not match:
        return {}
    try:
        value = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def _not_found_result() -> dict[str, Any]:
    return {
        "answer": NOT_FOUND_MESSAGE,
        "found": False,
        "section_title": "",
        "clause_number": "",
        "page_start": None,
        "page_end": None,
        "retrieved_text_snippet": "",
        "sources": [],
    }