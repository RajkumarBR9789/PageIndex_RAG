"""Streamlit UI for Mavenir PageIndex RAG over technical PDF documents."""

from __future__ import annotations

import hashlib
import io
import os
import re
import tempfile
from pathlib import Path
from typing import Any

import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader

from pageindex_client import PageIndexError, get_tree, upload_document
from pdf_utils import get_pdf_metadata
from rag_pipeline import GroundingError, answer_question, register_local_document

load_dotenv()

st.set_page_config(page_title="Mavenir 3GPP RAG Assistant", page_icon="📡", layout="wide")
st.title("Mavenir PageIndex RAG — 3GPP Standards Assistant")
st.caption("Tree-based PageIndex retrieval & page-grounded Groq answers for 3GPP Telecom Specifications with zero hallucinations.")


def initialize_state() -> None:
    defaults = {
        "doc_id": None,
        "tree": None,
        "metadata": None,
        "upload_hash": None,
        "chat_history": [],
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def build_index(uploaded_file: Any) -> None:
    pdf_bytes = uploaded_file.getvalue()
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temporary_file:
        temporary_file.write(pdf_bytes)
        temporary_path = Path(temporary_file.name)

    try:
        with st.status("Uploading PDF to PageIndex...", expanded=True) as status:
            doc_id = upload_document(temporary_path)
            status.write(f"Uploaded. PageIndex document id: `{doc_id}`")

            def update_status(processing_status: str) -> None:
                status.update(
                    label=f"Building tree index: {processing_status}",
                    state="running",
                    expanded=True,
                )

            tree_response = get_tree(doc_id, status_callback=update_status)
            status.update(label="PageIndex tree index ready", state="complete", expanded=False)

        local_pdf = register_local_document(doc_id, temporary_path)
        st.session_state.doc_id = doc_id
        st.session_state.tree = tree_response["result"]
        st.session_state.metadata = get_pdf_metadata(local_pdf)
        st.session_state.chat_history = []
        st.success("Index built. Inspect the extracted structure below, then ask a question.")
    finally:
        temporary_path.unlink(missing_ok=True)


def clause_number(title: str) -> str:
    match = re.match(r"^(\d+(?:\.\d+)*)", title.strip())
    return match.group(1) if match else ""


def display_tree(tree: list[dict[str, Any]], page_count: int) -> None:
    starts = sorted(
        {
            int(node["page_index"])
            for node in flatten_tree(tree)
            if isinstance(node.get("page_index"), int) and node["page_index"] > 0
        }
    )

    def page_range(node: dict[str, Any]) -> str:
        page_start = node.get("page_index")
        if not isinstance(page_start, int):
            return "page unavailable"
        later_starts = [start for start in starts if start > page_start]
        page_end = min(page_count, later_starts[0] - 1 if later_starts else page_count)
        return f"PDF pages {page_start}-{page_end}"

    def render(nodes: list[dict[str, Any]]) -> None:
        for node in nodes:
            title = str(node.get("title", "Untitled section"))
            clause = clause_number(title)
            label = f"{clause + ' - ' if clause else ''}{title} ({page_range(node)})"
            with st.expander(label, expanded=False):
                st.caption(f"Node ID: {node.get('node_id', 'not provided')}")
                summary = node.get("summary") or node.get("text")
                if summary:
                    st.write(str(summary))
                children = node.get("nodes", [])
                if isinstance(children, list) and children:
                    render(children)

    render(tree)


def flatten_tree(tree: list[dict[str, Any]]) -> list[dict[str, Any]]:
    nodes: list[dict[str, Any]] = []
    for node in tree:
        nodes.append(node)
        children = node.get("nodes", [])
        if isinstance(children, list):
            nodes.extend(flatten_tree(children))
    return nodes


def render_answer(result: dict[str, Any]) -> None:
    st.write(result["answer"])
    if not result["found"]:
        return
    for source in result["sources"]:
        clause = source["clause_number"] or "not numbered"
        st.caption(
            f"Source: {source['section_title']} | Clause/Section: {clause} | "
            f"PDF pages: {source['page_start']}-{source['page_end']}"
        )
    with st.expander("View retrieved context"):
        for source in result["sources"]:
            st.markdown(
                f"**{source['section_title']} - PDF pages "
                f"{source['page_start']}-{source['page_end']}**"
            )
            st.text(source["text"])


initialize_state()
uploaded_file = st.file_uploader("Upload a PDF document (3GPP Specification / Technical Standard)", type=["pdf"])

if uploaded_file:
    current_hash = hashlib.sha256(uploaded_file.getvalue()).hexdigest()
    if st.session_state.upload_hash and st.session_state.upload_hash != current_hash:
        st.session_state.doc_id = None
        st.session_state.tree = None
        st.session_state.metadata = None
        st.session_state.chat_history = []
    st.session_state.upload_hash = current_hash

    # Page-count check
    try:
        pdf_reader = PdfReader(io.BytesIO(uploaded_file.getvalue()))
        page_count = len(pdf_reader.pages)
        if page_count > 200:
            st.warning(
                f"This document contains {page_count} pages. To ensure fast indexing and stay within "
                "API usage limits, consider uploading a smaller subset or splitting the document by section."
            )
        else:
            st.success(f"Document uploaded successfully ({page_count} pages).")
    except Exception:
        pass

    if st.button("Build Index", type="primary", disabled=not os.getenv("PAGEINDEX_API_KEY")):
        try:
            build_index(uploaded_file)
        except (PageIndexError, GroundingError, OSError) as error:
            st.error(str(error))
else:
    st.info("Upload a PDF to begin.")

if st.session_state.tree:
    metadata = st.session_state.metadata or {}
    st.subheader("Extracted PageIndex Tree")
    st.caption(f"{metadata.get('filename', '')} | {metadata.get('page_count', '?')} PDF pages")
    display_tree(st.session_state.tree, int(metadata.get("page_count", 1)))

if st.session_state.doc_id:
    st.subheader("Ask the Document")
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant":
                render_answer(message["result"])
            else:
                st.write(message["content"])

    question = st.chat_input("Ask a question about the uploaded document")
    if question:
        st.session_state.chat_history.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.write(question)
        with st.chat_message("assistant"):
            with st.spinner("Reasoning over the PageIndex tree and grounding in source pages..."):
                try:
                    result = answer_question(question, st.session_state.doc_id)
                    render_answer(result)
                    st.session_state.chat_history.append({"role": "assistant", "result": result})
                except (PageIndexError, GroundingError, OSError) as error:
                    st.error(str(error))

if not os.getenv("PAGEINDEX_API_KEY"):
    st.warning("PAGEINDEX_API_KEY is missing. Add it to .env before building an index.")