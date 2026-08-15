"""Run evaluation test suite against an uploaded PDF document."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv

from pageindex_client import get_tree, upload_document
from rag_pipeline import answer_question, register_local_document

# Evaluation questions below test standard RAG accuracy against uploaded specification PDFs.
QUESTIONS = [
    "What information is included in a tracking area update request?",
    "When shall a UE perform a normal tracking area updating procedure?",
    "What is the purpose of a periodic tracking area update?",
    "What actions does the UE take after receiving a tracking area update reject?",
    "Which EMM states are defined for a UE?",
    "What is the difference between EMM-DEREGISTERED and EMM-REGISTERED?",
    "When does a UE enter EMM-DEREGISTERED.ATTEMPTING-TO-ATTACH?",
    "What initiates the EPS authentication and key agreement procedure?",
    "What shall the UE do when the authentication token is invalid?",
    "What is the purpose of the AUTHENTICATION REJECT message?",
    "What is timer T3416 used for?",
    "What is timer T3418 used for?",
    "What is timer T3420 used for?",
    "When is timer T3410 started and stopped?",
    "What happens when timer T3411 expires?",
    "What NAS security context is used after successful authentication?",
    "Under what conditions is a service request procedure initiated?",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate answers on any 3GPP specification PDF using this RAG pipeline."
    )
    parser.add_argument("--pdf", required=True, help="Path to the PDF document.")
    parser.add_argument("--doc-id", help="Existing PageIndex document ID; omit to upload and index.")
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()
    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    doc_id = args.doc_id
    if not doc_id:
        print("Uploading PDF and waiting for the PageIndex tree...")
        doc_id = upload_document(pdf_path)
        get_tree(doc_id, poll=True, status_callback=lambda status: print(f"Status: {status}"))
    register_local_document(doc_id, pdf_path)

    print("| # | Question | Result | Clause | Pages | Manual label |")
    print("| --- | --- | --- | --- | --- | --- |")
    for number, question in enumerate(QUESTIONS, start=1):
        try:
            result = answer_question(question, doc_id)
            result_text = result["answer"].replace("|", "\\|").replace("\n", " ")
            pages = (
                f"{result['page_start']}-{result['page_end']}" if result["found"] else "-"
            )
            clause = result["clause_number"] or "-"
        except Exception as error:  # Keep the remaining manual test cases running.
            result_text = f"ERROR: {error}".replace("|", "\\|")
            pages = "-"
            clause = "-"
        print(f"| {number} | {question} | {result_text} | {clause} | {pages} |  |")


if __name__ == "__main__":
    main()