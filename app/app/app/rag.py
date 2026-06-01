"""
app/rag.py — Ingest, Retrieve, Generate pipeline for ResearchPie.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import IO, List, Tuple

import ollama
import pdfplumber
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DEFAULT_CHAT_MODEL,
    EMBEDDING_MODEL,
    OLLAMA_HOST,
    RETRIEVAL_TOP_K,
    SUPPORTED_EXTENSIONS,
    VECTOR_STORE_DIR,
)

logger = logging.getLogger(__name__)


def _embeddings() -> OllamaEmbeddings:
    return OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_HOST)


def _splitter(chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )


def extract_model_names() -> List[str]:
    """Return locally available Ollama chat models."""
    try:
        client = ollama.Client(host=OLLAMA_HOST)
        models = client.list().get("models", [])
        names = [m["name"] for m in models if "embed" not in m["name"].lower()]
        return names or [DEFAULT_CHAT_MODEL]
    except Exception as exc:
        logger.warning("Could not list Ollama models: %s", exc)
        return [DEFAULT_CHAT_MODEL]


def _read_pdf(file: IO[bytes], filename: str) -> List[Document]:
    docs: List[Document] = []
    with pdfplumber.open(file) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            text = (page.extract_text() or "").strip()
            if text:
                docs.append(
                    Document(
                        page_content=text,
                        metadata={"source": filename, "page": i},
                    )
                )
    return docs


def _read_text(file: IO[bytes], filename: str, ext: str) -> List[Document]:
    raw = file.read()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        text = raw.decode("latin-1")

    if ext == ".json":
        try:
            data = json.loads(text)
            text = json.dumps(data, indent=2, ensure_ascii=False)
        except json.JSONDecodeError:
            pass

    return [Document(page_content=text.strip(), metadata={"source": filename, "page": 1})]


def extract_documents(uploads) -> Tuple[List[Document], List[str]]:
    """Parse uploaded Streamlit file objects into LangChain Documents."""
    all_docs: List[Document] = []
    file_names: List[str] = []

    for upload in uploads:
        ext = Path(upload.name).suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            logger.warning("Skipping unsupported file: %s", upload.name)
            continue
        try:
            if ext == ".pdf":
                docs = _read_pdf(upload, upload.name)
            else:
                docs = _read_text(upload, upload.name, ext)
            all_docs.extend(docs)
            file_names.append(upload.name)
            logger.info("Extracted %d pages from %s", len(docs), upload.name)
        except Exception as exc:
            logger.error("Failed to parse %s: %s", upload.name, exc)
            raise RuntimeError(f"Could not read '{upload.name}': {exc}") from exc

    return all_docs, file_names


def build_vector_store(documents: List[Document]) -> None:
    """Chunk documents and persist a FAISS index to disk."""
    if not documents:
        raise ValueError("No documents provided.")

    splitter = _splitter()
    chunks = splitter.split_documents(documents)
    logger.info("Split into %d chunks", len(chunks))

    embeddings = _embeddings()
    VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)

    store = FAISS.from_documents(chunks, embeddings)
    store.save_local(str(VECTOR_STORE_DIR))
    logger.info("Vector store saved to %s", VECTOR_STORE_DIR)


def _load_vector_store() -> FAISS:
    if not VECTOR_STORE_DIR.exists():
        raise FileNotFoundError("Vector store not found. Build it first.")
    return FAISS.load_local(
        str(VECTOR_STORE_DIR),
        _embeddings(),
        allow_dangerous_deserialization=True,
    )


def clear_vector_store() -> None:
    """Delete the FAISS index from disk."""
    import shutil
    if VECTOR_STORE_DIR.exists():
        shutil.rmtree(VECTOR_STORE_DIR)
        logger.info("Vector store cleared.")


_SYSTEM_PROMPT = """\
You are ResearchPie, a precise document assistant.
Answer ONLY using the context provided below.
If the answer cannot be found in the context, say: "I could not find this in the uploaded documents."
Always cite which document/page your answer comes from using [Source: filename, p.N] notation.
Be concise and factual.
"""

_RAG_TEMPLATE = """\
CONTEXT:
{context}

QUESTION:
{question}

ANSWER (cite sources inline):"""


def answer_question(question: str, model: str = DEFAULT_CHAT_MODEL) -> str:
    """Retrieve relevant chunks and generate a grounded answer."""
    store = _load_vector_store()
    results = store.similarity_search_with_score(question, k=RETRIEVAL_TOP_K)

    if not results:
        return "No relevant content found in your documents for this question."

    context_parts = []
    for i, (doc, score) in enumerate(results, start=1):
        src = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "?")
        context_parts.append(
            f"[{i}] Source: {src}, Page {page} (relevance: {score:.2f})\n{doc.page_content}"
        )
    context = "\n\n---\n\n".join(context_parts)
    prompt = _RAG_TEMPLATE.format(context=context, question=question)

    client = ollama.Client(host=OLLAMA_HOST)
    t0 = time.perf_counter()

    response = client.chat(
        model=model,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )

    elapsed = time.perf_counter() - t0
    answer = response["message"]["content"].strip()
    logger.info("Generated answer in %.2fs using %s", elapsed, model)

    sources_seen: dict[str, set] = {}
    for doc, _ in results:
        src = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "?")
        sources_seen.setdefault(src, set()).add(str(page))

    footer_lines = ["\n\n---\n**Sources consulted:**"]
    for src, pages in sources_seen.items():
        footer_lines.append(f"- `{src}` — page(s) {', '.join(sorted(pages))}")

    return answer + "\n".join(footer_lines)