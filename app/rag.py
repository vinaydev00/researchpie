import json
import logging
import os
import time
from pathlib import Path
from typing import IO, List, Tuple

import chromadb
import ollama
import pdfplumber
from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langsmith import traceable

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
CHROMA_COLLECTION = "researchpie"


def _get_chroma_client():
    VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(VECTOR_STORE_DIR))


def _get_collection(client):
    return client.get_or_create_collection(
        name=CHROMA_COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )


def _embed(texts: List[str]) -> List[List[float]]:
    client = ollama.Client(host=OLLAMA_HOST)
    embeddings = []
    for text in texts:
        response = client.embeddings(model=EMBEDDING_MODEL, prompt=text)
        embeddings.append(response["embedding"])
    return embeddings


def _splitter(chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )


def extract_model_names() -> List[str]:
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
    all_docs: List[Document] = []
    file_names: List[str] = []
    for upload in uploads:
        ext = Path(upload.name).suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            continue
        try:
            if ext == ".pdf":
                docs = _read_pdf(upload, upload.name)
            else:
                docs = _read_text(upload, upload.name, ext)
            all_docs.extend(docs)
            file_names.append(upload.name)
        except Exception as exc:
            raise RuntimeError(f"Could not read '{upload.name}': {exc}") from exc
    return all_docs, file_names


def build_vector_store(documents: List[Document]) -> None:
    if not documents:
        raise ValueError("No documents provided.")
    splitter = _splitter()
    chunks = splitter.split_documents(documents)
    logger.info("Split into %d chunks", len(chunks))
    texts = [c.page_content for c in chunks]
    metadatas = [c.metadata for c in chunks]
    ids = [f"chunk_{i}" for i in range(len(chunks))]
    embeddings = _embed(texts)
    client = _get_chroma_client()
    collection = _get_collection(client)
    collection.upsert(
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=ids,
    )
    logger.info("ChromaDB vector store saved.")


def clear_vector_store() -> None:
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


@traceable(name="ResearchPie RAG")
def answer_question(question: str, model: str = DEFAULT_CHAT_MODEL) -> str:
    client = _get_chroma_client()
    collection = _get_collection(client)
    query_embedding = _embed([question])[0]
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=RETRIEVAL_TOP_K,
        include=["documents", "metadatas", "distances"],
    )
    docs = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    if not docs:
        return "No relevant content found in your documents."

    context_parts = []
    for i, (doc, meta, dist) in enumerate(zip(docs, metadatas, distances), start=1):
        src = meta.get("source", "unknown")
        page = meta.get("page", "?")
        context_parts.append(f"[{i}] Source: {src}, Page {page}\n{doc}")
    context = "\n\n---\n\n".join(context_parts)
    prompt = _RAG_TEMPLATE.format(context=context, question=question)

    ollama_client = ollama.Client(host=OLLAMA_HOST)
    t0 = time.perf_counter()
    response = ollama_client.chat(
        model=model,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )
    elapsed = time.perf_counter() - t0
    answer = response["message"]["content"].strip()
    logger.info("Generated answer in %.2fs", elapsed)

    sources_seen: dict = {}
    for meta in metadatas:
        src = meta.get("source", "unknown")
        page = str(meta.get("page", "?"))
        sources_seen.setdefault(src, set()).add(page)

    footer = ["\n\n---\n**Sources consulted:**"]
    for src, pages in sources_seen.items():
        footer.append(f"- `{src}` — page(s) {', '.join(sorted(pages))}")

    return answer + "\n".join(footer)