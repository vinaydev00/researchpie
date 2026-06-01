"""
tests/test_rag.py — Unit tests for the RAG pipeline.
Run with: pytest tests/ -v
"""
import json
from unittest.mock import MagicMock

from langchain.schema import Document
from app.rag import _splitter, extract_documents


def _make_upload(name: str, content: bytes):
    f = MagicMock()
    f.name = name
    f.read.return_value = content
    return f


class TestSplitter:
    def test_splits_long_text(self):
        splitter = _splitter(chunk_size=100, overlap=10)
        docs = [Document(page_content="word " * 200, metadata={"source": "test.txt"})]
        chunks = splitter.split_documents(docs)
        assert len(chunks) > 1

    def test_preserves_metadata(self):
        splitter = _splitter(chunk_size=100, overlap=10)
        docs = [Document(page_content="word " * 200, metadata={"source": "test.txt", "page": 3})]
        chunks = splitter.split_documents(docs)
        for chunk in chunks:
            assert chunk.metadata["source"] == "test.txt"

    def test_short_doc_single_chunk(self):
        splitter = _splitter(chunk_size=1000, overlap=100)
        docs = [Document(page_content="Short text.", metadata={})]
        chunks = splitter.split_documents(docs)
        assert len(chunks) == 1


class TestExtractDocuments:
    def test_txt_extraction(self):
        upload = _make_upload("notes.txt", b"Hello world. This is a test.")
        docs, names = extract_documents([upload])
        assert names == ["notes.txt"]
        assert "Hello world" in docs[0].page_content

    def test_json_extraction(self):
        data = json.dumps({"key": "value"}).encode()
        upload = _make_upload("data.json", data)
        docs, names = extract_documents([upload])
        assert names == ["data.json"]

    def test_unsupported_extension_skipped(self):
        upload = _make_upload("image.png", b"fakeimage")
        docs, names = extract_documents([upload])
        assert docs == []
        assert names == []

    def test_empty_uploads(self):
        docs, names = extract_documents([])
        assert docs == []
        assert names == []


class TestConfig:
    def test_defaults_are_sensible(self):
        from app.config import CHUNK_OVERLAP, CHUNK_SIZE, RETRIEVAL_TOP_K
        assert 200 <= CHUNK_SIZE <= 4000
        assert 0 <= CHUNK_OVERLAP < CHUNK_SIZE
        assert 1 <= RETRIEVAL_TOP_K <= 20