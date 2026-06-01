"""
Central configuration — all settings read from environment / .env file.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ── Ollama ────────────────────────────────────────────────────────────────────
OLLAMA_HOST: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
DEFAULT_CHAT_MODEL: str = os.getenv("DEFAULT_CHAT_MODEL", "llama3.2:3b")

# ── Vector store ──────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
STORAGE_DIR: Path = BASE_DIR / "storage"
VECTOR_STORE_DIR: Path = STORAGE_DIR / "vector_store"
CHAT_HISTORY_PATH: Path = STORAGE_DIR / "chat_history.json"

# ── Chunking ──────────────────────────────────────────────────────────────────
CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "100"))

# ── Retrieval ─────────────────────────────────────────────────────────────────
RETRIEVAL_TOP_K: int = int(os.getenv("RETRIEVAL_TOP_K", "5"))

# ── Supported file types ──────────────────────────────────────────────────────
SUPPORTED_EXTENSIONS: set[str] = {".pdf", ".txt", ".md", ".json"}