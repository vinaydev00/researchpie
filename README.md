# 🥧 ResearchPie

> Local RAG app — upload documents, build a private knowledge base, chat with grounded answers. Everything runs locally via Ollama. Nothing leaves your machine.

## Features
- 📄 Upload PDF, TXT, Markdown, JSON files
- 🧠 Local embeddings via Ollama
- 💬 Answers grounded in your documents
- 📌 Source citations with file + page number
- 💾 Persistent chat history
- ⚙️ Tunable chunking settings in sidebar
- 🐳 One command Docker deploy

## Project Structure

researchpie/
├── app/
│   ├── __init__.py
│   ├── config.py
│   ├── rag.py
│   └── history.py
├── tests/
│   └── test_rag.py
├── storage/
├── main.py
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── .env.example

## Quick Start

### Local
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
ollama pull llama3.2:3b
ollama pull nomic-embed-text
streamlit run main.py

### Docker
docker compose up --build

Open http://localhost:8501

## Configuration

Copy .env.example to .env and edit:

- OLLAMA_HOST — default http://localhost:11434
- EMBEDDING_MODEL — default nomic-embed-text
- DEFAULT_CHAT_MODEL — default llama3.2:3b
- CHUNK_SIZE — default 800
- CHUNK_OVERLAP — default 100
- RETRIEVAL_TOP_K — default 5

## Run Tests

pip install pytest
pytest tests/ -v