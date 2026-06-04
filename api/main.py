import logging
from pathlib import Path
from typing import List

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.rag import (
    answer_question,
    build_vector_store,
    clear_vector_store,
    extract_documents,
    extract_model_names,
)
from app.history import append_message, clear_history, load_history

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="ResearchPie API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class QuestionRequest(BaseModel):
    question: str
    model: str = "llama3.2:3b"


class QuestionResponse(BaseModel):
    answer: str


@app.get("/")
def root():
    return {"status": "ok", "app": "ResearchPie API"}


@app.get("/models")
def get_models():
    return {"models": extract_model_names()}


@app.post("/upload")
async def upload_files(files: List[UploadFile] = File(...)):
    try:
        documents, file_names = extract_documents(files)
        build_vector_store(documents)
        return {"status": "ok", "indexed_files": file_names}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/chat", response_model=QuestionResponse)
def chat(request: QuestionRequest):
    try:
        answer = answer_question(question=request.question, model=request.model)
        append_message("user", request.question)
        append_message("assistant", answer)
        return QuestionResponse(answer=answer)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/history")
def get_history():
    return {"messages": load_history()}


@app.delete("/history")
def delete_history():
    clear_history()
    return {"status": "cleared"}


@app.delete("/knowledge-base")
def delete_knowledge_base():
    clear_vector_store()
    return {"status": "cleared"}