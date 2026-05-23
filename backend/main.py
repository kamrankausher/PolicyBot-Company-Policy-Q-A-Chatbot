"""
FastAPI application — the main entry point for PolicyBot backend.
Exposes endpoints for PDF upload, question answering, chat history, and session management.
"""

import os
import uuid
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

from backend.models import (
    UploadResponse,
    QuestionRequest,
    AnswerResponse,
    ChatMessage,
)
from backend.ingestor import extract_text_by_page, chunk_text, embed_and_store
from backend.retriever import search_similar_chunks
from backend.answerer import generate_answer

app = FastAPI(
    title="PolicyBot API",
    description="Company Policy Q&A Chatbot powered by RAG + Gemini",
    version="1.0.0",
)

# Allow all origins so the frontend can call from file:// or any host
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory chat history store: { session_id: [ChatMessage, ...] }
chat_histories: dict[str, list[ChatMessage]] = {}

MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB


@app.post("/upload", response_model=UploadResponse)
async def upload_pdf(file: UploadFile = File(...)):
    """Accept a PDF upload, extract text, chunk it, embed it, and store in ChromaDB."""
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    file_bytes = await file.read()
    if len(file_bytes) > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail="File size exceeds 10MB limit.")

    try:
        pages = extract_text_by_page(file_bytes)
        chunks = chunk_text(pages)
        session_id = uuid.uuid4().hex[:16]
        total_chunks = embed_and_store(chunks, session_id)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process PDF: {str(e)}")

    chat_histories[session_id] = []

    return UploadResponse(
        message="Document uploaded and indexed successfully.",
        document_name=file.filename,
        total_chunks=total_chunks,
        session_id=session_id,
    )


@app.post("/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    """Retrieve relevant chunks and generate an answer using Gemini."""
    if not request.session_id.strip() or not request.question.strip():
        raise HTTPException(status_code=400, detail="session_id and question must not be empty.")

    source_chunks = search_similar_chunks(request.question, request.session_id)

    history_dicts = [
        {"role": msg.role, "content": msg.content}
        for msg in chat_histories.get(request.session_id, [])
    ]

    answer, confidence = generate_answer(request.question, source_chunks, history_dicts)

    # Store conversation in memory
    if request.session_id not in chat_histories:
        chat_histories[request.session_id] = []

    chat_histories[request.session_id].append(
        ChatMessage(role="user", content=request.question)
    )
    chat_histories[request.session_id].append(
        ChatMessage(role="assistant", content=answer, sources=source_chunks)
    )

    return AnswerResponse(
        answer=answer,
        sources=source_chunks,
        confidence=confidence,
        session_id=request.session_id,
    )


@app.get("/history/{session_id}", response_model=list[ChatMessage])
async def get_history(session_id: str):
    """Return the full chat history for a given session."""
    return chat_histories.get(session_id, [])


@app.delete("/session/{session_id}")
async def clear_session(session_id: str):
    """Delete the ChromaDB collection and chat history for a session."""
    from backend.database import chroma_client

    try:
        chroma_client.delete_collection(name=f"policy_{session_id}")
    except Exception:
        pass  # Collection may not exist — that's fine

    chat_histories.pop(session_id, None)
    return {"message": "Session cleared"}


@app.get("/health")
async def health_check():
    """Return service health status and model info."""
    return {
        "status": "ok",
        "model": "gemini-1.5-flash",
        "embeddings": "all-MiniLM-L6-v2",
    }
