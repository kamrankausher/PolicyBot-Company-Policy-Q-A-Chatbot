"""
Pydantic models for PolicyBot API request and response validation.
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class UploadResponse(BaseModel):
    """Response returned after a PDF is successfully uploaded and indexed."""
    message: str
    document_name: str
    total_chunks: int
    session_id: str


class QuestionRequest(BaseModel):
    """Request body for asking a question about the uploaded policy document."""
    session_id: str = Field(..., min_length=1, description="Session ID from the upload step")
    question: str = Field(..., min_length=1, max_length=500, description="Question text")


class SourceChunk(BaseModel):
    """A single source excerpt retrieved from the vector database."""
    text: str
    page_number: int
    chunk_index: int


class AnswerResponse(BaseModel):
    """Response containing the generated answer, sources, and confidence level."""
    answer: str
    sources: List[SourceChunk]
    confidence: str = Field(..., description="One of: High, Medium, Low")
    session_id: str


class ChatMessage(BaseModel):
    """A single message in the chat history (user or assistant)."""
    role: str = Field(..., description="Either 'user' or 'assistant'")
    content: str
    sources: Optional[List[SourceChunk]] = None
