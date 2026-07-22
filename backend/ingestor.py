"""
PDF ingestion module: extracts text from PDFs, chunks it, embeds it, and stores in ChromaDB.
"""

import io
from pypdf import PdfReader
from fastembed import TextEmbedding
from backend.database import chroma_client

# Lazy load fastembed model to prevent blocking uvicorn startup
embedding_model = None

def get_embedding_model():
    global embedding_model
    if embedding_model is None:
        # BAAI/bge-small-en-v1.5 is the default, highly efficient local ONNX model
        embedding_model = TextEmbedding()
    return embedding_model

def get_embeddings(texts: list[str]) -> list[list[float]]:
    """Get embeddings using local FastEmbed model (no rate limits)."""
    model = get_embedding_model()
    # fastembed returns an iterable of numpy arrays, we convert it to lists
    embeddings = list(model.embed(texts))
    return [e.tolist() for e in embeddings]


def extract_text_by_page(file_bytes: bytes) -> list[dict]:
    """Extract text from each page of a PDF. Skips pages with less than 50 characters."""
    pages = []
    try:
        reader = PdfReader(io.BytesIO(file_bytes))
        for page_num, page in enumerate(reader.pages):
            text = page.extract_text()
            text = text.strip() if text else ""
            if len(text) >= 50:
                pages.append({"page_number": page_num + 1, "text": text})
    except Exception:
        return []
    if not pages:
        raise ValueError("No readable text found in the PDF.")
    return pages


def chunk_text(pages: list[dict], chunk_size: int = 400, overlap: int = 50) -> list[dict]:
    """Split page texts into overlapping chunks of fixed character length."""
    chunks = []
    chunk_index = 0
    for page in pages:
        text = page["text"]
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk_content = text[start:end].strip()
            if chunk_content:
                chunks.append({
                    "text": chunk_content,
                    "page_number": page["page_number"],
                    "chunk_index": chunk_index,
                })
                chunk_index += 1
            start += chunk_size - overlap
    return chunks


def embed_and_store(chunks: list[dict], session_id: str) -> int:
    """Embed chunk texts using Gemini and store them in a ChromaDB collection."""
    collection_name = f"policy_{session_id}"
    collection = chroma_client.get_or_create_collection(name=collection_name)

    texts = [chunk["text"] for chunk in chunks]
    embeddings = get_embeddings(texts)

    ids = [f"chunk_{i}" for i in range(len(chunks))]
    metadatas = [
        {"page_number": chunk["page_number"], "chunk_index": chunk["chunk_index"]}
        for chunk in chunks
    ]

    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=texts,
        metadatas=metadatas,
    )
    return len(chunks)
