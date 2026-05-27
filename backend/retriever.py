"""
Retriever module: searches ChromaDB for the most relevant policy chunks given a question.
"""

from fastapi import HTTPException
from backend.models import SourceChunk
from backend.ingestor import get_embedding_model
from backend.database import chroma_client


def search_similar_chunks(question: str, session_id: str, top_k: int = 4) -> list[SourceChunk]:
    """Embed the question and find the top_k most similar chunks in ChromaDB."""
    collection_name = f"policy_{session_id}"

    try:
        collection = chroma_client.get_collection(name=collection_name)
    except Exception:
        raise HTTPException(
            status_code=404,
            detail=f"No document found for session '{session_id}'. Please upload a PDF first."
        )

    count = collection.count()
    if count == 0:
        return []

    model = get_embedding_model()
    question_embedding = model.encode([question]).tolist()

    results = collection.query(
        query_embeddings=question_embedding,
        n_results=min(top_k, count),
    )

    source_chunks = []
    for i in range(len(results["documents"][0])):
        source_chunks.append(
            SourceChunk(
                text=results["documents"][0][i],
                page_number=results["metadatas"][0][i]["page_number"],
                chunk_index=results["metadatas"][0][i]["chunk_index"],
            )
        )
    return source_chunks
