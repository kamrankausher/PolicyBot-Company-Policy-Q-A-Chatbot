"""
Shared ChromaDB client instance used across the application.
Using a persistent client ensures that ingestor, retriever, and main all access the same data.
"""

from chromadb.config import Settings
import chromadb

# Single persistent client shared by all modules
# Disable telemetry to prevent network hanging on Render
chroma_client = chromadb.PersistentClient(
    path="./chroma_db",
    settings=Settings(anonymized_telemetry=False)
)
