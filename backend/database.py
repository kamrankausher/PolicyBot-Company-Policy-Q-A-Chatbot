"""
Shared ChromaDB client instance used across the application.
Using a persistent client ensures that ingestor, retriever, and main all access the same data.
"""

import chromadb

# Single persistent client shared by all modules
chroma_client = chromadb.PersistentClient(path="./chroma_db")
