"""ChromaDB vector store for document embeddings."""

import logging
import uuid
from typing import List, Optional

import chromadb
from sentence_transformers import SentenceTransformer

from config import (
    CHROMA_DB_DIR,
    EMBEDDING_MODEL,
    EMBEDDING_PASSAGE_PREFIX,
    EMBEDDING_QUERY_PREFIX,
    RETRIEVAL_TOP_K,
)

logger = logging.getLogger("tutor")

_client: Optional[chromadb.PersistentClient] = None
_collection: Optional[chromadb.Collection] = None
_st_model: Optional[SentenceTransformer] = None

# Batch size for adding documents (ChromaDB can struggle with large batches)
ADD_BATCH_SIZE = 100


def preload_model() -> None:
    """Pre-load the embedding model (avoids timeout on first request)."""
    global _st_model
    if _st_model is None:
        logger.info(f"Embedding model laden: {EMBEDDING_MODEL}")
        _st_model = SentenceTransformer(EMBEDDING_MODEL)
        logger.info(
            f"Embedding model geladen: {EMBEDDING_MODEL} "
            f"(dim={_st_model.get_sentence_embedding_dimension()})"
        )


def _encode_passages(texts: List[str]) -> List[List[float]]:
    """Encode document passages with the passage prefix."""
    preload_model()
    prefixed = [f"{EMBEDDING_PASSAGE_PREFIX}{t}" for t in texts]
    embeddings = _st_model.encode(prefixed, show_progress_bar=False, normalize_embeddings=True)
    return embeddings.tolist()


def _encode_query(query: str) -> List[float]:
    """Encode a search query with the query prefix."""
    preload_model()
    prefixed = f"{EMBEDDING_QUERY_PREFIX}{query}"
    embedding = _st_model.encode(prefixed, show_progress_bar=False, normalize_embeddings=True)
    return embedding.tolist()


def _get_collection() -> chromadb.Collection:
    """Get or create the ChromaDB collection (without embedding function — we manage embeddings ourselves)."""
    global _client, _collection
    if _collection is None:
        _client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))
        _collection = _client.get_or_create_collection(
            name="tutor_documents",
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


def add_document(chunks: List[str], doc_name: str, extra_metadatas: List[dict] = None) -> int:
    """Add document chunks to the vector store in batches. Returns number of chunks added.

    Args:
        chunks: List of text chunks to embed and store.
        doc_name: Document filename for metadata.
        extra_metadatas: Optional list of extra metadata dicts (one per chunk) to merge
                         with the default metadata. Used for image descriptions etc.
    """
    collection = _get_collection()
    total = len(chunks)

    # Process in batches to avoid memory issues with large documents
    for start in range(0, total, ADD_BATCH_SIZE):
        end = min(start + ADD_BATCH_SIZE, total)
        batch_chunks = chunks[start:end]
        batch_ids = [f"{doc_name}_{i}_{uuid.uuid4().hex[:8]}" for i in range(start, end)]
        batch_meta = []
        for i in range(start, end):
            meta = {"source": doc_name, "chunk_index": i}
            if extra_metadatas and i < len(extra_metadatas) and extra_metadatas[i]:
                meta.update(extra_metadatas[i])
            batch_meta.append(meta)

        # Compute embeddings with passage prefix
        batch_embeddings = _encode_passages(batch_chunks)

        collection.add(
            documents=batch_chunks,
            embeddings=batch_embeddings,
            ids=batch_ids,
            metadatas=batch_meta,
        )
        if total > ADD_BATCH_SIZE:
            logger.info(f"  Batch {start // ADD_BATCH_SIZE + 1}: chunks {start + 1}-{end} van {total}")

    return total


def search(query: str, top_k: int = RETRIEVAL_TOP_K) -> List[dict]:
    """Search for relevant chunks. Returns list of {text, source, score}."""
    collection = _get_collection()
    if collection.count() == 0:
        return []

    # Compute query embedding with query prefix
    query_embedding = _encode_query(query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, collection.count()),
    )
    items = []
    for i in range(len(results["documents"][0])):
        items.append({
            "text": results["documents"][0][i],
            "source": results["metadatas"][0][i]["source"],
            "metadata": results["metadatas"][0][i],  # full metadata (content_type, image_path, etc.)
            "score": 1 - results["distances"][0][i],  # cosine similarity
        })
    return items


def delete_document(doc_name: str) -> int:
    """Delete all chunks for a document. Returns number deleted."""
    collection = _get_collection()
    results = collection.get(where={"source": doc_name})
    if results["ids"]:
        collection.delete(ids=results["ids"])
    return len(results["ids"])


def reset_collection() -> None:
    """Delete and recreate the collection. Use after changing embedding model."""
    global _client, _collection
    if _client is None:
        _client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))
    try:
        _client.delete_collection("tutor_documents")
        logger.info("Oude collectie verwijderd")
    except Exception:
        pass  # Collection didn't exist
    _collection = None  # Force recreation on next access


def list_documents() -> List[str]:
    """List all unique document names in the store."""
    collection = _get_collection()
    if collection.count() == 0:
        return []
    all_meta = collection.get()["metadatas"]
    return sorted(set(m["source"] for m in all_meta))


def get_document_count() -> int:
    """Get total number of chunks in the store."""
    return _get_collection().count()
