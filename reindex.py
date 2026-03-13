"""Standalone script to re-index all documents with the current embedding model.

Run this after changing the embedding model in config.py:
    python reindex.py

The script will:
1. Delete the old ChromaDB collection
2. Re-load and re-chunk all documents from the documents/ directory
3. Create new embeddings with the configured model
"""

import os
import sys
import time

# Fix Windows console encoding
if sys.platform == "win32":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from config import DOCUMENTS_DIR, EMBEDDING_MODEL, CHUNK_SIZE, CHUNK_OVERLAP
from rag.document_loader import load_document
from rag.chunker import chunk_text
from rag.vector_store import reset_collection, add_document, preload_model, get_document_count


def main():
    print(f"{'='*50}")
    print(f"  AI Tutor - Document Herindexering")
    print(f"{'='*50}")
    print(f"Embedding model: {EMBEDDING_MODEL}")
    print(f"Chunk grootte: {CHUNK_SIZE} (overlap: {CHUNK_OVERLAP})")
    print()

    # Find documents
    files = [f for f in DOCUMENTS_DIR.iterdir() if f.is_file() and f.suffix.lower() in {
        ".pdf", ".docx", ".doc", ".txt", ".md", ".html", ".htm"
    }]

    if not files:
        print("Geen documenten gevonden in", DOCUMENTS_DIR)
        sys.exit(1)

    print(f"Gevonden: {len(files)} document(en)")
    for f in files:
        size_kb = f.stat().st_size / 1024
        print(f"  - {f.name} ({size_kb:.0f} KB)")
    print()

    # Step 1: Pre-load embedding model
    print("Stap 1/3: Embedding model laden...")
    start = time.time()
    preload_model()
    print(f"  Model geladen in {time.time()-start:.1f}s")
    print()

    # Step 2: Reset collection
    print("Stap 2/3: Oude embeddings verwijderen...")
    reset_collection()
    print("  Collectie gereset")
    print()

    # Step 3: Re-index all documents
    print("Stap 3/3: Documenten herindexeren...")
    total_chunks = 0
    for i, file_path in enumerate(files, 1):
        print(f"  [{i}/{len(files)}] {file_path.name}...", end=" ", flush=True)
        start = time.time()
        try:
            text = load_document(file_path)
            if text and text.strip():
                chunks = chunk_text(text)
                num_chunks = add_document(chunks, file_path.name)
                total_chunks += num_chunks
                elapsed = time.time() - start
                print(f"✓ {num_chunks} chunks ({elapsed:.1f}s)")
            else:
                print("⚠ Leeg document")
        except Exception as e:
            print(f"✗ Fout: {e}")

    print()
    print(f"{'='*50}")
    print(f"Herindexering voltooid!")
    print(f"Totaal: {len(files)} documenten, {total_chunks} chunks")
    print(f"Vector store bevat nu: {get_document_count()} chunks")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
