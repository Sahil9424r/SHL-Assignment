"""
build_index.py
Run this ONCE before starting the server.
Reads catalog.json, embeds every item using Gemini,
stores vectors + metadata in ChromaDB (chroma_db/ folder).
"""

import json
import os
import time
from dotenv import load_dotenv
import google.generativeai as genai
import chromadb
from chromadb.utils import embedding_functions

load_dotenv()

genai.configure(api_key=os.environ["GEMINI_API_KEY"])

CATALOG_PATH = "catalog.json"
CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "shl_catalog"
BATCH_SIZE = 5 # embed 5 items at a time (free tier rate limit)

KEY_TO_TYPE = {
    "Knowledge & Skills": "K",
    "Personality & Behavior": "P",
    "Ability & Aptitude": "A",
    "Biodata & Situational Judgment": "B",
    "Competencies": "C",
    "Development & 360": "D",
    "Assessment Exercises": "E",
}


def get_test_type(keys: list) -> str:
    for key in keys:
        if key in KEY_TO_TYPE:
            return KEY_TO_TYPE[key]
    return "K"


def item_to_document(item: dict) -> str:
    """Convert catalog item to a single string for embedding."""
    return (
        f"Name: {item.get('name', '')}\n"
        f"Description: {item.get('description', '')}\n"
        f"Test Type: {', '.join(item.get('keys', []))}\n"
        f"Job Levels: {', '.join(item.get('job_levels', []))}\n"
        f"Duration: {item.get('duration', 'Not specified')}\n"
        f"Languages: {', '.join(item.get('languages', []))}\n"
        f"Remote: {item.get('remote', 'unknown')} | Adaptive: {item.get('adaptive', 'unknown')}"
    )


def embed_batch(texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts using Gemini text-embedding-004."""
    result = genai.embed_content(
        model="models/gemini-embedding-001",
        content=texts,
        task_type="retrieval_document",
    )
    return result["embedding"]


def main():
    # Load catalog
    print(f"Loading {CATALOG_PATH} ...")
    with open(CATALOG_PATH, "r", encoding="utf-8") as f:
        catalog = json.load(f)

    # Filter valid items only
    valid_items = [
        item for item in catalog
        if item.get("description") and item.get("status") == "ok"
    ]
    print(f"Valid items: {len(valid_items)} / {len(catalog)}")

    # Setup ChromaDB
    client = chromadb.PersistentClient(path=CHROMA_DIR)

    # Delete existing collection if rebuilding
    try:
        client.delete_collection(COLLECTION_NAME)
        print("Deleted existing collection.")
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},  # cosine similarity
    )

    # Process in batches
    total = len(valid_items)
    for i in range(0, total, BATCH_SIZE):
        batch = valid_items[i : i + BATCH_SIZE]
        batch_num = i // BATCH_SIZE + 1
        total_batches = (total + BATCH_SIZE - 1) // BATCH_SIZE
        print(f"Batch {batch_num}/{total_batches} — embedding {len(batch)} items ...")

        # Build documents (text to embed)
        documents = [item_to_document(item) for item in batch]

        # Build embeddings
        embeddings = embed_batch(documents)

        # Build IDs (must be unique strings)
        ids = [str(item["entity_id"]) for item in batch]

        # Build metadata (stored as flat dict — Chroma requires flat key:value)
        metadatas = []
        for item in batch:
            metadatas.append({
                "name": item.get("name", ""),
                "url": item.get("link", ""),
                "test_type": get_test_type(item.get("keys", [])),
                "keys": ", ".join(item.get("keys", [])),
                "job_levels": ", ".join(item.get("job_levels", [])),
                "duration": item.get("duration", ""),
                "languages": ", ".join(item.get("languages", [])),
                "remote": item.get("remote", ""),
                "adaptive": item.get("adaptive", ""),
                "description": item.get("description", ""),
            })

        # Add to ChromaDB
        collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )

        # Respect Gemini free tier rate limits
        time.sleep(15)

    print(f"\nDone! {collection.count()} items stored in ChromaDB at ./{CHROMA_DIR}/")


if __name__ == "__main__":
    main()
