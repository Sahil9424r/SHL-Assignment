"""
agent.py
Core agent logic:
1. Embed conversation context using Gemini
2. Query ChromaDB for top relevant catalog items
3. Call Gemini with system prompt + catalog context + conversation history
4. Parse and return structured response
"""

import json
import os
import re
import google.generativeai as genai
import chromadb
from prompts import build_prompt

CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "shl_catalog"
TOP_K = 25  # retrieve top 25, Gemini picks best 1-10

# Loaded once at startup
_collection = None


def load_index():
    """Load ChromaDB collection into memory (called once at startup)."""
    global _collection
    if _collection is None:
        if not os.path.exists(CHROMA_DIR):
            raise FileNotFoundError(
                "chroma_db/ not found. Run `python build_index.py` first."
            )
        client = chromadb.PersistentClient(path=CHROMA_DIR)
        _collection = client.get_collection(name=COLLECTION_NAME)
        print(f"ChromaDB loaded: {_collection.count()} items ready.")


def _embed_query(text: str) -> list[float]:
    """Embed a single query string using Gemini."""
    result = genai.embed_content(
        model="models/gemini-embedding-001",
        content=text,
        task_type="retrieval_query",
    )
    return result["embedding"]


def _build_query_from_history(messages: list[dict]) -> str:
    """
    Combine recent user messages into one query string for retrieval.
    Uses last 6 turns, user messages only.
    """
    user_parts = [
        msg["content"]
        for msg in messages[-6:]
        if msg["role"] == "user"
    ]
    return " ".join(user_parts)


def _retrieve(query: str) -> list[dict]:
    """Query ChromaDB and return top-K catalog items as list of dicts."""
    query_embedding = _embed_query(query)

    results = _collection.query(
        query_embeddings=[query_embedding],
        n_results=TOP_K,
        include=["metadatas", "documents", "distances"],
    )

    # results["metadatas"][0] is a list of metadata dicts for our single query
    items = []
    for metadata in results["metadatas"][0]:
        items.append(metadata)  # already a plain dict with all fields
    return items


def _format_catalog_context(items: list[dict]) -> str:
    """Format retrieved items into a readable block for the system prompt."""
    lines = []
    for i, item in enumerate(items, 1):
        lines.append(
            f"{i}. Name: {item['name']}\n"
            f"   URL: {item['url']}\n"
            f"   Test Type: {item['test_type']} ({item['keys']})\n"
            f"   Job Levels: {item['job_levels'] or 'Not specified'}\n"
            f"   Duration: {item['duration'] or 'Not specified'}\n"
            f"   Remote: {item['remote']} | Adaptive: {item['adaptive']}\n"
            f"   Description: {item['description']}\n"
        )
    return "\n".join(lines)


def _parse_response(raw: str) -> dict:
    """Extract JSON from Gemini's response."""
    # Strip markdown fences if present
    cleaned = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group())
            except json.JSONDecodeError:
                return {
                    "reply": cleaned or "Sorry, I encountered an error.",
                    "recommendations": [],
                    "end_of_conversation": False,
                }
        else:
            return {
                "reply": cleaned or "Sorry, I encountered an error.",
                "recommendations": [],
                "end_of_conversation": False,
            }

    # If reply field itself contains a JSON string, parse it again
    reply_field = data.get("reply", "")
    if isinstance(reply_field, str) and reply_field.strip().startswith("{"):
        try:
            inner = json.loads(reply_field)
            if "reply" in inner:
                data = inner  # use the inner JSON instead
        except json.JSONDecodeError:
            pass

    # Sanitize fields
    reply = str(data.get("reply", "")).strip()
    recommendations = data.get("recommendations", [])
    end_of_conversation = bool(data.get("end_of_conversation", False))

    if not isinstance(recommendations, list):
        recommendations = []

    clean_recs = []
    for rec in recommendations[:10]:
        if isinstance(rec, dict) and rec.get("name") and rec.get("url"):
            clean_recs.append({
                "name": str(rec["name"]),
                "url": str(rec["url"]),
                "test_type": str(rec.get("test_type", "K")),
            })

    return {
        "reply": reply,
        "recommendations": clean_recs,
        "end_of_conversation": end_of_conversation,
    }

def get_agent_response(messages: list[dict]) -> dict:
    """
    Main entry point called by FastAPI endpoint.

    Args:
        messages: Full conversation history
                  [{"role": "user"|"assistant", "content": "..."}]

    Returns:
        dict: { reply, recommendations, end_of_conversation }
    """
    # Step 1: Build search query from conversation
    query = _build_query_from_history(messages)

    # Step 2: Retrieve relevant items from ChromaDB
    retrieved_items = _retrieve(query)

    # Step 3: Format catalog context for system prompt
    catalog_context = _format_catalog_context(retrieved_items)

    # Step 4: Build system prompt with catalog items injected
    system_prompt = build_prompt(catalog_context)

    # Step 5: Build Gemini conversation history
    # Gemini uses "model" not "assistant"
    gemini_history = []
    for msg in messages[:-1]:  # everything except last message
        role = "model" if msg["role"] == "assistant" else "user"
        gemini_history.append({"role": role, "parts": [msg["content"]]})

    last_user_message = messages[-1]["content"] if messages else ""

    # Step 6: Call Gemini 1.5 Flash
    model = genai.GenerativeModel(
        model_name="gemini-2.5-flash",
        system_instruction=system_prompt,
        generation_config=genai.GenerationConfig(
            temperature=0.2,         # low = consistent structured JSON output
            max_output_tokens=2048,
            # response_mime_type="application/json",
        ),
    )

    chat = model.start_chat(history=gemini_history)
    response = chat.send_message(last_user_message)

    # Step 7: Parse and return
    return _parse_response(response.text)
