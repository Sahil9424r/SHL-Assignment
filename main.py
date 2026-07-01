"""
main.py
FastAPI server exposing:
  GET  /health  → {"status": "ok"}
  POST /chat    → agent response with reply + recommendations
"""

import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
import google.generativeai as genai

from agent import load_index, get_agent_response

load_dotenv()

# Configure Gemini API key at startup
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY not found in environment. Check your .env file.")

genai.configure(api_key=GEMINI_API_KEY)


# ── Pydantic models ──────────────────────────────────────────────────────────

class Message(BaseModel):
    role: str   # "user" or "assistant"
    content: str


class ChatRequest(BaseModel):
    messages: list[Message]


class Recommendation(BaseModel):
    name: str
    url: str
    test_type: str


class ChatResponse(BaseModel):
    reply: str
    recommendations: list[Recommendation]
    end_of_conversation: bool


# ── App lifecycle ────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load FAISS index once when server starts."""
    print("Loading FAISS index...")
    load_index()
    print("Server ready.")
    yield


app = FastAPI(
    title="SHL Assessment Recommender",
    version="1.0.0",
    lifespan=lifespan,
)


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    # Validate
    if not request.messages:
        raise HTTPException(status_code=400, detail="messages cannot be empty")

    # Must start with a user message
    if request.messages[0].role != "user":
        raise HTTPException(status_code=400, detail="First message must be from user")

    # Convert pydantic models to plain dicts for agent
    messages = [{"role": m.role, "content": m.content} for m in request.messages]

    # try:
    #     result = get_agent_response(messages)
    # except Exception as e:
    #     raise HTTPException(status_code=500, detail=f"Agent error: {str(e)}")
    try:
        result = get_agent_response(messages)
    except Exception as e:
        import traceback
        traceback.print_exc()  # prints full error in Terminal 1
        raise HTTPException(status_code=500, detail=f"Agent error: {str(e)}")
    return ChatResponse(
        reply=result["reply"],
        recommendations=[
            Recommendation(**rec) for rec in result["recommendations"]
        ],
        end_of_conversation=result["end_of_conversation"],
    )


# ── Run directly ─────────────────────────────────────────────────────────────

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=False)
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    host = "0.0.0.0" if os.environ.get("PORT") else "127.0.0.1"
    uvicorn.run("main:app", host=host, port=port, reload=False)