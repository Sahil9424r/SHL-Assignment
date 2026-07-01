# SHL Assessment Recommender

Conversational agent that helps hiring managers find the right SHL assessments.

## Stack
- **FastAPI** — API server
- **Gemini 1.5 Flash** — LLM for conversation
- **Gemini text-embedding-004** — embeddings
- **ChromaDB** — vector store (local folder, no separate server needed)

## Project Structure
```
shl-recommender/
├── catalog.json        ← your SHL catalog data
├── .env                ← your Gemini API key
├── .env.example        ← template
├── .gitignore
├── requirements.txt
├── build_index.py      ← run ONCE to embed catalog into ChromaDB
├── main.py             ← FastAPI server
├── agent.py            ← retrieval + Gemini logic
├── prompts.py          ← system prompt
├── test_local.py       ← test against sample conversations
└── chroma_db/          ← auto-created by build_index.py
```

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Add your Gemini API key
Create a `.env` file:
```
GEMINI_API_KEY=your_actual_key_here
```

### 3. Place your catalog
Make sure `catalog.json` is in the project root (JSON array of assessment objects).

### 4. Build the ChromaDB index (run ONCE)
```bash
python build_index.py
```
Creates `chroma_db/` folder automatically. Takes a few minutes.

### 5. Start the server
```bash
python main.py
```
Server runs at `http://localhost:8000`

### 6. Test locally
```bash
python test_local.py
```

## API

### GET /health
```json
{"status": "ok"}
```

### POST /chat
**Request:**
```json
{
  "messages": [
    {"role": "user", "content": "I need to hire a Java developer"},
    {"role": "assistant", "content": "Sure. What is the seniority level?"},
    {"role": "user", "content": "Mid-level, around 4 years"}
  ]
}
```

**Response:**
```json
{
  "reply": "Here are assessments for a mid-level Java developer.",
  "recommendations": [
    {
      "name": "Java 8 (New)",
      "url": "https://www.shl.com/products/product-catalog/view/java-8-new/",
      "test_type": "K"
    }
  ],
  "end_of_conversation": false
}
```

## Deployment (Render)

1. Push to GitHub (`.env` and `chroma_db/` in `.gitignore` — see note below)
2. Create Web Service on Render
3. Set `GEMINI_API_KEY` in Render environment variables
4. Build command: `pip install -r requirements.txt && python build_index.py`
5. Start command: `python main.py`

> **Note:** Either commit `chroma_db/` to your repo (easiest), or let the build command regenerate it on each deploy. Free tier Render has enough disk for it.
