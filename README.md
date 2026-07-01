# SHL Assessment Recommender API

> An intelligent multi-turn conversational API that recommends SHL assessments to hiring managers and recruiters based on their role, seniority, and hiring context.

**Live API:** https://shl-assignment-fscp.onrender.com  
**Swagger Docs:** https://shl-assignment-fscp.onrender.com/docs  
**GitHub:** https://github.com/Sahil9424r/SHL-Assignment

---

## Table of Contents

- [Overview](#overview)
- [Tech Stack](#tech-stack)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Local Setup](#local-setup)
- [API Reference](#api-reference)
- [Testing](#testing)
- [Deployment on Render](#deployment-on-render)
- [Tools Used](#tools-used)
- [Evaluation Results](#evaluation-results)

---

## Overview

This project is a FastAPI-based conversational agent that helps hiring managers find the right SHL assessments. The agent:

- Holds a **multi-turn conversation** to understand the hiring need
- Asks **clarifying questions** when the query is vague (role, seniority, purpose)
- Returns **1–10 SHL catalog assessments** once it has enough context
- **Refuses off-topic questions** (salary, legal advice, general HR)
- Sets `end_of_conversation: true` when the user confirms the shortlist
- Caps conversations at **8 turns maximum**

---

## Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| **API Framework** | FastAPI 0.115.0 | REST API server, schema validation, Swagger UI |
| **Data Validation** | Pydantic 2.7.4 | Request/response schema enforcement |
| **LLM** | Gemini 1.5 Flash (`google-generativeai 0.7.2`) | Conversational agent, recommendation logic |
| **Embeddings** | Gemini Embedding 001 | Semantic search over catalog |
| **Vector Database** | ChromaDB 0.6.3 | Stores catalog embeddings, cosine similarity search |
| **Server** | Uvicorn 0.30.6 | ASGI server for FastAPI |
| **Environment** | python-dotenv 1.0.1 | Loads `.env` for local development |
| **HTTP Client** | requests 2.32.3 | Used in test scripts |
| **Deployment** | Render (Web Service) | Cloud hosting with auto-deploy from GitHub |
| **Runtime** | Python 3.11.0 | Pinned for stable pre-built wheels |

---

## Architecture

```
User (multi-turn conversation history)
            │
            ▼
    ┌───────────────┐
    │  POST /chat   │  FastAPI endpoint — validates schema via Pydantic
    └───────┬───────┘
            │
            ▼
    ┌───────────────────────┐
    │  Query Builder        │  Combines ALL user messages into retrieval query
    └───────┬───────────────┘
            │
            ▼
    ┌───────────────────────┐
    │  ChromaDB Retrieval   │  Gemini Embedding 001 → cosine similarity
    │  TOP_K = 40 items     │  Returns most relevant catalog items
    └───────┬───────────────┘
            │
            ▼
    ┌───────────────────────────────────────┐
    │  Gemini 1.5 Flash (LLM)              │
    │  System Prompt (10 rules injected)   │
    │  + Catalog context (top 40 items)    │
    │  + Full conversation history         │
    │  + JSON mode enforced                │
    └───────┬───────────────────────────────┘
            │
            ▼
    ┌───────────────────────┐
    │  JSON Parser          │  Handles malformed output, strips markdown fences
    └───────┬───────────────┘
            │
            ▼
    ┌───────────────────────┐
    │  Pydantic Response    │  reply + recommendations + end_of_conversation
    └───────────────────────┘
```

---

## Project Structure

```
shl-recommender/
│
├── main.py                # FastAPI app — /health and /chat endpoints
├── agent.py               # Core RAG logic — retrieval, LLM call, JSON parse
├── prompts.py             # System prompt with 10 behavioral rules
├── catalog.json           # Full SHL product catalog (~500 items)
│
├── chroma_db/             # Pre-built ChromaDB vector index (committed to repo)
│   ├── chroma.sqlite3     # ChromaDB metadata
│   └── */                 # HNSW index binary files
│
├── requirements.txt       # Python dependencies
├── render.yaml            # Render deployment config
├── Procfile               # Start command for Render
├── .python-version        # Pins Python 3.11.0 for Render
└── .gitignore             # Excludes .env, test scripts, __pycache__
```

> **Note:** `build_index.py` and test scripts (`test_local.py`, `test_conversations.py`) are excluded from the repo via `.gitignore` — they are only needed locally and not required for the API to run.

---

## Local Setup

### Prerequisites
- Python 3.11+
- Gemini API key (paid tier recommended — free tier has 20 req/day limit)

### Step 1 — Clone the repo
```bash
git clone https://github.com/Sahil9424r/SHL-Assignment.git
cd SHL-Assignment
```

### Step 2 — Create virtual environment
```bash
python -m venv myenv

# Windows
myenv\Scripts\activate

# Mac/Linux
source myenv/bin/activate
```

### Step 3 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 4 — Set up environment variable
Create a `.env` file in the project root:
```
GEMINI_API_KEY=your_gemini_api_key_here
```
> The `.env` file is gitignored — never commit it.

### Step 5 — Start the server
```bash
python main.py
# OR
uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

### Step 6 — Verify
```
http://127.0.0.1:8000/health   →  {"status": "ok"}
http://127.0.0.1:8000/docs     →  Swagger UI
```

---

## API Reference

### `GET /health`
Health check endpoint.

**Response:**
```json
{"status": "ok"}
```

---

### `POST /chat`
Main conversational endpoint. Send the full conversation history on every call.

**Request Body:**
```json
{
  "messages": [
    {"role": "user", "content": "string"},
    {"role": "assistant", "content": "string"}
  ]
}
```

**Response:**
```json
{
  "reply": "string",
  "recommendations": [
    {
      "name": "string",
      "url": "string",
      "test_type": "K | P | A | B | C | D | E"
    }
  ],
  "end_of_conversation": false
}
```

**Test Type Codes:**
| Code | Meaning |
|---|---|
| `K` | Knowledge & Skills |
| `P` | Personality & Behavior |
| `A` | Ability & Aptitude |
| `B` | Biodata & Situational Judgment |
| `C` | Competencies |
| `D` | Development & 360 |
| `E` | Assessment Exercises |

**Example — Single turn:**
```json
{
  "messages": [
    {"role": "user", "content": "I need assessments for a senior Java developer"}
  ]
}
```

**Example — Multi-turn conversation:**
```json
{
  "messages": [
    {"role": "user", "content": "I need assessments for a developer"},
    {"role": "assistant", "content": "What technology stack will they be working with?"},
    {"role": "user", "content": "Java and Spring Boot, senior level, 5 years experience"}
  ]
}
```

**Test via Swagger UI:**
```
https://shl-assignment-fscp.onrender.com/docs
```
Click `POST /chat` → "Try it out" → paste body → "Execute"

---

## Testing

The project includes an automated test harness that replays 10 sample multi-turn conversations against the live API and evaluates:

- ✅ Correct recommendations returned (matched by URL)
- ✅ `end_of_conversation` flag accuracy
- ✅ Null-turn compliance (clarifying turns return no recommendations)

**Run all 10 conversations (requires server running):**
```bash
python test_conversations.py

# Single file
python test_conversations.py --file C1.md

# With full reply output
python test_conversations.py --verbose

# Custom delay between turns (for rate limit safety)
python test_conversations.py --delay 10
```

**Sample conversations tested:**
| File | Scenario |
|---|---|
| C1.md | Senior leadership (CXO) selection |
| C2.md | Senior Rust/networking engineer |
| C3.md | 500 entry-level contact center agents |
| C4.md | Graduate financial analysts |
| C5.md | Sales organization re-skilling audit |
| C6.md | Chemical plant operators (safety-critical) |
| C7.md | Bilingual healthcare admin (HIPAA) |
| C8.md | Admin assistants (Excel/Word) |
| C9.md | Senior full-stack Java engineer (7-turn) |
| C10.md | Graduate management trainee battery |

---

## Deployment on Render

### Auto-deploy (recommended)
Render automatically redeploys on every push to `main` branch.

### Manual deploy
1. Go to [render.com](https://render.com) → Your service
2. Click **"Manual Deploy"** → **"Deploy latest commit"**

### Environment variable required
| Key | Value |
|---|---|
| `GEMINI_API_KEY` | Your Gemini API key |

Set this in: Render Dashboard → Your Service → **Environment** tab

### Render configuration (`render.yaml`)
```yaml
services:
  - type: web
    name: shl-recommender
    runtime: python
    pythonVersion: "3.11.0"
    buildCommand: pip install -r requirements.txt
    startCommand: uvicorn main:app --host 0.0.0.0 --port $PORT
    envVars:
      - key: GEMINI_API_KEY
        sync: false
    healthCheckPath: /health
```

> **Why Python 3.11?** Render defaults to Python 3.14 which has no pre-built wheels for `pydantic-core 2.18.4`. Python 3.11 is pinned via `.python-version` and `render.yaml` to ensure successful builds.

---

## Tools Used

| Tool | Used For |
|---|---|
| **Google Antigravity (AI Coding IDE)** | Codebase analysis, writing test harness, prompt iteration, gap analysis against grading rubric, deployment configuration |
| **Gemini 1.5 Flash** | LLM powering the recommender agent at runtime |
| **Gemini Embedding 001** | Catalog indexing (`retrieval_document`) and query embedding (`retrieval_query`) |
| **ChromaDB** | Local vector store — cosine similarity search over catalog embeddings |
| **FastAPI + Pydantic** | Schema-enforced REST API with auto-generated Swagger docs |
| **Render** | Cloud deployment with GitHub auto-deploy |

---

## Evaluation Results

### Automated test harness results (16 turns executed)
| Metric | Result |
|---|---|
| PASS (both recs + eoc correct) | 4 turns |
| PARTIAL (one of two correct) | 11 turns |
| FAIL | 1 turn |
| ERROR (rate limit) | 6 turns |

### Behavior probes
| Probe | Result |
|---|---|
| Refuses off-topic queries (salary, legal) | ✅ PASS |
| No recommendations on vague turn 1 | ✅ PASS |
| Honors surgical edits (add/remove items) | ✅ PASS |
| Correct end_of_conversation detection | ✅ PASS |
| No hallucinated product names/URLs | ✅ PASS |

### Key issues fixed during development
1. **OPQ32r missing** — Model recommended report products without the base questionnaire. Fixed via explicit Rule 3 in system prompt.
2. **Name paraphrasing** — "Verify - G+" instead of "SHL Verify Interactive G+". Fixed via Rule 10 (exact copy instruction).
3. **Short answer retrieval** — Single-word replies ("US.", "Selection.") retrieved poor results. Fixed by querying with full conversation history.
4. **end_of_conversation misses** — "Final list confirmed" not detected. Fixed by expanding trigger phrases in Rule 8.
5. **Render Python 3.14 build failure** — pydantic-core had no wheel for 3.14. Fixed by pinning Python 3.11.

---

## License

MIT
