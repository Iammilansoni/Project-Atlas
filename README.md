<div align="center">

# ◈ PROJECT ATLAS

### AI-Powered Learning Path Recommender

*Tell it what you want to master. Get a personalized 5-step roadmap — with reasoning.*

[![Live Demo](https://img.shields.io/badge/Live%20Demo-project--atlas--v1p3.onrender.com-1D9E75?style=for-the-badge&logo=render&logoColor=white)](https://project-atlas-v1p3.onrender.com/)
[![API Docs](https://img.shields.io/badge/API%20Docs-Swagger%20UI-orange?style=for-the-badge&logo=swagger&logoColor=white)](https://project-atlas-v1p3.onrender.com/docs)
[![Python](https://img.shields.io/badge/Python-3.11+-blue?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-teal?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![LangChain](https://img.shields.io/badge/LangChain-LCEL-purple?style=for-the-badge)](https://langchain.com)

</div>

---

## What is Project Atlas?

You type: *"I want to learn Machine Learning from scratch."*

Atlas searches 3,404 real Coursera courses using vector similarity, analyzes the gap between where you are and where you want to be, and returns a structured 5-step learning path — each course with a reason *why it's there*, *why it's at this step*, the skills it builds, and a realistic timeline.

No generic lists. No random recommendations. Reasoned, ordered, personalized paths.

---

## Demo

```
POST /recommend-path
{ "goal": "I want to learn Machine Learning from scratch" }
```

**Returns:**

```json
{
  "goal": "I want to learn Machine Learning from scratch",
  "skill_gap_analysis": {
    "current_level": "Beginner — no prior ML or programming assumed",
    "target_skills": ["Python", "Statistics", "ML Algorithms", "Deep Learning", "Model Deployment"],
    "missing_skills": ["Neural Networks", "Feature Engineering", "Model Evaluation"],
    "recommended_focus": "Start with Python and statistics before touching ML algorithms."
  },
  "learning_path": [
    {
      "step": 1,
      "course_title": "Machine Learning Specialization",
      "institution": "DeepLearning.AI",
      "why_this_course": "Establishes the core supervised learning foundations before any specialization. Andrew Ng's approach makes complex math accessible to beginners.",
      "why_at_this_step": "Everything else in this path depends on knowing this first.",
      "skills_gained": ["Supervised Learning", "Regression", "Neural Networks"],
      "duration": "3 months",
      "rating": 4.9,
      "level": "Beginner",
      "review_highlight": "Best ML course available online — clear, practical, and well-paced."
    }
    // ... 4 more steps
  ],
  "total_hours_estimate": "120–150 hours",
  "estimated_timeline": "3–4 months at 10 hrs/week",
  "pro_tip": "Build a mini-project after each course to cement the skills.",
  "llm_mode": "claude"
}
```

---

## How It Works

```
Your Goal (natural language)
         │
         ▼
  MiniLM Embedding          all-MiniLM-L6-v2 → 384-dim vector
         │
         ▼
  FAISS Vector Search        Similarity search across 3,404 Coursera courses → top 15
         │
         ▼
  Enrichment Layer           VADER sentiment analysis + review highlights injected
         │
         ▼
  LangChain LCEL Chain
    ├─ Tier 1 → Claude Sonnet        (primary — ChatAnthropic)
    ├─ Tier 2 → Gemini 2.5 Flash     (fallback — ChatOpenAI compat)
    └─ Tier 3 → Deterministic Ranker (zero-cost fallback, no API needed)
         │
         ▼
  Pydantic Response Model    Structured JSON — validated, typed, consistent
         │
         ▼
  FastAPI  POST /recommend-path
         │
         ▼
  Glassmorphism UI           Live at onrender.com
```

**Why 3 LLM tiers?** Production reliability. Claude Sonnet does the reasoning. If Anthropic's API is unavailable, Gemini 2.5 Flash takes over automatically via OpenAI-compatible endpoint. If both are down, the deterministic ranker returns a valid path without any API call. The system never goes offline.

---

## Tech Stack

| Layer | Technology | Why |
|---|---|---|
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` | Free, fast, runs on CPU, 384-dim — perfect for 3K courses |
| Vector DB | `LangChain FAISS` | No server needed, file-based, you've loaded 3,404 vectors |
| Primary LLM | `Claude Sonnet` via `ChatAnthropic` | Best reasoning quality for structured path generation |
| Fallback LLM | `Gemini 2.5 Flash` via `ChatOpenAI` (OpenAI-compat) | Auto-fallback, keeps system live if Anthropic is down |
| Chain | `LangChain LCEL` | Composable, async, clean prompt → LLM → parser flow |
| API | `FastAPI` | Auto Swagger docs, async, Pydantic validation |
| Enrichment | `VADER Sentiment` + `reviews.csv` | Injects real learner sentiment into LLM context |
| Dataset | Coursera (3,404 courses) | Real titles, descriptions, ratings, difficulty, duration |
| Deploy | `Render.com` | Free tier, auto-deploy from GitHub |

---

## Project Structure

```
project-atlas/
│
├── app/
│   ├── main.py                  # FastAPI app factory + lifespan + static serving
│   └── config.py                # Pydantic settings — reads .env
│
├── api/
│   ├── routes/
│   │   ├── recommend.py         # POST /recommend-path
│   │   └── health.py            # GET /health, /index/status
│   └── schemas/                 # Pydantic request + response models
│
├── models/                      # Course + LearningPath domain dataclasses
│
├── services/
│   ├── embedding_service.py     # LangChain MiniLM singleton
│   ├── enrichment_service.py    # VADER sentiment + review highlight join
│   ├── llm_service.py           # LCEL chain: prompt | llm | parser (3-tier)
│   └── rag_service.py           # Async pipeline orchestrator
│
├── utils/
│   ├── data_loader.py           # Coursera CSV → List[Course]
│   ├── faiss_index.py           # LangChain FAISS vectorstore wrapper
│   └── sentiment.py             # VADER scorer + highlight extractor
│
├── scripts/
│   └── build_index.py           # One-time FAISS index builder (run once)
│
├── frontend/
│   ├── index.html               # Single-page UI
│   ├── style.css                # Glassmorphism design system
│   └── app.js                   # Fetch API + interactive renderer
│
├── indexes/
│   └── faiss_store/             # Generated by build_index.py — git-ignored
│
├── .env.example                 # Copy this → .env, add your keys
├── requirements.txt
└── README.md
```

---

## Quick Start

### Prerequisites

- Python 3.11+
- At least one API key (optional — works without one via deterministic fallback)

### 1 — Clone and install

```bash
git clone https://github.com/Iammilansoni/Project-Atlas.git
cd Project-Atlas
pip install -r requirements.txt
```

> First install takes 3–5 minutes — PyTorch + sentence-transformers are ~2 GB.  
> Windows users: use `python -m pip install -r requirements.txt` if `pip` isn't found.

### 2 — Add your API keys

```bash
cp .env.example .env        # macOS / Linux
copy .env.example .env      # Windows
```

Open `.env` and fill in at least one:

```env
ANTHROPIC_API_KEY=sk-ant-...     # Claude Sonnet — primary LLM
GEMINI_API_KEY=AIza...           # Gemini 2.5 Flash — fallback LLM
```

Get your keys free:
- Claude (Anthropic): https://console.anthropic.com
- Gemini: https://aistudio.google.com/apikey

> No keys? The system still works using the built-in deterministic ranker.

### 3 — Build the FAISS index *(one-time, ~2–4 minutes)*

```bash
python scripts/build_index.py
```

Embeds all 3,404 Coursera courses with MiniLM and saves the LangChain FAISS vectorstore to `indexes/faiss_store/`. You only need to run this once.

### 4 — Start the server

```bash
uvicorn app.main:app --reload
```

Open **http://localhost:8000** — UI and API are both served from this single command.

---

## API Reference

### `POST /recommend-path`

Takes a learner's goal and returns a structured 5-step learning path.

**Request body:**
```json
{ "goal": "string" }
```

**Response fields:**

| Field | Type | Description |
|---|---|---|
| `goal` | string | The original input echoed back |
| `skill_gap_analysis` | object | Current level, target skills, missing skills, recommended focus |
| `learning_path` | array[5] | Ordered steps, each with course + reasoning |
| `learning_path[].why_this_course` | string | Why this specific course was selected |
| `learning_path[].why_at_this_step` | string | Why it belongs at this position in the sequence |
| `learning_path[].skills_gained` | array | What the learner will be able to do after completing it |
| `total_hours_estimate` | string | Estimated total effort |
| `estimated_timeline` | string | Realistic completion time at a given weekly pace |
| `pro_tip` | string | Personalized advice for this specific learning goal |
| `llm_mode` | string | Which tier generated the response: `claude`, `gemini`, or `deterministic` |

### `GET /health`

```json
{
  "status": "ok",
  "index_loaded": true,
  "model_loaded": true,
  "llm_mode": "claude",
  "course_count": 3404
}
```

### `GET /index/status`

Returns FAISS index path, vector count, and last-built timestamp.

### `GET /docs`

Auto-generated Swagger UI — try the API directly in the browser.

---

## LLM Fallback Logic

The system tries tiers in order. The first one that succeeds is used. The `llm_mode` field in every response tells you which tier ran.

```
Request received
      │
      ▼
  Claude Sonnet available? ──Yes──→ Generate path → return {llm_mode: "claude"}
      │No
      ▼
  Gemini 2.5 Flash available? ──Yes──→ Generate path → return {llm_mode: "gemini"}
      │No
      ▼
  Deterministic Ranker ──────────→ Generate path → return {llm_mode: "deterministic"}
```

This design means the API never returns a 503 due to an LLM outage.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `503 Index not built` | Run `python scripts/build_index.py` first |
| `pip` not found (Windows) | Use `python -m pip install -r requirements.txt` |
| `ModuleNotFoundError: langchain_anthropic` | Re-run `pip install -r requirements.txt` |
| Slow first request (~10s) | Normal — MiniLM loads on first query. Fast on all subsequent requests. |
| Claude returns an error | Check `ANTHROPIC_API_KEY` in `.env` — system auto-falls back to Gemini |
| Output shows `llm_mode: gemini` | Claude API key missing or quota exceeded — add `ANTHROPIC_API_KEY` to `.env` |

---

## About

Built as part of the KraftX Works CLAUDEpreneur trial — Project ATLAS (EdTech Vertical).

**Milan Soni** — Final year CS student, MERN + AI/ML developer, SIH 2023 Winner  
[GitHub](https://github.com/Iammilansoni) · [LinkedIn](https://linkedin.com/in/sonimilan) · milansoni96946@gmail.com
