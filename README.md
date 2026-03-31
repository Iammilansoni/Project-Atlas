# PROJECT ATLAS — AI-Powered Learning Path Recommender

> RAG pipeline · LangChain + FAISS · Claude Sonnet + Gemini 2.5 Flash · FastAPI + Glassmorphism UI

---

## 🧠 Architecture

```
User Goal
  │
  ▼
[LangChain HuggingFaceEmbeddings]   all-MiniLM-L6-v2 → 384-dim vector
  │
  ▼
[LangChain FAISS Vectorstore]       similarity_search_with_relevance_scores, top-15
  │
  ▼
[EnrichmentService]                 VADER sentiment + review highlights from reviews.csv
  │
  ▼
[LangChain LCEL Chain]
  ├─ Tier 1: ChatPromptTemplate | ChatAnthropic   (Claude Sonnet)
  ├─ Tier 2: ChatPromptTemplate | ChatOpenAI      (Gemini 2.5 Flash, OpenAI-compat)
  └─ Tier 3: Deterministic Ranker                 (zero-cost fallback)
       │ StrOutputParser → JSON → LearningPath domain model
  ▼
[FastAPI]     POST /recommend-path → LearningPathResponse (Pydantic)
  │
  ▼
[Frontend]    Dark glassmorphism UI at http://localhost:8000
```

---

## ⚙️ Prerequisites

| Requirement | Version | Check |
|---|---|---|
| Python | **3.11+** | `python --version` |
| pip | any | `python -m pip --version` |

> **Windows users:** if `pip` isn't recognised, always use `python -m pip` instead.

---

## ⚡ Quick Start

### 1. Install dependencies

```bash
cd project-atlas

# Windows (PowerShell / CMD)
python -m pip install -r requirements.txt

# macOS / Linux
pip install -r requirements.txt
```

> ⏳ First install takes 3–5 min — PyTorch + sentence-transformers are ~2 GB.

### 2. Configure API keys

```bash
# Windows
copy .env.example .env

# macOS / Linux
cp .env.example .env
```

Open `.env` and add at least one key:

```env
ANTHROPIC_API_KEY=sk-ant-...    # Primary: Claude Sonnet
GEMINI_API_KEY=AIza...          # Fallback: Gemini 2.5 Flash
```

> 💡 The system works **without any API key** using the deterministic ranker.

**Get free keys:**
- Anthropic (Claude): https://console.anthropic.com
- Gemini: https://aistudio.google.com/apikey ← uses your Google account

### 3. Build the FAISS index *(one-time, ~2–4 min)*

```bash
python scripts/build_index.py
```

Embeds all Coursera courses with LangChain MiniLM → saves LangChain FAISS vectorstore to `indexes/faiss_store/`.

### 4. Start the server

```bash
uvicorn app.main:app --reload
```

Open **http://localhost:8000** — the full UI and API are served from this single command.

---

## 🔌 API Reference

### `POST /recommend-path`

**Request:**
```json
{ "goal": "Learn Machine Learning from scratch" }
```

**Response:**
```json
{
  "goal": "Learn Machine Learning from scratch",
  "skill_gap_analysis": {
    "current_level": "Beginner",
    "target_skills": ["Python", "Statistics", "ML Algorithms", "Model Evaluation", "Deep Learning"],
    "missing_skills": ["Neural Networks", "Model Deployment", "Feature Engineering"],
    "recommended_focus": "Start with Python and statistics before touching ML algorithms."
  },
  "learning_path": [
    {
      "step": 1,
      "course_title": "Machine Learning Specialization",
      "institution": "DeepLearning.AI",
      "why_this_course": "This course establishes the core supervised learning foundations you need before any specialisation. Andrew Ng's pedagogical approach makes complex math accessible to beginners.",
      "skills_gained": ["Supervised Learning", "Regression", "Neural Networks"],
      "duration": "3 months",
      "rating": 4.9,
      "rating_label": "⭐ Exceptional",
      "review_highlight": "Best ML course available online — clear, practical, and well-paced.",
      "level": "Beginner"
    }
  ],
  "total_hours_estimate": "120-150 hours",
  "estimated_timeline": "3-4 months at 10 hrs/week",
  "pro_tip": "Build a mini-project after each course to cement the skills.",
  "llm_mode": "claude"
}
```

### `GET /health`
Returns system status, index loaded state, course count, and active LLM mode.

### `GET /index/status`
Returns FAISS index path, vector count, and last-built timestamp.

### `GET /docs`
Auto-generated Swagger UI for all endpoints.

---

## 🎚 LLM Fallback Tiers

| Tier | Engine | LangChain Class | Requires |
|------|--------|-----------------|----------|
| 1 | **Claude Sonnet** | `ChatAnthropic` | `ANTHROPIC_API_KEY` |
| 2 | **Gemini 2.5 Flash** | `ChatOpenAI` (OpenAI-compat) | `GEMINI_API_KEY` |
| 3 | **Deterministic Ranker** | — | Nothing |

---

## 🗂 Project Structure

```
project-atlas/
├── app/
│   ├── main.py              # FastAPI factory + lifespan + static serving
│   └── config.py            # Pydantic settings (reads .env)
├── api/
│   ├── routes/
│   │   ├── recommend.py     # POST /recommend-path
│   │   └── health.py        # GET /health, /index/status
│   └── schemas/             # Pydantic request/response models
├── models/                  # Course + LearningPath domain dataclasses
├── services/
│   ├── embedding_service.py # LangChain HuggingFaceEmbeddings singleton
│   ├── enrichment_service.py# VADER sentiment + review highlight join
│   ├── llm_service.py       # LangChain LCEL: prompt | llm | parser
│   └── rag_service.py       # Async pipeline orchestrator
├── utils/
│   ├── data_loader.py       # Coursera.csv → List[Course]
│   ├── faiss_index.py       # LangChain FAISS vectorstore wrapper
│   └── sentiment.py         # VADER scorer + highlight extractor
├── scripts/
│   └── build_index.py       # One-time LangChain FAISS index builder
├── frontend/
│   ├── index.html           # Dark glassmorphism single-page UI
│   ├── style.css            # Design system (CSS custom properties)
│   └── app.js               # Fetch API + interactive renderer
└── indexes/
    └── faiss_store/         # Generated by build_index.py (git-ignored)
```

---

## 🔧 Troubleshooting

| Problem | Fix |
|---|---|
| `pip` not found (Windows) | Use `python -m pip install -r requirements.txt` |
| `python` not found | Install Python 3.11+ from https://python.org — tick "Add to PATH" |
| `503 Index not built` | Run `python scripts/build_index.py` first |
| `ModuleNotFoundError: langchain_anthropic` | Re-run `python -m pip install -r requirements.txt` |
| Claude returns an error | Check `ANTHROPIC_API_KEY` in `.env`. System falls back to Gemini automatically. |
| Slow first request | Normal — MiniLM model loads on first query (~10s). Warm on subsequent requests. |
