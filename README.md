# hirring-project-s
---

## 🏗️ System Architecture & Workflow

markdown
# Agentic Compliance Auditor & Multi-Agent RAG Pipeline

A contract-review prototype built with **FastAPI** and **LangGraph**. It parses PDF documents, retrieves contract passages with a parent-child RAG design, and uses an auditor/critic workflow. Automated results are decision support only: citations, interpretation, and amendment precedence require human verification.

## 🚀 Key Architectural Highlights (The Hiring "Moat")
* **Stateful Cyclic Architecture:** Uses LangGraph to run an Auditor Agent and a Critic Agent, with bounded retries when review fails.
* **Parent-Child Chunking Strategy:** Dodge token truncation and loss-of-context issues by indexing granular child chunks ($400$ characters) for high-precision semantic matching while feeding larger parent contexts ($2000$ characters) to the LLM generation layer.
* **Asynchronous Streaming Gateway:** Written natively in FastAPI utilizing asynchronous I/O and server-side events (`StreamingResponse`) to stream token-by-token agent reasoning steps to the client.
* **Evaluation visibility:** Reports expose audit and citation-review status and stop with an incomplete/manual-review warning when either AI agent is unavailable.

---

## 🏗️ System Architecture & Workflow

Use code with caution.
┌──────────────────────────────────────────────────────────┐
│                  PDF Document Upload                    │
└─────────────────────────────┬────────────────────────────┘
│
[Parent-Child Chunking Engine]
│
▼
┌──────────────────────────────────────────────────────────┐
│               Qdrant Vector Database                     │
└─────────────────────────────┬────────────────────────────┘
│
[Semantic Query Fetch]
│
▼
┌─────────────────────────┐
┌─────►│  Auditor Agent (Node)   │
│      └────────────┬────────────┘
│                   │ Generates Structured Report
If "fail"  │                   ▼
Loop Retry └───────   Critic Agent (Node)   │
└────────────┬────────────┘
│ Validates Citations
│
▼ (Passes Quality Threshold)
┌─────────────────────────┐
│      Final Output       │
└─────────────────────────┘

---

## 🛠️ Deep Tech Stack & Ecosystem

* **Backend Engine:** FastAPI (Python 3.11+, fully asynchronous runtime execution)
* **Agentic Orchestration:** LangGraph (Stateful, multi-agent network topologies)
* **LLM Layer Integration:** LiteLLM with configurable timeout and retry/backoff settings
* **Vector Infrastructure:** Qdrant Cloud/Self-Hosted (Dense embeddings with metadata payload mapping)
* **Document Parsing Core:** PyPDF / Structured layout extraction
* **Guardrails:** Prompt-injection checks, budget limits, and deterministic citation validation; provider use is exposed in report metadata

---

## 📂 Production Codebase Structure

```text
agentic-compliance-auditor/
├── app/
│   ├── __init__.py
│   ├── main.py            # FastAPI Application Gateway & Streaming Endpoints
│   ├── config.py          # Unified Env/Pydantic-Settings Configurations
│   ├── agents/            # State Graphs, Node logic, and Routing functions
│   │   ├── __init__.py
│   │   ├── graph.py       # LangGraph Blueprint and conditional state definitions
│   │   └── nodes.py       # Auditor and Critic execution runtimes
│   └── services/          # Data Access and Analytical Utilities
│       ├── __init__.py
│       ├── vector_db.py   # Qdrant Indexing, Schema Management, and Parent-Child RAG
│       └── parser.py      # PDF parsing algorithms and window splitting logic
├── Dockerfile             # Application container
├── docker-compose.yml     # Local orchestration for FastAPI and Qdrant instances
├── requirements.txt       # Strict pin-controlled dependencies list
├── .env.example           # Safe environment-variable template; never put real keys in Git
└── tests/live_gemini_smoke.py # Optional live-model workflow check
```

---

## ⚡ Quickstart Setup (Cloud Dev / Local Docker)

This repository is optimized for isolated cloud environments (GitHub Codespaces, Gitpod, Project IDX) and local development. 

### 1. Environmental Variables Configuration
Copy `.env.example` to `.env` and add a valid OpenRouter API key locally. `.env` is ignored by Git; never commit real credentials.
```env
OPENROUTER_API_KEY=sk-or-v1-your-key
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
GEMINI_API_KEY=your-key          # optional: only used for Qdrant embeddings
```
The auditor and critic use free OpenRouter models. At runtime the app queries `GET /models`, keeps the `:free` entries, and tries up to five candidates (instruction-following families first) with bounded retries. If discovery fails, it falls back to the slugs listed in `FALLBACK_MODELS` inside `app/agents/nodes.py`. The model that produced the report is recorded in the report metadata.

`OPENROUTER_API_KEY` must start with `sk-or-v1-`. Do not hard-code the key inside `${...}` in `docker-compose.yml`; Docker Compose treats `${sk-or-v1-...}` as "variable `sk` with default `or-v1-...`", silently removes the `sk-` prefix, and OpenRouter then answers `401 Missing Authentication header`. Compose already reads `.env` from the project root, so the correct declaration is:
```yaml
environment:
  - OPENROUTER_API_KEY=${OPENROUTER_API_KEY}
```

Authentication failures (HTTP 401/403) abort the run immediately with an actionable message instead of retrying every candidate model. Rate limits, timeouts, and 5xx responses are retried; other 4xx model errors move on to the next free model.
If OpenRouter returns `free-models-per-day`, the account-wide daily free-model quota is exhausted; the app stops retrying because switching free models will not bypass it. Wait for the quota to reset or add credits to the OpenRouter account to unlock more free-model requests.

Run the local test suite with `pytest -q`. The auditor and critic require a live OpenRouter key; mock/local tests do not prove live-provider success.

If the auditor or critic is unavailable, the audit ends incomplete, citation review is marked `NOT RUN` when it did not execute, and the PDF withholds findings rather than substituting heuristic verdicts. Successful reports distinguish `Gap — Missing Protection` findings, include an executive summary, missing-risk descriptions, and per-finding confidence.

The Streamlit dashboard classifies uploaded contracts as consulting, SaaS/subscription, vendor/supply, NDA, or lease/real estate before auditing. Users can confirm or change the detected type. For consulting agreements, it flags breach-notification and data-retention rules as potentially inapplicable and lets the user remove those rules or run them anyway.

### 2. Stand Up Infrastructure
Spin up the backend API gateway alongside the Qdrant database clusters with a single command:
```bash
docker-compose up --build
```
The contract review dashboard is at `http://localhost:8000/dashboard`; upload a PDF, enter the rules, and click **Compile and audit contract** to download a PDF report. The API documentation is available at `http://localhost:8000/docs`, and the optional Streamlit UI is at `http://localhost:8501`.

---

## 🧪 Evaluation and client-facing claims

Do not present accuracy, latency, or cost-improvement percentages as measured results unless you have benchmark data to support them. The live smoke test records request duration for the synthetic test case; it is not a comparison against manual review and does not establish accuracy.

Client-facing materials and reports should state that this is automated decision support, not legal advice. A qualified reviewer should verify all findings, citations, and amendment precedence against the signed agreement.

