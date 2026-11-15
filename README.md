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
* **Evaluation visibility:** Reports expose the analysis and citation-review modes, token/cost estimates, and prominent warnings when local fallback logic is used.

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
Copy `.env.example` to `.env` and add a valid Google AI Studio API key locally. `.env` is ignored by Git; never commit real credentials.
```env
GEMINI_API_KEY=your-key
```
The Gemini primary model and comma-separated fallback models are configurable through `GEMINI_MODEL` and `GEMINI_FALLBACK_MODELS`. The defaults use `gemini-3.8-flash` first, then `gemini-3.6-flash` and `gemini-3.5-flash`. On quota exhaustion or provider unavailability, the auditor and critic move to the next configured Gemini model; connection timeouts use the configured retries and exponential backoff (`GEMINI_TIMEOUT_SECONDS`, `GEMINI_MAX_RETRIES`, and `GEMINI_RETRY_BACKOFF_SECONDS`). The selected model is recorded in report metadata.

Run the local test suite with `pytest -q`. To make 5–10 live requests using only the script's synthetic sample contract, run `python tests/live_gemini_smoke.py --runs 5`. This live check requires your local API key and may incur provider charges. It reports whether both auditor and critic used Gemini; do not treat mock/local tests as evidence of live-provider success.

If every configured Gemini model is unavailable, the API returns HTTP 503 and does not produce or deliver heuristic findings. Check model availability and quota in Google AI Studio, then retry or configure another Gemini model available to your account.

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

