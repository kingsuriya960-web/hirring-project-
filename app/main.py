"""FastAPI gateway for the multi-agent compliance auditor.

The static dashboard in ``app/static/index.html`` posts a PDF plus a rule set to
``/api/v1/auto-audit`` and expects a generated PDF report in the response.
"""

import json
import os
import re
import tempfile
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, RedirectResponse, Response, StreamingResponse
from starlette.concurrency import run_in_threadpool

from app.agents.graph import create_compliance_graph
from app.agents.nodes import auditor_node, critic_node, gatekeeper_node
from app.services.parser import DocumentParserService
from app.services.report_generator import AuditReportGenerator

STATIC_DIR = Path(__file__).parent / "static"
MAX_CONTRACT_CHARS = 35000

app = FastAPI(
    title="Agentic Compliance Auditor Platform",
    description="Multi-agent document analysis gateway (auditor + critic).",
    version="1.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_GRAPH = None
_VECTOR_STORE = None


def get_graph():
    """Compiles the LangGraph workflow once and reuses it across requests."""
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = create_compliance_graph(auditor_node, critic_node, gatekeeper_node)
    return _GRAPH


def get_vector_store():
    """Lazily connects to Qdrant; vector search is an optional enhancement."""
    global _VECTOR_STORE
    if _VECTOR_STORE is None:
        from app.services.vector_db import VectorStoreService

        _VECTOR_STORE = VectorStoreService()
    return _VECTOR_STORE


def parse_rules(raw_rules: str) -> list[str]:
    """Treat each non-empty line as one complete rule without splitting clauses."""
    rules = [
        line.strip()
        for line in (raw_rules or "").replace("\r\n", "\n").split("\n")
        if line.strip()
    ]
    if not rules:
        raise HTTPException(status_code=400, detail="Enter at least one compliance rule.")
    return rules


def finding_counts(final_state: Dict[str, Any], rules_count: int) -> Dict[str, int]:
    report = final_state.get("audit_report") or {}
    findings = report.get("findings") if isinstance(report, dict) else []
    findings = [item for item in (findings or []) if isinstance(item, dict)]
    reviewed = sum(
        1
        for item in findings
        if not str(item.get("result", "")).lower().startswith("gap")
        and str(item.get("citation") or "").strip()
    )
    try:
        verified = int(final_state.get("verified_citations") or 0)
    except (TypeError, ValueError):
        verified = 0
    counts = {
        "rules": rules_count,
        "citations_reviewed": reviewed,
        "citations_verified": max(0, min(verified, reviewed)),
        "compliant": 0,
        "non_compliant": 0,
        "gap": 0,
    }
    for finding in findings:
        result = str(finding.get("result", "")).strip().lower()
        if result == "compliant":
            counts["compliant"] += 1
        elif result == "non-compliant":
            counts["non_compliant"] += 1
        elif result.startswith("gap"):
            counts["gap"] += 1
        else:
            counts["gap"] += 1
    return counts


def analysis_is_fallback(final_state: Dict[str, Any]) -> bool:
    model = str(final_state.get("model_used") or "").lower()
    report_analysis = (
        (final_state.get("audit_report") or {}).get("analysis")
        if isinstance(final_state.get("audit_report"), dict)
        else {}
    )
    mode = str(
        report_analysis.get("mode") if isinstance(report_analysis, dict) else ""
    ).lower()
    return (
        "heuristic" in model
        or "fallback" in model
        or mode == "heuristic_fallback"
    )


def render_report_pdf(document_name: str, final_state: Dict[str, Any]) -> bytes:
    """Writes the report to a temporary file and returns its bytes."""
    handle, path = tempfile.mkstemp(prefix="veritas_report_", suffix=".pdf")
    os.close(handle)
    try:
        AuditReportGenerator.generate_pdf_report(path, document_name, final_state)
        return Path(path).read_bytes()
    finally:
        Path(path).unlink(missing_ok=True)


def safe_filename_stem(filename: Optional[str]) -> str:
    stem = Path(filename or "contract").stem
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", stem).strip("_")
    return cleaned or "contract"


def run_graph(initial_state: Dict[str, Any]) -> Dict[str, Any]:
    """Runs the LangGraph workflow and merges every node update into one state."""
    final_state = dict(initial_state)
    for output in get_graph().stream(initial_state):
        for state_update in output.values():
            if isinstance(state_update, dict):
                final_state.update(state_update)
    return final_state


def build_contract_context(
    file_bytes: bytes, rules: list, sources: Dict[str, Any]
) -> str:
    """Uses optional Qdrant retrieval when available, otherwise the full text."""
    full_text = sources["source_text"]
    try:
        store = get_vector_store()
        chunks = DocumentParserService.extract_and_chunk_pdf(file_bytes)
        if not chunks:
            return full_text
        document_id = str(uuid.uuid4())
        store.index_chunks(document_id, chunks)
        contexts = []
        for rule in rules:
            context = store.query_relevant_context(rule, document_id, limit=3)
            if context and context not in contexts:
                contexts.append(context)
        if contexts:
            sources["rag_used"] = True
            return "\n\n---\n\n".join(contexts)
    except Exception:
        # Vector search is an optional enhancement; the audit can run on raw text.
        pass
    return full_text


@app.get("/", include_in_schema=False)
async def root() -> RedirectResponse:
    return RedirectResponse(url="/dashboard")


@app.get("/dashboard", response_class=HTMLResponse)
async def serve_dashboard() -> HTMLResponse:
    dashboard_path = STATIC_DIR / "index.html"
    if not dashboard_path.is_file():
        raise HTTPException(status_code=404, detail="Static dashboard not found.")
    return HTMLResponse(content=dashboard_path.read_text(encoding="utf-8"))


@app.get("/health")
async def health_check() -> dict:
    return {"status": "healthy", "environment": "container"}


@app.post("/api/v1/auto-audit")
async def auto_audit(file: UploadFile = File(...), rules: str = Form(...)) -> Response:
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF documents are supported.")

    parsed_rules = parse_rules(rules)
    try:
        file_bytes = await file.read()
    except Exception as exc:
        raise HTTPException(
            status_code=400, detail=f"Could not read the uploaded file: {exc}"
        ) from exc

    try:
        contract_pages = DocumentParserService.extract_pdf_pages(file_bytes)
        source_text = DocumentParserService.format_pdf_pages(contract_pages)
    except Exception as exc:
        raise HTTPException(
            status_code=400, detail=f"Could not parse the PDF: {exc}"
        ) from exc
    if not source_text.strip():
        raise HTTPException(
            status_code=400, detail="No readable text found in the PDF."
        )

    contract_type = DocumentParserService.classify_contract_type(source_text)
    rule_alignment_warning = DocumentParserService.rule_alignment_warning(
        contract_type, parsed_rules
    )

    sources = {"source_text": source_text, "rag_used": False}
    try:
        contract_text = await run_in_threadpool(
            build_contract_context, file_bytes, parsed_rules, sources
        )
    except Exception:
        contract_text = source_text
    contract_text = contract_text[:MAX_CONTRACT_CHARS]

    initial_state = {
        "contract_text": contract_text,
        "source_contract_text": source_text,
        "source_pages": contract_pages,
        "retrieval_context": contract_text,
        "compliance_rules": parsed_rules,
        "contract_type": contract_type,
        "rule_alignment_warning": rule_alignment_warning,
        "total_pages": len(contract_pages),
        "audit_report": {},
        "critic_feedback": "",
        "citation_review_executed": False,
        "citation_review_status": "NOT RUN",
        "citation_review_detail": "",
        "verified_citations": 0,
        "total_citations": 0,
        "model_used": "",
        "ai_error": None,
        "loop_count": 0,
        "status": "pending",
    }

    try:
        # The graph makes blocking HTTP calls; keep the event loop responsive.
        final_state = await run_in_threadpool(run_graph, initial_state)
    except Exception as exc:
        raise HTTPException(
            status_code=502, detail=f"Automated audit failed: {exc}"
        ) from exc

    final_state["contract_type"] = contract_type
    final_state["rule_alignment_warning"] = rule_alignment_warning

    if final_state.get("guardrail_status") == "blocked":
        raise HTTPException(
            status_code=400,
            detail=(
                "Audit blocked by the prompt-injection guardrail: "
                f"{final_state.get('guardrail_reason') or 'malicious instruction detected'}"
            ),
        )

    if final_state.get("ai_error"):
        raise HTTPException(
            status_code=502,
            detail=(
                "Automated audit could not be completed: "
                f"{final_state['ai_error']}"
            ),
        )
    has_findings = bool((final_state.get("audit_report") or {}).get("findings"))
    review_incomplete = (
        final_state.get("status") in {"incomplete", "fail"}
        and final_state.get("citation_review_status") != "PASS"
    )
    if not has_findings and not review_incomplete:
        raise HTTPException(
            status_code=502,
            detail="Automated audit returned no findings; the report was not generated.",
        )

    try:
        pdf_bytes = await run_in_threadpool(
            render_report_pdf, file.filename, final_state
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"Report generation failed: {exc}"
        ) from exc

    counts = finding_counts(final_state, len(parsed_rules))
    report_name = f"COMPLIANCE_AUDIT_{safe_filename_stem(file.filename)}.pdf"
    fallback = analysis_is_fallback(final_state)
    headers = {
        "Content-Disposition": f'attachment; filename="{report_name}"',
        "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
        "Pragma": "no-cache",
        "X-Audit-Run-Id": uuid.uuid4().hex,
        "X-Finding-Counts": json.dumps(counts),
        "X-Rule-Alignment": "warning" if rule_alignment_warning else "ok",
        "X-Citation-Review": str(
            final_state.get("citation_review_status") or "NOT RUN"
        ).encode("ascii", "ignore").decode("ascii"),
        "X-Model-Used": str(final_state.get("model_used") or "unknown")
        .encode("ascii", "ignore")
        .decode("ascii"),
        "X-Analysis-Status": (
            "fallback"
            if fallback
            else (
                "verified"
                if final_state.get("citation_review_status") == "PASS"
                and final_state.get("status") == "pass"
                else "incomplete"
            )
        ),
    }
    return Response(content=pdf_bytes, media_type="application/pdf", headers=headers)
