import json
import re
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, RedirectResponse, Response, StreamingResponse

from app.agents.graph import create_compliance_graph
from app.agents.nodes import auditor_node, citation_review_from_state, critic_node
from app.services.parser import DocumentParserService
from app.services.report_generator import AuditReportGenerator
from app.services.vector_db import VectorStoreService

app = FastAPI(
	title="Agentic Compliance Auditor Platform",
	description="Production-grade multi-agent document analysis gateway.",
	version="1.0.0",
)

GEMINI_AUTH_DETAIL = (
	"Gemini rejected GEMINI_API_KEY. Set it to a valid Google AI Studio API key, "
	"not an OAuth access token, then recreate the API container."
)


def is_gemini_auth_error(exc: Exception) -> bool:
	return (
		type(exc).__name__ == "AuthenticationError"
		or getattr(exc, "status_code", None) == 401
		or "ACCESS_TOKEN_TYPE_UNSUPPORTED" in str(exc)
		or "API_KEY_INVALID" in str(exc)
		or "API key not valid" in str(exc)
	)

app.add_middleware(
	CORSMiddleware,
	allow_origins=["*"],
	allow_credentials=True,
	allow_methods=["*"],
	allow_headers=["*"],
)

vector_service = VectorStoreService()
agent_graph = create_compliance_graph(auditor_node, critic_node)


@app.get("/", include_in_schema=False)
async def root() -> RedirectResponse:
	return RedirectResponse(url="/docs")


@app.get("/dashboard", response_class=HTMLResponse)
async def serve_dashboard() -> HTMLResponse:
	dashboard_path = Path(__file__).parent / "static" / "index.html"
	try:
		return HTMLResponse(content=dashboard_path.read_text(encoding="utf-8"))
	except FileNotFoundError as exc:
		raise HTTPException(status_code=404, detail="Static dashboard not found.") from exc


@app.get("/health")
async def health_check() -> dict[str, str]:
	return {"status": "healthy", "environment": "cloud_container"}


@app.post("/api/v1/auto-audit")
async def auto_audit(file: UploadFile = File(...), rules: str = Form(...)) -> Response:
	if not file.filename or not file.filename.lower().endswith(".pdf"):
		raise HTTPException(status_code=400, detail="Only PDF documents are supported.")

	parsed_rules = [rule.strip() for line in rules.splitlines() for rule in line.split(",") if rule.strip()]
	if not parsed_rules:
		raise HTTPException(status_code=400, detail="Enter at least one compliance rule.")

	try:
		file_bytes = await file.read()
		source_contract_text = DocumentParserService.extract_pdf_text(file_bytes)
		chunks = DocumentParserService.extract_and_chunk_pdf(file_bytes)
		if not chunks:
			raise HTTPException(status_code=400, detail="No readable text found in the PDF.")

		document_id = str(uuid.uuid4())
		vector_service.index_chunks(document_id, chunks)
		contexts = []
		for rule in parsed_rules:
			context = vector_service.query_relevant_context(rule, document_id, limit=3)
			if context and context not in contexts:
				contexts.append(context)
		contract_context = "\n\n---\n\n".join(contexts)
		raw_context_fallback = False
		if not contract_context:
			contract_context = source_contract_text
			raw_context_fallback = True

		initial_state = {
			"contract_text": contract_context,
			"source_contract_text": source_contract_text,
			"retrieval_context": contract_context,
			"compliance_rules": parsed_rules,
			"audit_report": {},
			"critic_feedback": "",
			"loop_count": 0,
			"status": "pending",
		}
		final_state = dict(initial_state)
		for node_output in agent_graph.stream(initial_state):
			for state_update in node_output.values():
				final_state.update(state_update)

		audit_report = dict(final_state.get("audit_report") or {})
		findings = audit_report.get("findings", [])
		if final_state.get("guardrail_status") == "blocked":
			compliance_status = "blocked"
		elif findings and all(finding.get("compliant") is True for finding in findings):
			compliance_status = "compliant"
		elif any(finding.get("compliant") is False for finding in findings):
			compliance_status = "non_compliant"
		else:
			compliance_status = "undetermined"

		citation_review = citation_review_from_state(final_state)
		analysis = dict(audit_report.get("analysis") or {})
		if raw_context_fallback:
			warning = "Vector search returned no matches; parsed contract text was used directly."
			analysis["warning"] = f"{analysis['warning']} {warning}".strip() if analysis.get("warning") else warning

		report_data = {
			**audit_report,
			"document_name": Path(file.filename).name,
			"compliance_status": compliance_status,
			"audit_completed": True,
			"citation_review": citation_review,
			"analysis": analysis,
			"rules_evaluated": len(parsed_rules),
			"chunks_indexed": len(chunks),
		}
		pdf_bytes = AuditReportGenerator.generate_pdf_report(Path(file.filename).name, report_data)
		filename_stem = Path(file.filename).stem
		filename_stem = re.sub(r"[^A-Za-z0-9_-]+", "_", filename_stem).strip("_") or "contract"
		return Response(
			content=pdf_bytes,
			media_type="application/pdf",
			headers={
				"Content-Disposition": f'attachment; filename="COMPLIANCE_AUDIT_{filename_stem}.pdf"',
				"X-Compliance-Status": compliance_status,
				"X-Citation-Review": str(citation_review["status"]),
				"X-Analysis-Mode": str(analysis.get("mode", "unknown")),
				"X-Audit-Completed": "true",
			},
		)
	except HTTPException:
		raise
	except Exception as exc:
		if is_gemini_auth_error(exc):
			raise HTTPException(status_code=502, detail=GEMINI_AUTH_DETAIL) from exc
		raise HTTPException(status_code=500, detail=f"Automated audit failed: {exc}") from exc


@app.post("/api/v1/ingest")
async def ingest_document(file: UploadFile = File(...)) -> dict[str, str | int]:
	if not file.filename or not file.filename.lower().endswith(".pdf"):
		raise HTTPException(status_code=400, detail="Only PDF documents are supported.")

	try:
		file_bytes = await file.read()
		document_id = str(uuid.uuid4())
		chunks = DocumentParserService.extract_and_chunk_pdf(file_bytes)
		if not chunks:
			raise HTTPException(status_code=400, detail="No readable text found in document.")
		vector_service.index_chunks(document_id, chunks)
	except HTTPException:
		raise
	except Exception as exc:
		if is_gemini_auth_error(exc):
			raise HTTPException(status_code=502, detail=GEMINI_AUTH_DETAIL) from exc
		raise HTTPException(status_code=500, detail=f"Ingestion failed: {exc}") from exc

	return {
		"status": "success",
		"document_id": document_id,
		"total_chunks_indexed": len(chunks),
	}


@app.post("/api/v1/audit")
async def run_audit(document_id: str = Form(...), rules: str = Form(...)) -> object:
	parsed_rules = [rule.strip() for rule in rules.split(",") if rule.strip()]
	if not parsed_rules:
		raise HTTPException(status_code=400, detail="At least one compliance rule is required.")

	try:
		contexts = []
		for rule in parsed_rules:
			context = vector_service.query_relevant_context(rule, document_id, limit=3)
			if context and context not in contexts:
				contexts.append(context)
		rag_context = "\n\n---\n\n".join(contexts)
		if not rag_context:
			raise HTTPException(status_code=404, detail="No matching indexed document context found.")

		initial_state = {
			"contract_text": rag_context,
			"retrieval_context": rag_context,
			"compliance_rules": parsed_rules,
			"audit_report": {},
			"critic_feedback": "",
			"loop_count": 0,
			"status": "pending",
		}
	except HTTPException:
		raise
	except Exception as exc:
		if is_gemini_auth_error(exc):
			raise HTTPException(status_code=502, detail=GEMINI_AUTH_DETAIL) from exc
		raise HTTPException(status_code=500, detail=f"Audit preparation failed: {exc}") from exc

	def agent_stream_generator():
		final_report = {}
		try:
			for output in agent_graph.stream(initial_state):
				for node_name, state_update in output.items():
					if "audit_report" in state_update:
						final_report = state_update["audit_report"]
					if node_name == "critic_agent":
						final_report["citation_review"] = citation_review_from_state(state_update)
					update = json.dumps({"node": node_name, "update": state_update})
					yield f"event: agent_update\ndata: {update}\n\n"
		except Exception as exc:
			detail = GEMINI_AUTH_DETAIL if is_gemini_auth_error(exc) else f"Audit failed: {exc}"
			yield f"event: error\ndata: {json.dumps({'detail': detail})}\n\n"
			return
		findings = final_report.get("findings", [])
		if final_report.get("guardrail_status") == "blocked":
			final_report["compliance_status"] = "blocked"
		elif findings and all(finding.get("compliant") is True for finding in findings):
			final_report["compliance_status"] = "compliant"
		elif any(finding.get("compliant") is False for finding in findings):
			final_report["compliance_status"] = "non_compliant"
		else:
			final_report["compliance_status"] = "undetermined"
		final_report["audit_completed"] = True
		if not final_report.get("citation_review"):
			final_report["citation_review"] = citation_review_from_state({})
		yield f"event: final_report\ndata: {json.dumps(final_report)}\n\n"

	return StreamingResponse(
		agent_stream_generator(),
		media_type="text/event-stream",
		headers={"Cache-Control": "no-cache"},
	)
