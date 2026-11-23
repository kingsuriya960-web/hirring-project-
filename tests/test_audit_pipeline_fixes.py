"""Regression tests for the audit pipeline fixes.

Covers: Gap findings instead of Undetermined, per-row risk/confidence,
critic citation-status logic (gap citations excluded), withholding unverified
reports, and the auditor's amendment/precedence rules.
"""

from pathlib import Path
import sys

from pypdf import PdfReader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import app.agents.nodes as nodes
from app.agents.graph import should_continue
from app.services.parser import DocumentParserService
from app.services.report_generator import AuditReportGenerator, summarize_findings

CONTRACT = (
    "SECTION 1. PAYMENT TERMS\n"
    "Customer shall pay each invoice within thirty (30) days of receipt.\n"
)


def _finding(**overrides):
    finding = {
        "rule": "Payment terms must be Net-30 or better.",
        "result": "Compliant",
        "priority": "Medium",
        "citation": "Customer shall pay each invoice within thirty (30) days of receipt.",
        "risk_if_missing": "",
        "confidence": "High",
        "confidence_reason": "Verbatim citation located.",
        "reasoning": "30 days <= 30 days.",
        "next_step": "No action required",
    }
    finding.update(overrides)
    return finding


def _state(findings, **overrides):
    state = {
        "contract_text": CONTRACT,
        "source_contract_text": CONTRACT,
        "compliance_rules": [item["rule"] for item in findings],
        "contract_type": "Consulting Agreement",
        "rule_alignment_warning": None,
        "audit_report": {"findings": findings},
        "critic_feedback": "",
        "citation_review_executed": True,
        "citation_review_status": "PASS",
        "citation_review_detail": "",
        "verified_citations": 1,
        "total_citations": 1,
        "model_used": "test/model:free",
        "ai_error": None,
        "loop_count": 1,
        "status": "pass",
    }
    state.update(overrides)
    return state


def _pdf_text(tmp_path, filename, state):
    path = tmp_path / filename
    AuditReportGenerator.generate_pdf_report(str(path), "contract.pdf", state)
    reader = PdfReader(str(path))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def test_undetermined_becomes_a_critical_gap_with_business_risk():
    findings = nodes._normalize_findings(
        [
            {
                "rule": "Aggregate liability must be at least 3x fees.",
                "result": "Undetermined",
                "citation": "",
            }
        ]
    )

    assert len(findings) == 1
    finding = findings[0]
    assert finding["result"] == "Gap — Missing Protection"
    assert finding["priority"] == "Critical"
    assert "uncapped" in finding["risk_if_missing"].lower()
    assert finding["next_step"] == "Add clause or negotiate before execution"
    assert "Contract contains no provision" in finding["reasoning"]
    assert finding["confidence"] == "High"


def test_confidence_defaults_reflect_evidence():
    findings = nodes._normalize_findings(
        [
            {
                "rule": "Payment terms must be Net-30 or better.",
                "result": "Compliant",
                "citation": "pay within thirty (30) days",
            },
            {
                "rule": "Vendor must maintain cyber insurance.",
                "result": "Gap",
                "citation": "",
            },
            {
                "rule": "Termination requires 30 days notice.",
                "result": "Non-Compliant",
                "citation": "",
            },
        ]
    )

    assert findings[0]["confidence"] == "High"
    assert findings[1]["confidence"] == "High"
    assert findings[2]["confidence"] == "Medium"
    assert findings[2]["priority"] == "High"


def test_critic_ignores_gap_citations_and_verifies_real_ones(monkeypatch):
    findings = [
        _finding(),
        _finding(
            rule="Vendor must maintain cyber insurance.",
            result="Gap — Missing Protection",
            priority="Critical",
            citation="No provision found",
            risk_if_missing="No insurance requirement.",
            next_step="Add clause or negotiate before execution",
        ),
    ]
    monkeypatch.setattr(
        nodes,
        "_call_free_llm_with_retry",
        lambda prompt, max_retries=2: (
            {"status": "pass", "verified_count": 99, "feedback": "ok"},
            "critic/model:free",
        ),
    )

    update = nodes.critic_node(
        {
            "contract_text": CONTRACT,
            "compliance_rules": [item["rule"] for item in findings],
            "audit_report": {"findings": findings},
            "ai_error": None,
        }
    )

    # The gap's "No provision found" text must not be treated as a citation,
    # and the model's unreliable count must not fail an exact-match audit.
    assert update["citation_review_executed"] is True
    assert update["citation_review_status"] == "PASS"
    assert update["status"] == "pass"
    assert update["total_citations"] == 1
    assert update["verified_citations"] == 1


def test_critic_failure_withholds_findings_and_marks_report_incomplete(monkeypatch):
    def boom(prompt, max_retries=2):
        raise RuntimeError("critic provider down")

    monkeypatch.setattr(nodes, "_call_free_llm_with_retry", boom)
    findings = [_finding()]

    update = nodes.critic_node(
        {
            "contract_text": CONTRACT,
            "compliance_rules": [item["rule"] for item in findings],
            "audit_report": {"findings": findings},
            "ai_error": None,
        }
    )

    assert update["status"] == "incomplete"
    assert update["citation_review_executed"] is False
    assert update["citation_review_status"] == "NOT RUN"
    assert "Report withheld" in update["citation_review_detail"]
    assert update["audit_report"]["findings"] == []
    assert "Critic Agent unavailable" in update["ai_error"]
    assert should_continue(update) == "end"


def test_auditor_prompt_requires_amendment_precedence_and_gap_vocabulary(
    monkeypatch,
):
    captured = {}

    def fake(prompt, max_retries=2):
        captured["prompt"] = prompt
        return ({"findings": [_finding()]}, "auditor/model:free")

    monkeypatch.setattr(nodes, "_call_free_llm_with_retry", fake)

    nodes.auditor_node(
        {
            "contract_text": CONTRACT,
            "compliance_rules": ["Payment terms must be Net-30 or better."],
            "contract_type": "Consulting Agreement",
            "loop_count": 0,
        }
    )

    prompt = captured["prompt"].lower()
    assert "amendment" in prompt
    assert "exhibit" in prompt
    assert "cross-reference" in prompt
    assert "gap — missing protection" in prompt
    assert "undetermined" in prompt
    assert "confidence" in prompt
    assert '"type": "or_compound"' in prompt
    assert "never merge separate input rules" in prompt


def test_auditor_rejects_merged_or_dropped_rules(monkeypatch):
    monkeypatch.setattr(
        nodes,
        "_call_free_llm_with_retry",
        lambda prompt, max_retries=2: (
            {"findings": [_finding()]},
            "auditor/model:free",
        ),
    )

    update = nodes.auditor_node(
        {
            "contract_text": CONTRACT,
            "compliance_rules": [
                "Payment terms must be Net-30 or better.",
                "Liability cap must be at least 3x fees.",
            ],
        }
    )

    assert update["status"] == "error"
    assert update["audit_report"]["findings"] == []
    assert "Finding count mismatch" in update["ai_error"]


def test_executive_summary_reports_gaps_and_recommendation():
    summary = summarize_findings(
        [
            _finding(),
            _finding(
                rule="Aggregate liability must be at least 3x fees.",
                result="Gap — Missing Protection",
                priority="Critical",
                citation="",
                risk_if_missing="Uncapped financial exposure.",
                next_step="Add clause or negotiate before execution",
            ),
        ]
    )

    assert summary["counts"]["gap"] == 1
    assert summary["counts"]["critical"] == 1
    assert summary["recommendation"] == "DO NOT EXECUTE without legal review"
    assert "1 critical" in summary["risk_summary"]
    assert summary["key_issue"] == "Uncapped financial exposure."


def test_unverified_report_withholds_findings(tmp_path):
    findings = [
        _finding(),
        _finding(
            rule="Aggregate liability must be at least 3x fees.",
            result="Gap — Missing Protection",
            priority="Critical",
            citation="",
            risk_if_missing="Uncapped financial exposure.",
            next_step="Add clause or negotiate before execution",
        ),
    ]
    state = _state(
        findings,
        status="incomplete",
        citation_review_executed=False,
        citation_review_status="NOT RUN",
        citation_review_detail="Critic Agent unavailable; findings are UNVERIFIED.",
    )

    text = _pdf_text(tmp_path, "unverified.pdf", state)

    assert "ANALYSIS STATUS: INCOMPLETE" in text
    assert "MANUAL REVIEW REQUIRED" in text
    assert "No findings are presented" in text
    assert "Aggregate liability" not in text
    assert "Uncapped" not in text
    assert "All citations exactly match contract text" not in text


def test_verified_report_claims_exact_citations_only_when_passed(tmp_path):
    findings = [_finding()]
    state = _state(findings)

    text = _pdf_text(tmp_path, "verified.pdf", state)

    assert "Citation review: PASS" in text
    assert "All citations exactly match contract text" in text
    assert "UNVERIFIED" not in text
    assert "ANALYSIS STATUS: INCOMPLETE" not in text


def test_auditor_failure_produces_incomplete_banner_without_findings(tmp_path):
    state = _state(
        [],
        status="error",
        ai_error="All free OpenRouter models failed after retries.",
        citation_review_executed=False,
        citation_review_status="NOT RUN",
    )

    text = _pdf_text(tmp_path, "failed.pdf", state)

    assert "ANALYSIS STATUS: INCOMPLETE" in text
    assert "All free OpenRouter models failed" in text
    assert "Compliance rule" not in text


def test_rule_alignment_warning_flags_consulting_mismatch():
    rules = [
        "Payment terms must be Net-30.",
        "Breach notification must occur within 24 hours.",
        "Data retention must be at least 2 years.",
    ]

    mismatched = DocumentParserService.mismatched_rules("Consulting Agreement", rules)
    warning = DocumentParserService.rule_alignment_warning(
        "Consulting Agreement", rules
    )

    assert mismatched == rules[1:]
    assert warning is not None
    assert "Consulting Agreement" in warning
    assert "Remove or confirm" in warning


def test_api_routes_for_dashboard_and_audit_are_registered():
    # Inspected from source because importing FastAPI depends on the local
    # starlette version, which is pinned only inside the Docker image.
    source = (REPO_ROOT / "app" / "main.py").read_text(encoding="utf-8")

    assert '@app.get("/dashboard"' in source
    assert '@app.post("/api/v1/auto-audit"' in source
    assert '@app.get("/health"' in source
    assert "render_report_pdf(" in source
    assert "run_in_threadpool(run_graph, initial_state)" in source


def test_compose_and_env_still_forward_the_openrouter_key():
    compose = (REPO_ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    assert compose.count("OPENROUTER_API_KEY=${OPENROUTER_API_KEY}") == 2
