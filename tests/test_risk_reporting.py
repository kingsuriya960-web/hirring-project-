from io import BytesIO
import os
import sys

import pytest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import app.agents.nodes as nodes
from app.services.parser import DocumentParserService


def report_tools():
    pytest.importorskip("reportlab")
    from pypdf import PdfReader

    from app.services.report_generator import AuditReportGenerator

    return PdfReader, AuditReportGenerator


def test_contract_type_classifier_recognizes_supported_types():
    cases = {
        "CONSULTING AGREEMENT for professional services": "Consulting Agreement",
        "Software as a Service subscription agreement": "SaaS / Subscription Agreement",
        "Vendor supply agreement for goods": "Vendor / Supply Agreement",
        "Non-disclosure agreement for confidential information": "NDA",
        "Lease agreement between landlord and tenant": "Lease / Real Estate",
    }

    for contract_text, expected_type in cases.items():
        assert DocumentParserService.classify_contract_type(contract_text) == expected_type

    assert DocumentParserService.classify_contract_type("General commercial terms") is None


def test_legacy_undetermined_is_normalized_to_critical_gap():
    finding = nodes._normalize_findings(
        [{
            "rule": "Liability cap",
            "result": "Undetermined",
            "citation": "",
        }]
    )[0]

    assert finding["result"] == "Gap — Missing Protection"
    assert finding["priority"] == "Critical"
    assert finding["reasoning"] == (
        "Contract contains no provision for Liability cap. "
        "Without this protection, your exposure is uncapped/unregulated."
    )
    assert finding["next_step"] == "Add clause or negotiate before execution"
    assert "uncapped" in finding["risk_if_missing"]


def test_primary_model_failure_returns_no_findings(monkeypatch):
    def fail_auditor(_prompt):
        raise RuntimeError("model unavailable")

    monkeypatch.setattr(nodes, "_call_free_llm_with_retry", fail_auditor)
    state_update = nodes.auditor_node({
        "contract_text": "Payment is due Net-30.",
        "compliance_rules": ["Payment must be Net-30."],
    })

    assert state_update["status"] == "error"
    assert state_update["audit_report"]["findings"] == []
    assert state_update["ai_error"] == "model unavailable"


def test_critic_failure_is_not_converted_to_heuristic_pass(monkeypatch):
    def fail_critic(_prompt):
        raise RuntimeError("critic service unavailable")

    monkeypatch.setattr(nodes, "_call_free_llm_with_retry", fail_critic)
    state_update = nodes.critic_node({
        "audit_report": {
            "findings": [{"rule": "Payment", "citation": "Net-30"}],
        },
        "contract_text": "Payment is due Net-30.",
        "compliance_rules": ["Payment must be Net-30."],
    })

    assert state_update["status"] == "incomplete"
    assert state_update["citation_review_status"] == "NOT RUN"
    assert state_update["citation_review_executed"] is False
    assert "Critic Agent unavailable" in state_update["ai_error"]
    assert state_update["audit_report"]["findings"] == []


def test_critic_requires_reported_and_exact_citation_matches(monkeypatch):
    monkeypatch.setattr(
        nodes,
        "_call_free_llm_with_retry",
        lambda _prompt: (
            {"status": "pass", "verified_count": 1, "feedback": "Looks good."},
            "test model",
        ),
    )
    state_update = nodes.critic_node({
        "audit_report": {
            "findings": [
                {"rule": "Payment", "citation": "Invented Net-30 clause."},
                {"rule": "Liability cap", "citation": ""},
            ],
        },
        "contract_text": "Payment is due Net-30.",
        "compliance_rules": ["Payment must be Net-30.", "Liability cap"],
    })

    assert state_update["citation_review_executed"] is True
    assert state_update["citation_review_status"] == "FAIL"
    assert state_update["status"] == "fail"
    assert state_update["total_citations"] == 2
    assert state_update["audit_report"]["findings"] == []


def test_critic_passes_only_verified_nonempty_citations(monkeypatch):
    monkeypatch.setattr(
        nodes,
        "_call_free_llm_with_retry",
        lambda _prompt: (
            {"status": "pass", "verified_count": 1, "feedback": "Citation verified."},
            "test model",
        ),
    )
    state_update = nodes.critic_node({
        "audit_report": {
            "findings": [
                {"rule": "Payment must be Net-30.", "citation": "Payment is due Net-30."},
                {
                    "rule": "Liability cap",
                    "result": "Gap — Missing Protection",
                    "citation": "",
                },
            ],
        },
        "contract_text": "Payment is due Net-30.",
        "compliance_rules": ["Payment must be Net-30.", "Liability cap"],
    })

    assert state_update["status"] == "pass"
    assert state_update["citation_review_executed"] is True
    assert state_update["citation_review_status"] == "PASS"
    assert state_update["verified_citations"] == 1
    assert state_update["total_citations"] == 1


def test_critic_requires_literal_citation_and_checks_full_source_text(monkeypatch):
    monkeypatch.setattr(
        nodes,
        "_call_free_llm_with_retry",
        lambda _prompt: (
            {"status": "pass", "feedback": "Looks good."},
            "test model",
        ),
    )
    state_update = nodes.critic_node({
        "audit_report": {
            "findings": [{
                "rule": "Payment",
                "citation": "Payment is due  Net-30.",
            }],
        },
        "contract_text": "Payment is due",
        "source_contract_text": "Payment is due Net-30.",
        "compliance_rules": ["Payment"],
    })

    assert state_update["citation_review_status"] == "FAIL"
    assert state_update["verified_citations"] == 0
    assert state_update["audit_report"]["findings"] == []


def test_critic_passes_verbatim_citation_from_full_source(monkeypatch):
    monkeypatch.setattr(
        nodes,
        "_call_free_llm_with_retry",
        lambda _prompt: (
            {"status": "pass", "feedback": "Looks good."},
            "test model",
        ),
    )
    source = "Opening text. Payment is due Net-30. Closing text."
    state_update = nodes.critic_node({
        "audit_report": {
            "findings": [{
                "rule": "Payment",
                "citation": "Payment is due Net-30.",
            }],
        },
        "contract_text": "Opening text.",
        "source_contract_text": source,
        "compliance_rules": ["Payment"],
    })

    assert state_update["citation_review_status"] == "PASS"
    assert state_update["verified_citations"] == 1


def test_report_withholds_findings_when_citation_review_did_not_run(tmp_path):
    PdfReader, AuditReportGenerator = report_tools()
    report_path = tmp_path / "incomplete.pdf"
    AuditReportGenerator.generate_pdf_report(
        str(report_path),
        "contract.pdf",
        {
            "status": "error",
            "ai_error": "Critic Agent unavailable",
            "citation_review_executed": False,
            "citation_review_status": "NOT RUN",
            "audit_report": {
                "findings": [{
                    "rule": "Liability cap",
                    "result": "Gap — Missing Protection",
                }]
            },
        },
    )
    text = "\n".join(
        page.extract_text() or ""
        for page in PdfReader(BytesIO(report_path.read_bytes())).pages
    )

    assert "ANALYSIS STATUS: INCOMPLETE" in text
    assert "Manual review required" in text
    assert "DO NOT rely on this output" in text
    assert "All citations exactly match contract text" not in text
    assert "Liability cap" not in text


def test_report_withholds_findings_when_citation_review_fails(tmp_path):
    PdfReader, AuditReportGenerator = report_tools()
    report_path = tmp_path / "failed-citations.pdf"
    AuditReportGenerator.generate_pdf_report(
        str(report_path),
        "contract.pdf",
        {
            "status": "incomplete",
            "citation_review_executed": True,
            "citation_review_status": "FAIL",
            "citation_review_detail": "0/1 citations matched verbatim.",
            "audit_report": {
                "findings": [{
                    "rule": "Payment terms",
                    "result": "Non-Compliant",
                    "citation": "Paraphrased citation",
                    "reasoning": "Actionable but unverified.",
                    "next_step": "Renegotiate immediately.",
                }]
            },
        },
    )
    text = "\n".join(
        page.extract_text() or ""
        for page in PdfReader(BytesIO(report_path.read_bytes())).pages
    )

    assert "ANALYSIS STATUS: INCOMPLETE" in text
    assert "MANUAL REVIEW REQUIRED" in text
    assert "Citation verification failed" in text
    assert "Payment terms" not in text
    assert "Renegotiate immediately" not in text


def test_report_includes_executive_summary_gap_risk_and_confidence(tmp_path):
    PdfReader, AuditReportGenerator = report_tools()
    report_path = tmp_path / "complete.pdf"
    AuditReportGenerator.generate_pdf_report(
        str(report_path),
        "contract.pdf",
        {
            "status": "pass",
            "contract_type": "SaaS / Subscription Agreement",
            "model_used": "test model",
            "citation_review_executed": True,
            "citation_review_status": "PASS",
            "verified_citations": 1,
            "total_citations": 1,
            "audit_report": {
                "findings": [
                    {
                        "rule": "Liability cap",
                        "result": "Gap — Missing Protection",
                        "priority": "Critical",
                        "citation": "",
                        "reasoning": "No liability cap was identified.",
                        "risk_if_missing": "Potentially uncapped financial exposure.",
                        "confidence": "High",
                        "confidence_reason": "No liability cap language appears in the supplied text.",
                        "next_step": "Add clause or negotiate before execution",
                    },
                    {
                        "rule": "Payment terms",
                        "result": "Compliant",
                        "priority": "Low",
                        "citation": "Invoices are payable on Net-30 terms.",
                        "reasoning": "The payment term matches the rule.",
                        "risk_if_missing": "",
                        "confidence": "High",
                        "confidence_reason": "Exact payment text was identified.",
                        "next_step": "No action required",
                    },
                ]
            },
        },
    )
    text = "\n".join(
        page.extract_text() or ""
        for page in PdfReader(BytesIO(report_path.read_bytes())).pages
    )

    assert "RECOMMENDATION: DO NOT EXECUTE without legal review" in text
    assert "VERDICT SUMMARY: 1 compliant, 0 non-compliant, 1 missing-protection gaps" in text
    assert "RISK SEVERITY: 1 critical, 0 high, 0 medium, 1 low" in text
    assert "Gap — Missing Protection" in text
    assert "Potentially uncapped financial exposure" in text
    assert "High" in text
    assert "No liability cap" in text
    assert "language" in text
    assert "All citations exactly match contract text" in text


def test_auditor_prompt_receives_amendment_exhibit_and_defined_terms(monkeypatch):
    contract = (
        "SECTION 2: Payment is Net-30.\n"
        "EXHIBIT A: Enterprise tier payment is Net-15 and governs in case of conflict "
        "with Section 2.\n"
        "AMENDMENT NO. 1: Section 3 is amended; breach notice is 48 hours.\n"
        "SECTION 3: Breach notice is 5 days.\n"
        "SECTION 7.2: Cure Period means 10 days.\n"
        "SECTION 8: Provider must act within the Cure Period."
    )
    captured = {}

    def fake_completion(prompt):
        captured["prompt"] = prompt
        return (
            {
                "findings": [{
                    "rule": "Payment terms",
                    "result": "Compliant",
                    "citation": "EXHIBIT A: Enterprise tier payment is Net-15",
                    "reasoning": "The exhibit expressly controls the conflict.",
                    "confidence": "High",
                    "confidence_reason": "The exhibit states both the term and precedence.",
                }]
            },
            "test model",
        )

    monkeypatch.setattr(nodes, "_call_free_llm_with_retry", fake_completion)
    nodes.auditor_node({
        "contract_text": contract,
        "contract_type": "SaaS / Subscription Agreement",
        "compliance_rules": ["Payment terms must be Net-15."],
    })

    assert "AMENDMENT NO. 1" in captured["prompt"]
    assert "EXHIBIT A" in captured["prompt"]
    assert "Cure Period means 10 days" in captured["prompt"]
    assert "explicit amendment or exhibit precedence clause controls" in captured["prompt"]


def test_report_marks_nested_heuristic_fallback_without_verified_label(tmp_path):
    PdfReader, AuditReportGenerator = report_tools()
    report_path = tmp_path / "fallback.pdf"
    AuditReportGenerator.generate_pdf_report(
        str(report_path),
        "contract.pdf",
        {
            "status": "pass",
            "model_used": "local heuristic",
            "citation_review_executed": True,
            "citation_review_status": "PASS",
            "verified_citations": 1,
            "total_citations": 1,
            "audit_report": {
                "analysis": {"mode": "heuristic_fallback", "model": "local heuristic"},
                "findings": [
                    {
                        "rule": "Payment terms",
                        "result": "Compliant",
                        "priority": "Low",
                        "citation": "Invoices are payable on Net-30 terms.",
                        "reasoning": "The payment term matches the rule.",
                        "next_step": "No action required",
                    }
                ],
            },
        },
    )
    text = "\n".join(
        page.extract_text() or ""
        for page in PdfReader(BytesIO(report_path.read_bytes())).pages
    )

    assert "AI ANALYSIS UNAVAILABLE" in text
    assert "NOT generated by the compliance AI" in text
    assert "FALLBACK — NOT AI-VERIFIED" in text
    assert "[VERIFIED]" not in text
    assert "STATUS NOTE" in text
    assert "VERDICT SUMMARY" in text
    assert "RISK SEVERITY" in text
