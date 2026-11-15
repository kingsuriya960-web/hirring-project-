import os
import sys
from io import BytesIO
from types import SimpleNamespace

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.agents.nodes import (
    _evaluate_rules_heuristically,
    citation_review_from_state,
    critic_node,
)
from app.agents.nodes import auditor_node
from app.agents.graph import create_compliance_graph
from app.services.parser import DocumentParserService
from app.services.report_generator import AuditReportGenerator
from app.services.vector_db import VectorStoreService
from pypdf import PdfReader
from reportlab.pdfgen import canvas


def test_fallback_compares_contract_values_and_cites_exact_clauses(monkeypatch):
    monkeypatch.setenv("MOCK_LLM", "1")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    contract = (
        "Invoices are payable on Net-45 terms. "
        "In no event shall either party's aggregate liability exceed the total fees paid "
        "or payable by Customer in the preceding twelve (12) months. "
        "Vendor shall notify Customer within seventy-two (72) hours of discovery of a breach. "
        "Any arbitration shall be seated in Delaware."
    )
    rules = [
        "Payment terms must be Net-30 or better.",
        "Aggregate liability must be at least 3x fees paid in the preceding 12 months.",
        "Breach notification must occur within 24 hours.",
        "Arbitration must be seated outside Delaware.",
        "The contract must allow Customer to choose forum.",
    ]

    findings = _evaluate_rules_heuristically(contract, rules)
    assert [finding["compliant"] for finding in findings] == [False] * 5
    for finding in findings:
        assert finding["citation"]
        assert finding["citation"] in contract
    assert "45 days" in findings[0]["reason"]
    assert "1x" in findings[1]["reason"]
    assert "72 hours" in findings[2]["reason"]
    assert "Delaware" in findings[3]["citation"]
    assert findings[4]["citation"] == "Any arbitration shall be seated in Delaware."

    reviewed = critic_node({
        "audit_report": {"findings": findings},
        "contract_text": contract,
        "source_contract_text": contract,
        "loop_count": 1,
    })
    review = citation_review_from_state(reviewed)

    assert reviewed["status"] == "pass"
    assert review["status"] == "local_check_pass"
    assert review["mode"] == "local_citation_check"
    assert review["citations_reviewed"] == 5
    assert review["citations_verified"] == 5
    assert all(finding["compliant"] is False for finding in reviewed["audit_report"]["findings"])


def test_fallback_does_not_invent_absent_citations(monkeypatch):
    monkeypatch.setenv("MOCK_LLM", "1")
    contract = "This office lease covers chairs, tables, monthly rent, and termination notice."
    finding = _evaluate_rules_heuristically(
        contract,
        ["The agreement must maintain an annual cyber liability insurance policy."],
    )[0]

    assert finding["compliant"] is None
    assert finding["citation"] == ""
    assert "manual review" in finding["reason"].lower()


def test_mixed_compliance_rules_with_numbers_paraphrase_and_or_clause():
    contract = (
        "SECTION 1. PAYMENT TERMS. Customer shall pay each invoice within thirty (30) days of receipt.\n"
        "SECTION 2. LIMITATION OF LIABILITY. Aggregate liability is capped at four (4) times the fees "
        "paid in the preceding twelve months.\n"
        "SECTION 3. SECURITY INCIDENTS. Vendor shall notify Customer within twelve (12) hours of discovery; "
        "the full written report is due within seventy-two (72) hours.\n"
        "SECTION 4. DISPUTE RESOLUTION. Any arbitration shall be seated in Kolkata.\n"
        "SECTION 6. DATA RETENTION. Vendor shall retain Customer records for a period of three (3) years."
    )
    rules = [
        "Payment terms must be Net-30 or better.",
        "Liability cap must be at least 3x the fees paid in the preceding 12 months.",
        "Security breach notification must be within 24 hours of confirmation.",
        "Arbitration must be seated outside Delaware OR Customer may choose the forum.",
        "Data retention must be at least 2 years.",
    ]

    findings = _evaluate_rules_heuristically(contract, rules)

    assert len(findings) == len(rules)
    assert [finding["compliant"] for finding in findings] == [True] * 5
    assert all(finding["citation"] in contract for finding in findings)
    assert "30 days <= 30 days" in findings[0]["reason"]
    assert "4x fees >= 3x" in findings[1]["reason"]
    assert "12 hours <= 24 hours" in findings[2]["reason"]
    assert "OR condition" in findings[3]["reason"]
    assert "3 >= 2 years" in findings[4]["reason"]
    assert "Kolkata" in findings[3]["citation"]
    assert "seventy-two" not in findings[2]["reason"]


def test_synthetic_mixed_compliance_pdf_has_five_compliant_cited_rules(monkeypatch):
    monkeypatch.setenv("MOCK_LLM", "1")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    rules = [
        "Payment terms must be Net-30 or better.",
        "Liability cap must be at least 3x the fees paid in the preceding 12 months.",
        "Security breach notification must be within 24 hours of confirmation.",
        "Arbitration must be seated outside Delaware OR Customer may choose the forum.",
        "Data retention must be at least 2 years.",
    ]
    page_texts = [
        [
            "SECTION 1. PAYMENT TERMS",
            "Customer shall pay each invoice within thirty (30) days of receipt.",
        ],
        [
            "SECTION 2. LIMITATION OF LIABILITY",
            "Aggregate liability is capped at four (4) times the fees paid in the preceding twelve months.",
        ],
        [
            "SECTION 3. SECURITY INCIDENTS",
            "Vendor shall notify Customer within twelve (12) hours of discovery; the full written report is due within seventy-two (72) hours.",
        ],
        [
            "SECTION 4. DISPUTE RESOLUTION",
            "Any arbitration shall be seated in Kolkata.",
        ],
        [
            "SECTION 6. DATA RETENTION",
            "Vendor shall retain Customer records for a period of three (3) years.",
        ],
    ]
    buffer = BytesIO()
    pdf = canvas.Canvas(buffer)
    for lines in page_texts:
        y = 750
        for line in lines:
            pdf.drawString(55, y, line)
            y -= 18
        pdf.showPage()
    pdf.save()

    pages = DocumentParserService.extract_pdf_pages(buffer.getvalue())
    contract = "\n\n".join(str(page["text"]) for page in pages)
    workflow = create_compliance_graph(auditor_node, critic_node)
    final_state = workflow.invoke({
        "contract_text": contract,
        "source_contract_text": contract,
        "contract_pages": pages,
        "retrieval_context": contract,
        "compliance_rules": rules,
        "loop_count": 0,
    })
    findings = final_state["audit_report"]["findings"]
    citation_review = citation_review_from_state(final_state)

    assert len(findings) == len(rules) == 5
    assert [finding["compliant"] for finding in findings] == [True] * 5
    assert all(finding["citation"] in contract for finding in findings)
    assert [finding["source_reference"].split(",")[0] for finding in findings] == [
        "Page 1", "Page 2", "Page 3", "Page 4", "Page 5",
    ]
    assert "12 hours <= 24 hours" in findings[2]["reason"]
    assert "seventy-two" not in findings[2]["reason"]
    assert "4x fees >= 3x" in findings[1]["reason"]
    assert citation_review["status"] == "local_check_pass"
    assert citation_review["citations_reviewed"] == 5
    assert citation_review["citations_verified"] == 5


def test_fallback_one_rule_with_or_conditions_produces_one_finding():
    findings = _evaluate_rules_heuristically(
        "SECTION 4. Any arbitration shall be seated in Kolkata.",
        [
            "Arbitration must be seated outside Delaware OR Customer may choose the forum.",
        ],
    )

    assert len(findings) == 1
    assert findings[0]["compliant"] is True
    assert "OR condition" in findings[0]["reason"]
    assert "Kolkata" in findings[0]["citation"]


def test_fallback_handles_compliant_contract_values():
    contract = (
        "Invoices are payable on Net-30 terms. "
        "Aggregate liability is capped at three times the fees paid in the prior 12 months. "
        "Vendor shall notify Customer within 24 hours of a security breach. "
        "Any arbitration shall be seated in New York. "
        "Customer may choose the forum for any court proceeding."
    )
    rules = [
        "Payment terms must be Net-30 or better.",
        "Aggregate liability must be at least 3x fees paid.",
        "Breach notification must occur within 24 hours.",
        "Arbitration must be seated outside Delaware.",
        "The contract must allow Customer to choose forum.",
    ]

    findings = _evaluate_rules_heuristically(contract, rules)

    assert [finding["compliant"] for finding in findings] == [True] * 5
    assert all(finding["citation"] in contract for finding in findings)


def test_fallback_evaluates_word_and_parenthetical_liability_multiple():
    finding = _evaluate_rules_heuristically(
        "SECTION 2. Aggregate liability is capped at four (4) times the fees paid.",
        ["Liability must be capped at ≥3x fees."],
    )[0]

    assert finding["compliant"] is True
    assert "4x fees >= 3x" in finding["reason"]


def test_amendment_clause_overrides_base_liability_clause():
    contract = (
        "SECTION 5. LIMITATION OF LIABILITY\n"
        "Aggregate liability is capped at one times the fees paid in the prior 12 months.\n"
        "AMENDMENT NO. 1\n"
        "Section 5 is amended and restated as follows. Aggregate liability is capped at three times "
        "the fees paid in the prior 12 months."
    )

    finding = _evaluate_rules_heuristically(
        contract,
        ["Aggregate liability must be at least 3x fees paid."],
    )[0]

    assert finding["compliant"] is True
    assert "three times" in finding["citation"]


def test_auditor_attaches_page_and_section_reference_to_citation(monkeypatch):
    monkeypatch.setenv("MOCK_LLM", "1")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    contract = "SECTION 2. PAYMENT TERMS\nInvoices are payable on Net-30 terms."

    result = auditor_node({
        "contract_text": contract,
        "source_contract_text": contract,
        "contract_pages": [{"page": 4, "text": contract}],
        "compliance_rules": ["Payment terms must be Net-30 or better."],
    })
    finding = result["audit_report"]["findings"][0]

    assert finding["citation"] == "Invoices are payable on Net-30 terms."
    assert finding["source_reference"] == "Page 4, SECTION 2. PAYMENT TERMS"
    assert finding["review_priority"] == "Medium"
    assert finding["next_step"]


def test_pdf_parser_preserves_page_numbers_for_citation_sources():
    sample_path = os.path.join(PROJECT_ROOT, "sample_enterprise_agreement.pdf")
    with open(sample_path, "rb") as sample_file:
        pages = DocumentParserService.extract_pdf_pages(sample_file.read())
    chunks = DocumentParserService.chunk_pages(pages)

    assert pages and pages[0]["page"] == 1
    assert "ENTERPRISE CLOUD SERVICES AGREEMENT" in pages[0]["text"]
    assert chunks and all(chunk["page"] == 1 for chunk in chunks)


def test_long_multipage_agreement_uses_later_amendment_clause(monkeypatch):
    monkeypatch.setenv("MOCK_LLM", "1")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    buffer = BytesIO()
    pdf = canvas.Canvas(buffer)
    contract_pages_text = [
        [
            "MASTER SERVICES AGREEMENT",
            "SECTION 1. DEFINITIONS",
            *[f"Defined term {index}: commercial services language." for index in range(35)],
        ],
        [
            "SECTION 9. LIMITATION OF LIABILITY",
            "Aggregate liability is capped at one times the fees paid in the prior 12 months.",
            *[f"SECTION {index}. STANDARD COMMERCIAL TERMS" for index in range(10, 35)],
        ],
        [
            "AMENDMENT NO. 1",
            "SECTION 9. LIMITATION OF LIABILITY (AS AMENDED)",
            "Section 9 is amended and restated in its entirety.",
            "Aggregate liability is capped at three times the fees paid in the prior 12 months.",
        ],
    ]
    for lines in contract_pages_text:
        y = 750
        for line in lines:
            pdf.drawString(55, y, line)
            y -= 18
        pdf.showPage()
    pdf.save()

    pages = DocumentParserService.extract_pdf_pages(buffer.getvalue())
    contract = "\n\n".join(str(page["text"]) for page in pages)
    result = auditor_node({
        "contract_text": contract,
        "source_contract_text": contract,
        "contract_pages": pages,
        "compliance_rules": ["Aggregate liability must be at least 3x fees paid."],
    })
    finding = result["audit_report"]["findings"][0]

    assert len(pages) == 3
    assert len(contract) > 2000
    assert finding["compliant"] is True
    assert "three times" in finding["citation"]
    assert finding["source_reference"].startswith("Page 3")


def test_critic_rejects_spliced_non_contiguous_citation(monkeypatch):
    monkeypatch.setenv("MOCK_LLM", "1")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    contract = (
        "Aggregate liability shall not exceed fees paid in the preceding twelve months. "
        "Vendor shall notify Customer within seventy-two hours."
    )
    report = {
        "findings": [{
            "rule": "Liability cap",
            "compliant": False,
            "citation": "Aggregate liability shall not exceed fees paid within seventy-two hours.",
            "reason": "The cap is too low.",
        }]
    }

    result = critic_node({
        "audit_report": report,
        "contract_text": contract,
        "source_contract_text": contract,
        "loop_count": 1,
    })

    assert result["status"] == "fail"
    assert result["critic_mode"] == "local_citation_check"
    assert "not an exact contract excerpt" in result["critic_feedback"]
    assert result["critic_warning"]
    finding = result["audit_report"]["findings"][0]
    assert finding["compliant"] is None
    assert "Manual review" in finding["next_step"]


def test_critic_rejects_exact_but_unrelated_forum_citation(monkeypatch):
    monkeypatch.setenv("MOCK_LLM", "1")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    party_clause = "This Agreement is between Customer and Meridian Data Services LLC."
    result = critic_node({
        "audit_report": {
            "findings": [{
                "rule": "The contract must allow Customer to choose forum.",
                "compliant": False,
                "citation": party_clause,
                "reason": "The forum is fixed.",
            }]
        },
        "contract_text": party_clause,
        "source_contract_text": party_clause,
        "loop_count": 1,
    })

    assert result["status"] == "fail"
    assert result["audit_report"]["findings"][0]["citation"] == ""
    assert result["audit_report"]["findings"][0]["compliant"] is None


def test_unexecuted_critic_cannot_be_reported_as_pass():
    review = citation_review_from_state({"status": "pass"})

    assert review["status"] == "not_run"
    assert review["mode"] == "not_run"


def test_vector_search_returns_deduplicated_parent_context(monkeypatch):
    child = "Invoices are payable on Net-45 terms."
    parent = f"{child} Late payments accrue interest after the due date."

    class SearchClient:
        def search(self, **kwargs):
            return [
                SimpleNamespace(payload={
                    "child_text": child,
                    "parent_text": parent,
                    "page_number": 2,
                }),
                SimpleNamespace(payload={
                    "child_text": "Late payments accrue interest after the due date.",
                    "parent_text": parent,
                    "page_number": 2,
                }),
            ]

    service = object.__new__(VectorStoreService)
    service.collection_name = "compliance_documents"
    service.client = SearchClient()
    monkeypatch.setattr(service, "_embed_with_retry", lambda texts, api_key: [[0.0]])

    context = service.query_relevant_context("payment terms")

    assert context.count(parent) == 1
    assert "[Page 2 | Retrieved contract context]" in context
    assert context.endswith(parent)


def test_report_summary_counts_partition_all_findings_and_warns_first():
    pdf = AuditReportGenerator.generate_pdf_report("sample.pdf", {
        "compliance_status": "undetermined",
        "analysis": {"mode": "gemini"},
        "citation_review": {
            "mode": "gemini",
            "status": "verified",
            "citations_reviewed": 2,
            "citations_verified": 2,
        },
        "findings": [
            {"rule": "Rule one", "compliant": True, "citation": "Clause one.", "source_reference": "Page 1", "reason": "Condition satisfied.", "next_step": "Confirm current version."},
            {"rule": "Rule two", "compliant": False, "citation": "Clause two.", "source_reference": "Page 2", "reason": "Condition not satisfied.", "next_step": "Renegotiate."},
            {"rule": "Rule three", "compliant": None, "citation": "", "next_step": "Manual review."},
        ],
    })
    text = "\n".join(page.extract_text() or "" for page in PdfReader(BytesIO(pdf)).pages)

    assert "Rules reviewed: 3" in text
    assert "Compliant: 1" in text
    assert "Non-compliant: 1" in text
    assert "Undetermined: 1" in text
    assert "Gemini (multi-agent LangGraph pipeline)" in text
    assert "2/2 cited excerpts verified" in text
    assert "Reasoning" in text
    assert "Next step" in text
    assert "Manual review." in text


def test_report_generator_refuses_fallback_findings():
    import pytest

    with pytest.raises(ValueError, match="cannot be generated from heuristic fallback"):
        AuditReportGenerator.generate_pdf_report("sample.pdf", {
            "analysis": {"mode": "heuristic_fallback"},
            "findings": [{"rule": "Rule", "compliant": False}],
        })


def test_blocked_audit_is_not_mislabeled_as_fallback():
    pdf = AuditReportGenerator.generate_pdf_report("sample.pdf", {
        "guardrail_status": "blocked",
        "compliance_status": "blocked",
        "analysis": {},
        "citation_review": {"mode": "not_run", "status": "not_run"},
        "findings": [],
    })
    text = "\n".join(page.extract_text() or "" for page in PdfReader(BytesIO(pdf)).pages)

    assert "local heuristic analysis" not in text
    assert "Citation review was not run." not in text


def test_gemini_fails_over_after_rate_limit(monkeypatch):
    import app.agents.nodes as nodes

    class RateLimitError(Exception):
        status_code = 429

    calls = []

    def fake_completion(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise RateLimitError()
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content='{"findings": []}'))]
        )

    monkeypatch.setattr(nodes, "completion", fake_completion)
    monkeypatch.setattr(nodes, "GEMINI_MODEL", "gemini/primary")
    monkeypatch.setenv("GEMINI_FALLBACK_MODELS", "gemini/fallback")

    _, parsed, model = nodes._gemini_json_completion("test prompt", "test-key")

    assert len(calls) == 2
    assert [call["model"] for call in calls] == ["gemini/primary", "gemini/fallback"]
    assert all(call["num_retries"] == 0 for call in calls)
    assert model == "gemini/fallback"
    assert parsed == {"findings": []}


def test_gemini_fails_over_after_service_unavailable(monkeypatch):
    import app.agents.nodes as nodes

    class ServiceUnavailableError(Exception):
        pass

    calls = []

    def fake_completion(**kwargs):
        calls.append(kwargs)
        if kwargs["model"] == "gemini/primary":
            raise ServiceUnavailableError()
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content='{"findings": []}'))]
        )

    monkeypatch.setattr(nodes, "completion", fake_completion)
    monkeypatch.setattr(nodes, "GEMINI_MODEL", "gemini/primary")
    monkeypatch.setenv("GEMINI_FALLBACK_MODELS", "gemini/fallback")

    _, parsed, model = nodes._gemini_json_completion("test prompt", "test-key")

    assert len(calls) == 2
    assert model == "gemini/fallback"
    assert parsed == {"findings": []}


def test_gemini_retries_connection_timeout_before_failure(monkeypatch):
    import app.agents.nodes as nodes
    import pytest

    calls = []

    def fake_completion(**kwargs):
        calls.append(kwargs)
        raise TimeoutError("provider timed out")

    monkeypatch.setattr(nodes, "completion", fake_completion)
    monkeypatch.setattr(nodes.time, "sleep", lambda delay: None)
    monkeypatch.setattr(nodes, "GEMINI_MODEL", "gemini/primary")
    monkeypatch.setenv("GEMINI_FALLBACK_MODELS", "")
    monkeypatch.setenv("GEMINI_MAX_RETRIES", "1")

    with pytest.raises(TimeoutError):
        nodes._gemini_json_completion("test prompt", "test-key")

    assert len(calls) == 2


def test_gemini_key_is_read_from_local_dotenv(monkeypatch, tmp_path):
    import app.agents.nodes as nodes

    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text("GEMINI_API_KEY=synthetic-local-key\n", encoding="utf-8")

    assert nodes.get_gemini_api_key() == "synthetic-local-key"


def test_langgraph_uses_gemini_for_auditor_and_critic(monkeypatch):
    import app.agents.nodes as nodes

    monkeypatch.setenv("GEMINI_API_KEY", "synthetic-test-key")
    monkeypatch.delenv("MOCK_LLM", raising=False)
    contract = "Invoices are payable on Net-30 terms."
    calls = []

    def fake_completion(**kwargs):
        prompt = kwargs["messages"][0]["content"]
        calls.append(prompt)
        if "Senior Legal Quality Controller" in prompt:
            content = '{"status": "pass", "feedback": "Citation verified."}'
        else:
            content = (
                '{"findings": [{"rule": "Payment terms must be Net-30 or better.", '
                '"compliant": true, "citation": "Invoices are payable on Net-30 terms.", '
                '"reason": "The contract meets the stated limit."}]}'
            )
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
        )

    monkeypatch.setattr(nodes, "completion", fake_completion)
    workflow = create_compliance_graph(auditor_node, critic_node)
    final_state = workflow.invoke({
        "contract_text": contract,
        "source_contract_text": contract,
        "contract_pages": [{"page": 1, "text": contract}],
        "compliance_rules": ["Payment terms must be Net-30 or better."],
        "loop_count": 0,
    })

    assert len(calls) == 2
    assert final_state["audit_report"]["analysis"]["mode"] == "gemini"
    assert final_state["critic_mode"] == "gemini"
    assert final_state["status"] == "pass"
    assert final_state["audit_report"]["findings"][0]["source_reference"] == "Page 1"
