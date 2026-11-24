from io import BytesIO
from pathlib import Path
import sys

from pypdf import PdfReader
from reportlab.pdfgen import canvas

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import app.agents.nodes as nodes
from app.main import parse_rules
from app.services.parser import DocumentParserService
from app.services.report_generator import AuditReportGenerator


def _pdf_with_two_pages():
    buffer = BytesIO()
    pdf = canvas.Canvas(buffer)
    pdf.drawString(50, 750, "First page clause.")
    pdf.showPage()
    pdf.drawString(50, 750, "Second page clause.")
    pdf.save()
    return buffer.getvalue()


def test_pdf_page_extraction_preserves_real_page_numbers():
    pages = DocumentParserService.extract_pdf_pages(_pdf_with_two_pages())

    assert len(pages) == 2
    assert [page["page"] for page in pages] == [1, 2]
    assert "First page clause." in DocumentParserService.format_pdf_pages(pages)
    assert "--- PAGE 2 ---" in DocumentParserService.format_pdf_pages(pages)


def test_api_rule_parser_keeps_commas_inside_compound_rule():
    rules = parse_rules(
        "Governing law requires arbitration outside Delaware, or customer choice.\n"
        "Data must be retained for at least 2 years."
    )

    assert rules == [
        "Governing law requires arbitration outside Delaware, or customer choice.",
        "Data must be retained for at least 2 years.",
    ]


def test_critic_rejects_citation_page_beyond_pdf_page_count(monkeypatch):
    def unexpected_model_call(_prompt, max_retries=3):
        raise AssertionError("Invalid page references should fail before the LLM call.")

    monkeypatch.setattr(nodes, "_call_groq_with_retry", unexpected_model_call)
    state = {
        "audit_report": {
            "findings": [
                {
                    "rule": "Payment must be due within 30 days.",
                    "result": "Compliant",
                    "citation": "Invoices are due within 30 days. (Page 8)",
                }
            ]
        },
        "compliance_rules": ["Payment must be due within 30 days."],
        "contract_text": "--- PAGE 1 ---\nInvoices are due within 30 days.",
        "source_contract_text": "--- PAGE 1 ---\nInvoices are due within 30 days.",
        "total_pages": 5,
    }

    result = nodes.critic_node(state)

    assert result["status"] == "fail"
    assert result["citation_review_status"] == "FAIL"
    assert "Page 8" in result["citation_review_detail"]
    assert result["audit_report"]["findings"] == []


def test_auditor_replaces_model_quote_and_page_with_exact_source_location(monkeypatch):
    source_pages = [
        {"page": 1, "text": "Invoices are payable on delivery."},
        {"page": 2, "text": "Invoices are due\nwithin 30 days."},
    ]
    monkeypatch.setattr(
        nodes,
        "_call_groq_with_retry",
        lambda _prompt, max_retries=3: (
            {
                "findings": [
                    {
                        "rule": "Payment must be due within 30 days.",
                        "result": "Compliant",
                        "citation": "Invoices are due within 30 days. (Page 8)",
                    }
                ]
            },
            "groq/test-model",
        ),
    )

    result = nodes.auditor_node(
        {
            "contract_text": "--- PAGE 1 ---\nInvoices are payable on delivery.\n"
            "--- PAGE 2 ---\nInvoices are due\nwithin 30 days.",
            "source_pages": source_pages,
            "total_pages": 2,
            "compliance_rules": ["Payment must be due within 30 days."],
            "loop_count": 0,
        }
    )

    finding = result["audit_report"]["findings"][0]
    assert finding["citation"] == "Invoices are due\nwithin 30 days."
    assert finding["citation_page"] == 2
    assert source_pages[1]["text"][
        finding["citation_start"] : finding["citation_end"]
    ] == finding["citation"]
    assert finding["source_reference"] == "Page 2"


def test_auditor_grounds_multiple_quotes_and_builds_gap_citation(monkeypatch):
    source_pages = [
        {"page": 2, "text": "The base liability cap is one times fees."},
        {"page": 5, "text": "Amendment A1 increases the cap to two times fees."},
    ]
    prompts = []

    def fake_completion(prompt, max_retries=3):
        prompts.append(prompt)
        if len(prompts) == 1:
            return (
                {
                    "findings": [
                        {
                            "rule": "The liability cap must be at least three times fees.",
                            "result": "Non-Compliant",
                            "citation_sentence_ids": [
                                "P2-S1",
                                "P5-S1",
                            ],
                        }
                    ]
                },
                "groq/test-model",
            )
        return (
            {
                "findings": [
                    {
                        "rule": "Data must be retained for at least 2 years.",
                        "result": "Gap — Missing Protection",
                        "citation_sentence_ids": [],
                    }
                ]
            },
            "groq/test-model",
        )

    monkeypatch.setattr(nodes, "_call_groq_with_retry", fake_completion)
    state = {
        "contract_text": "[Page 2]: The base liability cap is one times fees.\n\n"
        "---\n\n[Page 5]: Amendment A1 increases the cap to two times fees.",
        "source_pages": source_pages,
        "total_pages": 5,
        "compliance_rules": [
            "The liability cap must be at least three times fees.",
        ],
        "loop_count": 0,
    }

    liability = nodes.auditor_node(state)["audit_report"]["findings"][0]

    assert [item["page"] for item in liability["citations"]] == [2, 5]
    assert [item["quote"] for item in liability["citations"]] == [
        "The base liability cap is one times fees.",
        "Amendment A1 increases the cap to two times fees.",
    ]
    assert liability["source_reference"] == "Page 2, Page 5"
    assert nodes._citation_issues(
        [liability],
        "--- PAGE 2 ---\nThe base liability cap is one times fees.\n"
        "--- PAGE 5 ---\nAmendment A1 increases the cap to two times fees.",
        5,
        source_pages,
    ) == []
    assert "[P2-S1] The base liability cap is one times fees." in prompts[0]
    assert "[P5-S1] Amendment A1 increases the cap to two times fees." in prompts[0]
    assert "citation_sentence_ids containing the ID(s)" in prompts[0]

    gap = nodes.auditor_node(
        {
            **state,
            "compliance_rules": ["Data must be retained for at least 2 years."],
        }
    )["audit_report"]["findings"][0]
    assert gap["citation"] == (
        "No relevant language found in Contract Text (Pages 1-5)."
    )
    assert gap["citations"] == []


def test_auditor_withholds_citation_over_sixty_words(monkeypatch):
    source_quote = " ".join(f"term{index}" for index in range(61))
    monkeypatch.setattr(
        nodes,
        "_call_groq_with_retry",
        lambda _prompt, max_retries=3: (
            {
                "findings": [
                    {
                        "rule": "Payment terms",
                        "result": "Compliant",
                        "citation_quotes": [source_quote],
                    }
                ]
            },
            "groq/test-model",
        ),
    )
    result = nodes.auditor_node(
        {
            "contract_text": f"--- PAGE 1 ---\n{source_quote}",
            "source_pages": [{"page": 1, "text": source_quote}],
            "total_pages": 1,
            "compliance_rules": ["Payment terms"],
            "loop_count": 0,
        }
    )

    finding = result["audit_report"]["findings"][0]
    assert finding["citation_grounding_failed"] is True
    assert nodes._citation_issues(
        [finding], f"--- PAGE 1 ---\n{source_quote}", 1, [{"page": 1, "text": source_quote}]
    )


def test_unknown_sentence_id_is_rejected_without_using_model_quote(monkeypatch):
    monkeypatch.setattr(
        nodes,
        "_call_groq_with_retry",
        lambda _prompt, max_retries=3: (
            {
                "findings": [
                    {
                        "rule": "Payment terms",
                        "result": "Compliant",
                        "citation_sentence_ids": ["P8-S1"],
                    }
                ]
            },
            "groq/test-model",
        ),
    )
    source_pages = [{"page": 1, "text": "Invoices are due within 30 days."}]
    finding = nodes.auditor_node(
        {
            "contract_text": "--- PAGE 1 ---\nInvoices are due within 30 days.",
            "source_pages": source_pages,
            "total_pages": 1,
            "compliance_rules": ["Payment terms"],
            "loop_count": 0,
        }
    )["audit_report"]["findings"][0]

    assert finding["citation_grounding_failed"] is True
    assert finding.get("citation") is None
    assert nodes._citation_issues(
        [finding],
        "--- PAGE 1 ---\nInvoices are due within 30 days.",
        1,
        source_pages,
    )


def test_critic_requires_literal_whitespace_match_after_grounding():
    issues = nodes._citation_issues(
        [
            {
                "citation": "Invoices are due within 30 days.",
                "citation_page": 2,
            }
        ],
        "--- PAGE 2 ---\nInvoices are due\nwithin 30 days.",
        2,
        [{"page": 2, "text": "Invoices are due\nwithin 30 days."}],
    )

    assert any("does not appear verbatim" in issue for issue in issues)


def test_report_displays_source_page_with_exact_citation(tmp_path):
    report_path = tmp_path / "audit.pdf"
    AuditReportGenerator.generate_pdf_report(
        str(report_path),
        "contract.pdf",
        {
            "status": "pass",
            "citation_review_executed": True,
            "citation_review_status": "PASS",
            "citation_review_detail": "1/1 citations verified.",
            "verified_citations": 1,
            "total_citations": 1,
            "model_used": "test/model",
            "audit_report": {
                "findings": [
                    {
                        "rule": "Payment terms",
                        "result": "Compliant",
                        "citation": "Invoices are due within 30 days.",
                        "source_reference": "Page 2",
                        "citations": [
                            {
                                "quote": "Invoices are due within 30 days.",
                                "page": 2,
                                "start": 0,
                                "end": 32,
                            },
                            {
                                "quote": "Payment is due by the end of the month.",
                                "page": 3,
                                "start": 0,
                                "end": 40,
                            },
                        ],
                    }
                ]
            },
        },
    )
    pdf_text = "\n".join(
        page.extract_text() or "" for page in PdfReader(str(report_path)).pages
    )

    assert "Invoices are due within 30 days." in " ".join(pdf_text.split())
    assert "Page 2" in pdf_text
    assert "Payment is due by the end of the month." in " ".join(pdf_text.split())
    assert "Page 3" in pdf_text


def test_critic_rejects_quote_assigned_to_the_wrong_source_page(monkeypatch):
    def unexpected_model_call(_prompt, max_retries=3):
        raise AssertionError("Wrong-page citations should fail before the LLM call.")

    monkeypatch.setattr(nodes, "_call_groq_with_retry", unexpected_model_call)
    for citation, metadata in (
        ("Invoices are due within 30 days. (Page 2)", {}),
        ("Invoices are due within 30 days.", {"citation_page": 2}),
    ):
        state = {
            "audit_report": {
                "findings": [
                    {
                        "rule": "Payment must be due within 30 days.",
                        "result": "Compliant",
                        "citation": citation,
                        **metadata,
                    }
                ]
            },
            "compliance_rules": ["Payment must be due within 30 days."],
            "contract_text": "--- PAGE 1 ---\nInvoices are due within 30 days.\n"
            "--- PAGE 2 ---\nOther clause.",
            "source_contract_text": "--- PAGE 1 ---\nInvoices are due within 30 days.\n"
            "--- PAGE 2 ---\nOther clause.",
            "source_pages": [
                {"page": 1, "text": "Invoices are due within 30 days."},
                {"page": 2, "text": "Other clause."},
            ],
            "total_pages": 2,
        }

        result = nodes.critic_node(state)

        assert result["status"] == "fail"
        assert result["citation_review_status"] == "FAIL"
        assert "does not appear verbatim on cited Page 2" in result["citation_review_detail"]


def test_critic_rejects_page_reference_without_a_verbatim_quote(monkeypatch):
    def unexpected_model_call(_prompt, max_retries=3):
        raise AssertionError("A page label alone is not a citation.")

    monkeypatch.setattr(nodes, "_call_groq_with_retry", unexpected_model_call)
    state = {
        "audit_report": {
            "findings": [
                {
                    "rule": "Payment must be due within 30 days.",
                    "result": "Compliant",
                    "citation": "Page 1",
                }
            ]
        },
        "compliance_rules": ["Payment must be due within 30 days."],
        "contract_text": "--- PAGE 1 ---\nInvoices are due within 30 days.",
        "source_contract_text": "--- PAGE 1 ---\nInvoices are due within 30 days.",
        "total_pages": 1,
    }

    result = nodes.critic_node(state)

    assert result["status"] == "fail"
    assert "no verbatim quote" in result["citation_review_detail"]


def test_critic_does_not_require_a_quote_for_gap_findings(monkeypatch):
    monkeypatch.setattr(
        nodes,
        "_call_groq_with_retry",
        lambda _prompt, max_retries=3: (
            {"status": "pass", "feedback": "Rule and numeric checks passed."},
            "groq/test-model",
        ),
    )
    state = {
        "audit_report": {
            "findings": [
                {
                    "rule": "Data must be retained for at least 2 years.",
                    "result": "Gap — Missing Protection",
                    "citation": "No relevant language found in Contract Text (Pages 1-2).",
                }
            ]
        },
        "compliance_rules": ["Data must be retained for at least 2 years."],
        "contract_text": "--- PAGE 1 ---\nPayment is due within 30 days.",
        "source_contract_text": "--- PAGE 1 ---\nPayment is due within 30 days.",
        "total_pages": 2,
    }

    result = nodes.critic_node(state)

    assert result["status"] == "pass"
    assert result["citation_review_status"] == "PASS"
    assert result["verified_citations"] == 0
    assert result["total_citations"] == 0
