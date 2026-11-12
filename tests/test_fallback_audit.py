import os
import sys
from types import SimpleNamespace

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.agents.nodes import (
    _evaluate_rules_heuristically,
    citation_review_from_state,
    critic_node,
)
from app.services.vector_db import VectorStoreService


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
    assert review["status"] == "pass"
    assert all(finding["compliant"] is False for finding in reviewed["audit_report"]["findings"])


def test_fallback_does_not_invent_absent_citations(monkeypatch):
    monkeypatch.setenv("MOCK_LLM", "1")
    contract = "This office lease covers chairs, tables, monthly rent, and termination notice."
    finding = _evaluate_rules_heuristically(
        contract,
        ["The agreement must maintain an annual cyber liability insurance policy."],
    )[0]

    assert finding["compliant"] is False
    assert finding["citation"] == ""
    assert "No matching operative clause" in finding["reason"]


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
