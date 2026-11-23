from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import app.agents.nodes as nodes


def _finding(rule):
    return {
        "rule": rule,
        "result": "Compliant",
        "priority": "Medium",
        "citation": "Any arbitration shall be seated in Kolkata.",
        "reasoning": "The contract satisfies the rule.",
        "next_step": "No action required",
    }


def test_auditor_corrects_split_or_rule_into_one_finding(monkeypatch):
    rules = [
        "Payment terms must be Net-30 or better.",
        "Arbitration must be seated outside Delaware OR Customer may choose the forum.",
        "Vendor must provide notice within 24 hours.",
    ]
    split_findings = [
        _finding(rules[0]),
        _finding("Arbitration must be seated outside Delaware."),
        _finding("Customer may choose the forum."),
        _finding(rules[2]),
    ]
    corrected_findings = [_finding(rule) for rule in rules]
    prompts = []

    def fake_call(prompt, max_retries=3):
        prompts.append(prompt)
        if len(prompts) == 1:
            return {"findings": split_findings}, "groq/model-a"
        return {"findings": corrected_findings}, "groq/model-b"

    monkeypatch.setattr(nodes, "_call_groq_with_retry", fake_call)

    result = nodes.auditor_node(
        {
            "contract_text": "The contract text.",
            "compliance_rules": rules,
            "total_pages": 3,
            "loop_count": 0,
        }
    )

    findings = result["audit_report"]["findings"]
    assert len(prompts) == 2
    assert len(findings) == len(rules)
    assert [finding["rule"] for finding in findings] == rules
    assert "Gap — Missing Protection" in prompts[0]
    assert "Pages 1-3" in prompts[0]
    assert "Never split one input rule" in prompts[1]


def test_auditor_withholds_findings_if_correction_still_splits_or_rule(monkeypatch):
    rules = [
        "Arbitration must be seated outside Delaware OR Customer may choose the forum.",
    ]
    split_findings = [
        _finding("Arbitration must be seated outside Delaware."),
        _finding("Customer may choose the forum."),
    ]
    monkeypatch.setattr(
        nodes,
        "_call_groq_with_retry",
        lambda prompt, max_retries=3: ({"findings": split_findings}, "groq/model"),
    )

    result = nodes.auditor_node(
        {
            "contract_text": "The contract text.",
            "compliance_rules": rules,
            "loop_count": 0,
        }
    )

    assert result["status"] == "error"
    assert result["audit_report"]["findings"] == []
    assert "Finding count/rule mapping mismatch" in result["ai_error"]
