import argparse
import os
import sys
import time

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.agents.graph import create_compliance_graph
from app.agents.nodes import auditor_node, critic_node, get_gemini_api_key


SAMPLE_CONTRACT = (
    "SAMPLE MIXED-COMPLIANCE SERVICES AGREEMENT\n"
    "SECTION 1. PAYMENT TERMS\n"
    "Customer shall pay each invoice within thirty (30) days of receipt.\n"
    "SECTION 2. LIMITATION OF LIABILITY\n"
    "Aggregate liability is capped at four (4) times the fees paid in the preceding twelve months.\n"
    "SECTION 3. SECURITY INCIDENTS\n"
    "Vendor shall notify Customer within twelve (12) hours of discovery; "
    "the full written report is due within seventy-two (72) hours.\n"
    "SECTION 4. DISPUTE RESOLUTION\n"
    "Any arbitration shall be seated in Kolkata.\n"
    "SECTION 6. DATA RETENTION\n"
    "Vendor shall retain Customer records for a period of three (3) years."
)
SAMPLE_RULES = [
    "Payment terms must be Net-30 or better.",
    "Liability cap must be at least 3x the fees paid in the preceding 12 months.",
    "Security breach notification must be within 24 hours of confirmation.",
    "Arbitration must be seated outside Delaware OR Customer may choose the forum.",
    "Data retention must be at least 2 years.",
]


def run_smoke_test(workflow) -> tuple[bool, float, str]:
    state = {
        "contract_text": SAMPLE_CONTRACT,
        "source_contract_text": SAMPLE_CONTRACT,
        "contract_pages": [{"page": 1, "text": SAMPLE_CONTRACT}],
        "retrieval_context": SAMPLE_CONTRACT,
        "compliance_rules": SAMPLE_RULES,
        "audit_report": {},
        "critic_feedback": "",
        "loop_count": 0,
        "status": "pending",
    }
    started = time.perf_counter()
    try:
        final_state = workflow.invoke(state)
    except Exception as exc:
        return False, time.perf_counter() - started, type(exc).__name__
    elapsed = time.perf_counter() - started

    report = final_state.get("audit_report", {})
    findings = report.get("findings") or []
    analysis_mode = report.get("analysis", {}).get("mode")
    critic_mode = final_state.get("critic_mode")
    if analysis_mode != "gemini":
        warning = report.get("analysis", {}).get("warning")
        return False, elapsed, warning or "auditor used local fallback"
    if critic_mode != "gemini":
        return False, elapsed, "critic used local citation check"
    if len(findings) != len(SAMPLE_RULES):
        return False, elapsed, f"expected {len(SAMPLE_RULES)} rule findings, received {len(findings)}"
    if any(finding.get("compliant") is not True for finding in findings):
        return False, elapsed, "not all synthetic contract rules were marked compliant"
    if any(
        not finding.get("citation")
        or finding["citation"] not in SAMPLE_CONTRACT
        or not finding.get("source_reference")
        for finding in findings
    ):
        return False, elapsed, "a citation was missing or did not match the synthetic contract"
    if not all(
        token in findings[index].get("reason", "")
        for index, token in ((0, "30 days"), (1, "4x fees"), (2, "12 hours"), (4, "3"))
    ):
        return False, elapsed, "a numeric finding lacked an explicit comparison"
    if final_state.get("status") != "pass":
        return False, elapsed, "citation review did not pass"
    if final_state.get("citations_reviewed") != 5 or final_state.get("citations_verified") != 5:
        return False, elapsed, "not all five citations were independently verified"
    auditor_model = report.get("analysis", {}).get("model") or "unknown model"
    critic_model = final_state.get("critic_model") or "unknown model"
    return (
        True,
        elapsed,
        f"auditor={auditor_model}; critic={critic_model}; 5/5 citations verified",
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run 5-10 live Gemini auditor/critic checks on a synthetic contract."
    )
    parser.add_argument("--runs", type=int, default=5, help="Number of runs (5 through 10; default: 5)")
    args = parser.parse_args()
    if not 5 <= args.runs <= 10:
        parser.error("--runs must be between 5 and 10")
    if not get_gemini_api_key():
        print("GEMINI_API_KEY or GOOGLE_API_KEY is not configured in this environment.")
        return 2
    if os.getenv("MOCK_LLM"):
        print("MOCK_LLM is enabled; unset it before running a live smoke test.")
        return 2

    workflow = create_compliance_graph(auditor_node, critic_node)
    durations = []
    passed = 0
    for run_number in range(1, args.runs + 1):
        success, duration, result = run_smoke_test(workflow)
        durations.append(duration)
        passed += int(success)
        print(f"Run {run_number}/{args.runs}: {'PASS' if success else 'FAIL'} ({result}; {duration:.2f}s)")
        if run_number < args.runs:
            time.sleep(2)

    mean_duration = sum(durations) / len(durations)
    print(f"Summary: {passed}/{args.runs} clean Gemini runs; mean duration {mean_duration:.2f}s.")
    return 0 if passed == args.runs else 1


if __name__ == "__main__":
    raise SystemExit(main())
