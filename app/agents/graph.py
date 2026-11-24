from typing import TypedDict, Dict, Any, List, Optional
from langgraph.graph import StateGraph, END


class ComplianceState(TypedDict, total=False):
    contract_text: str
    source_contract_text: Optional[str]
    source_pages: Optional[List[Dict[str, Any]]]
    contract_pages: Optional[List[Dict[str, Any]]]
    retrieval_context: Optional[str]
    compliance_rules: List[str]
    contract_type: Optional[str]
    rule_alignment_warning: Optional[str]
    total_pages: Optional[int]
    audit_report: Dict[str, Any]
    critic_feedback: str
    citation_review_executed: bool
    citation_review_status: str
    citation_review_detail: str
    verified_citations: int
    total_citations: int
    citations_reviewed: int
    citations_verified: int
    model_used: str
    critic_mode: str
    critic_model: str
    ai_error: Optional[str]
    loop_count: int
    status: str
    guardrail_status: str
    guardrail_reason: str
    budget_limit: float
    estimated_cost: float
    budget_exceeded: bool
    total_tokens: int
    cost_breakdown: List[Dict[str, Any]]


def check_guardrails(state: ComplianceState) -> bool:
    """Returns False when the gatekeeper or budget guardrail has halted the run."""
    return (
        state.get("guardrail_status") != "blocked"
        and not state.get("budget_exceeded")
    )


def should_continue(state: ComplianceState) -> str:
    if state.get("guardrail_status") == "blocked":
        return "end"
    if state.get("budget_exceeded"):
        return "end"
    if state.get("status") in {"error", "incomplete"} or state.get("ai_error"):
        return "end"
    if state.get("loop_count", 0) >= 3:
        return "end"
    if state.get("status") == "fail":
        return "retry"
    return "end"


def create_compliance_graph(auditor_node, critic_node, gatekeeper_node=None):
    workflow = StateGraph(ComplianceState)
    workflow.add_node("auditor_agent", auditor_node)
    workflow.add_node("critic_agent", critic_node)

    if gatekeeper_node is not None:
        workflow.add_node("gatekeeper", gatekeeper_node)
        workflow.set_entry_point("gatekeeper")
        workflow.add_conditional_edges(
            "gatekeeper",
            lambda state: "blocked"
            if state.get("guardrail_status") == "blocked"
            else "auditor_agent",
            {"blocked": END, "auditor_agent": "auditor_agent"},
        )
    else:
        workflow.set_entry_point("auditor_agent")

    workflow.add_edge("auditor_agent", "critic_agent")
    workflow.add_conditional_edges(
        "critic_agent",
        should_continue,
        {
            "retry": "auditor_agent",
            "end": END,
        },
    )
    return workflow.compile()
