from typing import TypedDict, Dict, Any, List, Optional
from langgraph.graph import StateGraph, END


class ComplianceState(TypedDict):
    contract_text: str
    source_contract_text: Optional[str]
    source_pages: Optional[List[Dict[str, Any]]]
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
    model_used: str
    ai_error: Optional[str]
    loop_count: int
    status: str


def should_continue(state: ComplianceState) -> str:
    if state.get("status") in {"error", "incomplete"} or state.get("ai_error"):
        return "end"
    if state.get("loop_count", 0) >= 1:
        return "end"
    if state.get("status") == "fail":
        return "retry"
    return "end"


def create_compliance_graph(auditor_node, critic_node):
    workflow = StateGraph(ComplianceState)
    workflow.add_node("auditor_agent", auditor_node)
    workflow.add_node("critic_agent", critic_node)
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
