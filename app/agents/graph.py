from typing import TypedDict, Dict, Any, List, Optional
from langgraph.graph import StateGraph, END


class ComplianceState(TypedDict, total=False):
    contract_text: str
    source_contract_text: str
    retrieval_context: str
    compliance_rules: List[str]
    audit_report: Dict[str, Any]
    critic_feedback: str
    loop_count: int
    status: str
    # Enterprise Upgrades
    guardrail_status: str       # "passed" | "blocked"
    guardrail_reason: str       # Security explanation if triggered
    total_tokens: int           # Accumulated tokens across agent cycles
    estimated_cost: float       # Accumulated cost in USD
    budget_limit: float         # Single-run budget limit in USD (default: 0.05)
    budget_exceeded: bool       # Budget limit flag
    cost_breakdown: List[Dict[str, Any]]  # Step-by-step token and cost log


def check_guardrails(state: ComplianceState) -> str:
    """Conditional router that halts graph if Gatekeeper flags the input."""
    if state.get("guardrail_status") == "blocked":
        return "blocked"
    return "proceed"


def should_continue(state: ComplianceState) -> str:
    """Enterprise router enforcing budget constraints, loop limits, and critic quality."""
    # 1. Budget constraint check
    if state.get("budget_exceeded", False):
        return "end"
    # 2. Maximum loops guardrail
    if state.get("loop_count", 0) >= 3:
        return "end"
    # 3. Quality control verdict
    if state.get("status") == "fail":
        return "retry"
    return "end"


def create_compliance_graph(auditor_node, critic_node, gatekeeper_node=None):
    from app.agents.nodes import gatekeeper_node as default_gatekeeper
    gk = gatekeeper_node or default_gatekeeper

    workflow = StateGraph(ComplianceState)
    workflow.add_node("gatekeeper_agent", gk)
    workflow.add_node("auditor_agent", auditor_node)
    workflow.add_node("critic_agent", critic_node)

    workflow.set_entry_point("gatekeeper_agent")

    workflow.add_conditional_edges(
        "gatekeeper_agent",
        check_guardrails,
        {
            "blocked": END,
            "proceed": "auditor_agent"
        }
    )
    workflow.add_edge("auditor_agent", "critic_agent")
    workflow.add_conditional_edges(
        "critic_agent",
        should_continue,
        {
            "retry": "auditor_agent",
            "end": END
        }
    )
    return workflow.compile()
