import os
import json
import time
from typing import Dict, Any
from google import genai
from google.genai import types
from app.agents.graph import ComplianceState

GEMINI_MODELS = [
    "gemini-2.5-flash",
    "gemini-3.5-flash-lite",
]


def _get_client() -> genai.Client:
    api_key = (os.getenv("VERTEXAI_API_KEY") or os.getenv("GEMINI_API_KEY") or "").strip()
    if not api_key:
        raise RuntimeError("Missing API key. Set VERTEXAI_API_KEY or GEMINI_API_KEY.")
    if api_key.startswith("AQ."):
        return genai.Client(vertexai=True, api_key=api_key)
    return genai.Client(api_key=api_key)


def _call_gemini_with_retry(prompt: str, max_retries: int = 3) -> tuple[dict, str]:
    client = _get_client()
    last_error = None
    for model_name in GEMINI_MODELS:
        for attempt in range(max_retries):
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        temperature=0.1,
                    ),
                )
                if not response.text:
                    raise ValueError("Gemini returned an empty response.")
                return json.loads(response.text), model_name
            except Exception as e:
                last_error = f"[{model_name} attempt {attempt + 1}]: {type(e).__name__} - {str(e)}"
                time.sleep(2 ** attempt)
    raise RuntimeError(f"All Gemini models failed after retries. Last error: {last_error}")


def auditor_node(state: ComplianceState) -> Dict[str, Any]:
    current_count = state.get("loop_count", 0)
    feedback_context = (
        f"\nPrevious Critic Feedback to fix: {state.get('critic_feedback')}"
        if current_count > 0
        else ""
    )

    prompt = f"""
You are a Senior Enterprise Contract Compliance Auditor.
Evaluate the Contract Text against EVERY rule in the Compliance Rules list.

CRITICAL INSTRUCTIONS:
1. ONE VERDICT PER RULE: Produce exactly one finding object per input rule. NEVER split a compound rule (containing AND / OR) into two findings. For OR rules, if either branch is satisfied, the rule is Compliant.
2. EXPLICIT MATH & LOGIC COMPARISON: For numeric rules (e.g., Net-30, >=3x fees, <=24h notice, >=2 years retention), extract the exact number from the contract and explicitly state the mathematical comparison in 'reasoning'.
3. VERBATIM CITATIONS ONLY: Copy an exact, continuous verbatim quote from the Contract Text (including Section/Page reference if visible). NEVER echo the rule text back as a citation.
4. UNDETERMINED POLICY: Only mark "Undetermined" if the contract genuinely contains zero language addressing the topic. Set 'next_step' to "Escalate to Legal Counsel — treat as Non-Compliant until clarified" when Undetermined or Non-Compliant, and "No action required" when Compliant.

Compliance Rules (Evaluate each of these {len(state['compliance_rules'])} rules):
{json.dumps(state['compliance_rules'], indent=2)}

Contract Text:
{state['contract_text']}
{feedback_context}

Respond STRICTLY with valid JSON matching this schema:
{{
  "findings": [
    {{
      "rule": "exact rule text from input",
      "result": "Compliant" | "Non-Compliant" | "Undetermined",
      "priority": "High" | "Medium" | "Low",
      "citation": "exact verbatim snippet from contract + section/page",
      "reasoning": "explicit number-vs-number or condition-vs-text comparison",
      "next_step": "concrete action item"
    }}
  ]
}}
"""
    try:
        parsed_report, used_model = _call_gemini_with_retry(prompt)
        return {
            "audit_report": parsed_report,
            "model_used": used_model,
            "ai_error": None,
            "loop_count": current_count + 1,
        }
    except Exception as e:
        return {
            "audit_report": {"findings": []},
            "model_used": "NONE",
            "ai_error": str(e),
            "status": "error",
            "loop_count": current_count + 1,
        }


def critic_node(state: ComplianceState) -> Dict[str, Any]:
    if state.get("ai_error"):
        return {
            "status": "error",
            "critic_feedback": "Skipped due to upstream Gemini failure.",
            "citation_review_status": "NOT RUN — AI analysis failed",
            "verified_citations": 0,
        }

    report = state.get("audit_report", {})
    findings = report.get("findings", [])
    contract = state.get("contract_text", "")

    prompt = f"""
You are a Legal Quality Controller. Verify the Audit Report against the Raw Contract Text.
Check:
1. Are all {len(state['compliance_rules'])} rules accounted for (1 finding per rule, no split OR rules)?
2. Does every 'citation' actually appear in the Raw Contract Text without hallucination or rule-echoing?
3. Are all numeric comparisons (<=, >=) mathematically accurate?

Audit Report: {json.dumps(report)}
Raw Contract: {contract}

Respond strictly with JSON:
{{
  "status": "pass" | "fail",
  "verified_count": <integer count of confirmed citations>,
  "feedback": "Specific corrections needed if fail, or confirmation summary if pass"
}}
"""
    try:
        validation, critic_model = _call_gemini_with_retry(prompt)
        total = len(findings)
        verified = validation.get("verified_count", 0)
        if not isinstance(verified, int) or isinstance(verified, bool):
            verified = 0
        verified = min(max(verified, 0), total)
        return {
            "status": validation.get("status", "pass"),
            "critic_feedback": validation.get("feedback", ""),
            "citation_review_status": f"Verified via {critic_model} against source PDF — {verified}/{total} citations confirmed",
            "verified_citations": verified,
        }
    except Exception as e:
        normalized_contract = " ".join(contract.lower().split())
        confirmed = 0
        for item in findings:
            cit = " ".join(item.get("citation", "").lower().split()).strip('".')
            core_cit = cit.split("(sec")[0].strip().strip('"')
            if core_cit and len(core_cit) > 10 and core_cit in normalized_contract:
                confirmed += 1
        return {
            "status": "pass" if confirmed == len(findings) else "fail",
            "critic_feedback": f"Critic LLM unavailable ({str(e)}); ran deterministic substring check.",
            "citation_review_status": f"Local substring check only (Critic LLM unavailable) — {confirmed}/{len(findings)} exact quotes matched",
            "verified_citations": confirmed,
        }
