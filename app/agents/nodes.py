import json
import os
import re
from typing import Any, Dict, List
from litellm import completion
from app.agents.graph import ComplianceState

GEMINI_MODEL = "gemini/gemini-3.8-flash"
INPUT_TOKEN_RATE = 0.075 / 1_000_000
OUTPUT_TOKEN_RATE = 0.30 / 1_000_000
DEFAULT_BUDGET_LIMIT = 0.05

PROMPT_INJECTION_PATTERNS = [
    r"(?i)ignore\s+(all\s+)?(previous|prior)\s+instructions",
    r"(?i)disregard\s+(all\s+)?(previous|prior)\s+instructions",
    r"(?i)override\s+(all\s+)?(previous|prior)\s+rules",
    r"(?i)reveal\s+(your\s+)?(system|internal)\s+(prompt|instructions)",
    r"(?i)system\s+prompt\s+leak",
    r"(?i)print\s+(your\s+)?(system\s+prompt|initial\s+instructions)",
    r"(?i)what\s+are\s+your\s+(system|core)\s+instructions",
    r"(?i)bypass\s+safety\s+(filters|rules|guardrails)",
    r"(?i)you\s+are\s+now\s+(DAN|unfiltered|jailbroken)",
    r"(?i)developer\s+mode\s+enabled",
    r"(?i)drop\s+table\s+",
    r"(?i)<\s*script[^>]*>",
    r"---BEGIN\s+SYSTEM",
    r"\[SYSTEM\s+INSTRUCTION\]",
]

def get_gemini_api_key() -> str:
    return (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or "").strip().strip('"').strip("'")

def calculate_step_cost(prompt_tokens: int, completion_tokens: int) -> float:
    return (prompt_tokens * INPUT_TOKEN_RATE) + (completion_tokens * OUTPUT_TOKEN_RATE)

def gatekeeper_node(state: ComplianceState) -> Dict[str, Any]:
    contract = state.get("contract_text", "")
    rules = state.get("compliance_rules", [])
    combined_input = f"{contract} " + " ".join(rules)
    budget_limit = state.get("budget_limit", DEFAULT_BUDGET_LIMIT)

    for pattern in PROMPT_INJECTION_PATTERNS:
        match = re.search(pattern, combined_input)
        if match:
            trigger = match.group(0)
            reason = f"Malicious instruction / prompt injection detected matching pattern: '{trigger}'"
            blocked_report = {
                "status": "blocked",
                "guardrail_status": "blocked",
                "guardrail_reason": reason,
                "findings": [],
                "summary": f"Security Alert: Execution blocked by Enterprise Gatekeeper. {reason}",
                "cost_summary": {
                    "total_tokens": 0,
                    "estimated_cost_usd": 0.0,
                    "budget_limit_usd": budget_limit,
                    "budget_exceeded": False,
                    "cost_breakdown": [{"step": "gatekeeper", "tokens": 0, "cost_usd": 0.0}]
                }
            }
            return {
                "guardrail_status": "blocked",
                "guardrail_reason": reason,
                "status": "blocked",
                "audit_report": blocked_report,
                "total_tokens": 0,
                "estimated_cost": 0.0,
                "budget_limit": budget_limit,
                "budget_exceeded": False,
                "cost_breakdown": [{"step": "gatekeeper", "tokens": 0, "cost_usd": 0.0}]
            }

    return {
        "guardrail_status": "passed",
        "guardrail_reason": "Clean input: Passed Enterprise Gatekeeper injection and safety guardrails.",
        "budget_limit": budget_limit,
        "total_tokens": state.get("total_tokens", 0),
        "estimated_cost": state.get("estimated_cost", 0.0),
        "cost_breakdown": state.get("cost_breakdown", [])
    }

def _evaluate_rules_heuristically(contract: str, rules: List[str]) -> List[Dict[str, Any]]:
    def number_value(value: str) -> int | None:
        normalized = value.lower().replace("-", " ").strip()
        if normalized.isdigit():
            return int(normalized)
        words = normalized.split()
        values = {
            "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
            "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
            "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
            "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17,
            "eighteen": 18, "nineteen": 19, "twenty": 20, "thirty": 30,
            "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70,
            "eighty": 80, "ninety": 90,
        }
        if len(words) == 1:
            return values.get(words[0])
        if len(words) == 2 and words[0] in values and words[1] in values:
            return values[words[0]] + values[words[1]]
        return None

    def number_match(pattern: str, text: str) -> int | None:
        match = re.search(pattern, text, re.IGNORECASE)
        if not match:
            return None
        parenthetical = re.search(r"\((\d+)\)", match.group(0))
        return int(parenthetical.group(1)) if parenthetical else number_value(match.group(1))

    sentences = [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+|\n+", contract)
        if len(sentence.strip()) > 10
    ]
    findings: List[Dict[str, Any]] = []

    for rule in rules:
        clean_rule = rule.strip()
        lower_rule = clean_rule.lower()
        citation = ""
        compliant: bool | None = None
        reason = "The local fallback could not determine compliance from an explicit operative clause."
        candidate = ""

        if any(term in lower_rule for term in ("payment", "net-", "net ", "invoice")):
            required_days = number_match(r"net[- ]*(\d+)", lower_rule)
            if required_days is None:
                required_days = number_match(
                    r"(?:within|no later than|at most|maximum of)\s+([\w-]+)\s+days?",
                    lower_rule,
                )
            candidates = [
                sentence for sentence in sentences
                if re.search(r"\bnet\s*-?\s*\d+\b|\b(?:payment|invoice|payable|due)\b", sentence, re.I)
            ]
            for sentence in candidates:
                days = number_match(r"\bnet\s*-?\s*(\d+)\b", sentence)
                if days is None:
                    days = number_match(
                        r"\b(?:within|due in|payable within|no later than)\s+([\w-]+)(?:\s+\(\d+\))?\s+(?:calendar\s+)?days?",
                        sentence,
                    )
                if days is not None:
                    candidate = sentence
                    if required_days is not None:
                        compliant = days <= required_days
                        reason = (
                            f"Contract payment term is {days} days; the rule allows no more than "
                            f"{required_days} days."
                        )
                    break
            if not candidate and required_days is not None:
                compliant = False
                reason = "No payment-term clause was found in the contract."

        elif "liabilit" in lower_rule and any(term in lower_rule for term in ("cap", "limit", "fee")):
            required_multiple = number_match(r"\b(\d+)\s*x\b", lower_rule)
            if required_multiple is None:
                required_multiple = number_match(r"\b(\w+(?:-\w+)?)\s+times\s+(?:the\s+)?fees?\b", lower_rule)
            candidates = [
                sentence for sentence in sentences
                if "liabilit" in sentence.lower() and any(
                    term in sentence.lower() for term in ("fees", "fee", "aggregate", "cap")
                )
            ]
            if candidates:
                candidate = candidates[0]
                actual_multiple = number_match(r"\b(\d+)\s*x\b", candidate)
                if actual_multiple is None:
                    actual_multiple = number_match(
                        r"\b(\w+(?:-\w+)?)\s+times\s+(?:the\s+)?fees?\b", candidate
                    )
                if actual_multiple is None and re.search(
                    r"(?:in no event|not exceed|limited to|capped at).{0,120}\bexceed.{0,80}\bfees?\b"
                    r"|(?:limited to|capped at).{0,80}\bfees?\b",
                    candidate,
                    re.I,
                ):
                    actual_multiple = 1
                if required_multiple is not None and actual_multiple is not None:
                    compliant = actual_multiple >= required_multiple
                    reason = (
                        f"Contract liability cap is {actual_multiple}x the referenced fees; "
                        f"the rule requires at least {required_multiple}x."
                    )
            elif required_multiple is not None:
                compliant = False
                reason = "No liability-cap clause was found in the contract."

        elif any(term in lower_rule for term in ("breach notification", "security breach", "notify customer")):
            required_hours = number_match(r"\b(\d+)\s+hours?\b", lower_rule)
            candidates = [
                sentence for sentence in sentences
                if re.search(r"\b(?:breach|security incident|notify|notification)\b", sentence, re.I)
                and re.search(r"\bhours?\b", sentence, re.I)
            ]
            if candidates:
                candidate = candidates[0]
                actual_hours = number_match(r"\b([\w-]+)(?:\s+\(\d+\))?\s+hours?\b", candidate)
                if required_hours is not None and actual_hours is not None:
                    compliant = actual_hours <= required_hours
                    reason = (
                        f"Contract requires notification within {actual_hours} hours; "
                        f"the rule allows no more than {required_hours} hours."
                    )
            elif required_hours is not None:
                compliant = False
                reason = "No breach-notification deadline was found in the contract."

        elif any(term in lower_rule for term in ("governing law", "arbitration", "outside delaware")):
            candidates = [
                sentence for sentence in sentences
                if re.search(r"\b(?:governing law|governed by|laws of|arbitration|venue|jurisdiction|seat)\b", sentence, re.I)
            ]
            outside_delaware_required = bool(
                re.search(r"\boutside\s+(?:the\s+)?(?:state\s+of\s+)?delaware\b", lower_rule)
                or "not delaware" in lower_rule
            )
            if candidates:
                candidate = next(
                    (sentence for sentence in candidates if "delaware" in lower_rule and "delaware" in sentence.lower()),
                    candidates[0],
                )
                if outside_delaware_required and "delaware" in candidate.lower():
                    compliant = False
                    reason = "The operative law, arbitration, or venue clause fixes the matter in Delaware."
                elif outside_delaware_required and not re.search(
                    r"\b(?:customer|client)\s+(?:may|can|shall be entitled to)\s+(?:choose|select|elect)\b",
                    candidate,
                    re.I,
                ):
                    compliant = "delaware" not in candidate.lower()
                    reason = (
                        "The clause identifies a non-Delaware location."
                        if compliant
                        else "The clause does not establish a location outside Delaware."
                    )
            elif outside_delaware_required:
                compliant = False
                reason = "No governing-law, arbitration, or venue clause was found in the contract."

        elif any(term in lower_rule for term in ("choose forum", "choice of forum", "select forum", "forum")):
            candidates = [
                sentence for sentence in sentences
                if re.search(r"\b(?:forum|venue|jurisdiction|arbitration)\b", sentence, re.I)
            ]
            choice = re.compile(
                r"\b(?:customer|client)\s+(?:may|can|shall be entitled to)\s+"
                r"(?:choose|select|elect)\b.{0,100}\b(?:forum|venue|jurisdiction)\b",
                re.I,
            )
            if candidates:
                candidate = next((sentence for sentence in candidates if choice.search(sentence)), candidates[0])
                compliant = bool(choice.search(candidate))
                reason = (
                    "The clause expressly allows the Customer to choose the forum."
                    if compliant
                    else "The contract does not give the Customer a choice of forum."
                )
            else:
                compliant = False
                reason = "No forum-selection clause allowing the Customer to choose was found in the contract."

        elif re.search(r"\b(?:notice|cure period|termination).{0,50}\bday", lower_rule):
            required_days = number_match(r"\b(\d+)\s+days?\b", lower_rule)
            candidates = [
                sentence for sentence in sentences
                if re.search(r"\b(?:notice|cure|terminate|termination)\b", sentence, re.I)
            ]
            if candidates:
                candidate = candidates[0]
                actual_days = number_match(r"\b([\w-]+)(?:\s+\(\d+\))?\s+days?\b", candidate)
                if required_days is not None and actual_days is not None:
                    requires_minimum = any(term in lower_rule for term in ("at least", "minimum", "no less than"))
                    compliant = actual_days >= required_days if requires_minimum else actual_days <= required_days
                    reason = (
                        f"Contract specifies {actual_days} days; the rule "
                        f"{'requires at least' if requires_minimum else 'allows no more than'} {required_days} days."
                    )

        elif "aes-256" in lower_rule:
            candidate = next((s for s in sentences if "aes-256" in s.lower()), "")
            compliant = bool(candidate)
            reason = "Contract explicitly requires AES-256 encryption." if candidate else "AES-256 encryption is not specified in the contract."

        elif "consequential damages" in lower_rule or "damages must be waived" in lower_rule:
            candidate = next((s for s in sentences if "consequential damages" in s.lower()), "")
            compliant = bool(candidate and re.search(r"\b(?:waive|waived|shall not be liable|not liable)\b", candidate, re.I))
            reason = (
                "The contract explicitly waives consequential damages."
                if compliant
                else "No operative consequential-damages waiver was found in the contract."
            )

        if compliant is None and re.search(r"\b(?:must|shall|required|requires)\b", lower_rule):
            topic_words = [
                word for word in re.findall(r"\w+", lower_rule)
                if len(word) > 4 and word not in {
                    "agreement", "contract", "maintain", "required", "requires",
                    "shall", "must", "within", "under", "policy",
                }
            ]
            if topic_words and not any(word in contract.lower() for word in topic_words):
                compliant = False
                reason = "No matching operative clause was found in the contract."

        if candidate:
            citation = candidate
        elif compliant is False and "No " not in reason:
            reason = "No matching operative clause was found in the contract."

        findings.append({
            "rule": clean_rule,
            "compliant": compliant,
            "citation": citation,
            "reason": reason,
        })
    return findings

def auditor_node(state: ComplianceState) -> Dict[str, Any]:
    current_count = state.get("loop_count", 0)
    contract_text = state.get("retrieval_context") or state.get("contract_text", "")
    source_contract_text = state.get("source_contract_text") or state.get("contract_text", "")
    compliance_rules = state.get("compliance_rules", [])
    budget_limit = state.get("budget_limit", DEFAULT_BUDGET_LIMIT)
    accumulated_cost = state.get("estimated_cost", 0.0)
    accumulated_tokens = state.get("total_tokens", 0)
    cost_ledger = list(state.get("cost_breakdown", []))

    feedback_text = state.get("critic_feedback", "")
    feedback_context = ("\nPrevious Critic Feedback: " + feedback_text) if (current_count > 0 and feedback_text) else ""
    prompt = f"""
    You are an expert Corporate Compliance Auditor.
    Analyze each rule independently and compare explicit contract values with the rule's threshold.
    A mandatory provision that is absent is non-compliant. If the evidence is insufficient to decide,
    use null for compliant. Never infer a compliant result from keyword overlap alone.
    Each citation must be one contiguous, verbatim excerpt from the contract text below; do not
    combine separate clauses or cite instructions from the rules.
    Compliance Rules to Check: {compliance_rules}
    Contract Text: {contract_text}
    {feedback_context}
    Respond STRICTLY with a valid JSON object matching this schema:
    {{
      "findings": [
        {{"rule": "name of rule", "compliant": true, "citation": "exact text snippet from contract", "reason": "why"}}
      ]
    }}
    """

    parsed_report = None
    analysis_mode = "heuristic_fallback"
    analysis_warning = "Gemini was not called; local heuristic analysis was used."
    step_prompt_tokens = max(len(prompt) // 4, 1)
    step_completion_tokens = 150

    api_key = get_gemini_api_key()
    if api_key and not os.getenv("MOCK_LLM"):
        try:
            response = completion(
                model=GEMINI_MODEL,
                api_key=api_key,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                timeout=12
            )
            parsed_report = json.loads(response.choices[0].message.content)
            analysis_mode = "gemini"
            analysis_warning = None
            if hasattr(response, "usage") and response.usage:
                step_prompt_tokens = getattr(response.usage, "prompt_tokens", step_prompt_tokens)
                step_completion_tokens = getattr(response.usage, "completion_tokens", step_completion_tokens)
        except Exception as exc:
            parsed_report = None
            analysis_warning = f"Gemini analysis failed ({type(exc).__name__}); local heuristic analysis was used."
    elif api_key and os.getenv("MOCK_LLM"):
        analysis_warning = "MOCK_LLM is enabled; local heuristic analysis was used."
    elif not api_key:
        analysis_warning = "No Gemini API key is configured; local heuristic analysis was used."

    if not isinstance(parsed_report, dict) or not isinstance(parsed_report.get("findings"), list):
        if analysis_mode == "gemini":
            analysis_warning = "Gemini returned an invalid report; local heuristic analysis was used."
        analysis_mode = "heuristic_fallback"
        findings = _evaluate_rules_heuristically(source_contract_text, compliance_rules)
        parsed_report = {"findings": findings}
        step_completion_tokens = max(len(json.dumps(findings)) // 4, 50)

    parsed_report["analysis"] = {
        "mode": analysis_mode,
        "warning": analysis_warning,
    }

    step_cost = calculate_step_cost(step_prompt_tokens, step_completion_tokens)
    new_total_cost = accumulated_cost + step_cost
    new_total_tokens = accumulated_tokens + step_prompt_tokens + step_completion_tokens
    budget_exceeded = new_total_cost >= budget_limit

    cost_ledger.append({
        "step": f"auditor_loop_{current_count + 1}",
        "prompt_tokens": step_prompt_tokens,
        "completion_tokens": step_completion_tokens,
        "total_tokens": step_prompt_tokens + step_completion_tokens,
        "cost_usd": round(step_cost, 7)
    })

    parsed_report["cost_summary"] = {
        "total_tokens": new_total_tokens,
        "estimated_cost_usd": round(new_total_cost, 6),
        "budget_limit_usd": budget_limit,
        "budget_exceeded": budget_exceeded,
        "cost_breakdown": cost_ledger
    }

    return {
        "audit_report": parsed_report,
        "loop_count": current_count + 1,
        "total_tokens": new_total_tokens,
        "estimated_cost": new_total_cost,
        "budget_exceeded": budget_exceeded,
        "cost_breakdown": cost_ledger
    }

def critic_node(state: ComplianceState) -> Dict[str, Any]:
    report = state.get("audit_report", {})
    contract = state.get("source_contract_text") or state.get("contract_text", "")
    budget_limit = state.get("budget_limit", DEFAULT_BUDGET_LIMIT)
    accumulated_cost = state.get("estimated_cost", 0.0)
    accumulated_tokens = state.get("total_tokens", 0)
    cost_ledger = list(state.get("cost_breakdown", []))
    current_count = state.get("loop_count", 1)

    prompt = f"""
    You are a Senior Legal Quality Controller.
    Review this Audit Report against the raw Contract Text to verify that citations are exact and relevant
    to their associated rules. Reject text that is merely a party name or otherwise unrelated.
    Audit Report: {json.dumps(report)}
    Raw Contract: {contract}
    Respond strictly with a JSON object:
    {{
      "status": "pass",
      "feedback": "Your evaluation notes here"
    }}
    """

    validation = None
    critic_mode = "local_citation_check"
    critic_warning = None
    step_prompt_tokens = max(len(prompt) // 4, 1)
    step_completion_tokens = 50

    api_key = get_gemini_api_key()
    if api_key and not os.getenv("MOCK_LLM"):
        try:
            response = completion(
                model=GEMINI_MODEL,
                api_key=api_key,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                timeout=10
            )
            validation = json.loads(response.choices[0].message.content)
            critic_mode = "gemini"
            if hasattr(response, "usage") and response.usage:
                step_prompt_tokens = getattr(response.usage, "prompt_tokens", step_prompt_tokens)
                step_completion_tokens = getattr(response.usage, "completion_tokens", step_completion_tokens)
        except Exception as exc:
            validation = None
            critic_warning = (
                f"Gemini citation review failed ({type(exc).__name__}); "
                "deterministic local citation verification was used."
            )
    else:
        critic_warning = (
            "No Gemini API key is configured; deterministic local citation verification was used."
            if not api_key
            else "MOCK_LLM is enabled; deterministic local citation verification was used."
        )

    if not isinstance(validation, dict) or not validation:
        findings = report.get("findings", [])
        feedback_notes = []
        for f in findings:
            cite = f.get("citation", "")
            if _finding_has_no_citation_evidence(f):
                continue
            if cite and _citation_is_exact(cite, contract) and _citation_is_relevant(
                f.get("rule", ""), cite
            ):
                continue
            if cite and not _citation_is_exact(cite, contract):
                feedback_notes.append(f"Citation for rule '{f.get('rule')}' is not an exact contract excerpt.")
            else:
                feedback_notes.append(f"Citation for rule '{f.get('rule')}' is not relevant to that rule.")
        status = "fail" if feedback_notes or not findings else "pass"
        validation = {
            "status": status,
            "feedback": (
                "All cited excerpts exactly match contract text."
                if findings and not feedback_notes
                else "; ".join(feedback_notes) or "No findings were produced to verify."
            ),
        }
    else:
        validation["status"] = str(validation.get("status", "fail")).lower()
        if validation["status"] not in {"pass", "fail"}:
            validation["status"] = "fail"

    citation_issues = []
    for finding in report.get("findings", []):
        citation = finding.get("citation", "")
        if _finding_has_no_citation_evidence(finding):
            continue
        if not citation:
            citation_issues.append(f"No verifiable citation was provided for rule '{finding.get('rule')}'.")
        elif not _citation_is_exact(citation, contract):
            citation_issues.append(
                f"Citation for rule '{finding.get('rule')}' is not an exact contract excerpt."
            )
        elif not _citation_is_relevant(finding.get("rule", ""), citation):
            citation_issues.append(f"Citation for rule '{finding.get('rule')}' is not relevant to that rule.")
    if citation_issues and validation.get("status") == "pass":
        validation["status"] = "fail"
        validation["feedback"] = "; ".join(citation_issues)

    corrected_report = dict(report)
    corrected_findings = []
    for finding in report.get("findings", []):
        if finding.get("citation") and (
            not _citation_is_exact(finding["citation"], contract)
            or not _citation_is_relevant(finding.get("rule", ""), finding["citation"])
        ):
            finding = {
                **finding,
                "citation": "",
                "compliant": None,
                "reason": "Citation could not be verified as exact, relevant contract evidence; verdict is undetermined.",
            }
        corrected_findings.append(finding)
    corrected_report["findings"] = corrected_findings

    step_cost = calculate_step_cost(step_prompt_tokens, step_completion_tokens)
    new_total_cost = accumulated_cost + step_cost
    new_total_tokens = accumulated_tokens + step_prompt_tokens + step_completion_tokens
    budget_exceeded = new_total_cost >= budget_limit

    cost_ledger.append({
        "step": f"critic_loop_{current_count}",
        "prompt_tokens": step_prompt_tokens,
        "completion_tokens": step_completion_tokens,
        "total_tokens": step_prompt_tokens + step_completion_tokens,
        "cost_usd": round(step_cost, 7)
    })

    return {
        "audit_report": corrected_report,
        "status": validation.get("status", "pass"),
        "critic_feedback": validation.get("feedback", ""),
        "critic_mode": critic_mode,
        "critic_warning": critic_warning,
        "total_tokens": new_total_tokens,
        "estimated_cost": new_total_cost,
        "budget_exceeded": budget_exceeded,
        "cost_breakdown": cost_ledger
    }


def citation_review_from_state(state: Dict[str, Any]) -> Dict[str, str]:
    mode = state.get("critic_mode")
    if not mode:
        return {
            "status": "not_run",
            "feedback": "Citation review was not executed.",
            "mode": "not_run",
        }

    status = state.get("status")
    if status not in {"pass", "fail"}:
        status = "fail"
    return {
        "status": status,
        "feedback": state.get("critic_feedback", ""),
        "mode": str(mode),
        "warning": state.get("critic_warning", ""),
    }


def _citation_is_exact(citation: str, contract: str) -> bool:
    if not isinstance(citation, str):
        return False
    normalized_citation = " ".join(citation.split()).casefold()
    normalized_contract = " ".join(contract.split()).casefold()
    return bool(normalized_citation and normalized_citation in normalized_contract)


def _finding_has_no_citation_evidence(finding: Dict[str, Any]) -> bool:
    reason = str(finding.get("reason", "")).casefold()
    return (
        not finding.get("citation")
        and finding.get("compliant") is not True
        and (
            bool(re.search(r"\bno\b.{0,80}\b(?:clause|provision|deadline|term|evidence)\b", reason))
            or any(term in reason for term in ("could not determine", "insufficient evidence", "unable to verify"))
        )
    )


def _citation_is_relevant(rule: str, citation: str) -> bool:
    if not isinstance(rule, str) or not isinstance(citation, str):
        return False
    rule_text = rule.casefold()
    citation_text = citation.casefold()
    relevant_terms = (
        (("payment", "invoice", "net-"), ("net", "payment", "invoice", "payable", "due")),
        (("liability", "liability cap"), ("liability", "fees", "aggregate", "cap")),
        (("breach notification", "security breach", "notify customer"), ("breach", "security incident", "notify", "notification")),
        (("governing law", "arbitration", "outside delaware"), ("governing law", "governed by", "laws of", "arbitration", "venue", "jurisdiction", "seat")),
        (("forum", "choose forum", "choice of forum"), ("forum", "venue", "jurisdiction", "arbitration")),
        (("aes-256",), ("aes-256",)),
        (("consequential damages", "damages must be waived"), ("consequential damages",)),
        (("notice", "termination", "cure period"), ("notice", "termination", "cure")),
    )
    for rule_terms, citation_terms in relevant_terms:
        if any(term in rule_text for term in rule_terms):
            return any(term in citation_text for term in citation_terms)

    generic_terms = {
        "agreement", "contract", "must", "shall", "should", "required", "requires",
        "rule", "customer", "vendor", "within", "from", "that", "this", "with",
    }
    rule_words = {
        word for word in re.findall(r"\w+", rule_text)
        if len(word) > 3 and word not in generic_terms and not word.isdigit()
    }
    citation_words = set(re.findall(r"\w+", citation_text))
    overlap = rule_words & citation_words
    return len(overlap) >= 2 and len(overlap) / max(len(rule_words), 1) >= 0.25
