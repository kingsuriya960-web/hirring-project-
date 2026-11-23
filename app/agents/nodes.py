import os
import json
import re
import time
from collections import Counter
from typing import Dict, Any
from openai import OpenAI
from app.agents.graph import ComplianceState

GROQ_MODELS = [
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
]
_PAGE_REFERENCE = re.compile(
    r"[\(\[]?\s*pages?\s*#?\s*\d+(?:\s*(?:-|–|to|and|,)\s*\d+)*\s*[\)\]]?",
    re.IGNORECASE,
)
_PAGE_MARKER = re.compile(r"---\s*PAGE\s+\d+\s*---", re.IGNORECASE)
MAX_CITATION_WORDS = 60


def _get_client() -> OpenAI:
    api_key = (os.getenv("GROQ_API_KEY") or "").strip()
    if not api_key:
        raise RuntimeError("Missing GROQ_API_KEY in .env file.")
    return OpenAI(
        base_url="https://api.groq.com/openai/v1",
        api_key=api_key,
    )


def _extract_json(raw_text: str) -> dict:
    cleaned = raw_text.strip()
    if "</think>" in cleaned:
        cleaned = cleaned.split("</think>")[-1].strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[-1]
        if cleaned.endswith("```"):
            cleaned = cleaned.rsplit("```", 1)[0]
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1:
        cleaned = cleaned[start : end + 1]
    return json.loads(cleaned)


def _call_groq_with_retry(prompt: str, max_retries: int = 3) -> tuple[dict, str]:
    client = _get_client()
    last_error = None
    for model_name in GROQ_MODELS:
        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a strict legal compliance auditor. Always respond with valid JSON only.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.1,
                    timeout=60,
                )
                content = response.choices[0].message.content or ""
                actual_model = getattr(response, "model", None) or model_name
                return _extract_json(content), f"groq/{actual_model}"
            except Exception as e:
                last_error = f"[{model_name} attempt {attempt + 1}]: {type(e).__name__} - {str(e)[:220]}"
                if getattr(e, "status_code", None) == 404:
                    break
                time.sleep(1.5 ** attempt)
    raise RuntimeError(f"All Groq models failed after retries. Last error: {last_error}")


def _findings_match_rules(report: dict, rules: list[str]) -> bool:
    findings = report.get("findings")
    if not isinstance(findings, list) or not all(
        isinstance(finding, dict) for finding in findings
    ):
        return False

    def normalize_rule(value: Any) -> str:
        return " ".join(str(value or "").split()).casefold()

    finding_rules = [normalize_rule(finding.get("rule")) for finding in findings]
    return Counter(finding_rules) == Counter(normalize_rule(rule) for rule in rules)


def _source_quote_match(quote: str, page_text: str) -> re.Match[str] | None:
    parts = quote.split()
    if not parts:
        return None
    pattern = r"\s+".join(re.escape(part) for part in parts)
    return re.search(pattern, page_text)


def _citation_matches_page(
    finding: dict, quote: str, page_text: str
) -> bool:
    start = finding.get("citation_start")
    end = finding.get("citation_end")
    if start is None and end is None:
        return quote in page_text
    return (
        isinstance(start, int)
        and not isinstance(start, bool)
        and isinstance(end, int)
        and not isinstance(end, bool)
        and 0 <= start <= end <= len(page_text)
        and page_text[start:end] == quote
    )


def _build_sentence_index(source_pages: list[dict]) -> list[dict]:
    sentence_index = []
    for page in source_pages:
        page_number = page.get("page")
        page_text = str(page.get("text") or "")
        if not isinstance(page_number, int) or isinstance(page_number, bool):
            continue
        boundaries = [
            match.end()
            for match in re.finditer(r"(?<=[.!?])\s+|\n+", page_text)
        ]
        starts = [0, *boundaries]
        ends = [*boundaries, len(page_text)]
        sentence_number = 0
        for start, end in zip(starts, ends):
            while start < end and page_text[start].isspace():
                start += 1
            while end > start and page_text[end - 1].isspace():
                end -= 1
            if start >= end:
                continue
            sentence_number += 1
            sentence_index.append(
                {
                    "id": f"P{page_number}-S{sentence_number}",
                    "page": page_number,
                    "text": page_text[start:end],
                    "start": start,
                    "end": end,
                }
            )
    return sentence_index


def _format_sentence_context(sentence_index: list[dict]) -> str:
    return "\n".join(
        f'[{sentence["id"]}] {sentence["text"]}'
        for sentence in sentence_index
    )


def _citation_candidates(finding: dict) -> list[str]:
    candidates = finding.get("citation_quotes")
    if isinstance(candidates, list):
        return [str(candidate) for candidate in candidates if str(candidate).strip()]
    citations = finding.get("citations")
    if isinstance(citations, list):
        return [
            str(citation.get("quote") or "")
            if isinstance(citation, dict)
            else str(citation)
            for citation in citations
            if (
                isinstance(citation, dict)
                and str(citation.get("quote") or "").strip()
            )
            or (not isinstance(citation, dict) and str(citation).strip())
        ]
    citation = str(finding.get("citation") or "")
    return [citation] if citation.strip() else []


def _ground_findings_to_source_pages(
    findings: list[dict],
    source_pages: list[dict],
    total_pages: int | None,
    sentence_index: list[dict] | None = None,
) -> None:
    sentence_lookup = {
        sentence["id"]: sentence
        for sentence in (sentence_index or _build_sentence_index(source_pages))
    }
    for finding in findings:
        if str(finding.get("result", "")).casefold().startswith("gap"):
            page_span = (
                f"Pages 1-{total_pages}"
                if isinstance(total_pages, int) and total_pages > 0
                else "the complete contract"
            )
            finding["citation"] = (
                f"No relevant language found in Contract Text ({page_span})."
            )
            finding["citations"] = []
            finding["source_reference"] = page_span
            finding.pop("citation_quotes", None)
            continue

        if "citation_sentence_ids" in finding:
            selected_ids = finding.get("citation_sentence_ids")
            if not isinstance(selected_ids, list):
                finding["citation_grounding_failed"] = True
                continue
            if not selected_ids or any(
                not isinstance(sentence_id, str)
                or sentence_id not in sentence_lookup
                for sentence_id in selected_ids
            ):
                finding["citation_grounding_failed"] = True
                continue
            selected = [
                sentence_lookup[sentence_id] for sentence_id in selected_ids
            ]
            grounded_citations = [
                {
                    "quote": sentence["text"],
                    "page": sentence["page"],
                    "start": sentence["start"],
                    "end": sentence["end"],
                }
                for sentence in selected
            ]
            finding["citations"] = grounded_citations
            finding["citation_quotes"] = [
                item["quote"] for item in grounded_citations
            ]
            finding["citation"] = grounded_citations[0]["quote"]
            finding["citation_page"] = grounded_citations[0]["page"]
            finding["citation_start"] = grounded_citations[0]["start"]
            finding["citation_end"] = grounded_citations[0]["end"]
            page_numbers = list(
                dict.fromkeys(item["page"] for item in grounded_citations)
            )
            finding["source_reference"] = ", ".join(
                f"Page {page}" for page in page_numbers
            )
            continue

        quotes = _citation_candidates(finding)
        if not quotes:
            finding["citation_grounding_failed"] = True
            continue

        grounded_citations = []
        for citation in quotes:
            claimed_pages = [
                int(page_number)
                for page_reference in _PAGE_REFERENCE.finditer(citation)
                for page_number in re.findall(r"\d+", page_reference.group())
            ]
            quote = _PAGE_REFERENCE.sub(" ", citation).strip()
            quote = quote.strip(" \t\r\n()[],:;—–-\"'“”")
            if len(quote.split()) > MAX_CITATION_WORDS:
                finding["citation_grounding_failed"] = True
                break
            matches = [
                (page, match)
                for page in source_pages
                if (
                    match := _source_quote_match(
                        quote, str(page.get("text") or "")
                    )
                )
            ]
            if not matches:
                finding["citation_grounding_failed"] = True
                break

            selected_page, match = next(
                (
                    candidate
                    for candidate in matches
                    if candidate[0].get("page") in claimed_pages
                ),
                matches[0],
            )
            page_number = selected_page.get("page")
            if not isinstance(page_number, int):
                finding["citation_grounding_failed"] = True
                break
            grounded_citations.append(
                {
                    "quote": match.group(0),
                    "page": page_number,
                    "start": match.start(),
                    "end": match.end(),
                }
            )

        if finding.get("citation_grounding_failed") or not grounded_citations:
            continue

        finding["citations"] = grounded_citations
        finding["citation_quotes"] = [item["quote"] for item in grounded_citations]
        finding["citation"] = grounded_citations[0]["quote"]
        finding["citation_page"] = grounded_citations[0]["page"]
        finding["citation_start"] = grounded_citations[0]["start"]
        finding["citation_end"] = grounded_citations[0]["end"]
        page_numbers = list(dict.fromkeys(item["page"] for item in grounded_citations))
        finding["source_reference"] = ", ".join(
            f"Page {page}" for page in page_numbers
        )


def _citation_issues(
    findings: list[dict],
    source_text: str,
    total_pages: int | None,
    source_pages: list[dict] | None = None,
) -> list[str]:
    pages = source_pages or [
        {"page": int(number), "text": text}
        for number, text in re.findall(
            r"---\s*PAGE\s+(\d+)\s*---\s*(.*?)(?=---\s*PAGE\s+\d+\s*---|\Z)",
            source_text,
            flags=re.IGNORECASE | re.DOTALL,
        )
    ]
    issues = []
    for index, finding in enumerate(findings, start=1):
        if str(finding.get("result", "")).casefold().startswith("gap"):
            continue

        if finding.get("citation_grounding_failed"):
            issues.append(
                f"Finding {index} contains a citation that could not be matched "
                "to extracted PDF text."
            )
            continue

        citation_sources = finding.get("citations")
        if isinstance(citation_sources, list):
            if not citation_sources:
                issues.append(f"Finding {index} has no verbatim citations.")
                continue
            for citation_source in citation_sources:
                if not isinstance(citation_source, dict):
                    issues.append(
                        f"Finding {index} has an invalid citation source entry."
                    )
                    continue
                citation_quote = str(citation_source.get("quote") or "")
                source_check = {
                    "citation": citation_quote,
                    "citation_page": citation_source.get("page"),
                    "citation_start": citation_source.get("start"),
                    "citation_end": citation_source.get("end"),
                }
                source_issues = _citation_issues(
                    [source_check], source_text, total_pages, pages
                )
                issues.extend(
                    issue.replace("Finding 1", f"Finding {index}")
                    for issue in source_issues
                )
            continue

        citation = str(finding.get("citation") or "").strip()
        if not citation:
            issues.append(f"Finding {index} has no citation.")
            continue

        for page_reference in _PAGE_REFERENCE.finditer(citation):
            page_numbers = re.findall(r"\d+", page_reference.group())
            for page_number in page_numbers:
                page = int(page_number)
                if total_pages is not None and not 1 <= page <= total_pages:
                    issues.append(
                        f"Finding {index} cites Page {page}, but the PDF has "
                        f"{total_pages} pages."
                    )

        citation_page = finding.get("citation_page")
        if citation_page is not None and (
            not isinstance(citation_page, int)
            or isinstance(citation_page, bool)
            or (total_pages is not None and not 1 <= citation_page <= total_pages)
        ):
            issues.append(
                f"Finding {index} has an invalid source page ({citation_page})."
            )

        quote = _PAGE_REFERENCE.sub(" ", citation)
        quote = quote.strip(" \t\r\n()[],:;—–-\"'“”")
        if not quote.split():
            issues.append(f"Finding {index} has no verbatim quote.")
            continue

        referenced_pages = [
            int(page_number)
            for page_reference in _PAGE_REFERENCE.finditer(citation)
            for page_number in re.findall(r"\d+", page_reference.group())
        ]
        if (
            citation_page is not None
            and referenced_pages
            and any(page != citation_page for page in referenced_pages)
        ):
            issues.append(
                f"Finding {index} page label conflicts with its extracted source page."
            )
        expected_pages = (
            [citation_page]
            if citation_page is not None
            else referenced_pages
        )
        if expected_pages:
            page_matches = [
                page for page in pages if page.get("page") in expected_pages
            ]
            found = any(
                _citation_matches_page(
                    finding, quote, str(page.get("text") or "")
                )
                for page in page_matches
            )
            if not found:
                page_label = (
                    f"Page {expected_pages[0]}"
                    if len(expected_pages) == 1
                    else "Pages " + ", ".join(str(page) for page in expected_pages)
                )
                issues.append(
                    f"Finding {index} citation does not appear verbatim on "
                    f"cited {page_label}."
                )
        elif pages and not any(
            _citation_matches_page(finding, quote, str(page.get("text") or ""))
            for page in pages
        ):
            issues.append(
                f"Finding {index} citation does not appear verbatim in the source PDF."
            )
        elif not pages:
            source_body = _PAGE_MARKER.sub(" ", source_text)
            if quote not in source_body:
                issues.append(
                    f"Finding {index} citation does not appear verbatim in the source PDF."
                )
    return issues


def auditor_node(state: ComplianceState) -> Dict[str, Any]:
    current_count = state.get("loop_count", 0)
    feedback_context = (
        f"\nPrevious Critic Feedback to fix: {state.get('critic_feedback')}"
        if current_count > 0
        else ""
    )

    total_pages = state.get("total_pages")
    page_span = (
        f"Pages 1-{total_pages}"
        if isinstance(total_pages, int) and total_pages > 0
        else "the complete contract"
    )
    source_pages = state.get("source_pages") or [
        {"page": 1, "text": str(state["contract_text"])}
    ]
    sentence_index = _build_sentence_index(source_pages)
    source_context = _format_sentence_context(sentence_index)
    prompt = f"""
Evaluate the Contract Text against EVERY rule in the Compliance Rules list.

CRITICAL INSTRUCTIONS:
1. ONE VERDICT PER RULE: Produce exactly one finding object per input rule. NEVER split a compound rule (containing AND / OR) into two findings. Evaluate each branch in the reasoning. An OR rule is Compliant if any branch passes, and Non-Compliant only if all branches fail.
2. EXPLICIT MATH & LOGIC COMPARISON: For numeric rules (e.g., Net-30, >=3x fees, <=24h notice, >=2 years retention), extract the exact number from the contract and explicitly state the mathematical comparison in 'reasoning'.
3. EVIDENCE-ONLY REVIEW: Use only the sentence-indexed source text below. Each sentence ID maps to an exact span of extracted PDF text and a real page.
4. CITATIONS: Do not write, paraphrase, shorten, or combine any citation text. Return citation_sentence_ids containing the ID(s) of the source sentences that directly support the finding. Select the shortest sufficient set, and use multiple IDs only for distinct provisions needed to evaluate the rule.
5. PAGE NUMBERS: Do not return or invent page numbers. Python resolves each selected sentence ID to its extracted source page.
6. GAP POLICY: If no relevant provision exists, use result "Gap — Missing Protection" and return citation_sentence_ids as an empty list. Python creates the absence citation for {page_span}.
7. If a sentence does not support the finding, do not select its ID. Every selected ID must appear exactly in the source text below.
8. Set 'next_step' to "No action required" when Compliant. For Non-Compliant or Gap findings, give a concrete negotiation or escalation action.

Compliance Rules (Evaluate each of these {len(state['compliance_rules'])} rules):
{json.dumps(state['compliance_rules'], indent=2)}

Sentence-indexed source text:
<source>
{source_context}
</source>
{feedback_context}

Respond STRICTLY with valid JSON matching this schema:
{{
  "findings": [
    {{
      "rule": "exact rule text from input",
      "result": "Compliant" | "Non-Compliant" | "Undetermined" | "Gap — Missing Protection",
      "priority": "High" | "Medium" | "Low",
      "citation_sentence_ids": ["P1-S1"],
      "reasoning": "explicit number-vs-number or condition-vs-text comparison",
      "risk_if_missing": "business risk if the requirement is absent or not met",
      "confidence": "High" | "Medium" | "Low",
      "confidence_reason": "evidence basis for the confidence rating",
      "next_step": "concrete action item"
    }}
  ]
}}
"""
    try:
        parsed_report, used_model = _call_groq_with_retry(prompt)
        rules = state["compliance_rules"]
        if not isinstance(parsed_report, dict) or not _findings_match_rules(
            parsed_report, rules
        ):
            previous_findings = (
                parsed_report.get("findings", [])
                if isinstance(parsed_report, dict)
                else parsed_report
            )
            correction_prompt = f"""
The previous audit response did not contain exactly one finding for every input rule.
Correct it and return ONLY valid JSON using the same findings schema.

Requirements:
1. Return exactly {len(rules)} findings, one per input rule, with each "rule" field
   copied exactly from its corresponding input rule.
2. Never split one input rule into multiple findings. Evaluate each branch of a
   compound rule in one finding; an OR rule is Compliant if either branch passes.
3. Never merge separate input rules. Return citation_sentence_ids containing
   only IDs of source sentences that directly support the finding.
4. Leave citation_sentence_ids empty when the contract is silent, and emit one
   "Gap — Missing Protection" finding rather than omitting that rule.

Input rules:
{json.dumps(rules, indent=2)}

Sentence-indexed source text:
<source>
{source_context}
</source>

Previous findings to correct:
{json.dumps(previous_findings, indent=2)}

Return this shape:
{{
  "findings": [
    {{
      "rule": "exact input rule",
      "result": "Compliant" | "Non-Compliant" | "Undetermined" | "Gap — Missing Protection",
      "priority": "High" | "Medium" | "Low",
      "citation_sentence_ids": ["P1-S1"],
      "reasoning": "explicit condition-vs-text comparison",
      "risk_if_missing": "business risk if the requirement is absent or not met",
      "confidence": "High" | "Medium" | "Low",
      "confidence_reason": "evidence basis for the confidence rating",
      "next_step": "concrete action item"
    }}
  ]
}}
"""
            corrected_report, correction_model = _call_groq_with_retry(
                correction_prompt
            )
            if not isinstance(corrected_report, dict) or not _findings_match_rules(
                corrected_report, rules
            ):
                corrected_findings = (
                    corrected_report.get("findings")
                    if isinstance(corrected_report, dict)
                    else None
                )
                corrected_count = (
                    len(corrected_findings)
                    if isinstance(corrected_findings, list)
                    else 0
                )
                raise RuntimeError(
                    "Finding count/rule mapping mismatch after correction: "
                    f"expected {len(rules)} findings, received {corrected_count}."
                )
            parsed_report = corrected_report
            used_model = f"{used_model} (corrected via {correction_model})"
        findings = parsed_report.get("findings")
        if isinstance(findings, list):
            _ground_findings_to_source_pages(
                [finding for finding in findings if isinstance(finding, dict)],
                source_pages,
                total_pages,
                sentence_index,
            )
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
            "critic_feedback": "Skipped due to upstream model failure.",
            "citation_review_executed": False,
            "citation_review_status": "NOT RUN — AI analysis failed",
            "citation_review_detail": "Findings were withheld because the auditor failed.",
            "verified_citations": 0,
            "total_citations": 0,
            "audit_report": {"findings": []},
        }

    report = state.get("audit_report") or {}
    if not isinstance(report, dict):
        report = {}
    findings = report.get("findings") or []
    if not isinstance(findings, list):
        findings = []
    findings = [finding for finding in findings if isinstance(finding, dict)]
    rules = state.get("compliance_rules") or []
    total_citations = sum(
        (
            len(finding["citations"])
            if isinstance(finding.get("citations"), list)
            and finding["citations"]
            else 1
        )
        for finding in findings
        if not str(finding.get("result", "")).casefold().startswith("gap")
    )
    source_text = state.get("source_contract_text") or state.get("contract_text", "")
    total_pages = state.get("total_pages")
    source_pages = state.get("source_pages") or None

    def incomplete_review(
        detail: str,
        *,
        citation_review_executed: bool,
        citation_review_status: str,
        verified_citations: int = 0,
    ) -> Dict[str, Any]:
        return {
            "status": "incomplete",
            "critic_feedback": detail,
            "citation_review_executed": citation_review_executed,
            "citation_review_status": citation_review_status,
            "citation_review_detail": detail,
            "verified_citations": verified_citations,
            "total_citations": total_citations,
            "audit_report": {"findings": []},
        }

    if not _findings_match_rules(report, rules):
        return incomplete_review(
            f"Finding coverage failed: expected exactly one finding for each of "
            f"{len(rules)} input rules. Findings were withheld.",
            citation_review_executed=True,
            citation_review_status="FAIL",
        )

    citation_issues = _citation_issues(
        findings, source_text, total_pages, source_pages
    )
    if citation_issues:
        return incomplete_review(
            " ".join(citation_issues) + " Findings were withheld.",
            citation_review_executed=True,
            citation_review_status="FAIL",
        )

    prompt = f"""
You are a Legal Quality Controller. Verify the Audit Report against the Raw Contract Text.
Check:
1. Are all {len(rules)} rules evaluated correctly? Each input rule already has exactly one finding; do not split OR conditions or merge distinct rules.
2. Are both/all branches of each compound rule discussed in the one finding's reasoning? An OR passes if at least one branch passes.
3. Are all numeric comparisons (<=, >=) mathematically accurate?
4. For a Gap finding, is the contract actually silent on that requirement?

Audit Report: {json.dumps(report)}
Raw Contract:
<contract_document>
{source_text}
</contract_document>

Respond strictly with valid JSON:
{{
  "status": "pass" | "fail",
  "verified_count": <integer count of confirmed citations>,
  "feedback": "Specific corrections needed if fail, or confirmation summary if pass"
}}
"""
    try:
        validation, critic_model = _call_groq_with_retry(prompt)
        validation_status = (
            str(validation.get("status", "")).strip().casefold()
            if isinstance(validation, dict)
            else ""
        )
        if validation_status != "pass":
            feedback = (
                str(validation.get("feedback") or "Critic did not verify the audit.")
                if isinstance(validation, dict)
                else "Critic returned an invalid response."
            )
            return incomplete_review(
                f"{feedback} Findings were withheld.",
                citation_review_executed=True,
                citation_review_status="PASS",
                verified_citations=total_citations,
            )
        page_note = (
            f" All page references are within the {total_pages}-page PDF."
            if isinstance(total_pages, int)
            else ""
        )
        return {
            "status": "pass",
            "critic_feedback": str(validation.get("feedback") or ""),
            "citation_review_executed": True,
            "citation_review_status": "PASS",
            "citation_review_detail": (
                f"{total_citations}/{total_citations} non-gap citations matched "
                f"verbatim. Reviewed by {critic_model}.{page_note}"
            ),
            "verified_citations": total_citations,
            "total_citations": total_citations,
        }
    except Exception as e:
        return incomplete_review(
            f"Critic Agent unavailable ({type(e).__name__}: {e}). Findings were withheld.",
            citation_review_executed=False,
            citation_review_status="NOT RUN",
        )
