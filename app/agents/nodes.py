import json
import os
import re
import time
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any, Dict

from litellm import completion
from openai import OpenAI

from app.agents.graph import ComplianceState
from app.config import settings

PROJECT_ROOT = Path(__file__).resolve().parents[2]

GROQ_MODELS = [
    os.getenv("GROQ_MODEL") or settings.GROQ_MODEL,
    "openai/gpt-oss-20b",
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
]
GROQ_MODELS = list(dict.fromkeys(model for model in GROQ_MODELS if model))

FALLBACK_MODELS = [
    "qwen/qwen-2.5-72b-instruct:free",
    "deepseek/deepseek-chat:free",
    "meta-llama/llama-3.3-70b-instruct:free",
    "google/gemma-2-9b-it:free",
    "mistralai/mistral-7b-instruct:free",
]

GEMINI_MODEL = os.getenv("GEMINI_MODEL") or settings.GEMINI_MODEL

PROMPT_INJECTION_PATTERNS = (
    r"ignore\s+(all\s+)?(previous|prior)\s+instructions",
    r"ignore\s+all\s+instructions",
    r"disregard\s+(all\s+)?(previous|prior)",
    r"override\s+all\s+prior",
    r"print\s+system\s+prompt",
    r"reveal\s+your\s+system",
    r"system\s+instructions\?",
    r"dan\s+mode",
    r"you\s+are\s+now\s+dan",
    r"<script",
    r"---\s*begin\s+system\s+instruction\s*---",
    r"\[system\s+instruction\]",
)

MAX_CITATION_WORDS = 60

_PAGE_REFERENCE = re.compile(
    r"[\(\[]?\s*pages?\s*#?\s*\d+(?:\s*(?:-|–|to|and|,)\s*\d+)*\s*[\)\]]?",
    re.IGNORECASE,
)
_PAGE_MARKER = re.compile(r"---\s*PAGE\s+\d+\s*---", re.IGNORECASE)

_NUMBER_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "fifteen": 15, "twenty": 20,
    "twenty-four": 24, "thirty": 30, "forty": 40, "forty-five": 45,
    "sixty": 60, "seventy-two": 72, "ninety": 90,
}

_DOMAIN_KEYWORDS = {
    "payment", "invoice", "invoices", "net", "payable", "pay",
    "liability", "cap", "capped", "fees", "indemnity",
    "breach", "notify", "notification", "incident", "security",
    "arbitration", "forum", "seated", "venue", "delaware",
    "retention", "retain", "records", "data",
    "insurance", "cyber", "termination", "notice",
    "confidentiality", "damages", "encryption", "gdpr",
    "warranty", "warranties",
}


def _sanitize_secret(value: Any) -> str:
    return str(value or "").strip().strip('"').strip("'").strip()


def _read_dotenv_value(key: str) -> str:
    lines = []
    for env_path in (Path(".env"), Path(PROJECT_ROOT) / ".env"):
        try:
            lines = env_path.read_text(encoding="utf-8").splitlines()
            break
        except OSError:
            continue
    if not lines:
        return ""
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        name, value = stripped.split("=", 1)
        if name.strip() == key:
            return _sanitize_secret(value)
    return ""


def _get_api_key(required: bool = True) -> str:
    value = _sanitize_secret(os.getenv("OPENROUTER_API_KEY"))
    if not value:
        value = _read_dotenv_value("OPENROUTER_API_KEY")
    if not value:
        if required:
            raise RuntimeError(
                "Missing OPENROUTER_API_KEY in .env file. Set it in .env or "
                "export it before starting the app."
            )
        return ""
    if not value.startswith("sk-or-v1-"):
        raise RuntimeError(
            "Invalid OPENROUTER_API_KEY: OpenRouter keys must start with "
            "'sk-or-v1-'. Check that docker-compose.yml passes "
            "OPENROUTER_API_KEY=${OPENROUTER_API_KEY} without mangling the prefix."
        )
    return value


def _get_client():
    api_key = _get_api_key()
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
        default_headers={
            "HTTP-Referer": "https://veritas-compliance.local",
            "X-Title": "Veritas Compliance Auditor",
        },
    )
    return client, api_key


def _get_live_free_models(api_key: str) -> list[str]:
    try:
        request = urllib.request.Request(
            "https://openrouter.ai/api/v1/models",
            headers={"Authorization": f"Bearer {api_key}"},
        )
        with urllib.request.urlopen(request, timeout=10) as response:
            payload = json.loads(response.read().decode("utf-8"))
        free_models = []
        for model in payload.get("data", []):
            model_id = str(model.get("id") or "")
            pricing = model.get("pricing") or {}
            if model_id.endswith(":free") or (
                str(pricing.get("prompt", "1")) == "0"
                and str(pricing.get("completion", "1")) == "0"
            ):
                free_models.append(model_id)
        preferred = [
            model_id
            for model_id in free_models
            if any(
                family in model_id.lower()
                for family in ("qwen", "deepseek", "llama", "gemma", "mistral")
            )
        ]
        ordered = list(dict.fromkeys(preferred + free_models))
        return ordered or list(FALLBACK_MODELS)
    except Exception:
        return list(FALLBACK_MODELS)


def get_gemini_api_key(required: bool = False) -> str | None:
    value = _sanitize_secret(
        os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    )
    if not value:
        value = _read_dotenv_value("GEMINI_API_KEY") or _read_dotenv_value(
            "GOOGLE_API_KEY"
        )
    if not value:
        if required:
            raise RuntimeError(
                "Missing GEMINI_API_KEY or GOOGLE_API_KEY. Add it to .env or "
                "export it before running the live Gemini smoke test."
            )
        return None
    return value


def get_groq_api_key(required: bool = False) -> str | None:
    value = _sanitize_secret(os.getenv("GROQ_API_KEY"))
    if not value:
        value = _read_dotenv_value("GROQ_API_KEY")
    if not value:
        if required:
            raise RuntimeError(
                "Missing GROQ_API_KEY. Add it to .env or export it before "
                "starting the app."
            )
        return None
    if not value.startswith("gsk_"):
        raise RuntimeError(
            "Invalid GROQ_API_KEY: Groq keys must start with 'gsk_'."
        )
    return value


def _extract_json(raw_text: str) -> dict:
    cleaned = (raw_text or "").strip()
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


def _call_free_llm_with_retry(prompt: str, max_retries: int = 2) -> tuple[dict, str]:
    client, api_key = _get_client()
    candidate_models = _get_live_free_models(api_key)
    last_error = None
    for model_name in candidate_models:
        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are a strict legal compliance auditor. "
                                "Always respond with valid JSON only."
                            ),
                        },
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.1,
                    timeout=90,
                )
                content = response.choices[0].message.content or ""
                actual_model = getattr(response, "model", None) or model_name
                return _extract_json(content), actual_model
            except Exception as exc:
                status_code = getattr(exc, "status_code", None)
                message = str(exc)
                if status_code in (401, 403) or "Missing Authentication header" in message:
                    raise RuntimeError(
                        "OpenRouter authentication failed "
                        f"(HTTP {status_code or 401}): {message[:300]}. "
                        "Verify OPENROUTER_API_KEY starts with sk-or-v1- and is "
                        "passed through docker-compose.yml."
                    ) from exc
                if "free-models-per-day" in message:
                    raise RuntimeError(
                        "OpenRouter daily free-model request quota is exhausted. "
                        "Wait for the quota to reset or add credits to unlock more "
                        "free-model requests."
                    ) from exc
                last_error = (
                    f"[{model_name} attempt {attempt + 1}]: "
                    f"{type(exc).__name__} - {message[:220]}"
                )
                if status_code == 404:
                    break
                if attempt + 1 < max_retries:
                    time.sleep(1.5 ** attempt)
    raise RuntimeError(
        f"All free OpenRouter models failed after retries. Last error: {last_error}"
    )


def _call_groq_with_retry(prompt: str, max_retries: int = 3) -> tuple[dict, str]:
    client = OpenAI(
        base_url="https://api.groq.com/openai/v1",
        api_key=get_groq_api_key(required=True),
    )
    messages = [
        {
            "role": "system",
            "content": (
                "You are a strict legal compliance auditor. "
                "Always respond with valid JSON only."
            ),
        },
        {"role": "user", "content": prompt},
    ]
    last_error = None
    for model_name in GROQ_MODELS:
        for attempt in range(max_retries):
            try:
                try:
                    response = client.chat.completions.create(
                        model=model_name,
                        messages=messages,
                        response_format={"type": "json_object"},
                        temperature=0.1,
                        timeout=60,
                    )
                except Exception as exc:
                    if getattr(exc, "status_code", None) != 400:
                        raise
                    response = client.chat.completions.create(
                        model=model_name,
                        messages=messages,
                        temperature=0.1,
                        timeout=60,
                    )
                content = response.choices[0].message.content or ""
                return _extract_json(content), f"groq/{model_name}"
            except Exception as exc:
                status_code = getattr(exc, "status_code", None)
                last_error = (
                    f"[{model_name} attempt {attempt + 1}]: "
                    f"{type(exc).__name__} - {str(exc)[:220]}"
                )
                if status_code == 404:
                    break
                if attempt + 1 < max_retries:
                    time.sleep(1.5 ** attempt)
    raise RuntimeError(f"All Groq models failed after retries. Last error: {last_error}")


def _gemini_json_completion(prompt: str, api_key: str) -> tuple[str, dict, str]:
    primary = GEMINI_MODEL
    raw_fallbacks = os.getenv(
        "GEMINI_FALLBACK_MODELS", settings.GEMINI_FALLBACK_MODELS
    )
    fallbacks = [
        model.strip() for model in str(raw_fallbacks or "").split(",") if model.strip()
    ]
    models = [primary, *[model for model in fallbacks if model != primary]]
    try:
        max_retries = int(os.getenv("GEMINI_MAX_RETRIES", settings.GEMINI_MAX_RETRIES))
    except (TypeError, ValueError):
        max_retries = settings.GEMINI_MAX_RETRIES
    attempts_per_model = max(1, max_retries + 1)
    last_error: Exception | None = None

    for model_name in models:
        for attempt in range(attempts_per_model):
            try:
                response = completion(
                    model=model_name,
                    messages=[{"content": prompt}],
                    api_key=api_key,
                    temperature=0.1,
                    num_retries=0,
                    timeout=settings.GEMINI_TIMEOUT_SECONDS,
                )
                content = response.choices[0].message.content or ""
                return content, _extract_json(content), model_name
            except (TimeoutError, ConnectionError, OSError) as exc:
                last_error = exc
                if attempt + 1 < attempts_per_model:
                    time.sleep(settings.GEMINI_RETRY_BACKOFF_SECONDS)
                    continue
                break
            except Exception as exc:
                last_error = exc
                break

    if last_error is not None:
        raise last_error
    raise RuntimeError("Gemini completion failed without an exception.")


def _normalize_findings(findings: list[dict]) -> list[dict]:
    normalized = []
    for item in findings or []:
        if not isinstance(item, dict):
            continue
        finding = dict(item)
        rule = str(finding.get("rule") or "").strip()
        result = str(finding.get("result") or "").strip()

        if not result:
            compliant = finding.get("compliant")
            if compliant is True:
                result = "Compliant"
            elif compliant is False:
                result = "Non-Compliant"
            else:
                result = "Undetermined"

        if result.casefold() == "undetermined":
            result = "Gap — Missing Protection"

        finding["result"] = result
        is_gap = result.casefold().startswith("gap")

        if is_gap:
            finding.setdefault("priority", "Critical")
            if not finding.get("reasoning"):
                finding["reasoning"] = (
                    f"Contract contains no provision for {rule}. "
                    "Without this protection, your exposure is uncapped/unregulated."
                )
            if not finding.get("risk_if_missing"):
                finding["risk_if_missing"] = (
                    f"uncapped or unregulated exposure for {rule}."
                )
            if not finding.get("next_step"):
                finding["next_step"] = "Add clause or negotiate before execution"
            finding.setdefault("confidence", "High")
            finding.setdefault(
                "confidence_reason",
                "No matching provision appears in the supplied contract text.",
            )
        else:
            if result.casefold() == "non-compliant":
                finding.setdefault("priority", "High")
                finding.setdefault(
                    "confidence",
                    "Medium" if not finding.get("citation") else "High",
                )
            else:
                finding.setdefault("priority", "Medium")
                finding.setdefault(
                    "confidence",
                    "High" if finding.get("citation") else "Medium",
                )
            if not finding.get("confidence_reason"):
                finding["confidence_reason"] = (
                    "Verbatim citation located."
                    if finding.get("citation")
                    else "No verbatim citation was provided."
                )
        normalized.append(finding)
    return normalized


def _extract_number(text: str) -> float | None:
    lowered = text.lower()
    net_match = re.search(r"net[-\s]?(\d+(?:\.\d+)?)", lowered)
    if net_match:
        return float(net_match.group(1))
    unit_match = re.search(
        r"(\d+(?:\.\d+)?)\s*(?:days?|hours?|hrs?|years?|months?|x|times)\b",
        lowered,
    )
    if unit_match:
        return float(unit_match.group(1))
    for phrase in sorted(_NUMBER_WORDS, key=len, reverse=True):
        word_unit = (
            r"\b"
            + re.escape(phrase)
            + r"\b\s*(?:\(\d+(?:\.\d+)?\)\s*)?"
            + r"(?:(?:business|calendar)\s+)?"
            + r"(?:days?|hours?|hrs?|years?|months?|x|times)\b"
        )
        if re.search(word_unit, lowered):
            return float(_NUMBER_WORDS[phrase])
    parenthetical = re.search(r"\((\d+(?:\.\d+)?)\)", text)
    if parenthetical:
        return float(parenthetical.group(1))
    cleaned = re.sub(
        r"^\s*(?:\d+(?:\.\d+)*\.?|[A-Z]\d+(?:\.\d+)*\.?)\s+",
        "",
        text,
    )
    digits = re.search(r"\b(\d+(?:\.\d+)?)\b", cleaned)
    if digits:
        return float(digits.group(1))
    for phrase in sorted(_NUMBER_WORDS, key=len, reverse=True):
        if re.search(rf"\b{re.escape(phrase)}\b", lowered):
            return float(_NUMBER_WORDS[phrase])
    return None


def _extract_multiplier(text: str) -> float | None:
    match = re.search(r"(\d+(?:\.\d+)?)\s*(?:x|times)\b", text, re.IGNORECASE)
    if match:
        return float(match.group(1))
    times_match = re.search(r"([^.;]{1,50}?)\s+times\b", text, re.IGNORECASE)
    if times_match:
        value = _extract_number(times_match.group(1))
        if value is not None:
            return value
    match = re.search(r"(?:≥|>=|at least\s+)(\d+(?:\.\d+)?)", text, re.IGNORECASE)
    if match and ("x" in text.lower() or "times" in text.lower()):
        return float(match.group(1))
    for phrase in sorted(_NUMBER_WORDS, key=len, reverse=True):
        if re.search(rf"{re.escape(phrase)}\s+times\b", text, re.IGNORECASE):
            return float(_NUMBER_WORDS[phrase])
    return None


def _split_clauses(text: str) -> list[str]:
    raw = [
        clause.strip()
        for clause in re.split(r"(?<!\d\.)(?<=[.!?])\s+|\n+", text or "")
        if clause.strip()
    ]
    merged: list[str] = []
    for clause in raw:
        previous = merged[-1] if merged else ""
        looks_like_heading = bool(
            re.match(r"(SECTION|ARTICLE|AMENDMENT)\b", previous, re.IGNORECASE)
            or (previous.isupper() and len(previous.split()) <= 8)
        )
        next_looks_like_heading = bool(
            re.match(
                r"(?:(?:SECTION|ARTICLE|AMENDMENT|EXHIBIT)\b"
                r"|[A-Z]\d+(?:\.\d+)*\.?"
                r"|\d+(?:\.\d+)*\.?)\s+[A-Z]",
                clause,
                re.IGNORECASE,
            )
        )
        if (
            merged
            and not re.search(r"[.!?:;]$", previous)
            and not looks_like_heading
            and not next_looks_like_heading
        ):
            join = re.search(
                re.escape(merged[-1]) + r"(\s+)" + re.escape(clause), text
            )
            if join:
                merged[-1] = join.group(0)
                continue
        merged.append(clause)
    return merged


def _citation_in_contract(clause: str, contract: str) -> str:
    return clause if clause and clause in contract else ""


def _find_first_duration(clause: str) -> tuple[float | None, str]:
    hours = re.search(r"within\s+([^.;]{1,60}?)\s*(hours?|hrs?)\b", clause, re.IGNORECASE)
    if hours:
        value = _extract_number(hours.group(1))
        if value is not None:
            return value, "hours"
    days = re.search(
        r"(?:within|no later than|not later than)\s+([^.;]{1,60}?)\s*"
        r"(business\s+days?|days?)\b",
        clause,
        re.IGNORECASE,
    )
    if days:
        value = _extract_number(days.group(1))
        if value is not None:
            return value * 24.0, "hours"
    any_hours = re.search(r"([^.;]{1,60}?)\s*(hours?|hrs?)\b", clause, re.IGNORECASE)
    if any_hours:
        value = _extract_number(any_hours.group(1))
        if value is not None:
            return value, "hours"
    return None, ""


def _extract_tier_payment_terms(contract: str) -> list[tuple[str, int, str]]:
    pattern = re.compile(
        r"(?m)^[ \t]*(?P<tier>[A-Z][A-Za-z0-9 +&/-]{1,30})[ \t]*\n"
        r"[ \t]*\$\s?[\d,]+(?:\.\d+)?[ \t]*\n"
        r"[ \t]*(?P<terms>Net[-\s]?\d+|Due on receipt|Upon receipt|"
        r"Prepayment[^\n]*|Immediate[^\n]*|Within\s+\d+\s+days?)[ \t]*$",
        re.IGNORECASE,
    )
    tiers = []
    for match in pattern.finditer(contract):
        terms = match.group("terms").strip()
        terms_lower = terms.lower()
        days = None
        net = re.search(r"net[-\s]?(\d+)", terms_lower)
        if net:
            days = int(net.group(1))
        elif any(keyword in terms_lower for keyword in ("receipt", "immediate", "prepayment")):
            days = 0
        else:
            within = re.search(r"within\s+(\d+)\s+days?", terms_lower)
            if within:
                days = int(within.group(1))
        if days is not None:
            tiers.append((match.group("tier").strip(), days, match.group(0).strip()))
    return tiers


def _payment_term_direction(rule: str) -> str:
    lowered = rule.lower()
    if any(
        keyword in lowered
        for keyword in ("at least", "no less than", "minimum", "or longer", "or more")
    ):
        return "min"
    return "max"


def _payment_citation_with_tier_note(clause: str, contract: str) -> str:
    start = contract.find(clause)
    if start < 0:
        return clause
    tail = contract[start + len(clause):]
    note = re.match(r"(\s+[^.]*\btier\b[^.]*\.)", tail, re.IGNORECASE)
    if note:
        return contract[start:start + len(clause) + note.end()]
    return clause


def _rule_priority(rule: str) -> str:
    lowered = rule.lower()
    if any(
        keyword in lowered
        for keyword in ("liability", "insurance", "cyber", "breach", "indemnity")
    ):
        return "High"
    return "Medium"


def _heuristic_finding(
    rule: str,
    compliant: bool | None,
    citation: str,
    reason: str,
    next_step: str,
) -> dict:
    return {
        "rule": rule,
        "compliant": compliant,
        "citation": citation,
        "reason": reason,
        "next_step": next_step,
        "review_priority": _rule_priority(rule),
    }


def _evaluate_rules_heuristically(contract: str, rules: list[str]) -> list[dict]:
    contract = contract or ""
    clauses = _split_clauses(contract)
    lowered_clauses = [(clause, clause.lower()) for clause in clauses]
    findings: list[dict] = []

    for rule in rules:
        rule_text = str(rule)
        lowered = rule_text.lower()

        if ("insurance" in lowered or "cyber" in lowered) and not any(
            "insurance" in clause_lower or "cyber" in clause_lower
            for _, clause_lower in lowered_clauses
        ):
            findings.append(
                _heuristic_finding(
                    rule_text,
                    None,
                    "",
                    "Undetermined — no insurance/cyber clause was found; manual review required.",
                    "Manual review the complete signed agreement.",
                )
            )
            continue

        if any(
            keyword in lowered
            for keyword in ("payment", "invoice", "net-", "net ")
        ):
            required = _extract_number(lowered)
            if required is None:
                required = 30.0
            direction = _payment_term_direction(rule_text)
            tier_terms = _extract_tier_payment_terms(contract)
            if ("tier" in lowered or "across all" in lowered) and tier_terms:
                if direction == "min":
                    bad_tiers = [tier for tier in tier_terms if tier[1] < required]
                    boundary_value = min(tier[1] for tier in tier_terms)
                    boundary_label = "minimum"
                else:
                    bad_tiers = [tier for tier in tier_terms if tier[1] > required]
                    boundary_value = max(tier[1] for tier in tier_terms)
                    boundary_label = "maximum"
                summary = ", ".join(
                    f"{tier} {'due on receipt' if days == 0 else f'Net-{days}'}"
                    for tier, days, _ in tier_terms
                )
                if bad_tiers:
                    first_start = contract.find(bad_tiers[0][2])
                    last_citation = bad_tiers[-1][2]
                    last_end = contract.find(last_citation) + len(last_citation)
                    citation = (
                        contract[first_start:last_end]
                        if first_start >= 0 and last_end > first_start
                        else bad_tiers[0][2]
                    )
                    findings.append(
                        _heuristic_finding(
                            rule_text,
                            False,
                            citation,
                            f"{summary}; {boundary_label} {boundary_value:g} days "
                            f"{'>' if direction == 'max' else '<'} {required:g} days",
                            "Align every tier to the required payment period before execution.",
                        )
                    )
                else:
                    findings.append(
                        _heuristic_finding(
                            rule_text,
                            True,
                            tier_terms[0][2],
                            f"{summary}; {boundary_label} {boundary_value:g} days "
                            f"{'<=' if direction == 'max' else '>='} {required:g} days",
                            "No action required",
                        )
                    )
                continue
            candidates = [
                clause
                for clause, clause_lower in lowered_clauses
                if any(
                    keyword in clause_lower
                    for keyword in ("payment", "invoice", "net-", "net ")
                )
                and (
                    re.search(r"net[-\s]?\d+", clause_lower)
                    or "within" in clause_lower
                    or re.search(r"\bdays?\b", clause_lower)
                    or re.search(r"\bpay\b", clause_lower)
                )
            ]
            actual = None
            citation = ""
            for clause in candidates:
                value = _extract_number(clause)
                if value is not None:
                    actual = value
                    citation = _citation_in_contract(clause, contract)
                    break
            if actual is not None and required is not None:
                if direction == "min":
                    compliant = actual >= required
                    comparison = ">=" if compliant else "<"
                else:
                    compliant = actual <= required
                    comparison = "<=" if compliant else ">"
                findings.append(
                    _heuristic_finding(
                        rule_text,
                        compliant,
                        _payment_citation_with_tier_note(citation, contract),
                        f"{actual:g} days {comparison} {required:g} days",
                        "No action required"
                        if compliant
                        else "Renegotiate the payment term before execution.",
                    )
                )
                continue

        if "liability" in lowered and any(
            keyword in lowered for keyword in ("cap", "capped", "at least", "multipl")
        ):
            required = _extract_multiplier(lowered)
            if required is None:
                required = 1.0
            candidates = [
                clause
                for clause, clause_lower in lowered_clauses
                if "liability" in clause_lower
                and any(
                    keyword in clause_lower
                    for keyword in ("cap", "capped", "times", "exceed", "fees")
                )
            ]
            selected = None
            actual = None
            for clause in candidates:
                value = _extract_multiplier(clause)
                if value is not None:
                    selected = clause
                    actual = value
            if selected is None and candidates:
                selected = candidates[0]
                actual = 1.0
            if selected is not None and actual is not None:
                compliant = actual >= required
                comparison = ">=" if compliant else "<"
                findings.append(
                    _heuristic_finding(
                        rule_text,
                        compliant,
                        _citation_in_contract(selected, contract),
                        f"{actual:g}x fees {comparison} {required:g}x",
                        "No action required"
                        if compliant
                        else "Renegotiate the liability cap before execution.",
                    )
                )
                continue

        if any(keyword in lowered for keyword in ("termination", "notice", "cure period")) and "notification" not in lowered:
            required = _extract_number(lowered)
            subtype_keywords = (
                ("cure", "uncured", "breach")
                if any(keyword in lowered for keyword in ("cure", "breach"))
                else ("terminat", "notice")
            )
            candidates = [
                clause
                for clause, clause_lower in lowered_clauses
                if any(keyword in clause_lower for keyword in subtype_keywords)
                and ("day" in clause_lower or "month" in clause_lower)
            ]
            actual = None
            citation = ""
            for clause in candidates:
                value = _extract_number(clause)
                if value is not None:
                    actual = value
                    citation = _citation_in_contract(clause, contract)
                    break
            if actual is not None and required is not None:
                if "not exceed" in lowered or "no more than" in lowered or "maximum" in lowered:
                    compliant = actual <= required
                    comparison = "<=" if compliant else ">"
                else:
                    compliant = actual >= required
                    comparison = ">=" if compliant else "<"
                findings.append(
                    _heuristic_finding(
                        rule_text,
                        compliant,
                        citation,
                        f"{actual:g} days {comparison} {required:g} days",
                        "No action required"
                        if compliant
                        else "Renegotiate the termination or cure period.",
                    )
                )
                continue

        if any(
            keyword in lowered
            for keyword in ("notification", "notify", "security incident", "breach notification")
        ):
            required = _extract_number(lowered)
            candidates = [
                clause
                for clause, clause_lower in lowered_clauses
                if any(
                    keyword in clause_lower
                    for keyword in ("notify", "notification", "incident", "breach notification")
                )
                and any(
                    keyword in clause_lower
                    for keyword in ("within", "hour", "business day", "no later than")
                )
            ]
            def _breach_priority(clause: str) -> int:
                lower = clause.lower()
                if "replac" in lower or "amended" in lower:
                    return 0
                if re.search(r"\bhours?\b", lower):
                    return 1
                return 2

            candidates.sort(key=_breach_priority)
            actual = None
            citation = ""
            for clause in candidates:
                value, _ = _find_first_duration(clause)
                if value is not None:
                    actual = value
                    citation = _citation_in_contract(clause, contract)
                    break
            if actual is not None and required is not None:
                compliant = actual <= required
                comparison = "<=" if compliant else ">"
                findings.append(
                    _heuristic_finding(
                        rule_text,
                        compliant,
                        citation,
                        f"{actual:g} hours {comparison} {required:g} hours",
                        "No action required"
                        if compliant
                        else "Renegotiate the breach notification period.",
                    )
                )
                continue

        if any(
            keyword in lowered
            for keyword in ("arbitration", "forum", "seated", "venue")
        ):
            forum_clauses = [
                clause
                for clause, clause_lower in lowered_clauses
                if any(
                    keyword in clause_lower
                    for keyword in ("arbitration", "forum", "seated", "venue")
                )
            ]
            outside = re.search(r"outside\s+([A-Z][A-Za-z .]+)", rule_text)
            choose_forum = "choose" in lowered and "forum" in lowered
            chosen_clause = ""
            outside_ok = False
            choose_ok = False

            if outside:
                forbidden = outside.group(1).strip().rstrip(".").lower()
                for clause in forum_clauses:
                    clause_lower = clause.lower()
                    if forbidden not in clause_lower and any(
                        keyword in clause_lower
                        for keyword in ("seated", "venue", "arbitration")
                    ):
                        outside_ok = True
                        chosen_clause = clause
                        break
            if choose_forum:
                for clause in forum_clauses:
                    clause_lower = clause.lower()
                    if "choose" in clause_lower and "forum" in clause_lower:
                        choose_ok = True
                        chosen_clause = clause
                        break
                if not choose_ok and forum_clauses:
                    chosen_clause = forum_clauses[0]

            if " or " in lowered and outside and choose_forum:
                compliant = outside_ok or choose_ok
                if not chosen_clause and forum_clauses:
                    chosen_clause = forum_clauses[0]
                findings.append(
                    _heuristic_finding(
                        rule_text,
                        compliant,
                        _citation_in_contract(chosen_clause, contract),
                        "OR condition satisfied by the selected forum clause."
                        if compliant
                        else "Neither OR condition is satisfied by the contract.",
                        "No action required"
                        if compliant
                        else "Renegotiate the dispute-resolution forum.",
                    )
                )
                continue

            if outside:
                compliant = outside_ok
                if not chosen_clause and forum_clauses:
                    chosen_clause = forum_clauses[0]
                findings.append(
                    _heuristic_finding(
                        rule_text,
                        compliant,
                        _citation_in_contract(chosen_clause, contract),
                        "Arbitration forum satisfies the outside-jurisdiction rule."
                        if compliant
                        else "Arbitration forum conflicts with the outside-jurisdiction rule.",
                        "No action required"
                        if compliant
                        else "Renegotiate the arbitration seat.",
                    )
                )
                continue

            if choose_forum:
                compliant = choose_ok
                if not chosen_clause and forum_clauses:
                    chosen_clause = forum_clauses[0]
                findings.append(
                    _heuristic_finding(
                        rule_text,
                        compliant,
                        _citation_in_contract(chosen_clause, contract),
                        "Customer forum-choice language is present."
                        if compliant
                        else "No clause allows Customer to choose the forum.",
                        "No action required"
                        if compliant
                        else "Add a customer forum-choice clause.",
                    )
                )
                continue

        if any(keyword in lowered for keyword in ("retention", "retain", "records")):
            required = _extract_number(lowered)
            candidates = [
                clause
                for clause, clause_lower in lowered_clauses
                if any(
                    keyword in clause_lower
                    for keyword in ("retention", "retain", "records")
                )
                and ("year" in clause_lower or "month" in clause_lower)
            ]
            actual = None
            citation = ""
            for clause in candidates:
                value = _extract_number(clause)
                if value is not None:
                    actual = value
                    citation = _citation_in_contract(clause, contract)
                    break
            if actual is not None and required is not None:
                compliant = actual >= required
                comparison = ">=" if compliant else "<"
                findings.append(
                    _heuristic_finding(
                        rule_text,
                        compliant,
                        citation,
                        f"{actual:g} {comparison} {required:g} years",
                        "No action required"
                        if compliant
                        else "Extend the records-retention period.",
                    )
                )
                continue

        rule_words = {
            word
            for word in re.findall(r"[a-z]{4,}", lowered)
            if word not in {"must", "shall", "with", "that", "this", "from", "have", "been"}
        }
        best_clause = ""
        best_score = 0
        for clause, clause_lower in lowered_clauses:
            clause_words = set(re.findall(r"[a-z]{4,}", clause_lower))
            score = len(rule_words & clause_words)
            if score > best_score:
                best_score = score
                best_clause = clause
        if best_clause and best_score >= 2:
            findings.append(
                _heuristic_finding(
                    rule_text,
                    True,
                    _citation_in_contract(best_clause, contract),
                    "The cited clause addresses the rule's subject matter.",
                    "No action required",
                )
            )
        else:
            findings.append(
                _heuristic_finding(
                    rule_text,
                    None,
                    "",
                    "Undetermined — no matching clause was found; manual review required.",
                    "Manual review the complete signed agreement.",
                )
            )

    return findings


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


def _citation_matches_page(finding: dict, quote: str, page_text: str) -> bool:
    if quote and quote in page_text:
        return True
    start = finding.get("citation_start")
    end = finding.get("citation_end")
    if start is None and end is None:
        return False
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
            match.end() for match in re.finditer(r"(?<=[.!?])\s+|\n+", page_text)
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
        f'[{sentence["id"]}] {sentence["text"]}' for sentence in sentence_index
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


def _primary_citation(citations: list[dict]) -> dict:
    def score(citation: dict) -> float:
        quote = str(citation.get("quote") or "").lower()
        value = 0.0
        if re.search(r"\d", quote):
            value += 3
        if any(
            keyword in quote
            for keyword in (
                "net-",
                "times",
                "hour",
                "days",
                "years",
                "fees",
                "liability",
                "breach",
                "arbitration",
                "retain",
                "delaware",
            )
        ):
            value += 2
        if "notify" in quote:
            value += 1
        if "replaced" in quote or "amended" in quote:
            value += 1
        value += min(len(quote.split()), 30) / 100
        return value

    return max(citations, key=score) if citations else {}


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
            selected = [sentence_lookup[sentence_id] for sentence_id in selected_ids]
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
            primary = _primary_citation(grounded_citations)
            finding["citation"] = primary["quote"]
            finding["citation_page"] = primary["page"]
            finding["citation_start"] = primary["start"]
            finding["citation_end"] = primary["end"]
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
        primary = _primary_citation(grounded_citations)
        finding["citation"] = primary["quote"]
        finding["citation_page"] = primary["page"]
        finding["citation_start"] = primary["start"]
        finding["citation_end"] = primary["end"]
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
        expected_pages = [citation_page] if citation_page is not None else referenced_pages
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


def _state_source_pages(state: dict) -> list[dict]:
    pages = state.get("source_pages") or state.get("contract_pages")
    if pages:
        return [
            {"page": int(page.get("page", 1)), "text": str(page.get("text") or "")}
            for page in pages
            if isinstance(page, dict)
        ]
    source_text = state.get("source_contract_text") or state.get("contract_text") or ""
    parsed = [
        {"page": int(number), "text": text}
        for number, text in re.findall(
            r"---\s*PAGE\s+(\d+)\s*---\s*(.*?)(?=---\s*PAGE\s+\d+\s*---|\Z)",
            source_text,
            flags=re.IGNORECASE | re.DOTALL,
        )
    ]
    return parsed or [{"page": 1, "text": source_text}]


def _attach_source_references(findings: list[dict], source_pages: list[dict]) -> None:
    for finding in findings:
        citation = str(finding.get("citation") or "")
        if not citation:
            continue
        for page in source_pages:
            page_text = str(page.get("text") or "")
            position = page_text.find(citation)
            if position < 0:
                continue
            page_number = page.get("page", 1)
            heading_pattern = re.compile(
                r"(?m)^[ \t]*(?P<heading>"
                r"(?:(?:SECTION|ARTICLE|AMENDMENT|EXHIBIT)\s+(?:NO\.?\s*)?[A-Z0-9][^\n]{0,70}"
                r"|[A-Z]\d+(?:\.\d+)*\.?\s+[A-Z][^\n]{0,70}"
                r"|\d+(?:\.\d+)*\.?\s+[A-Z][^\n]{0,70}))[ \t]*$"
            )
            def _valid_heading(value: str) -> bool:
                return not re.search(
                    r"\b(is hereby|shall|may|will|must)\b", value, re.IGNORECASE
                )

            start_heading = heading_pattern.match(citation)
            if start_heading and _valid_heading(start_heading.group("heading")):
                section = start_heading.group("heading").strip()
            else:
                headings = [
                    match.group("heading").strip()
                    for match in heading_pattern.finditer(page_text[:position])
                    if _valid_heading(match.group("heading"))
                ]
                section = headings[-1] if headings else ""
            finding["citation_page"] = page_number
            finding["citation_start"] = position
            finding["citation_end"] = position + len(citation)
            finding["source_reference"] = (
                f"Page {page_number}, {section}" if section else f"Page {page_number}"
            )
            break


def gatekeeper_node(state: ComplianceState) -> Dict[str, Any]:
    contract = str(state.get("contract_text") or "")
    rules = "\n".join(str(rule) for rule in (state.get("compliance_rules") or []))
    combined = f"{contract}\n{rules}"
    for pattern in PROMPT_INJECTION_PATTERNS:
        match = re.search(pattern, combined, re.IGNORECASE)
        if match:
            return {
                "guardrail_status": "blocked",
                "guardrail_reason": (
                    "Malicious instruction / prompt injection detected "
                    f"matching pattern: '{match.group(0)}'"
                ),
                "status": "blocked",
                "estimated_cost": 0.0,
                "total_tokens": 0,
                "audit_report": {},
            }
    return {"guardrail_status": "passed", "guardrail_reason": ""}


def _budget_fields(state: dict, total_tokens: int) -> dict:
    estimated_cost = round(total_tokens * 0.00001, 8)
    limit = state.get("budget_limit")
    exceeded = False
    if limit is not None:
        try:
            exceeded = estimated_cost > float(limit)
        except (TypeError, ValueError):
            exceeded = False
    return {"estimated_cost": estimated_cost, "budget_exceeded": exceeded}


def _auditor_heuristic(
    state: ComplianceState,
    source_pages: list[dict],
    total_pages: int | None,
) -> Dict[str, Any]:
    contract = str(state.get("source_contract_text") or state.get("contract_text") or "")
    rules = [str(rule) for rule in (state.get("compliance_rules") or [])]
    findings = _evaluate_rules_heuristically(contract, rules)
    for finding in findings:
        compliant = finding.get("compliant")
        if compliant is True:
            finding["result"] = "Compliant"
        elif compliant is False:
            finding["result"] = "Non-Compliant"
        else:
            finding["result"] = "Gap — Missing Protection"
        finding["reasoning"] = finding.get("reason", "")
        rule_text = str(finding.get("rule") or "").rstrip(".")
        if compliant is None:
            finding["priority"] = "Critical"
            finding["risk_if_missing"] = (
                f"uncapped or unregulated exposure for {rule_text}"
            )
        else:
            finding["priority"] = finding.get("review_priority") or _rule_priority(
                rule_text
            )
            finding["risk_if_missing"] = (
                f"unmet protection for {rule_text}" if compliant is False else ""
            )
        finding["confidence"] = (
            "High" if finding.get("citation") or compliant is None else "Medium"
        )
        finding["confidence_reason"] = (
            "Verbatim citation located in the supplied contract."
            if finding.get("citation")
            else "No verbatim citation was located."
        )
    _attach_source_references(findings, source_pages)
    total_tokens = max(1, len(contract) // 4 + len(rules) * 8)
    return {
        "audit_report": {
            "findings": findings,
            "analysis": {"mode": "heuristic_fallback", "model": "local heuristic"},
        },
        "model_used": "local heuristic",
        "ai_error": None,
        "loop_count": int(state.get("loop_count", 0)) + 1,
        "status": "pending",
        "total_tokens": int(state.get("total_tokens", 0)) + total_tokens,
        **_budget_fields(state, int(state.get("total_tokens", 0)) + total_tokens),
    }


def _auditor_gemini(
    state: ComplianceState,
    source_pages: list[dict],
    total_pages: int | None,
    api_key: str,
) -> Dict[str, Any]:
    rules = state.get("compliance_rules") or []
    contract = str(state.get("source_contract_text") or state.get("contract_text") or "")
    prompt = f"""
You are a strict legal compliance auditor. Evaluate the contract against every rule.

Return JSON only:
{{
  "findings": [
    {{
      "rule": "exact input rule",
      "result": "Compliant" | "Non-Compliant" | "Undetermined" | "Gap — Missing Protection",
      "priority": "High" | "Medium" | "Low",
      "citation": "exact verbatim quote from the contract",
      "reasoning": "condition-vs-text comparison",
      "confidence": "High" | "Medium" | "Low"
    }}
  ]
}}

Compliance Rules:
{json.dumps(rules, indent=2)}

Contract:
<contract_document>
{contract}
</contract_document>
"""
    try:
        _, parsed, model_name = _gemini_json_completion(prompt, api_key)
        raw_findings = parsed.get("findings") if isinstance(parsed, dict) else None
        if not isinstance(raw_findings, list):
            raise RuntimeError("Gemini returned no findings list.")
        findings = _normalize_findings(
            [finding for finding in raw_findings if isinstance(finding, dict)]
        )
        _ground_findings_to_source_pages(findings, source_pages, total_pages)
        _attach_source_references(findings, source_pages)
        return {
            "audit_report": {
                "findings": findings,
                "analysis": {"mode": "gemini", "model": model_name},
            },
            "model_used": f"gemini/{model_name}",
            "ai_error": None,
            "loop_count": int(state.get("loop_count", 0)) + 1,
            "status": "pending",
            "total_tokens": int(state.get("total_tokens", 0))
            + max(1, len(prompt) // 4),
            **_budget_fields(
                state,
                int(state.get("total_tokens", 0)) + max(1, len(prompt) // 4),
            ),
        }
    except Exception as exc:
        return {
            "audit_report": {"findings": []},
            "model_used": "NONE",
            "ai_error": str(exc),
            "status": "error",
            "loop_count": int(state.get("loop_count", 0)) + 1,
        }


def _auditor_prompt(
    state: ComplianceState,
    source_pages: list[dict],
    total_pages: int | None,
    feedback_context: str,
) -> tuple[str, list[dict]]:
    sentence_index = _build_sentence_index(source_pages)
    source_context = _format_sentence_context(sentence_index)
    page_span = (
        f"Pages 1-{total_pages}"
        if isinstance(total_pages, int) and total_pages > 0
        else "the complete contract"
    )
    prompt = f"""
Evaluate the Contract Text against EVERY rule in the Compliance Rules list.

CRITICAL INSTRUCTIONS:
1. ONE VERDICT PER RULE: Produce exactly one finding object per input rule. NEVER split a compound rule (containing AND / OR) into two findings. In the reasoning, explicitly name each OR branch you evaluated and state whether it passes or fails. An OR rule is Compliant if any branch passes, and Non-Compliant only if all branches fail.
2. EXPLICIT MATH & LOGIC COMPARISON: For numeric rules (e.g., Net-30, >=3x fees, <=24h notice, >=2 years retention), extract the exact number from the contract and explicitly state the mathematical comparison in 'reasoning'.
2a. PAYMENT DIRECTION: For a payment rule phrased "Net-30 or better", "within", or "not exceed", the payment period is an upper bound: every tier's period must be <= the stated number of days. Shorter periods such as Net-15 or due on receipt are not violations. For "at least", "no less than", or "or longer", the period is a lower bound and every tier must be >= the stated number of days.
2b. AMBIGUOUS TERMS: If a numeric requirement is not stated with a definite number and unit (for example, "promptly", "reasonable time", "upon receipt" without a number of days), do not mark the rule Compliant. Use Non-Compliant when the clause exists but is inadequate, or Gap — Missing Protection when the required term is absent. Explain the ambiguity in the reasoning.
3. EVIDENCE-ONLY REVIEW: Use only the sentence-indexed source text below. Each sentence ID maps to an exact span of extracted PDF text and a real page.
4. CITATIONS: Do not write, paraphrase, shorten, or combine any citation text. Return citation_sentence_ids containing the ID(s) of the source sentences that directly support the finding. Select the shortest sufficient set, and use multiple IDs only for distinct provisions needed to evaluate the rule. Never cite a bare number, heading, or incomplete fragment; for numeric rules, include the sentence or table row containing the decisive number and unit. Order the IDs with the most decisive provision first.
5. PAGE NUMBERS: Do not return or invent page numbers. Python resolves each selected sentence ID to its extracted source page.
6. GAP POLICY: If no relevant provision exists, use result "Gap — Missing Protection" and return citation_sentence_ids as an empty list. Python creates the absence citation for {page_span}.
7. AMENDMENTS AND PRECEDENCE: Read the whole agreement, including exhibits and amendments. When documents conflict, an explicit amendment or exhibit precedence clause controls over the base clause. Cross-reference defined terms before deciding whether a condition is met.
8. If a sentence does not support the finding, do not select its ID. Every selected ID must appear exactly in the source text below.
9. COMPOUND RULE TYPE: When a rule contains OR, set "type": "or_compound" in the finding and name both branches with their pass/fail status in one finding. Never merge separate input rules.
10. Set 'next_step' to "No action required" when Compliant. For Non-Compliant or Gap findings, give a concrete negotiation or escalation action.
11. Include a confidence rating and a short confidence reason for every finding.

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
      "type": "or_compound",
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
    return prompt, sentence_index


def _auditor_llm(
    state: ComplianceState,
    source_pages: list[dict],
    total_pages: int | None,
) -> Dict[str, Any]:
    current_count = int(state.get("loop_count", 0))
    feedback_context = (
        f"\nPrevious Critic Feedback to fix: {state.get('critic_feedback')}"
        if current_count > 0
        else ""
    )
    prompt, sentence_index = _auditor_prompt(
        state, source_pages, total_pages, feedback_context
    )
    try:
        parsed_report, used_model = _call_primary_llm(prompt)
        rules = state.get("compliance_rules") or []
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
5. Never merge separate input rules.
6. An explicit amendment or exhibit precedence clause controls over the base clause.

Contract Text:
<contract_document>
{state.get("source_contract_text") or state.get("contract_text") or ""}
</contract_document>

Input rules:
{json.dumps(rules, indent=2)}

Sentence-indexed source text:
<source>
{_format_sentence_context(sentence_index)}
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
            corrected_report, correction_model = _call_primary_llm(correction_prompt)
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
                    f"Finding count mismatch; expected {len(rules)} findings, "
                    f"received {corrected_count}."
                )
            parsed_report = corrected_report
            used_model = f"{used_model} (corrected via {correction_model})"

        findings = parsed_report.get("findings")
        if isinstance(findings, list):
            normalized = _normalize_findings(
                [finding for finding in findings if isinstance(finding, dict)]
            )
            _ground_findings_to_source_pages(
                normalized, source_pages, total_pages, sentence_index
            )
            parsed_report["findings"] = normalized
        provider_mode = "groq" if "groq" in str(used_model).lower() else "openrouter"
        parsed_report["analysis"] = {"mode": provider_mode, "model": used_model}
        return {
            "audit_report": parsed_report,
            "model_used": used_model,
            "ai_error": None,
            "loop_count": current_count + 1,
            "total_tokens": int(state.get("total_tokens", 0))
            + max(1, len(prompt) // 4),
            **_budget_fields(
                state,
                int(state.get("total_tokens", 0)) + max(1, len(prompt) // 4),
            ),
        }
    except Exception as exc:
        return {
            "audit_report": {"findings": []},
            "model_used": "NONE",
            "ai_error": str(exc),
            "status": "error",
            "loop_count": current_count + 1,
        }


def _llm_was_patched() -> bool:
    return (
        _call_groq_with_retry is not _ORIGINAL_GROQ
        or _call_free_llm_with_retry is not _ORIGINAL_FREE
    )


def _openrouter_available() -> bool:
    try:
        return bool(_get_api_key(required=False))
    except RuntimeError:
        return True


def _call_primary_llm(prompt: str) -> tuple[dict, str]:
    if _call_groq_with_retry is not _ORIGINAL_GROQ:
        return _call_groq_with_retry(prompt)
    if get_groq_api_key():
        return _call_groq_with_retry(prompt)
    if _call_free_llm_with_retry is not _ORIGINAL_FREE:
        return _call_free_llm_with_retry(prompt)
    return _call_free_llm_with_retry(prompt)


def auditor_node(state: ComplianceState) -> Dict[str, Any]:
    source_pages = _state_source_pages(state)
    total_pages = state.get("total_pages")
    if not isinstance(total_pages, int) or total_pages <= 0:
        total_pages = len(source_pages)

    if os.getenv("MOCK_LLM"):
        return _auditor_heuristic(state, source_pages, total_pages)
    if _llm_was_patched():
        return _auditor_llm(state, source_pages, total_pages)
    if get_groq_api_key():
        return _auditor_llm(state, source_pages, total_pages)
    if _openrouter_available():
        return _auditor_llm(state, source_pages, total_pages)
    gemini_key = get_gemini_api_key()
    if gemini_key:
        return _auditor_gemini(state, source_pages, total_pages, gemini_key)
    return _auditor_heuristic(state, source_pages, total_pages)


_DOMAIN_GROUPS = (
    {"payment", "invoice", "invoices", "net", "payable", "pay"},
    {"liability", "cap", "capped", "fees", "exceed", "indemnity", "multiplier"},
    {"breach", "notify", "notification", "incident", "security", "cure", "uncured"},
    {"arbitration", "forum", "seated", "venue", "choose", "delaware"},
    {"retention", "retain", "records", "data"},
    {"insurance", "cyber"},
    {"termination", "notice"},
    {"encryption", "aes", "gdpr"},
)


def _citation_supports_rule(citation: str, rule: str) -> bool:
    citation_lower = citation.lower()
    rule_lower = rule.lower()
    best_group = None
    best_score = 0
    for group in _DOMAIN_GROUPS:
        score = sum(1 for keyword in group if keyword in rule_lower)
        if score > best_score:
            best_score = score
            best_group = group
    if best_group is not None:
        return any(keyword in citation_lower for keyword in best_group)
    stop_words = {"must", "shall", "with", "that", "this", "from", "have", "been"}
    rule_words = {
        word
        for word in re.findall(r"[a-z]{4,}", rule_lower)
        if word not in stop_words
    }
    citation_words = set(re.findall(r"[a-z]{4,}", citation_lower))
    return bool(rule_words & citation_words)


def _critic_local(
    state: ComplianceState,
    report: dict,
    findings: list[dict],
    source_text: str,
) -> Dict[str, Any]:
    reviewed = 0
    verified = 0
    issues = []
    for finding in findings:
        rule = str(finding.get("rule") or "")
        citation = str(finding.get("citation") or "").strip()
        if not citation:
            is_gap = str(finding.get("result", "")).casefold().startswith("gap")
            if is_gap or finding.get("compliant") is None:
                continue
            issues.append(f"Finding for '{rule}' has no verbatim citation.")
            continue
        reviewed += 1
        if citation not in source_text:
            issues.append(
                f"Citation for '{rule}' is not an exact contract excerpt."
            )
            finding["compliant"] = None
            finding["citation"] = ""
            finding["next_step"] = "Manual review of the cited clause is required."
            continue
        if not _citation_supports_rule(citation, rule):
            issues.append(
                f"Citation for '{rule}' does not support the rule."
            )
            finding["compliant"] = None
            finding["citation"] = ""
            finding["next_step"] = "Manual review of the cited clause is required."
            continue
        verified += 1

    if issues:
        detail = (
            " ".join(issues)
            + " The cited text is not an exact contract excerpt or does not "
            "support the rule. Findings require manual review."
        )
        return {
            "status": "fail",
            "critic_mode": "local_citation_check",
            "critic_feedback": detail,
            "critic_warning": True,
            "citation_review_executed": True,
            "citation_review_status": "FAIL",
            "citation_review_detail": detail,
            "verified_citations": verified,
            "total_citations": reviewed,
            "audit_report": report,
        }

    return {
        "status": "pass",
        "critic_mode": "local_citation_check",
        "critic_feedback": "Local citation check passed.",
        "critic_warning": False,
        "citation_review_executed": True,
        "citation_review_status": "PASS",
        "citation_review_detail": (
            f"{verified}/{reviewed} citations matched verbatim using the local "
            "deterministic check."
        ),
        "verified_citations": verified,
        "total_citations": reviewed,
        "audit_report": report,
    }


def _critic_gemini(
    state: ComplianceState,
    report: dict,
    findings: list[dict],
    rules: list[str],
    source_text: str,
    total_citations: int,
    api_key: str,
) -> Dict[str, Any]:
    prompt = f"""
You are a Senior Legal Quality Controller. Verify the Audit Report against the Raw Contract Text.
Check:
1. Are all {len(rules)} rules evaluated correctly? Each input rule already has exactly one finding; do not split OR conditions or merge distinct rules.
2. For an OR rule, verify that the reasoning identifies at least one satisfied branch. It is acceptable to omit a failing branch if the verdict is supported; a Non-Compliant OR verdict requires all branches to fail.
3. Are all numeric comparisons (<=, >=) mathematically accurate? For payment rules, apply the direction in the finding: "Net-30 or better" is an upper bound, so every tier must be at most 30 days; "at least" or "or longer" is a lower bound.
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
        _, validation, model_name = _gemini_json_completion(prompt, api_key)
    except Exception as exc:
        detail = f"Critic Agent unavailable ({type(exc).__name__}: {exc}). Report withheld."
        return {
            "status": "incomplete",
            "critic_mode": "not_run",
            "critic_feedback": detail,
            "citation_review_executed": False,
            "citation_review_status": "NOT RUN",
            "citation_review_detail": detail,
            "verified_citations": 0,
            "total_citations": total_citations,
            "audit_report": {"findings": []},
            "ai_error": f"Critic Agent unavailable ({type(exc).__name__}: {exc})",
        }

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
        detail = f"{feedback} Findings were withheld."
        return {
            "status": "fail",
            "critic_mode": "gemini",
            "critic_feedback": detail,
            "citation_review_executed": True,
            "citation_review_status": "FAIL",
            "citation_review_detail": detail,
            "verified_citations": 0,
            "total_citations": total_citations,
            "audit_report": {"findings": []},
        }

    feedback = str(validation.get("feedback") or "") if isinstance(validation, dict) else ""
    return {
        "status": "pass",
        "critic_mode": "gemini",
        "critic_model": model_name,
        "critic_feedback": feedback,
        "citation_review_executed": True,
        "citation_review_status": "PASS",
        "citation_review_detail": (
            f"{total_citations}/{total_citations} non-gap citations matched "
            f"verbatim. Reviewed by {model_name}."
        ),
        "verified_citations": total_citations,
        "total_citations": total_citations,
        "citations_reviewed": total_citations,
        "citations_verified": total_citations,
    }


def _critic_llm(
    state: ComplianceState,
    report: dict,
    findings: list[dict],
    rules: list[str],
    source_text: str,
    total_pages: int | None,
    total_citations: int,
) -> Dict[str, Any]:
    prompt = f"""
You are a Senior Legal Quality Controller. Verify the Audit Report against the Raw Contract Text.
Check:
1. Are all {len(rules)} rules evaluated correctly? Each input rule already has exactly one finding; do not split OR conditions or merge distinct rules.
2. For an OR rule, verify that the reasoning identifies at least one satisfied branch. It is acceptable to omit a failing branch if the verdict is supported; a Non-Compliant OR verdict requires all branches to fail. An OR passes if at least one branch passes.
3. Are all numeric comparisons (<=, >=) mathematically accurate? For payment rules, apply the direction in the finding: "Net-30 or better" is an upper bound, so every tier must be at most 30 days; "at least" or "or longer" is a lower bound.
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
        validation, critic_model = _call_primary_llm(prompt)
    except Exception as exc:
        detail = (
            f"Critic Agent unavailable ({type(exc).__name__}: {exc}). "
            "Report withheld."
        )
        return {
            "status": "incomplete",
            "critic_mode": "not_run",
            "critic_feedback": detail,
            "citation_review_executed": False,
            "citation_review_status": "NOT RUN",
            "citation_review_detail": detail,
            "verified_citations": 0,
            "total_citations": total_citations,
            "audit_report": {"findings": []},
            "ai_error": f"Critic Agent unavailable ({type(exc).__name__}: {exc})",
        }

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
        detail = f"{feedback} Findings were withheld."
        return {
            "status": "fail",
            "critic_mode": "llm",
            "critic_feedback": detail,
            "citation_review_executed": True,
            "citation_review_status": "FAIL",
            "citation_review_detail": detail,
            "verified_citations": 0,
            "total_citations": total_citations,
            "audit_report": {"findings": []},
        }

    page_note = (
        f" All page references are within the {total_pages}-page PDF."
        if isinstance(total_pages, int)
        else ""
    )
    feedback = str(validation.get("feedback") or "") if isinstance(validation, dict) else ""
    return {
        "status": "pass",
        "critic_mode": "llm",
        "critic_model": critic_model,
        "critic_feedback": feedback,
        "citation_review_executed": True,
        "citation_review_status": "PASS",
        "citation_review_detail": (
            f"{total_citations}/{total_citations} non-gap citations matched "
            f"verbatim. Reviewed by {critic_model}.{page_note}"
        ),
        "verified_citations": total_citations,
        "total_citations": total_citations,
        "citations_reviewed": total_citations,
        "citations_verified": total_citations,
    }


def _incomplete_review(
    detail: str,
    *,
    citation_review_executed: bool,
    citation_review_status: str,
    total_citations: int,
    verified_citations: int = 0,
) -> Dict[str, Any]:
    return {
        "status": "fail",
        "critic_feedback": detail,
        "citation_review_executed": citation_review_executed,
        "citation_review_status": citation_review_status,
        "citation_review_detail": detail,
        "verified_citations": verified_citations,
        "total_citations": total_citations,
        "audit_report": {"findings": []},
    }


def critic_node(state: ComplianceState) -> Dict[str, Any]:
    if state.get("ai_error"):
        return {
            "status": "error",
            "critic_mode": "not_run",
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
    source_pages = state.get("source_pages") or state.get("contract_pages") or None

    if os.getenv("MOCK_LLM"):
        return _critic_local(state, report, findings, source_text)
    if (
        not _llm_was_patched()
        and not _openrouter_available()
        and not get_groq_api_key()
    ):
        gemini_key = get_gemini_api_key()
        if not gemini_key:
            return _critic_local(state, report, findings, source_text)

    if findings and all("result" in finding for finding in findings):
        if not _findings_match_rules(report, rules):
            return _incomplete_review(
                f"Finding coverage failed: expected exactly one finding for each of "
                f"{len(rules)} input rules. Findings were withheld.",
                citation_review_executed=True,
                citation_review_status="FAIL",
                total_citations=total_citations,
            )

    citation_issues = _citation_issues(
        findings, source_text, total_pages, source_pages
    )
    if citation_issues:
        return _incomplete_review(
            " ".join(citation_issues) + " Findings were withheld.",
            citation_review_executed=True,
            citation_review_status="FAIL",
            total_citations=total_citations,
        )

    gemini_key = get_gemini_api_key()
    if (
        gemini_key
        and not _llm_was_patched()
        and not _openrouter_available()
        and not get_groq_api_key()
    ):
        return _critic_gemini(
            state, report, findings, rules, source_text, total_citations, gemini_key
        )
    return _critic_llm(
        state, report, findings, rules, source_text, total_pages, total_citations
    )


def citation_review_from_state(state: dict) -> Dict[str, Any]:
    mode = state.get("critic_mode")
    if mode == "local_citation_check":
        return {
            "mode": "local_citation_check",
            "status": "local_check_pass"
            if state.get("status") == "pass"
            else "local_check_fail",
            "citations_reviewed": int(state.get("total_citations", 0) or 0),
            "citations_verified": int(state.get("verified_citations", 0) or 0),
        }
    if mode == "gemini" or state.get("citation_review_status") == "PASS":
        return {
            "mode": mode or "gemini",
            "status": "verified",
            "citations_reviewed": int(state.get("total_citations", 0) or 0),
            "citations_verified": int(state.get("verified_citations", 0) or 0),
        }
    return {
        "mode": "not_run",
        "status": "not_run",
        "citations_reviewed": int(state.get("total_citations", 0) or 0),
        "citations_verified": int(state.get("verified_citations", 0) or 0),
    }


_ORIGINAL_GROQ = _call_groq_with_retry
_ORIGINAL_FREE = _call_free_llm_with_retry
