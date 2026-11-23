from pypdf import PdfReader
import io


CONTRACT_TYPES = (
    "Consulting Agreement",
    "SaaS / Subscription Agreement",
    "Vendor / Supply Agreement",
    "NDA",
    "Lease / Real Estate",
)

_CONTRACT_TYPE_SIGNALS = {
    "Consulting Agreement": (
        "consulting agreement",
        "consultant",
        "consulting services",
        "professional services",
    ),
    "SaaS / Subscription Agreement": (
        "software as a service",
        "saas",
        "subscription agreement",
        "subscription services",
        "cloud services",
        "service level agreement",
    ),
    "Vendor / Supply Agreement": (
        "supply agreement",
        "supplier",
        "vendor agreement",
        "purchase order",
        "goods",
    ),
    "NDA": (
        "non-disclosure agreement",
        "nondisclosure agreement",
        "confidentiality agreement",
        "confidential information",
    ),
    "Lease / Real Estate": (
        "lease agreement",
        "landlord",
        "tenant",
        "leased premises",
        "monthly rent",
    ),
}

_RULE_RELEVANCE_HINTS = {
    "Consulting Agreement": (
        (
            "breach notification",
            "security breach",
            "security incident",
            "data retention",
            "records retention",
            "service level",
            "uptime",
            "sla",
        ),
        "Consulting agreements usually address liability caps, indemnity, IP "
        "ownership, confidentiality, payment terms, and termination.",
    ),
    "SaaS / Subscription Agreement": (
        (
            "landlord",
            "tenant",
            "leased premises",
            "monthly rent",
            "bill of sale",
            "delivery of goods",
        ),
        "SaaS agreements usually address uptime/SLA, data protection, breach "
        "notification, data retention, subscription fees, and termination.",
    ),
    "Vendor / Supply Agreement": (
        ("service level", "uptime", "sla", "subscription", "saas"),
        "Vendor/supply agreements usually address delivery, acceptance, "
        "warranties, price adjustments, penalties, and termination.",
    ),
    "NDA": (
        (
            "service level",
            "uptime",
            "sla",
            "invoice",
            "payment terms",
            "net-30",
            "liability cap at",
        ),
        "NDAs usually address confidentiality scope, permitted disclosures, "
        "return or destruction of information, and survival periods.",
    ),
    "Lease / Real Estate": (
        (
            "liability cap",
            "breach notification",
            "data retention",
            "service level",
            "uptime",
            "saas",
        ),
        "Leases usually address rent, security deposits, maintenance, permitted "
        "use, assignment, and renewal or termination.",
    ),
}

_CONTRACT_TYPE_TITLE_SIGNALS = {
    "Consulting Agreement": ("consulting agreement", "consultancy agreement"),
    "SaaS / Subscription Agreement": (
        "saas agreement",
        "subscription agreement",
        "software as a service",
    ),
    "Vendor / Supply Agreement": (
        "supply agreement",
        "vendor agreement",
        "supplier agreement",
    ),
    "NDA": (
        "non-disclosure agreement",
        "nondisclosure agreement",
        "confidentiality agreement",
    ),
    "Lease / Real Estate": (
        "lease agreement",
        "commercial lease",
        "residential lease",
    ),
}


class DocumentParserService:
    @staticmethod
    def classify_contract_type(text: str) -> str | None:
        """Classifies a contract from common title and subject-matter signals."""
        normalized_text = " ".join((text or "").lower().split())
        title_text = normalized_text[:1500]
        scores = {
            contract_type: sum(
                min(normalized_text.count(signal), 2)
                for signal in signals
            ) + 10 * sum(
                title_text.count(signal)
                for signal in _CONTRACT_TYPE_TITLE_SIGNALS[contract_type]
            )
            for contract_type, signals in _CONTRACT_TYPE_SIGNALS.items()
        }
        best_type = max(scores, key=scores.get)
        return best_type if scores[best_type] else None

    @staticmethod
    def mismatched_rules(contract_type: str | None, rules: list[str]) -> list[str]:
        """Returns rules that look specific to a different contract type."""
        hints = _RULE_RELEVANCE_HINTS.get(contract_type or "")
        if not hints:
            return []
        keywords, _ = hints
        return [
            rule
            for rule in rules
            if any(keyword in rule.lower() for keyword in keywords)
        ]

    @staticmethod
    def rule_alignment_warning(
        contract_type: str | None, rules: list[str]
    ) -> str | None:
        """Explains rule-set mismatches so the user can remove them or run anyway."""
        hints = _RULE_RELEVANCE_HINTS.get(contract_type or "")
        if not hints:
            return None
        _, guidance = hints
        mismatched = DocumentParserService.mismatched_rules(contract_type, rules)
        if not mismatched:
            return None
        listed = "; ".join(f"'{rule}'" for rule in mismatched)
        return (
            f"Rule-set alignment warning: this document looks like a {contract_type}, "
            f"but these rules look specific to another contract type: {listed}. "
            f"{guidance} Remove or confirm these rules before relying on the findings."
        )

    @staticmethod
    def extract_pdf_text(file_bytes: bytes) -> str:
        """Returns the full text of the PDF, with a page marker before each page."""
        return DocumentParserService.format_pdf_pages(
            DocumentParserService.extract_pdf_pages(file_bytes)
        )

    @staticmethod
    def extract_pdf_pages(file_bytes: bytes) -> list[dict]:
        reader = PdfReader(io.BytesIO(file_bytes))
        return [
            {
                "page": page_idx + 1,
                "text": (page.extract_text() or "").strip(),
            }
            for page_idx, page in enumerate(reader.pages)
        ]

    @staticmethod
    def format_pdf_pages(pages: list[dict]) -> str:
        pages_text = [
            f"--- PAGE {page['page']} ---\n{page['text']}"
            for page in pages
            if str(page.get("text") or "").strip()
        ]
        return "\n\n".join(pages_text)

    @staticmethod
    def extract_and_chunk_pdf(file_bytes: bytes, child_size: int = 400, parent_size: int = 2000) -> list[dict]:
        reader = PdfReader(io.BytesIO(file_bytes))
        processed_chunks = []
        parent_step = max(parent_size - 400, 1)

        for page_idx, page in enumerate(reader.pages):
            page_text = page.extract_text() or ""
            if not page_text.strip():
                continue

            for i in range(0, len(page_text), parent_step):
                parent_chunk = page_text[i : i + parent_size]
                for j in range(0, len(parent_chunk), child_size):
                    child_chunk = parent_chunk[j : j + child_size]
                    if len(child_chunk.strip()) > 50:
                        processed_chunks.append({
                            "child_text": child_chunk.strip(),
                            "parent_text": parent_chunk.strip(),
                            "page": page_idx + 1,
                        })
        return processed_chunks
