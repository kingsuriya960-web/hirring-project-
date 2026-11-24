import html
from pathlib import Path

from reportlab.lib.pagesizes import letter, landscape
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

SEVERITY_ORDER = {"critical": 0, "high": 1, "medium": 2, "low": 3}


def summarize_findings(findings: list[dict]) -> dict:
    """Builds the executive summary shared by the PDF and the Streamlit UI."""
    items = [item for item in (findings or []) if isinstance(item, dict)]
    compliant = sum(
        1 for item in items if str(item.get("result", "")).lower() == "compliant"
    )
    non_compliant = sum(
        1 for item in items if str(item.get("result", "")).lower() == "non-compliant"
    )
    gap = sum(
        1 for item in items if str(item.get("result", "")).lower().startswith("gap")
    )
    critical = sum(
        1 for item in items if str(item.get("priority", "")).lower() == "critical"
    )
    high = sum(1 for item in items if str(item.get("priority", "")).lower() == "high")
    medium = sum(
        1 for item in items if str(item.get("priority", "")).lower() == "medium"
    )
    low = sum(1 for item in items if str(item.get("priority", "")).lower() == "low")

    if critical or non_compliant:
        recommendation = "DO NOT EXECUTE without legal review"
    elif high or gap:
        recommendation = "PROCEED WITH CAUTION"
    else:
        recommendation = "APPROVED"

    problem_items = [
        item
        for item in items
        if str(item.get("result", "")).lower() != "compliant"
    ]
    worst = min(
        problem_items,
        key=lambda item: SEVERITY_ORDER.get(
            str(item.get("priority", "low")).lower(), 4
        ),
        default={},
    )
    key_issue = (
        str(
            worst.get("risk_if_missing")
            or worst.get("reasoning")
            or worst.get("rule")
            or "No findings were produced."
        )
        if worst
        else ""
    )
    status_note = (
        ""
        if worst
        else "No blocking issues found; all rules are compliant."
    )
    next_step = (
        str(worst.get("next_step") or "Have qualified counsel review the agreement.")
        if worst
        else "Proceed with standard legal review."
    )
    return {
        "recommendation": recommendation,
        "verdict_summary": (
            f"{compliant} compliant, {non_compliant} non-compliant, "
            f"{gap} missing-protection gaps."
        ),
        "risk_summary": (
            f"{critical} critical, {high} high, {medium} medium, {low} low."
        ),
        "key_issue": key_issue,
        "status_note": status_note,
        "next_step": next_step,
        "counts": {
            "rules": len(items),
            "compliant": compliant,
            "non_compliant": non_compliant,
            "gap": gap,
            "critical": critical,
            "high": high,
            "medium": medium,
            "low": low,
        },
    }


class AuditReportGenerator:
    @staticmethod
    def _blocked_pdf_bytes(filename: str, document_name: str, final_state: dict) -> bytes:
        doc = SimpleDocTemplate(filename, pagesize=landscape(letter))
        styles = getSampleStyleSheet()
        story = [
            Paragraph("VERITAS COMPLIANCE — ENTERPRISE AUDIT REPORT", styles["Heading1"]),
            Paragraph(f"<b>Document Audited:</b> {html.escape(str(document_name))}", styles["BodyText"]),
            Spacer(1, 10),
            Paragraph(
                "<b>ANALYSIS STATUS: BLOCKED</b><br/>"
                "The document was not analyzed because a prompt-injection guardrail "
                "blocked the run. No findings are presented. Manual review is required.",
                styles["BodyText"],
            ),
        ]
        doc.build(story)
        return Path(filename).read_bytes()

    @staticmethod
    def _legacy_pdf_bytes(filename: str, document_name: str, final_state: dict) -> bytes:
        doc = SimpleDocTemplate(
            filename,
            pagesize=landscape(letter),
            rightMargin=30,
            leftMargin=30,
            topMargin=30,
            bottomMargin=30,
        )
        styles = getSampleStyleSheet()
        story = [
            Paragraph("VERITAS COMPLIANCE — ENTERPRISE AUDIT REPORT", styles["Heading1"]),
            Paragraph(f"<b>Document Audited:</b> {html.escape(str(document_name))}", styles["BodyText"]),
        ]
        findings = [
            item
            for item in (final_state.get("findings") or [])
            if isinstance(item, dict)
        ]
        compliant = sum(1 for item in findings if item.get("compliant") is True)
        non_compliant = sum(1 for item in findings if item.get("compliant") is False)
        undetermined = sum(1 for item in findings if item.get("compliant") is None)

        analysis = final_state.get("analysis") or {}
        mode = analysis.get("mode") if isinstance(analysis, dict) else None
        method = (
            "Gemini (multi-agent LangGraph pipeline)"
            if mode == "gemini"
            else str(mode or "Automated multi-agent LangGraph pipeline")
        )
        story.append(Paragraph(f"<b>Analysis method:</b> {html.escape(method)}", styles["BodyText"]))

        citation_review = final_state.get("citation_review") or {}
        if isinstance(citation_review, dict) and citation_review.get("status") == "verified":
            reviewed = int(citation_review.get("citations_reviewed") or 0)
            verified = int(citation_review.get("citations_verified") or 0)
            story.append(
                Paragraph(
                    f"<b>Citation review:</b> {verified}/{reviewed} cited excerpts verified",
                    styles["BodyText"],
                )
            )

        story.append(
            Paragraph(
                f"<b>Rules reviewed:</b> {len(findings)} | "
                f"<b>Compliant:</b> {compliant} | "
                f"<b>Non-compliant:</b> {non_compliant} | "
                f"<b>Undetermined:</b> {undetermined}",
                styles["BodyText"],
            )
        )
        story.append(Spacer(1, 10))

        table_data = [[
            "Compliance rule",
            "Result",
            "Citation",
            "Source",
            "Reasoning",
            "Next step",
        ]]
        for item in findings:
            value = item.get("compliant")
            result = (
                "Compliant"
                if value is True
                else "Non-compliant"
                if value is False
                else "Undetermined"
            )
            table_data.append([
                Paragraph(html.escape(str(item.get("rule") or "")), styles["BodyText"]),
                Paragraph(result, styles["BodyText"]),
                Paragraph(html.escape(str(item.get("citation") or "")), styles["BodyText"]),
                Paragraph(html.escape(str(item.get("source_reference") or "")), styles["BodyText"]),
                Paragraph(html.escape(str(item.get("reason") or "")), styles["BodyText"]),
                Paragraph(html.escape(str(item.get("next_step") or "")), styles["BodyText"]),
            ])
        table = Table(table_data, colWidths=[130, 80, 170, 100, 170, 130], repeatRows=1)
        table.setStyle(
            TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1E293B")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
            ])
        )
        story.append(table)
        doc.build(story)
        return Path(filename).read_bytes()

    @staticmethod
    def generate_pdf_report(
        filename: str,
        document_name: str | None = None,
        final_state: dict | None = None,
    ) -> bytes:
        if final_state is None and isinstance(document_name, dict):
            final_state = document_name
            document_name = Path(filename).name
        final_state = final_state or {}
        document_name = document_name or Path(filename).name

        analysis = final_state.get("analysis")
        if isinstance(analysis, dict) and analysis.get("mode") == "heuristic_fallback":
            raise ValueError(
                "A compliance report cannot be generated from heuristic fallback findings."
            )
        if final_state.get("guardrail_status") == "blocked":
            return AuditReportGenerator._blocked_pdf_bytes(
                filename, document_name, final_state
            )
        if "audit_report" not in final_state and any(
            key in final_state
            for key in ("findings", "analysis", "citation_review", "compliance_status")
        ):
            return AuditReportGenerator._legacy_pdf_bytes(
                filename, document_name, final_state
            )

        doc = SimpleDocTemplate(
            filename,
            pagesize=landscape(letter),
            rightMargin=30,
            leftMargin=30,
            topMargin=30,
            bottomMargin=30,
        )
        story = []
        styles = getSampleStyleSheet()

        title_style = ParagraphStyle(
            "DocTitle",
            parent=styles["Heading1"],
            fontSize=18,
            leading=22,
            textColor=colors.HexColor("#0F172A"),
            spaceAfter=8,
        )
        meta_style = ParagraphStyle(
            "DocMeta",
            parent=styles["BodyText"],
            fontSize=9.5,
            leading=13,
            textColor=colors.HexColor("#334155"),
            spaceAfter=4,
        )
        body_style = ParagraphStyle(
            "DocBody",
            parent=styles["BodyText"],
            fontSize=8.5,
            leading=11.5,
            textColor=colors.HexColor("#1E293B"),
        )
        error_style = ParagraphStyle(
            "ErrorBanner",
            parent=styles["BodyText"],
            fontSize=11,
            leading=15,
            textColor=colors.HexColor("#991B1B"),
            backColor=colors.HexColor("#FEE2E2"),
            borderPadding=12,
        )

        safe_doc_name = html.escape(str(document_name))
        story.append(Paragraph("VERITAS COMPLIANCE — ENTERPRISE AUDIT REPORT", title_style))
        story.append(Paragraph(f"<b>Document Audited:</b> {safe_doc_name}", meta_style))
        contract_type = final_state.get("contract_type")
        if contract_type:
            story.append(
                Paragraph(
                    f"<b>Contract type:</b> {html.escape(str(contract_type))}",
                    meta_style,
                )
            )
        rule_alignment_warning = final_state.get("rule_alignment_warning")
        if rule_alignment_warning:
            story.append(
                Paragraph(
                    f"<b>RULE-SET ALIGNMENT WARNING:</b> "
                    f"{html.escape(str(rule_alignment_warning))}",
                    meta_style,
                )
            )
        story.append(Spacer(1, 6))

        ai_error = final_state.get("ai_error")
        citation_review_executed = bool(
            final_state.get("citation_review_executed", False)
        )
        citation_review_status = str(
            final_state.get("citation_review_status") or "NOT RUN"
        ).upper()
        citation_review_passed = (
            citation_review_executed
            and citation_review_status == "PASS"
            and final_state.get("status") == "pass"
        )
        try:
            total_citations = int(final_state.get("total_citations") or 0)
        except (TypeError, ValueError):
            total_citations = 0
        try:
            verified_citations = int(final_state.get("verified_citations") or 0)
        except (TypeError, ValueError):
            verified_citations = 0

        if ai_error:
            status_label = (
                "Critic Agent unavailable"
                if "Critic Agent unavailable" in str(ai_error)
                else "Primary AI analysis unavailable"
            )
            banner_text = (
                f"<b>ANALYSIS STATUS: INCOMPLETE — {status_label}</b><br/>"
                f"<b>VERDICT:</b> Manual review required. "
                f"{html.escape(str(ai_error))[:800]}<br/>"
                "DO NOT rely on this output for execution decisions. No findings "
                "are presented because automated verification did not complete."
            )
            story.append(Paragraph(banner_text, error_style))
            doc.build(story)
            return Path(filename).read_bytes()

        if not citation_review_passed:
            citation_error = (
                "Citation verification failed. Automated audit could not be completed."
                if citation_review_executed
                else "Citation verification was not completed. Automated audit could not be completed."
            )
            citation_detail = str(
                final_state.get("citation_review_detail") or citation_error
            )
            story.append(
                Paragraph(
                    "<b>ANALYSIS STATUS: INCOMPLETE</b><br/>"
                    "<b>RECOMMENDATION: MANUAL REVIEW REQUIRED</b><br/>"
                    f"{html.escape(citation_error)}<br/>"
                    f"{html.escape(citation_detail)[:800]}<br/>"
                    "No findings are presented because citations or rule coverage "
                    "could not be verified.",
                    error_style,
                )
            )
            doc.build(story)
            return Path(filename).read_bytes()

        audit_report = final_state.get("audit_report") or {}
        if not isinstance(audit_report, dict):
            audit_report = {}
        raw_findings = audit_report.get("findings") or []
        if not isinstance(raw_findings, list):
            raw_findings = []
        findings = [item for item in raw_findings if isinstance(item, dict)]
        if not findings:
            banner_text = (
                "<b>ANALYSIS STATUS: INCOMPLETE — no findings were produced</b><br/>"
                "<b>VERDICT:</b> Manual review required. The auditor returned no "
                "rule-by-rule findings, so no compliance conclusion can be drawn."
            )
            story.append(Paragraph(banner_text, error_style))
            doc.build(story)
            return Path(filename).read_bytes()

        model_used_raw = str(final_state.get("model_used") or "").strip()
        report_analysis = (
            (final_state.get("audit_report") or {}).get("analysis")
            if isinstance(final_state.get("audit_report"), dict)
            else {}
        )
        analysis_mode = str(
            report_analysis.get("mode")
            if isinstance(report_analysis, dict)
            else ""
        ).lower()
        model_lower = model_used_raw.lower()
        analysis_is_ai = bool(model_used_raw) and not any(
            marker in model_lower
            for marker in ("local heuristic", "heuristic", "fallback", "none")
        ) and analysis_mode != "heuristic_fallback"
        model_used = html.escape(model_used_raw or "Unknown")
        citation_status = (
            "PASS"
            if citation_review_passed
            else ("NOT RUN" if not citation_review_executed else "FAIL")
        )
        citation_detail = str(
            final_state.get("citation_review_detail")
            or (
                f"{verified_citations}/{total_citations} citations verified."
                if total_citations
                else "No citation-backed findings to verify."
            )
        )
        compliant_cnt = sum(
            1 for f in findings if str(f.get("result", "")).lower() == "compliant"
        )
        gap_cnt = sum(
            1
            for f in findings
            if str(f.get("result", "")).lower().startswith("gap")
        )
        non_compliant_cnt = sum(
            1 for f in findings if str(f.get("result", "")).lower() == "non-compliant"
        )
        total_cnt = len(findings)

        critical_cnt = sum(
            1 for f in findings if str(f.get("priority", "")).lower() == "critical"
        )
        high_cnt = sum(
            1 for f in findings if str(f.get("priority", "")).lower() == "high"
        )
        medium_cnt = sum(
            1 for f in findings if str(f.get("priority", "")).lower() == "medium"
        )
        low_cnt = sum(
            1 for f in findings if str(f.get("priority", "")).lower() == "low"
        )
        if critical_cnt or non_compliant_cnt:
            recommendation = "DO NOT EXECUTE without legal review"
        elif high_cnt or gap_cnt:
            recommendation = "PROCEED WITH CAUTION"
        else:
            recommendation = "APPROVED"
        if not citation_review_passed:
            recommendation = (
                "MANUAL REVIEW REQUIRED — automated verification is incomplete"
            )
        if not analysis_is_ai:
            recommendation = (
                "MANUAL REVIEW REQUIRED — local fallback only, not AI analysis"
            )

        severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        problem_findings = [
            finding
            for finding in findings
            if str(finding.get("result", "")).lower() != "compliant"
        ]
        worst_finding = min(
            problem_findings,
            key=lambda f: severity_order.get(
                str(f.get("priority", "low")).lower(), 4
            ),
            default={},
        )
        if worst_finding:
            key_issue = str(
                worst_finding.get("risk_if_missing")
                or worst_finding.get("reasoning")
                or worst_finding.get("rule")
                or "No findings were produced."
            )
            next_step = str(
                worst_finding.get("next_step")
                or "Refer findings to qualified counsel."
            )
            issue_label = "KEY ISSUE"
        else:
            key_issue = "No blocking issues found; all rules are compliant."
            next_step = "Proceed with standard legal review."
            issue_label = "STATUS NOTE"
        if not analysis_is_ai:
            story.append(
                Paragraph(
                    "<b>AI ANALYSIS UNAVAILABLE — this report was NOT generated "
                    "by the compliance AI.</b><br/>"
                    "The results below come from the deterministic keyword/rule "
                    "fallback. They are not an AI legal review, have not been "
                    "endorsed by an AI model, and must be manually verified.",
                    error_style,
                )
            )
            story.append(Spacer(1, 8))
        summary_text = (
            f"<b>RECOMMENDATION:</b> {html.escape(recommendation)}.<br/>"
            f"<b>VERDICT SUMMARY:</b> {compliant_cnt} compliant, "
            f"{non_compliant_cnt} non-compliant, {gap_cnt} missing-protection gaps.<br/>"
            f"<b>RISK SEVERITY:</b> {critical_cnt} critical, {high_cnt} high, "
            f"{medium_cnt} medium, {low_cnt} low.<br/>"
            f"<b>{issue_label}:</b> {html.escape(key_issue)}<br/>"
            f"<b>NEXT STEP:</b> {html.escape(next_step)}"
        )
        story.append(Paragraph(summary_text, meta_style))
        story.append(Spacer(1, 8))
        if not analysis_is_ai:
            analysis_badge = "FALLBACK — NOT AI-VERIFIED"
        elif citation_review_passed:
            analysis_badge = "AI-VERIFIED"
        else:
            analysis_badge = "AI-UNVERIFIED"
        story.append(
            Paragraph(
                f"<b>Analysis method:</b> {model_used} "
                f"(multi-agent LangGraph pipeline) [{analysis_badge}]",
                meta_style,
            )
        )
        story.append(
            Paragraph(
                f"<b>Citation review:</b> {citation_status} — "
                f"{html.escape(citation_detail)}",
                meta_style,
            )
        )
        if citation_review_passed and total_citations:
            if analysis_is_ai:
                citation_note = "<b>All citations exactly match contract text.</b>"
            else:
                citation_note = (
                    "<b>Local citation check passed.</b> This is a deterministic "
                    "text match, not an AI verification of the analysis."
                )
            story.append(Paragraph(citation_note, meta_style))
        story.append(
            Paragraph(
                f"<b>Rules reviewed:</b> {total_cnt} | <b>Compliant:</b> {compliant_cnt} | "
                f"<b>Non-compliant:</b> {non_compliant_cnt} | "
                f"<b>Gap — Missing Protection:</b> {gap_cnt}",
                meta_style,
            )
        )
        story.append(Spacer(1, 12))

        result_header = (
            "Result (FALLBACK)"
            if not analysis_is_ai
            else ("Result (UNVERIFIED)" if not citation_review_passed else "Result")
        )
        table_data = [[
            "Compliance rule",
            result_header,
            "Priority",
            "Risk if missing",
            "Confidence (why)",
            "Citation",
            "Reasoning",
            "Next step",
        ]]

        for item in findings:
            res_val = str(item.get("result", "Gap — Missing Protection"))
            is_gap = res_val.lower().startswith("gap")
            color_hex = (
                "#15803D"
                if res_val.lower() == "compliant"
                else (
                    "#B91C1C"
                    if res_val.lower() == "non-compliant" or is_gap
                    else "#B45309"
                )
            )
            confidence = html.escape(str(item.get("confidence") or "Low"))
            confidence_reason = html.escape(
                str(item.get("confidence_reason") or "Evidence basis not provided.")
            )
            result_markup = (
                f"<font color='{color_hex}'><b>{html.escape(res_val)}</b></font>"
            )
            if not analysis_is_ai:
                result_markup = (
                    f"<font color='#B45309'><b>FALLBACK</b></font><br/>"
                    f"{result_markup}"
                )
            elif not citation_review_passed and res_val.lower() != "compliant":
                result_markup = (
                    f"<font color='#B45309'><b>UNVERIFIED</b></font><br/>"
                    f"{result_markup}"
                )
            citation_sources = item.get("citations")
            if isinstance(citation_sources, list) and citation_sources:
                citation_markup = "<br/><br/>".join(
                    f"<i>{html.escape(str(source.get('quote') or ''))}</i>"
                    f"<br/><b>Page {html.escape(str(source.get('page') or ''))}</b>"
                    for source in citation_sources
                    if isinstance(source, dict)
                )
            else:
                citation_markup = (
                    f"<i>{html.escape(str(item.get('citation', '')))}</i>"
                    + (
                        f"<br/><b>{html.escape(str(item['source_reference']))}</b>"
                        if item.get("source_reference")
                        else ""
                    )
                )
            table_data.append([
                Paragraph(html.escape(str(item.get("rule", ""))), body_style),
                Paragraph(result_markup, body_style),
                Paragraph(html.escape(str(item.get("priority", "Critical" if is_gap else "Medium"))), body_style),
                Paragraph(
                    html.escape(str(item.get("risk_if_missing") or "—"))
                    if is_gap else "—",
                    body_style,
                ),
                Paragraph(f"<b>{confidence}</b><br/>{confidence_reason}", body_style),
                Paragraph(citation_markup, body_style),
                Paragraph(html.escape(str(item.get("reasoning", ""))), body_style),
                Paragraph(html.escape(str(item.get("next_step", ""))), body_style),
            ])

        body_style.fontSize = 7.2
        body_style.leading = 9.2
        col_widths = [88, 68, 45, 112, 78, 118, 125, 91]
        results_table = Table(table_data, colWidths=col_widths, repeatRows=1)
        results_table.setStyle(
            TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1E293B")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, 0), 9),
                ("BOTTOMPADDING", (0, 0), (-1, 0), 8),
                ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#F8FAFC")),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
                ("TOPPADDING", (0, 1), (-1, -1), 7),
                ("BOTTOMPADDING", (0, 1), (-1, -1), 7),
            ])
        )

        story.append(results_table)
        doc.build(story)
        return Path(filename).read_bytes()
