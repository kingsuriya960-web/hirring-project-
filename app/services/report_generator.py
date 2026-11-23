import html
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

    if critical or non_compliant:
        recommendation = "DO NOT EXECUTE without legal review"
    elif high or gap:
        recommendation = "PROCEED WITH CAUTION"
    else:
        recommendation = "APPROVED"

    worst = min(
        items,
        key=lambda item: SEVERITY_ORDER.get(
            str(item.get("priority", "low")).lower(), 4
        ),
        default={},
    )
    key_issue = str(
        worst.get("risk_if_missing")
        or worst.get("reasoning")
        or worst.get("rule")
        or "No findings were produced."
    )
    next_step = str(
        worst.get("next_step") or "Have qualified counsel review the agreement."
    )
    return {
        "recommendation": recommendation,
        "risk_summary": (
            f"{critical} critical, {high} high, {compliant} compliant; "
            f"{gap} missing-protection gaps and {non_compliant} non-compliant findings."
        ),
        "key_issue": key_issue,
        "next_step": next_step,
        "counts": {
            "rules": len(items),
            "compliant": compliant,
            "non_compliant": non_compliant,
            "gap": gap,
            "critical": critical,
            "high": high,
        },
    }


class AuditReportGenerator:
    @staticmethod
    def generate_pdf_report(filename: str, document_name: str, final_state: dict) -> str:
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
            return filename

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
            return filename

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
            return filename

        model_used = html.escape(str(final_state.get("model_used") or "OpenRouter Free"))
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

        severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        worst_finding = min(
            findings,
            key=lambda f: severity_order.get(
                str(f.get("priority", "low")).lower(), 4
            ),
            default={},
        )
        key_issue = (
            str(
                worst_finding.get("risk_if_missing")
                or worst_finding.get("reasoning")
                or worst_finding.get("rule")
                or "No findings were produced."
            )
            if worst_finding
            else "No findings were produced."
        )
        next_step = (
            str(worst_finding.get("next_step") or "Refer findings to qualified counsel.")
            if worst_finding
            else "Have qualified counsel review the agreement."
        )
        summary_text = (
            f"<b>RECOMMENDATION:</b> {html.escape(recommendation)}.<br/>"
            f"<b>RISK SUMMARY:</b> {critical_cnt} critical, {high_cnt} high, "
            f"{compliant_cnt} compliant; {gap_cnt} missing-protection gaps and "
            f"{non_compliant_cnt} non-compliant findings.<br/>"
            f"<b>KEY ISSUE:</b> {html.escape(key_issue)}<br/>"
            f"<b>NEXT STEP:</b> {html.escape(next_step)}"
        )
        story.append(Paragraph(summary_text, meta_style))
        story.append(Spacer(1, 8))
        story.append(
            Paragraph(
                f"<b>Analysis method:</b> {model_used} (multi-agent LangGraph pipeline) "
                f"[{'VERIFIED' if citation_review_passed else 'UNVERIFIED'}]",
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
            story.append(
                Paragraph(
                    "<b>All citations exactly match contract text.</b>",
                    meta_style,
                )
            )
        story.append(
            Paragraph(
                f"<b>Rules reviewed:</b> {total_cnt} | <b>Compliant:</b> {compliant_cnt} | "
                f"<b>Non-compliant:</b> {non_compliant_cnt} | "
                f"<b>Gap — Missing Protection:</b> {gap_cnt}",
                meta_style,
            )
        )
        story.append(Spacer(1, 12))

        table_data = [[
            "Compliance rule",
            "Result (UNVERIFIED)" if not citation_review_passed else "Result",
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
            if not citation_review_passed and res_val.lower() != "compliant":
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
        return filename
