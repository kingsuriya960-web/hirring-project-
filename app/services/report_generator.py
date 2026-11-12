from io import BytesIO
from typing import Any
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


class AuditReportGenerator:
    @staticmethod
    def generate_pdf_report(document_name: str, audit_result: dict[str, Any]) -> bytes:
        buffer = BytesIO()
        document = SimpleDocTemplate(
            buffer,
            pagesize=letter,
            rightMargin=40,
            leftMargin=40,
            topMargin=42,
            bottomMargin=42,
            title="Compliance Audit Report",
            author="Compliance Review Workspace",
        )
        base_styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            "AuditTitle",
            parent=base_styles["Title"],
            alignment=TA_LEFT,
            fontSize=20,
            leading=25,
            textColor=colors.HexColor("#173b32"),
            spaceAfter=12,
        )
        section_style = ParagraphStyle(
            "AuditSection",
            parent=base_styles["Heading2"],
            fontSize=12,
            leading=16,
            textColor=colors.HexColor("#17684d"),
            spaceBefore=14,
            spaceAfter=7,
        )
        body_style = ParagraphStyle(
            "AuditBody",
            parent=base_styles["BodyText"],
            fontSize=9,
            leading=13,
            textColor=colors.HexColor("#29352f"),
            splitLongWords=1,
        )
        small_style = ParagraphStyle(
            "AuditSmall",
            parent=body_style,
            fontSize=8,
            leading=11,
            textColor=colors.HexColor("#5a6860"),
        )
        header_style = ParagraphStyle(
            "AuditHeader",
            parent=body_style,
            fontName="Helvetica-Bold",
            textColor=colors.white,
        )

        findings = audit_result.get("findings", [])
        citation_review = audit_result.get("citation_review", {})
        analysis = audit_result.get("analysis", {})
        status = str(audit_result.get("compliance_status", "undetermined")).replace("_", " ").upper()
        compliant_count = sum(finding.get("compliant") is True for finding in findings)
        non_compliant_count = sum(finding.get("compliant") is False for finding in findings)

        story = [
            Paragraph("COMPLIANCE AUDIT REPORT", title_style),
            Paragraph(f"<b>Document:</b> {escape(document_name)}", body_style),
            Paragraph(f"<b>Audit result:</b> {escape(status)}", body_style),
            Paragraph(f"<b>Rules reviewed:</b> {len(findings)}", body_style),
            Paragraph(
                f"<b>Compliant:</b> {compliant_count} &nbsp;&nbsp; "
                f"<b>Non-compliant:</b> {non_compliant_count}",
                body_style,
            ),
            Paragraph(
                f"<b>Citation review:</b> {escape(str(citation_review.get('status', 'not reviewed')).upper())}"
                f" ({escape(str(citation_review.get('mode', 'unknown')).replace('_', ' '))})",
                body_style,
            ),
        ]

        citation_feedback = citation_review.get("feedback")
        if citation_feedback:
            story.append(Paragraph(f"<b>Citation notes:</b> {escape(str(citation_feedback))}", small_style))
        if citation_review.get("warning"):
            story.append(Paragraph(f"<b>Citation-review warning:</b> {escape(str(citation_review['warning']))}", small_style))

        analysis_mode = analysis.get("mode", "unknown")
        story.append(Paragraph(f"<b>Analysis method:</b> {escape(str(analysis_mode).replace('_', ' '))}", body_style))
        if analysis.get("warning"):
            story.append(Paragraph(f"<b>Important:</b> {escape(str(analysis['warning']))}", small_style))

        if audit_result.get("guardrail_reason"):
            story.append(Paragraph(f"<b>Security review:</b> {escape(str(audit_result['guardrail_reason']))}", small_style))

        story.extend([
            Spacer(1, 10),
            Paragraph("Rule-by-rule findings", section_style),
        ])

        rows = [[
            Paragraph("Compliance rule", header_style),
            Paragraph("Result", header_style),
            Paragraph("Citation and rationale", header_style),
        ]]
        row_statuses = []
        for finding in findings:
            compliant = finding.get("compliant")
            finding_status = "COMPLIANT" if compliant is True else "NON-COMPLIANT" if compliant is False else "UNDETERMINED"
            citation = finding.get("citation") or "No matching clause located."
            reason = finding.get("reason", "No rationale provided.")
            details = (
                f"<i>&quot;{escape(str(citation))}&quot;</i><br/><br/>"
                f"<b>Reason:</b> {escape(str(reason))}"
            )
            rows.append([
                Paragraph(escape(str(finding.get("rule", "Unspecified rule"))), body_style),
                Paragraph(finding_status, body_style),
                Paragraph(details, body_style),
            ])
            row_statuses.append(compliant)

        if not findings:
            rows.append([
                Paragraph("No rules were evaluated.", body_style),
                Paragraph("UNDETERMINED", body_style),
                Paragraph("The audit did not produce rule-level findings.", body_style),
            ])
            row_statuses.append(None)

        table = Table(rows, colWidths=[145, 90, 277], repeatRows=1, splitByRow=1)
        table_commands = [
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#24483e")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("ALIGN", (0, 0), (-1, -1), "LEFT"),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cad5ce")),
            ("LEFTPADDING", (0, 0), (-1, -1), 7),
            ("RIGHTPADDING", (0, 0), (-1, -1), 7),
            ("TOPPADDING", (0, 0), (-1, -1), 8),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ]
        for row_index, compliant in enumerate(row_statuses, start=1):
            background = (
                colors.HexColor("#e8f3ec")
                if compliant is True
                else colors.HexColor("#f9eaea")
                if compliant is False
                else colors.HexColor("#f2f3f1")
            )
            table_commands.append(("BACKGROUND", (0, row_index), (-1, row_index), background))
        table.setStyle(TableStyle(table_commands))
        story.append(table)

        story.extend([
            Spacer(1, 16),
            Paragraph(
                "This report is automated decision support, not legal advice. "
                "Review the cited contract language and findings before making decisions.",
                small_style,
            ),
        ])
        document.build(story)
        return buffer.getvalue()