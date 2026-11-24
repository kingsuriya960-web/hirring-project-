import os
import sys
import time
import uuid
from datetime import datetime

import streamlit as st

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

if os.path.exists(".env"):
    with open(".env") as f:
        for line in f:
            if "=" in line and not line.strip().startswith("#"):
                key, value = line.strip().split("=", 1)
                os.environ.setdefault(key, value.strip("\"'"))

from app.services.parser import CONTRACT_TYPES, DocumentParserService
from app.services.report_generator import AuditReportGenerator, summarize_findings
from app.agents.graph import create_compliance_graph
from app.agents.nodes import auditor_node, critic_node

st.set_page_config(
    page_title="Veritas Compliance — Enterprise Review",
    page_icon="✓",
    layout="wide",
    initial_sidebar_state="collapsed",
)

if "active_tab" not in st.session_state:
    st.session_state["active_tab"] = "Dashboard"
if "uploader_key" not in st.session_state:
    st.session_state["uploader_key"] = 0
if "rules_draft" not in st.session_state:
    st.session_state["rules_draft"] = ""
if "audit_history" not in st.session_state:
    st.session_state["audit_history"] = []
if "selected_kpi" not in st.session_state:
    st.session_state["selected_kpi"] = None

st.markdown(
    """
    <style>
        .stApp { background-color: #F8FAFC; color: #0F172A; }
        #MainMenu, footer, header { visibility: hidden; }
        .veritas-brand {
            display: flex; align-items: center; gap: 10px; font-weight: 800;
            font-size: 19px; color: #0F172A; padding-top: 6px;
        }
        .veritas-check {
            background-color: #2563EB; color: white; padding: 3px 9px;
            border-radius: 7px; font-size: 14px;
        }
        .hero-pill {
            display: inline-block; background-color: #EFF6FF; color: #2563EB;
            font-size: 11px; font-weight: 700; padding: 4px 10px;
            border-radius: 6px; letter-spacing: 0.05em; text-transform: uppercase;
            margin-top: 12px; margin-bottom: 8px;
        }
        .hero-title { font-size: 32px; font-weight: 800; color: #0F172A; margin: 0; line-height: 1.2; }
        .hero-sub { color: #64748B; font-size: 15px; margin-top: 6px; margin-bottom: 20px; }
        div.stButton > button[kind="primary"] {
            background-color: #2563EB !important; color: #FFFFFF !important;
            border: 1px solid #2563EB !important; border-radius: 10px !important;
            padding: 12px 24px !important; font-weight: 700 !important;
            font-size: 15px !important; transition: all 0.2s ease !important;
        }
        div.stButton > button[kind="primary"]:hover {
            background-color: #1D4ED8 !important; border-color: #1D4ED8 !important;
        }
        div.stDownloadButton > button {
            background-color: #2563EB !important; color: #FFFFFF !important;
            border: 2px solid #2563EB !important; border-radius: 10px !important;
            padding: 12px 24px !important; font-weight: 700 !important;
            font-size: 15px !important; transition: all 0.2s ease !important;
        }
        div.stDownloadButton > button:hover {
            background-color: #FFFFFF !important; color: #2563EB !important;
            border: 2px solid #2563EB !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

nav_left, _, nav_dashboard, nav_history, nav_rules = st.columns([3, 2, 1.3, 1.4, 1.3])
with nav_left:
    st.markdown(
        '<div class="veritas-brand"><span class="veritas-check">✓</span>'
        "<span>Veritas Compliance</span></div>",
        unsafe_allow_html=True,
    )
for column, tab_name in (
    (nav_dashboard, "Dashboard"),
    (nav_history, "Audit History"),
    (nav_rules, "Rule Sets"),
):
    with column:
        if st.button(
            tab_name,
            use_container_width=True,
            type="primary" if st.session_state["active_tab"] == tab_name else "secondary",
        ):
            st.session_state["active_tab"] = tab_name
            st.rerun()

st.divider()


@st.cache_resource
def get_compiled_graph():
    return create_compliance_graph(auditor_node, critic_node)


history = st.session_state["audit_history"]
completed_audits = [item for item in history if item["status"] == "Passed"]
failed_audits = [
    item for item in history if item["status"] in {"Failed", "Incomplete"}
]
total_completed = len(completed_audits)
total_rules_checked = sum(item["rules_count"] for item in completed_audits)
avg_time_saved = (
    round((total_rules_checked * 8.2) / total_completed, 1)
    if total_completed
    else 0.0
)
total_verified_citations = sum(
    item["verified_citations"] for item in completed_audits
)
total_possible_citations = sum(item["total_citations"] for item in completed_audits)
accuracy_rate = (
    round((total_verified_citations / total_possible_citations) * 100, 1)
    if total_possible_citations
    else 0.0
)

if st.session_state["active_tab"] == "Dashboard":
    st.markdown(
        """
        <div class="hero-pill">AI-POWERED AUDIT</div>
        <h1 class="hero-title">Enterprise Compliance Review</h1>
        <p class="hero-sub">Upload a contract and your rule set once — get cited, rule-by-rule findings and a downloadable report in minutes, not hours.</p>
        """,
        unsafe_allow_html=True,
    )

    kpi_columns = st.columns(3, gap="medium")
    kpi_details = (
        (
            "Completed Audits",
            str(total_completed),
            f"{len(failed_audits)} failed runs" if failed_audits else "100% run health",
            "audits",
            "View Audit Breakdown",
        ),
        (
            "Avg. Review Time Saved",
            f"{avg_time_saved} min",
            f"{total_rules_checked} total clauses audited",
            "time",
            "View Time Calculation",
        ),
        (
            "Citation Accuracy Rate",
            f"{accuracy_rate}%" if total_possible_citations else "—",
            f"{total_verified_citations}/{total_possible_citations} quotes verified",
            "accuracy",
            "View Verification Stats",
        ),
    )
    for index, (column, (label, value, delta, key, button_label)) in enumerate(
        zip(kpi_columns, kpi_details)
    ):
        with column:
            with st.container(border=True):
                st.metric(label=label, value=value, delta=delta)
                if st.button(
                    button_label,
                    key=f"kpi_btn_{index}",
                    use_container_width=True,
                ):
                    st.session_state["selected_kpi"] = (
                        None if st.session_state["selected_kpi"] == key else key
                    )

    selected_kpi = st.session_state["selected_kpi"]
    if selected_kpi == "audits":
        st.info(
            f"**Completed Audits Detail:** {total_completed} contracts audited "
            f"successfully, {len(failed_audits)} failed runs recorded in this session."
        )
    elif selected_kpi == "time":
        st.info(
            f"**Time Saved Detail:** Calculated at 8.2 minutes saved per compliance "
            f"rule audited ({total_rules_checked} rules reviewed across "
            f"{total_completed} contracts)."
        )
    elif selected_kpi == "accuracy":
        st.info(
            f"**Citation Accuracy Detail:** {total_verified_citations} out of "
            f"{total_possible_citations} extracted citations were verified against "
            "source contract text."
        )

    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
    col1, col2 = st.columns(2, gap="large")
    with col1:
        with st.container(border=True):
            st.markdown("### Contract PDF")
            st.caption("Drag in a file or browse. 200MB limit, PDF only.")
            uploaded_file = st.file_uploader(
                "Drop your contract here",
                type=["pdf"],
                key=f"pdf_uploader_{st.session_state['uploader_key']}",
                label_visibility="collapsed",
            )
    with col2:
        with st.container(border=True):
            st.markdown("### Compliance Rules")
            st.caption(
                "One complete rule per line; commas and AND/OR clauses stay together. "
                "Clears automatically on page refresh."
            )
            rules_input = st.text_area(
                "Compliance Rules",
                value=st.session_state["rules_draft"],
                placeholder="Enter your compliance rules here (one rule per line)...",
                height=175,
                label_visibility="collapsed",
            )
            st.session_state["rules_draft"] = rules_input

    file_bytes = uploaded_file.getvalue() if uploaded_file else b""
    full_text = ""
    contract_pages = []
    parse_error = None
    detected_type = None
    selected_contract_type = "Select contract type..."
    if file_bytes:
        try:
            contract_pages = DocumentParserService.extract_pdf_pages(file_bytes)
            full_text = DocumentParserService.format_pdf_pages(contract_pages)
            detected_type = DocumentParserService.classify_contract_type(full_text)
        except Exception as error:
            parse_error = str(error)
            st.error(f"Could not read the uploaded PDF: {parse_error}")
        type_options = ["Select contract type..."] + list(CONTRACT_TYPES)
        if detected_type:
            st.info(
                f"Suggested contract type: **{detected_type}**. "
                "Confirm or change it before auditing."
            )
            initial_type_index = type_options.index(detected_type)
        else:
            st.warning(
                "Contract type could not be confidently identified. Select the "
                "correct type before running the audit."
            )
            initial_type_index = 0
        selected_contract_type = st.selectbox(
            "Contract type",
            type_options,
            index=initial_type_index,
            key=(
                f"contract_type_selector_{st.session_state['uploader_key']}_"
                f"{uploaded_file.name}"
            ),
        )

    input_rules = [
        line.strip()
        for line in rules_input.replace("\r\n", "\n").split("\n")
        if line.strip()
    ]
    effective_type = (
        selected_contract_type
        if selected_contract_type != "Select contract type..."
        else None
    )
    rules_to_exclude = DocumentParserService.mismatched_rules(
        effective_type, input_rules
    )
    alignment_warning = DocumentParserService.rule_alignment_warning(
        effective_type, input_rules
    )
    uploaded_name = uploaded_file.name if uploaded_file else "no-file"
    if rules_to_exclude:
        st.warning(alignment_warning)
        rule_handling = st.radio(
            "Remove these rules or run anyway?",
            ("Run all rules anyway", "Remove potentially inapplicable rules"),
            horizontal=True,
            key=(
                f"rule_handling_{st.session_state['uploader_key']}_"
                f"{uploaded_name}"
            ),
        )
    else:
        rule_handling = "Run all rules anyway"
    parsed_rules = input_rules
    if rule_handling == "Remove potentially inapplicable rules":
        parsed_rules = [rule for rule in input_rules if rule not in rules_to_exclude]
    st.caption(f"{len(parsed_rules)} rule(s) will be audited.")

    st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
    if st.button("⚡ Compile and Audit Contract", type="primary", use_container_width=True):
        if not uploaded_file:
            st.warning("Please upload a Contract PDF first.")
        elif not rules_input.strip():
            st.warning("Please enter at least one compliance rule.")
        elif parse_error:
            st.error(f"Could not read the uploaded PDF: {parse_error}")
        elif not full_text.strip():
            st.error("No readable text found inside this PDF.")
        elif selected_contract_type == "Select contract type...":
            st.warning("Select a contract type before running the audit.")
        elif not parsed_rules:
            st.warning("No rules remain after filtering; add or restore rules.")
        else:
            start_ts = time.time()
            with st.spinner(
                "Multi-Agent Pipeline Active: Extracting clauses, verifying citations "
                "via OpenRouter, and compiling your report..."
            ):
                try:
                    # The audit runs on the full contract text, so the optional
                    # Qdrant/embedding indexing step is intentionally skipped here:
                    # an unavailable embedding provider must never stall the audit.
                    initial_state = {
                        "contract_text": full_text[:35000],
                        "source_contract_text": full_text,
                        "retrieval_context": full_text[:35000],
                        "compliance_rules": parsed_rules,
                        "contract_type": selected_contract_type,
                        "rule_alignment_warning": alignment_warning,
                        "total_pages": len(contract_pages),
                        "audit_report": {},
                        "critic_feedback": "",
                        "citation_review_executed": False,
                        "citation_review_status": "NOT RUN",
                        "citation_review_detail": "",
                        "verified_citations": 0,
                        "total_citations": 0,
                        "model_used": "",
                        "ai_error": None,
                        "loop_count": 0,
                        "status": "pending",
                    }

                    final_state = dict(initial_state)
                    for output in get_compiled_graph().stream(initial_state):
                        for state_update in output.values():
                            final_state.update(state_update)

                    report_name = f"VERITAS_REPORT_{uploaded_file.name}"
                    pdf_bytes = None
                    report_path = None
                    try:
                        report_path = f"/tmp/VERITAS_REPORT_{uuid.uuid4()}.pdf"
                        AuditReportGenerator.generate_pdf_report(
                            report_path, uploaded_file.name, final_state
                        )
                        with open(report_path, "rb") as pdf_file:
                            pdf_bytes = pdf_file.read()
                    except Exception as pdf_error:
                        # Findings are still valuable even if PDF layout fails.
                        st.warning(
                            "The audit completed, but the PDF report could not be "
                            f"generated: {pdf_error}"
                        )
                    finally:
                        if report_path:
                            try:
                                os.remove(report_path)
                            except OSError:
                                pass

                    elapsed = round(time.time() - start_ts, 1)
                    ai_error = final_state.get("ai_error")
                    report_payload = final_state.get("audit_report") or {}
                    findings = [
                        item
                        for item in (report_payload.get("findings") or [])
                        if isinstance(item, dict)
                    ]
                    compliant_count = sum(
                        finding.get("result", "").lower() == "compliant"
                        for finding in findings
                    )
                    verified_count = final_state.get("verified_citations", 0)
                    audit_passed = (
                        not ai_error
                        and final_state.get("status") == "pass"
                        and final_state.get("citation_review_executed") is True
                        and final_state.get("citation_review_status") == "PASS"
                    )
                    if not audit_passed:
                        findings = []
                    failure_reason = (
                        ai_error
                        or final_state.get("critic_feedback")
                        or f"Audit ended with status: {final_state.get('status', 'unknown')}"
                    )
                    record = {
                        "id": f"AUD-{len(history) + 101}",
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "document": uploaded_file.name,
                        "contract_type": selected_contract_type,
                        "status": (
                            "Passed"
                            if audit_passed
                            else (
                                "Incomplete"
                                if final_state.get("status") == "incomplete"
                                else "Failed"
                            )
                        ),
                        "failure_reason": "None — all checks completed"
                        if audit_passed
                        else failure_reason,
                        "model": final_state.get("model_used", "NONE"),
                        "findings": findings,
                        "summary": summarize_findings(findings),
                        "rule_alignment_warning": alignment_warning,
                        "citation_review_executed": bool(
                            final_state.get("citation_review_executed")
                        ),
                        "citation_review_status": final_state.get(
                            "citation_review_status", "NOT RUN"
                        ),
                        "citation_review_detail": final_state.get(
                            "citation_review_detail", ""
                        ),
                        "rules": parsed_rules,
                        "rules_count": len(parsed_rules),
                        "compliant_count": compliant_count,
                        "verified_citations": verified_count,
                        "total_citations": final_state.get(
                            "total_citations",
                            sum(
                                1
                                for finding in findings
                                if not str(finding.get("result", ""))
                                .lower()
                                .startswith("gap")
                                and finding.get("citation")
                            ),
                        ),
                        "duration_sec": elapsed,
                        "pdf_bytes": pdf_bytes,
                        "report_name": report_name,
                    }
                    st.session_state["audit_history"].insert(0, record)
                    st.session_state["latest_report"] = record
                    st.session_state["uploader_key"] += 1

                    if ai_error:
                        st.error(f"Automated audit could not be completed: {ai_error}")
                    elif not findings:
                        st.error(
                            "Citation or rule-coverage verification did not pass. "
                            "The report contains no findings; manual review is required."
                        )
                    st.rerun()
                except Exception as error:
                    st.error(f"Pipeline execution error: {error!r}")
                    st.exception(error)

    if "latest_report" in st.session_state:
        latest = st.session_state["latest_report"]
        if latest["status"] == "Failed":
            st.error(
                f"⚠️ Audit Failed for {latest['document']} — "
                f"Cause: {latest['failure_reason']}"
            )
        elif latest["status"] == "Incomplete":
            st.warning(
                f"⚠️ Audit INCOMPLETE for {latest['document']} — "
                f"{latest['failure_reason']}"
            )
        else:
            st.success(
                f"✅ Audit Complete for {latest['document']} "
                f"({latest['compliant_count']}/{latest['rules_count']} rules "
                f"compliant in {latest['duration_sec']}s)"
            )

        if latest.get("rule_alignment_warning"):
            st.warning(latest["rule_alignment_warning"])

        if latest.get("findings"):
            summary = latest.get("summary") or summarize_findings(latest["findings"])
            verified = (
                latest.get("citation_review_executed") is True
                and latest.get("citation_review_status") == "PASS"
            )
            model_used = str(latest.get("model_used") or "").lower()
            report_analysis = (
                (latest.get("audit_report") or {}).get("analysis")
                if isinstance(latest.get("audit_report"), dict)
                else {}
            )
            analysis_mode = str(
                report_analysis.get("mode")
                if isinstance(report_analysis, dict)
                else ""
            ).lower()
            is_fallback = (
                "heuristic" in model_used
                or "fallback" in model_used
                or analysis_mode == "heuristic_fallback"
            )
            if is_fallback:
                st.error(
                    "AI ANALYSIS UNAVAILABLE — this report was NOT generated by "
                    "the compliance AI. It contains deterministic fallback results "
                    "only. Manual review is required."
                )
            with st.container(border=True):
                st.markdown("### Executive summary")
                st.markdown(
                    f"**RECOMMENDATION:** {summary['recommendation']}"
                    if verified and not is_fallback
                    else "**RECOMMENDATION:** MANUAL REVIEW REQUIRED — "
                    + (
                        "local fallback only, not AI analysis"
                        if is_fallback
                        else "automated verification is incomplete"
                    )
                )
                st.markdown(
                    f"**VERDICT SUMMARY:** {summary.get('verdict_summary', '')}"
                )
                st.markdown(f"**RISK SEVERITY:** {summary['risk_summary']}")
                if summary.get("key_issue"):
                    st.markdown(f"**KEY ISSUE:** {summary['key_issue']}")
                elif summary.get("status_note"):
                    st.markdown(f"**STATUS NOTE:** {summary['status_note']}")
                st.markdown(f"**NEXT STEP:** {summary['next_step']}")
                st.caption(
                    f"Contract type: {latest.get('contract_type') or 'Unclassified'} "
                    f"• Citation review: {latest.get('citation_review_status') or 'NOT RUN'}"
                    f" — {latest.get('citation_review_detail') or 'no detail recorded'}"
                )

            st.markdown("#### Rule-by-rule findings")
            st.dataframe(
                [
                    {
                        "Rule": item.get("rule", ""),
                        "Result": (
                            f"FALLBACK — {item.get('result', '')}"
                            if is_fallback
                            else (
                                item.get("result", "")
                                if verified or item.get("result") == "Compliant"
                                else f"UNVERIFIED — {item.get('result', '')}"
                            )
                        ),
                        "Priority": item.get("priority", ""),
                        "Risk if missing": item.get("risk_if_missing") or "—",
                        "Confidence": item.get("confidence", ""),
                        "Why": item.get("confidence_reason", ""),
                        "Citation": (
                            "\n".join(
                                f"{source.get('quote', '')} (Page {source.get('page')})"
                                for source in item.get("citations", [])
                                if isinstance(source, dict)
                            )
                            if item.get("citations")
                            else (
                                f"{item.get('citation') or '—'}"
                                + (
                                    f" ({item['source_reference']})"
                                    if item.get("source_reference")
                                    else ""
                                )
                            )
                        ),
                        "Reasoning": item.get("reasoning", ""),
                        "Next step": item.get("next_step", ""),
                    }
                    for item in latest["findings"]
                ],
                use_container_width=True,
                hide_index=True,
            )

        if latest.get("pdf_bytes"):
            st.download_button(
                label=f"📥 Download {latest['report_name']}",
                data=latest["pdf_bytes"],
                file_name=latest["report_name"],
                mime="application/pdf",
                key="latest_dl_btn",
                use_container_width=True,
            )
        else:
            st.info(
                "The PDF report could not be generated for this run. The findings "
                "shown above are complete and were saved to Audit History."
            )

elif st.session_state["active_tab"] == "Audit History":
    st.markdown("## 📜 Audit History")
    st.caption(
        "Complete log of every contract audit run, including pass/fail outcomes "
        "and failure causes."
    )
    if not history:
        st.info(
            "No audits run yet in this session. Go to Dashboard to upload a contract "
            "and run your first audit."
        )
    else:
        for index, item in enumerate(history):
            with st.container(border=True):
                col1, col2, col3, col4 = st.columns([2.5, 1.5, 2, 1.5])
                with col1:
                    st.markdown(f"{item['id']} — {item['document']}")
                    st.caption(
                        f"Run at {item['timestamp']} • Duration: {item['duration_sec']}s "
                        f"• Model: {item['model']}"
                    )
                with col2:
                    if item["status"] == "Passed":
                        st.success(
                            f"✅ PASSED ({item['compliant_count']}/"
                            f"{item['rules_count']} Compliant)"
                        )
                    elif item["status"] == "Incomplete":
                        st.warning("⚠️ INCOMPLETE")
                    else:
                        st.error("❌ FAILED")
                with col3:
                    if item["status"] == "Failed":
                        st.markdown(f"Failure Cause: {item['failure_reason']}")
                    else:
                        st.markdown(
                            f"Citations Verified: {item['verified_citations']}/"
                            f"{item['total_citations']}"
                        )
                with col4:
                    st.download_button(
                        label="📥 Download PDF",
                        data=item["pdf_bytes"],
                        file_name=item["report_name"],
                        mime="application/pdf",
                        key=f"hist_dl_{index}",
                        use_container_width=True,
                    )

else:
    st.markdown("## 📋 Rule Set History")
    st.caption(
        "View the exact compliance rule sets applied to each audited contract and "
        "reload any rule set into the Dashboard."
    )
    if not history:
        st.info("No rule sets recorded yet. Run an audit from the Dashboard to log them.")
    else:
        for index, item in enumerate(history):
            with st.container(border=True):
                rule_col, action_col = st.columns([4, 1.2])
                with rule_col:
                    st.markdown(
                        f"Audit {item['id']} • Contract: {item['document']} • "
                        f"Status: {item['status']} ({item['timestamp']})"
                    )
                    for rule_index, rule in enumerate(item["rules"], 1):
                        st.markdown(f"{rule_index}. {rule}")
                with action_col:
                    if st.button(
                        "Use in Dashboard",
                        key=f"reuse_rules_{index}",
                        use_container_width=True,
                    ):
                        st.session_state["rules_draft"] = "\n".join(item["rules"])
                        st.session_state["active_tab"] = "Dashboard"
                        st.rerun()

st.markdown(
    "<p style='text-align:center; color:#94A3B8; font-size:12px; margin-top:24px;'>"
    "Automated decision support, not legal advice. Review cited contract language "
    "before making decisions.</p>",
    unsafe_allow_html=True,
)
