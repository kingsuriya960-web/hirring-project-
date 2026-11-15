import io
import os
import sys
import time
import uuid
from datetime import datetime

import streamlit as st
from pypdf import PdfReader

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

if os.path.exists(".env"):
    with open(".env") as f:
        for line in f:
            if "=" in line and not line.strip().startswith("#"):
                key, value = line.strip().split("=", 1)
                os.environ.setdefault(key, value.strip("\"'"))

from app.services.parser import DocumentParserService
from app.services.report_generator import AuditReportGenerator
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
failed_audits = [item for item in history if item["status"] == "Failed"]
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
            st.caption("One rule per line. Clears automatically on page refresh.")
            rules_input = st.text_area(
                "Compliance Rules",
                value=st.session_state["rules_draft"],
                placeholder="Enter your compliance rules here (one rule per line)...",
                height=175,
                label_visibility="collapsed",
            )
            st.session_state["rules_draft"] = rules_input

    st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
    if st.button("⚡ Compile and Audit Contract", type="primary", use_container_width=True):
        if not uploaded_file:
            st.warning("Please upload a Contract PDF first.")
        elif not rules_input.strip():
            st.warning("Please enter at least one compliance rule.")
        else:
            start_ts = time.time()
            with st.spinner(
                "Multi-Agent Pipeline Active: Extracting clauses, verifying citations "
                "via Gemini, and compiling your report..."
            ):
                try:
                    file_bytes = uploaded_file.getvalue()
                    reader = PdfReader(io.BytesIO(file_bytes))
                    pages_text = []
                    for page_index, page in enumerate(reader.pages):
                        page_text = page.extract_text() or ""
                        if page_text.strip():
                            pages_text.append(
                                f"--- PAGE {page_index + 1} ---\n{page_text.strip()}"
                            )
                    full_text = "\n\n".join(pages_text)

                    if not full_text.strip():
                        st.error("No readable text found inside this PDF.")
                    else:
                        try:
                            from app.services.vector_db import VectorStoreService

                            vector_store = VectorStoreService()
                            chunks = DocumentParserService.extract_and_chunk_pdf(file_bytes)
                            if chunks:
                                vector_store.index_chunks(str(uuid.uuid4()), chunks)
                        except Exception as error:
                            st.warning(
                                f"Optional vector indexing was skipped: {error}"
                            )

                        parsed_rules = [
                            line.strip()
                            for line in rules_input.replace("\r\n", "\n").split("\n")
                            if line.strip()
                        ]
                        initial_state = {
                            "contract_text": full_text[:35000],
                            "compliance_rules": parsed_rules,
                            "audit_report": {},
                            "critic_feedback": "",
                            "citation_review_status": "",
                            "verified_citations": 0,
                            "model_used": "",
                            "ai_error": None,
                            "loop_count": 0,
                            "status": "pending",
                        }

                        final_state = dict(initial_state)
                        for output in get_compiled_graph().stream(initial_state):
                            for state_update in output.values():
                                final_state.update(state_update)

                        report_path = f"/tmp/VERITAS_REPORT_{uuid.uuid4()}.pdf"
                        AuditReportGenerator.generate_pdf_report(
                            report_path, uploaded_file.name, final_state
                        )
                        with open(report_path, "rb") as pdf_file:
                            pdf_bytes = pdf_file.read()

                        elapsed = round(time.time() - start_ts, 1)
                        ai_error = final_state.get("ai_error")
                        findings = final_state.get("audit_report", {}).get("findings", [])
                        compliant_count = sum(
                            finding.get("result", "").lower() == "compliant"
                            for finding in findings
                        )
                        verified_count = final_state.get("verified_citations", 0)
                        audit_passed = (
                            not ai_error and final_state.get("status") == "pass"
                        )
                        failure_reason = (
                            ai_error
                            or final_state.get("critic_feedback")
                            or f"Audit ended with status: {final_state.get('status', 'unknown')}"
                        )
                        record = {
                            "id": f"AUD-{len(history) + 101}",
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "document": uploaded_file.name,
                            "status": "Passed" if audit_passed else "Failed",
                            "failure_reason": "None — all checks completed"
                            if audit_passed
                            else failure_reason,
                            "model": final_state.get("model_used", "NONE"),
                            "rules": parsed_rules,
                            "rules_count": len(parsed_rules),
                            "compliant_count": compliant_count,
                            "verified_citations": verified_count,
                            "total_citations": len(findings),
                            "duration_sec": elapsed,
                            "pdf_bytes": pdf_bytes,
                            "report_name": f"VERITAS_REPORT_{uploaded_file.name}",
                        }
                        st.session_state["audit_history"].insert(0, record)
                        st.session_state["latest_report"] = record
                        st.session_state["uploader_key"] += 1
                        st.rerun()
                except Exception as error:
                    st.error(f"Pipeline execution error: {error}")

    if "latest_report" in st.session_state:
        latest = st.session_state["latest_report"]
        if latest["status"] == "Failed":
            st.error(
                f"⚠️ Audit Failed for {latest['document']} — "
                f"Cause: {latest['failure_reason']}"
            )
        else:
            st.success(
                f"✅ Audit Complete for {latest['document']} "
                f"({latest['compliant_count']}/{latest['rules_count']} rules "
                f"compliant in {latest['duration_sec']}s)"
            )
        st.download_button(
            label=f"📥 Download {latest['report_name']}",
            data=latest["pdf_bytes"],
            file_name=latest["report_name"],
            mime="application/pdf",
            key="latest_dl_btn",
            use_container_width=True,
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
