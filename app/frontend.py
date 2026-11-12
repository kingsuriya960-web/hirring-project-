import os

import requests
import streamlit as st


BACKEND_URL = os.getenv("BACKEND_URL", "http://api:8000").rstrip("/")

st.set_page_config(page_title="Compliance Auditor", page_icon="C", layout="wide")
st.title("Enterprise compliance review")
st.caption("Upload a contract and rules once; receive the agent findings and a downloadable PDF report.")

uploaded_file = st.file_uploader("Contract PDF", type=["pdf"])
rules = st.text_area(
    "Compliance rules",
    placeholder="Governing law must be Delaware; termination requires 30 days notice; late fees cannot exceed 5%",
    height=120,
)

if st.button("Compile and audit contract", type="primary", use_container_width=True):
    if uploaded_file is None or not rules.strip():
        st.warning("Choose a PDF and enter at least one compliance rule.")
    else:
        try:
            with st.spinner("Parsing, indexing, auditing, and generating the report..."):
                response = requests.post(
                    f"{BACKEND_URL}/api/v1/auto-audit",
                    files={"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")},
                    data={"rules": rules},
                    timeout=(10, 300),
                )
            if not response.ok:
                try:
                    detail = response.json().get("detail", response.text)
                except ValueError:
                    detail = response.text
                st.error(f"Audit failed ({response.status_code}): {detail}")
            else:
                compliance_status = response.headers.get("X-Compliance-Status", "undetermined")
                citation_status = response.headers.get("X-Citation-Review", "unknown")
                analysis_mode = response.headers.get("X-Analysis-Mode", "unknown")
                st.subheader("Audit result")
                st.metric("Compliance", compliance_status.replace("_", " ").title())
                st.write(f"Citation review: **{citation_status.title()}**")
                st.caption(f"Analysis method: {analysis_mode.replace('_', ' ')}")
                st.download_button(
                    "Download audit report PDF",
                    data=response.content,
                    file_name=response.headers.get(
                        "Content-Disposition", "attachment; filename=compliance_audit_report.pdf"
                    ).split("filename=")[-1].strip('"'),
                    mime="application/pdf",
                    type="primary",
                    use_container_width=True,
                )
        except requests.RequestException as exc:
            st.error(f"Could not reach the API: {exc}")