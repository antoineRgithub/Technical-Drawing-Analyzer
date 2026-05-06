"""
app.py — Main Streamlit application
POC: Industrial Technical Drawing Analyser & Cost Estimator

Reads a PDF technical drawing, extracts text and schematic images,
uses GPT-4o to identify dimensions / material / surface finish, then
computes an estimated manufacturing cost.
"""

from __future__ import annotations

import os

import streamlit as st
from dotenv import load_dotenv

from modules.ai_analyzer import analyse_document
from modules.cost_calculator import (
    CostParameters,
    estimate_cost,
    MATERIAL_DB,
)
from modules.pdf_processor import load_pdf

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
load_dotenv()

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Technical Drawing Analyser",
    page_icon="⚙️",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Sidebar — configuration
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("⚙️ Configuration")

    # API key
    api_key = st.text_input(
        "OpenAI API Key",
        value=os.getenv("OPENAI_API_KEY", ""),
        type="password",
        help="Provide your OpenAI API key. It is never stored.",
    )

    st.divider()
    st.subheader("Cost Parameters")

    hourly_rate = st.number_input(
        "CNC Hourly Rate (€/h)", min_value=10.0, max_value=500.0, value=65.0, step=5.0
    )
    setup_time = st.number_input(
        "Setup Time (h)", min_value=0.0, max_value=8.0, value=0.5, step=0.25
    )
    overhead_rate = st.slider(
        "Overhead Rate (%)", min_value=0, max_value=60, value=25
    ) / 100.0
    margin_rate = st.slider(
        "Profit Margin (%)", min_value=0, max_value=60, value=15
    ) / 100.0
    quantity = st.number_input(
        "Quantity", min_value=1, max_value=10_000, value=1, step=1,
        help="Overrides the quantity found in the drawing.",
    )

    cost_params = CostParameters(
        hourly_rate=hourly_rate,
        setup_time=setup_time,
        overhead_rate=overhead_rate,
        margin_rate=margin_rate,
    )

    st.divider()
    st.subheader("Material Override")
    material_override = st.selectbox(
        "Force material (optional)",
        options=["(use drawing value)"] + list(MATERIAL_DB.keys()),
    )

    st.divider()
    st.caption(
        "This is a proof-of-concept. "
        "Cost estimates are indicative only and should not be used for quoting."
    )

# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------
st.title("📐 Industrial Technical Drawing Analyser")
st.markdown(
    "Upload a **PDF technical drawing** of a machined part. "
    "The app will extract all text, analyse the schematic with GPT-4o, "
    "identify dimensions and materials, and compute an estimated manufacturing cost."
)

uploaded_file = st.file_uploader(
    "Drop your PDF here",
    type=["pdf"],
    help="Multi-page PDFs are supported.",
)

if uploaded_file is None:
    st.info("Upload a PDF file to get started.")
    st.stop()

# ---------------------------------------------------------------------------
# Load & display PDF
# ---------------------------------------------------------------------------
file_bytes = uploaded_file.read()

with st.spinner("Reading PDF…"):
    doc = load_pdf(file_bytes, filename=uploaded_file.name, dpi=150)

st.success(f"Loaded **{doc.filename}** — {len(doc.pages)} page(s)")

tab_pages, tab_text, tab_analysis, tab_cost = st.tabs(
    ["📄 Pages", "📝 Extracted Text", "🔍 AI Analysis", "💶 Cost Estimate"]
)


def _apply_sidebar_overrides(analysis) -> None:
    """Apply sidebar material and quantity overrides to an analysis object in-place."""
    if material_override != "(use drawing value)":
        analysis.material = material_override
    analysis.quantity = int(quantity)

# ---------------------------------------------------------------------------
# Tab 1 – Page images
# ---------------------------------------------------------------------------
with tab_pages:
    cols_per_row = 2
    for i in range(0, len(doc.pages), cols_per_row):
        cols = st.columns(cols_per_row)
        for j, page in enumerate(doc.pages[i : i + cols_per_row]):
            with cols[j]:
                st.image(
                    f"data:image/png;base64,{page.image_b64}",
                    caption=f"Page {page.page_number}",
                    use_container_width=True,
                )

# ---------------------------------------------------------------------------
# Tab 2 – Extracted text
# ---------------------------------------------------------------------------
with tab_text:
    if doc.full_text.strip():
        st.text_area(
            "Full extracted text",
            value=doc.full_text,
            height=500,
        )
    else:
        st.warning(
            "No text could be extracted from this PDF. "
            "This might be a scanned document — the AI will rely on the images only."
        )

# ---------------------------------------------------------------------------
# Tab 3 – AI analysis
# ---------------------------------------------------------------------------
with tab_analysis:
    if not api_key:
        st.warning("Please enter your OpenAI API Key in the sidebar to run the analysis.")
        st.stop()

    run_analysis = st.button("▶ Run AI Analysis", type="primary")

    if "analysis" not in st.session_state:
        st.session_state.analysis = None

    if run_analysis:
        with st.spinner("Sending document to GPT-4o… This may take 15–60 s."):
            try:
                analysis = analyse_document(doc, api_key=api_key)
                st.session_state.analysis = analysis
            except Exception as exc:
                st.error(f"Analysis failed: {exc}")
                st.stop()

    analysis = st.session_state.analysis

    if analysis is None:
        st.info("Click **Run AI Analysis** to extract information from the drawing.")
    else:
        _apply_sidebar_overrides(analysis)

        col_info, col_dims = st.columns([1, 2])

        with col_info:
            st.subheader("Part Information")
            st.markdown(f"**Part Name:** {analysis.part_name}")
            st.markdown(f"**Material:** {analysis.material}")
            st.markdown(f"**Surface Finish:** {analysis.surface_finish}")
            st.markdown(f"**Heat Treatment:** {analysis.heat_treatment}")
            st.markdown(f"**Quantity:** {analysis.quantity}")

            if analysis.notes:
                st.subheader("Notes")
                for note in analysis.notes:
                    st.markdown(f"- {note}")

        with col_dims:
            st.subheader("Extracted Dimensions")
            if analysis.dimensions:
                rows = [
                    {
                        "Name": d.name,
                        "Value": d.value,
                        "Unit": d.unit,
                        "Tol +": d.tolerance_plus if d.tolerance_plus is not None else "—",
                        "Tol −": d.tolerance_minus if d.tolerance_minus is not None else "—",
                    }
                    for d in analysis.dimensions
                ]
                st.dataframe(rows, use_container_width=True, hide_index=True)
            else:
                st.info("No dimensions were extracted.")

        with st.expander("Raw JSON response from GPT-4o"):
            st.json(analysis.raw_json)

# ---------------------------------------------------------------------------
# Tab 4 – Cost estimate
# ---------------------------------------------------------------------------
with tab_cost:
    analysis = st.session_state.get("analysis")

    if analysis is None:
        st.info("Run the AI analysis first (tab **🔍 AI Analysis**).")
    else:
        _apply_sidebar_overrides(analysis)

        unit_cost = estimate_cost(analysis, params=cost_params)
        breakdown = unit_cost.as_dict()
        total_cost = breakdown.pop("TOTAL")

        st.subheader("Unit Cost Breakdown")
        col_chart, col_table = st.columns([1, 1])

        with col_chart:
            st.bar_chart(breakdown)

        with col_table:
            for label, value in breakdown.items():
                st.metric(label=label, value=f"€ {value:,.2f}")

        st.divider()
        col1, col2, col3 = st.columns(3)
        col1.metric("Unit Cost", f"€ {total_cost:,.2f}")
        col2.metric("Quantity", str(analysis.quantity))
        col3.metric(
            "Total Order Cost",
            f"€ {total_cost * analysis.quantity:,.2f}",
        )

        st.caption(
            "⚠️ These estimates are generated automatically from the drawing data and are "
            "indicative only. Always validate with a qualified engineer before quoting."
        )
