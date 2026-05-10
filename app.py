"""
app.py — Main Streamlit application
POC: Industrial Technical Drawing Analyser & Cost Estimator

Reads a PDF technical drawing, extracts text and schematic images,
uses GPT-4o to identify dimensions / material / surface finish, then
computes an estimated manufacturing cost.
"""

from __future__ import annotations
from transformers import AutoProcessor, AutoModelForCausalLM 
import torch
import argparse
import pandas as pd
import os
import cv2
import numpy as np
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
import streamlit as st
from dotenv import load_dotenv

from modules.ai_analyzer import analyse_document
from modules.cost_calculator import (
    CostParameters,
    estimate_cost,
    MATERIAL_DB,
    lookup_material,
)
from modules.plan_analysis import (load_florence2, run_florence2_ocr, extract_dimensions_from_text, run_vqa_questions)
from modules.pdf_processor import load_pdf, preprocess_image_for_ocr
from modules.AI_analysis import return_thickness, return_length, return_height, return_number_of_bendings, return_number_of_holes

from modules.RAG import extract_entities
from modules.qwen_chat import ask_qwen
# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
load_dotenv()

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Analyse de Plans Techniques",
    page_icon="⚙️",
    layout="wide",
)

@st.cache_resource
def load_model():
    print("Loading Qwen... (only once)")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen3-VL-2B-Instruct",
        torch_dtype="auto",
        device_map="auto"
    )

    processor = AutoProcessor.from_pretrained(
        "Qwen/Qwen3-VL-2B-Instruct"
    )
    print("Qwen loaded !")

    return model, processor

model, processor = load_model()
# ---------------------------------------------------------------------------
# Sidebar — configuration
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("⚙️ Configuration")


    st.divider()
    st.subheader("Paramètres de coût")


    hourly_rate = st.number_input(
        "Taux horaire (€/h)", min_value=10.0, max_value=500.0, value=40.0, step=5.0
    )
    setup_time = st.number_input(
        "Temps de préparation (min)", min_value=0, max_value=120, value=30, step=5
    ) / 60 # convert to hours
    overhead_rate = st.slider(
        "Taux de coûts indirects (%)", min_value=0, max_value=60, value=25
    ) / 100.0
    margin_rate = st.slider(
        "Taux de marge (%)", min_value=0, max_value=60, value=15
    ) / 100.0
    quantity = st.number_input(
        "Quantité", min_value=1, max_value=10_000, value=1, step=1,
        help="Overrides the quantity found in the drawing.",
    )

    cost_params = CostParameters(
        hourly_rate=hourly_rate,
        setup_time=setup_time,
        overhead_rate=overhead_rate,
        margin_rate=margin_rate,
    )

    st.divider()
    st.subheader("Matériaux")
    material = st.selectbox(
        "Sélectionnez un matériau",
        options= list(MATERIAL_DB.keys()),
    )

    st.divider()
    st.caption(
        "This is a proof-of-concept. "
        "Cost estimates are indicative only and should not be used for quoting."
    )

# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------
st.title("📐 Etude de cas : outil d'analyse de plan techniques")
print("gpu available:", torch.cuda.is_available())
print(torch.__version__)
print(torch.cuda.get_device_name(0))
print(torch.cuda.get_device_capability(0))
st.markdown(
    "Chargez le **plan technique PDF** d'un composant. "
    "L'application va extraire les dimensions et le texte du fichier,"
    " et calculer une estimation du coût de fabrication."
)

uploaded_file = st.file_uploader(
    "Déposez votre PDF ici",
    type=["pdf"],
    help="Les PDFs multi-pages sont pris en charge.",
)

if uploaded_file is None:
    st.info("Déposez un fichier PDF.")
    st.stop()

# ---------------------------------------------------------------------------
# Load & display PDF
# ---------------------------------------------------------------------------
file_bytes = uploaded_file.read()

with st.spinner("Lecture du PDF…"):
    doc = load_pdf(file_bytes, filename=uploaded_file.name, dpi=150)

st.success(f"Loaded **{doc.filename}** — {len(doc.pages)} page(s)")

tab_pages, tab_text, tab_analysis, tab_cost, sandbag, chat = st.tabs(
    ["📄 Pages", "📝 Texte extrait", "🔍 Analyse IA", "💶 Estimation des coûts", "OCR + RAG (Texte)", "Chat" ]
)


def _apply_sidebar_overrides(analysis) -> None:
    """Apply sidebar material and quantity overrides to an analysis object in-place."""
    if material != "(use drawing value)":
        analysis["material"] = material
    
    analysis["quantity"] = int(quantity)

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

with sandbag:
    img = doc.pages[0].img
    
    width, height = img.size

    # left half
    left_img = img.crop((0, 0, width // 2, height))

    # right half
    right_img = img.crop((width // 2, 0, width, height))

    # Crop the top and the bottom of right_img
    middle_right_img = right_img.crop((0, height * 0.35, width // 2, height * 0.85))
    top_right_img = right_img.crop((0, 0, width // 2, height * 0.35))

    
    cols = st.columns(2)
    with cols[0]:
        
        st.markdown("**Preprocessed Text Region**")
        st.image(middle_right_img, caption="Preprocessed Text Region", use_container_width=True)
    with cols[1]:
        
        st.markdown("**Preprocessed Drawing Region**")
        st.image(top_right_img, caption="Preprocessed Table Region", use_container_width=True)
    
    text = doc.pages[0].text.strip()
    st.markdown("**OCR Extracted Text**")
    st.text_area("Extracted Text", value=text, height=200)

    labels = ["BENDING RADIUS",
              "UNDIMENSIONNED RADIUS",
              "ISO",]
    
    rag_extract = st.button("▶ Extraire les entités avec le modèle RAG", type="primary")
    if rag_extract:
        entities = extract_entities(text, labels)
        # print(f"Extracted entities: {entities}")
    

        st.markdown("**Extracted Entities**")
        rows = []

        for key, values in entities["entities"].items():
            if values:
                for v in values:
                    rows.append({"Field": key, "Value": v})
            else:
                rows.append({"Field": key, "Value": None})

        df = pd.DataFrame(rows)

        st.dataframe(df, use_container_width=True)

with chat:
    st.markdown("** **")
    img = doc.pages[0].img
    
    width, height = img.size

    # left half
    left_img = img.crop((0, 0, width // 2, height))

    # middle of left half
    middle_left_img = left_img.crop((0, height * 0.45, width // 2, height * 0.65))
    top_left_img = left_img.crop((0, 0, width // 2, height * 0.45))
    bottom_left_img = left_img.crop((0, height * 0.65, width // 2, height))

    image_choice = st.selectbox(
        "Choose the image to chat with",
        (
            "Full left image",
            "Top left image",
            "Middle left image",
            "Bottom left image",
        ),
    )

    selected_chat_image = {
        "Full left image": left_img,
        "Top left image": top_left_img,
        "Middle left image": middle_left_img,
        "Bottom left image": bottom_left_img,
    }[image_choice]

    st.image(selected_chat_image, caption=image_choice, use_container_width=True)
    
    st.session_state.messages = []

    user_question = st.text_input("Ask something about the image:")
    send = st.button("Send")

    
    if user_question and selected_chat_image is not None and send:
        print("asking qwen ...")
        answer = ask_qwen(
            selected_chat_image,
            user_question, 
            model,
            processor,
        )
        print(f"Qwen answer: {answer}")
        # store chat history
        st.session_state.messages.append(("You", user_question))
        st.session_state.messages.append(("Qwen", answer))

    print(st.session_state.messages)
    # ----------------------------
    # Display chat history
    # ----------------------------
    for role, msg in st.session_state.messages:
        if role == "You":
            st.markdown(f"🧑 **You:** {msg}")
        else:
            st.markdown(f"🤖 **Qwen:** {msg}")

    st.markdown("**Qwen-VL analysis of the drawing**")
    # st.text_area("Qwen-VL Output", value=output_text[0], height=200)
    



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

    run_analysis = st.button("▶ Exécuter l'analyse IA", type="primary")

    cols = st.columns(2)
    with cols[1]:
        st.markdown("**Image à analyser**")
        st.image(left_img, caption="Image à analyser")

    if "analysis" not in st.session_state:
        st.session_state.analysis = {}

    if run_analysis:
        with cols[0]:
                with st.spinner("Analyzing document with Qwen-VL… This may take 1–10 s."):
                    thickness = return_thickness(doc.pages[0].img, model, processor)
                    st.session_state.analysis['thickness'] = thickness
                    length = return_length(doc.pages[0].img, model, processor)
                    st.session_state.analysis['length'] = length
                    height = return_height(doc.pages[0].img, model, processor)
                    st.session_state.analysis['height'] = height
                    st.session_state.analysis['volume'] = int(height) * int(length) * int(thickness)
                    estimated_mass = st.session_state.analysis['volume'] / 1_000_000 * MATERIAL_DB[material][0] # volume in m3 * density in g/m3
                    st.session_state.analysis['estimated_mass'] = estimated_mass
                    st.session_state.analysis['number_of_bendings'] = return_number_of_bendings(doc.pages[0].img, model, processor)[0]
                    # st.session_state.analysis['bending_angle'] = return_number_of_bendings(doc.pages[0].img, model, processor)[1]
                    st.session_state.analysis['number_of_holes'] = return_number_of_holes(doc.pages[0].img, model, processor)
                analysis = st.session_state.analysis

                if analysis == {}:
                    st.info("Cliquez sur **Exécuter l'analyse IA** pour extraire les informations de la dessin.")
                else:

                    col_info, col_dims = st.columns([1, 2])

                    
                    st.subheader("Analyse sur l'image")
                    rows = [
                        {
                            "Dimensions": "Epaisseur",
                            "Valeur": analysis['thickness'] + "mm",
                        },
                        {
                            "Dimensions": "Longueur",
                            "Valeur": analysis['length'] + "mm",
                        },
                        {
                            "Dimensions": "Hauteur",
                            "Valeur": analysis['height'] + "mm",
                        },
                        
                    ]
                    st.dataframe(rows, use_container_width=True, hide_index=True)

                    rows2 = [
                        {
                            "Features": "Pliages",
                            "Quantité": analysis['number_of_bendings'],
                        },
                        {
                            "Features": "Trous",
                            "Quantité": analysis['number_of_holes'],
                        },
                    ]
                    st.dataframe(rows2, use_container_width=True, hide_index=True)
                    st.markdown(f"**Estimated Volume:** {analysis['volume']} mm³")
                    st.markdown(f"**Estimated Mass:** {analysis['estimated_mass']:.2f} g (using density of {material})")
    #         if analysis.notes:
    #             st.subheader("Notes")
    #             for note in analysis.notes:
    #                 st.markdown(f"- {note}")

    #     with col_dims:
    #         st.subheader("Extracted Dimensions")
    #         if analysis.dimensions:
    #             rows = [
    #                 {
    #                     "Name": d.name,
    #                     "Value": d.value,
    #                     "Unit": d.unit,
    #                     "Tol +": d.tolerance_plus if d.tolerance_plus is not None else "—",
    #                     "Tol −": d.tolerance_minus if d.tolerance_minus is not None else "—",
    #                 }
    #                 for d in analysis.dimensions
    #             ]
    #             st.dataframe(rows, use_container_width=True, hide_index=True)
    #         else:
    #             st.info("No dimensions were extracted.")

    #     with st.expander("Raw JSON response from GPT-4o"):
    #         st.json(analysis.raw_json)

# ---------------------------------------------------------------------------
# Tab 4 – Cost estimate
# ---------------------------------------------------------------------------
with tab_cost:
    
    cost_analysis = st.session_state.get("analysis")

    if cost_analysis== {}:
        st.info("Run the AI cost_analysis first (tab **🔍 AI cost_analysis**).")
    else:
        # run_cost = st.button("▶ Run Cost Estimate", type="primary")
        # if not run_cost:
        #     st.info("Click **Run Cost Estimate** to compute the manufacturing cost based on the extracted dimensions and material.")
    #     st.stop()
        _apply_sidebar_overrides(cost_analysis)

        unit_cost = estimate_cost(cost_analysis, params=cost_params)
        breakdown = unit_cost.as_dict()
        total_cost = breakdown.pop("TOTAL")
        breakdown.pop("Temps de découpage (h)")

        st.subheader("Répartition des coûts :")
        col_chart, col_table = st.columns([1, 1])

        with col_chart:
            st.bar_chart(breakdown)

        with col_table:
            # Present breakdown as a clean two-column table
            rows = []
            for label, value in breakdown.items():
                rows.append({"Coût": label, "Montant": f"€ {value:,.2f}"})

            # Use st.table for a compact, well-aligned display
            st.table(rows)

        st.divider()
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Coût unitaire", f"€ {total_cost:,.2f}")
        col2.metric("Temps d'usinage : ", f"{unit_cost.total_time_hours * 60:.2f} min")
        col3.metric("Quantité", str(cost_analysis["quantity"]))
        col4.metric(
            "Coût total du devis",
            f"€ {total_cost * cost_analysis['quantity']:,.2f}",
        )


        st.caption(
            "⚠️ These estimates are generated automatically from the drawing data and are "
            "indicative only. Always validate with a qualified engineer before quoting."
        )


