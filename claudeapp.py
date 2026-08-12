import streamlit as st
from PIL import Image
import json
import os
import pandas as pd

from predict import predict_image
from recommendation import generate_recommendation


# ==========================================================
# Page Configuration
# ==========================================================

st.set_page_config(
    page_title="AgroVision AI | Plant Disease Diagnostics",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ==========================================================
# Load Disease Database
# ==========================================================

with open("disease_database.json", "r", encoding="utf-8") as file:
    DISEASE_DATABASE = json.load(file)

os.makedirs("temp", exist_ok=True)


# ==========================================================
# Design tokens
# ----------------------------------------------------------
# Palette grounded in the subject: canopy green (primary),
# moss (secondary), soil brown (accent for warnings/warmth),
# and a soft parchment background rather than a stock white
# or the generic cream+terracotta combo.
# ==========================================================

COLORS = {
    "canopy": "#1B4332",
    "moss": "#40916C",
    "sprout": "#95D5B2",
    "soil": "#A9702C",
    "soil_dark": "#7A4E1D",
    "paper": "#F7F7F2",
    "card": "#FFFFFF",
    "ink": "#1F2A24",
    "muted": "#5B6B62",
    "border": "#E1E5DE",
    "danger": "#B3261E",
    "warning": "#A9702C",
    "success": "#2D6A4F",
}


# ==========================================================
# Global Styles
# ==========================================================

st.markdown(
    f"""
    <style>

    @import url('https://fonts.googleapis.com/css2?family=Sora:wght@500;600;700;800&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@500;600&display=swap');

    html, body, [class*="css"] {{
        font-family: 'Inter', sans-serif;
        color: {COLORS['ink']};
    }}

    .stApp {{
        background-color: {COLORS['paper']};
    }}

    /* ---------- Sidebar ---------- */
    section[data-testid="stSidebar"] {{
        background-color: {COLORS['canopy']};
    }}
    section[data-testid="stSidebar"] * {{
        color: #EAF3EC !important;
    }}
    section[data-testid="stSidebar"] hr {{
        border-color: rgba(255,255,255,0.15);
    }}

    /* ---------- Hero header ---------- */
    .av-hero {{
        display: flex;
        align-items: center;
        gap: 18px;
        padding: 28px 32px;
        border-radius: 16px;
        background: linear-gradient(135deg, {COLORS['canopy']} 0%, {COLORS['moss']} 100%);
        margin-bottom: 28px;
    }}
    .av-hero-badge {{
        font-size: 40px;
        line-height: 1;
    }}
    .av-hero-title {{
        font-family: 'Sora', sans-serif;
        font-weight: 800;
        font-size: 30px;
        color: #FFFFFF;
        margin: 0;
        letter-spacing: -0.02em;
    }}
    .av-hero-sub {{
        font-family: 'Inter', sans-serif;
        font-size: 15px;
        color: #DDEDE2;
        margin-top: 4px;
    }}

    /* ---------- Section labels ---------- */
    .av-eyebrow {{
        font-family: 'JetBrains Mono', monospace;
        font-size: 12px;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: {COLORS['moss']};
        font-weight: 600;
        margin-bottom: 2px;
    }}
    .av-section-title {{
        font-family: 'Sora', sans-serif;
        font-weight: 700;
        font-size: 22px;
        color: {COLORS['canopy']};
        margin-top: 0;
        margin-bottom: 14px;
    }}

    /* ---------- Cards ---------- */
    div[data-testid="stVerticalBlockBorderWrapper"] {{
        background-color: {COLORS['card']};
        border: 1px solid {COLORS['border']} !important;
        border-radius: 14px !important;
    }}

    /* ---------- Metrics ---------- */
    div[data-testid="stMetric"] {{
        background-color: {COLORS['card']};
        border: 1px solid {COLORS['border']};
        border-radius: 12px;
        padding: 14px 16px 10px 16px;
    }}
    div[data-testid="stMetricLabel"] {{
        font-family: 'JetBrains Mono', monospace;
        font-size: 12px;
        color: {COLORS['muted']};
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }}
    div[data-testid="stMetricValue"] {{
        font-family: 'Sora', sans-serif;
        color: {COLORS['canopy']};
        font-weight: 700;
    }}

    /* ---------- Buttons / uploader ---------- */
    .stButton>button, .stDownloadButton>button {{
        background-color: {COLORS['canopy']};
        color: white;
        border-radius: 8px;
        border: none;
        font-weight: 600;
    }}
    .stButton>button:hover {{
        background-color: {COLORS['moss']};
    }}
    section[data-testid="stSidebar"] div[data-testid="stFileUploaderDropzone"] {{
        background-color: rgba(255,255,255,0.06);
        border: 1.5px dashed rgba(255,255,255,0.35);
    }}

    /* ---------- Tabs ---------- */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 4px;
        border-bottom: 1px solid {COLORS['border']};
    }}
    .stTabs [data-baseweb="tab"] {{
        font-family: 'Sora', sans-serif;
        font-weight: 600;
        color: {COLORS['muted']};
        padding: 8px 4px;
    }}
    .stTabs [aria-selected="true"] {{
        color: {COLORS['canopy']} !important;
        border-bottom: 2px solid {COLORS['moss']} !important;
    }}

    /* ---------- Confidence pill ---------- */
    .av-pill {{
        display: inline-block;
        padding: 6px 14px;
        border-radius: 999px;
        font-family: 'JetBrains Mono', monospace;
        font-weight: 600;
        font-size: 13px;
        margin-bottom: 14px;
    }}
    .av-pill-high {{ background:#E6F4EA; color:{COLORS['success']}; }}
    .av-pill-mid  {{ background:#FBF0DE; color:{COLORS['warning']}; }}
    .av-pill-low  {{ background:#FBE6E4; color:{COLORS['danger']}; }}

    /* ---------- List rows ---------- */
    .av-row {{
        display: flex;
        gap: 10px;
        padding: 7px 0;
        border-bottom: 1px solid {COLORS['border']};
        font-size: 14.5px;
    }}
    .av-row:last-child {{ border-bottom: none; }}
    .av-row-dot {{ color: {COLORS['moss']}; font-weight: 700; }}

    /* ---------- Footer ---------- */
    .av-footer {{
        text-align: center;
        color: {COLORS['muted']};
        font-size: 13px;
        padding: 18px 0 6px 0;
    }}
    .av-footer b {{ color: {COLORS['canopy']}; }}

    /* ---------- Empty state ---------- */
    .av-empty {{
        text-align: center;
        padding: 60px 20px;
        color: {COLORS['muted']};
    }}
    .av-empty-icon {{ font-size: 46px; margin-bottom: 10px; }}

    </style>
    """,
    unsafe_allow_html=True,
)


# ==========================================================
# Helpers
# ==========================================================

def render_list(items, empty_text):
    if not items:
        st.caption(empty_text)
        return
    rows = "".join(
        f'<div class="av-row"><span class="av-row-dot">›</span><span>{item}</span></div>'
        for item in items
    )
    st.markdown(rows, unsafe_allow_html=True)


def confidence_pill(confidence):
    if confidence >= 80:
        cls, label = "av-pill-high", f"🟢 High confidence · {confidence:.1f}%"
    elif confidence >= 50:
        cls, label = "av-pill-mid", f"🟡 Moderate confidence · {confidence:.1f}%"
    else:
        cls, label = "av-pill-low", f"🔴 Low confidence · {confidence:.1f}%"
    st.markdown(f'<span class="av-pill {cls}">{label}</span>', unsafe_allow_html=True)


# ==========================================================
# Sidebar — Upload & Context
# ==========================================================

with st.sidebar:
    st.markdown("### 🌿 AgroVision AI")
    st.caption("AI-powered plant leaf disease diagnostics")
    st.divider()

    st.markdown("**1. Upload a leaf image**")
    uploaded_file = st.file_uploader(
        "Drag and drop or browse",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
    )
    st.caption("Best results: single leaf, plain background, natural light.")

    st.divider()
    st.markdown("**About**")
    st.caption(
        "AgroVision AI classifies crop leaf images and returns a disease "
        "diagnosis, confidence score, and treatment guidance."
    )
    st.caption("Built with TensorFlow · EfficientNetB0 · Streamlit")


# ==========================================================
# Header
# ==========================================================

st.markdown(
    """
    <div class="av-hero">
        <div class="av-hero-badge">🌿</div>
        <div>
            <p class="av-hero-title">AgroVision AI</p>
            <p class="av-hero-sub">Upload a leaf photo to get an instant disease diagnosis and treatment plan.</p>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ==========================================================
# Empty State
# ==========================================================

if uploaded_file is None:
    st.markdown(
        """
        <div class="av-empty">
            <div class="av-empty-icon">📷</div>
            <div style="font-family:'Sora',sans-serif; font-weight:700; font-size:18px; color:#1B4332;">
                No image uploaded yet
            </div>
            <div style="margin-top:6px;">Use the panel on the left to upload a plant leaf photo and start a diagnosis.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ==========================================================
# Process Uploaded Image
# ==========================================================

else:
    original_filename = os.path.basename(uploaded_file.name)
    image_path = os.path.join("temp", original_filename)
    with open(image_path, "wb") as file:
        file.write(uploaded_file.getbuffer())

    result = predict_image(image_path)
    info = DISEASE_DATABASE.get(result["prediction"], {})
    recommendation = generate_recommendation(result, info)
    confidence = result["confidence"]

    # ------------------------------------------------------
    # Image + Key Metrics
    # ------------------------------------------------------
    left, right = st.columns([1, 1.4], gap="large")

    with left:
        with st.container(border=True):
            image = Image.open(uploaded_file)
            st.image(image, use_container_width=True)
            st.caption(f"Source: {original_filename}")

    with right:
        with st.container(border=True):
            st.markdown('<p class="av-eyebrow">Diagnosis</p>', unsafe_allow_html=True)
            st.markdown('<p class="av-section-title">Prediction Result</p>', unsafe_allow_html=True)
            confidence_pill(confidence)

            c1, c2, c3 = st.columns(3)
            c1.metric("🌱 Plant", result["plant"])
            c2.metric("🦠 Disease", result["disease"])
            c3.metric("🎯 Confidence", f"{confidence:.1f}%")

            st.progress(min(confidence / 100, 1.0))

            if confidence < 50:
                st.info(
                    "The model is uncertain about this prediction. For a clearer "
                    "result, upload a well-lit photo showing a single leaf against "
                    "a plain background."
                )

    st.write("")

    # ------------------------------------------------------
    # Tabbed detail views
    # ------------------------------------------------------
    tab_overview, tab_details, tab_recommendation = st.tabs(
        ["📊 Top Predictions", "📖 Disease Details", "🤖 Recommendation"]
    )

    # ---------------- Top Predictions ----------------
    with tab_overview:
        st.markdown('<p class="av-section-title">Top 3 Predictions</p>', unsafe_allow_html=True)
        rows = [
            {
                "Plant": item["plant"],
                "Disease": item["disease"],
                "Confidence (%)": item["confidence"],
            }
            for item in result["top3"]
        ]
        df = pd.DataFrame(rows)
        st.dataframe(
            df,
            hide_index=True,
            use_container_width=True,
            column_config={
                "Confidence (%)": st.column_config.ProgressColumn(
                    "Confidence (%)", min_value=0, max_value=100, format="%.1f%%"
                )
            },
        )

    # ---------------- Disease Details ----------------
    with tab_details:
        if info:
            st.markdown('<p class="av-section-title">Disease Information</p>', unsafe_allow_html=True)

            desc_col, sev_col = st.columns([3, 1])
            with desc_col:
                st.info(info.get("description", "No description available."))
            with sev_col:
                st.metric("⚠ Severity", info.get("severity", "Unknown"))

            col1, col2 = st.columns(2, gap="large")

            with col1:
                with st.container(border=True):
                    st.markdown("**⚠ Symptoms**")
                    render_list(info.get("symptoms", []), "No specific symptoms listed.")

                with st.container(border=True):
                    st.markdown("**🦠 Causes**")
                    render_list(info.get("causes", []), "No specific causes listed.")

                with st.container(border=True):
                    st.markdown("**🌿 Organic Treatment**")
                    render_list(info.get("organic_treatment", []), "No organic treatment information available.")

            with col2:
                with st.container(border=True):
                    st.markdown("**🧪 Chemical Treatment**")
                    render_list(info.get("chemical_treatment", []), "No chemical treatment information available.")

                with st.container(border=True):
                    st.markdown("**🛡 Prevention**")
                    render_list(info.get("prevention", []), "No prevention information available.")
        else:
            st.warning("Disease information is not available for this prediction.")

    # ---------------- AI Recommendation ----------------
    with tab_recommendation:
        st.markdown('<p class="av-section-title">AI Recommendation</p>', unsafe_allow_html=True)
        if result["disease"].lower() == "healthy":
            st.success(recommendation)
        else:
            st.warning(recommendation)


# ==========================================================
# Footer
# ==========================================================

st.divider()
st.markdown(
    """
    <div class="av-footer">
        🌿 <b>AgroVision AI</b> &nbsp;·&nbsp; Built with TensorFlow · EfficientNetB0 · Streamlit
    </div>
    """,
    unsafe_allow_html=True,
)