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
    page_title="AgroVision AI — Plant Health Intelligence",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ==========================================================
# Load Disease Database & Ensure Temp Directory
# ==========================================================

@st.cache_data
def load_database():
    with open("disease_database.json", "r", encoding="utf-8") as file:
        return json.load(file)

DISEASE_DATABASE = load_database()

os.makedirs("temp", exist_ok=True)


# ==========================================================
# Inject High-End Modern CSS
# ==========================================================

st.markdown(
    """
    <style>
    /* Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Outfit:wght@600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    #MainMenu, footer, header { visibility: hidden; }

    .block-container {
        padding-top: 1.2rem !important;
        padding-bottom: 2rem !important;
        max-width: 1250px !important;
    }

    /* Top Brand Ribbon */
    .nav-bar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        background: linear-gradient(90deg, #0F1D17 0%, #152A22 100%);
        border: 1px solid #234235;
        border-radius: 14px;
        padding: 14px 28px;
        margin-bottom: 1.2rem;
        box-shadow: 0 8px 24px rgba(0,0,0,0.3);
    }
    .nav-brand {
        font-family: 'Outfit', sans-serif;
        font-size: 22px;
        font-weight: 800;
        color: #ECFDF5;
        letter-spacing: -0.3px;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .nav-badge {
        background: rgba(16, 185, 129, 0.15);
        border: 1px solid rgba(16, 185, 129, 0.35);
        color: #34D399;
        font-size: 12px;
        font-weight: 600;
        padding: 5px 14px;
        border-radius: 20px;
    }

    /* Hero Card */
    .hero-card {
        background: linear-gradient(135deg, #12241D 0%, #0A1410 100%);
        border: 1px solid #244437;
        border-radius: 16px;
        padding: 1.8rem 2.2rem;
        margin-bottom: 1.5rem;
    }
    .status-pill {
        display: inline-block;
        background: rgba(16, 185, 129, 0.15);
        border: 1px solid rgba(16, 185, 129, 0.3);
        color: #34D399;
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 1px;
        padding: 4px 12px;
        border-radius: 20px;
        margin-bottom: 10px;
        text-transform: uppercase;
    }
    .hero-title {
        font-family: 'Outfit', sans-serif;
        font-size: 34px;
        font-weight: 800;
        color: #FFFFFF;
        margin-bottom: 6px;
    }
    .hero-subtitle {
        font-size: 14px;
        color: #94A3B8;
        margin: 0;
    }

    /* Styled Navigation Ribbon Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #0E1A15;
        padding: 8px;
        border-radius: 12px;
        border: 1px solid #1E382D;
    }
    .stTabs [data-baseweb="tab"] {
        height: 44px;
        border-radius: 8px;
        color: #94A3B8;
        font-weight: 600;
        font-size: 14px;
        background-color: transparent;
        padding: 0px 20px;
        border: none !important;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1A3328 !important;
        color: #34D399 !important;
        border: 1px solid rgba(52, 211, 153, 0.3) !important;
        box-shadow: 0 4px 12px rgba(0,0,0,0.2);
    }

    /* Fixed Image Container Constraint */
    .img-preview-box {
        max-width: 320px;
        margin: 0 auto;
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid #28473B;
        box-shadow: 0 8px 20px rgba(0,0,0,0.4);
    }

    /* Section Cards */
    .agri-card {
        background: #112019;
        border: 1px solid #213B30;
        border-radius: 14px;
        padding: 1.25rem;
        margin-bottom: 1rem;
    }

    /* Diagnostic Metric Cards */
    .metric-card {
        background: #162B22;
        border: 1px solid #2A4C3D;
        border-radius: 12px;
        padding: 1.1rem;
        text-align: left;
    }
    .metric-label {
        font-size: 11px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        color: #94A3B8;
        margin-bottom: 4px;
    }
    .metric-value {
        font-family: 'Outfit', sans-serif;
        font-size: 22px;
        font-weight: 700;
        color: #FFFFFF;
    }

    /* List Items */
    .list-item {
        color: #CBD5E1;
        font-size: 14px;
        margin-bottom: 8px;
        padding: 8px 12px;
        background: rgba(0,0,0,0.2);
        border-radius: 8px;
        border-left: 3px solid #10B981;
    }

    /* AI Advice Box */
    .ai-rec-card {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.1) 0%, #112019 100%);
        border: 1px solid rgba(16, 185, 129, 0.4);
        border-radius: 14px;
        padding: 1.8rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ==========================================================
# Brand Header & Hero Section
# ==========================================================

st.markdown(
    """
    <div class="nav-bar">
        <div class="nav-brand">🌿 AgroVision AI</div>
        <div class="nav-badge">AI Powered • Computer Vision</div>
    </div>
    <div class="hero-card">
        <div class="status-pill">● AI SYSTEM ONLINE</div>
        <div class="hero-title">AI-Powered Plant Health Intelligence</div>
        <div class="hero-subtitle">Detect plant leaf pathology instantly and access actionable treatment advice.</div>
    </div>
    """,
    unsafe_allow_html=True
)


# ==========================================================
# File Upload Area
# ==========================================================

left, right = st.columns([1, 1.3], gap="large")

with left:
    st.markdown("### 📷 Upload Leaf Image")
    uploaded_file = st.file_uploader(
        "Choose a clear leaf photo (JPG, JPEG, PNG)",
        type=["jpg", "jpeg", "png"]
    )


# ==========================================================
# Execution & Interactive Tabs
# ==========================================================

if uploaded_file is not None:

    # 1. Image Preview with Constraint (Prevents image from blowing up large)
    image = Image.open(uploaded_file)
    
    with left:
        st.markdown('<div class="img-preview-box">', unsafe_allow_html=True)
        st.image(image, caption="Uploaded Leaf Preview", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # 2. Save Original Uploaded File (Preserving exact backend requirements)
    original_filename = os.path.basename(uploaded_file.name)
    image_path = os.path.join("temp", original_filename)

    with open(image_path, "wb") as file:
        file.write(uploaded_file.getbuffer())

    # 3. Predict & Fetch Data
    with st.spinner("Analyzing plant leaf..."):
        result = predict_image(image_path)
        info = DISEASE_DATABASE.get(result["prediction"], {})
        recommendation = generate_recommendation(result, info)

    # 4. Top Ribbon Navigation Tabs
    with right:
        tab_diag, tab_top3, tab_info, tab_ai = st.tabs([
            "🔍 Diagnosis",
            "📊 Predictions",
            "📖 Treatment & Info",
            "✨ AI Advisor"
        ])

        # TAB 1: DIAGNOSIS
        with tab_diag:
            st.write("")
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown(f'<div class="metric-card"><div class="metric-label">🌱 Plant</div><div class="metric-value">{result["plant"]}</div></div>', unsafe_allow_html=True)
            with c2:
                st.markdown(f'<div class="metric-card"><div class="metric-label">🦠 Disease</div><div class="metric-value">{result["disease"]}</div></div>', unsafe_allow_html=True)
            with c3:
                confidence = result["confidence"]
                st.markdown(f'<div class="metric-card"><div class="metric-label">🎯 Confidence</div><div class="metric-value">{confidence:.2f}%</div></div>', unsafe_allow_html=True)

            st.write("")
            st.progress(min(confidence / 100, 1.0))

            if confidence >= 80:
                st.success(f"🟢 High Confidence Prediction — {confidence:.2f}%")
            elif confidence >= 50:
                st.warning(f"🟡 Moderate Confidence Prediction — {confidence:.2f}%")
            else:
                st.error(f"🔴 Low Confidence Prediction — {confidence:.2f}%")
                st.info("The model is uncertain. Upload a clearer leaf image with good lighting.")

        # TAB 2: ALTERNATIVE PREDICTIONS
        with tab_top3:
            st.write("")
            st.markdown("#### Top 3 Probable Conditions")
            rows = []
            for item in result["top3"]:
                rows.append({
                    "Plant": item["plant"],
                    "Disease": item["disease"],
                    "Confidence": f"{item['confidence']:.2f}%"
                })
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

        # TAB 3: DISEASE INFO & TREATMENTS
        with tab_info:
            st.write("")
            if info:
                st.markdown(f'<div class="agri-card"><div style="font-weight:700; color:#FFF; margin-bottom:4px;">Overview</div><div style="color:#CBD5E1; font-size:14px;">{info.get("description", "N/A")}</div></div>', unsafe_allow_html=True)
                
                severity = info.get("severity", "Unknown")
                st.metric("⚠ Severity Level", severity)

                col_a, col_b = st.columns(2)
                with col_a:
                    st.markdown("##### ⚠ Symptoms")
                    for s in info.get("symptoms", ["No symptoms listed."]):
                        st.markdown(f'<div class="list-item">{s}</div>', unsafe_allow_html=True)
                    
                    st.markdown("##### 🌿 Organic Treatment")
                    for ot in info.get("organic_treatment", ["No organic treatment listed."]):
                        st.markdown(f'<div class="list-item">{ot}</div>', unsafe_allow_html=True)

                with col_b:
                    st.markdown("##### 🧪 Chemical Treatment")
                    for ct in info.get("chemical_treatment", ["No chemical treatment listed."]):
                        st.markdown(f'<div class="list-item">{ct}</div>', unsafe_allow_html=True)

                    st.markdown("##### 🛡 Prevention")
                    for p in info.get("prevention", ["No prevention listed."]):
                        st.markdown(f'<div class="list-item">{p}</div>', unsafe_allow_html=True)
            else:
                st.warning("No detailed disease database entry found for this condition.")

        # TAB 4: AI ADVISOR
        with tab_ai:
            st.write("")
            st.markdown(
                f"""
                <div class="ai-rec-card">
                    <div style="font-weight: 800; font-size: 18px; color: #FFFFFF; margin-bottom: 8px;">✨ Automated Action Plan</div>
                    <div style="color: #E2E8F0; font-size: 14px; line-height: 1.6;">{recommendation}</div>
                </div>
                """,
                unsafe_allow_html=True
            )


# ==========================================================
# Footer
# ==========================================================

st.divider()
st.markdown(
    """
    <div style="text-align:center; color: #64748B; font-size: 12px;">
        🌿 <b>AgroVision AI</b> • Precision AgriTech Platform
    </div>
    """,
    unsafe_allow_html=True
)