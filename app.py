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
    page_title="AgroVision AI",
    page_icon="🌿",
    layout="wide"
)


# ==========================================================
# Load Disease Database
# ==========================================================

with open(
    "disease_database.json",
    "r",
    encoding="utf-8"
) as file:

    DISEASE_DATABASE = json.load(file)


# ==========================================================
# Create Temp Folder
# ==========================================================

os.makedirs(
    "temp",
    exist_ok=True
)


# ==========================================================
# Custom CSS
# ==========================================================

st.markdown(
    """
    <style>

    .main-title {
        text-align: center;
        font-size: 42px;
        color: #2E8B57;
        font-weight: bold;
    }

    .sub-title {
        text-align: center;
        color: gray;
        font-size: 18px;
        margin-bottom: 30px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ==========================================================
# Header
# ==========================================================

st.markdown(
    "<div class='main-title'>🌿 AgroVision AI</div>",
    unsafe_allow_html=True
)

st.markdown(
    "<div class='sub-title'>AI Powered Plant Disease Detection System</div>",
    unsafe_allow_html=True
)

st.divider()


# ==========================================================
# Upload Section
# ==========================================================

left, right = st.columns([1, 2])


uploaded_file = left.file_uploader(
    "📷 Upload Plant Leaf Image",
    type=[
        "jpg",
        "jpeg",
        "png"
    ]
)


# ==========================================================
# Process Uploaded Image
# ==========================================================

if uploaded_file is not None:

    # ------------------------------------------------------
    # Load Image
    # ------------------------------------------------------

    image = Image.open(uploaded_file)

    left.image(
        image,
        caption="Uploaded Plant Leaf",
        use_container_width=True
    )


    # ------------------------------------------------------
    # Save Original Uploaded Image
    # ------------------------------------------------------
    #
    # We preserve the original uploaded file instead of
    # re-encoding it as JPEG.
    # ------------------------------------------------------

    original_filename = os.path.basename(
        uploaded_file.name
    )

    image_path = os.path.join(
        "temp",
        original_filename
    )

    with open(
        image_path,
        "wb"
    ) as file:

        file.write(
            uploaded_file.getbuffer()
        )


    # ======================================================
    # Prediction
    # ======================================================

    result = predict_image(
        image_path
    )


    # ======================================================
    # Find Disease Information
    # ======================================================

    info = DISEASE_DATABASE.get(
        result["prediction"],
        {}
    )


    # ======================================================
    # Generate Recommendation
    # ======================================================

    recommendation = generate_recommendation(
        result,
        info
    )


    # ======================================================
    # Prediction Result
    # ======================================================

    right.success(
        "Prediction Completed"
    )


    c1, c2, c3 = right.columns(3)


    # ------------------------------------------------------
    # Plant
    # ------------------------------------------------------

    c1.metric(
        "🌱 Plant",
        result["plant"]
    )


    # ------------------------------------------------------
    # Disease
    # ------------------------------------------------------

    c2.metric(
        "🦠 Disease",
        result["disease"]
    )


    # ------------------------------------------------------
    # Confidence
    # ------------------------------------------------------

    confidence = result["confidence"]

    c3.metric(
        "🎯 Confidence",
        f"{confidence:.2f}%"
    )


    # ------------------------------------------------------
    # Confidence Progress Bar
    # ------------------------------------------------------

    right.progress(
        min(
            confidence / 100,
            1.0
        )
    )


    # ======================================================
    # Confidence Assessment
    # ======================================================

    if confidence >= 80:

        st.success(
            f"🟢 High Confidence Prediction — "
            f"{confidence:.2f}%"
        )

    elif confidence >= 50:

        st.warning(
            f"🟡 Moderate Confidence Prediction — "
            f"{confidence:.2f}%"
        )

    else:

        st.error(
            f"🔴 Low Confidence Prediction — "
            f"{confidence:.2f}%"
        )

        st.info(
            "The model is uncertain about this prediction. "
            "For better results, upload a clear image showing "
            "a single plant leaf with good lighting."
        )


    # ======================================================
    # Top 3 Predictions
    # ======================================================

    st.divider()

    st.subheader(
        "📊 Top 3 Predictions"
    )


    rows = []


    for item in result["top3"]:

        rows.append(
            {
                "Plant": item["plant"],
                "Disease": item["disease"],
                "Confidence (%)": item["confidence"]
            }
        )


    df = pd.DataFrame(
        rows
    )


    st.dataframe(
        df,
        hide_index=True,
        use_container_width=True
    )


    # ======================================================
    # Disease Information
    # ======================================================

    st.divider()


    if info:

        st.header(
            "📖 Disease Information"
        )


        # --------------------------------------------------
        # Description
        # --------------------------------------------------

        st.info(
            info.get(
                "description",
                "No description available."
            )
        )


        # --------------------------------------------------
        # Severity
        # --------------------------------------------------

        severity = info.get(
            "severity",
            "Unknown"
        )


        st.metric(
            "⚠ Severity",
            severity
        )


        # ==================================================
        # Disease Details
        # ==================================================

        col1, col2 = st.columns(2)


        # ==================================================
        # Left Column
        # ==================================================

        with col1:

            # --------------------------------------------------
            # Symptoms
            # --------------------------------------------------

            st.subheader(
                "⚠ Symptoms"
            )


            symptoms = info.get(
                "symptoms",
                []
            )


            if symptoms:

                for symptom in symptoms:

                    st.write(
                        "•",
                        symptom
                    )

            else:

                st.write(
                    "No specific symptoms listed."
                )


            # --------------------------------------------------
            # Causes
            # --------------------------------------------------

            st.subheader(
                "🦠 Causes"
            )


            causes = info.get(
                "causes",
                []
            )


            if causes:

                for cause in causes:

                    st.write(
                        "•",
                        cause
                    )

            else:

                st.write(
                    "No specific causes listed."
                )


            # --------------------------------------------------
            # Organic Treatment
            # --------------------------------------------------

            st.subheader(
                "🌿 Organic Treatment"
            )


            organic_treatment = info.get(
                "organic_treatment",
                []
            )


            if organic_treatment:

                for treatment in organic_treatment:

                    st.write(
                        "•",
                        treatment
                    )

            else:

                st.write(
                    "No organic treatment information available."
                )


        # ==================================================
        # Right Column
        # ==================================================

        with col2:

            # --------------------------------------------------
            # Chemical Treatment
            # --------------------------------------------------

            st.subheader(
                "🧪 Chemical Treatment"
            )


            chemical_treatment = info.get(
                "chemical_treatment",
                []
            )


            if chemical_treatment:

                for treatment in chemical_treatment:

                    st.write(
                        "•",
                        treatment
                    )

            else:

                st.write(
                    "No chemical treatment information available."
                )


            # --------------------------------------------------
            # Prevention
            # --------------------------------------------------

            st.subheader(
                "🛡 Prevention"
            )


            prevention = info.get(
                "prevention",
                []
            )


            if prevention:

                for item in prevention:

                    st.write(
                        "•",
                        item
                    )

            else:

                st.write(
                    "No prevention information available."
                )


    else:

        st.warning(
            "Disease information is not available "
            "for this prediction."
        )


    # ======================================================
    # AI Recommendation
    # ======================================================

    st.divider()

    st.header(
        "🤖 AI Recommendation"
    )


    if result["disease"].lower() == "healthy":

        st.success(
            recommendation
        )

    else:

        st.warning(
            recommendation
        )


# ==========================================================
# Footer
# ==========================================================

st.divider()


st.markdown(
    """
    <div style="text-align:center; color:gray;">

    🌿 <b>AgroVision AI</b>

    <br><br>

    Built with TensorFlow • EfficientNetB0 • Streamlit

    </div>
    """,
    unsafe_allow_html=True
)
