import streamlit as st
import numpy as np
import cv2
import pandas as pd
from backend import process_image
from datetime import datetime
import time

# PAGE CONFIG
st.set_page_config(
    page_title="DermalScan",
    page_icon="🧴",
    layout="wide"
)

# CSS 
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&display=swap');

html, body, [data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #1b1f4b, #2c2f6c);
    font-family: 'Poppins', sans-serif;
}

/* Titles */
.main-title {
    text-align: center;
    font-size: 44px;
    font-weight: 700;
    color: #ffffff;
}

.subtitle {
    text-align: center;
    font-size: 17px;
    color: #c7d2fe;
    margin-bottom: 30px;
}

/* Cards */
.card {
    background: rgba(255,255,255,0.12);
    backdrop-filter: blur(12px);
    border-radius: 16px;
    padding: 20px;
    margin-bottom: 25px;
    box-shadow: 0 8px 30px rgba(0,0,0,0.25);
}

/* Section headers */
.section-title {
    color: #ffffff;
    font-size: 22px;
    font-weight: 600;
    margin-bottom: 15px;
}

/* Probability text */
.prob-text {
    color: #e0f2fe;
    font-size: 18px;
    font-weight: 500;
}

/* Fix metric colors if used */
[data-testid="stMetricLabel"] {
    color: #c7d2fe !important;
}
[data-testid="stMetricValue"] {
    color: #ffffff !important;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

# HEADER
st.markdown("<div class='main-title'>🧴 DermalScan</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='subtitle'>AI-Powered Facial Skin Condition Analysis</div>",
    unsafe_allow_html=True
)

# IMAGE RESIZE
def resize_for_ui(image, max_height=820):
    h, w = image.shape[:2]
    if h > max_height:
        scale = max_height / h
        new_w = int(w * scale)
        new_h = int(h * scale)
        image = cv2.resize(image, (new_w, new_h))
    return image

# FILE UPLOAD
uploaded = st.file_uploader(
    "Upload a facial image",
    type=["jpg", "jpeg", "png"]
)

if uploaded:
    file_bytes = np.frombuffer(uploaded.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    col1, col2 = st.columns(2)

    # ---------------- ORIGINAL IMAGE ----------------
    with col1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>Uploaded Image</div>", unsafe_allow_html=True)

        ui_img = resize_for_ui(img)

        st.markdown("<div style='display:flex; justify-content:center;'>", unsafe_allow_html=True)
        st.image(cv2.cvtColor(ui_img, cv2.COLOR_BGR2RGB))
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

    # ---------------- PROCESS IMAGE ----------------
    start_time = time.time()
    annotated_img, results = process_image(img)
    processing_time = time.time() - start_time

    # ---------------- ANNOTATED IMAGE ----------------
    with col2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>Annotated Output</div>", unsafe_allow_html=True)

        ui_annotated = resize_for_ui(annotated_img)

        st.markdown("<div style='display:flex; justify-content:center;'>", unsafe_allow_html=True)
        st.image(cv2.cvtColor(ui_annotated, cv2.COLOR_BGR2RGB))
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

    
    # RESULTS
    if results and len(results) > 0:
        r = results[0]

        # ---------------- DETECTED CONDITION ----------------
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>Detected Skin Condition</div>", unsafe_allow_html=True)

        st.markdown(
            f"""
            <div style="
                background: rgba(0,0,0,0.35);
                padding: 16px;
                border-radius: 12px;
                color: #ffffff;
                font-size: 20px;
                font-weight: 600;
            ">
                🧴 {r['label']}
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            f"""
            <div style="
                margin-top: 12px;
                background: rgba(255,255,255,0.12);
                padding: 14px;
                border-radius: 12px;
                color: #ffffff;
                font-size: 18px;
            ">
                📊 Confidence: <b>{r['confidence']:.2f}%</b>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("</div>", unsafe_allow_html=True)

        # ---------------- PROBABILITY DISTRIBUTION ----------------
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>📈 Probability Distribution</div>", unsafe_allow_html=True)

        probs = r.get("probabilities", {})

        pcol1, pcol2 = st.columns(2)
        items = list(probs.items())

        for i, (name, val) in enumerate(items):
            block = f"<div class='prob-text'><b>{name}</b>: {val:.1f}%</div>"
            if i % 2 == 0:
                pcol1.markdown(block, unsafe_allow_html=True)
            else:
                pcol2.markdown(block, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

        # ---------------- PROCESSING TIME ----------------
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>⏱ Processing Time</div>", unsafe_allow_html=True)

        st.markdown(
            f"""
            <div style="
                background: rgba(0,0,0,0.35);
                padding: 16px;
                border-radius: 12px;
                color: #ffffff;
                font-size: 18px;
                font-weight: 500;
                text-align: center;
            ">
                Image analyzed in <b>{processing_time:.2f} seconds</b>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("</div>", unsafe_allow_html=True)

        # ---------------- DOWNLOAD SECTION ----------------
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>⬇ Download Results</div>", unsafe_allow_html=True)

        dcol1, dcol2 = st.columns(2)

        with dcol1:
            df = pd.DataFrame(results)
            df["probabilities"] = df["probabilities"].apply(str)
            csv_bytes = df.to_csv(index=False).encode("utf-8")

            st.download_button(
                label="📄 Download CSV Report",
                data=csv_bytes,
                file_name=f"dermalscan_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )

        with dcol2:
            _, png_img = cv2.imencode(".png", annotated_img)
            st.download_button(
                label="🖼 Download Annotated Image",
                data=png_img.tobytes(),
                file_name=f"dermalscan_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                mime="image/png"
            )

        st.markdown("</div>", unsafe_allow_html=True)

    else:
        st.warning("No face detected. Please upload a clear facial image.")

else:
    st.info("⬆ Upload an image to start analysis")
