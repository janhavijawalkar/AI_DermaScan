import streamlit as st
import numpy as np
import cv2
import pandas as pd
from backend import process_image
from datetime import datetime
import time

#  PAGE CONFIG 
st.set_page_config(
    page_title="DermalScan",
    page_icon="🧴",
    layout="wide"
)

#  APPLE-STYLE CSS 
st.markdown("""
<style>

/* Background */
html, body, [data-testid="stAppViewContainer"] {
    background: radial-gradient(circle at top, #1f2a55, #0b1025);
    font-family: -apple-system, BlinkMacSystemFont, "Helvetica Neue",
                 Arial, sans-serif;
}

/* Header */
.main-title {
    text-align: center;
    font-size: 44px;
    font-weight: 700;
    color: white;
}
.subtitle {
    text-align: center;
    font-size: 18px;
    color: #c7d2fe;
    margin-bottom: 35px;
}

/* Apple-style card */
.apple-card {
    background: rgba(255,255,255,0.12);
    backdrop-filter: blur(18px);
    border-radius: 22px;
    padding: 24px;
    margin-bottom: 28px;
    border: 1px solid rgba(255,255,255,0.18);
    box-shadow: 0 20px 40px rgba(0,0,0,0.4);
}

/* Titles */
.apple-title {
    font-size: 22px;
    font-weight: 600;
    color: white;
    margin-bottom: 14px;
}

/* Text */
.apple-text {
    color: #e5e7eb;
    font-size: 17px;
}

/* Badge */
.apple-badge {
    display: inline-block;
    padding: 10px 16px;
    border-radius: 16px;
    background: rgba(255,255,255,0.2);
    color: white;
    font-weight: 600;
    font-size: 18px;
}

/* Confidence pill */
.apple-pill {
    margin-top: 12px;
    padding: 10px 18px;
    border-radius: 18px;
    background: linear-gradient(135deg, #4f8cff, #7aa7ff);
    color: white;
    font-weight: 600;
    display: inline-block;
}

/* FORCE WIDGET TEXT WHITE */
[data-testid="stRadio"] *,
[data-testid="stFileUploader"] *,
[data-testid="stCameraInput"] *,
label {
    color: white !important;
}

</style>
""", unsafe_allow_html=True)

#  HEADER 
st.markdown("<div class='main-title'>🧴 DermalScan</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='subtitle'>AI-Powered Facial Skin Condition Analysis</div>",
    unsafe_allow_html=True
)

#  INPUT SOURCE 
st.markdown("<div class='apple-card'>", unsafe_allow_html=True)
st.markdown("<div class='apple-title'>📸 Choose Input Source</div>", unsafe_allow_html=True)

source = st.radio(
    "Select image source",
    ["Upload Image", "Use Live Camera"],
    horizontal=True
)

st.markdown("</div>", unsafe_allow_html=True)

img = None

#  UPLOAD IMAGE 
if source == "Upload Image":
    uploaded = st.file_uploader(
        "Upload a facial image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded:
        file_bytes = np.frombuffer(uploaded.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

#  LIVE CAMERA
else:
    camera_img = st.camera_input("Capture image using camera")
    if camera_img:
        file_bytes = np.frombuffer(camera_img.getvalue(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

# PROCESS IMAGE 
if img is not None:

    col1, col2 = st.columns(2)

    # ORIGINAL IMAGE
    with col1:
        st.markdown("<div class='apple-card'>", unsafe_allow_html=True)
        st.markdown("<div class='apple-title'>Input Image</div>", unsafe_allow_html=True)
        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        st.markdown("</div>", unsafe_allow_html=True)

    # MODEL PROCESSING
    start_time = time.time()
    annotated_img, results = process_image(img)
    processing_time = time.time() - start_time

    # OUTPUT IMAGE
    with col2:
        st.markdown("<div class='apple-card'>", unsafe_allow_html=True)
        st.markdown("<div class='apple-title'>Analysis Output</div>", unsafe_allow_html=True)
        st.image(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB))
        st.markdown("</div>", unsafe_allow_html=True)

    # RESULTS
    if results:
        r = results[0]

        # CONDITION
        st.markdown("<div class='apple-card'>", unsafe_allow_html=True)
        st.markdown("<div class='apple-title'>Detected Skin Condition</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='apple-badge'>🧴 {r['label']}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='apple-pill'>Confidence: {r['confidence']:.2f}%</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # PROBABILITIES
        st.markdown("<div class='apple-card'>", unsafe_allow_html=True)
        st.markdown("<div class='apple-title'>Probability Distribution</div>", unsafe_allow_html=True)

        for k, v in r["probabilities"].items():
            st.markdown(f"<div class='apple-text'><b>{k}</b>: {v:.1f}%</div>", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

        # PROCESSING TIME
        st.markdown("<div class='apple-card'>", unsafe_allow_html=True)
        st.markdown("<div class='apple-title'>Processing Time</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='apple-text'>⏱ {processing_time:.2f} seconds</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # DOWNLOAD
        st.markdown("<div class='apple-card'>", unsafe_allow_html=True)
        st.markdown("<div class='apple-title'>Download Results</div>", unsafe_allow_html=True)

        df = pd.DataFrame(results)
        df["probabilities"] = df["probabilities"].apply(str)

        st.download_button(
            "📄 Download CSV Report",
            df.to_csv(index=False).encode("utf-8"),
            file_name=f"dermalscan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

        _, png_img = cv2.imencode(".png", annotated_img)
        st.download_button(
            "🖼 Download Annotated Image",
            png_img.tobytes(),
            file_name=f"dermalscan_result.png",
            mime="image/png"
        )

        st.markdown("</div>", unsafe_allow_html=True)

else:
    st.info("⬆ Please upload or capture an image to start analysis")
