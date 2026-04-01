import streamlit as st
import cv2
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# ==========================================
# PAGE CONFIG
# ==========================================
st.set_page_config(
    page_title="Saliva Glucose Monitoring Platform",
    layout="wide"
)

st.title("Saliva Glucose Monitoring Platform")

# ==========================================
# SESSION STATE FOR HISTORY
# ==========================================
if "history" not in st.session_state:
    st.session_state.history = []

# ==========================================
# RGB TO HSV
# ==========================================
def rgb_to_hsv(rgb):
    rgb = np.array(rgb)
    maxc = rgb.max(axis=1)
    minc = rgb.min(axis=1)
    v = maxc
    s = (maxc - minc) / (maxc + 1e-6)
    s[maxc == 0] = 0

    rc = (maxc - rgb[:, 0]) / (maxc - minc + 1e-6)
    gc = (maxc - rgb[:, 1]) / (maxc - minc + 1e-6)
    bc = (maxc - rgb[:, 2]) / (maxc - minc + 1e-6)

    h = np.zeros_like(maxc)

    mask = maxc == rgb[:, 0]
    h[mask] = (bc - gc)[mask]

    mask = maxc == rgb[:, 1]
    h[mask] = 2.0 + (rc - bc)[mask]

    mask = maxc == rgb[:, 2]
    h[mask] = 4.0 + (gc - rc)[mask]

    h = (h / 6.0) % 1.0
    h[minc == maxc] = 0.0

    return np.stack([h, s, v], axis=1)

# ==========================================
# FEATURE EXTRACTION
# ==========================================
def extract_bubble_features(image_path, top_n=20):
    img = cv2.imread(image_path)

    if img is None:
        raise ValueError("Cannot read image.")

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.GaussianBlur(img_gray, (3, 3), 0)

    circles = cv2.HoughCircles(
        img_gray,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=20,
        param1=50,
        param2=35,
        minRadius=5,
        maxRadius=50
    )

    if circles is None:
        raise ValueError("No bubbles detected.")

    circles = np.around(circles).astype(int)

    candidates = []

    for x, y, r in circles[0]:
        r = int(r * 0.9)

        Y, X = np.ogrid[:img_rgb.shape[0], :img_rgb.shape[1]]
        mask = (X - x)**2 + (Y - y)**2 <= r**2

        roi_rgb = img_rgb[mask]
        roi_hsv = rgb_to_hsv(roi_rgb / 255.0)

        h_mean, s_mean, v_mean = roi_hsv.mean(axis=0)

        if 252/360 <= h_mean <= 290/360 and s_mean >= 0.04 and v_mean >= 0.60:
            score = (h_mean**8) * s_mean * v_mean * r
            candidates.append({"roi_hsv": roi_hsv, "score": score})

    if len(candidates) == 0:
        raise ValueError("No valid bubbles detected.")

    candidates = sorted(
        candidates,
        key=lambda b: b["score"],
        reverse=True
    )[:top_n]

    avg_hsv = np.mean(
        [b["roi_hsv"].mean(axis=0) for b in candidates],
        axis=0
    )

    return avg_hsv, img_rgb

# ==========================================
# MODEL TRAINER
# ==========================================
def train_model(glucose_range):
    calibration_data = pd.DataFrame({
        "Glucose": glucose_range,
        "H": [0.722795, 0.731712, 0.730700, 0.743624, 0.786134],
        "S": [0.086949, 0.093759, 0.097361, 0.107223, 0.121588]
    })

    y = calibration_data["Glucose"].values

    model_H = LinearRegression().fit(calibration_data[["H"]], y)
    model_S = LinearRegression().fit(calibration_data[["S"]], y)
    model_HS = LinearRegression().fit(calibration_data[["H", "S"]], y)

    return model_H, model_S, model_HS

# ==========================================
# PREDICTION FUNCTION
# ==========================================
def predict_glucose(uploaded_file, glucose_range, reagent_name):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    temp_path = f"temp_{timestamp}.jpg"

    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    avg_hsv, img_rgb = extract_bubble_features(temp_path)

    H_avg, S_avg, _ = avg_hsv

    model_H, model_S, model_HS = train_model(glucose_range)

    g_H = max(model_H.predict(pd.DataFrame({"H": [H_avg]}))[0], 0)
    g_S = max(model_S.predict(pd.DataFrame({"S": [S_avg]}))[0], 0)
    g_HS = max(
        model_HS.predict(pd.DataFrame({"H": [H_avg], "S": [S_avg]}))[0],
        0
    )

    glucose = 0.5*g_H + 0.3*g_S + 0.2*g_HS

    st.image(img_rgb, caption="Uploaded Image", use_container_width=True)

    st.metric("Estimated Saliva Glucose", f"{glucose:.1f} µM")

    if glucose < 150:
        st.success("Within expected healthy saliva glucose range.")
    else:
        st.warning("Elevated glucose detected. Please consult a healthcare professional.")

    st.caption("This application is for research screening use only and not for diagnosis.")

    st.session_state.history.append({
        "Time": datetime.now(),
        "Reagent": reagent_name,
        "Glucose": glucose
    })

# ==========================================
# TABS
# ==========================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Home",
    "Reagent Set A",
    "Reagent Set D",
    "Historical Results",
    "Tips & Tricks"
])

# ==========================================
# TAB 1 HOME
# ==========================================
with tab1:
    st.header("About Diabetes")
    st.write("""
    Diabetes mellitus is a chronic metabolic disorder characterized by elevated glucose levels due to insufficient insulin production or reduced insulin sensitivity.

    This application estimates saliva glucose concentration using microfluidic bubble-based colorimetric analysis.
    """)

    st.subheader("Workflow")
    st.write("""
    1. Choose reagent set
    2. Upload microfluidic bubble image
    3. Automatic bubble detection
    4. HSV color analysis
    5. Glucose estimation
    6. View interpretation and history
    """)

# ==========================================
# TAB 2 REAGENT A
# ==========================================
with tab2:
    st.header("Reagent Set A (Non-Diabetic Range)")
    file_A = st.file_uploader("Upload image", type=["jpg", "png", "jpeg"], key="A")

    if file_A:
        predict_glucose(file_A, [25, 50, 75, 100, 125], "Set A")

# ==========================================
# TAB 3 REAGENT D
# ==========================================
with tab3:
    st.header("Reagent Set D (Diabetic Range)")
    file_D = st.file_uploader("Upload image", type=["jpg", "png", "jpeg"], key="D")

    if file_D:
        predict_glucose(file_D, [150, 200, 250, 300, 350], "Set D")

# ==========================================
# TAB 4 HISTORY
# ==========================================
with tab4:
    st.header("Historical Results Log")

    if len(st.session_state.history) > 0:
        df = pd.DataFrame(st.session_state.history)

        st.dataframe(df)

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(df["Time"], df["Glucose"], marker="o")
        ax.set_ylabel("Glucose (µM)")
        ax.set_xlabel("Time")

        st.pyplot(fig)
    else:
        st.info("No historical data available yet.")

# ==========================================
# TAB 5 TIPS
# ==========================================
with tab5:
    st.header("Tips & Tricks to Manage Diabetes")

    st.write("""
    - Maintain a balanced diet with controlled sugar intake
    - Exercise regularly
    - Stay hydrated
    - Monitor glucose consistently
    - Follow prescribed medication
    - Schedule regular medical reviews
    """)
