import streamlit as st
import cv2
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.linear_model import LinearRegression

# ==========================================
# PAGE CONFIG + THEME
# ==========================================
st.set_page_config(page_title="Saliva Glucose Monitoring Platform", layout="wide")

st.markdown("""
<style>
.stApp {
    background: linear-gradient(180deg, #f3fff3 0%, #e8fbe8 100%);
}
div[data-testid='stMetric'] {
    background-color: #e6ffe6;
    border: 1px solid #b6e3b6;
    padding: 12px;
    border-radius: 14px;
}
</style>
""", unsafe_allow_html=True)

st.title("🧪 Saliva Glucose Monitoring Platform")

if "history" not in st.session_state:
    st.session_state.history = []

# ==========================================
# FUNCTIONS
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


def extract_bubble_features(image_path, top_n=20):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Cannot read image.")

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.GaussianBlur(img_gray, (3, 3), 0)

    circles = cv2.HoughCircles(
        img_gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=20,
        param1=50, param2=35, minRadius=5, maxRadius=50
    )

    if circles is None:
        raise ValueError("No bubbles detected.")

    circles = np.around(circles).astype(int)
    candidates = []

    for x, y, r in circles[0]:
        r = int(r * 0.9)
        Y, X = np.ogrid[:img_rgb.shape[0], :img_rgb.shape[1]]
        mask = (X - x) ** 2 + (Y - y) ** 2 <= r ** 2
        roi_rgb = img_rgb[mask]
        roi_hsv = rgb_to_hsv(roi_rgb / 255.0)

        h_mean, s_mean, v_mean = roi_hsv.mean(axis=0)

        if 252 / 360 <= h_mean <= 290 / 360 and s_mean >= 0.04 and v_mean >= 0.60:
            score = (h_mean ** 8) * s_mean * v_mean * r
            candidates.append({"roi_hsv": roi_hsv, "score": score})

    if len(candidates) == 0:
        raise ValueError("No valid bubbles detected.")

    candidates = sorted(candidates, key=lambda b: b["score"], reverse=True)[:top_n]
    avg_hsv = np.mean([b["roi_hsv"].mean(axis=0) for b in candidates], axis=0)

    return avg_hsv, img_rgb


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


def predict_glucose(uploaded_file, glucose_range, reagent_name):
    temp_path = f"temp_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    avg_hsv, img_rgb = extract_bubble_features(temp_path)
    H_avg, S_avg, _ = avg_hsv

    model_H, model_S, model_HS = train_model(glucose_range)

    g_H = max(model_H.predict(pd.DataFrame({"H": [H_avg]}))[0], 0)
    g_S = max(model_S.predict(pd.DataFrame({"S": [S_avg]}))[0], 0)
    g_HS = max(model_HS.predict(pd.DataFrame({"H": [H_avg], "S": [S_avg]}))[0], 0)

    glucose = 0.5 * g_H + 0.3 * g_S + 0.2 * g_HS

    st.image(img_rgb, caption="📷 Uploaded Microfluidic Bubble Image", use_container_width=True)
    st.metric("🩸 Estimated Saliva Glucose", f"{glucose:.1f} µM")

    if glucose < 150:
        st.success("✅ Result within expected range")
    else:
        st.warning("⚠️ Elevated glucose detected. Please consult a healthcare professional.")

    st.session_state.history.append({
        "Time": datetime.now(),
        "Reagent": reagent_name,
        "Glucose": glucose
    })

# ==========================================
# TABS
# ==========================================
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🏠 Home",
    "🧪 Reagent Set A",
    "🧪 Reagent Set B",
    "📈 History",
    "🩸 Finger Prick",
    "🌿 Lifestyle & Diet"
])

with tab1:
    st.header("💡 About Diabetes")
    st.markdown("""
**Type 1 Diabetes**: autoimmune destruction of pancreatic beta cells leading to little or no insulin production.

**Type 2 Diabetes**: reduced insulin sensitivity and impaired glucose regulation, commonly associated with lifestyle and genetic factors.
""")

    st.subheader("🗺️ How to Use This Web App")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.image("https://img.icons8.com/fluency/96/test-tube.png")
        st.caption("1️⃣ Select reagent set")
    with col2:
        st.image("https://img.icons8.com/fluency/96/image-file.png")
        st.caption("2️⃣ Upload image")
    with col3:
        st.image("https://img.icons8.com/fluency/96/microscope.png")
        st.caption("3️⃣ Automatic analysis")
    with col4:
        st.image("https://img.icons8.com/fluency/96/combo-chart.png")
        st.caption("4️⃣ View result")

with tab2:
    st.header("🧪 Reagent Set A")
    file_A = st.file_uploader("Upload image", type=["jpg", "png", "jpeg"], key="A")
    if file_A:
        predict_glucose(file_A, [25, 50, 75, 100, 125], "Set A")

with tab3:
    st.header("🧪 Reagent Set B")
    file_B = st.file_uploader("Upload image", type=["jpg", "png", "jpeg"], key="B")
    if file_B:
        predict_glucose(file_B, [150, 200, 250, 300, 350], "Set B")

with tab4:
    st.header("📈 Historical Results")
    if st.session_state.history:
        df = pd.DataFrame(st.session_state.history)
        st.dataframe(df)
        chart_df = df.set_index("Time")[["Glucose"]]
        st.line_chart(chart_df)
    else:
        st.info("No previous results yet.")

with tab5:
    st.header("🩸 Conventional Glucose Monitoring")
    cols = st.columns(5)
    steps = [
        ("https://img.icons8.com/fluency/96/wash-your-hands.png", "Wash hands"),
        ("https://img.icons8.com/fluency/96/syringe.png", "Prepare lancet"),
        ("https://img.icons8.com/fluency/96/drop-of-blood.png", "Finger prick"),
        ("https://img.icons8.com/fluency/96/glucometer.png", "Read glucose"),
        ("https://img.icons8.com/fluency/96/notebook.png", "Log result")
    ]
    for col, (img, txt) in zip(cols, steps):
        with col:
            st.image(img)
            st.caption(txt)

with tab6:
    st.header("🌿 Tips for Diabetes Management")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🏃 Lifestyle")
        st.image("https://images.unsplash.com/photo-1517836357463-d25dfeac3438", use_container_width=True)
        st.markdown("- Exercise regularly\n- Sleep adequately\n- Reduce stress\n- Monitor glucose")
    with col2:
        st.subheader("🥗 Diet")
        st.image("https://images.unsplash.com/photo-1490645935967-10de6ba17061", use_container_width=True)
        st.markdown("- Reduce refined sugar\n- Increase fibre\n- Balanced meals\n- Stay hydrated")
