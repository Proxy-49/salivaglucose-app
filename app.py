import streamlit as st
import cv2
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.linear_model import LinearRegression

# =====================================
# PAGE STYLE
# =====================================
st.set_page_config(
    page_title="Saliva Glucose Monitoring Platform",
    layout="wide"
)

st.markdown("""
<style>
.stApp {
    background: #eaf8ea;
    color: black;
}
html, body, [class*="css"] {
    color: black !important;
}
button[data-baseweb="tab"] {
    color: black !important;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

st.title("🧪 Saliva Glucose Monitoring Platform")

# =====================================
# RGB TO HSV
# =====================================
def rgb_to_hsv(rgb):
    rgb = np.array(rgb)
    maxc = rgb.max(axis=1)
    minc = rgb.min(axis=1)

    v = maxc
    s = (maxc - minc) / (maxc + 1e-6)
    s[maxc == 0] = 0

    rc = (maxc - rgb[:,0]) / (maxc - minc + 1e-6)
    gc = (maxc - rgb[:,1]) / (maxc - minc + 1e-6)
    bc = (maxc - rgb[:,2]) / (maxc - minc + 1e-6)

    h = np.zeros_like(maxc)

    mask = maxc == rgb[:,0]
    h[mask] = (bc - gc)[mask]

    mask = maxc == rgb[:,1]
    h[mask] = 2.0 + (rc - bc)[mask]

    mask = maxc == rgb[:,2]
    h[mask] = 4.0 + (gc - rc)[mask]

    h = (h / 6.0) % 1.0
    h[minc == maxc] = 0.0

    return np.stack([h, s, v], axis=1)

# =====================================
# BUBBLE FEATURE EXTRACTION
# =====================================
def extract_bubble_features(image_path, top_n=20):
    img = cv2.imread(image_path)

    if img is None:
        raise ValueError(f"Cannot read image {image_path}")

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.GaussianBlur(img_gray, (3,3), 0)

    circles = cv2.HoughCircles(
        img_gray, cv2.HOUGH_GRADIENT,
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
            candidates.append({
                "roi_hsv": roi_hsv,
                "score": score
            })

    if len(candidates) == 0:
        raise ValueError("No bubbles passed HSV filter.")

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

# =====================================
# CALIBRATION MODEL
# =====================================
calibration_data = pd.DataFrame({
    "Glucose": [25, 50, 75, 100, 125],
    "H": [0.722795, 0.731712, 0.730700, 0.743624, 0.786134],
    "S": [0.086949, 0.093759, 0.097361, 0.107223, 0.121588]
})

H_blank_deg = 12
S_blank_percent = 0.6

H_blank = H_blank_deg / 360.0
S_blank = S_blank_percent / 100.0

calibration_data["H_corr"] = calibration_data["H"] - H_blank
calibration_data["S_corr"] = calibration_data["S"] - S_blank

y_glucose = calibration_data["Glucose"].values

model_H  = LinearRegression().fit(calibration_data[["H_corr"]], y_glucose)
model_S  = LinearRegression().fit(calibration_data[["S_corr"]], y_glucose)
model_HS = LinearRegression().fit(calibration_data[["H_corr","S_corr"]], y_glucose)

# =====================================
# TABS
# =====================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🏠 Home",
    "🧪 Saliva Glucose Estimation",
    "🍽 Diet",
    "🩸 Finger Prick Guide",
    "❓ Myth Buster"
])

# =====================================
# HOME
# =====================================
with tab1:
    st.header("About Diabetes")

    st.write("""
Diabetes mellitus is a chronic metabolic disorder characterized by elevated glucose levels.

**Type 1 diabetes** results from insufficient insulin production.

**Type 2 diabetes** results from insulin resistance.

This application estimates saliva glucose.
""")

    st.subheader("Workflow")

    st.write("""
```text
Upload / Take image under Saliva Glucose Estimation Tab
        ↓
Automatic bubble detection
        ↓
Automatic HSV colour extraction
        ↓
Automatic Regression prediction
        ↓
Given estimated saliva glucose
""")


# =====================================
# Saliva Glucose Estimation TAB
# =====================================
with tab2:
    uploaded_file = st.file_uploader(
        "Upload an image of your saliva bubbles",
        type=["jpg","png","jpeg"]
    )

    if uploaded_file:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        temp_path = f"temp_{timestamp}.jpg"

        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        try:
            avg_hsv, img_rgb = extract_bubble_features(temp_path)
            H_avg, S_avg, V_avg = avg_hsv

            df_H  = pd.DataFrame({"H_corr":[H_avg]})
            df_S  = pd.DataFrame({"S_corr":[S_avg]})
            df_HS = pd.DataFrame({"H_corr":[H_avg], "S_corr":[S_avg]})

            g_H  = max(model_H.predict(df_H)[0], 0)
            g_S  = max(model_S.predict(df_S)[0], 0)
            g_HS = max(model_HS.predict(df_HS)[0], 0)

            glucose_weighted = 0.5*g_H + 0.3*g_S + 0.2*g_HS

            st.image(img_rgb, caption="Uploaded Image")
            st.subheader("Estimated Glucose (µM)")
            st.write(f"**{glucose_weighted:.1f} µM**")

            if 10 <= glucose_weighted <= 150:
                st.success("✅ This is within expected healthy physiological saliva range.")
            elif 150 < glucose_weighted <= 250:
                st.warning("⚠️ Elevated saliva glucose detected. Consider confirmatory finger-prick or clinical assessment.")
            else:
                st.error("❌ Unusual Glucose Level detected, do upload another image.")

        except Exception as e:
            st.error(f"No bubbles detected in image, do upload another image: {e}")


# =====================================
# DIET TAB
# =====================================
with tab3:
    st.header("Food Serving Size Estimation")
    st.image("8.png")
    serving_df = pd.DataFrame({
    "Food Group": ["Vegetables", "Protein", "Carbohydrates", "Fats"],
    "Recommended ": ["2-3 Servings", "1-2 Servings", "1 Serving", "1 Serving"]
    })

    st.table(serving_df)

    st.subheader("Carbohydrate Counter")

    carb_foods = {
        "Rice (1 serving)": 45,
        "Bread (2 slices)": 30,
        "Apple (1 piece)": 15,
        "Banana (1 piece)": 27,
        "Noodles (1 serving)": 40,
        "Pasta (1 serving)": 42,
        "Crackers (5 pieces)": 15,
        "Milk (1 cup)": 12
    }

    food = st.selectbox(
        "Select food item",
        list(carbs_per_serving.keys())
    )

    quantity = st.number_input(
        "Number of servings",
        min_value=1,
        value=1
    )

    total_carbs = carbs_per_serving[food] * quantity

    st.metric("Estimated Carbohydrate Intake", f"{total_carbs} g")


# =====================================
# FINGER PRICK GUIDE
# =====================================
with tab4:
    st.header("Conventional Finger-Prick Monitoring")

    steps = [
        ("1.png", "1. Warming up finger"),
        ("2.png", "2. Pricking finger"),
        ("3.png", "3. Squeezing finger"),
        ("4.png", "4. Measuring blood glucose"),
        ("5.png", "5. Calculating insulin dose"),
        ("6.png", "6. Dialling insulin pen"),
        ("7.png", "7. Delivering insulin")
    ]

    for img, caption in steps:
        st.image(img)
        st.caption(caption)


# =====================================
# MYTH BUSTER TAB
# =====================================
with tab5:
    st.header("Diabetes Myth Buster")

    myths = {
        "Myth 1: Avoid all starchy foods at all cost":
            "All starchy foods break down to glucose, which is the body's preferred source of energy. Choose high-fibre wholegrain options.",

        "Myth 2: All sugar-free products are calorie-free":
            "Sugar-free foods may still contain starch and carbohydrates that raise blood glucose.",

        "Myth 3: People with diabetes should never consume sugar":
            "Sugar may still be consumed in controlled amounts through carbohydrate counting.",

        "Myth 4: Bitter foods lower glucose":
            "Bitter foods do not directly lower blood glucose. Carbohydrate intake is the main driver.",

        "Myth 5: Wholegrain rice means more rice can be eaten":
            "Wholegrain rice has similar carbohydrate content but higher fibre.",

        "Myth 6: Unlimited protein is acceptable":
            "Excess protein still contributes calories and may lead to weight gain.",

        "Myth 7: Favourite sweets should treat hypoglycaemia":
            "Treat low glucose with 15 g fast-acting carbohydrates only.",

        "Myth 8: Fat content does not matter":
            "Fat slows carbohydrate digestion and may prolong elevated glucose.",

        "Myth 9: People with diabetes should not eat fruits":
            "Fresh fruits should still be consumed in controlled portions.",

        "Myth 10: Weight loss cures diabetes":
            "Weight loss improves control but does not necessarily cure diabetes."
    }

    for myth, fact in myths.items():
        with st.expander(myth):
            st.write(f"**FACT:** {fact}")
