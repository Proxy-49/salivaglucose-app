import streamlit as st
import cv2
import numpy as np
import pandas as pd
from datetime import datetime
from zoneinfo import ZoneInfo
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# =====================================
# PAGE STYLE
# =====================================
st.set_page_config(
    page_title="Saliva Glucose Monitoring Platform",
    layout="wide"
)


# app_glucose_only.py

import streamlit as st
import cv2
import numpy as np
import pandas as pd
import os
from datetime import datetime
from sklearn.linear_model import LinearRegression

# --------------------------
# Streamlit UI config
# --------------------------
st.set_page_config(page_title="Saliva Glucose Estimator", layout="wide")
st.title("Glucose Estimator from Microfluidic Bubbles")

# --------------------------
# RGB to HSV conversion
# --------------------------
def rgb_to_hsv(rgb):
    """Convert Nx3 array of RGB [0-1] to HSV [0-1]"""
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

# --------------------------
# Bubble feature extraction (no visualization)
# --------------------------
def extract_bubble_features(image_path, top_n=20):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Cannot read image {image_path}")

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.GaussianBlur(img_gray, (3,3), 0)

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
        mask = (X - x)**2 + (Y - y)**2 <= r**2
        roi_rgb = img_rgb[mask]
        roi_hsv = rgb_to_hsv(roi_rgb / 255.0)
        h_mean, s_mean, v_mean = roi_hsv.mean(axis=0)

        # HSV pink filter
        if (h_mean >= 252/360) and (h_mean <= 290/360) and (s_mean >= 0.07) and (v_mean >= 0.60):
            score = (h_mean**8)* s_mean * v_mean  * r
            candidates.append({
                "roi_hsv": roi_hsv,
                "score": score
            })

    if len(candidates) == 0:
        raise ValueError("No bubbles passed HSV filter.")

    candidates = sorted(candidates, key=lambda b: b["score"], reverse=True)[:top_n]
    avg_hsv = np.mean([b["roi_hsv"].mean(axis=0) for b in candidates], axis=0)
    return avg_hsv, img_rgb

# --------------------------
# Calibration data (pure glucose)
# --------------------------
calibration_data = pd.DataFrame({
    "Glucose": [25, 50, 75, 100, 125],
    "H": [0.722795, 0.731712, 0.730700, 0.743624, 0.786134],
    "S": [0.086949, 0.093759, 0.097361, 0.107223, 0.121588]
})

y_glucose = calibration_data["Glucose"].values

model_H  = LinearRegression().fit(calibration_data[["H"]], y_glucose)
model_S  = LinearRegression().fit(calibration_data[["S"]], y_glucose)
model_HS = LinearRegression().fit(calibration_data[["H","S"]], y_glucose)

# --------------------------
# Streamlit file uploader
# --------------------------
uploaded_file = st.file_uploader("Upload an image of your microfluidic bubbles", type=["jpg","png","jpeg"])

if uploaded_file:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    temp_path = f"temp_{timestamp}.jpg"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    try:
        avg_hsv, img_rgb = extract_bubble_features(temp_path)
        H_avg, S_avg, V_avg = avg_hsv

        # Convert to DataFrames for prediction
        df_H  = pd.DataFrame({"H":[H_avg]})
        df_S  = pd.DataFrame({"S":[S_avg]})
        df_HS = pd.DataFrame({"H":[H_avg], "S":[S_avg]})

        # Predict glucose
        g_H  = max(model_H.predict(df_H)[0], 0)
        g_S  = max(model_S.predict(df_S)[0], 0)
        g_HS = max(model_HS.predict(df_HS)[0], 0)

        # Weighted glucose using H, S, and H+S model
        glucose_weighted = 0.5*g_H + 0.3*g_S + 0.2*g_HS

        # Display original image only
        st.image(img_rgb, caption="Uploaded Image", use_column_width=True)
        st.subheader("Estimated Glucose (µM)")
        st.write(f"{glucose_weighted:.1f} µM")

    except Exception as e:
        st.error(f"Error processing image: {e}")


Okay I want to fix the hough transform function - I’m trying to isolate bubbles in my microfluidic chip 

I want a step by step for my python code- I want to run this in anaconda not stream lit and GitHub 

I want to include images of bubbles selected - before filtering 

- Bubbles selected after filtering  

The region of selection is always too large, I want to reduce to diameter of my circle selection instead of simply reducing the radius at the end 

I want to improve the accuracy of bubbles selection as well

Below is a structured way to rebuild the bubble-detection section step by step in plain Python for Anaconda/Jupyter, while also improving circle accuracy and showing before/after filtering images.

Your current issue is not just “radius too large at the end.” The main problem is that the Hough transform is detecting circles that are already oversized, and then you shrink them afterward with:

r = int(r * 0.9)
That is a patch, not a fix.

The correct fix is to improve the circle detection stage itself.

1) Move from Streamlit → plain Python script
Create a normal .py file, for example:

# bubble_detection.py
import cv2
import numpy as np
import matplotlib.pyplot as plt
Then use:

image_path = "your_image.jpg"
img = cv2.imread(image_path)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

plt.imshow(img_rgb)
plt.title("Original image")
plt.axis("off")
plt.show()
This runs directly in Anaconda Spyder / Jupyter Notebook.

2) Improve preprocessing before Hough transform
This is the biggest factor in accuracy.

Your current preprocessing is:

img_gray = cv2.GaussianBlur(img_gray, (3,3), 0)
This is too weak.

Use:

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# stronger denoising
gray_blur = cv2.GaussianBlur(gray, (7, 7), 1.5)

# optional contrast enhancement
gray_blur = cv2.equalizeHist(gray_blur)
This improves edge definition.

3) Fix the HoughCircles parameters
This is where your oversized bubbles come from.

Replace your current block with this:

circles = cv2.HoughCircles(
    gray_blur,
    cv2.HOUGH_GRADIENT,
    dp=1.1,
    minDist=15,
    param1=80,
    param2=22,
    minRadius=8,
    maxRadius=30
)
Why this is better
dp
Current:

dp=1.2
Try:

dp=1.1
Lower values improve resolution.

param2
This is the most important one.

Current:

param2=35
This is probably too strict and unstable.

Lower it:

param2=22
lower = more circles detected

higher = fewer but stricter

Typical range:

18–30
maxRadius
This is critical.

Instead of shrinking radius afterward, limit radius here.

For example:

maxRadius=30
This directly prevents large false circles.

This is much better than:

r = int(r * 0.9)
4) Show ALL circles before filtering
You asked for “bubbles selected before filtering.”

Use this:

img_before = img_rgb.copy()

if circles is not None:
    circles = np.round(circles[0]).astype(int)

    for x, y, r in circles:
        cv2.circle(img_before, (x, y), r, (0, 255, 0), 2)

plt.figure(figsize=(8,8))
plt.imshow(img_before)
plt.title("Bubbles before HSV filtering")
plt.axis("off")
plt.show()
This gives visual debugging.

5) Improve region selection (THIS solves your oversized ROI issue)
This part is very important.

Instead of shrinking radius after detection, define the ROI more precisely.

Use:

diameter_scale = 0.75
r_new = int(r * diameter_scale)
Then:

mask = (X - x)**2 + (Y - y)**2 <= r_new**2
This selects a smaller inner bubble region.

For example:

diameter_scale = 0.7
means use 70% of detected diameter.

This is mathematically much cleaner.

6) Improve bubble filtering accuracy
Your HSV filter is currently:

if (h_mean >= 252/360) and (h_mean <= 290/360) and ...
This is too rigid.

Real microscopy images vary.

Use a scoring-based approach instead.

Better filtering method
filtered_circles = []
img_after = img_rgb.copy()

for x, y, r in circles:
    r_new = int(r * 0.75)

    Y, X = np.ogrid[:img_rgb.shape[0], :img_rgb.shape[1]]
    mask = (X - x)**2 + (Y - y)**2 <= r_new**2

    roi_rgb = img_rgb[mask]
    roi_hsv = rgb_to_hsv(roi_rgb / 255.0)

    h_mean, s_mean, v_mean = roi_hsv.mean(axis=0)

    if (
        0.68 <= h_mean <= 0.82 and
        0.05 <= s_mean <= 0.20 and
        v_mean >= 0.55
    ):
        filtered_circles.append((x, y, r_new))

        cv2.circle(img_after, (x, y), r_new, (255, 0, 0), 2)
This is much more robust.

7) Show circles AFTER filtering
plt.figure(figsize=(8,8))
plt.imshow(img_after)
plt.title("Bubbles after HSV filtering")
plt.axis("off")
plt.show()
This directly gives before/after comparison.

8) Major accuracy improvement: use contour refinement after Hough
This is the best way to improve accuracy.

Hough gives approximate circles.

Contours refine boundaries.

After detecting each circle:

x1 = max(0, x-r)
x2 = min(img.shape[1], x+r)
y1 = max(0, y-r)
y2 = min(img.shape[0], y+r)

roi = gray[y1:y2, x1:x2]

_, thresh = cv2.threshold(roi, 0, 255, cv2.THRESH_OTSU)

contours, _ = cv2.findContours(
    thresh,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)
This improves edge localization.

For microfluidic bubbles this is usually better than pure Hough alone.

9) My honest academic recommendation
For your application, Hough transform alone is probably not the best primary detector.

Microfluidic bubbles often have:

partial edges

merged bubbles

lighting artifacts

reflections

A better pipeline is:

CLAHE / histogram equalization
→ adaptive threshold
→ contour detection
→ circularity filtering
→ HSV validation
This often outperforms Hough.

Circle Hough is best when edges are very clean.

Microfluidic imaging often is not.

So if accuracy is the priority, I would strongly recommend contour + circularity analysis.

If you want, I can help you rewrite your full detection function into a research-grade OpenCV pipeline for microfluidic bubble detection.


Nvm keep to hough, add in a step to view the grayscale and the improve constrast versions 

Add explanations to why these steps were done as well

Understood. If you want to stay with Hough circle detection, the best approach is to make the preprocessing steps visible and explain why each one is included.

Below is a step-by-step Python workflow for Anaconda / Jupyter / Spyder, with image outputs for:

original image

grayscale

contrast-improved grayscale

bubbles before filtering

bubbles after filtering

I’ll also explain the reasoning for each step.

1) Import libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt
2) Load and display original image
image_path = "your_bubble_image.jpg"

img = cv2.imread(image_path)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(6,6))
plt.imshow(img_rgb)
plt.title("Original Image")
plt.axis("off")
plt.show()
3) Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

plt.figure(figsize=(6,6))
plt.imshow(gray, cmap="gray")
plt.title("Grayscale Image")
plt.axis("off")
plt.show()
Why this step is done
The Hough transform works on intensity gradients, not color.

Circle detection relies on edges, and edges are much easier to detect in grayscale because:

the image becomes one intensity channel

noise from RGB color variation is removed

computational load is lower

So this is a necessary preprocessing step.

4) Improve contrast
Use histogram equalization:

gray_contrast = cv2.equalizeHist(gray)

plt.figure(figsize=(6,6))
plt.imshow(gray_contrast, cmap="gray")
plt.title("Contrast Enhanced Grayscale")
plt.axis("off")
plt.show()
Why this step is done
Microfluidic bubbles often have weak edge contrast, especially under uneven illumination.

Contrast enhancement helps by:

making bright bubbles brighter

making dark background darker

sharpening boundary intensity differences

This makes the circle edges more visible to the Hough detector.

In practical terms, it improves the gradient peaks used by Hough.

5) Blur to reduce noise
gray_blur = cv2.GaussianBlur(gray_contrast, (5,5), 1.2)

plt.figure(figsize=(6,6))
plt.imshow(gray_blur, cmap="gray")
plt.title("Blurred + Contrast Enhanced")
plt.axis("off")
plt.show()
Why this step is done
Raw images contain:

pixel noise

reflections

tiny artifacts

camera sensor grain

These can create false circle edges.

Gaussian blur suppresses high-frequency noise while preserving larger circular boundaries.

This improves stability.

6) Run Hough circle detection
circles = cv2.HoughCircles(
    gray_blur,
    cv2.HOUGH_GRADIENT,
    dp=1.1,
    minDist=15,
    param1=80,
    param2=22,
    minRadius=8,
    maxRadius=25
)
Why these parameters
dp=1.1
Controls resolution of the accumulator.

Lower values improve positional precision.

minDist=15
Minimum distance between detected bubble centers.

This helps prevent multiple detections on the same bubble.

param1=80
Edge threshold for the internal Canny detector.

Higher values reduce weak/noisy edges.

param2=22
Detection confidence threshold.

Lower = more bubbles
Higher = fewer but stricter bubbles

This usually needs tuning.

minRadius, maxRadius
This is the most important fix for your “circle too large” problem.

Instead of shrinking afterward, constrain the detector directly.

Example:

minRadius=8
maxRadius=25
This ensures circles are found within your expected bubble size.

7) Show circles before filtering
img_before = img_rgb.copy()

if circles is not None:
    circles = np.round(circles[0]).astype(int)

    for x, y, r in circles:
        cv2.circle(img_before, (x, y), r, (0, 255, 0), 2)

plt.figure(figsize=(6,6))
plt.imshow(img_before)
plt.title("Bubbles Before Filtering")
plt.axis("off")
plt.show()
Why this step is important
This gives visual confirmation of what Hough is actually detecting.

You should always inspect this before applying HSV filters.

Otherwise you may be debugging the wrong stage.

8) Reduce selected diameter for ROI analysis
Instead of changing the detected circle size itself, use a smaller analysis diameter.

img_after = img_rgb.copy()
filtered = []

for x, y, r in circles:
    r_roi = int(r * 0.75)

    Y, X = np.ogrid[:img_rgb.shape[0], :img_rgb.shape[1]]
    mask = (X - x)**2 + (Y - y)**2 <= r_roi**2

    roi = img_rgb[mask]

    mean_intensity = roi.mean()

    if mean_intensity > 120:
        filtered.append((x, y, r_roi))
        cv2.circle(img_after, (x, y), r_roi, (255, 0, 0), 2)
Why this step is done
This solves your ROI issue.

The detected circle boundary is the estimated bubble edge.

But for color analysis, you usually want the inner region only.

So using:

r_roi = int(r * 0.75)
means:

keep detection circle accurate

use smaller sampling region

avoid boundary artifacts

This is much better than modifying the detected radius itself.

9) Show circles after filtering
plt.figure(figsize=(6,6))
plt.imshow(img_after)
plt.title("Bubbles After Filtering")
plt.axis("off")
plt.show()
10) Recommended workflow for tuning accuracy
Tune in this order:

first
maxRadius
minRadius
second
param2
third
Gaussian blur kernel
This order usually gives the fastest improvement.

My direct technical advice
The biggest source of inaccuracy is usually incorrect radius bounds, not the Hough algorithm itself.

Measure a few bubbles manually in pixels and use those values for:

minRadius
maxRadius
That will improve accuracy more than arbitrary post-scaling.

If you want, I can help you build a parameter sweep script to automatically test multiple Hough settings and compare detection results.


#Setup and Library Import

import sys
!{sys.executable} -m pip install opencv-python
import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from datetime import datetime
from matplotlib.colors import rgb_to_hsv
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# Define Input Data
image_files = [
    "25.jpg",
    "50.jpg",
    "75.jpg",
    "100.jpg",
    "125.jpg"
]

# Corresponding glucose concentrations (µM)
glucose_uM = np.array([25, 50, 75, 100, 125])

# Create timestamped folder for outputs
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
save_folder = f"detected_bubbles_{timestamp}"
os.makedirs(save_folder, exist_ok=True)

#Bubble Feature Extraction
def extract_bubble_features(image_path, visualize=True, save_images=True, top_n=20):
    #Read and preprocess the image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Cannot read image {image_path}")

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.GaussianBlur(img_gray, (3, 3), 0)

    # Detect circular bubbles
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
        raise ValueError(f"No bubbles detected in {image_path}")

    circles = np.uint16(np.around(circles))
    candidates = []
    
#Shrink radius and compute bubble ROI
    for x, y, r in circles[0]:
        r = int(r * 0.9)  # shrink to 90%

        Y, X = np.ogrid[:img_rgb.shape[0], :img_rgb.shape[1]]
        mask = (X - x)**2 + (Y - y)**2 <= r**2

        roi_rgb = img_rgb[mask]
        roi_hsv = rgb_to_hsv(roi_rgb / 255.0)

        h_mean, s_mean, v_mean = roi_hsv.mean(axis=0)

        # HSV-based pink filter
        if (h_mean >= 252/360) and (h_mean <= 290/360) and \
           (s_mean >= 0.07) and (v_mean >= 0.60):

            # Pinkness score: 8× weight on hue
            score = (h_mean ** 8) *s_mean * v_mean* r

            candidates.append({
                "x": int(x),
                "y": int(y),
                "r": r,
                "score": score,
                "rgb": roi_rgb.mean(axis=0),
                "hsv": roi_hsv.mean(axis=0)
            })

    if len(candidates) == 0:
        raise ValueError(f"No bubbles passed HSV filter in {image_path}")

    # Sort by pinkness score
    candidates = sorted(candidates, key=lambda b: b["score"], reverse=True)

    # Enforce non-overlapping selection
    selected = []
    for b in candidates:
        overlap = False
        for sb in selected:
            dx = float(b["x"] - sb["x"])
            dy = float(b["y"] - sb["y"])
            dist = np.sqrt(dx*dx + dy*dy)
            if dist < (b["r"] + sb["r"]):
                overlap = True
                break
        if not overlap:
            selected.append(b)
        if len(selected) >= top_n:
            break

    # Visualization
    img_vis = img_rgb.copy()
    for b in selected:
        cv2.circle(img_vis, (b["x"], b["y"]), b["r"], (255, 0, 0), 2)

    if visualize or save_images:
        plt.figure(figsize=(6, 6))
        plt.imshow(img_vis)
        plt.title(f"{image_path}: {len(selected)} bubbles selected")
        plt.axis("off")
        plt.show()

    if save_images:
        out_path = os.path.join(save_folder, f"bubbles_{os.path.basename(image_path)}")
        cv2.imwrite(out_path, cv2.cvtColor(img_vis, cv2.COLOR_RGB2BGR))

    avg_rgb = np.mean([b["rgb"] for b in selected], axis=0)
    avg_hsv = np.mean([b["hsv"] for b in selected], axis=0)

    return avg_rgb, avg_hsv

rgb_list = []
hsv_list = []

for img in image_files:
    rgb, hsv = extract_bubble_features(img)
    rgb_list.append(rgb)
    hsv_list.append(hsv)

rgb_arr = np.array(rgb_list)
hsv_arr = np.array(hsv_list)

data = pd.DataFrame({
    "Glucose_uM": glucose_uM,
    "R": rgb_arr[:,0],
    "G": rgb_arr[:,1],
    "B": rgb_arr[:,2],
    "H": hsv_arr[:,0],
    "S": hsv_arr[:,1],
    "V": hsv_arr[:,2]
})

data

X = data["Glucose_uM"].values.reshape(-1,1)

results = []

for ch in ["R","G","B","H","S","V"]:
    y = data[ch].values
    model = LinearRegression().fit(X, y)
    y_pred = model.predict(X)

    r2 = r2_score(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))

    results.append([ch, r2, rmse])

results_df = pd.DataFrame(results, columns=["Channel","R2","RMSE"])
results_df

plt.figure(figsize=(10,4))

for i, ch in enumerate(["H","S"], start=1):
    y = data[ch].values
    model = LinearRegression().fit(X, y)
    y_pred = model.predict(X)

    plt.subplot(1,2,i)
    plt.scatter(data["Glucose_uM"], y, s=70)
    plt.plot(data["Glucose_uM"], y_pred, "k--")
    plt.xlabel("Glucose (µM)")
    plt.ylabel(ch)
    plt.title(f"{ch} vs Glucose")

plt.tight_layout()
plt.show()

now as mentioned above edit the code for the imporved hough detection as well as the viewing of intermediate steps

Good — this is now at the stage where a proper structural edit is more useful than isolated snippets.

I’ll rewrite your extract_bubble_features() function directly so you can paste it into your notebook/script with:

improved Hough preprocessing

grayscale visualization

contrast-enhanced visualization

blurred image visualization

circles before HSV filtering

circles after filtering

improved radius control at detection stage

smaller ROI diameter for feature extraction

explanations embedded as comments

Replace your function with this version
def extract_bubble_features(image_path, visualize=True, save_images=True, top_n=20):
    # -----------------------------
    # 1. Read image
    # -----------------------------
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Cannot read image {image_path}")

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # -----------------------------
    # 2. Convert to grayscale
    # Hough works on intensity gradients
    # -----------------------------
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if visualize:
        plt.figure(figsize=(5,5))
        plt.imshow(gray, cmap="gray")
        plt.title(f"{image_path} - Grayscale")
        plt.axis("off")
        plt.show()

    # -----------------------------
    # 3. Improve contrast
    # makes bubble edges clearer
    # -----------------------------
    gray_contrast = cv2.equalizeHist(gray)

    if visualize:
        plt.figure(figsize=(5,5))
        plt.imshow(gray_contrast, cmap="gray")
        plt.title(f"{image_path} - Contrast Enhanced")
        plt.axis("off")
        plt.show()

    # -----------------------------
    # 4. Blur to reduce noise
    # -----------------------------
    gray_blur = cv2.GaussianBlur(gray_contrast, (5, 5), 1.2)

    if visualize:
        plt.figure(figsize=(5,5))
        plt.imshow(gray_blur, cmap="gray")
        plt.title(f"{image_path} - Blurred")
        plt.axis("off")
        plt.show()

    # -----------------------------
    # 5. Improved Hough detection
    # tighter radius control
    # -----------------------------
    circles = cv2.HoughCircles(
        gray_blur,
        cv2.HOUGH_GRADIENT,
        dp=1.1,
        minDist=15,
        param1=80,
        param2=22,
        minRadius=8,
        maxRadius=25
    )

    if circles is None:
        raise ValueError(f"No bubbles detected in {image_path}")

    circles = np.round(circles[0]).astype(int)

    # -----------------------------
    # 6. Show circles BEFORE filtering
    # -----------------------------
    img_before = img_rgb.copy()

    for x, y, r in circles:
        cv2.circle(img_before, (x, y), r, (0, 255, 0), 2)

    if visualize:
        plt.figure(figsize=(6,6))
        plt.imshow(img_before)
        plt.title(f"{image_path} - Before HSV Filtering")
        plt.axis("off")
        plt.show()

    candidates = []

    # -----------------------------
    # 7. ROI extraction with smaller
    # diameter for analysis
    # -----------------------------
    for x, y, r in circles:
        # use smaller inner sampling radius
        r_roi = int(r * 0.75)

        Y, X = np.ogrid[:img_rgb.shape[0], :img_rgb.shape[1]]
        mask = (X - x)**2 + (Y - y)**2 <= r_roi**2

        roi_rgb = img_rgb[mask]

        if roi_rgb.size == 0:
            continue

        roi_hsv = rgb_to_hsv(roi_rgb / 255.0)

        h_mean, s_mean, v_mean = roi_hsv.mean(axis=0)

        # improved HSV filter
        if (
            252/360 <= h_mean <= 290/360 and
            s_mean >= 0.07 and
            v_mean >= 0.60
        ):
            score = (h_mean ** 8) * s_mean * v_mean * r_roi

            candidates.append({
                "x": int(x),
                "y": int(y),
                "r": r_roi,
                "score": score,
                "rgb": roi_rgb.mean(axis=0),
                "hsv": roi_hsv.mean(axis=0)
            })

    if len(candidates) == 0:
        raise ValueError(f"No bubbles passed HSV filter in {image_path}")

    # -----------------------------
    # 8. Sort by score
    # -----------------------------
    candidates = sorted(
        candidates,
        key=lambda b: b["score"],
        reverse=True
    )

    # -----------------------------
    # 9. Non-overlapping selection
    # -----------------------------
    selected = []

    for b in candidates:
        overlap = False

        for sb in selected:
            dx = b["x"] - sb["x"]
            dy = b["y"] - sb["y"]
            dist = np.sqrt(dx**2 + dy**2)

            if dist < (b["r"] + sb["r"]):
                overlap = True
                break

        if not overlap:
            selected.append(b)

        if len(selected) >= top_n:
            break

    # -----------------------------
    # 10. Show circles AFTER filtering
    # -----------------------------
    img_after = img_rgb.copy()

    for b in selected:
        cv2.circle(
            img_after,
            (b["x"], b["y"]),
            b["r"],
            (255, 0, 0),
            2
        )

    if visualize:
        plt.figure(figsize=(6,6))
        plt.imshow(img_after)
        plt.title(f"{image_path} - After HSV Filtering")
        plt.axis("off")
        plt.show()

    # -----------------------------
    # 11. Save final image
    # -----------------------------
    if save_images:
        out_path = os.path.join(
            save_folder,
            f"bubbles_{os.path.basename(image_path)}"
        )
        cv2.imwrite(
            out_path,
            cv2.cvtColor(img_after, cv2.COLOR_RGB2BGR)
        )

    avg_rgb = np.mean([b["rgb"] for b in selected], axis=0)
    avg_hsv = np.mean([b["hsv"] for b in selected], axis=0)

    return avg_rgb, avg_hsv
What changed technically
These are the important academic improvements.

1) Contrast enhancement added
gray_contrast = cv2.equalizeHist(gray)
This improves edge separability.

For bubble microscopy this is often essential.

2) Better Hough parameters
Old:

dp=1.2
param1=50
param2=35
maxRadius=50
New:

dp=1.1
param1=80
param2=22
maxRadius=25
This improves:

localization

edge strictness

radius realism

The radius constraint is the major fix.

3) No more fake radius correction
You previously had:

r = int(r * 0.9)
This modifies detection geometry.

Now we separate:

detection radius = actual bubble

ROI radius = analysis region

Much cleaner.

4) Intermediate steps now visible
You specifically requested:

grayscale

contrast enhanced

blurred

before filtering

after filtering

All are now included.

This is the correct debugging workflow.

My technical recommendation
The next thing you should tune is this:

param2=22
This is the single most sensitive Hough parameter.

Try:

18, 20, 22, 24, 26
and compare outputs.

That usually gives the biggest performance gain.


what about the intermeidate steps like the convert to gray scale, improving of contraast , bluring to reduce noirse i wan to see theese as well

Yes — those intermediate images should absolutely be visualized, especially if you are tuning Hough parameters.
At the moment, the function I rewrote already includes those steps under if visualize:, but I’ll show you exactly where they appear and how to make them clearer as a sequential diagnostic pipeline.

The best way is to display them side-by-side in one figure, so you can compare how each preprocessing step changes the image.

Replace the visualization section for preprocessing with this block
Put this right after the blur step and before cv2.HoughCircles(...).

    # -----------------------------
    # Show preprocessing pipeline
    # -----------------------------
    if visualize:
        fig, axes = plt.subplots(1, 4, figsize=(18, 5))

        # Original
        axes[0].imshow(img_rgb)
        axes[0].set_title("Original RGB")
        axes[0].axis("off")

        # Grayscale
        axes[1].imshow(gray, cmap="gray")
        axes[1].set_title("Grayscale")
        axes[1].axis("off")

        # Contrast enhanced
        axes[2].imshow(gray_contrast, cmap="gray")
        axes[2].set_title("Contrast Enhanced")
        axes[2].axis("off")

        # Blurred
        axes[3].imshow(gray_blur, cmap="gray")
        axes[3].set_title("Blurred (Noise Reduced)")
        axes[3].axis("off")

        plt.suptitle(f"{image_path} - Preprocessing Steps")
        plt.tight_layout()
        plt.show()
What each image is showing
This is the important analytical part.

1) Original RGB
axes[0].imshow(img_rgb)
This is your raw microscope image.

Purpose:

visual reference

compare with processed versions

check lighting consistency

This is your baseline.

2) Grayscale
axes[1].imshow(gray, cmap="gray")
This removes color information.

Why:

Hough circles work from edge gradients, not RGB channels.

The circle transform internally relies on intensity transitions.

Mathematically, it detects radial edge accumulations.

So color is unnecessary at this stage.

3) Contrast enhanced
axes[2].imshow(gray_contrast, cmap="gray")
This step:

gray_contrast = cv2.equalizeHist(gray)
redistributes pixel intensities.

Why this matters:

bubble boundaries are often only slightly brighter/darker than background.

Contrast enhancement amplifies those differences.

This improves edge strength before Hough.

You should see:

darker background

brighter bubble rims

sharper transitions

4) Blurred
axes[3].imshow(gray_blur, cmap="gray")
This step:

gray_blur = cv2.GaussianBlur(gray_contrast, (5,5), 1.2)
reduces high-frequency noise.

Why:

small bright speckles and sensor noise can create false circles.

Blur smooths those artifacts.

Ideally:

bubble edges remain visible

isolated bright pixels disappear

Why this is important academically
For image analysis, preprocessing should always be auditable.

You do not want to tune Hough parameters blindly.

The correct sequence is:

original
→ grayscale
→ contrast enhanced
→ denoised
→ circle detection
→ HSV filtering
By visualizing each stage, you can identify whether errors come from:

poor contrast

over-blurring

wrong radius bounds

incorrect Hough thresholds

rather than guessing.

Optional: save these intermediate images
If you want to save them into your timestamp folder:

cv2.imwrite(os.path.join(save_folder, "gray.jpg"), gray)
cv2.imwrite(os.path.join(save_folder, "contrast.jpg"), gray_contrast)
cv2.imwrite(os.path.join(save_folder, "blur.jpg"), gray_blur)
This is useful for reports and GitHub documentation.

This preprocessing panel is usually the most useful diagnostic figure when optimizing Hough detection.


now the bubbles looks good do not need to reduce the size so much

Good — that means the Hough radius bounds are now doing the main correction, which is exactly what you want.

In that case, you should stop shrinking the ROI as aggressively.

Right now the function still uses:

r_roi = int(r * 0.75)
That was useful when the detected circles were oversized, but if the circles now align well with the actual bubble boundary, this reduction is probably cutting away valid signal.

Recommended change
Change this line:

r_roi = int(r * 0.75)
to:

r_roi = int(r * 0.95)
This keeps almost the full bubble while still avoiding edge artifacts.

Why 0.95 instead of 1.0
Using exactly:

r_roi = r
can include the bubble boundary ring and surrounding background pixels.

Those edge pixels often have:

reflection artifacts

partial edge gradients

darker rim pixels

illumination bleed

For HSV feature extraction, those can distort your mean values.

Using:

0.95
keeps nearly the full bubble interior while excluding the extreme outer ring.

This is usually the most stable choice.

Updated code section
Replace this block:

for x, y, r in circles:
    r_roi = int(r * 0.75)
with:

for x, y, r in circles:
    # keep almost full bubble diameter
    r_roi = int(r * 0.95)
If circles are now very accurate
You can even test:

r_roi = int(r * 0.98)
or

r_roi = r
Then compare the extracted HSV values.

A good diagnostic test is:

print(f"H={h_mean:.4f}, S={s_mean:.4f}, V={v_mean:.4f}")
for several bubbles.

If values remain stable across images, then the ROI size is appropriate.

My technical recommendation
For quantitative work, I would test these three settings:

0.90
0.95
1.00
and compare the regression fit:

R²
RMSE
against glucose.

Choose the one giving the best model performance rather than relying only on visual appearance.

That is the more rigorous approach.


why my Channel	R2	RMSE
0	R	1.266259e-03	8.251545
1	G	6.720670e-02	9.590355
2	B	9.304981e-08	9.454485
3	H	6.029419e-01	0.011101
4	S	7.484710e-01	0.008206
5	V	1.024465e-08	0.037070

how to put the R2 in terms of 0. smth for  

What you are seeing is already in decimal form between 0 and 1.
The reason some values look unusual is because Python is displaying them in scientific notation.

For example:

1.266259e-03
means:

1.266259 × 10⁻³ = 0.001266259
So your table is actually:

Channel	R² (decimal)
R	0.0013
G	0.0672
B	0.0000
H	0.6029
S	0.7485
V	0.0000
So the good fits are clearly H and S.

Why Python shows e-03
This is just display formatting.

Very small numbers are automatically shown in exponential notation.

Examples:

1e-01 = 0.1
1e-02 = 0.01
1e-03 = 0.001
So nothing is wrong mathematically.

How to force decimal format
Use pandas formatting before displaying the dataframe.

Replace:

results_df
with:

results_df.style.format({
    "R2": "{:.4f}",
    "RMSE": "{:.4f}"
})
This will show:

0.0013
0.0672
0.0000
0.6029
0.7485
which is easier to read.

If you want to permanently round the dataframe
Use:

results_df = results_df.round(4)
results_df
This changes the stored values to 4 decimal places.

If you want percentages instead
Sometimes R² is reported as percentage explained variance.

Use:

results_df["R2_percent"] = results_df["R2"] * 100
results_df = results_df.round(2)
results_df
Then:

H = 60.29%
S = 74.85%
Interpretation of your results
This is the more important part.

R, G, B are essentially useless
These values:

R = 0.0013
B ≈ 0
V ≈ 0
mean almost no linear relationship with glucose.

So these channels are not informative.

H is moderate
H = 0.6029
This means about 60% of the variance in hue is explained by glucose concentration.

Reasonable.

S is strongest
S = 0.7485
This is the best predictor.

About 75% explained variance, which is quite decent for image-derived biochemical estimation.

This supports using saturation as your primary signal.

Scientifically, this suggests bubble color intensity / chromatic purity changes more consistently than brightness.


okay now based on this i wan to adapt to my streamlit app # =====================================
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
    "🧪 Reagent Set A",
    "🍽 Diet",
    "🩸 Finger Prick Guide",
    "❓ Myth Buster"
])
=====================================
REAGENT SET A
=====================================

with tab2:
uploaded_file = st.file_uploader(
"Upload an image of your saliva bubbles",
type=["jpg","png","jpeg"]
) if uploaded_file:     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")     temp_path = f"temp_{timestamp}.jpg"      with open(temp_path, "wb") as f:         f.write(uploaded_file.getbuffer())      try:         avg_hsv, img_rgb = extract_bubble_features(temp_path)         H_avg, S_avg, V_avg = avg_hsv          df_H  = pd.DataFrame({"H_corr":[H_avg]})         df_S  = pd.DataFrame({"S_corr":[S_avg]})         df_HS = pd.DataFrame({"H_corr":[H_avg], "S_corr":[S_avg]})          g_H  = max(model_H.predict(df_H)[0], 0)         g_S  = max(model_S.predict(df_S)[0], 0)         g_HS = max(model_HS.predict(df_HS)[0], 0)          glucose_weighted = 0.5*g_H + 0.3*g_S + 0.2*g_HS          st.image(img_rgb, caption="Uploaded Image")         st.subheader("Estimated Glucose (µM)")         st.write(f"**{glucose_weighted:.1f} µM**")          if 10 <= glucose_weighted <= 150:             st.success("✅ This is within expected healthy physiological saliva range.")         elif 150 < glucose_weighted <= 250:             st.warning("⚠️ Elevated saliva glucose detected. Consider confirmatory finger-prick or clinical assessment.")         else:             st.error("❌ Highly elevated glucose level detected. Seek clinical review.")      except Exception as e:         st.error(f"Error processing image: {e}") 
i wan the updated hough transform selection with filtering but i wan to keep the base line correction for saliva with corrected wording 

Understood. You want to retain the saliva baseline correction and calibration logic, but replace the older Hough detection with the improved pipeline and also correct the wording around the “corrected” variables.

The main issue in your current Streamlit section is that you are not actually applying the blank correction to the uploaded sample before prediction.

You trained on:

H_corr
S_corr
but you are predicting using raw:

H_avg
S_avg
This is scientifically inconsistent.

That needs to be fixed.

1) Updated bubble extraction function for Streamlit
Replace your current extract_bubble_features() with this:

# =====================================
# BUBBLE FEATURE EXTRACTION (UPDATED)
# =====================================
def extract_bubble_features(image_path, top_n=20):
    img = cv2.imread(image_path)

    if img is None:
        raise ValueError(f"Cannot read image {image_path}")

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Improve contrast
    gray_contrast = cv2.equalizeHist(gray)

    # Reduce noise
    gray_blur = cv2.GaussianBlur(gray_contrast, (5, 5), 1.2)

    # Improved Hough circle detection
    circles = cv2.HoughCircles(
        gray_blur,
        cv2.HOUGH_GRADIENT,
        dp=1.1,
        minDist=15,
        param1=80,
        param2=22,
        minRadius=8,
        maxRadius=25
    )

    if circles is None:
        raise ValueError("No bubbles detected.")

    circles = np.round(circles[0]).astype(int)

    candidates = []

    for x, y, r in circles:
        # Keep almost full bubble region
        r_roi = int(r * 0.95)

        Y, X = np.ogrid[:img_rgb.shape[0], :img_rgb.shape[1]]
        mask = (X - x)**2 + (Y - y)**2 <= r_roi**2

        roi_rgb = img_rgb[mask]

        if roi_rgb.size == 0:
            continue

        roi_hsv = rgb_to_hsv(roi_rgb / 255.0)

        h_mean, s_mean, v_mean = roi_hsv.mean(axis=0)

        # HSV pink filter
        if (
            252/360 <= h_mean <= 290/360 and
            s_mean >= 0.04 and
            v_mean >= 0.60
        ):
            score = (h_mean**8) * s_mean * v_mean * r_roi

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
2) Keep baseline correction (scientifically correct)
This part is correct in principle:

H_blank_deg = 12
S_blank_percent = 0.6

H_blank = H_blank_deg / 360.0
S_blank = S_blank_percent / 100.0
This represents saliva blank/reference subtraction.

I would rename the wording to be clearer:

# Blank saliva reference correction
H_blank_deg = 12
S_blank_percent = 0.6
This is better terminology than just “baseline.”

3) IMPORTANT FIX — apply correction to uploaded sample
This is the most important scientific correction.

Replace this block:

df_H  = pd.DataFrame({"H_corr":[H_avg]})
df_S  = pd.DataFrame({"S_corr":[S_avg]})
df_HS = pd.DataFrame({"H_corr":[H_avg], "S_corr":[S_avg]})
with:

# Apply saliva blank correction to uploaded sample
H_corr_sample = H_avg - H_blank
S_corr_sample = S_avg - S_blank

df_H = pd.DataFrame({
    "H_corr": [H_corr_sample]
})

df_S = pd.DataFrame({
    "S_corr": [S_corr_sample]
})

df_HS = pd.DataFrame({
    "H_corr": [H_corr_sample],
    "S_corr": [S_corr_sample]
})
This is essential.

Otherwise your prediction model is mathematically inconsistent.

4) Correct wording for results
Replace:

st.subheader("Estimated Glucose (µM)")
with:

st.subheader("Estimated Salivary Glucose Concentration (µM)")
This is scientifically more precise.

5) Add corrected HSV display (recommended)
I strongly recommend showing the corrected values for transparency:

st.write(f"Hue (corrected): {H_corr_sample:.4f}")
st.write(f"Saturation (corrected): {S_corr_sample:.4f}")
This helps validate the model.

6) Better interpretation wording
Replace this:

st.success("✅ This is within expected healthy physiological saliva range.")
with:

st.success("Within the expected physiological salivary glucose range.")
Replace:

st.warning("⚠️ Elevated saliva glucose detected. Consider confirmatory finger-prick or clinical assessment.")
with:

st.warning("Elevated salivary glucose estimated. Consider confirmatory blood glucose testing.")
Replace:

st.error("❌ Highly elevated glucose level detected. Seek clinical review.")
with:

st.error("Markedly elevated salivary glucose estimated. Clinical follow-up is advised.")
This wording is more clinical and avoids overclaiming.

Most important scientific issue fixed
The major correction is this:

H_corr_sample = H_avg - H_blank
S_corr_sample = S_avg - S_blank
Without this, your calibration model and prediction model are on different scales.

That was the main technical problem.


so im suppose to change this part?

Yes — this is the part you should definitely change, because it directly affects the glucose prediction.

You trained your regression models on blank-corrected features:

calibration_data["H_corr"]
calibration_data["S

# =====================================
# CALIBRATION MODEL
# =====================================
calibration_data = pd.DataFrame({
    "Glucose": [25, 50, 75, 100, 125],
    "H": [0.735825, 0.745060, 0.740868, 0.743964, 0.784736],
    "S": [0.085632, 0.090197, 0.090234, 0.100153, 0.130699]
})

# saliva reference correction
H_blank_deg = 10
S_blank_percent = 0.5

H_blank = H_blank_deg / 360.0
S_blank = S_blank_percent / 100.0

calibration_data["H_corr"] = calibration_data["H"] - H_blank
calibration_data["S_corr"] = calibration_data["S"] - S_blank

y_glucose = calibration_data["Glucose"].values

model_H  = LinearRegression().fit(calibration_data[["H_corr"]], y_glucose)
model_S  = LinearRegression().fit(calibration_data[["S_corr"]], y_glucose)
model_HS = LinearRegression().fit(calibration_data[["H_corr","S_corr"]], y_glucose)

Yes — this calibration block is already correct and should remain as it is.

The part you need to change is later, when predicting from the uploaded image.

Right now, the likely issue is that the uploaded sample is still being passed as raw H_avg and S_avg instead of corrected values.

So:

keep this calibration block

change the prediction block

Keep this block unchanged
This part is correct:

# saliva reference correction
H_blank_deg = 10
S_blank_percent = 0.5

H_blank = H_blank_deg / 360.0
S_blank = S_blank_percent / 100.0

calibration_data["H_corr"] = calibration_data["H"] - H_blank
calibration_data["S_corr"] = calibration_data["S"] - S_blank
This defines your saliva blank/reference subtraction.

Then you correctly train on:

model_H
model_S
model_HS
using corrected variables.

No need to edit this section further.

The section you must change
You need to edit the uploaded image prediction section.

Find this part:

df_H  = pd.DataFrame({"H_corr":[H_avg]})
df_S  = pd.DataFrame({"S_corr":[S_avg]})
df_HS = pd.DataFrame({"H_corr":[H_avg], "S_corr":[S_avg]})
and replace it with:

# apply saliva blank correction to uploaded sample
H_corr_sample = H_avg - H_blank
S_corr_sample = S_avg - S_blank

df_H = pd.DataFrame({
    "H_corr": [H_corr_sample]
})

df_S = pd.DataFrame({
    "S_corr": [S_corr_sample]
})

df_HS = pd.DataFrame({
    "H_corr": [H_corr_sample],
    "S_corr": [S_corr_sample]
})
This is the actual fix.

Why this matters
Your model was trained on:

H - H_blank
S - S_blank
So the uploaded sample must also be transformed the same way.

Otherwise you are feeding values on a different scale.

That can significantly bias prediction.

Formally, this is a feature space mismatch.

Simple analogy
If training used:

temperature in °C above room temperature
then prediction must also use:

measured temperature - room temperature
not raw temperature.

Same principle here.

So in short:

calibration section → leave as is

uploaded sample dataframe section → change to corrected sample values


no im correcting the prediction cause i have nothing to base agasint but my sample is already accounted for, so the corretion was taking a saliva sample and seeing how much more muted the colour was comapred then based on that shift my calibration so my new saliva samples need not be corrected

Under that experimental design, your current logic is defensible.

What you are describing is a one-time calibration transfer correction, not a per-sample blank subtraction.

That is an important distinction.

You are saying:

You first measured how saliva matrix effects shift the color relative to your standard glucose calibration

You quantified that offset

You embedded that offset into the calibration dataset

Future saliva samples are predicted directly against that saliva-adjusted calibration

If that is your method, then do not correct each new sample again.

Doing so would indeed double-correct.

So yes — in your case the prediction block should remain with raw extracted sample values.

Conceptually what you are doing
This is essentially matrix-adjusted calibration.

Mathematically:

standard calibration → saliva correction offset → adjusted calibration curve
Then prediction uses:

raw sample HSV → adjusted model
This is correct.

Why my earlier suggestion does not apply here
My earlier suggestion assumed this workflow:

raw calibration → raw sample → subtract blank every time
That is a classical analytical chemistry blank-correction workflow.

But your workflow is different.

You already transformed the calibration space itself:

calibration_data["H_corr"]
calibration_data["S_corr"]
So the model coefficients already incorporate saliva matrix suppression.

Therefore prediction should stay:

df_H  = pd.DataFrame({"H_corr":[H_avg]})
df_S  = pd.DataFrame({"S_corr":[S_avg]})
df_HS = pd.DataFrame({"H_corr":[H_avg], "S_corr":[S_avg]})
This is consistent with your stated method.

Suggested wording correction in code comments
Your current comment:

# saliva reference correction
is slightly misleading.

I would change it to:

# saliva matrix-adjusted calibration shift
This better reflects what was actually done.

Better explanation for documentation / GitHub
I would document it as:

# Calibration values were shifted using experimentally
# determined saliva matrix offset.
# Future saliva samples are predicted directly without
# additional per-sample correction.
This avoids confusion later.

Scientific caution
This approach is valid only if saliva matrix effects are stable across users and collection conditions.

That is the main assumption.

If saliva varies significantly with:

hydration

pH

food residue

mucin concentration

lighting conditions

then a fixed offset may not generalize.

So academically, note that this is a global matrix correction model.

That assumption should be acknowledged in your report or README.


so is this # =====================================
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
       
        sg_time = datetime.now(ZoneInfo("Asia/Singapore"))

        st.session_state.history.append({
            "Time": sg_time.strftime("%Y-%m-%d %H:%M:%S"),
            "Glucose": round(glucose_weighted, 1)
            }) correct?

Yes — given your calibration-transfer design, this prediction block is methodologically consistent.

You are using the saliva-adjusted calibration model and then predicting directly from the raw extracted HSV of future saliva samples.

That part is correct.

However, there are two technical issues in the Streamlit block that should be fixed.

1) The prediction logic is correct
This section is fine:

df_H  = pd.DataFrame({"H_corr":[H_avg]})
df_S  = pd.DataFrame({"S_corr":[S_avg]})
df_HS = pd.DataFrame({"H_corr":[H_avg], "S_corr":[S_avg]})
Even though the column names say H_corr and S_corr, in your workflow these are really:

raw sample HSV values mapped onto a saliva-adjusted model space
So mathematically this is acceptable.

The naming is a little confusing, but the computation is valid.

2) Better variable naming (recommended)
For clarity, I strongly recommend renaming:

H_corr
S_corr
to:

H_adj
S_adj
because these are calibration-adjusted features, not per-sample corrected values.

For example:

df_H  = pd.DataFrame({"H_adj":[H_avg]})
But this requires retraining the model with the same column names.

Not mandatory, just cleaner.

3) IMPORTANT BUG — history append placement
This part:

st.session_state.history.append({
    "Time": sg_time.strftime("%Y-%m-%d %H:%M:%S"),
    "Glucose": round(glucose_weighted, 1)
})
is currently outside the try block.

This is risky.

If bubble detection fails, glucose_weighted may not exist.

Then Streamlit will throw another error.

Move it inside the try block after successful prediction.

Correct structure
Use this:

with tab2:
    uploaded_file = st.file_uploader(
        "Upload an image of your saliva bubbles",
        type=["jpg", "png", "jpeg"]
    )

    if uploaded_file:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        temp_path = f"temp_{timestamp}.jpg"

        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        try:
            avg_hsv, img_rgb = extract_bubble_features(temp_path)
            H_avg, S_avg, V_avg = avg_hsv

            df_H = pd.DataFrame({"H_corr": [H_avg]})
            df_S = pd.DataFrame({"S_corr": [S_avg]})
            df_HS = pd.DataFrame({
                "H_corr": [H_avg],
                "S_corr": [S_avg]
            })

            g_H = max(model_H.predict(df_H)[0], 0)
            g_S = max(model_S.predict(df_S)[0], 0)
            g_HS = max(model_HS.predict(df_HS)[0], 0)

            glucose_weighted = 0.5*g_H + 0.3*g_S + 0.2*g_HS

            st.image(img_rgb, caption="Uploaded Image")
            st.subheader("Estimated Salivary Glucose (µM)")
            st.write(f"**{glucose_weighted:.1f} µM**")

            if 10 <= glucose_weighted <= 150:
                st.success("Within the expected physiological salivary range.")
            elif 150 < glucose_weighted <= 250:
                st.warning("Elevated salivary glucose estimated. Consider confirmatory blood glucose testing.")
            else:
                st.error("Unusual glucose estimate detected. Please upload another image.")

            sg_time = datetime.now(ZoneInfo("Asia/Singapore"))

            st.session_state.history.append({
                "Time": sg_time.strftime("%Y-%m-%d %H:%M:%S"),
                "Glucose": round(glucose_weighted, 1)
            })

        except Exception as e:
            st.error(f"No bubbles detected in image. Please upload another image. ({e})")
Final academic assessment
The model logic is correct for your stated calibration approach.

The main fix was structural robustness of the Streamlit code, especially the history logging location.


back to my python code X_HS = data[["H","S"]]
y = data["Glucose_uM"].values

model_HS = LinearRegression().fit(X_HS, y)
y_pred = model_HS.predict(X_HS)

r2 = r2_score(y, y_pred)
rmse = np.sqrt(mean_squared_error(y, y_pred))

print(f"H+S → Glucose: R² = {r2:.4f}, RMSE = {rmse:.4f} µM")
X = data["Glucose_uM"].values.reshape(-1,1)

results = []

for ch in ["R","G","B","H","S","V"]:
    y = data[ch].values
    model = LinearRegression().fit(X, y)
    y_pred = model.predict(X)

    r2 = r2_score(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))

    results.append([ch, r2, rmse])

results_df = pd.DataFrame(results, columns=["Channel","R2","RMSE"])
results_df.style.format({
    "R2": "{:.4f}",
    "RMSE": "{:.4f}"
})

how to interpret the 2 different RMSE?

Good question. These two RMSE values are not directly the same quantity, even though both are called RMSE.

The key difference is what variable is being predicted.

1) RMSE from H+S → Glucose
This block:

X_HS = data[["H","S"]]
y = data["Glucose_uM"].values

model_HS = LinearRegression().fit(X_HS, y)
y_pred = model_HS.predict(X_HS)

rmse = np.sqrt(mean_squared_error(y, y_pred))
gives:

RMSE in units of glucose (µM)
This is the prediction error of your actual analytical model.

For example, if:

RMSE = 6.5
that means:

predicted glucose is typically off by about ±6.5 µM
This is the most important RMSE for your application.

Interpretation
Example:

actual glucose = 100 µM
predicted = 93 µM

error = 7 µM

RMSE summarizes the average magnitude of such errors.

So this directly reflects estimation accuracy.

2) RMSE in results_df
This block:

for ch in ["R","G","B","H","S","V"]:
fits:

Glucose → Channel
For example:

Glucose → H
Glucose → S
So here the RMSE is in the units of the channel itself, not glucose.

Example
For H:

RMSE = 0.0111
means:

predicted hue differs by ~0.0111 hue units
This is NOT µM.

It is hue-space error.

Similarly for S:

RMSE = 0.0082
means saturation-space error.

Important distinction
So:

Model 1
H,S → glucose
RMSE unit:

µM
Use this for app accuracy.

Model 2
glucose → H
glucose → S
RMSE unit:

channel units
Use this for biomarker linearity analysis.

Practical meaning
The second table helps answer:

Which image channel correlates best with glucose?
That is why H and S had better R².

This is feature evaluation.

The first RMSE answers:

How accurate is my final glucose estimator?
This is predictive performance.

Academic interpretation
You can phrase it like this:

The channel-wise regression RMSE values quantify the linearity of each optical feature with glucose concentration, whereas the multivariate H+S regression RMSE represents the actual predictive error of glucose estimation in µM.

That distinction is the important one.


oh then i want the same RSME for H and S 

Got it. You want the prediction error of glucose using only H or only S, just like you did for H+S. That way, the RMSE is in µM, not in channel units.

Here’s how you can do it:

from sklearn.metrics import r2_score, mean_squared_error
import numpy as np

# Single-channel glucose prediction
X_H = data[["H"]]
X_S = data[["S"]]
y = data["Glucose_uM"].values

# H → Glucose
model_H = LinearRegression().fit(X_H, y)
y_pred_H = model_H.predict(X_H)
r2_H = r2_score(y, y_pred_H)
rmse_H = np.sqrt(mean_squared_error(y, y_pred_H))

# S → Glucose
model_S = LinearRegression().fit(X_S, y)
y_pred_S = model_S.predict(X_S)
r2_S = r2_score(y, y_pred_S)
rmse_S = np.sqrt(mean_squared_error(y, y_pred_S))

print(f"H → Glucose:   R² = {r2_H:.4f}, RMSE = {rmse_H:.4f} µM")
print(f"S → Glucose:   R² = {r2_S:.4f}, RMSE = {rmse_S:.4f} µM")
✅ This will give you the RMSE in µM for H alone and S alone, directly comparable to the H+S → Glucose RMSE.

You can then compare RMSE for H, S, and H+S to see which feature or combination predicts glucose most accurately.

If you want, I can also rewrite your entire results table to include H, S, and H+S RMSE all in µM for easy comparison. Do you want me to do that?


so how i interpret the prev non glucose RMSE to justify that i chose H and S to be the one to use fo rglucose>

Exactly — the old RMSE values in your results_df (the ones in channel units) are not directly about glucose accuracy, but they are still very useful for feature selection. Here’s how to interpret them academically:

1. What those RMSE values mean
You regressed Glucose → Channel:

Glucose_uM → R, G, B, H, S, V
RMSE here is in channel units, e.g., hue fraction (0–1) or saturation fraction (0–1), not µM.

A smaller RMSE means that the channel changes linearly and consistently with glucose.

Conversely, a large RMSE means that the channel is noisy or non-linear with glucose.

2. How this justifies choosing H and S
Look at the table you shared:

Channel	R2	RMSE
R	0.0013	8.25
G	0.0672	9.59
B	9.30e-08	9.45
H	0.6029	0.0111
S	0.7485	0.0082
V	1.02e-08	0.0371
R² indicates linear correlation with glucose:

H: 0.60 → good correlation

S: 0.75 → best correlation

Other channels: near zero → no correlation

RMSE indicates how tightly the channel tracks glucose:

H: 0.0111 → tight

S: 0.0082 → very tight

R/G/B/V: much higher or meaningless

Therefore, H and S are the channels that consistently reflect glucose concentration, while RGB or V are too noisy or insensitive.

3. How to justify feature choice
You can write this as:

“Hue (H) and saturation (S) were selected as predictive features because they exhibit the highest linear correlation with glucose concentration (R² = 0.60 and 0.75, respectively) and the lowest channel RMSE, indicating a strong and consistent optical response to glucose. Other channels (R, G, B, V) showed negligible correlation or high variability, making them unsuitable for accurate estimation.”

4. Linking to actual glucose prediction
Once you combine H and S in a multivariate model (H+S → Glucose), you are using the best predictors identified via channel R² and RMSE, and then you evaluate glucose prediction in µM, which is the practical metric for your assay.

In short:

Channel RMSE / R² → feature selection (H & S)

H+S → Glucose RMSE → predictive accuracy in µM

If you want, I can make a diagram-style interpretation showing:
Channel RMSE → selection → H+S → Glucose RMSE
so you can use it in your GitHub README or lab report. This makes the reasoning very clear.

Do you want me to do that?


write as a paragrapgh for my resutls and discussion paragraph to enahnce the current To enable quantitative interpretation of the colorimetric response, digital image processing techniques were applied to extract colour information from the droplets.

Figure 1: A Image converted to grayscale B image contrast enhanced C image Blurred D Image before HSV filtering

Microbubbles within the droplets were automatically detected using the Hough Transform algorithm, which identifies circular features within the captured images. Once droplets were identified, regions corresponding to each droplet were isolated and analysed.
For each droplet, average RGB and HSV colour values were computed. These values were then aggregated across all droplets within each image to obtain representative colour parameters corresponding to each glucose concentration.

Glucose Concentration (µM)	Images of droplets formed	RGB Values
(to 1 decimal place)	HSV Values
(to 3 decimal place)
		R	G	B	H	S	V
25	 	184.2	177.4	193.2	0.736	0.086	0.758
50	 	185.1	176.6	193.8	0.745	0.090	0.760
75	 	183.7	176.2	193.1	0.741	0.090	0.757
100	 	201.8	192.0	212.7	0.744	0.100	0.834
125	 	176.9	160.6	183.7	0.785	0.131	0.721
Table 1: Data Collected via digital image processing

To evaluate the predictive capability of each colour parameter, linear regression analysis was performed and the resulting coefficient of determination (R²) and root mean square error (RMSE) values were calculated.

Channel	R2	RMSE
R	0.0013	8.2515
G	0.0672	9.5904
B	0.0000	9.4545
H	0.6029	0.0111
S	0.7485	0.0082
V	0.0000	0.0371
Table 2: Results of linear regression

Analysis of the extracted colour parameters indicates that the RGB channels exhibit very weak correlation with glucose concentration. This suggests that the raw red, green and blue intensity values are not reliable indicators of the colorimetric reaction. One likely reason is that RGB values are highly sensitive to variations in lighting conditions, camera exposure as well as reflections within the microfluidic chip. These factors alter pixel intensity without necessarily reflecting changes in the chemical reaction, thereby reducing the predictive power of RGB-based measurements.

In contrast, the HSV colour space demonstrated substantially stronger correlations with glucose concentration, particularly for the saturation (S) parameter. Hue (H) also showed moderate correlation Unlike RGB, HSV separates colour properties into three components. Hue represents the dominant colour wavelength, saturation represents colour intensity and value represents brightness. Because hue and saturation describe colour characteristics independently of illumination intensity, they are less affected by variations in lighting conditions during image capture.

The superior performance of saturation in this study suggests that the colorimetric reaction primarily influences the intensity of the developed colour rather than the brightness of the image. As glucose concentration increases, the enzymatic oxidation reaction involving glucose oxidase and horseradish peroxidase produces a stronger coloured product from Amplex Red, resulting in increased colour saturation within the droplets. Consequently, saturation provides a more direct representation of the reaction progress compared with raw RGB intensities.

Channel	R2	RMSE
H	0.6029	0.0111
S	0.7485	0.0082
H+S	0.0000	0.0371
Table 2: Results of linear regression

Combining the Hue and Saturation (H+S) parameters in a multivariate regression model significantly improved prediction accuracy, yielding an R² value of 0.993 and an extremely low RMSE. This result suggests that both colour tone and colour intensity contribute complementary information for distinguishing glucose concentrations.

The strong predictive performance of the H+S model highlights the effectiveness of HSV-based colour analysis for microfluidic biosensing applications, particularly in imaging environments where illumination conditions may vary.

Here’s a polished paragraph you can insert into your Results and Discussion section to integrate the feature selection reasoning and RMSE interpretation:

Analysis of the linear regression results across individual colour channels indicates that the RGB channels exhibit very weak correlation with glucose concentration, with R² values near zero and high RMSE in channel units, confirming that raw red, green, and blue intensities are unreliable indicators of the colorimetric reaction. In contrast, the HSV colour space demonstrated substantially stronger correlation, particularly for the saturation (S) parameter (R² = 0.7485, RMSE = 0.0082) and hue (H) (R² = 0.6029, RMSE = 0.0111), suggesting that these features track the enzymatic reaction more consistently. The low RMSE values in channel units indicate that H and S vary linearly and reproducibly with glucose concentration, making them the most informative optical features. Accordingly, a multivariate regression model combining H and S (H+S → Glucose) was developed to predict glucose concentration in µM, yielding a near-perfect R² of 0.993 and a very low RMSE in µM. This demonstrates that while individual HSV channels are useful for feature selection, the combined H+S model provides complementary information from both colour tone (H) and intensity (S), enabling highly accurate estimation of glucose concentration. These findings justify the selection of H and S as the primary predictors for quantitative analysis in microfluidic droplet-based biosensing, particularly under variable illumination conditions where RGB intensities are unreliable.

This paragraph clearly explains:

Why RGB is poor (low R², high channel RMSE)

Why H and S were chosen (linear, reproducible, low RMSE)

How H+S improves predictive accuracy (RMSE in µM)

The scientific rationale connecting colour physics to enzyme reaction

You can drop it after your existing discussion of Tables 1 and 2.

If you want, I can also merge it with your figure captions and previous paragraph to make one cohesive Results section ready for publication. Do you want me to do that?


rerewrite the entire paragraph in better academic standard, note that 3.2.2 Data Collection 
Saliva samples were collected from a volunteer where the participant expectorated directly into sterile collection tubes. No fasting, mouth rinsing or centrifugation steps were performed in order to evaluate the assay under minimally processed, real world sampling conditions.

Collected saliva samples were directly introduced into the prepared reagent master mix. The enzymatic reaction between endogenous glucose and the reagent system resulted in formation of the chromogenic resorufin product.

Images of the resulting colorimetric response were captured using an iPhone 14 smartphone camera with the flash enabled to provide consistent illumination and reduce variation caused by ambient lighting conditions.

Although controlled illumination systems such as LED-based imaging enclosures are commonly employed in smartphone-based assays to improve reproducibility, the use of smartphone flash illumination provides a simplified and accessible approach suitable for point-of-care applications.


3.2.3 Data Analysis and Processing 
Digital colour analysis in microfluidic colorimetric assays involves converting visual information into quantitative numerical values using established colour models. In this study, the RGB colour model was employed to represent each pixel as a combination of Red (R), Green (G) and Blue (B) channels, with intensity values ranging from 0 to 255. 
Complementarily, the HSV colour model was applied, providing Hue (H), Saturation (S) and Value (V) parameters. Hue reflects the colour tone and is expressed as an angular position on the colour wheel ranging from 0° to 360°. Saturation captures the purity of the colour and Value represents brightness. The HSV model is particularly useful for detecting subtle colour changes corresponding to analyte concentration because it separates chromatic information from illumination intensity and is therefore less sensitive to variations in lighting conditions.

Images of the microfluidic assay were captured under consistent illumination conditions using a smartphone with flash enabled to minimize ambient light fluctuations. Each image was first converted to grayscale and subjected to Gaussian blurring to reduce noise and facilitate robust feature detection. 

Circular regions corresponding to microbubbles were automatically identified using the Hough Transform, which detects circular features based on geometric constraints and edge gradients. This method is well suited to the approximately spherical morphology of microfluidic droplets and offers improved robustness compared to simple intensity-based segmentation in the presence of reflections and uneven illumination.

Each detected bubble was then evaluated in HSV space to select those corresponding to the characteristic pink hue of the reaction product. A non-overlapping selection strategy was applied to avoid redundant measurements and the radii of selected bubbles were slightly reduced to exclude peripheral pixels thereby ensuring accurate representation of the reaction mixture. This selective filtering reduces the inclusion of non-reactive regions and imaging artefacts. A scoring-based ranking approach was subsequently used to prioritise the most representative bubbles.

For each selected bubble, average RGB and HSV values were computed and aggregated per image to obtain representative colour features for each glucose concentration. These values were compiled into a structured dataset that combined the known glucose concentration with the image derived colour features. Univariate linear regression models were first applied to each channel individually to evaluate the strength of correlation between colour intensity and glucose levels, with performance metrics including the coefficient of determination ("R" 2) and the root mean square error (RMSE) used to assess predictive accuracy. Subsequently, multivariate regression incorporating the most responsive channels (H and S) was performed to improve model robustness and account for colour interactions.

This workflow effectively transforms raw images into quantitative metrics that capture the colorimetric response of the assay. By averaging over multiple regions of interest and applying selective weighting of responsive channels, the method minimizes variability and provides a reproducible approach for glucose prediction. The combination of automated bubble detection with HSV-based colour filtering and regression-based analysis ensures that colorimetric information from saliva-based microfluidic assays is translated into precise, statistically meaningful glucose measurements.

3.2.4 Application Development 
Building on this calibrated dataset, a Streamlit-based smartphone application was developed to allow real time glucose estimation from uploaded images. The application follows the same fundamental workflow as the initial analysis. Images are read then converted to RGB and grayscale and blurred to reduce noise before detecting via the Hough Transform. For saliva samples, an additional baseline correction step compensates for muted or less saturated coloration inherent to biological samples, ensuring that only the colour change due to glucose is analysed. Detected bubbles are filtered and scored based on pink hue intensity before averaging the top N bubbles to extract representative H and S values. These baseline corrected values are then input into the pretrained regression models to predict glucose concentration, with a weighted combination of H and S predictions enhancing accuracy.

The final glucose estimate is displayed alongside the uploaded image, enabling rapid interpretable measurement directly on a smartphone. By linking the initial calibration using known pure glucose samples to the real time saliva assay, this workflow ensures that colorimetric information is reliably translated into quantitative glucose measurements while accounting for sample specific variations and image artifacts. was already mentioned in the methodology

Here’s a polished, academically structured version of your Results and Discussion paragraph that integrates your methodology, data analysis, and justification for selecting H and S for glucose prediction:

To quantitatively interpret the colorimetric response, digital image analysis was applied to extract numerical colour information from the microfluidic droplets. The images were first converted to grayscale and subjected to Gaussian blurring to reduce noise, enhancing the robustness of subsequent feature detection. Microbubbles within each droplet were automatically identified using the Hough Transform, which leverages geometric constraints and edge gradients to detect circular features. This method is well suited to the approximately spherical morphology of the droplets and provides greater robustness against reflections and uneven illumination compared with simple intensity-based segmentation.

Once detected, each bubble was evaluated in HSV space to identify those corresponding to the characteristic pink colour of the resorufin product. A non-overlapping selection strategy was applied to avoid redundant measurements, and peripheral pixels were excluded through minor radius adjustments to ensure accurate representation of the reaction. Scoring-based ranking was subsequently used to prioritise the most representative bubbles. For each selected bubble, average RGB and HSV values were computed and aggregated across all bubbles within an image to yield representative colour features corresponding to each glucose concentration.

Univariate regression analysis of individual channels revealed that the RGB components exhibited very weak correlation with glucose concentration (R² ≈ 0, high RMSE in channel units), indicating that raw intensity values are strongly affected by lighting conditions and imaging artefacts and therefore do not reliably track the enzymatic reaction. In contrast, the HSV parameters showed substantially stronger correlation, with saturation (S) demonstrating the highest linearity (R² = 0.7485, RMSE = 0.0082) and hue (H) showing moderate correlation (R² = 0.6029, RMSE = 0.0111). These low RMSE values in channel units indicate that H and S vary reproducibly with glucose concentration and provide robust optical markers for reaction progress.

Building on these findings, a multivariate regression model combining H and S (H+S → Glucose) was implemented to predict glucose concentration in µM. This approach yielded a near-perfect coefficient of determination (R² = 0.993) and very low RMSE, demonstrating that combining hue and saturation provides complementary information about both colour tone and intensity, substantially improving prediction accuracy over single-channel models. The strong predictive performance of the H+S model highlights the effectiveness of HSV-based colour analysis in translating microfluidic assay images into precise and reproducible quantitative glucose measurements. This approach is particularly advantageous in smartphone-based imaging environments, where variations in illumination and sample-specific effects can compromise the reliability of raw RGB intensities.

This version:

Integrates image preprocessing and bubble detection steps with analytical reasoning.

Explains why RGB is unreliable and why H and S were chosen.

Shows the link between single-channel evaluation (R², RMSE in channel units) and predictive performance (H+S → Glucose in µM).

Maintains an academic tone suitable for Results and Discussion.

If you want, I can also rewrite the whole Results section, integrating Figures 1 and 2, Tables 1–2, and the H+S discussion so it reads as one cohesive, publication-ready section. Do you want me to do that next?


add more about how the grayscale enahnce constrast and blurring help to sharpen the image in methodolog?

Here’s a refined version of your methodology section that explicitly explains the purpose of grayscale conversion, contrast enhancement, and blurring:

Images of the microfluidic assay were captured under consistent illumination conditions using a smartphone with flash enabled to minimize ambient light fluctuations. Each image was first converted to grayscale, which simplifies the image by reducing it to intensity values while preserving structural information. This facilitates subsequent processing by removing colour information that is not required for bubble detection. To further enhance the features of interest, contrast enhancement was applied, increasing the distinction between bubble edges and the background, thereby improving the visibility of faint or partially developed droplets. Subsequently, Gaussian blurring was employed to reduce high-frequency noise and small artefacts in the image, which might otherwise lead to false-positive detections during automated feature identification. These preprocessing steps collectively sharpen the image representation of microbubbles, improving the robustness and accuracy of the Hough Transform in detecting circular droplets.

Circular regions corresponding to microbubbles were then automatically identified using the Hough Transform, which detects circular features based on edge gradients and geometric constraints. This approach is well suited to the approximately spherical morphology of microfluidic droplets and is more robust than simple intensity-based segmentation, particularly in the presence of reflections or uneven illumination.

This explicitly justifies each preprocessing step:

Grayscale → simplifies data, retains structure for detection.

Contrast enhancement → strengthens edges, improves visibility of bubbles.

Blurring → reduces noise, prevents false detections.

If you want, I can now rewrite your full “Data Analysis and Processing” section integrating this explanation seamlessly with bubble detection, HSV filtering, and regression analysis. This will make your methods read cohesively for a publication-ready manuscript. Do you want me to do that?


is ti correct to say that For each selected bubble, average RGB and HSV values were computed and aggregated across all bubbles within an image to yield representative colour features corresponding to each glucose concentration. Each image has a seleected of N bubbles up to 20

Yes, that statement is correct, but it can be clarified for precision and academic style. You want to make it clear that:

Each image may have up to 20 bubbles selected.

For each bubble, you compute average RGB and HSV values.

Then, you aggregate across all selected bubbles in that image to get a single representative feature vector.

A more precise version would be:

"For each detected bubble, the average RGB and HSV values were computed. For each image, these values were then aggregated across all selected bubbles—up to a maximum of 20 per image—to obtain a single representative set of colour features corresponding to the glucose concentration of that sample."

This wording makes it clear that the aggregation happens per image, that N ≤ 20 bubbles are used, and that the resulting feature vector represents the glucose-dependent colour response.

If you want, I can rewrite the entire Results paragraph including this clarification and integrate it seamlessly with the regression and H+S explanation.


Channel	R2	RMSE
H	0.6029	22.2783
S	0.7485	17.7316
H+S	0.8476	13.8009
Table 2: Results of linear regression

Combining the Hue and Saturation (H+S) parameters in a multivariate regression model significantly improved prediction accuracy, yielding an R² value of 0.993 and an extremely low RMSE. This result suggests that both colour tone and colour intensity contribute complementary information for distinguishing glucose concentrations.

There’s an inconsistency in the text you quoted. Your table shows:

Channel	R²	RMSE
H	0.6029	22.2783
S	0.7485	17.7316
H+S	0.8476	13.8009
…but the paragraph says:

“Combining the Hue and Saturation (H+S) parameters in a multivariate regression model significantly improved prediction accuracy, yielding an R² value of 0.993 and an extremely low RMSE.”

This doesn’t match your table, which shows R² = 0.8476 and RMSE = 13.8 µM for H+S. The paragraph should be updated to accurately reflect your data. Here’s a corrected version:

Combining the Hue and Saturation (H+S) parameters in a multivariate regression model improved prediction accuracy compared with single-channel models, yielding an R² value of 0.848 and an RMSE of 13.8 µM. This indicates that both colour tone (H) and colour intensity (S) contribute complementary information for distinguishing glucose concentrations. The improvement over the individual H (R² = 0.603, RMSE = 22.3 µM) and S (R² = 0.748, RMSE = 17.7 µM) models demonstrates that multivariate analysis can capture synergistic effects between hue and saturation, providing a more reliable quantitative representation of the enzymatic colorimetric response.

If you want, I can rewrite the entire Results and Discussion paragraph to integrate this corrected R²/RMSE discussion with the image processing and bubble selection explanation for a cohesive, publication-ready section. Do you want me to do that?


yes the table is correct reqriete the apargarph 

Here’s a fully rewritten, academically polished Results and Discussion paragraph that aligns with your table and methodology, integrating bubble selection, image preprocessing, HSV analysis, and regression interpretation:

To quantitatively interpret the colorimetric response of the saliva-based microfluidic assay, digital image analysis was performed on images captured under controlled smartphone flash illumination. Each image was first converted to grayscale to simplify structural information, followed by contrast enhancement to increase the distinction between bubble edges and the background. Gaussian blurring was then applied to reduce high-frequency noise and minor artefacts, thereby sharpening the features of interest and improving the robustness of automated bubble detection.

Microbubbles were subsequently identified using the Hough Transform, which leverages geometric constraints and edge gradients to detect circular features. This method is particularly well suited to the approximately spherical morphology of microfluidic droplets and provides greater robustness than simple intensity-based segmentation in the presence of reflections or uneven illumination. Each detected bubble was evaluated in HSV space to select those corresponding to the characteristic pink hue of the enzymatic reaction product. A non-overlapping selection strategy was applied to avoid redundant measurements, and a slight reduction in bubble radius excluded peripheral pixels, ensuring that the analysis focused on the reactive core. Up to 20 bubbles per image were selected, and for each, average RGB and HSV values were computed and then aggregated to generate a representative feature vector for that sample.

Univariate regression analysis of the individual colour channels revealed that the RGB components exhibited very weak correlation with glucose concentration (R² ≈ 0, high RMSE), indicating that raw intensity values are highly susceptible to lighting variations and imaging artefacts. In contrast, the HSV parameters demonstrated stronger linear relationships, with saturation (S) showing the highest correlation (R² = 0.7485, RMSE = 17.73 µM) and hue (H) exhibiting moderate correlation (R² = 0.6029, RMSE = 22.28 µM). These findings indicate that both hue and saturation vary consistently with glucose concentration, reflecting changes in colour tone and intensity produced by the enzymatic reaction.

A multivariate regression model combining H and S (H+S → Glucose) further improved predictive performance, achieving an R² of 0.8476 and an RMSE of 13.80 µM. The improvement over individual channel models demonstrates that hue and saturation provide complementary information, capturing both chromatic and intensity-related aspects of the colorimetric response. This result highlights the suitability of HSV-based analysis for quantitative microfluidic biosensing, enabling robust and reproducible estimation of glucose concentrations under variable imaging conditions.

This version:

Clarifies why grayscale, contrast enhancement, and blurring were used.

Explains bubble detection and selection, including the N = 20 limit.

Justifies selection of H and S based on R² and RMSE from your table.

Integrates H+S multivariate regression as a more accurate predictive model.

Maintains formal academic style suitable for publication.

If you like, I can also rewrite the full “Tables 1–2 and Figures” section together so the paragraph and data presentation are seamless and ready for your manuscript. Do you want me to do that?


as such what should i assgin my weightages to be?

Given your table and R²/RMSE results:

Channel	R²	RMSE (µM)
H	0.6029	22.28
S	0.7485	17.73
H+S	0.8476	13.80
You want a weighted combination of H and S predictions for your Streamlit app. A rational approach is:

Assign higher weight to the channel with lower RMSE / higher predictive power, because it is more reliable.

Use relative performance to guide the weighting.

A simple calculation based on RMSE inverse:

w
H
=
1
/
R
M
S
E
H
(
1
/
R
M
S
E
H
)
+
(
1
/
R
M
S
E
S
)
,
w
S
=
1
−
w
H
w 
H
​
 = 
(1/RMSE 
H
​
 )+(1/RMSE 
S
​
 )
1/RMSE 
H
​
 
​
 ,w 
S
​
 =1−w 
H
​
 
Plug in your numbers:

1
/
R
M
S
E
H
=
1
/
22.28
≈
0.0449
1/RMSE 
H
​
 =1/22.28≈0.0449

1
/
R
M
S
E
S
=
1
/
17.73
≈
0.0564
1/RMSE 
S
​
 =1/17.73≈0.0564

Total = 0.0449 + 0.0564 = 0.1013

w
H
=
0.0449
/
0.1013
≈
0.443
w 
H
​
 =0.0449/0.1013≈0.443
w
S
=
0.0564
/
0.1013
≈
0.557
w 
S
​
 =0.0564/0.1013≈0.557
So a weighted glucose prediction could be:

G
l
u
c
o
s
e
w
e
i
g
h
t
e
d
=
0.44
⋅
g
H
+
0.56
⋅
g
S
Glucose 
weighted
​
 =0.44⋅g 
H
​
 +0.56⋅g 
S
​
 
This gives slightly more influence to S, which is consistent with its lower RMSE.

If you want, you can also include H+S multivariate prediction as a third component, e.g., 0.2 weight, and rescale H and S accordingly.

Using this method is quantitatively justified and aligns with your regression results.

If you want, I can write the exact Python code snippet for your Streamlit app to compute the weighted glucose using these weights. Do you want me to do that?


should i include all 3? if yes why and what should the weightages be?

Yes, you can include all three (H, S, H+S) in your weighted glucose prediction, but it should be justified:

Why include all three:
H (Hue) – captures the color tone of the reaction, moderately correlated with glucose (R² = 0.603, RMSE = 22.28 µM).

S (Saturation) – captures the color intensity, strongest single-channel predictor (R² = 0.7485, RMSE = 17.73 µM).

H+S (Multivariate model) – captures combined information and interactions between hue and saturation, giving the best overall performance (R² = 0.8476, RMSE = 13.80 µM).

By including all three, you are leveraging:

H for color tone

S for intensity

H+S for synergistic interaction

This can smooth out errors from individual channels and produce a more robust prediction.

Suggested weighting strategy:
A rational approach is to use inverse RMSE as weight, giving more influence to more accurate predictors:

w
i
=
1
/
R
M
S
E
i
∑
1
/
R
M
S
E
j
w 
i
​
 = 
∑1/RMSE 
j
​
 
1/RMSE 
i
​
 
​
 
Plugging your RMSE values:

1
/
R
M
S
E
H
=
1
/
22.28
≈
0.0449
1/RMSE 
H
​
 =1/22.28≈0.0449

1
/
R
M
S
E
S
=
1
/
17.73
≈
0.0564
1/RMSE 
S
​
 =1/17.73≈0.0564

1
/
R
M
S
E
H
+
S
=
1
/
13.80
≈
0.0725
1/RMSE 
H+S
​
 =1/13.80≈0.0725

Total = 0.0449 + 0.0564 + 0.0725 = 0.1738

Normalized weights:

w
H
=
0.0449
/
0.1738
≈
0.258
w 
H
​
 =0.0449/0.1738≈0.258
w
S
=
0.0564
/
0.1738
≈
0.325
w 
S
​
 =0.0564/0.1738≈0.325
w
H
+
S
=
0.0725
/
0.1738
≈
0.417
w 
H+S
​
 =0.0725/0.1738≈0.417
So the weighted glucose prediction formula becomes:

G
l
u
c
o
s
e
w
e
i
g
h
t
e
d
=
0.26
⋅
g
H
+
0.33
⋅
g
S
+
0.42
⋅
g
H
+
S
Glucose 
weighted
​
 =0.26⋅g 
H
​
 +0.33⋅g 
S
​
 +0.42⋅g 
H+S
​
 
H+S gets the largest weight, consistent with its lowest RMSE.

S is second most important, and H contributes moderately.

If you want, I can rewrite your Streamlit code to include all three channels using these calculated weights so it’s ready to deploy. Do you want me to do that?


write in text onli why i included all 3 and give weightabe 0.2 0.3 0.5 to H S and H+S

In the glucose prediction model, all three colour parameters—Hue (H), Saturation (S), and the combined multivariate H+S—were included to leverage their complementary information. Hue captures the colour tone of the enzymatic reaction, providing moderate correlation with glucose levels. Saturation reflects the intensity of the developed colour and exhibits the strongest single-channel predictive performance. The H+S multivariate model incorporates both hue and saturation simultaneously, capturing synergistic interactions that improve overall prediction accuracy. Assigning weights of 0.2, 0.3, and 0.5 to H, S, and H+S, respectively, ensures that the final glucose estimate gives proportionally greater influence to the most robust predictor (H+S), while still incorporating the informative contributions of the individual H and S channels.


okay now possible to have a darker shade of green for the tabs? and another colour whihc complinets the light green for the title?

{}

{}

We can choose colors manually for now.

For the tabs: a darker shade of green could be something like #2E7D32 (a deep forest green).

For the title to complement a light green, a muted yellow or soft gold works well, for example #FBC02D.

So in your Streamlit app, you could set:

Tabs: #2E7D32

Title or accent elements: #FBC02D

This pairing gives a strong contrast while remaining visually harmonious.

If you want, I can give you the exact Streamlit code snippet to apply these colors to your tabs and title.


yes do so

Here’s how you can apply custom colors in your Streamlit app for the tabs and the title using CSS styling:

import streamlit as st

# Set page title
st.set_page_config(page_title="Saliva Glucose Estimator", layout="wide")

# Apply custom CSS for title and tabs
st.markdown(
    """
    <style>
    /* Title color */
    .css-10trblm {  /* This class targets the main title; may vary depending on Streamlit version */
        color: #FBC02D;  /* Soft gold/yellow */
    }

    /* Tabs background color and font color */
    div[role="tab"] {
        background-color: #2E7D32 !important;  /* Dark green */
        color: white !important;
        font-weight: bold;
    }

    /* Tabs selected highlight */
    div[role="tab"].st-bf {
        background-color: #66BB6A !important;  /* Slightly lighter green for selected tab */
        color: white !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# App title
st.title("Glucose Estimator from Microfluidic Bubbles")

# Tabs example
tab1, tab2, tab3 = st.tabs(["🏠 Home", "🧪 Reagent Set A", "🍽 Diet"])

with tab1:
    st.write("Home content here")

with tab2:
    st.write("Reagent Set A content here")

with tab3:
    st.write("Diet content here")
Notes:
#2E7D32 → dark green for unselected tabs.

#66BB6A → lighter green for the active tab, giving contrast.

#FBC02D → soft gold for the title to complement the green.

You may need to inspect the CSS classes in your Streamlit version if .css-10trblm doesn’t target the title; the rest (tabs using role="tab") should work consistently.

This will make your app visually cohesive with a green tab theme and a complementary gold title.

I can also suggest a full color palette for your app so all buttons, tabs, and headings match nicely. Do you want me to do that?


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

st.title("🧪 Saliva Glucose Monitoring Platform") 5his is the current code

You can adapt your current CSS to include the darker green tabs and a complementary accent color for the title like this:

st.markdown("""
<style>
/* App background */
.stApp {
    background: #eaf8ea;  /* light green background */
    color: black;
}

/* Default text color */
html, body, [class*="css"] {
    color: black !important;
}

/* Tab text and background */
button[data-baseweb="tab"] {
    background-color: #2E7D32 !important;  /* dark green for tabs */
    color: white !important;
    font-weight: 600;
}

/* Selected tab highlight */
button[data-baseweb="tab"][data-selected="true"] {
    background-color: #66BB6A !important;  /* lighter green for active tab */
    color: white !important;
}

/* Title color */
h1 {
    color: #FBC02D !important;  /* complementary soft gold */
}
</style>
""", unsafe_allow_html=True)

st.title("🧪 Saliva Glucose Monitoring Platform")
Key Changes:
#2E7D32 → darker green for tabs.

#66BB6A → lighter green for active tab.

#FBC02D → gold accent for title.

Your app background remains light green (#eaf8ea) for contrast.

This keeps your existing color scheme but adds contrast and hierarchy for tabs and headings.

If you want, I can also tweak it so buttons and other UI elements match the green/gold palette for a fully cohesive look. Do you want me to do that?


i want all words to be black 

Got it. You can ensure all text in the app is black, while keeping your tabs and title colors for emphasis. Here's the updated CSS:

st.markdown("""
<style>
/* App background */
.stApp {
    background: #eaf8ea;  /* light green background */
    color: black !important;  /* all default text black */
}

/* Force all text to black */
html, body, [class*="css"] {
    color: black !important;
}

/* Tab text and background */
button[data-baseweb="tab"] {
    background-color: #2E7D32 !important;  /* dark green for tabs */
    color: black !important;               /* tab text black */
    font-weight: 600;
}

/* Selected tab highlight */
button[data-baseweb="tab"][data-selected="true"] {
    background-color: #66BB6A !important;  /* lighter green for active tab */
    color: black !important;                /* selected tab text black */
}

/* Title color */
h1 {
    color: #FBC02D !important;  /* complementary soft gold */
}
</style>
""", unsafe_allow_html=True)

st.title("🧪 Saliva Glucose Monitoring Platform")
Summary:
All text is forced to black (color: black !important;).

Tabs have dark and light green backgrounds but tab labels remain black.

The title keeps the gold accent for emphasis.

This ensures readability while keeping your green/gold color theme consistent.

If you want, I can also apply the same black text rule to any buttons, inputs, and other interactive widgets so everything is uniform. Do you want me to do that?


wrong alraedy, i want the background of the title to be the yellowish not the words itself, also i want the whole row for tabs to be green not jus the shape of the tab isteslf 

Ah, understood now — you want:

Title row background to be yellow/gold, not the text color. Text itself remains black.

Entire tab row background (the container behind all tabs) to be green, not just the tab shapes.

Here’s how you can do that in Streamlit with CSS:

st.markdown("""
<style>
/* App background */
.stApp {
    background: #eaf8ea;  /* light green background for app */
    color: black !important;  /* all default text black */
}

/* Force all text to black */
html, body, [class*="css"] {
    color: black !important;
}

/* Title background */
h1 {
    background-color: #FBC02D !important;  /* yellow/gold background */
    padding: 10px;
    border-radius: 5px;
}

/* Entire tabs row background */
div[role="tablist"] {
    background-color: #2E7D32 !important;  /* dark green row behind tabs */
    border-radius: 5px;
    padding: 5px;
}

/* Tab text color */
button[data-baseweb="tab"] {
    color: black !important;  /* tab text black */
    font-weight: 600;
}

/* Selected tab highlight */
button[data-baseweb="tab"][data-selected="true"] {
    background-color: #66BB6A !important;  /* lighter green for active tab */
    color: black !important;
}
</style>
""", unsafe_allow_html=True)

st.title("🧪 Saliva Glucose Monitoring Platform")
What this does:
The title has a yellow/gold background (#FBC02D) and black text.

The entire tab row has a dark green background (#2E7D32).

Individual tabs still have the selected lighter green (#66BB6A) but text stays black.

App text elsewhere remains black.

This achieves the visual effect you described: a highlighted title row and a full green tab row with good contrast.

If you want, I can also make the green tab row extend edge-to-edge for a cleaner look on wide screens. Do you want me to do that?


okay change the yellow to another complinematy colour so that it is not so glaring, also the dark green to a lighter shade, only to be slightyly darker than the back groun to differentiate it 

Got it — we can make it softer and more harmonious:

Title row background: a soft muted gold/yellow → #FEEFB3 (less glaring than the previous gold).

Tab row background: slightly darker than the app background → #C6EAC6 (just enough to differentiate from the light green #eaf8ea app background).

Selected tab: slightly brighter green → #A5D6A7 for contrast.

Here’s the updated CSS for Streamlit:

st.markdown("""
<style>
/* App background */
.stApp {
    background-color: #eaf8ea;  /* light green background for app */
    color: black !important;    /* all default text black */
}

/* Force all text to black */
html, body, [class*="css"] {
    color: black !important;
}

/* Title background */
h1 {
    background-color: #FEEFB3 !important;  /* soft muted gold */
    padding: 10px;
    border-radius: 5px;
}

/* Entire tabs row background */
div[role="tablist"] {
    background-color: #C6EAC6 !important;  /* slightly darker green than background */
    border-radius: 5px;
    padding: 5px;
}

/* Tab text color */
button[data-baseweb="tab"] {
    color: black !important;  /* tab text black */
    font-weight: 600;
}

/* Selected tab highlight */
button[data-baseweb="tab"][data-selected="true"] {
    background-color: #A5D6A7 !important;  /* slightly brighter green for active tab */
    color: black !important;
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
# BUBBLE FEATURE EXTRACTION (UPDATED)
# =====================================
def extract_bubble_features(image_path, top_n=20):
    img = cv2.imread(image_path)

    if img is None:
        raise ValueError(f"Cannot read image {image_path}")

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Improve contrast
    gray_contrast = cv2.equalizeHist(gray)

    # Reduce noise
    gray_blur = cv2.GaussianBlur(gray_contrast, (5, 5), 1.2)

    # Improved Hough circle detection
    circles = cv2.HoughCircles(
        gray_blur,
        cv2.HOUGH_GRADIENT,
        dp=1.1,
        minDist=15,
        param1=80,
        param2=22,
        minRadius=8,
        maxRadius=25
    )

    if circles is None:
        raise ValueError("No bubbles detected.")

    circles = np.round(circles[0]).astype(int)

    candidates = []

    for x, y, r in circles:
        # Keep almost full bubble region
        r_roi = int(r * 0.95)

        Y, X = np.ogrid[:img_rgb.shape[0], :img_rgb.shape[1]]
        mask = (X - x)**2 + (Y - y)**2 <= r_roi**2

        roi_rgb = img_rgb[mask]

        if roi_rgb.size == 0:
            continue

        roi_hsv = rgb_to_hsv(roi_rgb / 255.0)

        h_mean, s_mean, v_mean = roi_hsv.mean(axis=0)

        # HSV pink filter
        if (
            252/360 <= h_mean <= 290/360 and
            s_mean >= 0.04 and
            v_mean >= 0.60
        ):
            score = (h_mean**8) * s_mean * v_mean * r_roi

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
    "H": [0.735825, 0.745060, 0.740868, 0.743964, 0.784736],
    "S": [0.085632, 0.090197, 0.090234, 0.100153, 0.130699]
})

# saliva reference correction
H_blank_deg = 10
S_blank_percent = 0.5

H_blank = H_blank_deg / 360.0
S_blank = S_blank_percent / 100.0

calibration_data["H_corr"] = calibration_data["H"] - H_blank
calibration_data["S_corr"] = calibration_data["S"] - S_blank

y_glucose = calibration_data["Glucose"].values

model_H  = LinearRegression().fit(calibration_data[["H_corr"]], y_glucose)
model_S  = LinearRegression().fit(calibration_data[["S_corr"]], y_glucose)
model_HS = LinearRegression().fit(calibration_data[["H_corr","S_corr"]], y_glucose)


# Initialize session state
if "history" not in st.session_state:
    st.session_state.history = []
    
# =====================================
# TABS
# =====================================
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🏠 Home",
    "🧪 Saliva Glucose Estimation",
    "📈 History",
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

    st.subheader("🗺️ How to Use This Web App")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.image("https://img.icons8.com/fluency/96/test-tube.png")
        st.caption("1️⃣ Go to Saliva Glucose Estimation Tab")
    with col2:
        st.image("https://img.icons8.com/fluency/96/image-file.png")
        st.caption("2️⃣ Upload/Take image")
    with col3:
        st.image("https://img.icons8.com/fluency/96/microscope.png")
        st.caption("3️⃣ Automatic analysis")
    with col4:
        st.image("https://img.icons8.com/fluency/96/combo-chart.png")
        st.caption("4️⃣ View result")


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

            glucose_weighted = 0.2*g_H + 0.3*g_S + 0.5*g_HS

            st.image(img_rgb, caption="Uploaded Image")
            st.subheader("Estimated Glucose (µM)")
            st.write(f"**{glucose_weighted:.1f} µM**")

            if 10 <= glucose_weighted <= 150:
                st.success("✅ This is within expected healthy physiological saliva range.")
            elif 150 < glucose_weighted <= 250:
                st.warning("⚠️ Elevated saliva glucose detected. Consider confirmatory finger-prick or clinical assessment.")
            else:
                st.error("❌ Unusual Glucose Level detected, do upload another image.")
           
            sg_time = datetime.now(ZoneInfo("Asia/Singapore"))
            
            st.session_state.history.append({
            "Time": sg_time.strftime("%Y-%m-%d %H:%M:%S"),
            "Glucose": round(glucose_weighted, 1)
            })
        
        except Exception as e:
            st.error(f"No bubbles detected in image, do upload another image: {e}")
        
# ==========================================
#  HISTORY TAB
# ==========================================
with tab3:
    st.header("Historical Results Log")

    if len(st.session_state.history) > 0:
        df = pd.DataFrame(st.session_state.history)

        st.dataframe(df)

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(df["Time"], df["Glucose"], marker="o")
        ax.set_ylabel("Glucose (µM)")
        ax.set_xlabel("Date | Time")

        st.pyplot(fig)
    else:
        st.info("No historical data available yet.")
# =====================================
# DIET TAB
# =====================================
with tab4:
    st.header("Food Serving Size Estimation")
    st.image("8.png")
    serving_df = pd.DataFrame({
    "Food Group": ["Vegetables", "Protein", "Carbohydrates", "Fats"],
    "Recommended ": ["2-3 Servings", "1-2 Servings", "1 Serving", "1 Serving"]
    })

    st.table(serving_df)

    st.subheader("Carbohydrate Counter")

    carbs_per_serving = {
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
with tab5:
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
with tab6:
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
