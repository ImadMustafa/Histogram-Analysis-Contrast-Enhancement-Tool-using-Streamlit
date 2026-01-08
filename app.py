# Histogram Analysis and Contrast Enhancement Tool Tasks:
#    1- Apply histogram equalization
#    2- Compute and display the image histogram
#    3- Normalize the histogram
#    4- Comparison Before and After (Contrast, Entropy, Histograms)


import streamlit as st
import cv2
import numpy as np
import plotly.graph_objects as go
from PIL import Image

st.set_page_config(
    page_title="Histogram & Contrast Enhancement Tool",
    page_icon="📊",
    layout="wide"
)

# Sidebar Controls (User inputs)

st.sidebar.title("⚙️ Controls")

uploaded_file = st.sidebar.file_uploader(
    "Upload Image",
    type=["jpg", "jpeg", "png", "bmp"]
)

processing_mode = st.sidebar.radio(
    "Processing Mode",
    ["Grayscale", "Color"]
)

contrast_method = st.sidebar.radio(
    "Contrast Enhancement Method",
    [
        "None",
        "Standard Histogram Equalization",
        "Adaptive Histogram Equalization (CLAHE)"
    ]
)

clahe_clip = 2.0
clahe_tile = 8

if contrast_method == "Adaptive Histogram Equalization (CLAHE)":
    st.sidebar.markdown("### 🧩 CLAHE Parameters")

    clahe_clip = st.sidebar.slider(
        "Clip Limit",
        min_value=0.5,
        max_value=5.0,
        value=2.0,
        step=0.1
    )

    clahe_tile = st.sidebar.slider(
        "Tile Grid Size",
        min_value=4,
        max_value=16,
        value=8,
        step=2
    )

noise_filter = st.sidebar.selectbox(
    "Noise Reduction",
    [
        "None",
        "Gaussian Blur",
        "Median Filter",
        "Bilateral Filter",
        "Non-Local Means (Best Quality)"
    ]
)

filter_strength = st.sidebar.slider(
    "Filter Strength",
    min_value=1,
    max_value=15,
    value=7,
    step=2
)

st.sidebar.markdown("---")
st.sidebar.info(
    "Noise reduction is applied BEFORE contrast enhancement "
    "to prevent noise amplification."
)

# Main Page Title
st.title("📊 Histogram Analysis & Contrast Enhancement Tool")
st.caption("Streamlit • OpenCV • Histogram Equalization • CLAHE")


if uploaded_file is None:
    st.info("👈 Upload an image to begin.")
    st.stop()

# Load Image
image = Image.open(uploaded_file)
image_array = np.array(image)

# Check if image is color (3 channels) or grayscale
is_color = len(image_array.shape) == 3


# Denoiseing Function
def apply_denoising(img, method, strength):
    """
    Apply noise reduction BEFORE contrast enhancement.
    This avoids amplifying noise.
    """
    if method == "Gaussian Blur":
        # Kernel size must be odd
        k = strength if strength % 2 == 1 else strength + 1
        return cv2.GaussianBlur(img, (k, k), 0)

    elif method == "Median Filter":
        k = strength if strength % 2 == 1 else strength + 1
        return cv2.medianBlur(img, k)

    elif method == "Bilateral Filter":
        # Preserves edges better than Gaussian
        return cv2.bilateralFilter(img, 9, 75, 75)

    elif method == "Non-Local Means (Best Quality)":
        # Best quality denoising but slow
        h = strength * 2 + 5
        if len(img.shape) == 2:
            return cv2.fastNlMeansDenoising(img, None, h, 7, 21)
        else:
            return cv2.fastNlMeansDenoisingColored(img, None, h, h, 7, 21)

    return img


def entropy(img):
    """
    Measures information content of the image.
    Higher entropy = more details.
    """
                    #image, grascale only, entire img (no mask), hist size, range
    hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    #nomralize
    hist = hist / hist.sum()
    return -np.sum(hist * np.log2(hist + 1e-10))
















# ==================================================
# 1- Apply histogram equalization
# ==================================================
def apply_contrast(img, method, clip, tile):
    """
    Apply contrast enhancement.
    CLAHE works locally, standard equalization is global.
    """
    if method == "Standard Histogram Equalization":
        return cv2.equalizeHist(img)

    elif method == "Adaptive Histogram Equalization (CLAHE)":
        clahe = cv2.createCLAHE(
            clipLimit=clip,
            tileGridSize=(tile, tile)
        )
        return clahe.apply(img)

    return img





















# ==================================================
# 2- Compute and display the image histogram
# ==================================================


# Image Processing
# ----------------------------------------------
if processing_mode == "Grayscale" or not is_color:
    # Convert color image to grayscale if needed
    img_gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY) if is_color else image_array

    # Apply noise reduction
    img_denoised = apply_denoising(img_gray, noise_filter, filter_strength)

    # Apply contrast enhancement
    img_processed = apply_contrast(
        img_denoised,
        contrast_method,
        clahe_clip,
        clahe_tile
    )

    display_original = img_gray
    display_processed = img_processed


    # Histograms
    hist_original = cv2.calcHist([img_gray], [0], None, [256], [0, 256])
    hist_processed = cv2.calcHist([img_processed], [0], None, [256], [0, 256])

    # Metrics
    contrast_original = np.std(img_gray)
    contrast_processed = np.std(img_processed)

    entropy_original = entropy(img_gray)
    entropy_processed = entropy(img_processed)

else:
    # ----------------------------------------------
    # COLOR MODE (IMPORTANT PART)
    # ----------------------------------------------
    # Convert RGB → YCrCb
    # Y = luminance (brightness)
    # Cr, Cb = color information
    img_ycrcb = cv2.cvtColor(image_array, cv2.COLOR_RGB2YCrCb)
    y, cr, cb = cv2.split(img_ycrcb)

    # Apply processing ONLY on luminance
    y_denoised = apply_denoising(y, noise_filter, filter_strength)
    y_processed = apply_contrast(
        y_denoised,
        contrast_method,
        clahe_clip,
        clahe_tile
    )

    # Merge channels back
    img_ycrcb_processed = cv2.merge([y_processed, cr, cb])
    img_processed = cv2.cvtColor(img_ycrcb_processed, cv2.COLOR_YCrCb2RGB)

    display_original = image_array
    display_processed = img_processed

    # Histograms are computed on luminance only
    hist_original = cv2.calcHist([y], [0], None, [256], [0, 256])
    hist_processed = cv2.calcHist([y_processed], [0], None, [256], [0, 256])

    contrast_original = np.std(y)
    contrast_processed = np.std(y_processed)

    entropy_original = entropy(y)
    entropy_processed = entropy(y_processed)














# ==================================================
# 3- Normalize the histogram
# ==================================================

# Normalize the histograms so their values sum to 1
hist_original = cv2.normalize(hist_original, hist_original).flatten()
hist_processed = cv2.normalize(hist_processed, hist_processed).flatten()

cdf_original = np.cumsum(hist_original)
cdf_processed = np.cumsum(hist_processed)















# ==================================================
# 4- Comparison Before and After (Contrast, Entropy, Histograms)
# ==================================================


# Image Comparison
# ----------------------------------------------
st.markdown("## 🖼️ Image Comparison")

c1, c2 = st.columns(2)
with c1:
    st.subheader("Original Image")
    st.image(display_original, use_column_width=True)
    st.metric("Std Dev", f"{contrast_original:.2f}")
    st.metric("Entropy", f"{entropy_original:.2f}")

with c2:
    st.subheader("Processed Image")
    st.image(display_processed, use_column_width=True)
    st.metric("Std Dev", f"{contrast_processed:.2f}")
    st.metric("Entropy", f"{entropy_processed:.2f}")




# Difference Map
# ----------------------------------------------
st.markdown("## 🔍 Difference Map")

if processing_mode == "Grayscale" or not is_color:
    diff = cv2.absdiff(display_original, display_processed)
else:
    # Compare luminance difference only
    #Take ALL rows, ALL columns, but ONLY the first channel
    orig_y = cv2.cvtColor(display_original, cv2.COLOR_RGB2YCrCb)[:, :, 0]
    proc_y = cv2.cvtColor(display_processed, cv2.COLOR_RGB2YCrCb)[:, :, 0]
    diff = cv2.absdiff(orig_y, proc_y)

st.image(diff, use_column_width=True, caption="Absolute Difference (Luminance)")




# Histogram Analysis
# ----------------------------------------------
st.markdown("## 📈 Histogram Analysis")

h1, h2 = st.columns(2)
with h1:
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=hist_original, fill="tozeroy", name="Original"))
    fig.update_layout(title="Normalized Histogram (Original)")
    st.plotly_chart(fig, use_container_width=True)

with h2:
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=hist_processed, fill="tozeroy", name="Processed"))
    fig.update_layout(title="Normalized Histogram (Processed)")
    st.plotly_chart(fig, use_container_width=True)




# CDF Visualization
# ----------------------------------------------
st.markdown("## 📉 Cumulative Distribution Function (CDF)")

fig_cdf = go.Figure()
fig_cdf.add_trace(go.Scatter(y=cdf_original, name="Original CDF"))
fig_cdf.add_trace(go.Scatter(y=cdf_processed, name="Processed CDF"))
fig_cdf.update_layout(
    xaxis_title="Pixel Intensity",
    yaxis_title="Cumulative Probability"
)
st.plotly_chart(fig_cdf, use_container_width=True)




# RGB Histograms (always shown in Color mode)
# ----------------------------------------------
if processing_mode == "Color" and is_color:
    st.markdown("## 🎨 RGB Channel Histograms")

    fig_rgb = go.Figure()
    channels = [("Red", 0), ("Green", 1), ("Blue", 2)]

    for name, idx in channels:
        hist = cv2.calcHist([image_array], [idx], None, [256], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        fig_rgb.add_trace(go.Scatter(y=hist, name=name))

    fig_rgb.update_layout(
        height=300,
        xaxis_title="Pixel Intensity",
        yaxis_title="Normalized Frequency"
    )

    st.plotly_chart(fig_rgb, use_container_width=True)




















# Summary
# ----------------------------------------------
st.markdown("## ✅ Enhancement Summary")

improvement = (
    (contrast_processed - contrast_original) /
    contrast_original * 100
    if contrast_original > 0 else 0
)

st.success(f"Contrast Improvement: **{improvement:.2f}%**")
st.info(
    f"Method: **{contrast_method}** | "
    f"CLAHE Clip: **{clahe_clip}** | Tile: **{clahe_tile}** | "
    f"Noise Reduction: **{noise_filter}**"
)

st.caption("Academic Image Processing Project • OpenCV & Streamlit")
