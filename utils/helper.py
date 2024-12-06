import os
import numpy as np
import cv2
import streamlit as st
from tensorflow.keras.models import load_model
import time
from sklearn.metrics import jaccard_score, f1_score



def load_css(file_path):
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            css_content = f.read()
        st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)
    else:
        st.error(f"CSS file not found at {file_path}")

# Fungsi untuk memuat JavaScript (opsional)
def load_js(file_path):
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            js_content = f.read()
        st.markdown(f"<script>{js_content}</script>", unsafe_allow_html=True)
    else:
        st.error(f"JavaScript file not found at {file_path}")

# Fungsi tambahan
def rgb_to_hex(rgb):
    return "#{:02x}{:02x}{:02x}".format(rgb[0], rgb[1], rgb[2])

def load_images(uploaded_files, target_size=None):
    images = []
    filenames = []

    for uploaded_file in uploaded_files:
        img = cv2.cvtColor(cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        if target_size:
            img = cv2.resize(img, target_size)
        images.append(img)
        filenames.append(uploaded_file.name)

    return np.array(images), filenames

def convert_to_rgb(mask, colormap):
    h, w = mask.shape
    rgb_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for cls, color in colormap.items():
        rgb_mask[mask == cls] = color
    return rgb_mask

def overlay_images(original, mask, opacity=0.4):
    return cv2.addWeighted(original, 1 - opacity, mask, opacity, 0)

def calculate_metrics(y_true, y_pred):
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()

    y_true_flat = (y_true_flat > 0).astype(int)
    y_pred_flat = (y_pred_flat > 0).astype(int)

    iou = jaccard_score(y_true_flat, y_pred_flat, average='weighted')
    f1 = f1_score(y_true_flat, y_pred_flat, average='weighted')

    intersection = np.sum(y_true_flat * y_pred_flat)
    dice = 2 * intersection / (np.sum(y_true_flat) + np.sum(y_pred_flat) + 1e-6)

    return iou, f1, dice

def calculate_corrosion_domination(predicted_mask, corrosion_color=[255, 0, 0]):
    corrosion_area = np.sum(np.all(predicted_mask == corrosion_color, axis=-1))
    total_area = predicted_mask.shape[0] * predicted_mask.shape[1]
    return (corrosion_area / total_area) * 100


def get_severity_message(severity):
    """Return a handling message based on severity"""
    messages = {
        "Low": "The corrosion is minimal. Regular monitoring is recommended.",
        "Moderate": "Corrosion is noticeable. Action should be taken soon to prevent further damage.",
        "High": "Severe corrosion detected. Immediate intervention is required to avoid further deterioration."
    }
    return messages.get(severity, "No message available for this severity.")


def evaluate_images(model, images, filenames, colormap):
    st.write("### Results:")
    results = []

    for i, image in enumerate(images):
        # Start timer for each image processing
        start_time = time.time()

        # Predict the mask
        predicted_mask_class = np.argmax(model.predict(image[None, ...]), axis=-1)[0]
        predicted_mask_rgb = convert_to_rgb(predicted_mask_class, colormap)
        overlay = overlay_images(image, predicted_mask_rgb)

        # Calculate corrosion domination
        corrosion_percentage = calculate_corrosion_domination(predicted_mask_rgb, corrosion_color=colormap[0])
        severity = (
            "Low" if corrosion_percentage < 20 else
            "Moderate" if corrosion_percentage < 50 else
            "High"
        )

        # Get the handling message based on severity
        handling_message = get_severity_message(severity)

        # End timer after processing
        end_time = time.time()
        processing_time = end_time - start_time  # Calculate the time taken for processing this image

        # Append result
        results.append({
            "filename": filenames[i],
            "corrosion_percentage": corrosion_percentage,
            "severity": severity,
            "processing_time": processing_time,  # Include processing time in the result
            "handling_message": handling_message  # Add handling message to results
        })

        # Display results with message centered using CSS class
        message = f"""
        <div class="center-message">
            <p>File: {filenames[i]}</p>
            <h3>Corrosion Area (%): {corrosion_percentage:.2f}%</h3>
            <h3>{severity}</h3>
            <p><strong>{handling_message}</strong></p>  <!-- Display severity handling message -->
            <p> Processing Time:{processing_time:.2f} seconds</p>  <!-- Display processing time -->
        </div>
        """
        st.markdown(message, unsafe_allow_html=True)
        # Display original image and overlay
        with st.container():
            col1, col2 = st.columns(2)
            with col1:
                st.image(image, caption="Original Image", use_container_width=True)
            with col2:
                st.image(overlay, caption=f"Corrosion Overlay ({severity} Severity)", use_container_width=True)

        st.write("---")

        time.sleep(0.5)  # Simulate processing delay

    return results


def main():
    load_css("styles/style.css")
    load_js("javascript/script.js")

    st.title("Corrosion Segmentation and Severity Assessment")


    # Definisikan path ke gambar logo
    LOGO_URL_LARGE = "assets/img/logo_undiksha.png"  # Ganti dengan path logo besar
    LOGO_URL_SMALL = "https://github.com/Adiwid20/Semanctic-Segmentation-with-Streamlit-/blob/e6a042457914a51fee7bb12a01928427b1708f74/assets/img/logo_dagoengineering.png"         # Ganti dengan path logo kecil

    # Menampilkan dua logo berdampingan di sidebar
    st.sidebar.markdown(
        """
        <div style="text-align: center; display: flex; justify-content: center; align-items: center;">
            <img src="{}" alt="Large Logo" style="width: 80px; margin-right: 10px;">
            <img src="{}" alt="Small Logo" style="width: 40px;">
        </div>
        """.format(LOGO_URL_LARGE, LOGO_URL_SMALL),
        unsafe_allow_html=True,
    )

    st.sidebar.title("Navigation")
    st.sidebar.markdown("""
        **Instructions**:
        1. Upload your images.
        2. See segmentation results and corrosion analysis.
    """)

    # Menu
    menu = st.sidebar.radio("Menu", ("Image Prediction", "About"))
    model_option = st.sidebar.selectbox("Select Model:", ("U-Net", "DeepLab V3+"))

    model_path = (
        "/Users/macbook-air/M2/DAGO PHKT (AIDA PHASE 3 BACKUP)/Deployment/Streamlit/model/UNET/Model_B8E100.h5" if model_option == "U-Net" else
        "/Users/macbook-air/M2/DAGO PHKT (AIDA PHASE 3 BACKUP)/Deployment/Streamlit/model/DeepLabV3+/Model_B16E20.h5"
    )
    model = load_model(model_path)

    if menu == "Image Prediction":
        st.write("### Upload Images for Corrosion Analysis")
        uploaded_files = st.file_uploader("Upload Images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

        if uploaded_files:
            images, filenames = load_images(uploaded_files, target_size=(256, 256))
            colormap = {
                0: [255, 0, 0], 
                2: [0, 255, 0], 
                1: [0, 0, 255]}  # Define your colormap here

            with st.spinner("Processing images..."):
                evaluate_images(model, images, filenames, colormap)

    elif menu == "About":
        st.write("### About this App")
        st.write("""
            This application uses deep learning to segment corrosion areas in images.
            It also assesses the severity of corrosion based on the segmented area.
        """)

if __name__ == "__main__":
    main()