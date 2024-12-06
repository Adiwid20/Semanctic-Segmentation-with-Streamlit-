import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
import time

def load_css(file_path):
    with open(file_path, "r") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Fungsi untuk memuat JavaScript
def load_js(file_path):
    with open(file_path, "r") as f:
        st.markdown(f"<script>{f.read()}</script>", unsafe_allow_html=True)


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

def calculate_corrosion_domination(predicted_mask, corrosion_color=[255, 0, 0]):
    corrosion_area = np.sum(np.all(predicted_mask == corrosion_color, axis=-1))
    total_area = predicted_mask.shape[0] * predicted_mask.shape[1]
    return (corrosion_area / total_area) * 100

def get_severity_message(severity):
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
        start_time = time.time()

        predicted_mask_class = np.argmax(model.predict(image[None, ...]), axis=-1)[0]
        predicted_mask_rgb = convert_to_rgb(predicted_mask_class, colormap)
        overlay = overlay_images(image, predicted_mask_rgb)

        corrosion_percentage = calculate_corrosion_domination(predicted_mask_rgb, corrosion_color=colormap[0])
        severity = (
            "Low" if corrosion_percentage < 20 else
            "Moderate" if corrosion_percentage < 50 else
            "High"
        )

        handling_message = get_severity_message(severity)

        end_time = time.time()
        processing_time = end_time - start_time

        results.append({
            "filename": filenames[i],
            "corrosion_percentage": corrosion_percentage,
            "severity": severity,
            "processing_time": processing_time,
            "handling_message": handling_message
        })

        st.markdown(f"""
        <div style="text-align: center;">
            <p>File: {filenames[i]}</p>
            <h3>Corrosion Area (%): {corrosion_percentage:.2f}%</h3>
            <h3>{severity}</h3>
            <p><strong>{handling_message}</strong></p>  
            <p> Processing Time: {processing_time:.2f} seconds</p>
        </div>
        """, unsafe_allow_html=True)

        with st.container():
            col1, col2 = st.columns(2)
            with col1:
                st.image(image, caption="Original Image", use_container_width=True)
            with col2:
                st.image(overlay, caption=f"Corrosion Overlay ({severity} Severity)", use_container_width=True)

        st.write("---")

        time.sleep(0.5)

    return results

def load_model_based_on_selection():
    model_option = st.selectbox("Select Model:", ("U-Net", "DeepLab V3+"))
    model_path = (
        "models/3_Classes/Best_Performa_Model/unet/Model_B8E100.h5" if model_option == "U-Net" else
        "models/3_Classes/Best_Performa_Model/deeplabv3+/Model_B16E20.h5"
    )
    return load_model(model_path)

def process_uploaded_images(model):
    uploaded_files = st.file_uploader("Upload Images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

    if uploaded_files:
        images, filenames = load_images(uploaded_files, target_size=(256, 256))

        colormap = {
            0: [255, 0, 0],  # Red for class 0 (corrosion)
            2: [0, 255, 0],  # Green for class 1 (healthy area)
            1: [0, 0, 255]   # Blue for class 2 (other class)
        }

        with st.spinner("Processing images..."):
            evaluate_images(model, images, filenames, colormap)

def render():
    st.title("Predict Image")
    model = load_model_based_on_selection()
    process_uploaded_images(model)
    

if __name__ == "__main__":
    render()