import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
import time

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

def calculate_area(mask, color):
    return np.sum(np.all(mask == color, axis=-1))

def calculate_corrosion_domination(predicted_mask, colormap):
    corrosion_color = colormap[0]  # Red for corrosion
    asset_color = colormap[1]  # Blue for asset (not corrosion)
    background_color = colormap[2]  # Green for background
    
    corrosion_area = calculate_area(predicted_mask, corrosion_color)
    asset_area = calculate_area(predicted_mask, asset_color)
    background_area = calculate_area(predicted_mask, background_color)
    
    total_area = predicted_mask.shape[0] * predicted_mask.shape[1]
    
    corrosion_percentage = (corrosion_area / total_area) * 100
    asset_percentage = (asset_area / total_area) * 100
    background_percentage = (background_area / total_area) * 100
    
    return corrosion_percentage, asset_percentage, background_percentage

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

        # Get corrosion and asset area percentages
        corrosion_percentage, asset_percentage, background_percentage = calculate_corrosion_domination(predicted_mask_rgb, colormap)
        
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
            "asset_percentage": asset_percentage,
            "background_percentage": background_percentage,
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

    return overlay, image, corrosion_percentage, asset_percentage, background_percentage, severity, handling_message, processing_time

def load_model_based_on_option(model_option):
    model_paths = {
        "U-Net": "models/3_Classes/Best_Performa_Model/unet/Model_B8E100.h5",
        "DeepLab V3+": "models/3_Classes/Best_Performa_Model/deeplabv3+/Model_B16E20.h5"
    }
    return load_model(model_paths.get(model_option))

def render():
    st.title("Corrosion Segmentation with Two Models")
    st.tabs (["Predict", "Evaluation"])

    # Use columns for selecting models
    col1, col2 = st.columns(2)

    with col1:
        model_option_unet = st.selectbox("Select Model for Column 1", ["U-Net", "DeepLab V3+"], key="unet")

    with col2:
        model_option_deeplab = st.selectbox("Select Model for Column 2", ["U-Net", "DeepLab V3+"], key="deeplab")

    # Load models based on selection
    model_unet = load_model_based_on_option(model_option_unet)
    model_deeplab = load_model_based_on_option(model_option_deeplab)
    
    # File upload 
    uploaded_files = st.file_uploader("Upload Images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

    if uploaded_files:
        images, filenames = load_images(uploaded_files, target_size=(256, 256))

        colormap = {
            0: [255, 0, 0],  # Red for corrosion
            1: [0, 0, 255],  # Blue for asset
            2: [0, 255, 0]   # Green for background
        }

        # Display results below image upload
        st.write("### Processing Results:")

        # Create columns for displaying U-Net and DeepLab results side by side
        result_col1, result_col2 = st.columns(2)

        # Placeholder for summary data
        summary_data = []

        with result_col1:
            st.subheader(f"Results for {model_option_unet} Model")
            with st.spinner(f"Processing with {model_option_unet}..."):
                overlay_unet, image_unet, corrosion_percentage_unet, asset_percentage_unet, background_percentage_unet, severity_unet, handling_message_unet, processing_time_unet = evaluate_images(
                    model_unet, images, filenames, colormap
                )
                
                # Menggunakan st.columns untuk menampilkan gambar secara horizontal
                cols = st.columns(2)
                with cols[0]:
                    st.image(image_unet, caption="Original Image (U-Net)", use_container_width=True)
                with cols[1]:
                    st.image(overlay_unet, caption=f"Corrosion Overlay (U-Net) - {severity_unet} Severity", use_container_width=True)

                # st.markdown(f"**Corrosion Area: {corrosion_percentage_unet:.2f}%**")
                # st.markdown(f"**Processing Time: {processing_time_unet:.2f} seconds**")
                # st.markdown(f"**{handling_message_unet}**")

                summary_data.append({
                    "Model Name": model_option_unet,
                    "Image Size": f"{image_unet.shape[1]}x{image_unet.shape[0]}",
                    "Processing Time (s)": f"{processing_time_unet:.2f}",
                    "Corrosion Area (%)": f"{corrosion_percentage_unet:.2f}",
                    "Asset Area (%)": f"{asset_percentage_unet:.2f}",
                    "Background Area (%)": f"{background_percentage_unet:.2f}",
                    "Severity": severity_unet,
                    "Message": handling_message_unet
                })
        with result_col2:
            st.subheader(f"Results for {model_option_deeplab} Model")
            with st.spinner(f"Processing with {model_option_deeplab}..."):
                overlay_deeplab, image_deeplab, corrosion_percentage_deeplab, asset_percentage_deeplab, background_percentage_deeplab, severity_deeplab, handling_message_deeplab, processing_time_deeplab = evaluate_images(
                    model_deeplab, images, filenames, colormap
                )
                
                # Menggunakan st.columns untuk menampilkan gambar secara horizontal
                cols = st.columns(2)
                with cols[0]:
                    st.image(image_deeplab, caption="Original Image (DeepLab V3+)", use_container_width=True)
                with cols[1]:
                    st.image(overlay_deeplab, caption=f"Corrosion Overlay (DeepLab V3+) - {severity_deeplab} Severity", use_container_width=True)

                # st.markdown(f"**Corrosion Area: {corrosion_percentage_deeplab:.2f}%**")
                # st.markdown(f"**Processing Time: {processing_time_deeplab:.2f} seconds**")
                # st.markdown(f"**{handling_message_deeplab}**")

                summary_data.append({
                    "Model Name": model_option_deeplab,
                    "Image Size": f"{image_deeplab.shape[1]}x{image_deeplab.shape[0]}",
                    "Processing Time (s)": f"{processing_time_deeplab:.2f}",
                    "Corrosion Area (%)": f"{corrosion_percentage_deeplab:.2f}",
                    "Asset Area (%)": f"{asset_percentage_deeplab:.2f}",
                    "Background Area (%)": f"{background_percentage_deeplab:.2f}",
                    "Severity": severity_deeplab,
                    "Message": handling_message_deeplab
                })
        # Display summary table at the bottom
        st.write("### Summary of Results")
        st.table(summary_data)

if __name__ == "__main__":
    render()