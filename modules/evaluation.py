import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
import time
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score

# Fungsi untuk memuat gambar dari file upload
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

# Fungsi untuk mengubah mask menjadi format RGB
def convert_to_rgb(mask, colormap):
    h, w = mask.shape
    rgb_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for cls, color in colormap.items():
        rgb_mask[mask == cls] = color
    return rgb_mask

# Fungsi untuk overlay gambar (overlay asli dengan mask)
def overlay_images(original, mask, opacity=0.4):
    return cv2.addWeighted(original, 1 - opacity, mask, opacity, 0)

# Fungsi untuk menghitung area dari setiap kelas
def calculate_area(mask, color):
    return np.sum(np.all(mask == color, axis=-1))

# Fungsi untuk menghitung dominasi korosi dalam hasil segmentasi
def calculate_corrosion_domination(predicted_mask, colormap):
    corrosion_color = colormap[0]  # Merah untuk korosi
    asset_color = colormap[1]  # Biru untuk aset (bukan korosi)
    background_color = colormap[2]  # Hijau untuk latar belakang
    
    corrosion_area = calculate_area(predicted_mask, corrosion_color)
    asset_area = calculate_area(predicted_mask, asset_color)
    background_area = calculate_area(predicted_mask, background_color)
    
    total_area = predicted_mask.shape[0] * predicted_mask.shape[1]
    
    corrosion_percentage = (corrosion_area / total_area) * 100
    asset_percentage = (asset_area / total_area) * 100
    background_percentage = (background_area / total_area) * 100
    
    return corrosion_percentage, asset_percentage, background_percentage

# Fungsi untuk memberikan pesan berdasarkan tingkat keparahan
def get_severity_message(severity):
    messages = {
        "Low": "Korosi minimal. Pemantauan reguler disarankan.",
        "Moderate": "Korosi terlihat. Tindakan harus segera diambil untuk mencegah kerusakan lebih lanjut.",
        "High": "Korosi parah terdeteksi. Intervensi segera diperlukan untuk menghindari kerusakan lebih lanjut."
    }
    return messages.get(severity, "Pesan tidak tersedia untuk tingkat keparahan ini.")

# Fungsi untuk mengonversi mask RGB menjadi class mask
def convert_to_class_mask(rgb_mask, colormap):
    class_mask = np.zeros((rgb_mask.shape[0], rgb_mask.shape[1]), dtype=np.uint8)
    for cls, color in colormap.items():
        class_mask[np.all(rgb_mask == color, axis=-1)] = cls
    return class_mask

# Fungsi untuk menghitung IoU
def calculate_iou(pred_mask, true_mask, num_classes=3):
    iou_per_class = []
    for cls in range(num_classes):
        intersection = np.sum((pred_mask == cls) & (true_mask == cls))
        union = np.sum((pred_mask == cls) | (true_mask == cls))
        iou = intersection / union if union != 0 else 0
        iou_per_class.append(iou)
    return np.array(iou_per_class)

# Fungsi untuk menghitung metrik evaluasi
def calculate_metrics(pred_mask, true_mask, num_classes=3):
    metrics = {
        "precision": [],
        "recall": [],
        "f1_score": []
    }
    for cls in range(num_classes):
        true_positive = np.sum((pred_mask == cls) & (true_mask == cls))
        false_positive = np.sum((pred_mask == cls) & (true_mask != cls))
        false_negative = np.sum((pred_mask != cls) & (true_mask == cls))
        
        precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) != 0 else 0
        recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) != 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
        
        metrics["precision"].append(precision)
        metrics["recall"].append(recall)
        metrics["f1_score"].append(f1)
    
    return metrics

# Fungsi untuk evaluasi gambar
def evaluate_images(model, images, groundtruth_images, filenames, colormap):
    st.write("### Hasil Evaluasi:")
    results = []
    iou_per_model = []
    metrics_per_model = []

    for i, image in enumerate(images):
        start_time = time.time()

        # Prediksi mask
        predicted_mask_class = np.argmax(model.predict(image[None, ...]), axis=-1)[0]
        predicted_mask_rgb = convert_to_rgb(predicted_mask_class, colormap)
        overlay = overlay_images(image, predicted_mask_rgb)

        # Ground truth untuk perhitungan IoU
        groundtruth_rgb = groundtruth_images[i]
        groundtruth_class = convert_to_class_mask(groundtruth_rgb, colormap)

        # Hitung IoU untuk setiap kelas
        iou = calculate_iou(predicted_mask_class, groundtruth_class)
        
        # Hitung area korosi dan tingkat keparahannya
        corrosion_percentage, asset_percentage, background_percentage = calculate_corrosion_domination(predicted_mask_rgb, colormap)
        
        severity = (
            "Low" if corrosion_percentage < 20 else
            "Moderate" if corrosion_percentage < 50 else
            "High"
        )

        handling_message = get_severity_message(severity)

        # Hitung metrik evaluasi
        metrics = calculate_metrics(predicted_mask_class, groundtruth_class)

        end_time = time.time()
        processing_time = end_time - start_time

        iou_per_model.append(iou)
        metrics_per_model.append(metrics)

        results.append({
            "filename": filenames[i],
            "corrosion_percentage": corrosion_percentage,
            "asset_percentage": asset_percentage,
            "background_percentage": background_percentage,
            "severity": severity,
            "processing_time": processing_time,
            "handling_message": handling_message,
            "IoU Corrosion": iou[0],
            "IoU Asset": iou[1],
            "IoU Background": iou[2],
            "Precision Corrosion": metrics["precision"][0],
            "Recall Corrosion": metrics["recall"][0],
            "F1-Score Corrosion": metrics["f1_score"][0],
            "Precision Asset": metrics["precision"][1],
            "Recall Asset": metrics["recall"][1],
            "F1-Score Asset": metrics["f1_score"][1],
            "Precision Background": metrics["precision"][2],
            "Recall Background": metrics["recall"][2],
            "F1-Score Background": metrics["f1_score"][2]
        })

        st.markdown(f"""
        <div style="text-align: center;">
            <p>File: {filenames[i]}</p>
            <h3>Korosi (%): {corrosion_percentage:.2f}%</h3>
            <h3>{severity}</h3>
            <p><strong>{handling_message}</strong></p>  
            <p> Waktu Proses: {processing_time:.2f} detik</p>
            <p>IoU Korosi: {iou[0]:.2f}</p>
            <p>IoU Aset: {iou[1]:.2f}</p>
            <p>IoU Latar Belakang: {iou[2]:.2f}</p>
        </div>
        """, unsafe_allow_html=True)

        st.image(overlay, caption=f"Segmentation Result for {filenames[i]}", use_column_width=True)

    # Hitung rata-rata IoU dan metrik untuk setiap model
    avg_iou = np.mean(iou_per_model, axis=0)
    avg_metrics = {
        "precision": np.mean([m["precision"] for m in metrics_per_model], axis=0),
        "recall": np.mean([m["recall"] for m in metrics_per_model], axis=0),
        "f1_score": np.mean([m["f1_score"] for m in metrics_per_model], axis=0),
    }
    
    return results, avg_iou, avg_metrics

# Fungsi untuk memuat model berdasarkan pilihan
def load_model_based_on_option(model_option):
    model_paths = {
        "U-Net": "path_to_unet_model.h5",
        "DeepLabV3": "path_to_deeplab_model.h5"
    }
    model_path = model_paths.get(model_option)
    model = load_model(model_path)
    return model

# Fungsi utama untuk aplikasi
def render():
    st.title("Evaluasi Model Segmentasi Citra")

    uploaded_files = st.file_uploader("Pilih gambar untuk diupload", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

    if uploaded_files:
        images, filenames = load_images(uploaded_files, target_size=(256, 256))

        model_option_deeplab = st.selectbox("Pilih Model", ["U-Net", "DeepLabV3"])

        if st.button("Prediksi"):
            with st.spinner(f"Memproses dengan {model_option_deeplab}..."):
                model_deeplab = load_model_based_on_option(model_option_deeplab)
                colormap = {
                    0: [255, 0, 0],  # Korosi: Merah
                    1: [0, 0, 255],  # Aset: Biru
                    2: [0, 255, 0],  # Latar Belakang: Hijau
                }
                results_deeplab, avg_iou_deeplab, avg_metrics_deeplab = evaluate_images(
                    model_deeplab, images, images, filenames, colormap)

                # Tampilkan summary hasil
                summary_data = pd.DataFrame(results_deeplab)
                st.write("### Summary Hasil Evaluasi:")
                st.dataframe(summary_data)
                st.write(f"Rata-rata IoU: {avg_iou_deeplab}")
                st.write(f"Rata-rata Precision: {avg_metrics_deeplab['precision']}")
                st.write(f"Rata-rata Recall: {avg_metrics_deeplab['recall']}")
                st.write(f"Rata-rata F1-Score: {avg_metrics_deeplab['f1_score']}")

# Jalankan aplikasi
if __name__ == "__main__":
    render()