import streamlit as st
from modules import predict, compare, about, evaluation
# from modules.predict import load_css, load_js, load_model, load_images, evaluate_images


# Fungsi untuk memuat CSS
def load_css(file_path):
    with open(file_path, "r") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Fungsi untuk memuat JavaScript
def load_js(file_path):
    with open(file_path, "r") as f:
        st.markdown(f"<script>{f.read()}</script>", unsafe_allow_html=True)

def main():
    st.set_page_config(page_title="Segmenntation", page_icon="assets/img/favicon.ico", layout="wide")
    # Memuat CSS dan JavaScript
    load_css("styles/style.css")
    load_js("js/script.js")

    
    
    col1, col2 = st.sidebar.columns(2)  # Membagi sidebar menjadi dua kolom
    with col1:
        st.image("assets/img/logo_undiksha.png", width=90)  # Logo pertama
    with col2:
        st.image("assets/img/logo_dagoengineering.png", width=500)  # Logo kedua

    st.sidebar.title("Segmentation Corrosion")

    # # Menambahkan dua logo di sidebar
    # st.sidebar.image("assets/img/logo_undiksha.png", width=100)  # Logo pertama
    # st.sidebar.image("assets/img/logo_dagoengineering.png", width=100)  # Logo kedua

    # st.sidebar.markdown("""
    #     **Instructions**:
    #     1. Pilih menu di bawah ini.
    #     2. Ikuti instruksi pada halaman yang dipilih.
    # """)

    


    # Menu Navigasi
    menu = st.sidebar.radio("Menu", ("Predict", "Compare", "Evaluation","About"))

    # Halaman berdasarkan menu
    if menu == "Predict":
        predict.render()
    elif menu == "Compare":
        compare.render()
    elif menu == "Evaluation":
        evaluation.render()
    elif menu == "About":
        about.render()



    
if __name__ == "__main__":
    main()