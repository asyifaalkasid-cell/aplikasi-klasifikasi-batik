import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ====================================================================
# 1. PENGATURAN HALAMAN & TAMPILAN (STREAMLIT)
# ====================================================================
st.set_page_config(
    page_title="Klasifikasi Batik Modern vs Tradisional",
    page_icon="⚡",
    layout="centered"
)

st.title("⚡ Klasifikasi Batik Modern vs Tradisional ⚡")
st.write("Aplikasi web mandiri untuk menguji performa prediksi model CNN secara langsung.")
st.write("---")

# ====================================================================
# 2. LOAD MODEL YANG SUDAH DI-DOWNLOAD
# ====================================================================
@st.cache_resource
def load_my_model():
    # Mengambil model yang berada di folder yang sama dengan file app.py
    return tf.keras.models.load_model("model_batik_terbaik.h5")

try:
    model = load_my_model()
    st.success("🤖 Model CNN (128, 128, 64, 64, 32, 32) Berhasil Dimuat!")
except Exception as e:
    st.error("❌ Model gagal dimuat. Pastikan file 'model_batik_terbaik.h5' sudah berada di folder yang sama.")

# ====================================================================
# 3. INTERFACE INPUT GAMBAR UJI
# ====================================================================
uploaded_file = st.file_uploader("Unggah Foto Batik (.jpg/.png)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Membuka dan menampilkan gambar yang diunggah
    img = Image.open(uploaded_file)
    st.image(img, caption="Gambar Batik yang Diunggah", use_container_width=True)
    
    # Tombol pemicu prediksi
    if st.button("Mulai Analisis Prediksi"):
        with st.spinner("Model sedang menghitung fitur spasial piksel..."):
            # Pemrosesan gambar agar sesuai input CNN (150x150)
            img_resized = img.resize((150, 150))
            img_array = np.array(img_resized)
            
            # Jika gambar memiliki channel RGBA, konversi ke RGB
            if img_array.shape[-1] == 4:
                img_array = img_array[..., :3]
                
            # Normalisasi & penambahan dimensi batch
            img_array = img_array / 255.0
            img_tensor = np.expand_dims(img_array, axis=0)
            
            # Prediksi dengan model
            prediction = model.predict(img_tensor)[0][0]
            
            # Logika Sigmoid (Dekat ke 1 = Tradisional, Dekat ke 0 = Modern)
            prob_traditional = float(prediction)
            prob_modern = 1.0 - prob_traditional
            
            # Tampilan Hasil Akhir
            st.write("### **Hasil Analisis Model:**")
            if prob_modern > prob_traditional:
                st.info(f"🏆 Kesimpulan: **Batik Modern** (Keyakinan: {prob_modern*100:.2f}%)")
            else:
                st.success(f"🏆 Kesimpulan: **Batik Tradisional** (Keyakinan: {prob_traditional*100:.2f}%)")
                
            # Menampilkan Grafik Batang Probabilitas
            chart_data = {
                "Kategori": ["Batik Modern", "Batik Tradisional"],
                "Probabilitas (%)": [prob_modern * 100, prob_traditional * 100]
            }
            st.bar_chart(data=chart_data, x="Kategori", y="Probabilitas (%)")