import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# ==============================
# 1️⃣  LOAD DATA DAN MODEL
# ==============================
@st.cache_data
def load_data():
    df = pd.read_csv("data_no2.csv")
    df['date'] = pd.to_datetime(df['date'])
    return df

@st.cache_resource
def load_model():
    knn_h3 = joblib.load("model_h3.pkl")
    scaler_h3 = joblib.load("scaler_h3.pkl")
    return knn_h3, scaler_h3

# Load
new_df = load_data()
knn_h3, scaler_h3 = load_model()


# ==============================
# 2️⃣  SIDEBAR
# ==============================
st.sidebar.title("🌍 NO₂ Forecasting App")
st.sidebar.write("Prediksi kadar NO₂ untuk hari berikutnya menggunakan model KNN (h1–h3).")

show_data = st.sidebar.checkbox("Tampilkan Data Awal", value=False)
if show_data:
    st.subheader("📊 Data Awal")
    st.dataframe(new_df.tail(10))


# ==============================
# 3️⃣  VISUALISASI DATA
# ==============================
st.subheader("📈 Tren Konsentrasi NO₂")
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(new_df['date'], new_df['NO2'], label='NO₂ (mol/m²)')
ax.set_xlabel("Tanggal")
ax.set_ylabel("Konsentrasi NO₂ (mol/m²)")
ax.legend()
st.pyplot(fig)


# ==============================
# 4️⃣  FUNGSI PREDIKSI
# ==============================
def prediksi_hari_berikutnya(data, model, scaler, n_hari=3):
    last_vals = data['NO2'].values[-n_hari:]
    feature_names = [f"h{i}" for i in range(1, n_hari + 1)]
    input_df = pd.DataFrame([last_vals], columns=feature_names)
    input_scaled = scaler.transform(input_df)
    return model.predict(input_scaled)[0]


# ==============================
# 5️⃣  PREDIKSI
# ==============================
st.subheader("🔮 Prediksi Hari Berikutnya")

if st.button("Prediksi Sekarang"):
    predicted_no2 = prediksi_hari_berikutnya(new_df, knn_h3, scaler_h3, n_hari=3)

    median_val = new_df['NO2'].quantile(0.50)
    upper_quantile_val = new_df['NO2'].quantile(0.75)

    if predicted_no2 <= median_val:
        kategori = "Baik"
        warna = "🟢"
    elif predicted_no2 <= upper_quantile_val:
        kategori = "Sedang"
        warna = "🟡"
    else:
        kategori = "Tinggi (Tidak Baik)"
        warna = "🔴"

    st.success(f"**Prediksi Konsentrasi NO₂:** {predicted_no2:.8f} mol/m²")
    st.info(f"**Kategori Kualitas Udara:** {warna} {kategori}")
    st.write("---")

    # Tampilkan batas-batas statistik
    st.write(f"Median (Batas Baik): `{median_val:.8f}` mol/m²")
    st.write(f"Kuantil Atas (Batas Sedang): `{upper_quantile_val:.8f}` mol/m²")

    # Visual prediksi
    st.subheader("📉 Visualisasi Prediksi")
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    ax2.plot(new_df['date'], new_df['NO2'], label="Data Historis", color='blue')
    ax2.axhline(y=predicted_no2, color='red', linestyle='--', label="Prediksi Hari Berikutnya")
    ax2.legend()
    st.pyplot(fig2)


# ==============================
# 6️⃣  FOOTER
# ==============================
st.markdown("---")
st.caption("Dibuat oleh: **Dwi Prasetya Mumtaz** | Model: KNN Forecast NO₂ | Streamlit Deployment ✅")
