import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from datetime import timedelta

# ===============================
# Load model dan scaler
# ===============================
MODEL_PATH = "model_h3.pkl"
SCALER_PATH = "scaler_h3.pkl"
DATA_PATH = "sample_data/data_no2.csv"

# Load model dan scaler
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# ===============================
# Load data historis
# ===============================
try:
    df = pd.read_csv(DATA_PATH, parse_dates=['date'])
    df = df.sort_values('date').reset_index(drop=True)

    if 'NO2' not in df.columns:
        raise ValueError("Kolom 'NO2' tidak ditemukan dalam dataset.")

    df['h1'] = df['NO2'].shift(1)
    df['h2'] = df['NO2'].shift(2)
    df['h3'] = df['NO2'].shift(3)
    df = df.dropna().reset_index(drop=True)

except Exception as e:
    df = None
    st.error(f"⚠️ Gagal memuat data historis: {e}")

# ===============================
# Fungsi Kategori Udara WHO
# ===============================
def kategori_no2(value):
    if value < 0.000020:
        return "🟢 Baik"
    elif value < 0.000040:
        return "🟡 Sedang ⚠️"
    elif value < 0.000060:
        return "🟠 Tidak Sehat 🚫"
    else:
        return "🔴 Sangat Tidak Sehat ☠️"

# ===============================
# UI Streamlit
# ===============================
st.title("🌫️ Aplikasi Prediksi Konsentrasi NO₂ Harian")
st.write("Gunakan aplikasi ini untuk melakukan prediksi otomatis atau manual terhadap kadar NO₂ harian berdasarkan model yang sudah dilatih.")

mode = st.radio("Pilih Mode Prediksi:", ["Prediksi Otomatis", "Prediksi Manual Interaktif"])

# ===============================
# MODE 1: Prediksi Otomatis
# ===============================
if mode == "Prediksi Otomatis":
    st.subheader("📆 Prediksi Otomatis Hari Berikutnya")

    if df is not None and not df.empty:
        last_row = df.iloc[-1]
        next_day = last_row['date'] + timedelta(days=1)

        last_features = pd.DataFrame([[last_row['h1'], last_row['h2'], last_row['h3']]],
                                     columns=['h1', 'h2', 'h3'])

        last_scaled = scaler.transform(last_features)
        pred = model.predict(last_scaled)[0]

        kategori = kategori_no2(pred)

        st.success(f"Prediksi konsentrasi NO₂ untuk tanggal **{next_day.strftime('%Y-%m-%d')}** adalah **{pred:.6f} mol/m²**")
        st.info(f"Kategori Udara (WHO): **{kategori}**")

        # Plot hasil
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(df['date'], df['NO2'], label='Data Historis', marker='o')
        ax.scatter(next_day, pred, color='red', label='Prediksi', s=80)
        ax.set_title("Prediksi Otomatis Konsentrasi NO₂")
        ax.set_xlabel("Tanggal")
        ax.set_ylabel("Konsentrasi NO₂ (mol/m²)")
        ax.legend()
        st.pyplot(fig)
    else:
        st.error("❌ Data historis tidak ditemukan atau kosong.")

# ===============================
# MODE 2: Prediksi Manual Interaktif
# ===============================
else:
    st.subheader("🧮 Prediksi Manual Interaktif")
    st.write("Masukkan nilai konsentrasi NO₂ dari 3 hari terakhir:")

    h1 = st.number_input("Hari ke-1 (H-1)", min_value=0.0, format="%.8f")
    h2 = st.number_input("Hari ke-2 (H-2)", min_value=0.0, format="%.8f")
    h3 = st.number_input("Hari ke-3 (H-3)", min_value=0.0, format="%.8f")

    if st.button("Prediksi Sekarang"):
        manual_features = pd.DataFrame([[h1, h2, h3]], columns=['h1', 'h2', 'h3'])
        manual_scaled = scaler.transform(manual_features)
        pred = model.predict(manual_scaled)[0]

        kategori = kategori_no2(pred)

        st.success(f"Perkiraan konsentrasi NO₂ adalah **{pred:.6f} mol/m²**")
        st.info(f"Kategori Udara (WHO): **{kategori}**")

# ===============================
# Footer
# ===============================
st.markdown("---")
st.caption("Dibuat oleh Dwi Prasetya Mumtaz menggunakan Streamlit")
