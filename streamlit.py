import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import streamlit as st
st.title("Tes Streamlit")
st.write("Jika ini muncul, aplikasi Streamlit Anda berjalan dengan baik!")


# Judul aplikasi
st.title("Analisis dan Prediksi Penjualan")

# Upload file dataset
data_file = st.file_uploader("Upload file dataset (XLSX)", type=["xlsx"])

if data_file is not None:
    # Membaca dataset
    try:
        df = pd.read_excel(data_file)
        st.success("Dataset berhasil dimuat!")

        # Menampilkan data
        st.subheader("Data")
        st.write(df.head())

        # Informasi dataset
        st.subheader("Informasi Dataset")
        st.write(df.info())

        # Statistik deskriptif
        st.subheader("Statistik Deskriptif")
        st.write(df.describe())

        # Mengecek missing values
        st.subheader("Cek Missing Values")
        st.write(df.isnull().sum())

        # Konversi tipe data jika diperlukan
        if 'Curah Hujan (mm)' in df.columns:
            df['Curah Hujan (mm)'] = df['Curah Hujan (mm)'].astype('int')

        # Visualisasi Data
        st.subheader("Visualisasi Data")
        selected_column = st.selectbox("Pilih Kolom untuk Visualisasi", df.columns)
        if selected_column:
            fig, ax = plt.subplots(1, 2, figsize=(12, 4))
            sns.countplot(df[selected_column], ax=ax[0])
            ax[0].set_title(f"Countplot: {selected_column}")
            ax[1].boxplot(df[selected_column])
            ax[1].set_title(f"Boxplot: {selected_column}")
            st.pyplot(fig)

        # Korelasi Data
        st.subheader("Korelasi Antar Variabel")
        st.write(df.corr())

        # Pemodelan Data
        st.subheader("Pemodelan Data")
        if 'Penjualan (pcs)' in df.columns:
            x = df.drop(columns='Penjualan (pcs)')
            y = df['Penjualan (pcs)']

            # Split data
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=4)

            # Linear Regression
            lin_reg = LinearRegression()
            lin_reg.fit(x_train, y_train)

            # Koefisien dan Intercept
            st.write("Koefisien Model:", lin_reg.coef_)
            st.write("Intercept Model:", lin_reg.intercept_)

            # Akurasi Model
            score = lin_reg.score(x_test, y_test)
            st.write("Akurasi Model (R²):", score)

            # Prediksi Penjualan Optimal
            st.subheader("Prediksi Penjualan Optimal")
            hari = st.number_input("Masukkan Hari", value=0)
            tanggal = st.number_input("Masukkan Tanggal", value=1)
            kegiatan = st.number_input("Masukkan Kegiatan", value=1)
            curah_hujan = st.number_input("Masukkan Curah Hujan (mm)", value=0)

            if st.button("Prediksi"):
                prediksi = lin_reg.predict([[hari, tanggal, kegiatan, curah_hujan]])
                st.success(f"Hasil Prediksi Penjualan: {prediksi[0]} pcs")

    except Exception as e:
        st.error(f"Terjadi kesalahan saat membaca dataset: {e}")
else:
    st.info("Silakan upload file dataset untuk memulai.")
