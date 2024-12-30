import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Judul aplikasi
st.title("Prediksi Harga Rumah dengan Streamlit")

# Upload file data
data_file = st.file_uploader("Upload file dataset (CSV)", type=["csv"])

if data_file is not None:
    # Membaca dataset
    try:
        df = pd.read_csv(data_file)
        st.success("Dataset berhasil dimuat!")

        # Menampilkan data
        st.subheader("Data")
        st.write(df.head())

        # Menampilkan informasi dataset
        st.subheader("Informasi Dataset")
        st.write(df.describe())

        # Memilih fitur dan target
        st.subheader("Pilih Fitur dan Target")
        all_columns = df.columns.tolist()
        target_column = st.selectbox("Pilih kolom target (harga rumah)", all_columns)
        feature_columns = st.multiselect("Pilih kolom fitur", all_columns, default=all_columns[:-1])

        if target_column and feature_columns:
            X = df[feature_columns]
            y = df[target_column]

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Melatih model
            st.subheader("Pelatihan Model")
            model = RandomForestRegressor(random_state=42)
            model.fit(X_train, y_train)

            # Evaluasi model
            st.subheader("Evaluasi Model")
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            st.write(f"Mean Squared Error: {mse}")
            st.write(f"R2 Score: {r2}")

            # Prediksi
            st.subheader("Prediksi Harga Rumah")
            input_data = {col: st.number_input(f"Masukkan nilai untuk {col}", value=0) for col in feature_columns}

            if st.button("Prediksi"):
                input_df = pd.DataFrame([input_data])
                prediction = model.predict(input_df)
                st.success(f"Prediksi Harga Rumah: {prediction[0]}")

    except Exception as e:
        st.error(f"Terjadi kesalahan saat memuat dataset: {e}")
else:
    st.info("Silakan upload file dataset untuk memulai.")
