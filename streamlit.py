# Import necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Set the title of the app
st.title("Sales Prediction App")

# File upload section
st.sidebar.header("Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Upload your Excel file", type=["xlsx"])

if uploaded_file is not None:
    # Read the uploaded file
    df = pd.read_excel(uploaded_file, engine='openpyxl')

    # Show the first 5 rows of the dataset
    st.header("Dataset Overview")
    st.write(df.head())

    # Show general information about the dataset
    st.subheader("Data Information")
    st.write(df.info())

    # Show basic statistics
    st.subheader("Statistical Summary")
    st.write(df.describe())

    # Show missing values
    st.subheader("Missing Values")
    st.write(df.isnull().sum())

    # Data visualization section
    st.header("Exploratory Data Analysis (EDA)")

    # Visualizing distributions for features
    st.subheader("Visualize Data Distribution")
    columns = df.columns
    feature = st.selectbox("Choose a feature to analyze", columns)
    fig = plt.figure(figsize=(10, 4))
    sns.histplot(df[feature], kde=True)
    st.pyplot(fig)

    # Correlation heatmap
    st.subheader("Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    # Data preprocessing for modeling
    st.header("Sales Prediction Model")

    # Preprocess the data (convert 'Curah Hujan' to int, if needed)
    if 'Curah Hujan (mm)' in df.columns:
        df['Curah Hujan (mm)'] = df['Curah Hujan (mm)'].astype(int)

    # Split data into features (X) and target (y)
    X = df.drop(columns='Penjualan (pcs)')
    y = df['Penjualan (pcs)']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)

    # Create and train the linear regression model
    lin_reg = LinearRegression()
    lin_reg.fit(X_train, y_train)

    # Model evaluation
    st.subheader("Model Performance")
    accuracy = lin_reg.score(X_test, y_test)
    st.write(f"Model R² Score: {accuracy:.2f}")

    # Predict sales based on user inputs
    st.header("Sales Prediction")
    st.write("Provide the following values to predict the sales:")

    hari = st.number_input("Hari", min_value=0, max_value=6, value=0)
    tanggal = st.number_input("Tanggal", min_value=1, max_value=31, value=1)
    kegiatan = st.number_input("Kegiatan", min_value=0, max_value=10, value=1)
    curah_hujan = st.number_input("Curah Hujan (mm)", min_value=0, max_value=500, value=0)

    # Predict the sales using the model
    prediction = lin_reg.predict([[hari, tanggal, kegiatan, curah_hujan]])

    st.write(f"Predicted Sales: {prediction[0]:,.0f} pcs")

