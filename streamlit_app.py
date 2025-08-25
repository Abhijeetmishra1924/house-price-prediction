import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib
import os

# -----------------------------
# App Title
# -----------------------------
st.title("House Price Prediction")
st.write("Click on arrow to Predict the sale price of a house based on its features. Model trained on Ames, Iowa housing data.")

# -----------------------------
# Load Training Data
# -----------------------------
@st.cache_data
def load_data():
    if not os.path.exists('train.csv'):
        st.error("❌ train.csv not found. Please ensure it's in the same folder as this app.")
        return None
    return pd.read_csv('train.csv')

train_df = load_data()

if train_df is None:
    st.stop()

# -----------------------------
# Preprocess and Train Model
# -----------------------------
@st.cache_resource
def train_model(df):
    # Features we'll use for prediction
    selected_features = [
        'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF',
        'YearBuilt', 'FullBath', 'TotRmsAbvGrd', 'YearRemodAdd',
        'ExterQual', 'Foundation', 'CentralAir'
    ]

    X = df[selected_features].copy()
    y = df['SalePrice']

    # Handle missing values
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    categorical_cols = X.select_dtypes(include=['object']).columns

    for col in numeric_cols:
        X[col] = X[col].fillna(X[col].median())
    for col in categorical_cols:
        X[col] = X[col].fillna(X[col].mode()[0])

    # One-hot encode categorical variables
    X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    return model, X.columns.tolist()

model, feature_columns = train_model(train_df)

# -----------------------------
# User Input Form
# -----------------------------
st.sidebar.header("Enter House Details")

with st.sidebar.form("prediction_form"):
    OverallQual = st.slider("Overall Quality (1-10)", 1, 10, 5, help="1=Poor, 10=Excellent")
    GrLivArea = st.number_input("Above Ground Living Area (sq ft)", 500, 5000, 1500)
    GarageCars = st.slider("Garage Size (Car Capacity)", 0, 4, 2)
    TotalBsmtSF = st.number_input("Total Basement Area (sq ft)", 0, 3000, 1000)
    YearBuilt = st.slider("Year Built", 1880, 2020, 1990)
    FullBath = st.slider("Full Bathrooms", 0, 4, 2)
    TotRmsAbvGrd = st.slider("Total Rooms Above Ground", 2, 12, 6)
    YearRemodAdd = st.slider("Remodel Year", 1950, 2020, 2000)
    ExterQual = st.selectbox("Exterior Quality", ["Ex", "Gd", "TA", "Fa"])
    Foundation = st.selectbox("Foundation Type", ["PConc", "CBlock", "BrkTil", "Wood", "Slab", "Stone"])
    CentralAir = st.radio("Central Air Conditioning", ["Y", "N"])

    submitted = st.form_submit_button("Predict Price")

# -----------------------------
# Make Prediction
# -----------------------------
if submitted:
    with st.spinner("Predicting house price..."):

        # Create input dataframe
        input_data = pd.DataFrame({
            'OverallQual': [OverallQual],
            'GrLivArea': [GrLivArea],
            'GarageCars': [GarageCars],
            'TotalBsmtSF': [TotalBsmtSF],
            'YearBuilt': [YearBuilt],
            'FullBath': [FullBath],
            'TotRmsAbvGrd': [TotRmsAbvGrd],
            'YearRemodAdd': [YearRemodAdd],
            'ExterQual': [ExterQual],
            'Foundation': [Foundation],
            'CentralAir': [CentralAir]
        })

        # One-hot encode
        input_data = pd.get_dummies(input_data)

        # Align with training data
        for col in feature_columns:
            if col not in input_data.columns:
                input_data[col] = 0
        input_data = input_data[feature_columns]

        # Predict
        predicted_price = model.predict(input_data)[0]

        # Display result
        st.success("Prediction Complete")
        st.write(f"### Predicted Sale Price: ${predicted_price:,.2f}")
        st.write(f"_Model trained on {len(train_df)} house sales in Ames, Iowa._")

# -----------------------------
# Optional: Show Data Info
# -----------------------------
with st.sidebar.expander("Dataset Info"):
    st.write(f"Training samples: {len(train_df)}")
    st.write(f"Features used: {len([c for c in feature_columns if 'OverallQual' in c or 'GrLivArea' in c or 'GarageCars' in c]) + 8}")
    st.write("Top predictors: Quality, Size, Year Built, Basement, Garage")
