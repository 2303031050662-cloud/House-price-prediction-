import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

import streamlit as st
import joblib
import pandas as pd
from src.features import add_features

model = joblib.load(ROOT / "models" / "model.pkl")

st.title("California House Price Predictor")

income = st.slider("Median income (x $10k)", 0.5, 15.0, 4.0)
age = st.slider("Housing median age", 1, 52, 20)
rooms = st.number_input("Total rooms", 1, 40000, 2000)
bedrooms = st.number_input("Total bedrooms", 1, 7000, 400)
population = st.number_input("Population", 1, 40000, 1200)
households = st.number_input("Households", 1, 7000, 400)
lat = st.number_input("Latitude", 32.5, 42.0, 34.0)
lon = st.number_input("Longitude", -124.5, -114.0, -118.0)
ocean = st.selectbox("Ocean proximity",
                     ["<1H OCEAN", "INLAND", "ISLAND", "NEAR BAY", "NEAR OCEAN"])

if st.button("Predict"):
    row = pd.DataFrame([{
        "longitude": lon, "latitude": lat,
        "housing_median_age": age, "total_rooms": rooms,
        "total_bedrooms": bedrooms, "population": population,
        "households": households, "median_income": income,
        "ocean_proximity": ocean,
    }])
    st.success(f"Estimated value: ${model.predict(row)[0]:,.0f}")
