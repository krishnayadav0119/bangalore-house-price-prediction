import json

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Bangalore House Price Predictor", page_icon="🏠")


@st.cache_resource
def load_model():
    return joblib.load("house_price_model.pkl")


@st.cache_data
def load_options():
    with open("locations.json") as f:
        return json.load(f)


model = load_model()
options = load_options()

st.title("🏠 Bangalore House Price Predictor")
st.write("Enter the property details to get an estimated price (in lakhs of rupees).")

location = st.selectbox("Location", options["locations"])
total_sqft = st.number_input("Total area (sqft)", min_value=300, max_value=10000, value=1200, step=50)

col1, col2, col3 = st.columns(3)
bhk = col1.number_input("BHK", min_value=1, max_value=10, value=2)
bath = col2.number_input("Bathrooms", min_value=1, max_value=10, value=2)
balcony = col3.number_input("Balconies", min_value=0, max_value=3, value=1)

area_type = st.selectbox("Area type", options["area_types"])
availability = st.selectbox("Availability", ["Ready To Move", "Not Ready"])

if st.button("Predict price"):
    row = pd.DataFrame([{
        "total_sqft": total_sqft,
        "bath": bath,
        "balcony": balcony,
        "bhk": bhk,
        "location": location,
        "area_type": area_type,
        "availability": availability,
    }])
    price = model.predict(row)[0]
    st.success(f"Estimated price: ₹{price:,.1f} lakhs (about ₹{price / 100:,.2f} crore)")
    st.caption("Typical error is about 22 lakhs. Best for typical homes; luxury homes are under-predicted.")
