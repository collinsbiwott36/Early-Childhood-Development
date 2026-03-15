import streamlit as st
import requests
import os

st.title("Individual Prediction")
st.write("Enter child/household info to generate risk score + explanation.")

API_URL = os.getenv("API_URL", "http://127.0.0.1:5000")

if st.button("Predict Risk"):
    # Example payload
    payload = {"child_age_months": 48, "county": "Nairobi"}
    try:
        resp = requests.post(f"{API_URL}/predict", json=payload)
        if resp.status_code == 200:
            st.json(resp.json())
        else:
            st.error(f"API Error: {resp.status_code}")
    except Exception as e:
        st.error(f"Connection Failed: {e}")
