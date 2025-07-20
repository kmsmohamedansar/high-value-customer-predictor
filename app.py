import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap
import matplotlib.pyplot as plt

# Load model and encoders
with open("model.pkl", "rb") as f:
    model = pickle.load(f)
with open("le_region.pkl", "rb") as f:
    le_region = pickle.load(f)
with open("le_segment.pkl", "rb") as f:
    le_segment = pickle.load(f)

st.set_page_config(page_title="💎 High-Value Customer Predictor")
st.title("💎 High-Value Customer Predictor")
st.markdown("Enter customer details below to predict if they are high-value.")

# Layout inputs in two columns
col1, col2 = st.columns(2)
with col1:
    recency_days = st.number_input("📅 Recency (days since last purchase)", min_value=0, value=30)
    frequency = st.number_input("🔁 Frequency (number of orders)", min_value=1, value=5)
    monetary_value = st.number_input("💰 Monetary Value (total sales)", min_value=0.0, value=1000.0)
    avg_order_value = st.number_input("🛒 Average Order Value", min_value=0.0, value=200.0)

with col2:
    total_profit = st.number_input("📈 Total Profit", min_value=0.0, value=100.0)
    avg_days_between_orders = st.number_input("⏳ Avg Days Between Orders", min_value=0.0, value=30.0)
    region = st.selectbox("📍 Region", le_region.classes_)
    segment = st.selectbox("👤 Segment", le_segment.classes_)

# Encode categorical inputs
region_enc = le_region.transform([region])[0]
segment_enc = le_segment.transform([segment])[0]

input_data = pd.DataFrame([[
    recency_days, frequency, monetary_value,
    avg_order_value, total_profit, avg_days_between_orders,
    region_enc, segment_enc
]], columns=[
    'recency_days', 'frequency', 'monetary_value',
    'avg_order_value', 'total_profit', 'avg_days_between_orders',
    'region_enc', 'segment_enc'
])

if st.button("🚀 Predict"):
    pred = model.predict(input_data)[0]
    proba = model.predict_proba(input_data)[0][1]
    
    if pred == 1:
        st.success(f"✅ Predicted HIGH VALUE with {proba:.2%} confidence.")
    else:
        st.info(f"ℹ️ Predicted NOT high value ({proba:.2%} confidence).")

    # SHAP explanation
    explainer = shap.Explainer(model)
    shap_values = explainer(input_data)

    st.subheader("🔍 Feature Contribution (SHAP)")
    fig, ax = plt.subplots()
    shap.plots.waterfall(shap_values[0], max_display=8, show=False)
    st.pyplot(fig)
