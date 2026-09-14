import json
import pandas as pd
import streamlit as st
from src.config import ARTIFACTS
from src.inference import predict

st.set_page_config(page_title="RideSurge", page_icon="🚕", layout="wide")
st.title("🚕 RideSurge — Demand-Adaptive Pricing")
summary_file = ARTIFACTS / "run_summary.json"
if not summary_file.exists():
    st.error("Run `python train.py` first."); st.stop()
s = json.loads(summary_file.read_text())
m = s["measured"]
c1, c2, c3 = st.columns(3)
c1.metric("Duration–cost correlation", f'{m["correlation"]:.3f}')
c2.metric("CatBoost R²", f'{m["catboost"]["r2"]:.3f}')
c3.metric("Simulated profit uplift", f'{m["revenue"]["simulated_profit_uplift_pct"]:.1f}%')

with st.form("fare"):
    a, b = st.columns(2)
    riders = a.number_input("Riders requesting", 1, 10000, 70)
    drivers = a.number_input("Available drivers", 1, 10000, 25)
    duration = a.number_input("Expected duration (minutes)", 1.0, 500.0, 45.0)
    past = a.number_input("Past rides", 0, 10000, 20)
    rating = b.slider("Average rating", 1.0, 5.0, 4.2)
    location = b.selectbox("Location", ["Urban", "Suburban", "Rural"])
    loyalty = b.selectbox("Loyalty", ["Regular", "Silver", "Gold"])
    booking = b.selectbox("Booking time", ["Morning", "Afternoon", "Evening", "Night"])
    vehicle = b.selectbox("Vehicle", ["Economy", "Premium"])
    submitted = st.form_submit_button("Recommend fare", type="primary")
if submitted:
    result = predict({"number_of_riders": riders, "number_of_drivers": drivers,
        "number_of_past_rides": past, "average_ratings": rating,
        "expected_ride_duration": duration, "location_category": location,
        "customer_loyalty_status": loyalty, "time_of_booking": booking, "vehicle_type": vehicle})
    st.success(f'Recommended fare: {result["recommended_fare"]:,.2f}')
    st.write(result)
st.subheader("Seven-model benchmark")
st.dataframe(pd.read_csv(ARTIFACTS / "seven_model_comparison.csv"), hide_index=True, use_container_width=True)
st.subheader("SHAP global importance")
st.bar_chart(pd.read_csv(ARTIFACTS / "shap_rankings.csv").set_index("feature"))

