"""
RideGuard AI - Utility & Helper Functions
Plotly styling, CSS injection, KPI card rendering, and synthetic dataset generation.
"""

from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import random

from config.settings import THEME, ASSETS_DIR, RAW_DATA_DIR, DEFAULT_DATASET_PATH
from utils.ui_components import render_kpi_card as ui_render_kpi_card, render_risk_badge as ui_render_risk_badge


def load_custom_css():
    """Injects custom CSS from assets/styles.css into Streamlit app."""
    css_path = ASSETS_DIR / "styles.css"
    if css_path.exists():
        with open(css_path, "r", encoding="utf-8") as f:
            css_content = f.read()
        st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)


def apply_plotly_theme(fig: go.Figure) -> go.Figure:
    """Applies a consistent sleek dark theme to any Plotly figure."""
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(17, 27, 43, 0.4)",
        font=dict(
            family="Inter, -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif",
            color=THEME["text_secondary"],
            size=11
        ),
        title=dict(
            font=dict(
                family="Inter, -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif",
                color=THEME["text_primary"],
                size=13,
                weight=600
            )
        ),
        legend=dict(
            bgcolor="rgba(17, 27, 43, 0.9)",
            bordercolor=THEME["border"],
            borderwidth=1,
            font=dict(color=THEME["text_secondary"], size=10)
        ),
        margin=dict(l=24, r=24, t=40, b=30),
        xaxis=dict(
            gridcolor="rgba(255, 255, 255, 0.04)",
            linecolor=THEME["border"],
            zerolinecolor="rgba(255, 255, 255, 0.05)"
        ),
        yaxis=dict(
            gridcolor="rgba(255, 255, 255, 0.04)",
            linecolor=THEME["border"],
            zerolinecolor="rgba(255, 255, 255, 0.05)"
        )
    )
    return fig


def render_kpi_card(title: str, value: str, subtitle: str = "", variant: str = "neutral", icon_key: str = "rides"):
    """Renders a modern enterprise metric card."""
    ui_render_kpi_card(title, value, subtitle, icon_key=icon_key, variant=variant)


def render_risk_badge(probability: float, thresholds: dict) -> str:
    """Returns HTML badge according to cancellation risk category."""
    return ui_render_risk_badge(probability, thresholds)


def get_risk_label(probability: float, thresholds: dict) -> str:
    """Returns 'Low', 'Medium', or 'High' based on risk threshold."""
    if probability < thresholds.get("low", 0.30):
        return "Low"
    elif probability <= thresholds.get("medium", 0.60):
        return "Medium"
    else:
        return "High"


def generate_synthetic_rides(n_samples: int = 3000, random_seed: int = 42) -> pd.DataFrame:
    """
    Generates a realistic, statistically grounded ride-hailing dataset
    mirroring platforms like Uber/Ola with genuine real-world cancellation dynamics.
    """
    np.random.seed(random_seed)
    random.seed(random_seed)

    vehicle_types = ["Auto", "Mini / Go", "Prime Sedan", "Prime SUV", "Bike Taxi"]
    vehicle_weights = [0.28, 0.32, 0.20, 0.12, 0.08]

    locations = [
        "Airport Terminal 1",
        "Tech Park Phase 2",
        "Central Railway Station",
        "Financial District",
        "Downtown Arts District",
        "Suburban Residential West",
        "University Campus",
        "Metro Transit Hub",
        "Shopping Galleria"
    ]

    payment_methods = ["UPI / Digital Wallet", "Credit / Debit Card", "Cash", "Corporate Account"]
    payment_weights = [0.55, 0.25, 0.15, 0.05]

    start_date = datetime(2026, 8, 1)
    records = []

    for i in range(1, n_samples + 1):
        booking_id = f"RG-{20260000 + i}"
        days_offset = random.randint(0, 90)
        date_obj = start_date + timedelta(days=days_offset)
        booking_date = date_obj.strftime("%Y-%m-%d")

        # Peak hours: 8-10 AM and 5-9 PM
        hour_weights = np.array([
            0.01, 0.01, 0.01, 0.01, 0.02, 0.03, # 0-5
            0.05, 0.08, 0.10, 0.09, 0.06, 0.04, # 6-11
            0.04, 0.04, 0.04, 0.05, 0.07, 0.09, # 12-17
            0.10, 0.08, 0.04, 0.02, 0.01, 0.01  # 18-23
        ])
        hour_p = hour_weights / hour_weights.sum()
        hour = int(np.random.choice(range(24), p=hour_p))
        minute = random.randint(0, 59)
        booking_time = f"{hour:02d}:{minute:02d}"

        vehicle = np.random.choice(vehicle_types, p=vehicle_weights)
        pickup = np.random.choice(locations)
        drop = np.random.choice([loc for loc in locations if loc != pickup])

        # Distance logic: Airport trips are longer
        if "Airport" in pickup or "Airport" in drop:
            ride_dist = np.round(np.random.normal(28.0, 7.0), 1)
        else:
            ride_dist = np.round(np.random.gamma(shape=2.5, scale=3.5), 1)
        ride_dist = max(1.2, ride_dist)

        # Fare pricing per km based on vehicle
        fare_multiplier = {
            "Bike Taxi": 8.0,
            "Auto": 14.0,
            "Mini / Go": 20.0,
            "Prime Sedan": 26.0,
            "Prime SUV": 36.0
        }[vehicle]

        base_fare = 40.0
        surge = 1.0
        is_peak = (8 <= hour <= 10) or (17 <= hour <= 21)
        if is_peak:
            surge = np.random.choice([1.2, 1.4, 1.6, 1.8], p=[0.4, 0.3, 0.2, 0.1])

        booking_val = np.round((base_fare + (ride_dist * fare_multiplier)) * surge, 0)

        # Driver Arrival Time (ETA in minutes)
        if is_peak or "Airport" in pickup:
            driver_arrival_time = int(np.clip(np.random.normal(12.5, 5.0), 2, 35))
        else:
            driver_arrival_time = int(np.clip(np.random.normal(6.5, 3.0), 1, 25))

        # Waiting Time (pickup delay beyond arrival)
        waiting_time = int(np.clip(np.random.exponential(scale=3.5), 0, 25))

        customer_rating = round(float(np.clip(np.random.normal(4.6, 0.45), 2.5, 5.0)), 1)
        payment = np.random.choice(payment_methods, p=payment_weights)

        # Cancellation probability based on realistic features
        # Core drivers: High driver arrival time, excessive waiting, peak hour surge, distance extremes
        logit = -2.2  # baseline ~10% cancellation
        logit += 0.12 * (driver_arrival_time - 6.0)
        logit += 0.15 * (waiting_time - 2.0)
        if is_peak:
            logit += 0.45
        if vehicle in ["Auto", "Bike Taxi"]:
            logit += 0.35
        if "Airport" in pickup and driver_arrival_time > 15:
            logit += 0.60
        if surge > 1.3:
            logit += 0.30

        prob_cancel = 1.0 / (1.0 + np.exp(-logit))
        prob_cancel = np.clip(prob_cancel, 0.02, 0.95)

        is_cancelled = np.random.binomial(1, prob_cancel)
        booking_status = "Cancelled" if is_cancelled == 1 else "Completed"

        cancellation_reason = "Not Applicable"
        if is_cancelled == 1:
            reasons = [
                "Driver delayed / ETA increased",
                "Customer changed plans",
                "Driver requested offline payment / direct cancellation",
                "Expected shorter waiting time",
                "Found alternative transport",
                "Driver heading opposite direction"
            ]
            cancellation_reason = np.random.choice(reasons)

        records.append({
            "Booking_ID": booking_id,
            "Booking_Date": booking_date,
            "Booking_Time": booking_time,
            "Hour": hour,
            "Vehicle_Type": vehicle,
            "Pickup_Location": pickup,
            "Drop_Location": drop,
            "Ride_Distance": ride_dist,
            "Booking_Value": booking_val,
            "Driver_Arrival_Time": driver_arrival_time,
            "Waiting_Time": waiting_time,
            "Customer_Rating": customer_rating,
            "Payment_Method": payment,
            "Booking_Status": booking_status,
            "Cancellation_Reason": cancellation_reason
        })

    df = pd.DataFrame(records)
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(DEFAULT_DATASET_PATH, index=False)
    return df
