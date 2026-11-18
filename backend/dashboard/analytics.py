"""
RideGuard AI - Page 2: Cancellation Analytics
Enterprise mobility analytics with responsive two-column grid and clean filtering.
"""

import streamlit as st
import pandas as pd
from utils.ui_components import render_top_navbar, render_empty_state
from analytics.eda import RideAnalytics


def render_analytics_page():
    if "df" not in st.session_state or st.session_state.df is None:
        render_top_navbar("Cancellation Analytics", breadcrumb="Analytics", row_count=0)
        render_empty_state(
            "No Dataset Available",
            "Please navigate to 'Dataset Management' to load or upload ride records before analyzing cancellation trends."
        )
        return

    df = st.session_state.df.copy()
    target_col = st.session_state.get("target_col", "Booking_Status")
    is_custom = st.session_state.get("is_custom_dataset", False)
    best_name = st.session_state.get("best_model_name", "XGBoost")

    # Top Navbar
    render_top_navbar(
        "Cancellation Analytics",
        breadcrumb="Analytics",
        is_custom_data=is_custom,
        row_count=len(df),
        model_name=best_name
    )

    # Clean Filter Bar in Elevated Panel
    with st.expander("Filter Data Records", expanded=False):
        f_col1, f_col2, f_col3 = st.columns(3)
        with f_col1:
            vehicles = ["All Vehicle Classes"] + sorted(df["Vehicle_Type"].dropna().unique().tolist()) if "Vehicle_Type" in df.columns else ["All Vehicle Classes"]
            selected_vehicle = st.selectbox("Vehicle Category", vehicles)
        with f_col2:
            locations = ["All Pickup Hubs"] + sorted(df["Pickup_Location"].dropna().unique().tolist()) if "Pickup_Location" in df.columns else ["All Pickup Hubs"]
            selected_loc = st.selectbox("Pickup Location", locations)
        with f_col3:
            hour_range = st.slider("Booking Hour Window", 0, 23, (0, 23))

    filtered_df = df.copy()
    if selected_vehicle != "All Vehicle Classes":
        filtered_df = filtered_df[filtered_df["Vehicle_Type"] == selected_vehicle]
    if selected_loc != "All Pickup Hubs":
        filtered_df = filtered_df[filtered_df["Pickup_Location"] == selected_loc]
    if "Hour" in filtered_df.columns:
        filtered_df = filtered_df[(filtered_df["Hour"] >= hour_range[0]) & (filtered_df["Hour"] <= hour_range[1])]

    analytics = RideAnalytics(filtered_df, target_col=target_col)

    # Row 1: Vehicle Analysis & Hourly Trend
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(analytics.plot_vehicle_cancellations(), use_container_width=True)
    with col2:
        st.plotly_chart(analytics.plot_hourly_trend(), use_container_width=True)

    # Row 2: Location Analysis & Waiting Time Impact
    col3, col4 = st.columns(2)
    with col3:
        st.plotly_chart(analytics.plot_location_analysis(top_n=8), use_container_width=True)
    with col4:
        st.plotly_chart(analytics.plot_waiting_time_impact(), use_container_width=True)

    # Row 3: Distance Analysis & Correlation Matrix
    col5, col6 = st.columns(2)
    with col5:
        st.plotly_chart(analytics.plot_fare_distance_distribution(), use_container_width=True)
    with col6:
        st.plotly_chart(analytics.plot_correlation_heatmap(), use_container_width=True)
