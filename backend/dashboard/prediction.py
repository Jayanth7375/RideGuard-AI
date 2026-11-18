"""
RideGuard AI - Page 3: Real-Time AI Prediction Engine
Two-column decision-support interface with single simulation, circular risk gauge, and live SHAP attribution.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from models.predict import RidePredictor
from explainability.shap_explainer import ShapExplainer
from utils.ui_components import render_top_navbar, render_risk_badge, render_empty_state
from utils.helpers import apply_plotly_theme
from config.settings import THEME, DEFAULT_RISK_THRESHOLDS


def render_prediction_page():
    if "df" not in st.session_state or st.session_state.df is None:
        render_top_navbar("AI Prediction", breadcrumb="AI Prediction", row_count=0)
        render_empty_state("No Dataset Available", "Please navigate to 'Dataset Management' to load training data.")
        return

    df_sample = st.session_state.df
    is_custom = st.session_state.get("is_custom_dataset", False)
    best_name = st.session_state.get("best_model_name", "XGBoost")

    # Top Navbar
    render_top_navbar(
        "AI Prediction Engine",
        breadcrumb="AI Prediction",
        is_custom_data=is_custom,
        row_count=len(df_sample),
        model_name=best_name
    )

    # Validate trained models
    if "trained_models" not in st.session_state or st.session_state.trained_models is None:
        st.info("Machine learning models are not yet compiled. Click below to initialize training on the active dataset.")
        if st.button("Initialize & Train ML Pipeline"):
            with st.spinner("Training models..."):
                from models.train_models import ModelTrainer
                from models.evaluate import ModelEvaluator
                from preprocessing.data_processor import DataProcessor

                dp = DataProcessor()
                X_train_p, X_test_p, y_train, y_test, _, _ = dp.prepare_train_test(
                    df_sample, target_col=st.session_state.get("target_col", "Booking_Status")
                )
                dp.save()

                trainer = ModelTrainer()
                trained = trainer.train_all(X_train_p, y_train)
                trainer.save_models()

                evaluator = ModelEvaluator(trained, X_test_p, y_test)
                benchmark_df = evaluator.evaluate_all()
                best_model = evaluator.get_best_model_name()

                st.session_state.preprocessor = dp
                st.session_state.trained_models = trained
                st.session_state.benchmark_df = benchmark_df
                st.session_state.evaluator = evaluator
                st.session_state.best_model_name = best_model
                st.session_state.X_train_proc = X_train_p
                st.session_state.X_test_proc = X_test_p
                st.session_state.y_test = y_test
                st.session_state.best_model_metric = benchmark_df[benchmark_df["Model"] == best_model].iloc[0].to_dict()
                st.rerun()
        return

    models = st.session_state.trained_models
    preprocessor = st.session_state.preprocessor

    # Top Controls Bar: Model & Configurable Thresholds
    ctl_col1, ctl_col2 = st.columns([1.5, 1])
    with ctl_col1:
        model_names = list(models.keys())
        default_idx = model_names.index(best_name) if best_name in model_names else 0
        selected_model_name = st.selectbox("Active Inference Model", model_names, index=default_idx)
        selected_model = models[selected_model_name]
    with ctl_col2:
        with st.expander("Threshold Settings", expanded=False):
            low_thresh = st.slider("Low Risk Cap (%)", 10, 40, int(DEFAULT_RISK_THRESHOLDS["low"] * 100)) / 100.0
            med_thresh = st.slider("Medium Risk Cap (%)", 40, 80, int(DEFAULT_RISK_THRESHOLDS["medium"] * 100)) / 100.0
            thresholds = {"low": low_thresh, "medium": med_thresh, "high": med_thresh}

    tab_single, tab_batch = st.tabs(["Single Trip Simulation", "Batch CSV Scoring"])

    with tab_single:
        # Two-Column Layout (Left Form, Right Prediction)
        col_form, col_result = st.columns([1.15, 1])

        # Retrieve dynamic dropdown options
        veh_options = sorted(df_sample["Vehicle_Type"].dropna().unique().tolist()) if "Vehicle_Type" in df_sample.columns else ["Auto", "Mini / Go", "Prime Sedan", "Prime SUV", "Bike Taxi"]
        loc_options = sorted(df_sample["Pickup_Location"].dropna().unique().tolist()) if "Pickup_Location" in df_sample.columns else ["Airport Terminal 1", "Tech Park Phase 2", "Central Railway Station", "Downtown Arts District"]
        pay_options = sorted(df_sample["Payment_Method"].dropna().unique().tolist()) if "Payment_Method" in df_sample.columns else ["UPI / Digital Wallet", "Credit / Debit Card", "Cash"]

        with col_form:
            st.markdown("""
            <div class="panel-header" style="margin-bottom: 12px;">
                <span class="panel-title">Trip Parameters</span>
                <span class="panel-desc">Pre-Dispatch Features</span>
            </div>
            """, unsafe_allow_html=True)

            sub_c1, sub_c2 = st.columns(2)
            with sub_c1:
                vehicle = st.selectbox("Vehicle Category", veh_options)
                pickup = st.selectbox("Pickup Hub", loc_options)
                distance = st.number_input("Ride Distance (km)", min_value=0.5, max_value=150.0, value=12.5, step=0.5)
                driver_eta = st.number_input("Driver Arrival Time (ETA mins)", min_value=1, max_value=60, value=8, step=1)
                payment = st.selectbox("Payment Method", pay_options)

            with sub_c2:
                drop_candidates = [l for l in loc_options if l != pickup] or loc_options
                drop = st.selectbox("Destination Hub", drop_candidates)
                fare = st.number_input("Estimated Fare ($ / ₹)", min_value=10.0, max_value=5000.0, value=260.0, step=10.0)
                waiting_time = st.number_input("Estimated Waiting Delay (mins)", min_value=0, max_value=60, value=3, step=1)
                hour = st.slider("Dispatch Hour", 0, 23, 17)
                cust_rating = st.slider("Customer Rating", 1.0, 5.0, 4.6, step=0.1)

            st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
            predict_btn = st.button("⚡ Predict Cancellation Risk", use_container_width=True)

        with col_result:
            st.markdown("""
            <div class="panel-header" style="margin-bottom: 12px;">
                <span class="panel-title">Prediction Diagnostics</span>
                <span class="panel-desc">Real-time Risk Classification</span>
            </div>
            """, unsafe_allow_html=True)

            if predict_btn or "last_prediction" in st.session_state:
                if predict_btn:
                    input_dict = {
                        "Vehicle_Type": vehicle,
                        "Pickup_Location": pickup,
                        "Drop_Location": drop,
                        "Ride_Distance": distance,
                        "Booking_Value": fare,
                        "Hour": hour,
                        "Driver_Arrival_Time": driver_eta,
                        "Waiting_Time": waiting_time,
                        "Payment_Method": payment,
                        "Customer_Rating": cust_rating
                    }

                    predictor = RidePredictor(selected_model, preprocessor, thresholds=thresholds)
                    st.session_state.last_prediction = predictor.predict_single(input_dict)
                    st.session_state.last_model_used = selected_model_name

                pred_res = st.session_state.last_prediction
                prob = pred_res["cancellation_probability"]
                status = pred_res["predicted_status"]
                risk = pred_res["risk_category"]

                # Refined Circular Gauge Chart
                gauge_color = THEME["success"] if prob < thresholds["low"] else (THEME["warning"] if prob <= thresholds["medium"] else THEME["danger"])
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=prob * 100.0,
                    number={"suffix": "%", "font": {"size": 34, "color": "#F8FAFC", "family": "Inter, sans-serif"}},
                    title={"text": "Cancellation Probability", "font": {"size": 13, "color": THEME["text_secondary"]}},
                    gauge={
                        "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": THEME["border"]},
                        "bar": {"color": gauge_color, "thickness": 0.28},
                        "bgcolor": "rgba(22, 34, 53, 0.4)",
                        "borderwidth": 1,
                        "bordercolor": THEME["border"],
                        "steps": [
                            {"range": [0, thresholds["low"]*100], "color": "rgba(34, 197, 94, 0.12)"},
                            {"range": [thresholds["low"]*100, thresholds["medium"]*100], "color": "rgba(245, 158, 11, 0.12)"},
                            {"range": [thresholds["medium"]*100, 100], "color": "rgba(239, 68, 68, 0.12)"}
                        ]
                    }
                ))
                fig_gauge.update_layout(height=210, margin=dict(l=15, r=15, t=30, b=15))
                st.plotly_chart(apply_plotly_theme(fig_gauge), use_container_width=True)

                # Prediction Summary Card
                st.markdown(f"""
                <div class="panel-card" style="margin-bottom: 12px; padding: 14px 16px;">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                        <span style="color: var(--text-secondary); font-size: 0.82rem;">Predicted Outcome:</span>
                        <strong style="color: {'#EF4444' if status == 'Cancelled' else '#22C55E'}; font-size: 1rem;">{status}</strong>
                    </div>
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                        <span style="color: var(--text-secondary); font-size: 0.82rem;">Risk Classification:</span>
                        <div>{render_risk_badge(prob, thresholds)}</div>
                    </div>
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <span style="color: var(--text-secondary); font-size: 0.82rem;">Evaluator Model:</span>
                        <span style="color: var(--primary-blue); font-size: 0.82rem; font-family: monospace;">{st.session_state.get('last_model_used', selected_model_name)}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Real SHAP Local Breakdown
                with st.spinner("Calculating SHAP marginal values..."):
                    try:
                        explainer = ShapExplainer(selected_model, st.session_state.X_train_proc)
                        shap_local = explainer.explain_prediction(pred_res["processed_features"], top_k=5)
                        st.plotly_chart(explainer.plot_local_waterfall(shap_local), use_container_width=True)
                    except Exception as e:
                        st.caption(f"SHAP explanation: {str(e)}")

            else:
                # Clean Empty State before prediction
                st.markdown("""
                <div class="empty-state-card" style="margin-top: 20px;">
                    <div class="empty-state-title">Awaiting Parameters</div>
                    <div class="empty-state-desc">
                        Configure dispatch features on the left panel and click <b>Predict Cancellation Risk</b> 
                        to execute real-time model inference and generate SHAP attribution vectors.
                    </div>
                </div>
                """, unsafe_allow_html=True)

    with tab_batch:
        st.markdown("#### Bulk Ride Cancellation Scoring")
        st.caption("Upload a CSV file containing unfulfilled ride requests to generate batch probability scores.")
        batch_file = st.file_uploader("Upload ride bookings CSV", type=["csv"], key="batch_scoring_uploader")

        if batch_file is not None:
            batch_df = pd.read_csv(batch_file)
            st.write(f"Loaded **{len(batch_df):,}** booking records.")

            if st.button("Score Batch Bookings"):
                with st.spinner("Processing batch predictions..."):
                    predictor = RidePredictor(selected_model, preprocessor, thresholds=thresholds)
                    scored_df = predictor.predict_batch(batch_df)

                    st.dataframe(scored_df.head(25), use_container_width=True)

                    csv_data = scored_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Download Scored Results (CSV)",
                        data=csv_data,
                        file_name="rideguard_scored_results.csv",
                        mime="text/csv"
                    )
