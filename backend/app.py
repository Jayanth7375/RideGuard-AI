"""
RideGuard AI - Main Streamlit Application Entrypoint
Enterprise Mobility Intelligence Platform for Ride Cancellation Prediction and Policy Recommendation.
"""

import streamlit as st
import pandas as pd
from pathlib import Path

# Streamlit Page Config
st.set_page_config(
    page_title="RIDEGUARD AI | Predictive Mobility Intelligence",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

from config.settings import DEFAULT_DATASET_PATH, ARTIFACTS_DIR
from utils.helpers import load_custom_css, generate_synthetic_rides
from utils.ui_components import ICONS
from models.train_models import ModelTrainer
from models.evaluate import ModelEvaluator
from preprocessing.data_processor import DataProcessor

# Import Redesigned Dashboard Pages
from dashboard.overview import render_overview_page
from dashboard.analytics import render_analytics_page
from dashboard.prediction import render_prediction_page
from dashboard.model_performance import render_model_performance_page
from dashboard.explainable_ai import render_explainable_ai_page
from dashboard.recommendations import render_recommendations_page
from dashboard.dataset_management import render_dataset_management_page


def initialize_session_state():
    """Initializes dataset, preprocessor, and models in Streamlit session state."""
    # 1. Dataset Initialization
    if "df" not in st.session_state or st.session_state.df is None:
        if DEFAULT_DATASET_PATH.exists():
            st.session_state.df = pd.read_csv(DEFAULT_DATASET_PATH)
            st.session_state.is_custom_dataset = False
            st.session_state.uploaded_file_name = "Benchmark Dataset"
        else:
            st.session_state.df = generate_synthetic_rides(n_samples=3000)
            st.session_state.is_custom_dataset = False
            st.session_state.uploaded_file_name = "Benchmark Dataset"

    if "target_col" not in st.session_state:
        st.session_state.target_col = "Booking_Status"

    # 2. Check for Serialized Artifacts or Train on First Run
    if "trained_models" not in st.session_state or st.session_state.trained_models is None:
        preprocessor_path = ARTIFACTS_DIR / "preprocessor.joblib"
        loaded_models = ModelTrainer.load_models(ARTIFACTS_DIR)

        if preprocessor_path.exists() and len(loaded_models) == 4:
            st.session_state.preprocessor = DataProcessor.load(preprocessor_path)
            st.session_state.trained_models = loaded_models
            dp = st.session_state.preprocessor
            X_train_p, X_test_p, y_train, y_test, _, _ = dp.prepare_train_test(
                st.session_state.df, target_col=st.session_state.target_col
            )
            evaluator = ModelEvaluator(loaded_models, X_test_p, y_test)
            st.session_state.benchmark_df = evaluator.evaluate_all()
            st.session_state.evaluator = evaluator
            st.session_state.best_model_name = evaluator.get_best_model_name()
            st.session_state.X_train_proc = X_train_p
            st.session_state.X_test_proc = X_test_p
            st.session_state.y_test = y_test
            st.session_state.best_model_metric = st.session_state.benchmark_df[
                st.session_state.benchmark_df["Model"] == st.session_state.best_model_name
            ].iloc[0].to_dict()
        else:
            dp = DataProcessor()
            X_train_p, X_test_p, y_train, y_test, _, _ = dp.prepare_train_test(
                st.session_state.df, target_col=st.session_state.target_col
            )
            dp.save()

            trainer = ModelTrainer()
            trained = trainer.train_all(X_train_p, y_train)
            trainer.save_models()

            evaluator = ModelEvaluator(trained, X_test_p, y_test)
            benchmark_df = evaluator.evaluate_all()
            best_model_name = evaluator.get_best_model_name()

            st.session_state.preprocessor = dp
            st.session_state.trained_models = trained
            st.session_state.benchmark_df = benchmark_df
            st.session_state.evaluator = evaluator
            st.session_state.best_model_name = best_model_name
            st.session_state.X_train_proc = X_train_p
            st.session_state.X_test_proc = X_test_p
            st.session_state.y_test = y_test
            st.session_state.best_model_metric = benchmark_df[
                benchmark_df["Model"] == best_model_name
            ].iloc[0].to_dict()


def main():
    # Load Enterprise Custom CSS
    load_custom_css()

    # Initialize State
    initialize_session_state()

    # Sidebar Navigation
    with st.sidebar:
        # Elegant Header Branding with SVG Shield Logo
        st.markdown(f"""
        <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 22px; padding: 2px 0;">
            <div style="background: rgba(59, 130, 246, 0.12); width: 36px; height: 36px; border-radius: 8px; border: 1px solid rgba(59, 130, 246, 0.25); display: flex; align-items: center; justify-content: center;">
                {ICONS['shield']}
            </div>
            <div>
                <div style="font-weight: 700; font-size: 1.05rem; color: #F8FAFC; letter-spacing: -0.02em;">RIDEGUARD AI</div>
                <div style="font-size: 0.70rem; color: #64748B; font-weight: 500;">Predictive Mobility Intelligence</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        pages = {
            "Executive Overview": "Executive Overview",
            "Cancellation Analytics": "Cancellation Analytics",
            "AI Prediction": "AI Prediction",
            "Model Performance": "Model Performance",
            "Explainable AI": "Explainable AI",
            "Policy Recommendations": "Policy Recommendations",
            "Dataset Management": "Dataset Management",
        }

        selected_page = st.radio(
            "PLATFORM MODULES",
            list(pages.keys()),
            format_func=lambda x: pages[x],
            label_visibility="visible"
        )

        st.markdown("<div style='height: 24px;'></div>", unsafe_allow_html=True)

        # System Telemetry Panel at Bottom
        active_rows = len(st.session_state.df) if st.session_state.df is not None else 0
        best_name = st.session_state.get("best_model_name", "Ready")
        is_custom = st.session_state.get("is_custom_dataset", False)
        data_label = "Custom Upload" if is_custom else "Benchmark Data"

        st.markdown(f"""
        <div style="background: #111B2B; border: 1px solid #243247; border-radius: 6px; padding: 12px 14px; font-size: 0.74rem;">
            <div style="color: #64748B; font-weight: 600; text-transform: uppercase; letter-spacing: 0.04em; margin-bottom: 8px;">SYSTEM TELEMETRY</div>
            <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                <span style="color: #94A3B8;">Dataset Source:</span>
                <span style="color: #F8FAFC; font-weight: 500;">{data_label}</span>
            </div>
            <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                <span style="color: #94A3B8;">Sample Size:</span>
                <span style="color: #22D3EE; font-family: monospace;">{active_rows:,} rides</span>
            </div>
            <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                <span style="color: #94A3B8;">Champion Model:</span>
                <span style="color: #22C55E; font-weight: 600;">{best_name}</span>
            </div>
            <div style="display: flex; justify-content: space-between;">
                <span style="color: #94A3B8;">SHAP Explainer:</span>
                <span style="color: #3B82F6; font-weight: 500;">Online</span>
            </div>
        </div>
        <div style="margin-top: 18px; font-size: 0.70rem; color: #475569; text-align: center;">
            RideGuard AI &bull; Enterprise Release v1.2
        </div>
        """, unsafe_allow_html=True)

    # Render Active Page Module
    if selected_page == "Executive Overview":
        render_overview_page()
    elif selected_page == "Cancellation Analytics":
        render_analytics_page()
    elif selected_page == "AI Prediction":
        render_prediction_page()
    elif selected_page == "Model Performance":
        render_model_performance_page()
    elif selected_page == "Explainable AI":
        render_explainable_ai_page()
    elif selected_page == "Policy Recommendations":
        render_recommendations_page()
    elif selected_page == "Dataset Management":
        render_dataset_management_page()


if __name__ == "__main__":
    main()
