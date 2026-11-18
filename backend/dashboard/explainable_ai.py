"""
RideGuard AI - Page 5: Explainable AI (SHAP)
Enterprise interpretability suite with global importance, single ride forensics, and executive factor narratives.
"""

import streamlit as st
import pandas as pd
import numpy as np

from explainability.shap_explainer import ShapExplainer
from utils.ui_components import render_top_navbar, render_empty_state
from utils.helpers import apply_plotly_theme


def render_explainable_ai_page():
    if ("df" not in st.session_state or
            "trained_models" not in st.session_state or
            "X_train_proc" not in st.session_state or
            st.session_state.trained_models is None):
        render_top_navbar("Explainable AI", breadcrumb="Explainable AI", row_count=0)
        render_empty_state(
            "Models Not Yet Initialized",
            "Please train or initialize machine learning models in 'Model Performance' before executing SHAP game-theoretic attribution analysis."
        )
        return

    models = st.session_state.trained_models
    X_train_proc = st.session_state.X_train_proc
    X_test_proc = st.session_state.X_test_proc
    best_name = st.session_state.get("best_model_name", "XGBoost")
    is_custom = st.session_state.get("is_custom_dataset", False)

    # Top Navbar
    render_top_navbar(
        "Explainable AI (SHAP)",
        breadcrumb="Explainability",
        is_custom_data=is_custom,
        row_count=len(st.session_state.df),
        model_name=best_name
    )

    # Model and Sampling Controls Bar
    c_col1, c_col2 = st.columns([1.5, 1])
    with c_col1:
        selected_model_name = st.selectbox(
            "Select Classifier for SHAP Attribution",
            list(models.keys()),
            index=list(models.keys()).index(best_name) if best_name in models else 0
        )
        model = models[selected_model_name]
    with c_col2:
        sample_size = st.slider("Attribution Test Sample Size", 40, min(250, len(X_test_proc)), 100)

    st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)

    with st.spinner(f"Computing Shapley Additive Explanations for {selected_model_name}..."):
        try:
            explainer = ShapExplainer(model, X_train_proc)
            X_sample = X_test_proc.sample(sample_size, random_state=42)

            tab_global, tab_local = st.tabs(["Global Feature Importance", "Individual Trip Attribution"])

            with tab_global:
                # Executive Narrative Box
                st.markdown("""
                <div class="panel-card" style="margin-bottom: 16px;">
                    <div class="panel-header">
                        <span class="panel-title">Why Does the Model Predict Cancellation?</span>
                        <span class="panel-desc">Executive Explanatory Rationale</span>
                    </div>
                    <p style="color: var(--text-secondary); font-size: 0.85rem; line-height: 1.6; margin: 0;">
                        Shapley Additive Explanations allocate mathematical credit to each feature vector based on cooperative game theory. 
                        Features positioned higher on the importance scale possess the strongest marginal capacity to tip an otherwise fulfilled ride 
                        into a cancellation. In empirical dispatch records, <b>Driver Arrival Time (ETA)</b> and <b>Pickup Waiting Delays</b> 
                        act as the dominant cancellation drivers, followed by dynamic surge price differentials.
                    </p>
                </div>
                """, unsafe_allow_html=True)

                col_chart, col_table = st.columns([1.4, 1])
                with col_chart:
                    st.plotly_chart(explainer.plot_global_bar(X_sample, max_display=10), use_container_width=True)

                with col_table:
                    df_imp = explainer.compute_global_importance(X_sample, max_display=10)
                    st.markdown("##### Leading Feature Attributions")
                    st.dataframe(
                        df_imp.rename(columns={"Feature": "Dispatch Feature", "Importance": "Mean |SHAP|"}),
                        use_container_width=True
                    )

            with tab_local:
                st.markdown("#### Test Partition Trip Forensics")
                st.caption("Select an individual test trip to inspect exact positive and negative SHAP marginal contributions.")

                test_idx = st.slider("Select Record Index from Sample Partition", 0, len(X_sample) - 1, 0)
                selected_ride = X_sample.iloc[[test_idx]]

                local_exp = explainer.explain_prediction(selected_ride, top_k=6)

                st.plotly_chart(explainer.plot_local_waterfall(local_exp), use_container_width=True)

                st.markdown("##### Feature Impact Directional Matrix")
                st.dataframe(
                    local_exp["all_factors"].head(8)[["feature", "shap_value", "impact"]]
                    .rename(columns={"feature": "Feature", "shap_value": "SHAP Weight", "impact": "Effect on Risk"}),
                    use_container_width=True
                )

        except Exception as e:
            st.error(f"Error computing SHAP values: {str(e)}")
