"""
RideGuard AI - Page 4: Model Performance & Algorithmic Benchmarking
Evaluates and benchmarks Logistic Regression, Decision Tree, Random Forest, and XGBoost on test holdout.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from models.train_models import ModelTrainer
from models.evaluate import ModelEvaluator
from preprocessing.data_processor import DataProcessor
from utils.ui_components import render_top_navbar, render_empty_state
from utils.helpers import apply_plotly_theme
from config.settings import THEME


def render_model_performance_page():
    if "df" not in st.session_state or st.session_state.df is None:
        render_top_navbar("Model Performance", breadcrumb="Performance", row_count=0)
        render_empty_state("No Dataset Available", "Please navigate to 'Dataset Management' to ingest data.")
        return

    df = st.session_state.df
    target_col = st.session_state.get("target_col", "Booking_Status")
    is_custom = st.session_state.get("is_custom_dataset", False)
    best_name = st.session_state.get("best_model_name", "XGBoost")

    # Top Navbar
    render_top_navbar(
        "Model Performance Benchmark",
        breadcrumb="Model Performance",
        is_custom_data=is_custom,
        row_count=len(df),
        model_name=best_name
    )

    # Retrain Action Header
    top_c1, top_c2 = st.columns([3, 1])
    with top_c1:
        st.caption("Empirical benchmark across 4 classifiers on a stratified 20% holdout test partition.")
    with top_c2:
        retrain = st.button("Retrain All 4 Models", use_container_width=True)

    if retrain or "benchmark_df" not in st.session_state or st.session_state.benchmark_df is None:
        with st.spinner("Training models across 4 algorithms with balanced class weights..."):
            dp = DataProcessor()
            X_train_p, X_test_p, y_train, y_test, _, _ = dp.prepare_train_test(df, target_col=target_col)
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
            st.session_state.best_model_metric = benchmark_df[benchmark_df["Model"] == best_model_name].iloc[0].to_dict()

    benchmark_df = st.session_state.benchmark_df
    evaluator = st.session_state.evaluator
    best_name = st.session_state.best_model_name

    # 1. Best Model Identification Banner
    best_row = benchmark_df[benchmark_df["Model"] == best_name].iloc[0]
    st.markdown(f"""
    <div class="panel-card" style="border-left: 3px solid var(--success); margin-bottom: 20px;">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <span style="font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.05em; color: var(--success); font-weight: 700;">Selected Champion Model</span>
                <div style="font-size: 1.25rem; font-weight: 700; color: #FFFFFF; margin-top: 2px;">{best_name}</div>
                <div style="color: var(--text-secondary); font-size: 0.82rem; margin-top: 4px;">
                    Optimal composite rank balancing F1-Score ({best_row['F1 Score']:.3f}) and ROC-AUC ({best_row['ROC-AUC']:.3f}) under class imbalance.
                </div>
            </div>
            <div style="text-align: right;">
                <div style="font-size: 1.7rem; font-weight: 700; color: var(--success);">{best_row['ROC-AUC']:.3f}</div>
                <div style="font-size: 0.74rem; color: var(--text-muted);">ROC-AUC Index</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 2. Four Model Comparison Cards (Equal Columns)
    st.markdown("#### Classifier Comparison Cards")
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    model_columns = [m_col1, m_col2, m_col3, m_col4]
    model_order = ["Logistic Regression", "Decision Tree", "Random Forest", "XGBoost"]

    for idx, m_name in enumerate(model_order):
        match = benchmark_df[benchmark_df["Model"] == m_name]
        if not match.empty:
            row = match.iloc[0]
            with model_columns[idx]:
                is_champ = (m_name == best_name)
                badge_html = f'<span class="telemetry-pill success" style="padding: 2px 8px; font-size: 0.68rem;">CHAMPION</span>' if is_champ else ''
                st.markdown(f"""
                <div class="metric-card" style="min-height: 140px; {'border-color: rgba(34, 197, 94, 0.4);' if is_champ else ''}">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                        <span style="font-weight: 600; font-size: 0.88rem; color: #FFFFFF;">{m_name}</span>
                        {badge_html}
                    </div>
                    <div style="display: flex; flex-direction: column; gap: 4px; font-size: 0.78rem;">
                        <div style="display: flex; justify-content: space-between;">
                            <span style="color: var(--text-secondary);">Accuracy:</span>
                            <span style="font-family: monospace; color: #FFFFFF;">{row['Accuracy']:.3f}</span>
                        </div>
                        <div style="display: flex; justify-content: space-between;">
                            <span style="color: var(--text-secondary);">Precision:</span>
                            <span style="font-family: monospace; color: #FFFFFF;">{row['Precision']:.3f}</span>
                        </div>
                        <div style="display: flex; justify-content: space-between;">
                            <span style="color: var(--text-secondary);">Recall:</span>
                            <span style="font-family: monospace; color: #FFFFFF;">{row['Recall']:.3f}</span>
                        </div>
                        <div style="display: flex; justify-content: space-between;">
                            <span style="color: var(--text-secondary);">F1-Score:</span>
                            <span style="font-family: monospace; color: var(--primary-blue);">{row['F1 Score']:.3f}</span>
                        </div>
                        <div style="display: flex; justify-content: space-between;">
                            <span style="color: var(--text-secondary);">ROC-AUC:</span>
                            <span style="font-family: monospace; color: var(--cyan-accent);">{row['ROC-AUC']:.3f}</span>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

    # 3. Model Benchmark Bar Chart
    metrics_melt = benchmark_df.melt(
        id_vars=["Model"],
        value_vars=["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"],
        var_name="Metric",
        value_name="Score"
    )

    fig_bar = px.bar(
        metrics_melt,
        x="Metric",
        y="Score",
        color="Model",
        barmode="group",
        color_discrete_sequence=["#3B82F6", "#F59E0B", "#6366F1", "#22C55E"]
    )
    fig_bar.update_layout(
        title="Multi-Metric Cross-Validation Breakdown",
        yaxis=dict(range=[0, 1.05], title="Score"),
        xaxis=dict(title=""),
        height=330
    )
    st.plotly_chart(apply_plotly_theme(fig_bar), use_container_width=True)

    # 4. ROC Curves & Confusion Matrix Inspector
    col_roc, col_cm = st.columns([1.1, 1])

    with col_roc:
        st.markdown("#### Multi-Model ROC Curves")
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode="lines",
            line=dict(dash="dash", color=THEME["text_muted"], width=1.5),
            name="Random Baseline (0.50)"
        ))

        colors = {"Logistic Regression": "#3B82F6", "Decision Tree": "#F59E0B", "Random Forest": "#6366F1", "XGBoost": "#22C55E"}
        for name, curve_data in evaluator.curves.items():
            fig_roc.add_trace(go.Scatter(
                x=curve_data["fpr"],
                y=curve_data["tpr"],
                mode="lines",
                name=f"{name} ({curve_data['auc']:.3f})",
                line=dict(color=colors.get(name, "#22D3EE"), width=2)
            ))

        fig_roc.update_layout(
            title="Receiver Operating Characteristic",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            height=330
        )
        st.plotly_chart(apply_plotly_theme(fig_roc), use_container_width=True)

    with col_cm:
        st.markdown("#### Confusion Matrix")
        inspect_model = st.selectbox("Select Model", list(evaluator.confusion_matrices.keys()), key="cm_inspect_select")
        cm = evaluator.confusion_matrices[inspect_model]

        fig_cm = go.Figure(data=go.Heatmap(
            z=cm,
            x=["Completed (0)", "Cancelled (1)"],
            y=["Completed (0)", "Cancelled (1)"],
            colorscale=[[0, "#111B2B"], [0.5, "#162235"], [1, "#3B82F6"]],
            text=cm,
            texttemplate="%{text}",
            textfont=dict(size=14, color="#FFFFFF")
        ))
        fig_cm.update_layout(
            title=f"Confusion Matrix ({inspect_model})",
            xaxis_title="Predicted Class",
            yaxis_title="Actual Ground Truth",
            height=330
        )
        st.plotly_chart(apply_plotly_theme(fig_cm), use_container_width=True)
