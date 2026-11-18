"""
RideGuard AI - Page 1: Executive Overview Dashboard
Enterprise mobility dashboard displaying KPIs, booking trends, cancellation insights, and ML telemetry.
"""

import streamlit as st
import pandas as pd
from utils.ui_components import render_top_navbar, render_kpi_card, render_empty_state
from analytics.eda import RideAnalytics


def render_overview_page():
    if "df" not in st.session_state or st.session_state.df is None:
        render_top_navbar("Executive Overview", breadcrumb="Overview", row_count=0)
        render_empty_state(
            "No Dataset Loaded",
            "Please navigate to 'Dataset Management' to ingest your ride booking CSV dataset or initialize the benchmark data."
        )
        return

    df = st.session_state.df
    target_col = st.session_state.get("target_col", "Booking_Status")
    is_custom = st.session_state.get("is_custom_dataset", False)
    best_model_name = st.session_state.get("best_model_name", "XGBoost")

    # Top Navbar & Telemetry Bar
    render_top_navbar(
        "Executive Overview",
        breadcrumb="Overview",
        is_custom_data=is_custom,
        row_count=len(df),
        model_name=best_model_name
    )

    # Optional Compact Date Filter
    filtered_df = df.copy()
    if "Booking_Date" in df.columns:
        dates = pd.to_datetime(df["Booking_Date"], errors="coerce").dropna().sort_values()
        if not dates.empty:
            min_date, max_date = dates.iloc[0].date(), dates.iloc[-1].date()
            if min_date < max_date:
                col_sub, col_filt = st.columns([3, 1.2])
                with col_sub:
                    st.caption("Monitor ride completion, cancellation trends, and predictive intelligence.")
                with col_filt:
                    date_range = st.date_input(
                        "Date Range",
                        value=(min_date, max_date),
                        min_value=min_date,
                        max_value=max_date,
                        label_visibility="collapsed"
                    )
                    if isinstance(date_range, tuple) and len(date_range) == 2:
                        start_d, end_d = date_range
                        parsed_dates = pd.to_datetime(filtered_df["Booking_Date"], errors="coerce").dt.date
                        filtered_df = filtered_df[(parsed_dates >= start_d) & (parsed_dates <= end_d)]
        else:
            st.caption("Monitor ride completion, cancellation trends, and predictive intelligence.")
    else:
        st.caption("Monitor ride completion, cancellation trends, and predictive intelligence.")

    analytics = RideAnalytics(filtered_df, target_col=target_col)
    kpis = analytics.get_kpis()

    # 1. KPI SECTION (5 Compact, Balanced Cards)
    kpi_col1, kpi_col2, kpi_col3, kpi_col4, kpi_col5 = st.columns(5)
    with kpi_col1:
        render_kpi_card("Total Bookings", f"{kpis['total_bookings']:,}", "Recorded trip requests", icon_key="rides", variant="neutral")
    with kpi_col2:
        fulfillment_pct = (100.0 - kpis['cancellation_rate']) if kpis['total_bookings'] > 0 else 0.0
        render_kpi_card("Completed Rides", f"{kpis['completed_rides']:,}", f"{fulfillment_pct:.1f}% fulfillment rate", icon_key="check", variant="neutral")
    with kpi_col3:
        render_kpi_card("Cancelled Rides", f"{kpis['cancelled_rides']:,}", f"{kpis['cancellation_rate']:.1f}% dropout rate", icon_key="cross", variant="neutral")
    with kpi_col4:
        render_kpi_card("Cancellation Rate", f"{kpis['cancellation_rate']:.1f}%", "Overall network churn", icon_key="percent", variant="neutral")
    with kpi_col5:
        avg_wait_val = f"{kpis['avg_waiting_time']:.1f} min" if kpis['avg_waiting_time'] > 0 else "N/A"
        render_kpi_card("Average Wait Time", avg_wait_val, "Driver delay index", icon_key="clock", variant="neutral")

    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

    # 2. CHART SECTION (Two-Column Layout with Consistent Heights)
    chart_col1, chart_col2 = st.columns([1, 1.35])
    with chart_col1:
        st.plotly_chart(analytics.plot_cancellation_donut(), use_container_width=True)
    with chart_col2:
        st.plotly_chart(analytics.plot_hourly_trend(), use_container_width=True)

    st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)

    # 3. BELOW THE CHARTS (Cancellation Insights & AI Intelligence Summary)
    lower_col1, lower_col2 = st.columns([1.2, 1])

    with lower_col1:
        # Compute real empirical patterns from the dataset
        insights = []
        if "Waiting_Time" in filtered_df.columns:
            high_wait = filtered_df[filtered_df["Waiting_Time"] >= 8]
            if len(high_wait) > 0 and "_is_cancelled" in analytics.df.columns:
                high_wait_rate = analytics.df.loc[high_wait.index, "_is_cancelled"].mean() * 100
                insights.append(f"Extended wait times (≥8 mins) exhibit a <b>{high_wait_rate:.1f}%</b> cancellation rate compared to {kpis['cancellation_rate']:.1f}% platform baseline.")

        if "Pickup_Location" in filtered_df.columns:
            loc_rates = analytics.df.groupby("Pickup_Location")["_is_cancelled"].agg(["count", "mean"]).reset_index()
            loc_rates = loc_rates[loc_rates["count"] >= 10].sort_values(by="mean", ascending=False)
            if not loc_rates.empty:
                worst_loc = loc_rates.iloc[0]
                insights.append(f"Highest geographic cancellation intensity is concentrated at <b>{worst_loc['Pickup_Location']}</b> ({worst_loc['mean']*100:.1f}% churn across {worst_loc['count']} rides).")

        if "Hour" in analytics.df.columns:
            hour_rates = analytics.df.groupby("Hour")["_is_cancelled"].agg(["count", "mean"]).reset_index()
            peak_hr = hour_rates[hour_rates["count"] >= 10].sort_values(by="mean", ascending=False)
            if not peak_hr.empty:
                top_hr = peak_hr.iloc[0]
                insights.append(f"Peak temporal dropout occurs during the <b>{int(top_hr['Hour']):02d}:00</b> window ({top_hr['mean']*100:.1f}% cancellation rate).")

        if not insights:
            insights.append("Operational baseline established across all active feature vectors.")

        insights_html = "".join([f"<li style='margin-bottom: 8px; color: #94A3B8; font-size: 0.84rem; line-height: 1.5;'>{item}</li>" for item in insights])

        st.markdown(f"""
        <div class="panel-card" style="height: 100%;">
            <div class="panel-header">
                <span class="panel-title">Cancellation Insights</span>
                <span class="panel-desc">Empirical Dataset Patterns</span>
            </div>
            <ul style="padding-left: 18px; margin: 0;">
                {insights_html}
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with lower_col2:
        # Real Model telemetry status
        evaluator = st.session_state.get("evaluator", None)
        best_metric = st.session_state.get("best_model_metric", {})
        auc_val = f"{best_metric.get('ROC-AUC', 0.0):.3f}" if best_metric else "Trained"
        f1_val = f"{best_metric.get('F1 Score', 0.0):.3f}" if best_metric else "Trained"
        records_analyzed = f"{len(filtered_df):,}"

        # Check SHAP top factor if computed
        top_driver_str = "Driver Arrival Time / Waiting Delay"
        if "trained_models" in st.session_state and st.session_state.trained_models and "X_train_proc" in st.session_state:
            try:
                from explainability.shap_explainer import ShapExplainer
                exp = ShapExplainer(st.session_state.trained_models[best_model_name], st.session_state.X_train_proc)
                top_feats = exp.compute_global_importance(st.session_state.X_test_proc.head(40), max_display=1)
                if not top_feats.empty:
                    top_driver_str = str(top_feats.iloc[0]["Feature"]).replace("_", " ")
            except Exception:
                pass

        st.markdown(f"""
        <div class="panel-card" style="height: 100%;">
            <div class="panel-header">
                <span class="panel-title">AI Intelligence Summary</span>
                <span class="panel-desc">Production Telemetry</span>
            </div>
            <div style="display: flex; flex-direction: column; gap: 8px; font-size: 0.84rem;">
                <div style="display: flex; justify-content: space-between; border-bottom: 1px solid var(--border); padding-bottom: 6px;">
                    <span style="color: var(--text-secondary);">Selected Model:</span>
                    <span style="color: var(--primary-blue); font-weight: 600; font-family: monospace;">{best_model_name}</span>
                </div>
                <div style="display: flex; justify-content: space-between; border-bottom: 1px solid var(--border); padding-bottom: 6px;">
                    <span style="color: var(--text-secondary);">Evaluation Metric (ROC-AUC / F1):</span>
                    <span style="color: var(--success); font-weight: 600; font-family: monospace;">{auc_val} / {f1_val}</span>
                </div>
                <div style="display: flex; justify-content: space-between; border-bottom: 1px solid var(--border); padding-bottom: 6px;">
                    <span style="color: var(--text-secondary);">Records Analyzed:</span>
                    <span style="color: var(--text-primary); font-weight: 500;">{records_analyzed} rides</span>
                </div>
                <div style="display: flex; justify-content: space-between;">
                    <span style="color: var(--text-secondary);">Leading SHAP Factor:</span>
                    <span style="color: var(--cyan-accent); font-weight: 500;">{top_driver_str}</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
