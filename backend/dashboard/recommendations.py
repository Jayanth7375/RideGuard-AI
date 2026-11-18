"""
RideGuard AI - Page 6: Intelligent Policy Recommendation Engine
Enterprise policy advisory derived from empirical delay cohorts, fleet friction, and geospatial imbalances.
"""

import streamlit as st
import pandas as pd
from recommendations.recommendation_engine import PolicyRecommendationEngine
from utils.ui_components import render_top_navbar, render_empty_state


def render_recommendations_page():
    if "df" not in st.session_state or st.session_state.df is None:
        render_top_navbar("Policy Recommendations", breadcrumb="Policy Engine", row_count=0)
        render_empty_state("No Dataset Available", "Please navigate to 'Dataset Management' to ingest data.")
        return

    df = st.session_state.df
    target_col = st.session_state.get("target_col", "Booking_Status")
    is_custom = st.session_state.get("is_custom_dataset", False)
    best_name = st.session_state.get("best_model_name", "XGBoost")

    # Top Navbar
    render_top_navbar(
        "Intelligent Policy Recommendations",
        breadcrumb="Policy Recommendations",
        is_custom_data=is_custom,
        row_count=len(df),
        model_name=best_name
    )

    engine = PolicyRecommendationEngine(df, target_col=target_col)
    recommendations = engine.generate_recommendations()

    # Priority Tier Filter
    st.markdown("#### Operational Initiatives")
    priority_filter = st.radio(
        "Priority Filter",
        ["All Priorities", "Critical", "High", "Medium", "Low"],
        horizontal=True,
        label_visibility="collapsed"
    )

    filtered_recs = [
        r for r in recommendations
        if priority_filter == "All Priorities" or r.get("priority", "").lower() == priority_filter.lower()
    ]

    if not filtered_recs:
        st.info(f"No active recommendations under '{priority_filter}' priority tier.")
        return

    # Render Priority Cards with Clean Semantic Styling
    for rec in filtered_recs:
        priority = rec.get("priority", "Medium")
        border_class = {
            "Critical": "priority-critical",
            "High": "priority-high",
            "Medium": "priority-medium",
            "Low": "priority-low"
        }.get(priority, "priority-medium")

        badge_class = {
            "Critical": "badge-high",
            "High": "badge-high",
            "Medium": "badge-medium",
            "Low": "badge-low"
        }.get(priority, "badge-medium")

        st.markdown(f"""
        <div class="rec-card {border_class}">
            <div class="rec-meta">
                <span class="rec-tag">{rec['category']} &bull; {rec['id']}</span>
                <span class="status-badge {badge_class}">{priority} Priority</span>
            </div>
            
            <div class="rec-title">{rec['title']}</div>
            
            <div class="rec-evidence-box">
                <div style="margin-bottom: 4px;">
                    <strong style="color: var(--text-primary);">Identified Pattern:</strong> {rec['pattern']}
                </div>
                <div>
                    <strong style="color: var(--text-primary);">Supporting Evidence:</strong> {rec['evidence']}
                </div>
            </div>

            <div class="rec-action">
                <strong style="color: var(--primary-blue);">Recommended Action:</strong> {rec['recommended_action']}
            </div>

            <div class="rec-footer">
                <span style="color: var(--success); font-weight: 500;"><strong>Potential Benefit:</strong> {rec['expected_benefit']}</span>
                <span style="color: var(--text-muted); font-style: italic;">{rec['causal_disclaimer']}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
