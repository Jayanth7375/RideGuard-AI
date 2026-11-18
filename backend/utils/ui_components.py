"""
RideGuard AI - Enterprise UI Component System
Provides reusable, brand-consistent SVG icons, top navigations, KPI cards, badges, and empty states.
"""

from typing import Dict, Any, Optional
import streamlit as st
from datetime import datetime

# Clean Lucide-Style SVG Icons (16x16 / 20x20)
ICONS = {
    "shield": """<svg xmlns="http://www.w3.org/2000/svg" width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#3B82F6" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/></svg>""",
    "overview": """<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="3" width="7" height="9"/><rect x="14" y="3" width="7" height="5"/><rect x="14" y="12" width="7" height="9"/><rect x="3" y="16" width="7" height="5"/></svg>""",
    "analytics": """<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="18" y1="20" x2="18" y2="10"/><line x1="12" y1="20" x2="12" y2="4"/><line x1="6" y1="20" x2="6" y2="14"/></svg>""",
    "prediction": """<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2a4 4 0 0 1 4 4v2a4 4 0 0 1-8 0V6a4 4 0 0 1 4-4z"/><rect x="4" y="12" width="16" height="10" rx="2"/><circle cx="12" cy="17" r="1.5"/></svg>""",
    "performance": """<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg>""",
    "explainable": """<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"/><circle cx="12" cy="12" r="3"/></svg>""",
    "recommendations": """<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/></svg>""",
    "dataset": """<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3"/><path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5"/></svg>""",
    "rides": """<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#94A3B8" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M19 17h2c.6 0 1-.4 1-1v-3c0-.9-.7-1.7-1.5-1.9C18.7 10.6 16 10 16 10s-1.3-1.4-2.2-2.3c-.5-.4-1.1-.7-1.8-.7H5c-.6 0-1.1.4-1.4.9l-1.4 2.9A3.7 3.7 0 0 0 2 12v4c0 .6.4 1 1 1h2"/><circle cx="7" cy="17" r="2"/><circle cx="17" cy="17" r="2"/></svg>""",
    "check": """<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#22C55E" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>""",
    "cross": """<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#EF4444" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><line x1="15" y1="9" x2="9" y2="15"/><line x1="9" y1="9" x2="15" y2="15"/></svg>""",
    "percent": """<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#F59E0B" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="19" y1="5" x2="5" y2="19"/><circle cx="6.5" cy="6.5" r="2.5"/><circle cx="17.5" cy="17.5" r="2.5"/></svg>""",
    "clock": """<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#22D3EE" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg>""",
    "database": """<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#94A3B8" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3"/><path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5"/></svg>""",
    "cpu": """<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#3B82F6" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="4" y="4" width="16" height="16" rx="2"/><rect x="9" y="9" width="6" height="6"/><line x1="9" y1="1" x2="9" y2="4"/><line x1="15" y1="1" x2="15" y2="4"/><line x1="9" y1="20" x2="9" y2="23"/><line x1="15" y1="20" x2="15" y2="23"/><line x1="20" y1="9" x2="23" y2="9"/><line x1="20" y1="14" x2="23" y2="14"/><line x1="1" y1="9" x2="4" y2="9"/><line x1="1" y1="14" x2="4" y2="14"/></svg>""",
}


def render_top_navbar(page_title: str, breadcrumb: str = "Overview", is_custom_data: bool = False, row_count: int = 0, model_name: str = "XGBoost"):
    """
    Renders an enterprise-grade top navigation header with breadcrumbs,
    telemetry status pills, and system time.
    """
    data_badge = f'<span class="telemetry-pill success"><span class="dot live"></span> Custom: {row_count:,} rides</span>' if is_custom_data else f'<span class="telemetry-pill neutral"><span class="dot neutral"></span> Benchmark: {row_count:,} rides</span>'
    model_badge = f'<span class="telemetry-pill info">{ICONS["cpu"]} {model_name}</span>'

    html = f"""
    <div class="top-navbar-container">
        <div class="top-nav-left">
            <div class="breadcrumb">
                <span class="crumb-root">RideGuard AI</span>
                <span class="crumb-sep">/</span>
                <span class="crumb-current">{breadcrumb}</span>
            </div>
            <div class="page-headline">{page_title}</div>
        </div>
        <div class="top-nav-right">
            {data_badge}
            {model_badge}
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_kpi_card(title: str, value: str, subtitle: str = "", icon_key: str = "rides", variant: str = "neutral"):
    """
    Renders an elegant, compact metric card following modern SaaS design systems.
    """
    icon_svg = ICONS.get(icon_key, ICONS["rides"])
    card_html = f"""
    <div class="metric-card variant-{variant}">
        <div class="metric-header">
            <span class="metric-title">{title}</span>
            <div class="metric-icon-box">{icon_svg}</div>
        </div>
        <div class="metric-body">
            <div class="metric-value">{value}</div>
            {f'<div class="metric-subtitle">{subtitle}</div>' if subtitle else ''}
        </div>
    </div>
    """
    st.markdown(card_html, unsafe_allow_html=True)


def render_risk_badge(probability: float, thresholds: Dict[str, float]) -> str:
    """
    Returns an inline status badge with a live colored indicator dot.
    """
    low = thresholds.get("low", 0.30)
    medium = thresholds.get("medium", 0.60)

    if probability < low:
        return f'<span class="status-badge badge-low"><span class="dot green"></span>Low Risk ({probability*100:.1f}%)</span>'
    elif probability <= medium:
        return f'<span class="status-badge badge-medium"><span class="dot yellow"></span>Medium Risk ({probability*100:.1f}%)</span>'
    else:
        return f'<span class="status-badge badge-high"><span class="dot red"></span>High Risk ({probability*100:.1f}%)</span>'


def render_empty_state(title: str, message: str, action_text: str = None):
    """
    Renders a sophisticated empty state card with guidance.
    """
    html = f"""
    <div class="empty-state-card">
        <div class="empty-state-icon">{ICONS["dataset"]}</div>
        <div class="empty-state-title">{title}</div>
        <div class="empty-state-desc">{message}</div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)
