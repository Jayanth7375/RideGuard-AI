"""
RideGuard AI - Page 7: Dataset Management & Hygiene Auditing
Ingest custom CSV datasets, validate schema, audit missing values/duplicates, and map target variables.
"""

import streamlit as st
import pandas as pd
import numpy as np

from utils.ui_components import render_top_navbar, render_kpi_card, render_empty_state
from utils.helpers import generate_synthetic_rides
from config.settings import DEFAULT_DATASET_PATH, TARGET_COLUMN_CANDIDATES


def render_dataset_management_page():
    df = st.session_state.get("df", None)
    is_custom = st.session_state.get("is_custom_dataset", False)
    best_name = st.session_state.get("best_model_name", "XGBoost")

    # Top Navbar
    render_top_navbar(
        "Dataset Management & Schema Audit",
        breadcrumb="Dataset Management",
        is_custom_data=is_custom,
        row_count=len(df) if df is not None else 0,
        model_name=best_name
    )

    tab_upload, tab_synthetic, tab_schema = st.tabs(["Ingest Custom CSV", "Generate Benchmark Data", "Schema & Hygiene Audit"])

    # TAB 1: CSV Upload
    with tab_upload:
        st.markdown("#### Upload Production Ride Records")
        st.caption("Upload raw CSV booking logs from your fleet database. The schema validator will automatically detect columns.")

        uploaded_file = st.file_uploader("Select or drop CSV file", type=["csv"], key="custom_csv_uploader")
        if uploaded_file is not None:
            try:
                user_df = pd.read_csv(uploaded_file)
                st.session_state.df = user_df
                st.session_state.is_custom_dataset = True
                st.session_state.uploaded_file_name = uploaded_file.name
                # Reset models to trigger fresh training on new dataset
                st.session_state.trained_models = None
                st.session_state.benchmark_df = None
                st.success(f"Loaded '{uploaded_file.name}' successfully: {len(user_df):,} rows and {len(user_df.columns)} columns.")
                st.rerun()
            except Exception as e:
                st.error(f"Error parsing CSV file: {str(e)}")

    # TAB 2: Benchmark Ride Generator
    with tab_synthetic:
        st.markdown("#### Synthesize High-Fidelity Benchmark Data")
        st.caption("Generate a statistically realistic ride-hailing dataset modeling actual Uber and Ola cancellation dynamics.")

        gen_c1, gen_c2 = st.columns([1, 2])
        with gen_c1:
            n_samples = st.slider("Record Volume", 1000, 10000, 3000, step=500)
            seed = st.number_input("Reproducibility Seed", value=42, step=1)
            gen_btn = st.button("Generate Benchmark Data", use_container_width=True)

        if gen_btn:
            with st.spinner("Generating benchmark mobility data..."):
                syn_df = generate_synthetic_rides(n_samples=n_samples, random_seed=seed)
                st.session_state.df = syn_df
                st.session_state.is_custom_dataset = False
                st.session_state.uploaded_file_name = "Benchmark Synthetic Dataset"
                st.session_state.trained_models = None
                st.session_state.benchmark_df = None
                st.success(f"Generated {len(syn_df):,} benchmark records.")
                st.rerun()

    # Check active dataset
    if "df" not in st.session_state or st.session_state.df is None:
        render_empty_state("No Active Dataset", "Upload a CSV dataset or generate benchmark data to continue.")
        return

    df = st.session_state.df

    # Target Variable Mapping Section
    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
    st.markdown("#### Target Classification Variable")

    detected_target = None
    for cand in TARGET_COLUMN_CANDIDATES:
        match = [c for c in df.columns if c.lower().strip() == cand]
        if match:
            detected_target = match[0]
            break

    target_c1, target_c2 = st.columns([1.5, 1])
    with target_c1:
        target_options = df.columns.tolist()
        default_idx = target_options.index(detected_target) if detected_target in target_options else 0
        target_col = st.selectbox(
            "Target Column (Cancellation Indicator)",
            target_options,
            index=default_idx,
            help="Select the column representing whether the booking completed or was cancelled."
        )
        st.session_state.target_col = target_col

    with target_c2:
        unique_targets = df[target_col].dropna().unique().tolist()
        st.markdown(f"""
        <div class="panel-card" style="padding: 10px 14px; margin-top: 24px;">
            <div style="font-size: 0.75rem; color: var(--text-muted); text-transform: uppercase;">Observed Target Values</div>
            <div style="font-size: 0.88rem; color: var(--primary-blue); font-family: monospace; margin-top: 2px;">{unique_targets}</div>
        </div>
        """, unsafe_allow_html=True)

    # TAB 3: Schema & Hygiene Audit
    with tab_schema:
        st.markdown("#### Dataset Health Metrics")

        # 4 Hygiene Metric Cards
        h1, h2, h3, h4 = st.columns(4)
        with h1:
            render_kpi_card("Total Rows", f"{len(df):,}", "Active record count", icon_key="rides")
        with h2:
            render_kpi_card("Feature Columns", f"{len(df.columns)}", "Available attributes", icon_key="database")
        with h3:
            dup_count = int(df.duplicated().sum())
            render_kpi_card("Duplicates", f"{dup_count:,}", "Duplicate row count", icon_key="check" if dup_count == 0 else "cross")
        with h4:
            null_count = int(df.isnull().sum().sum())
            render_kpi_card("Missing Cells", f"{null_count:,}", "Missing value count", icon_key="check" if null_count == 0 else "cross")

        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

        # Feature Schema Breakdown
        st.markdown("##### Detailed Column Schema")
        schema_rows = []
        for col in df.columns:
            null_pct = (df[col].isnull().sum() / len(df)) * 100
            schema_rows.append({
                "Column": col,
                "Type": str(df[col].dtype),
                "Missing Count": int(df[col].isnull().sum()),
                "Missing (%)": f"{null_pct:.1f}%",
                "Unique Values": int(df[col].nunique()),
                "Sample": str(df[col].dropna().iloc[0]) if not df[col].dropna().empty else "None"
            })
        st.dataframe(pd.DataFrame(schema_rows), use_container_width=True)

        # Data Preview
        st.markdown("##### Sample Records Preview (First 15 Rows)")
        st.dataframe(df.head(15), use_container_width=True)

        # Statistical Summary
        st.markdown("##### Numerical Distribution Summary")
        st.dataframe(df.describe().T, use_container_width=True)
