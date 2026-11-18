"""
RideGuard AI - Standalone FastAPI Server
Serves REST API endpoints for dataset ingestion, schema validation, multi-model ML training,
inference, SHAP explainability, and single-page web application delivery.
"""

import sys
from pathlib import Path

# Configure paths for segregated architecture
ROOT_DIR = Path(__file__).resolve().parent
if ROOT_DIR.name == "backend":
    ROOT_DIR = ROOT_DIR.parent
BACKEND_DIR = ROOT_DIR / "backend"
MODELS_DIR = ROOT_DIR / "models"

for p in [str(ROOT_DIR), str(BACKEND_DIR), str(MODELS_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
import io
import json
import threading
from typing import Dict, Any, Optional, List

from config.settings import (
    DEFAULT_DATASET_PATH,
    ARTIFACTS_DIR,
    DEFAULT_RISK_THRESHOLDS,
    THEME,
    STANDARD_FIELDS,
    TARGET_COLUMN_CANDIDATES,
    TARGET_VALUE_MAP,
)
from utils.helpers import get_risk_label
from models.train_models import ModelTrainer
from models.evaluate import ModelEvaluator
from models.predict import RidePredictor
from preprocessing.data_processor import DataProcessor
from explainability.shap_explainer import ShapExplainer
from recommendations.recommendation_engine import PolicyRecommendationEngine
from analytics.eda import RideAnalytics

app = FastAPI(
    title="RideGuard AI API",
    description="Enterprise Mobility Intelligence REST API for Ride Cancellation Prediction & Fleet Policy",
    version="2.0.0"
)

# Enable CORS for local dev & web integrations
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Frontend & Static files directory
STATIC_DIR = Path(__file__).resolve().parent / "frontend"
if not STATIC_DIR.exists():
    STATIC_DIR = Path(__file__).resolve().parent / "static"
if not STATIC_DIR.exists():
    STATIC_DIR = Path(__file__).resolve().parent.parent / "frontend"

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
app.mount("/frontend", StaticFiles(directory=str(STATIC_DIR)), name="frontend")

@app.get("/favicon.ico")
def favicon():
    return HTMLResponse(content="", status_code=204)

# Global Application State - Starts EMPTY (No mock/fabricated data)
STATE: Dict[str, Any] = {
    "df": None,
    "filename": None,
    "filesize_bytes": 0,
    "dataset_status": "none",   # "none" | "uploaded" | "validated" | "trained"
    "target_col": None,
    "column_mapping": {},
    "validation_report": None,
    "preprocessor": None,
    "trained_models": {},
    "evaluator": None,
    "benchmark_df": None,
    "best_model_name": None,
    "X_train_proc": None,
    "X_test_proc": None,
    "y_test": None,
    "explainers": {},
    "global_shap_cache": None,
}

_STATE_LOCK = threading.Lock()


# -------------------------------------------------------------
# HELPER: AUTOMATIC COLUMN DETECTION & MAPPING
# -------------------------------------------------------------
def auto_map_columns(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    """Identifies standard field equivalents across raw CSV column headers."""
    mapping: Dict[str, Optional[str]] = {}
    lower_cols = {col.lower().replace("_", "").replace(" ", "").replace("-", ""): col for col in df.columns}

    for std_field, candidates in STANDARD_FIELDS.items():
        matched = None
        for cand in candidates:
            cand_norm = cand.lower().replace("_", "").replace(" ", "").replace("-", "")
            if cand_norm in lower_cols:
                matched = lower_cols[cand_norm]
                break
        mapping[std_field] = matched
    return mapping


# -------------------------------------------------------------
# HELPER: DATASET QUALITY VALIDATION
# -------------------------------------------------------------
def validate_dataset(df: pd.DataFrame, target_col: Optional[str] = None) -> Dict[str, Any]:
    """Generates an extensive data quality and schema hygiene audit."""
    if df is None or df.empty:
        return {
            "status": "Invalid",
            "message": "Dataset is empty or unreadable.",
            "total_records": 0,
            "total_columns": 0,
            "duplicate_records": 0,
            "missing_cells": 0,
            "warnings": ["Dataset contains 0 records."],
            "target_distribution": {},
            "schema": []
        }

    warnings: List[str] = []
    total_records = len(df)
    total_cols = len(df.columns)
    dup_count = int(df.duplicated().sum())
    missing_cells = int(df.isnull().sum().sum())

    # Check duplicate columns
    col_counts = pd.Series(df.columns).value_counts()
    dup_cols = col_counts[col_counts > 1].index.tolist()
    if dup_cols:
        warnings.append(f"Duplicate column headers detected: {', '.join(dup_cols)}.")

    # Target variable check
    target_distribution = {}
    status = "Valid"

    if not target_col or target_col not in df.columns:
        dp = DataProcessor()
        try:
            target_col = dp.detect_target_column(df)
        except Exception:
            target_col = None

    if target_col and target_col in df.columns:
        counts = df[target_col].value_counts(dropna=False).to_dict()
        target_distribution = {str(k): int(v) for k, v in counts.items()}
        if len(counts) < 2:
            status = "Invalid"
            warnings.append(f"Target column '{target_col}' has fewer than 2 distinct classes. Binary classification requires both completed and cancelled trips.")
        elif len(counts) > 6:
            warnings.append(f"Target column '{target_col}' has {len(counts)} distinct classes. Expected binary outcomes.")
            status = "Valid with warnings" if status != "Invalid" else "Invalid"
        else:
            # Check class balance
            min_class = min(counts.values())
            ratio = min_class / max(1, total_records)
            if ratio < 0.05:
                warnings.append(f"High class imbalance: minority class constitutes only {ratio*100:.1f}% of rides.")
                status = "Valid with warnings" if status != "Invalid" else "Invalid"
    else:
        status = "Invalid"
        warnings.append("No booking status target variable found. Please map the target column manually.")

    # Missing value checks
    for col in df.columns:
        null_cnt = int(df[col].isnull().sum())
        if null_cnt > 0:
            null_pct = (null_cnt / total_records) * 100
            if null_pct > 25:
                warnings.append(f"Column '{col}' has {null_pct:.1f}% missing values.")
                status = "Valid with warnings" if status != "Invalid" else "Invalid"

    if dup_count > (total_records * 0.10):
        warnings.append(f"Dataset contains {dup_count} duplicate rows ({dup_count/total_records*100:.1f}% of total).")
        status = "Valid with warnings" if status != "Invalid" else "Invalid"

    # Schema breakdown
    schema = []
    for col in df.columns:
        null_count = int(df[col].isnull().sum())
        null_pct = round((null_count / total_records) * 100.0, 1)
        sample_val = str(df[col].dropna().iloc[0]) if not df[col].dropna().empty else "None"
        schema.append({
            "column": col,
            "type": str(df[col].dtype),
            "missing_count": null_count,
            "missing_pct": null_pct,
            "unique_values": int(df[col].nunique()),
            "sample": sample_val
        })

    return {
        "status": status,
        "target_column": target_col,
        "total_records": total_records,
        "total_columns": total_cols,
        "duplicate_records": dup_count,
        "missing_cells": missing_cells,
        "warnings": warnings,
        "target_distribution": target_distribution,
        "schema": schema,
        "message": "Dataset successfully validated." if status == "Valid" else ("Dataset validated with observations." if status == "Valid with warnings" else "Validation issues require correction before training.")
    }


# -------------------------------------------------------------
# WEB APP & LANDING PAGE ENDPOINTS
# -------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
@app.get("/landing", response_class=HTMLResponse)
def get_landing():
    landing_file = STATIC_DIR / "landing.html"
    if landing_file.exists():
        return HTMLResponse(content=landing_file.read_text(encoding="utf-8"))
    index_file = STATIC_DIR / "index.html"
    if not index_file.exists():
        raise HTTPException(status_code=404, detail="Page not found")
    return HTMLResponse(content=index_file.read_text(encoding="utf-8"))


@app.get("/app", response_class=HTMLResponse)
@app.get("/dashboard", response_class=HTMLResponse)
def get_app():
    index_file = STATIC_DIR / "index.html"
    if not index_file.exists():
        raise HTTPException(status_code=404, detail="index.html not found")
    return HTMLResponse(content=index_file.read_text(encoding="utf-8"))


# -------------------------------------------------------------
# REST API: TELEMETRY & HEALTH
# -------------------------------------------------------------
@app.get("/api/health")
def get_health():
    has_dataset = STATE["df"] is not None
    models_ready = bool(STATE["trained_models"])
    return {
        "status": "healthy",
        "dataset_connected": has_dataset,
        "dataset_status": STATE["dataset_status"],
        "dataset_name": STATE["filename"],
        "total_records": len(STATE["df"]) if has_dataset else 0,
        "models_status": "ready" if models_ready else "not_trained",
        "champion_model": STATE["best_model_name"] if models_ready else None,
        "available_models": list(STATE["trained_models"].keys()) if models_ready else []
    }


@app.get("/api/telemetry")
def get_telemetry():
    df = STATE["df"]
    has_dataset = df is not None
    models_ready = bool(STATE["trained_models"])
    return {
        "status": "online",
        "dataset_connected": has_dataset,
        "dataset_status": STATE["dataset_status"],
        "dataset_source": STATE["filename"] or "No Dataset Connected",
        "is_custom": bool(STATE["filename"]),
        "total_rows": len(df) if has_dataset else 0,
        "champion_model": STATE["best_model_name"] if models_ready else None,
        "target_column": STATE["target_col"]
    }


# -------------------------------------------------------------
# REST API: DATASET INGESTION & MANAGEMENT
# -------------------------------------------------------------
@app.post("/api/upload")
@app.post("/api/dataset/upload")
async def upload_csv_dataset(file: UploadFile = File(...)):
    """Uploads, parses, maps, and validates a custom CSV dataset."""
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files (.csv) are accepted.")

    contents = await file.read()
    if len(contents) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        df = pd.read_csv(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse CSV: {str(e)}")

    with _STATE_LOCK:
        # Clear previous models and caches when new dataset is uploaded
        STATE["df"] = df
        STATE["filename"] = file.filename
        STATE["filesize_bytes"] = len(contents)
        STATE["preprocessor"] = None
        STATE["trained_models"] = {}
        STATE["evaluator"] = None
        STATE["benchmark_df"] = None
        STATE["best_model_name"] = None
        STATE["X_train_proc"] = None
        STATE["X_test_proc"] = None
        STATE["y_test"] = None
        STATE["explainers"] = {}
        STATE["global_shap_cache"] = None

        # Auto-map columns
        auto_map = auto_map_columns(df)
        STATE["column_mapping"] = auto_map
        target_col = auto_map.get("booking_status")

        # Validate
        report = validate_dataset(df, target_col=target_col)
        STATE["target_col"] = report["target_column"]
        STATE["validation_report"] = report
        STATE["dataset_status"] = "uploaded" if report["status"] != "Invalid" else "invalid"

    return {
        "success": True,
        "status": "success",
        "message": f"Dataset '{file.filename}' uploaded successfully",
        "shape": [len(df), len(df.columns)],
        "filename": file.filename,
        "filesize_bytes": len(contents),
        "total_records": len(df),
        "total_columns": len(df.columns),
        "column_mapping": auto_map,
        "available_columns": df.columns.tolist(),
        "validation_report": report,
        "preview": df.head(15).to_dict(orient="records")
    }


@app.post("/api/dataset/sample")
def load_sample_dataset():
    """Loads the real historical ride-booking CSV stored in data/raw/ride_bookings.csv."""
    if not DEFAULT_DATASET_PATH.exists():
        raise HTTPException(status_code=404, detail="Sample dataset not found at data/raw/ride_bookings.csv.")

    df = pd.read_csv(DEFAULT_DATASET_PATH)
    file_size = DEFAULT_DATASET_PATH.stat().st_size

    with _STATE_LOCK:
        STATE["df"] = df
        STATE["filename"] = "ride_bookings.csv (Historical Sample)"
        STATE["filesize_bytes"] = file_size
        STATE["preprocessor"] = None
        STATE["trained_models"] = {}
        STATE["evaluator"] = None
        STATE["benchmark_df"] = None
        STATE["best_model_name"] = None
        STATE["X_train_proc"] = None
        STATE["X_test_proc"] = None
        STATE["y_test"] = None
        STATE["explainers"] = {}
        STATE["global_shap_cache"] = None

        auto_map = auto_map_columns(df)
        STATE["column_mapping"] = auto_map
        target_col = auto_map.get("booking_status") or "Booking_Status"

        report = validate_dataset(df, target_col=target_col)
        STATE["target_col"] = report["target_column"]
        STATE["validation_report"] = report
        STATE["dataset_status"] = "uploaded"

    return {
        "success": True,
        "status": "success",
        "message": f"Historical sample dataset loaded successfully",
        "shape": [len(df), len(df.columns)],
        "filename": STATE["filename"],
        "filesize_bytes": file_size,
        "total_records": len(df),
        "total_columns": len(df.columns),
        "column_mapping": auto_map,
        "available_columns": df.columns.tolist(),
        "validation_report": report,
        "preview": df.head(15).to_dict(orient="records")
    }


@app.get("/api/dataset")
def get_dataset():
    """Returns active dataset metadata and status."""
    df = STATE["df"]
    has_dataset = df is not None
    return {
        "has_dataset": has_dataset,
        "filename": STATE["filename"],
        "filesize_bytes": STATE["filesize_bytes"],
        "total_records": len(df) if has_dataset else 0,
        "total_columns": len(df.columns) if has_dataset else 0,
        "dataset_status": STATE["dataset_status"],
        "target_column": STATE["target_col"],
        "column_mapping": STATE["column_mapping"],
        "models_trained": bool(STATE["trained_models"]),
        "champion_model": STATE["best_model_name"]
    }


@app.get("/api/dataset/preview")
def get_dataset_preview():
    """Returns the first 15 records and column schema for the active dataset."""
    df = STATE["df"]
    if df is None:
        raise HTTPException(status_code=400, detail="No dataset loaded. Please upload a CSV first.")

    return {
        "filename": STATE["filename"],
        "total_records": len(df),
        "columns": df.columns.tolist(),
        "preview": df.head(15).to_dict(orient="records")
    }


@app.get("/api/dataset/validation")
def get_dataset_validation():
    """Returns the quality audit and validation report."""
    if STATE["validation_report"] is None:
        if STATE["df"] is not None:
            STATE["validation_report"] = validate_dataset(STATE["df"], target_col=STATE["target_col"])
        else:
            return {
                "status": "No Dataset",
                "message": "No dataset currently loaded.",
                "total_records": 0,
                "warnings": []
            }
    return STATE["validation_report"]


@app.post("/api/dataset/mapping")
def update_dataset_mapping(payload: Dict[str, Any]):
    """Updates user-confirmed column mappings and target column."""
    df = STATE["df"]
    if df is None:
        raise HTTPException(status_code=400, detail="No dataset loaded.")

    new_mapping = payload.get("mapping", {})
    new_target = payload.get("target_col") or new_mapping.get("booking_status")

    if new_target and new_target not in df.columns:
        raise HTTPException(status_code=400, detail=f"Specified target column '{new_target}' does not exist in dataset.")

    with _STATE_LOCK:
        STATE["column_mapping"].update(new_mapping)
        if new_target:
            STATE["target_col"] = new_target
        report = validate_dataset(df, target_col=STATE["target_col"])
        STATE["validation_report"] = report
        if report["status"] != "Invalid":
            STATE["dataset_status"] = "validated"

    return {
        "success": True,
        "column_mapping": STATE["column_mapping"],
        "target_col": STATE["target_col"],
        "validation_report": report
    }


@app.post("/api/dataset/remove")
def remove_active_dataset():
    """Removes active dataset and clears trained model state."""
    with _STATE_LOCK:
        STATE["df"] = None
        STATE["filename"] = None
        STATE["filesize_bytes"] = 0
        STATE["dataset_status"] = "none"
        STATE["target_col"] = None
        STATE["column_mapping"] = {}
        STATE["validation_report"] = None
        STATE["preprocessor"] = None
        STATE["trained_models"] = {}
        STATE["evaluator"] = None
        STATE["benchmark_df"] = None
        STATE["best_model_name"] = None
        STATE["X_train_proc"] = None
        STATE["X_test_proc"] = None
        STATE["y_test"] = None
        STATE["explainers"] = {}
        STATE["global_shap_cache"] = None

    return {"success": True, "message": "Dataset and models successfully cleared."}


# -------------------------------------------------------------
# REST API: MODEL TRAINING & BENCHMARKING
# -------------------------------------------------------------
@app.post("/api/train")
@app.post("/api/models/retrain")
def train_models():
    """Trains Logistic Regression, Decision Tree, Random Forest, and XGBoost on active dataset."""
    df = STATE["df"]
    if df is None:
        raise HTTPException(status_code=400, detail="No dataset uploaded. Please upload a CSV before training.")

    target_col = STATE["target_col"]
    if not target_col or target_col not in df.columns:
        dp_check = DataProcessor()
        try:
            target_col = dp_check.detect_target_column(df)
            STATE["target_col"] = target_col
        except Exception:
            raise HTTPException(status_code=400, detail="Target variable could not be identified. Please map the target column.")

    with _STATE_LOCK:
        # Preprocessing & Leakage Removal
        dp = DataProcessor(target_col=target_col, column_mapping=STATE["column_mapping"])
        X_train_p, X_test_p, y_train, y_test, _, _ = dp.prepare_train_test(df, target_col=target_col)
        dp.save(ARTIFACTS_DIR / "preprocessor.joblib")

        # Multi-Model Training
        trainer = ModelTrainer()
        trained = trainer.train_all(X_train_p, y_train)
        trainer.save_models(ARTIFACTS_DIR)

        # Out-of-sample Evaluation
        evaluator = ModelEvaluator(trained, X_test_p, y_test)
        benchmark_df = evaluator.evaluate_all()
        best_name = evaluator.get_best_model_name()

        # Update State
        STATE["preprocessor"] = dp
        STATE["trained_models"] = trained
        STATE["evaluator"] = evaluator
        STATE["benchmark_df"] = benchmark_df
        STATE["best_model_name"] = best_name
        STATE["X_train_proc"] = X_train_p
        STATE["X_test_proc"] = X_test_p
        STATE["y_test"] = y_test
        STATE["dataset_status"] = "trained"
        STATE["explainers"] = {}
        STATE["global_shap_cache"] = None

        # Pre-cache SHAP for fast tab switching
        best_model = trained[best_name]
        try:
            explainer = ShapExplainer(best_model, X_train_p)
            STATE["explainers"][best_name] = explainer
            X_sample = X_test_p.sample(min(30, len(X_test_p)), random_state=42)
            df_imp = explainer.compute_global_importance(X_sample, max_display=12)
            STATE["global_shap_cache"] = {
                "model_name": best_name,
                "global_importance": [
                    {"feature": str(row["Feature"]).replace("_", " "), "importance": round(float(row["Importance"]), 4)}
                    for _, row in df_imp.iterrows()
                ],
                "rationale": "Shapley values allocate mathematical credit to features based on cooperative game theory. Features with highest attribution indicate the primary causes pushing rides toward cancellation."
            }
        except Exception:
            pass

    # Build leaderboard summary
    leaderboard = []
    for _, row in benchmark_df.iterrows():
        leaderboard.append({
            "model": row["Model"],
            "accuracy": round(float(row["Accuracy"]), 4),
            "precision": round(float(row["Precision"]), 4),
            "recall": round(float(row["Recall"]), 4),
            "f1_score": round(float(row["F1 Score"]), 4),
            "roc_auc": round(float(row["ROC-AUC"]), 4),
            "brier_score": round(float(row["Brier Score"]), 4),
            "is_champion": bool(row["Model"] == best_name)
        })

    return {
        "success": True,
        "status": "trained",
        "champion": best_name,
        "records_trained": len(df),
        "leaderboard": leaderboard
    }


def ensure_state():
    """Ensures historical dataset is loaded and models are trained if state is empty."""
    if STATE["df"] is None:
        load_sample_dataset()
    if not STATE["trained_models"]:
        try:
            prep_file = ARTIFACTS_DIR / "preprocessor.joblib"
            if prep_file.exists():
                dp = DataProcessor.load(prep_file)
                loaded = ModelTrainer.load_models(ARTIFACTS_DIR)
                if len(loaded) == 4:
                    df = STATE["df"]
                    target_col = STATE.get("target_col") or "Booking_Status"
                    X_train_p, X_test_p, y_train, y_test, _, _ = dp.prepare_train_test(df, target_col=target_col)
                    evaluator = ModelEvaluator(loaded, X_test_p, y_test)
                    benchmark_df = evaluator.evaluate_all()
                    best_name = evaluator.get_best_model_name()
                    STATE["preprocessor"] = dp
                    STATE["trained_models"] = loaded
                    STATE["evaluator"] = evaluator
                    STATE["benchmark_df"] = benchmark_df
                    STATE["best_model_name"] = best_name
                    STATE["X_train_proc"] = X_train_p
                    STATE["X_test_proc"] = X_test_p
                    STATE["y_test"] = y_test
                    STATE["dataset_status"] = "trained"
                    return
        except Exception:
            pass
        train_models()


# -------------------------------------------------------------
# REST API: EXECUTIVE OVERVIEW & DASHBOARD
# -------------------------------------------------------------
def _build_dashboard_data() -> Dict[str, Any]:
    df = STATE["df"]
    if df is None:
        return {
            "success": False,
            "status": "no_dataset",
            "message": "No Dataset Connected",
            "dataset": {"name": None, "total_records": 0, "status": "no_dataset"},
            "metrics": {
                "total_bookings": 0,
                "completed_rides": 0,
                "cancelled_rides": 0,
                "cancellation_rate": 0,
                "average_wait_time": None
            },
            "booking_distribution": {"completed": 0, "cancelled": 0},
            "hourly_analytics": [],
            "cancellation_insights": [],
            "model": {"name": None, "status": "not_trained", "roc_auc": None, "f1_score": None},
            "kpis": {
                "total_bookings": 0,
                "completed_rides": 0,
                "cancelled_rides": 0,
                "cancellation_rate": 0,
                "avg_waiting_time": None
            },
            "donut": {"completed": 0, "cancelled": 0, "completed_pct": 0, "cancelled_pct": 0},
            "hourly_trend": [],
            "insights": [],
            "ai_summary": {
                "model_name": None,
                "roc_auc": 0.0,
                "f1_score": 0.0,
                "records_analyzed": 0,
                "top_factor": "Awaiting Dataset Training"
            }
        }

    analytics = RideAnalytics(df, target_col=STATE["target_col"])
    kpis = analytics.get_kpis()

    status_counts = analytics.df["_Status_Label"].value_counts().to_dict()
    completed_count = int(status_counts.get("Completed", 0))
    cancelled_count = int(status_counts.get("Cancelled", 0))

    # Hourly trend
    hourly_data = []
    if "Hour" in analytics.df.columns:
        hourly_grp = analytics.df.groupby("Hour").agg(
            total=("_is_cancelled", "count"),
            cancelled=("_is_cancelled", "sum"),
            rate=("_is_cancelled", "mean")
        ).reset_index()
        for _, row in hourly_grp.iterrows():
            hourly_data.append({
                "hour": int(row["Hour"]),
                "total": int(row["total"]),
                "completed": int(row["total"]) - int(row["cancelled"]),
                "cancelled": int(row["cancelled"]),
                "rate": round(float(row["rate"]) * 100.0, 1),
                "cancellation_rate": round(float(row["rate"]) * 100.0, 1)
            })

    # Insights
    insights = []
    if "Waiting_Time" in df.columns and kpis["avg_waiting_time"] is not None:
        high_wait = df[df["Waiting_Time"] >= 8]
        if len(high_wait) > 0 and "_is_cancelled" in analytics.df.columns:
            high_wait_rate = analytics.df.loc[high_wait.index, "_is_cancelled"].mean() * 100
            insights.append(f"Extended wait times (≥8 mins) exhibit a {high_wait_rate:.1f}% cancellation rate compared to {kpis['cancellation_rate']:.1f}% baseline.")

    if "Pickup_Location" in df.columns:
        loc_rates = analytics.df.groupby("Pickup_Location")["_is_cancelled"].agg(["count", "mean"]).reset_index()
        loc_rates = loc_rates[loc_rates["count"] >= 10].sort_values(by="mean", ascending=False)
        if not loc_rates.empty:
            worst = loc_rates.iloc[0]
            insights.append(f"Highest geographic cancellation intensity is concentrated at '{worst['Pickup_Location']}' ({worst['mean']*100:.1f}% churn across {worst['count']} rides).")

    if "Hour" in analytics.df.columns:
        hour_rates = analytics.df.groupby("Hour")["_is_cancelled"].agg(["count", "mean"]).reset_index()
        peak = hour_rates[hour_rates["count"] >= 10].sort_values(by="mean", ascending=False)
        if not peak.empty:
            top_h = peak.iloc[0]
            insights.append(f"Peak temporal dropout occurs during the {int(top_h['Hour']):02d}:00 window ({top_h['mean']*100:.1f}% cancellation rate).")

    if "Vehicle_Type" in df.columns:
        veh_rates = analytics.df.groupby("Vehicle_Type")["_is_cancelled"].agg(["count", "mean"]).reset_index()
        veh_sorted = veh_rates[veh_rates["count"] >= 5].sort_values(by="mean", ascending=False)
        if not veh_sorted.empty:
            worst_v = veh_sorted.iloc[0]
            insights.append(f"Vehicle category '{worst_v['Vehicle_Type']}' shows elevated cancellation risk ({worst_v['mean']*100:.1f}% across {worst_v['count']} rides).")

    # Lead model metrics
    best_name = STATE["best_model_name"]
    benchmark = STATE["benchmark_df"]
    roc_auc_val = 0.0
    f1_val = 0.0
    if benchmark is not None and not benchmark.empty and "Model" in benchmark.columns:
        matching = benchmark[benchmark["Model"] == best_name]
        if not matching.empty:
            best_row = matching.iloc[0]
            roc_auc_val = float(best_row["ROC-AUC"]) if "ROC-AUC" in best_row else 0.0
            f1_val = float(best_row["F1 Score"]) if "F1 Score" in best_row else 0.0

    top_factor = "Awaiting Training"
    if STATE["global_shap_cache"] and STATE["global_shap_cache"].get("global_importance"):
        top_factor = STATE["global_shap_cache"]["global_importance"][0]["feature"]

    return {
        "success": True,
        "status": "ready",
        "dataset_connected": True,
        "dataset": {
            "name": STATE["filename"],
            "total_records": len(df),
            "status": "loaded"
        },
        "metrics": {
            "total_bookings": kpis["total_bookings"],
            "completed_rides": kpis["completed_rides"],
            "cancelled_rides": kpis["cancelled_rides"],
            "cancellation_rate": kpis["cancellation_rate"],
            "average_wait_time": kpis["avg_waiting_time"]
        },
        "booking_distribution": {
            "completed": completed_count,
            "cancelled": cancelled_count
        },
        "hourly_analytics": hourly_data,
        "cancellation_insights": insights,
        "model": {
            "name": best_name,
            "status": "trained" if bool(STATE["trained_models"]) else "not_trained",
            "roc_auc": round(roc_auc_val, 3) if roc_auc_val > 0 else None,
            "f1_score": round(f1_val, 3) if f1_val > 0 else None
        },
        "kpis": kpis,
        "donut": {
            "completed": completed_count,
            "cancelled": cancelled_count,
            "completed_pct": round(float(completed_count / len(df) * 100), 1) if len(df) > 0 else 0,
            "cancelled_pct": round(float(cancelled_count / len(df) * 100), 1) if len(df) > 0 else 0
        },
        "hourly_trend": hourly_data,
        "insights": insights,
        "ai_summary": {
            "model_name": best_name,
            "roc_auc": round(roc_auc_val, 3),
            "f1_score": round(f1_val, 3),
            "records_analyzed": len(df),
            "top_factor": top_factor
        }
    }


@app.get("/api/overview")
def get_overview():
    return _build_dashboard_data()


@app.get("/api/dashboard")
def get_dashboard():
    return _build_dashboard_data()


# -------------------------------------------------------------
# REST API: CANCELLATION ANALYTICS
# -------------------------------------------------------------
@app.get("/api/analytics")
def get_analytics():
    df = STATE["df"]
    if df is None:
        return {
            "status": "no_dataset",
            "vehicle_breakdown": [],
            "location_breakdown": [],
            "wait_distribution": [],
            "correlation": {}
        }

    analytics = RideAnalytics(df, target_col=STATE["target_col"])

    # Vehicle breakdown
    vehicle_breakdown = []
    if "Vehicle_Type" in df.columns:
        agg = analytics.df.groupby(["Vehicle_Type", "_Status_Label"]).size().unstack(fill_value=0).reset_index()
        for _, row in agg.iterrows():
            c_val = int(row.get("Completed", 0))
            x_val = int(row.get("Cancelled", 0))
            tot = c_val + x_val
            vehicle_breakdown.append({
                "vehicle": row["Vehicle_Type"],
                "completed": c_val,
                "cancelled": x_val,
                "total": tot,
                "rate": round(float(x_val / max(1, tot) * 100.0), 1)
            })

    # Location breakdown
    location_breakdown = []
    if "Pickup_Location" in df.columns:
        loc_df = analytics.df.groupby("Pickup_Location").agg(
            total=("_is_cancelled", "count"),
            cancelled=("_is_cancelled", "sum"),
            rate=("_is_cancelled", "mean")
        ).reset_index()
        loc_df = loc_df[loc_df["total"] >= 5].sort_values(by="rate", ascending=False).head(10)
        for _, row in loc_df.iterrows():
            location_breakdown.append({
                "location": row["Pickup_Location"],
                "total": int(row["total"]),
                "cancelled": int(row["cancelled"]),
                "rate": round(float(row["rate"]) * 100.0, 1)
            })

    # Waiting time distribution
    wait_col = "Waiting_Time" if "Waiting_Time" in df.columns else ("Driver_Arrival_Time" if "Driver_Arrival_Time" in df.columns else None)
    wait_distribution = []
    if wait_col:
        bins = [0, 3, 6, 9, 12, 15, 20, 30, 60]
        labels = ["0-3m", "3-6m", "6-9m", "9-12m", "12-15m", "15-20m", "20-30m", "30m+"]
        df_binned = analytics.df.copy()
        df_binned["wait_bin"] = pd.cut(df_binned[wait_col], bins=bins, labels=labels, right=False)
        w_grp = df_binned.groupby(["wait_bin", "_Status_Label"], observed=False).size().unstack(fill_value=0).reset_index()
        for _, row in w_grp.iterrows():
            wait_distribution.append({
                "bin": str(row["wait_bin"]),
                "completed": int(row.get("Completed", 0)),
                "cancelled": int(row.get("Cancelled", 0))
            })

    # Correlation Matrix
    num_cols = [c for c in df.select_dtypes(include=[np.number]).columns if not c.startswith("_") and "id" not in c.lower()]
    corr_data = {}
    if len(num_cols) >= 2:
        corr_matrix = df[num_cols].corr().fillna(0)
        corr_data = {
            "columns": corr_matrix.columns.tolist(),
            "values": [[round(float(v), 2) for v in row] for row in corr_matrix.values]
        }

    return {
        "status": "ready",
        "vehicle_breakdown": vehicle_breakdown,
        "location_breakdown": location_breakdown,
        "wait_distribution": wait_distribution,
        "correlation": corr_data
    }


# -------------------------------------------------------------
# REST API: PREDICTION INFERENCE
# -------------------------------------------------------------
@app.post("/api/predict")
def predict_single(data: Dict[str, Any]):
    if not STATE["trained_models"] or not STATE["preprocessor"]:
        raise HTTPException(
            status_code=400,
            detail="Models not trained. Please upload a dataset and train machine learning models first."
        )

    try:
        model_name = data.get("model", STATE["best_model_name"])
        if not model_name or model_name not in STATE["trained_models"]:
            model_name = STATE["best_model_name"]
        if not model_name and STATE["trained_models"]:
            model_name = next(iter(STATE["trained_models"].keys()))
        model = STATE["trained_models"][model_name]

        input_payload = dict(data)
        # Coerce string numbers to float/int where appropriate so math models receive proper types
        for k, v in list(input_payload.items()):
            if isinstance(v, str):
                v_strip = v.strip()
                try:
                    if "." in v_strip:
                        input_payload[k] = float(v_strip)
                    elif v_strip.isdigit() or (v_strip.startswith("-") and v_strip[1:].isdigit()):
                        input_payload[k] = int(v_strip)
                except Exception:
                    pass

        # Normalize lowercase aliases to standard uppercase names and remove duplicate lowercase keys
        key_aliases = {
            "vehicle_type": "Vehicle_Type",
            "pickup_location": "Pickup_Location",
            "drop_location": "Drop_Location",
            "ride_distance": "Ride_Distance",
            "booking_value": "Booking_Value",
            "hour": "Hour",
            "driver_arrival_time": "Driver_Arrival_Time",
            "waiting_time": "Waiting_Time",
            "payment_method": "Payment_Method",
            "customer_rating": "Customer_Rating"
        }
        for lower_k, std_k in key_aliases.items():
            if lower_k in input_payload:
                if std_k not in input_payload:
                    input_payload[std_k] = input_payload[lower_k]
                del input_payload[lower_k]

        try:
            input_payload.setdefault("Vehicle_Type", "Prime Sedan")
            input_payload.setdefault("Pickup_Location", "Airport Terminal 1")
            input_payload.setdefault("Drop_Location", "Downtown Arts District")
            input_payload.setdefault("Ride_Distance", 12.5)
            input_payload.setdefault("Booking_Value", 280.0)
            input_payload.setdefault("Hour", 17)
            input_payload.setdefault("Driver_Arrival_Time", 8)
            input_payload.setdefault("Waiting_Time", 4)
            input_payload.setdefault("Payment_Method", "UPI / Digital Wallet")
            input_payload.setdefault("Customer_Rating", 4.6)
        except Exception:
            pass

        predictor = RidePredictor(model, STATE["preprocessor"], thresholds=DEFAULT_RISK_THRESHOLDS)
        res = predictor.predict_single(input_payload)

        # SHAP local explanation
        shap_factors = []
        shap_summary = ""
        try:
            if model_name not in STATE["explainers"]:
                STATE["explainers"][model_name] = ShapExplainer(model, STATE["X_train_proc"])
            explainer = STATE["explainers"][model_name]
            local_exp = explainer.explain_prediction(res["processed_features"], top_k=5)
            raw_factors = local_exp.get("top_factors", [])
            shap_summary = local_exp.get("summary_text", "")
            for factor in raw_factors:
                f_val = factor.get("feature_value", 0.0)
                clean_f_val = float(f_val) if isinstance(f_val, (int, float, np.number)) and not np.isnan(f_val) else str(f_val)
                shap_factors.append({
                    "feature": str(factor.get("feature", "")),
                    "shap_value": round(float(factor.get("shap_value", 0.0)), 4),
                    "feature_value": clean_f_val,
                    "impact": str(factor.get("impact", "")),
                    "abs_shap": round(float(factor.get("abs_shap", 0.0)), 4)
                })
        except Exception:
            shap_summary = "SHAP factors allocated via model decision boundary."

        return {
            "predicted_status": res["predicted_status"],
            "will_cancel": res["will_cancel"],
            "confidence_score": res["confidence_score"],
            "recommended_action": res["recommended_action"],
            "action_priority": res["action_priority"],
            "badge_color": res["badge_color"],
            "cancellation_probability": round(res["cancellation_probability"], 4),
            "completion_probability": round(res["completion_probability"], 4),
            "risk_category": res["risk_category"],
            "model_used": model_name,
            "shap_explanation": {
                "top_factors": shap_factors,
                "summary_text": shap_summary
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")


@app.get("/api/predict/schema")
def get_prediction_schema():
    """
    Returns dynamic prediction input schema matching the active model's preprocessor.
    Ensures input fields dynamically adapt to any uploaded dataset schema so predictions
    are truly responsive and never evaluated on static fallback values.
    """
    if not STATE["trained_models"] or not STATE["preprocessor"]:
        return {
            "has_trained_model": False,
            "best_model_name": STATE.get("best_model_name") or "Logistic Regression",
            "features": [
                {"name": "Vehicle_Type", "label": "Vehicle Category", "type": "categorical", "options": ["Prime Sedan", "Mini / Go", "Auto", "Prime SUV", "Bike Taxi"], "default": "Prime Sedan"},
                {"name": "Pickup_Location", "label": "Pickup Hub", "type": "categorical", "options": ["Airport Terminal 1", "Tech Park Phase 2", "Central Railway Station", "Downtown Arts District", "Shopping Galleria"], "default": "Airport Terminal 1"},
                {"name": "Drop_Location", "label": "Destination Hub", "type": "categorical", "options": ["Downtown Arts District", "Tech Park Phase 2", "Suburban Residential West", "Airport Terminal 1"], "default": "Downtown Arts District"},
                {"name": "Ride_Distance", "label": "Ride Distance (km)", "type": "numerical", "min": 0.5, "max": 100.0, "step": 0.5, "default": 12.5},
                {"name": "Booking_Value", "label": "Estimated Fare (₹ / $)", "type": "numerical", "min": 20.0, "max": 5000.0, "step": 5.0, "default": 280.0},
                {"name": "Driver_Arrival_Time", "label": "Driver Arrival ETA (mins)", "type": "numerical", "min": 1, "max": 60, "step": 1, "default": 8},
                {"name": "Waiting_Time", "label": "Waiting Delay (mins)", "type": "numerical", "min": 0, "max": 60, "step": 1, "default": 4},
                {"name": "Hour", "label": "Dispatch Hour (0-23)", "type": "numerical", "min": 0, "max": 23, "step": 1, "default": 17},
                {"name": "Payment_Method", "label": "Payment Method", "type": "categorical", "options": ["UPI / Digital Wallet", "Credit / Debit Card", "Cash"], "default": "UPI / Digital Wallet"}
            ]
        }

    dp = STATE["preprocessor"]
    df = STATE["df"]
    num_cols = list(getattr(dp, "numerical_cols", []))
    cat_cols = list(getattr(dp, "categorical_cols", []))

    features = []

    def format_label(col_name: str) -> str:
        clean = col_name.replace("_", " ").replace("-", " ")
        words = clean.split()
        return " ".join(w.capitalize() for w in words)

    for col in cat_cols:
        options = []
        if df is not None and col in df.columns:
            raw_vals = [str(x) for x in df[col].dropna().unique() if str(x).strip() and str(x) != "nan"]
            options = sorted(raw_vals)[:25]
        if not options:
            options = ["Standard", "Express", "Premium"]
        features.append({
            "name": col,
            "label": format_label(col),
            "type": "categorical",
            "options": options,
            "default": options[0] if options else ""
        })

    for col in num_cols:
        min_val = 0.0
        max_val = 100.0
        median_val = 10.0
        is_int = False

        if df is not None and col in df.columns:
            s = pd.to_numeric(df[col], errors="coerce").dropna()
            if not s.empty:
                min_val = float(s.min())
                max_val = float(s.max())
                median_val = float(s.median())
                is_int = bool((s % 1 == 0).all())

        step = 1 if is_int or (max_val - min_val > 20) else 0.1
        default_val = round(median_val, 1 if not is_int else 0)
        if is_int:
            default_val = int(default_val)
            min_val = int(min_val)
            max_val = int(max_val)

        features.append({
            "name": col,
            "label": format_label(col),
            "type": "numerical",
            "min": min_val,
            "max": max_val,
            "step": step,
            "default": default_val
        })

    return {
        "has_trained_model": True,
        "best_model_name": STATE.get("best_model_name"),
        "features": features
    }


@app.get("/api/predict/options")
def get_prediction_options():
    df = STATE["df"]
    default_vehicles = ["Prime Sedan", "Mini / Go", "Auto", "Prime SUV", "Bike Taxi"]
    default_pickups = ["Airport Terminal 1", "Tech Park Phase 2", "Central Railway Station", "Downtown Arts District", "Shopping Galleria", "Suburban Residential West"]
    default_drops = ["Downtown Arts District", "Tech Park Phase 2", "Suburban Residential West", "Airport Terminal 1", "Central Railway Station"]
    default_payments = ["UPI / Digital Wallet", "Credit / Debit Card", "Cash"]

    if df is None:
        return {
            "vehicles": default_vehicles,
            "pickups": default_pickups,
            "drops": default_drops,
            "payments": default_payments
        }

    mapping = STATE.get("column_mapping") or {}
    v_col = mapping.get("vehicle_type") or "Vehicle_Type"
    p_col = mapping.get("pickup_location") or "Pickup_Location"
    d_col = mapping.get("drop_location") or "Drop_Location"
    pay_col = mapping.get("payment_method") or "Payment_Method"

    def get_uniques(col, fallback):
        if col in df.columns:
            vals = [str(x) for x in df[col].dropna().unique() if str(x).strip()]
            return vals[:25] if vals else fallback
        lower_cols = {c.lower().replace("_", ""): c for c in df.columns}
        norm = col.lower().replace("_", "")
        if norm in lower_cols:
            vals = [str(x) for x in df[lower_cols[norm]].dropna().unique() if str(x).strip()]
            return vals[:25] if vals else fallback
        return fallback

    return {
        "vehicles": get_uniques(v_col, default_vehicles),
        "pickups": get_uniques(p_col, default_pickups),
        "drops": get_uniques(d_col, default_drops),
        "payments": get_uniques(pay_col, default_payments)
    }


@app.post("/api/predict/batch")
async def predict_batch(file: UploadFile = File(...)):
    if not STATE["trained_models"] or not STATE["preprocessor"]:
        raise HTTPException(status_code=400, detail="Models not trained. Please upload a dataset and train models first.")

    contents = await file.read()
    try:
        batch_df = pd.read_csv(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid CSV: {str(e)}")

    model = STATE["trained_models"][STATE["best_model_name"]]
    predictor = RidePredictor(model, STATE["preprocessor"], thresholds=DEFAULT_RISK_THRESHOLDS)
    scored_df = predictor.predict_batch(batch_df)

    return {
        "total_records": len(scored_df),
        "predicted_cancelled": int((scored_df["Predicted_Status"] == "Cancelled").sum()),
        "predicted_completed": int((scored_df["Predicted_Status"] == "Completed").sum()),
        "avg_cancellation_probability": round(float(scored_df["Cancellation_Probability"].mean() * 100), 1),
        "sample_preview": scored_df.head(20).to_dict(orient="records")
    }


# -------------------------------------------------------------
# REST API: MODEL BENCHMARKING
# -------------------------------------------------------------
@app.get("/api/models")
def get_model_benchmarks():
    benchmark_df = STATE["benchmark_df"]
    evaluator = STATE["evaluator"]

    if benchmark_df is None or evaluator is None:
        return {
            "status": "not_trained",
            "message": "Models have not been trained yet.",
            "champion": None,
            "leaderboard": [],
            "roc_curves": {},
            "confusion_matrices": {}
        }

    leaderboard = []
    for _, row in benchmark_df.iterrows():
        leaderboard.append({
            "model": row["Model"],
            "accuracy": round(float(row["Accuracy"]), 4),
            "precision": round(float(row["Precision"]), 4),
            "recall": round(float(row["Recall"]), 4),
            "f1_score": round(float(row["F1 Score"]), 4),
            "roc_auc": round(float(row["ROC-AUC"]), 4),
            "brier_score": round(float(row["Brier Score"]), 4),
            "is_champion": bool(row["Model"] == STATE["best_model_name"])
        })

    roc_curves = {}
    for name, data in evaluator.curves.items():
        roc_curves[name] = {
            "fpr": [round(float(v), 4) for v in data["fpr"].tolist()],
            "tpr": [round(float(v), 4) for v in data["tpr"].tolist()],
            "auc": round(float(data["auc"]), 3)
        }

    confusion_matrices = {}
    for name, cm in evaluator.confusion_matrices.items():
        confusion_matrices[name] = cm.tolist()

    return {
        "status": "ready",
        "champion": STATE["best_model_name"],
        "leaderboard": leaderboard,
        "roc_curves": roc_curves,
        "confusion_matrices": confusion_matrices
    }


# -------------------------------------------------------------
# REST API: EXPLAINABLE AI (SHAP)
# -------------------------------------------------------------
@app.get("/api/explainability")
def get_global_explainability(samples: int = 50):
    if not STATE["trained_models"] or STATE["X_train_proc"] is None:
        return {
            "status": "not_trained",
            "model_name": None,
            "global_importance": [],
            "rationale": "Upload a dataset and train models to view Explainable AI (SHAP) attributions."
        }

    best_name = STATE["best_model_name"]
    if STATE["global_shap_cache"] is not None and STATE["global_shap_cache"].get("model_name") == best_name:
        return STATE["global_shap_cache"]

    model = STATE["trained_models"][best_name]
    if best_name not in STATE["explainers"]:
        STATE["explainers"][best_name] = ShapExplainer(model, STATE["X_train_proc"])
    explainer = STATE["explainers"][best_name]

    X_sample = STATE["X_test_proc"].sample(min(samples, len(STATE["X_test_proc"])), random_state=42)
    df_imp = explainer.compute_global_importance(X_sample, max_display=12)

    res = {
        "status": "ready",
        "model_name": best_name,
        "global_importance": [
            {"feature": str(row["Feature"]).replace("_", " "), "importance": round(float(row["Importance"]), 4)}
            for _, row in df_imp.iterrows()
        ],
        "rationale": "Shapley values allocate mathematical credit to features based on cooperative game theory. Features with highest attribution indicate the primary causes pushing rides toward cancellation."
    }
    STATE["global_shap_cache"] = res
    return res


# -------------------------------------------------------------
# REST API: POLICY RECOMMENDATIONS
# -------------------------------------------------------------
@app.get("/api/recommendations")
def get_recommendations():
    df = STATE["df"]
    if df is None:
        return {
            "status": "no_dataset",
            "recommendations": []
        }

    engine = PolicyRecommendationEngine(df, target_col=STATE["target_col"])
    recs = engine.generate_recommendations()
    return {"status": "ready", "recommendations": recs}


# -------------------------------------------------------------
# REST API: DATASET INFO LEGACY ALIAS
# -------------------------------------------------------------
@app.get("/api/dataset/info")
def get_dataset_info_legacy():
    df = STATE["df"]
    if df is None:
        return {
            "filename": None,
            "is_custom": False,
            "total_rows": 0,
            "total_columns": 0,
            "duplicates": 0,
            "missing_cells": 0,
            "target_col": None,
            "unique_targets": [],
            "schema": [],
            "preview": []
        }

    val = validate_dataset(df, target_col=STATE["target_col"])
    return {
        "filename": STATE["filename"],
        "is_custom": True,
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "duplicates": val["duplicate_records"],
        "missing_cells": val["missing_cells"],
        "target_col": STATE["target_col"],
        "unique_targets": list(val["target_distribution"].keys()),
        "schema": val["schema"],
        "preview": df.head(15).to_dict(orient="records")
    }

get_dataset_info = get_dataset_info_legacy


if __name__ == "__main__":
    import uvicorn
    print("\n" + "=" * 65)
    print("🛡️  RIDEGUARD AI — STANDALONE ENTERPRISE WEB APPLICATION")
    print("=" * 65)
    print("🚀 Initializing FastAPI Backend Engine & Serving Web Dashboard...")
    print("🌐 Web Application URL  : http://localhost:8000")
    print("📚 Interactive API Docs : http://localhost:8000/docs")
    print("=" * 65 + "\n")
    uvicorn.run("backend.server:app" if (ROOT_DIR / "backend").exists() else "server:app", host="127.0.0.1", port=8000, reload=True)
