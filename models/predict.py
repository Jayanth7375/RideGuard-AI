"""
RideGuard AI - Prediction Interface
Provides single-ride and batch inference using trained models and fitted preprocessors.
Includes decisive YES/NO verdict, confidence scoring, and operational action recommendations.
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
if ROOT_DIR.name == "models":
    ROOT_DIR = ROOT_DIR.parent
BACKEND_DIR = ROOT_DIR / "backend"
MODELS_DIR = ROOT_DIR / "models"
for p in [str(ROOT_DIR), str(BACKEND_DIR), str(MODELS_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from typing import Dict, Any, Union
import pandas as pd
import numpy as np

from preprocessing.data_processor import DataProcessor
from config.settings import DEFAULT_RISK_THRESHOLDS
from utils.helpers import get_risk_label


def derive_action_and_verdict(cancel_prob: float, risk_category: str, input_dict: dict = None) -> Dict[str, Any]:
    """
    Computes decisive YES/NO cancellation verdict, model confidence score,
    and priority operational recommendations.
    """
    input_dict = input_dict or {}
    will_cancel = "YES" if cancel_prob >= 0.50 else "NO"
    confidence = round(max(cancel_prob, 1.0 - cancel_prob) * 100, 1)

    eta = float(input_dict.get("Driver_Arrival_Time") or input_dict.get("driver_arrival_time") or 0)
    wait = float(input_dict.get("Waiting_Time") or input_dict.get("waiting_time") or 0)
    fare = float(input_dict.get("Booking_Value") or input_dict.get("booking_value") or 0)
    dist = max(float(input_dict.get("Ride_Distance") or input_dict.get("ride_distance") or 1), 0.5)
    fare_per_km = fare / dist if dist else 25

    traffic = str(input_dict.get("Traffic_Level") or input_dict.get("traffic_level") or "").strip().lower()
    weather = str(input_dict.get("Weather") or input_dict.get("weather") or "").strip().lower()
    surge = float(input_dict.get("Surge_Multiplier") or input_dict.get("surge_multiplier") or 1.0)
    driver_rating = float(input_dict.get("Driver_Rating") or input_dict.get("driver_rating") or 4.5)

    if will_cancel == "YES":
        if wait >= 5 or eta >= 9:
            action = "🚨 Re-dispatch Closer Driver & Issue Wait Token: ETA delay is the primary churn trigger. Immediately reassign a closer driver (< 3 km radius) and send an automated ₹30 in-app waiting compensation perk to retain passenger."
            priority = "Critical Intervention"
            badge_color = "red"
        elif traffic in ["very high", "high", "congested", "heavy"]:
            action = "🚨 Heavy Traffic Choke Point: Severe corridor congestion detected. Reroute via low-density arterial corridors and push live GPS progress tracking to prevent rider drop."
            priority = "Critical Intervention"
            badge_color = "red"
        elif surge >= 1.6:
            action = "⚡ Calibrate Surge Jump: Steep dynamic surge multiplier is driving price resistance. Smooth steep surge jumps or offer an in-app fare lock perk to prevent rider cancellation."
            priority = "Critical Intervention"
            badge_color = "red"
        elif weather in ["heavy rain", "rain", "storm", "snow"]:
            action = "🌧️ Weather Churn Alert: Rain-induced transit resistance. Dispatch weather-shielded vehicle priority and apply zero-cancellation penalty protection."
            priority = "Critical Intervention"
            badge_color = "red"
        elif fare_per_km < 18 and dist > 2:
            action = "⚡ Apply Driver Surge Multiplier: Low fare-to-distance ratio makes driver rejection likely. Apply a 1.25x driver earnings boost to guarantee pickup fulfillment."
            priority = "High Priority"
            badge_color = "red"
        elif driver_rating < 4.2:
            action = "⚠️ Driver Quality Warning: Assigned driver rating is below fleet standards. Automatically reassign to a top-tier driver (4.8★+) to avert rider rejection."
            priority = "High Priority"
            badge_color = "red"
        else:
            action = "🚨 Priority Churn Intervention: Pre-assign backup standby vehicle, transmit live driver tracking to rider, and lock guaranteed arrival window."
            priority = "High Priority"
            badge_color = "red"
    elif risk_category == "Medium":
        if wait >= 4 or eta >= 7:
            action = "⚠️ Proactive Rider Check-In: Send in-app reassurance update with live GPS progress and place nearest reserve driver on standby."
            priority = "Medium Priority"
            badge_color = "amber"
        elif traffic in ["very high", "high", "moderate"]:
            action = "⚠️ Traffic Delay Watch: Monitor corridor transit velocity; prepare automated backup driver assignment if arrival ETA slips past 7 minutes."
            priority = "Medium Priority"
            badge_color = "amber"
        elif surge >= 1.4:
            action = "⚠️ Surge Sensitivity Watch: Monitor rider acceptance window; offer flat pricing discount if booking is not accepted within 45 seconds."
            priority = "Medium Priority"
            badge_color = "amber"
        else:
            action = "⚠️ Active Monitoring: Track driver heading in real time; prepare automated re-routing if arrival ETA increases past 7 minutes."
            priority = "Medium Priority"
            badge_color = "amber"
    else:
        action = "✅ Proceed With Standard Dispatch: Trip health metrics are optimal. Standard automated dispatch and tracking active."
        priority = "Standard Dispatch"
        badge_color = "emerald"

    return {
        "will_cancel": will_cancel,
        "confidence_score": confidence,
        "recommended_action": action,
        "action_priority": priority,
        "badge_color": badge_color
    }


class RidePredictor:
    """Predicts ride cancellation probability and risk level using production models."""

    def __init__(self, model: Any, preprocessor: DataProcessor, thresholds: dict = None):
        self.model = model
        self.preprocessor = preprocessor
        self.thresholds = thresholds or DEFAULT_RISK_THRESHOLDS

    def predict_single(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Takes raw booking input dictionary, preprocesses it, and returns prediction details
        including decisive YES/NO, confidence score, and recommended operational action.
        """
        df_input = pd.DataFrame([input_data])
        X_proc = self.preprocessor.transform(df_input)

        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(X_proc)[0]
            cancel_prob = float(probs[1])
        else:
            pred = self.model.predict(X_proc)[0]
            cancel_prob = float(pred)

        risk_category = get_risk_label(cancel_prob, self.thresholds)
        predicted_status = "Cancelled" if cancel_prob >= 0.50 else "Completed"

        verdict_info = derive_action_and_verdict(cancel_prob, risk_category, input_data)

        return {
            "predicted_status": predicted_status,
            "will_cancel": verdict_info["will_cancel"],
            "confidence_score": verdict_info["confidence_score"],
            "recommended_action": verdict_info["recommended_action"],
            "action_priority": verdict_info["action_priority"],
            "badge_color": verdict_info["badge_color"],
            "cancellation_probability": cancel_prob,
            "completion_probability": 1.0 - cancel_prob,
            "risk_category": risk_category,
            "processed_features": X_proc,
            "raw_features": df_input
        }

    def predict_batch(self, df_input: pd.DataFrame) -> pd.DataFrame:
        """Runs batch inference over a dataframe and appends prediction and action columns."""
        df_clean = df_input.copy()
        X_proc = self.preprocessor.transform(df_clean)

        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(X_proc)[:, 1]
        else:
            probs = self.model.predict(X_proc)

        results = df_clean.copy()
        results["Cancellation_Probability"] = np.round(probs, 4)
        results["Predicted_Status"] = np.where(probs >= 0.50, "Cancelled", "Completed")
        results["Will_Cancel"] = np.where(probs >= 0.50, "YES", "NO")
        results["Confidence_Score"] = np.round(np.maximum(probs, 1.0 - probs) * 100, 1)
        results["Risk_Category"] = [get_risk_label(p, self.thresholds) for p in probs]

        actions = []
        priorities = []
        for idx, row in results.iterrows():
            info = derive_action_and_verdict(row["Cancellation_Probability"], row["Risk_Category"], row.to_dict())
            actions.append(info["recommended_action"])
            priorities.append(info["action_priority"])

        results["Recommended_Action"] = actions
        results["Action_Priority"] = priorities

        return results
