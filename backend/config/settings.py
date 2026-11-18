"""
RideGuard AI - Configuration & Settings
Defines design tokens, system paths, risk thresholds, and ML configuration.
"""

from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
if BASE_DIR.name == "backend":
    BASE_DIR = BASE_DIR.parent

DATA_DIR = BASE_DIR / "dataset" if (BASE_DIR / "dataset").exists() else BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"
ARTIFACTS_DIR = MODELS_DIR / "model_artifacts"
FRONTEND_DIR = BASE_DIR / "frontend" if (BASE_DIR / "frontend").exists() else BASE_DIR / "static"
ASSETS_DIR = FRONTEND_DIR / "assets" if (FRONTEND_DIR / "assets").exists() else BASE_DIR / "assets"

DEFAULT_DATASET_PATH = RAW_DATA_DIR / "ride_bookings.csv"

# UI Theme Color Palette (Enterprise SaaS Dark Theme)
THEME = {
    "bg_primary": "#080D18",      # Main background
    "bg_sidebar": "#0B1220",      # Sidebar background
    "bg_card": "#111B2B",         # Card background
    "bg_elevated": "#162235",     # Elevated surface
    "border": "#243247",          # Borders
    "accent_blue": "#3B82F6",     # Primary blue
    "accent_cyan": "#22D3EE",     # Cyan accent
    "success": "#22C55E",         # Success
    "warning": "#F59E0B",         # Warning
    "danger": "#EF4444",          # Danger
    "text_primary": "#F8FAFC",    # Primary text
    "text_secondary": "#94A3B8",  # Secondary text
    "text_muted": "#64748B",      # Muted text
}

# Risk Thresholds
DEFAULT_RISK_THRESHOLDS = {
    "low": 0.30,      # < 30% = Low Risk
    "medium": 0.60,   # 30% - 60% = Medium Risk
    "high": 0.60,     # > 60% = High Risk
}

# Leakage Columns Blacklist - Should NEVER be fed into ML model
LEAKAGE_COLUMNS = [
    "cancellation_reason",
    "reason_for_cancellation",
    "cancelled_by",
    "driver_cancellation_reason",
    "customer_cancellation_reason",
    "ride_rating",
    "driver_rating_given",
    "customer_rating_given",
    "actual_drop_time",
    "completion_timestamp",
    "final_fare_paid",
    "feedback",
]

# ID / Metadata Columns to Exclude from Feature Matrix
METADATA_COLUMNS = [
    "booking_id",
    "ride_id",
    "customer_id",
    "driver_id",
    "user_id",
    "order_id",
]

# Standard Column Detection Equivalents
STANDARD_FIELDS = {
    "booking_id": ["booking_id", "bookingid", "trip_id", "tripid", "id", "ride_id", "order_id"],
    "booking_date": ["booking_date", "date", "trip_date", "ride_date", "order_date"],
    "booking_time": ["booking_time", "time", "trip_time", "ride_time", "order_time"],
    "vehicle_type": ["vehicle_type", "vehicle", "car_type", "cab_type", "service_type", "fleet_type", "vehicle_category"],
    "pickup_location": ["pickup_location", "pickup", "from_location", "source", "origin", "pickup_area", "pickup_point", "pickup_address"],
    "drop_location": ["drop_location", "drop", "to_location", "destination", "dest", "drop_area", "drop_point", "drop_address"],
    "ride_distance": ["ride_distance", "distance", "trip_distance", "km", "distance_km", "trip_km", "miles"],
    "booking_value": ["booking_value", "booking_amount", "fare", "amount", "price", "cost", "trip_cost", "trip_fare", "total_fare"],
    "booking_status": ["booking_status", "status", "cancelled", "is_cancelled", "cancellation_status", "trip_status", "target", "outcome"],
    "waiting_time": ["waiting_time", "wait_time", "passenger_waiting_time", "customer_wait_time", "wait_mins", "customer_wait"],
    "driver_arrival_time": ["driver_arrival_time", "eta", "arrival_time", "driver_eta", "dispatch_time", "arrival_delay", "driver_wait"],
    "cancellation_reason": ["cancellation_reason", "cancel_reason", "reason", "cancellation_cause", "why_cancelled"]
}

# Common Target Names
TARGET_COLUMN_CANDIDATES = [
    "booking_status",
    "status",
    "is_cancelled",
    "cancellation",
    "ride_status",
    "cancelled",
    "target",
    "outcome",
]

# Value mappings for Target Encoding (1 = Cancelled, 0 = Completed)
TARGET_VALUE_MAP = {
    "cancelled": 1,
    "canceled": 1,
    "cancel": 1,
    "failure": 1,
    "failed": 1,
    "yes": 1,
    "true": 1,
    "1": 1,
    1: 1,
    True: 1,
    "completed": 0,
    "complete": 0,
    "success": 0,
    "successful": 0,
    "finished": 0,
    "fulfilled": 0,
    "no": 0,
    "false": 0,
    "0": 0,
    0: 0,
    False: 0,
}

# Machine Learning Models
SUPPORTED_MODELS = [
    "Logistic Regression",
    "Decision Tree",
    "Random Forest",
    "XGBoost",
]

RANDOM_SEED = 42
TEST_SIZE = 0.20
