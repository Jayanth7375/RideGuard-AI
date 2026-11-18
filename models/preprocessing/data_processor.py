"""
RideGuard AI - Data Preprocessing Pipeline
Handles target detection, leakage prevention, imputation, encoding, scaling, and train-test splitting.
"""

from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from config.settings import (
    LEAKAGE_COLUMNS,
    METADATA_COLUMNS,
    TARGET_COLUMN_CANDIDATES,
    TARGET_VALUE_MAP,
    ARTIFACTS_DIR,
    PROCESSED_DATA_DIR,
    RANDOM_SEED,
    TEST_SIZE,
)


class DataProcessor:
    """
    Robust preprocessing pipeline designed for ride-cancellation data.
    Ensures strict data leakage prevention and provides fitted transformers for inference.
    """

    def __init__(self, target_col: str = None, target_mapping: dict = None, column_mapping: dict = None):
        self.target_col = target_col
        self.target_mapping = target_mapping or TARGET_VALUE_MAP
        self.column_mapping = column_mapping or {}
        self.preprocessor = None
        self.feature_names = []
        self.categorical_cols = []
        self.numerical_cols = []
        self.target_encoder_classes = {}
        self.fitted = False

    def detect_target_column(self, df: pd.DataFrame) -> str:
        """Automatically detects the target column from candidates or returns None."""
        if self.target_col and self.target_col in df.columns:
            return self.target_col

        lower_cols = {col.lower().strip(): col for col in df.columns}
        for candidate in TARGET_COLUMN_CANDIDATES:
            if candidate in lower_cols:
                self.target_col = lower_cols[candidate]
                return self.target_col

        # Fallback: look for column with cancellation keywords
        for col_lower, orig_col in lower_cols.items():
            if "status" in col_lower or "cancel" in col_lower:
                if col_lower not in [c.lower() for c in LEAKAGE_COLUMNS]:
                    self.target_col = orig_col
                    return self.target_col

        raise ValueError("Could not automatically identify the target variable. Please select it manually.")

    def clean_and_sanitize(self, df: pd.DataFrame) -> pd.DataFrame:
        """Removes duplicates and standardizes mapped column names."""
        cleaned = df.copy()
        if self.column_mapping:
            rename_dict = {v: k for k, v in self.column_mapping.items() if v and v in cleaned.columns and k != v}
            if rename_dict:
                cleaned = cleaned.rename(columns=rename_dict)
                if self.target_col in rename_dict:
                    self.target_col = rename_dict[self.target_col]
        # Drop strict duplicate rows
        cleaned = cleaned.drop_duplicates()
        return cleaned

    def separate_features_and_target(self, df: pd.DataFrame, target_col: str = None):
        """
        Strips leakage columns and separates X and y.
        Encodes y such that 1 = Cancelled, 0 = Completed.
        """
        if target_col is None or target_col not in df.columns:
            if self.target_col and self.target_col in df.columns:
                target_col = self.target_col
            else:
                lower_map = {c.lower(): c for c in df.columns}
                if target_col and target_col.lower() in lower_map:
                    target_col = lower_map[target_col.lower()]
                else:
                    target_col = self.detect_target_column(df)

        self.target_col = target_col
        y_raw = df[target_col]

        # Target encoding
        if pd.api.types.is_numeric_dtype(y_raw):
            # Already numeric: assume 1 is positive/cancelled
            y = y_raw.astype(int)
        else:
            y_clean = y_raw.astype(str).str.strip().str.lower()
            y = y_clean.map(lambda val: self.target_mapping.get(val, 1 if "cancel" in val else 0)).astype(int)

        # Drop Target, Metadata (IDs), and Data Leakage columns
        cols_to_drop = [target_col]
        for col in df.columns:
            col_lower = col.lower().strip()
            # Leakage checks
            if any(leakage in col_lower for leakage in LEAKAGE_COLUMNS):
                cols_to_drop.append(col)
            # Metadata checks
            elif any(meta == col_lower for meta in METADATA_COLUMNS):
                cols_to_drop.append(col)

        cols_to_drop = list(set(cols_to_drop))
        X = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

        # Drop any date or timestamp strings if present or extract hour
        X = self._extract_date_features(X)

        return X, y

    def _extract_date_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Extracts hour or day-of-week if datetime columns exist, drops raw strings."""
        X_mod = X.copy()
        for col in X.columns:
            if "date" in col.lower() and not pd.api.types.is_numeric_dtype(X_mod[col]):
                try:
                    dt_series = pd.to_datetime(X_mod[col], errors='coerce')
                    if dt_series.notnull().mean() > 0.5:
                        X_mod[f"{col}_DayOfWeek"] = dt_series.dt.dayofweek
                        X_mod[f"{col}_IsWeekend"] = (dt_series.dt.dayofweek >= 5).astype(int)
                except Exception:
                    pass
                X_mod = X_mod.drop(columns=[col])

            elif "time" in col.lower() and col.lower() not in ["driver_arrival_time", "waiting_time"]:
                try:
                    # Parse time HH:MM
                    time_series = X_mod[col].astype(str).str.extract(r"(\d{1,2}):(\d{2})")
                    if time_series[0].notnull().mean() > 0.5:
                        X_mod["Booking_Hour"] = time_series[0].astype(int)
                except Exception:
                    pass
                X_mod = X_mod.drop(columns=[col])

        return X_mod

    def fit(self, X: pd.DataFrame):
        """Identifies column types and fits the preprocessing pipeline."""
        self.numerical_cols = X.select_dtypes(include=["int64", "float64", "int32", "float32"]).columns.tolist()
        self.categorical_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

        num_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])

        cat_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ])

        transformers = []
        if self.numerical_cols:
            transformers.append(("num", num_pipeline, self.numerical_cols))
        if self.categorical_cols:
            transformers.append(("cat", cat_pipeline, self.categorical_cols))

        self.preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
        self.preprocessor.fit(X)

        # Reconstruct transformed feature names
        feature_names = []
        if self.numerical_cols:
            feature_names.extend(self.numerical_cols)
        if self.categorical_cols:
            cat_encoder = self.preprocessor.named_transformers_["cat"].named_steps["encoder"]
            cat_feature_names = cat_encoder.get_feature_names_out(self.categorical_cols).tolist()
            feature_names.extend(cat_feature_names)

        self.feature_names = feature_names
        self.fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transforms features using the fitted preprocessor with resilient column mapping."""
        if not self.fitted:
            raise RuntimeError("DataProcessor must be fitted before transforming.")

        X_mod = X.copy()

        # 1. Apply column mapping if original columns are present
        if self.column_mapping:
            rename_dict = {v: k for k, v in self.column_mapping.items() if v and v in X_mod.columns and k != v}
            if rename_dict:
                X_mod = X_mod.rename(columns=rename_dict)

        # De-duplicate columns if aliases or mappings caused duplicate names
        X_mod = X_mod.loc[:, ~X_mod.columns.duplicated(keep="last")]

        # 2. Case-insensitive and alias matching for required features
        norm_X_cols = {c.lower().replace("_", "").replace(" ", "").replace("-", ""): c for c in X_mod.columns}

        # Check and resolve numerical columns
        for col in self.numerical_cols:
            if col not in X_mod.columns:
                norm_col = col.lower().replace("_", "").replace(" ", "").replace("-", "")
                if norm_col in norm_X_cols:
                    X_mod[col] = pd.to_numeric(X_mod[norm_X_cols[norm_col]], errors="coerce")
                else:
                    # Check common aliases
                    if norm_col in ["hour", "bookinghour"]:
                        for alt in ["Hour", "Booking_Hour", "hour", "booking_hour", "pickup_hour"]:
                            if alt in X_mod.columns:
                                X_mod[col] = pd.to_numeric(X_mod[alt], errors="coerce")
                                break
                    elif norm_col in ["ridedistance", "distance"]:
                        for alt in ["Ride_Distance", "Distance", "distance_km", "trip_distance"]:
                            if alt in X_mod.columns:
                                X_mod[col] = pd.to_numeric(X_mod[alt], errors="coerce")
                                break
                    elif norm_col in ["bookingvalue", "fare", "price"]:
                        for alt in ["Booking_Value", "Fare", "fare_amount", "trip_fare", "price"]:
                            if alt in X_mod.columns:
                                X_mod[col] = pd.to_numeric(X_mod[alt], errors="coerce")
                                break
                    elif norm_col in ["driverarrivaltime", "eta"]:
                        for alt in ["Driver_Arrival_Time", "eta", "driver_eta", "arrival_time"]:
                            if alt in X_mod.columns:
                                X_mod[col] = pd.to_numeric(X_mod[alt], errors="coerce")
                                break
                    elif norm_col in ["waitingtime", "delay", "wait"]:
                        for alt in ["Waiting_Time", "delay_time", "wait_time", "rider_wait"]:
                            if alt in X_mod.columns:
                                X_mod[col] = pd.to_numeric(X_mod[alt], errors="coerce")
                                break
                    elif norm_col in ["customerrating", "rating"]:
                        for alt in ["Customer_Rating", "Driver_Rating", "Passenger_Rating", "rating"]:
                            if alt in X_mod.columns:
                                X_mod[col] = pd.to_numeric(X_mod[alt], errors="coerce")
                                break

        # Check and resolve categorical columns
        for col in self.categorical_cols:
            if col not in X_mod.columns:
                norm_col = col.lower().replace("_", "").replace(" ", "").replace("-", "")
                if norm_col in norm_X_cols:
                    X_mod[col] = X_mod[norm_X_cols[norm_col]]
                else:
                    if norm_col in ["vehicletype", "vehicle"]:
                        for alt in ["Vehicle_Type", "vehicle", "car_type", "ride_type"]:
                            if alt in X_mod.columns:
                                X_mod[col] = X_mod[alt]
                                break
                    elif norm_col in ["pickuplocation", "pickup"]:
                        for alt in ["Pickup_Location", "pickup", "from_location", "source"]:
                            if alt in X_mod.columns:
                                X_mod[col] = X_mod[alt]
                                break
                    elif norm_col in ["droplocation", "drop", "destination"]:
                        for alt in ["Drop_Location", "drop", "to_location", "destination"]:
                            if alt in X_mod.columns:
                                X_mod[col] = X_mod[alt]
                                break
                    elif norm_col in ["paymentmethod", "payment"]:
                        for alt in ["Payment_Method", "payment_mode", "payment"]:
                            if alt in X_mod.columns:
                                X_mod[col] = X_mod[alt]
                                break

        # 3. Extract date features
        X_mod = self._extract_date_features(X_mod)

        # De-duplicate columns before final alignment
        X_mod = X_mod.loc[:, ~X_mod.columns.duplicated(keep="last")]

        # 4. Fill any remaining unsupplied columns safely
        for col in self.numerical_cols:
            if col not in X_mod.columns:
                X_mod[col] = np.nan
            else:
                series_data = X_mod[col].iloc[:, 0] if isinstance(X_mod[col], pd.DataFrame) else X_mod[col]
                X_mod[col] = pd.to_numeric(series_data, errors="coerce")

        for col in self.categorical_cols:
            if col not in X_mod.columns:
                X_mod[col] = "Unknown"
            else:
                series_data = X_mod[col].iloc[:, 0] if isinstance(X_mod[col], pd.DataFrame) else X_mod[col]
                X_mod[col] = series_data.astype(str)

        X_transformed_arr = self.preprocessor.transform(X_mod)
        return pd.DataFrame(X_transformed_arr, columns=self.feature_names, index=X_mod.index)

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Fits and transforms feature dataframe."""
        self.fit(X)
        return self.transform(X)

    def prepare_train_test(self, df: pd.DataFrame, target_col: str = None, test_size: float = TEST_SIZE):
        """Full preparation pipeline: sanitization, leakage removal, stratified train/test split, and scaling."""
        if target_col:
            self.target_col = target_col
        df_clean = self.clean_and_sanitize(df)
        X, y = self.separate_features_and_target(df_clean, target_col=self.target_col)

        # Stratified train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=RANDOM_SEED,
            stratify=y if len(np.unique(y)) > 1 else None
        )

        self.fit(X_train)
        X_train_proc = self.transform(X_train)
        X_test_proc = self.transform(X_test)

        return X_train_proc, X_test_proc, y_train, y_test, X_train, X_test

    def save(self, filepath: Path = None):
        """Serializes preprocessor state to disk."""
        if filepath is None:
            ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
            filepath = ARTIFACTS_DIR / "preprocessor.joblib"
        joblib.dump(self, filepath)

    @staticmethod
    def load(filepath: Path = None):
        """Loads serialized preprocessor from disk."""
        if filepath is None:
            filepath = ARTIFACTS_DIR / "preprocessor.joblib"
        return joblib.load(filepath)
