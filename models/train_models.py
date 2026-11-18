"""
RideGuard AI - Machine Learning Model Training Engine
Trains, tunes, and serializes Logistic Regression, Decision Tree, Random Forest, and XGBoost models.
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

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from config.settings import ARTIFACTS_DIR, RANDOM_SEED


class ModelTrainer:
    """
    Trains all four comparative algorithms with class imbalance management,
    consistent seeds, and model persistence.
    """

    def __init__(self, random_seed: int = RANDOM_SEED):
        self.random_seed = random_seed
        self.models = {}

    def get_model_definitions(self, pos_weight: float = 1.0) -> dict:
        """Instantiates the 4 algorithms with balanced weighting and hyperparameters."""
        return {
            "Logistic Regression": LogisticRegression(
                class_weight="balanced",
                max_iter=1000,
                C=1.0,
                solver="lbfgs",
                random_state=self.random_seed
            ),
            "Decision Tree": DecisionTreeClassifier(
                max_depth=7,
                min_samples_split=15,
                min_samples_leaf=8,
                class_weight="balanced",
                random_state=self.random_seed
            ),
            "Random Forest": RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=10,
                min_samples_leaf=4,
                class_weight="balanced",
                n_jobs=1,
                random_state=self.random_seed
            ),
            "XGBoost": XGBClassifier(
                n_estimators=150,
                max_depth=5,
                learning_rate=0.08,
                subsample=0.85,
                colsample_bytree=0.85,
                scale_pos_weight=max(1.0, float(pos_weight)),
                eval_metric="logloss",
                random_state=self.random_seed
            )
        }

    def train_all(self, X_train: pd.DataFrame, y_train: pd.Series) -> dict:
        """Trains all 4 models and returns trained model dictionary."""
        # Calculate positive class weight for XGBoost
        neg_count = (y_train == 0).sum()
        pos_count = (y_train == 1).sum()
        pos_weight = (neg_count / max(1, pos_count))

        model_defs = self.get_model_definitions(pos_weight=pos_weight)

        trained_models = {}
        for name, model in model_defs.items():
            model.fit(X_train, y_train)
            trained_models[name] = model

        self.models = trained_models
        return trained_models

    def save_models(self, artifacts_dir: Path = ARTIFACTS_DIR):
        """Saves all trained models to artifacts directory."""
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        for name, model in self.models.items():
            filename = name.lower().replace(" ", "_") + ".joblib"
            joblib.dump(model, artifacts_dir / filename)

    @staticmethod
    def load_models(artifacts_dir: Path = ARTIFACTS_DIR) -> dict:
        """Loads saved models from artifacts directory if available."""
        models = {}
        expected = {
            "Logistic Regression": "logistic_regression.joblib",
            "Decision Tree": "decision_tree.joblib",
            "Random Forest": "random_forest.joblib",
            "XGBoost": "xgboost.joblib"
        }
        for name, filename in expected.items():
            path = artifacts_dir / filename
            if path.exists():
                models[name] = joblib.load(path)
        return models
