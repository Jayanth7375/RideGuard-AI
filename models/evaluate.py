"""
RideGuard AI - Model Evaluation & Comparison Engine
Computes multi-dimensional classification metrics, ROC curves, confusion matrices, and selects optimal models.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
    brier_score_loss,
)
from sklearn.calibration import calibration_curve


class ModelEvaluator:
    """Evaluates and benchmarks multiple ML models for ride cancellation."""

    def __init__(self, models: dict, X_test: pd.DataFrame, y_test: pd.Series):
        self.models = models
        self.X_test = X_test
        self.y_test = y_test
        self.results = {}
        self.curves = {}
        self.confusion_matrices = {}

    def evaluate_all(self) -> pd.DataFrame:
        """Evaluates every model across standard metrics."""
        summary_rows = []

        for name, model in self.models.items():
            y_pred = model.predict(self.X_test)
            if hasattr(model, "predict_proba"):
                y_prob = model.predict_proba(self.X_test)[:, 1]
            else:
                y_prob = y_pred

            acc = accuracy_score(self.y_test, y_pred)
            prec = precision_score(self.y_test, y_pred, zero_division=0)
            rec = recall_score(self.y_test, y_pred, zero_division=0)
            f1 = f1_score(self.y_test, y_pred, zero_division=0)
            try:
                auc = roc_auc_score(self.y_test, y_prob)
            except Exception:
                auc = 0.5
            brier = brier_score_loss(self.y_test, y_prob)

            cm = confusion_matrix(self.y_test, y_pred)
            self.confusion_matrices[name] = cm

            fpr, tpr, _ = roc_curve(self.y_test, y_prob)
            prob_true, prob_pred = calibration_curve(self.y_test, y_prob, n_bins=10)

            self.curves[name] = {
                "fpr": fpr,
                "tpr": tpr,
                "cal_true": prob_true,
                "cal_pred": prob_pred,
                "auc": auc
            }

            self.results[name] = {
                "Accuracy": acc,
                "Precision": prec,
                "Recall": rec,
                "F1 Score": f1,
                "ROC-AUC": auc,
                "Brier Score": brier
            }

            summary_rows.append({
                "Model": name,
                "Accuracy": acc,
                "Precision": prec,
                "Recall": rec,
                "F1 Score": f1,
                "ROC-AUC": auc,
                "Brier Score": brier
            })

        df_summary = pd.DataFrame(summary_rows)
        # Rank primarily on F1 Score and ROC-AUC
        df_summary["Composite Rank"] = (df_summary["F1 Score"] * 0.5) + (df_summary["ROC-AUC"] * 0.5)
        df_summary = df_summary.sort_values(by="Composite Rank", ascending=False).reset_index(drop=True)
        return df_summary

    def get_best_model_name(self) -> str:
        """Selects optimal model balancing Precision, Recall (F1), and Discriminative Power (AUC)."""
        df = self.evaluate_all()
        return df.iloc[0]["Model"]
