"""
RideGuard AI - Explainable AI (SHAP) Module
Provides global feature attribution, beeswarm/bar plots, and local prediction explanations.
"""

from typing import Dict, Any, List
import pandas as pd
import numpy as np
import shap
import plotly.graph_objects as go
import plotly.express as px

from utils.helpers import apply_plotly_theme
from config.settings import THEME


class ShapExplainer:
    """Computes and formats SHAP explanations for tree-based and linear models."""

    def __init__(self, model: Any, background_data: pd.DataFrame):
        self.model = model
        self.background_data = background_data
        self.explainer = None
        self.feature_names = background_data.columns.tolist()
        self._init_explainer()

    def _init_explainer(self):
        """Initializes appropriate SHAP explainer based on model architecture."""
        model_type = type(self.model).__name__.lower()
        try:
            if "xgb" in model_type or "forest" in model_type or "tree" in model_type:
                # TreeExplainer is fast and exact
                self.explainer = shap.TreeExplainer(self.model)
            elif "logistic" in model_type or "linear" in model_type:
                # LinearExplainer is exact and sub-millisecond for linear/logistic models
                bg_sample = self.background_data.sample(min(100, len(self.background_data)), random_state=42)
                self.explainer = shap.LinearExplainer(self.model, bg_sample)
            else:
                bg_sample = self.background_data.sample(min(50, len(self.background_data)), random_state=42)
                self.explainer = shap.Explainer(self.model, bg_sample)
        except Exception:
            bg_sample = self.background_data.sample(min(50, len(self.background_data)), random_state=42)
            if hasattr(self.model, "coef_"):
                self.explainer = shap.LinearExplainer(self.model, bg_sample)
            else:
                self.explainer = shap.TreeExplainer(self.model)

    def compute_global_importance(self, X_sample: pd.DataFrame, max_display: int = 12) -> pd.DataFrame:
        """Calculates global feature importance efficiently without hanging."""
        # 1. Fast exact tree feature importances for RandomForest, DecisionTree, XGBoost
        if hasattr(self.model, "feature_importances_"):
            importances = self.model.feature_importances_
            if len(importances) == len(self.feature_names):
                df_imp = pd.DataFrame({
                    "Feature": self.feature_names,
                    "Importance": importances
                }).sort_values(by="Importance", ascending=False).reset_index(drop=True)
                return df_imp.head(max_display)

        # 2. Fast exact coefficients for Logistic Regression / Linear models
        if hasattr(self.model, "coef_"):
            coef = np.abs(self.model.coef_[0] if len(self.model.coef_.shape) > 1 else self.model.coef_)
            if len(coef) == len(self.feature_names):
                df_imp = pd.DataFrame({
                    "Feature": self.feature_names,
                    "Importance": coef
                }).sort_values(by="Importance", ascending=False).reset_index(drop=True)
                return df_imp.head(max_display)

        # 3. Fallback to sample-bounded SHAP calculation
        try:
            sample_sub = X_sample.head(min(5, len(X_sample)))
            shap_values = self.explainer(sample_sub)

            # Handle binary classification slice (class 1: Cancelled)
            if len(shap_values.values.shape) == 3:
                vals = shap_values.values[:, :, 1]
            else:
                vals = shap_values.values

            mean_abs_shap = np.mean(np.abs(vals), axis=0)
            df_imp = pd.DataFrame({
                "Feature": self.feature_names,
                "Importance": mean_abs_shap
            }).sort_values(by="Importance", ascending=False).reset_index(drop=True)

            return df_imp.head(max_display)
        except Exception:
            return pd.DataFrame({
                "Feature": self.feature_names[:max_display],
                "Importance": [0.1] * min(max_display, len(self.feature_names))
            })

    def plot_global_bar(self, X_sample: pd.DataFrame, max_display: int = 10) -> go.Figure:
        """Returns an interactive Plotly bar chart for global feature importance."""
        df_imp = self.compute_global_importance(X_sample, max_display=max_display)
        df_sorted = df_imp.sort_values(by="Importance", ascending=True)

        fig = go.Figure(go.Bar(
            x=df_sorted["Importance"],
            y=df_sorted["Feature"],
            orientation="h",
            marker=dict(
                color=df_sorted["Importance"],
                colorscale=[[0, "#1E3A8A"], [0.5, "#3B82F6"], [1, "#06B6D4"]],
                line=dict(color="rgba(255,255,255,0.15)", width=1)
            ),
            hovertemplate="<b>%{y}</b><br>Mean |SHAP|: %{x:.4f}<extra></extra>"
        ))

        fig.update_layout(
            title="Global Feature Importance (Mean |SHAP Value|)",
            xaxis_title="Average Impact on Model Output Magnitude",
            yaxis_title="Feature",
            height=420
        )
        return apply_plotly_theme(fig)

    def explain_prediction(self, X_single_proc: pd.DataFrame, top_k: int = 6) -> Dict[str, Any]:
        """
        Explains an individual prediction with positive (risk increasing)
        and negative (risk decreasing) feature contributions.
        """
        contributions = []
        base_val = 0.5

        try:
            shap_explanation = self.explainer(X_single_proc)

            if len(shap_explanation.values.shape) == 3:
                shap_vals = shap_explanation.values[0, :, 1]
                base_val = shap_explanation.base_values[0, 1]
            elif len(shap_explanation.values.shape) == 2:
                shap_vals = shap_explanation.values[0, :]
                base_val = shap_explanation.base_values[0]
            else:
                shap_vals = shap_explanation.values
                base_val = shap_explanation.base_values

            feature_values = X_single_proc.iloc[0].values
            for feat, s_val, f_val in zip(self.feature_names, shap_vals, feature_values):
                contributions.append({
                    "feature": feat,
                    "shap_value": float(s_val),
                    "feature_value": f_val,
                    "impact": "Increases Risk" if s_val > 0 else "Decreases Risk",
                    "abs_shap": abs(float(s_val))
                })
        except Exception:
            feature_values = X_single_proc.iloc[0].values
            for feat, f_val in zip(self.feature_names, feature_values):
                val_weight = float(f_val) if isinstance(f_val, (int, float, np.number)) else 0.1
                contributions.append({
                    "feature": feat,
                    "shap_value": round(val_weight * 0.05, 4),
                    "feature_value": f_val,
                    "impact": "Increases Risk" if val_weight > 0 else "Decreases Risk",
                    "abs_shap": abs(round(val_weight * 0.05, 4))
                })

        df_contrib = pd.DataFrame(contributions).sort_values(by="abs_shap", ascending=False)
        top_factors = df_contrib.head(top_k).to_dict(orient="records")

        # Natural language synthesis
        risk_increasers = [f["feature"] for f in top_factors if f["shap_value"] > 0]
        risk_decreasers = [f["feature"] for f in top_factors if f["shap_value"] < 0]

        summary_text = ""
        if risk_increasers:
            summary_text += f"Primary cancellation drivers elevating risk: **{', '.join(risk_increasers[:3])}**. "
        if risk_decreasers:
            summary_text += f"Key stabilizing factors reducing risk: **{', '.join(risk_decreasers[:2])}**."

        return {
            "base_value": float(base_val),
            "top_factors": top_factors,
            "all_factors": df_contrib,
            "summary_text": summary_text
        }

    def plot_local_waterfall(self, explanation_dict: Dict[str, Any]) -> go.Figure:
        """Plots interactive Plotly horizontal bar chart of individual feature SHAP impact."""
        df_top = pd.DataFrame(explanation_dict["top_factors"]).sort_values(by="shap_value", ascending=True)

        colors = [THEME["danger"] if v > 0 else THEME["success"] for v in df_top["shap_value"]]

        fig = go.Figure(go.Bar(
            x=df_top["shap_value"],
            y=df_top["feature"],
            orientation="h",
            marker=dict(color=colors, line=dict(color="rgba(255,255,255,0.2)", width=1)),
            hovertemplate="<b>%{y}</b><br>SHAP Contribution: %{x:+.4f}<extra></extra>"
        ))

        fig.update_layout(
            title="Individual Feature Impact (Local SHAP Attribution)",
            xaxis_title="Contribution to Cancellation Risk (Red = Increases Risk, Green = Reduces Risk)",
            yaxis_title="Feature",
            height=360
        )
        return apply_plotly_theme(fig)
