"""
RideGuard AI - Exploratory Data Analysis & Analytics Engine
Generates interactive, theme-consistent Plotly visualizations for cancellation dynamics.
"""

from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from utils.helpers import apply_plotly_theme
from config.settings import THEME


class RideAnalytics:
    """Computes aggregate metrics and builds rich Plotly analytics charts."""

    def __init__(self, df: pd.DataFrame, target_col: str = "Booking_Status"):
        self.df = df.copy()
        self.target_col = target_col
        self._prepare_data()

    def _prepare_data(self):
        """Standardizes binary target column and date fields."""
        if self.target_col in self.df.columns:
            val = self.df[self.target_col]
            if pd.api.types.is_numeric_dtype(val):
                self.df["_is_cancelled"] = val.astype(int)
            else:
                self.df["_is_cancelled"] = val.astype(str).str.lower().apply(
                    lambda x: 1 if "cancel" in x or x == "1" else 0
                )
        else:
            self.df["_is_cancelled"] = 0

        self.df["_Status_Label"] = np.where(self.df["_is_cancelled"] == 1, "Cancelled", "Completed")

        # Standardize common column aliases if missing
        lower_cols = {c.lower().replace("_", "").replace(" ", "").replace("-", ""): c for c in self.df.columns}
        alias_map = {
            "Hour": ["hour", "bookinghour", "pickuphour", "dispatchhour"],
            "Booking_Value": ["bookingvalue", "fare", "fareamount", "price", "tripfare"],
            "Ride_Distance": ["ridedistance", "distance", "distancekm", "tripdistance"],
            "Waiting_Time": ["waitingtime", "wait", "waittime", "delay", "delaymins"],
            "Pickup_Location": ["pickuplocation", "pickup", "fromlocation", "source", "startlocation"],
            "Drop_Location": ["droplocation", "drop", "tolocation", "destination", "endlocation"],
            "Vehicle_Type": ["vehicletype", "vehicle", "cartype", "ridetype", "fleetcategory"],
            "Payment_Method": ["paymentmethod", "payment", "paymentmode", "paytype"]
        }
        for target_name, aliases in alias_map.items():
            if target_name not in self.df.columns:
                for a in aliases:
                    if a in lower_cols:
                        self.df[target_name] = self.df[lower_cols[a]]
                        break

        # Parse hour if available
        if "Hour" not in self.df.columns and "Booking_Time" in self.df.columns:
            try:
                self.df["Hour"] = pd.to_datetime(self.df["Booking_Time"], format="%H:%M", errors="coerce").dt.hour
            except Exception:
                pass

        # Parse date if available
        if "Booking_Date" in self.df.columns:
            try:
                self.df["_Date_Parsed"] = pd.to_datetime(self.df["Booking_Date"], errors="coerce")
            except Exception:
                pass

    def get_kpis(self) -> Dict[str, Any]:
        """Calculates core KPI figures for dashboard overview."""
        total = len(self.df)
        cancelled = int(self.df["_is_cancelled"].sum())
        completed = total - cancelled
        rate = (cancelled / total * 100.0) if total > 0 else 0.0

        avg_fare = round(float(self.df["Booking_Value"].mean()), 2) if "Booking_Value" in self.df.columns and not self.df["Booking_Value"].dropna().empty else None
        avg_dist = round(float(self.df["Ride_Distance"].mean()), 2) if "Ride_Distance" in self.df.columns and not self.df["Ride_Distance"].dropna().empty else None
        avg_wait = round(float(self.df["Waiting_Time"].mean()), 1) if "Waiting_Time" in self.df.columns and not self.df["Waiting_Time"].dropna().empty else None

        return {
            "total_bookings": total,
            "completed_rides": completed,
            "cancelled_rides": cancelled,
            "cancellation_rate": round(rate, 2),
            "avg_booking_value": avg_fare,
            "avg_ride_distance": avg_dist,
            "avg_waiting_time": avg_wait,
        }

    def plot_cancellation_donut(self) -> go.Figure:
        """Donut chart illustrating overall cancellation vs completion proportion."""
        counts = self.df["_Status_Label"].value_counts().reset_index()
        counts.columns = ["Status", "Count"]

        color_map = {
            "Completed": THEME["success"],
            "Cancelled": THEME["danger"]
        }
        colors = [color_map.get(s, THEME["accent_blue"]) for s in counts["Status"]]

        fig = go.Figure(go.Pie(
            labels=counts["Status"],
            values=counts["Count"],
            hole=0.62,
            marker=dict(colors=colors, line=dict(color=THEME["bg_primary"], width=2)),
            textinfo="label+percent",
            hoverinfo="label+value+percent"
        ))

        fig.update_layout(
            title="Overall Booking Outcome Distribution",
            showlegend=True,
            height=330
        )
        return apply_plotly_theme(fig)

    def plot_vehicle_cancellations(self) -> go.Figure:
        """Grouped bar chart showing completion vs cancellation across fleet categories."""
        if "Vehicle_Type" not in self.df.columns:
            return go.Figure()

        agg = self.df.groupby(["Vehicle_Type", "_Status_Label"]).size().unstack(fill_value=0).reset_index()

        fig = go.Figure()
        if "Completed" in agg.columns:
            fig.add_trace(go.Bar(
                name="Completed",
                x=agg["Vehicle_Type"],
                y=agg["Completed"],
                marker_color=THEME["success"]
            ))
        if "Cancelled" in agg.columns:
            fig.add_trace(go.Bar(
                name="Cancelled",
                x=agg["Vehicle_Type"],
                y=agg["Cancelled"],
                marker_color=THEME["danger"]
            ))

        fig.update_layout(
            title="Rides by Vehicle Fleet Category",
            barmode="group",
            xaxis_title="Vehicle Category",
            yaxis_title="Number of Bookings",
            height=330
        )
        return apply_plotly_theme(fig)

    def plot_hourly_trend(self) -> go.Figure:
        """Dual-axis line / bar chart showing hourly booking volume and cancellation percentage."""
        if "Hour" not in self.df.columns:
            return go.Figure()

        hourly = self.df.groupby("Hour").agg(
            total=("_is_cancelled", "count"),
            cancelled=("_is_cancelled", "sum"),
            rate=("_is_cancelled", "mean")
        ).reset_index()

        hourly["rate_pct"] = hourly["rate"] * 100.0

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=hourly["Hour"],
            y=hourly["total"],
            name="Total Bookings",
            marker_color="rgba(59, 130, 246, 0.4)",
            yaxis="y"
        ))
        fig.add_trace(go.Scatter(
            x=hourly["Hour"],
            y=hourly["rate_pct"],
            name="Cancellation Rate (%)",
            mode="lines+markers",
            line=dict(color=THEME["danger"], width=2.5),
            marker=dict(size=5, color="#F87171"),
            yaxis="y2"
        ))

        fig.update_layout(
            title="Hourly Booking Volume & Cancellation Rate (%)",
            xaxis=dict(title="Hour of Day (0 - 23)", tickmode="linear", tick0=0, dtick=2),
            yaxis=dict(title="Booking Volume", side="left"),
            yaxis2=dict(
                title="Cancellation Rate (%)",
                overlaying="y",
                side="right",
                showgrid=False
            ),
            legend=dict(x=0.02, y=0.98),
            height=330
        )
        return apply_plotly_theme(fig)

    def plot_location_analysis(self, top_n: int = 8) -> go.Figure:
        """Horizontal bar chart showing top cancellation pickup locations."""
        if "Pickup_Location" not in self.df.columns:
            return go.Figure()

        loc_df = self.df.groupby("Pickup_Location").agg(
            total=("_is_cancelled", "count"),
            cancelled=("_is_cancelled", "sum"),
            rate=("_is_cancelled", "mean")
        ).reset_index()

        loc_df = loc_df[loc_df["total"] >= 15].sort_values(by="rate", ascending=True).tail(top_n)
        loc_df["rate_pct"] = loc_df["rate"] * 100.0

        fig = go.Figure(go.Bar(
            x=loc_df["rate_pct"],
            y=loc_df["Pickup_Location"],
            orientation="h",
            marker=dict(
                color=loc_df["rate_pct"],
                colorscale=[[0, "#3B82F6"], [0.6, "#F59E0B"], [1.0, "#EF4444"]]
            ),
            hovertemplate="<b>%{y}</b><br>Cancellation Rate: %{x:.1f}%<br>Total Rides: %{customdata}<extra></extra>",
            customdata=loc_df["total"]
        ))

        fig.update_layout(
            title=f"Top Pickup Locations by Cancellation Rate (%)",
            xaxis_title="Cancellation Rate (%)",
            yaxis_title="Pickup Location",
            height=330
        )
        return apply_plotly_theme(fig)

    def plot_fare_distance_distribution(self) -> go.Figure:
        """Box plot of ride distance split by completion status."""
        if "Ride_Distance" not in self.df.columns:
            return go.Figure()

        fig = px.box(
            self.df,
            x="_Status_Label",
            y="Ride_Distance",
            color="_Status_Label",
            color_discrete_map={"Completed": THEME["success"], "Cancelled": THEME["danger"]},
            points="outliers"
        )

        fig.update_layout(
            title="Ride Distance Distribution by Outcome",
            xaxis_title="Booking Status",
            yaxis_title="Distance (km)",
            showlegend=False,
            height=360
        )
        return apply_plotly_theme(fig)

    def plot_waiting_time_impact(self) -> go.Figure:
        """Histogram showing waiting time distribution split by completed vs cancelled."""
        col = "Waiting_Time" if "Waiting_Time" in self.df.columns else "Driver_Arrival_Time"
        if col not in self.df.columns:
            return go.Figure()

        fig = px.histogram(
            self.df,
            x=col,
            color="_Status_Label",
            barmode="overlay",
            nbins=30,
            color_discrete_map={"Completed": THEME["success"], "Cancelled": THEME["danger"]},
            opacity=0.7
        )

        fig.update_layout(
            title=f"{col.replace('_', ' ')} Impact on Cancellation",
            xaxis_title=f"{col.replace('_', ' ')} (Minutes)",
            yaxis_title="Number of Rides",
            height=360
        )
        return apply_plotly_theme(fig)

    def plot_correlation_heatmap(self) -> go.Figure:
        """Correlation matrix of numerical variables."""
        num_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        clean_num = [c for c in num_cols if not c.startswith("_") and "id" not in c.lower()]

        if len(clean_num) < 2:
            return go.Figure()

        corr = self.df[clean_num].corr()

        fig = go.Figure(data=go.Heatmap(
            z=corr.values,
            x=corr.columns,
            y=corr.index,
            colorscale=[[0, "#1E3A8A"], [0.5, "#0F172A"], [1, "#06B6D4"]],
            zmin=-1, zmax=1,
            text=np.round(corr.values, 2),
            texttemplate="%{text}",
            textfont=dict(size=10, color="#FFFFFF")
        ))

        fig.update_layout(
            title="Numerical Feature Correlation Matrix",
            height=420
        )
        return apply_plotly_theme(fig)
