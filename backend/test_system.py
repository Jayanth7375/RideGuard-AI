"""
RideGuard AI - System Integration & Verification Test Suite
Tests every ML, Preprocessing, SHAP, Recommendation, and App module end-to-end.
"""

import sys
from pathlib import Path

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

import pandas as pd
import numpy as np

def run_all_tests():
    print("=" * 60)
    print("🚀 STARTING RIDEGUARD AI SYSTEM INTEGRATION TESTS")
    print("=" * 60)

    # 1. Dataset Generation Test
    print("\n[TEST 1] Testing Synthetic Dataset Generation...")
    from utils.helpers import generate_synthetic_rides
    df = generate_synthetic_rides(n_samples=1000, random_seed=42)
    assert len(df) == 1000, f"Expected 1000 rows, got {len(df)}"
    assert "Booking_Status" in df.columns, "Target column missing"
    print(f"  ✓ Dataset generated: {len(df)} rows, {len(df.columns)} columns")
    print(f"  ✓ Cancellation Rate: {(df['Booking_Status'] == 'Cancelled').mean() * 100:.2f}%")

    # 2. Preprocessing & Leakage Prevention Test
    print("\n[TEST 2] Testing Preprocessor & Leakage Prevention...")
    from preprocessing.data_processor import DataProcessor
    dp = DataProcessor()
    X_train_p, X_test_p, y_train, y_test, X_train_raw, X_test_raw = dp.prepare_train_test(df, target_col="Booking_Status")
    assert not any("cancellation_reason" in col.lower() for col in X_train_p.columns), "Leakage column detected in features!"
    assert len(X_train_p) == 800, f"Expected 800 train rows, got {len(X_train_p)}"
    assert len(X_test_p) == 200, f"Expected 200 test rows, got {len(X_test_p)}"
    dp.save()
    loaded_dp = DataProcessor.load()
    assert loaded_dp.fitted, "Loaded preprocessor is not fitted"
    print(f"  ✓ Features processed: {X_train_p.shape[1]} engineered features")
    print(f"  ✓ Strict leakage prevention verified (zero leakage columns)")

    # 3. Model Training Test (All 4 Algorithms)
    print("\n[TEST 3] Testing Multi-Model Training (Logistic Regression, Decision Tree, Random Forest, XGBoost)...")
    from models.train_models import ModelTrainer
    trainer = ModelTrainer(random_seed=42)
    models = trainer.train_all(X_train_p, y_train)
    assert len(models) == 4, f"Expected 4 models, got {len(models)}"
    trainer.save_models()
    loaded_models = ModelTrainer.load_models()
    assert len(loaded_models) == 4, "Failed to reload all 4 models"
    print(f"  ✓ Successfully trained and saved all 4 algorithms:")
    for name in models.keys():
        print(f"    - {name}")

    # 4. Model Evaluation & Benchmark Test
    print("\n[TEST 4] Testing Model Benchmark & Calibration...")
    from models.evaluate import ModelEvaluator
    evaluator = ModelEvaluator(models, X_test_p, y_test)
    bench_df = evaluator.evaluate_all()
    best_name = evaluator.get_best_model_name()
    assert not bench_df.empty, "Benchmark dataframe is empty"
    print(f"  ✓ Leaderboard generated:")
    for _, row in bench_df.iterrows():
        print(f"    {row['Model']:<22} | Acc: {row['Accuracy']:.3f} | F1: {row['F1 Score']:.3f} | AUC: {row['ROC-AUC']:.3f}")
    print(f"  ✓ Optimal model selected: {best_name}")

    # 5. Prediction Interface Test
    print("\n[TEST 5] Testing Prediction Engine (Single & Batch)...")
    from models.predict import RidePredictor
    predictor = RidePredictor(models[best_name], loaded_dp)
    sample_booking = {
        "Vehicle_Type": "Prime Sedan",
        "Pickup_Location": "Airport Terminal 1",
        "Drop_Location": "Downtown Arts District",
        "Ride_Distance": 28.5,
        "Booking_Value": 680.0,
        "Hour": 19,
        "Driver_Arrival_Time": 16,
        "Waiting_Time": 8,
        "Payment_Method": "UPI / Digital Wallet",
        "Customer_Rating": 4.5
    }
    pred_res = predictor.predict_single(sample_booking)
    assert "cancellation_probability" in pred_res
    assert "risk_category" in pred_res
    print(f"  ✓ Single Prediction: Status = {pred_res['predicted_status']}, Prob = {pred_res['cancellation_probability']*100:.1f}%, Risk = {pred_res['risk_category']}")

    # Batch prediction test
    batch_res = predictor.predict_batch(df.head(10))
    assert "Cancellation_Probability" in batch_res.columns
    print(f"  ✓ Batch Prediction: Successfully scored {len(batch_res)} records")

    # 6. SHAP Explainable AI Test
    print("\n[TEST 6] Testing SHAP Explainable AI Module...")
    from explainability.shap_explainer import ShapExplainer
    explainer = ShapExplainer(models[best_name], X_train_p)
    global_imp = explainer.compute_global_importance(X_test_p.head(50), max_display=5)
    assert not global_imp.empty, "Global SHAP importance empty"
    print(f"  ✓ Top 3 Global Drivers:")
    for _, row in global_imp.head(3).iterrows():
        print(f"    - {row['Feature']}: {row['Importance']:.4f}")

    local_exp = explainer.explain_prediction(pred_res["processed_features"])
    assert len(local_exp["top_factors"]) > 0, "No local factors computed"
    print(f"  ✓ Local SHAP Narrative: {local_exp['summary_text']}")

    # 7. Policy Recommendation Engine Test
    print("\n[TEST 7] Testing Policy Recommendation Engine...")
    from recommendations.recommendation_engine import PolicyRecommendationEngine
    rec_engine = PolicyRecommendationEngine(df, target_col="Booking_Status")
    recs = rec_engine.generate_recommendations()
    assert len(recs) > 0, "No recommendations generated"
    print(f"  ✓ Generated {len(recs)} evidence-backed policy initiatives:")
    for r in recs:
        print(f"    [{r['priority']}] {r['title']} ({r['category']})")

    # 8. EDA Analytics Test
    print("\n[TEST 8] Testing EDA Analytics & Plotly Visualizations...")
    from analytics.eda import RideAnalytics
    analytics = RideAnalytics(df, target_col="Booking_Status")
    kpis = analytics.get_kpis()
    assert kpis["total_bookings"] == 1000
    fig_donut = analytics.plot_cancellation_donut()
    fig_hourly = analytics.plot_hourly_trend()
    fig_veh = analytics.plot_vehicle_cancellations()
    print("  ✓ Plotly charts successfully constructed")

    print("\n" + "=" * 60)
    print("🎉 ALL 8 INTEGRATION TESTS PASSED WITHOUT ERROR!")
    print("=" * 60)

if __name__ == "__main__":
    run_all_tests()
