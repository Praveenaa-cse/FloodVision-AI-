# scripts/train_models.py
#
# PURPOSE: Train Random Forest + XGBoost + Ensemble for FloodVision AI
#
# WHAT IT TRAINS:
#   1. Random Forest  → classifies flood risk (LOW/MEDIUM/HIGH)
#   2. XGBoost        → regresses water depth in cm
#   3. Ensemble logic → combines both for final prediction
#
# WHERE TO RUN:
#   From root floodvision-ai/ folder:
#   python scripts/train_models.py
#
# OUTPUT:
#   backend/models/rf_model.pkl
#   backend/models/xgb_model.pkl

import os
import sys
import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble        import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics         import (classification_report,
                                      confusion_matrix,
                                      mean_absolute_error,
                                      r2_score)
from sklearn.preprocessing   import StandardScaler
import xgboost as xgb

sys.path.insert(0, ".")

print("=" * 60)
print("FloodVision AI — Training ML Models")
print("Coverage: All Tamil Nadu regions + Coastal areas")
print("=" * 60)

# ─────────────────────────────────────────────────────────────
# STEP 1: Load training data
# ─────────────────────────────────────────────────────────────

DATA_PATH = "backend/data/training_ready.csv"

if not os.path.exists(DATA_PATH):
    print(f"ERROR: {DATA_PATH} not found.")
    print("Run Day 2 first: python scripts/build_features.py")
    sys.exit(1)

df = pd.read_csv(DATA_PATH)
print(f"\nLoaded {len(df)} training rows")
print(f"Regions covered: {df['region'].nunique()}")
print(f"Label distribution:\n{df['flood_label_text'].value_counts().to_string()}\n")

# ─────────────────────────────────────────────────────────────
# STEP 2: Define features
#
# These 8 features are what both models use.
# They are the same features computed in production
# when a real user sends their location.
# ─────────────────────────────────────────────────────────────

# Features the model uses to predict
FEATURES = [
    "rain_score_0_10",    # how much rain (scaled 0-10)
    "river_score_0_10",   # river level (scaled 0-10)
    "tide_score_0_10",    # tide height (scaled 0-10)
    "elevation_factor",   # 10 - elevation (lower = worse)
    "twi",                # topographic wetness index
    "dbi",                # drainage blockage index
    "drain_efficiency",   # how well drains are working
    "slope_deg",          # slope of the terrain
]

X = df[FEATURES].values
y_class = df["flood_label"].values       # for Random Forest (0/1/2)
y_depth = df["water_depth_cm"].values    # for XGBoost (depth in cm)

print(f"Features used: {FEATURES}")
print(f"Training samples: {len(X)}\n")

# ─────────────────────────────────────────────────────────────
# STEP 3: Split into train and test sets
# 80% training, 20% testing
# stratify=y_class ensures equal proportion of labels in both splits
# ─────────────────────────────────────────────────────────────

X_train, X_test, yc_train, yc_test, yd_train, yd_test = \
    train_test_split(X, y_class, y_depth,
                     test_size=0.20,
                     random_state=42,
                     stratify=y_class)

print(f"Training set:  {len(X_train)} rows")
print(f"Test set:      {len(X_test)} rows\n")

os.makedirs("backend/models", exist_ok=True)

# ─────────────────────────────────────────────────────────────
# STEP 4: Train Random Forest Classifier
#
# What it does: classifies flood risk as LOW(0), MEDIUM(1), HIGH(2)
# Gives a probability for each class — we use the HIGH probability
# as the main flood risk signal.
#
# n_estimators=200: 200 decision trees vote together
# max_depth=10: each tree can be up to 10 levels deep
# class_weight="balanced": handles unequal class counts
# ─────────────────────────────────────────────────────────────

print("=" * 40)
print("Training Random Forest Classifier...")
print("=" * 40)

rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_leaf=3,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1   # use all CPU cores
)
rf.fit(X_train, yc_train)

# Evaluate
rf_preds = rf.predict(X_test)
print("\nRandom Forest — Classification Report:")
print(classification_report(yc_test, rf_preds,
      target_names=["LOW", "MEDIUM", "HIGH"]))

print("Confusion Matrix (rows=actual, cols=predicted):")
print("           LOW  MED  HIGH")
cm = confusion_matrix(yc_test, rf_preds)
for i, label in enumerate(["LOW   ", "MEDIUM", "HIGH  "]):
    print(f"  {label}: {cm[i]}")

# Cross-validation score
cv_scores = cross_val_score(rf, X, y_class, cv=5, scoring="accuracy")
print(f"\nCross-validation accuracy: "
      f"{cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")

# Feature importance
print("\nFeature importance (which input matters most):")
importances = rf.feature_importances_
for feat, imp in sorted(zip(FEATURES, importances),
                         key=lambda x: -x[1]):
    bar = "█" * int(imp * 40)
    print(f"  {feat:<22} {imp:.4f}  {bar}")

# Save RF model
joblib.dump(rf, "backend/models/rf_model.pkl")
print("\nRandom Forest saved → backend/models/rf_model.pkl")


# ─────────────────────────────────────────────────────────────
# STEP 5: Train XGBoost Regressor
#
# What it does: predicts exact water depth in cm
# This is used to determine RED/YELLOW/GREEN threshold
#
# n_estimators=300: 300 boosting rounds
# learning_rate=0.05: small steps to avoid overfitting
# max_depth=6: depth of each tree
# ─────────────────────────────────────────────────────────────

print("\n" + "=" * 40)
print("Training XGBoost Regressor...")
print("=" * 40)

xgb_model = xgb.XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="reg:squarederror",
    eval_metric="rmse",
    early_stopping_rounds=20,
    random_state=42,
    verbosity=0
)

xgb_model.fit(
    X_train, yd_train,
    eval_set=[(X_test, yd_test)],
    verbose=False
)

# Evaluate
xgb_preds = xgb_model.predict(X_test)
mae  = mean_absolute_error(yd_test, xgb_preds)
r2   = r2_score(yd_test, xgb_preds)
print(f"\nXGBoost — Depth Prediction:")
print(f"  Mean Absolute Error: {mae:.2f} cm")
print(f"  R² Score:            {r2:.4f}")
print(f"  (R² of 1.0 = perfect, 0.0 = random)")

# Show some predictions vs actual
print("\nSample predictions (actual vs predicted depth):")
print(f"  {'Actual cm':>10}  {'Predicted cm':>12}  {'Error cm':>10}")
for i in range(min(10, len(yd_test))):
    err = abs(yd_test[i] - xgb_preds[i])
    print(f"  {yd_test[i]:>10.1f}  {xgb_preds[i]:>12.1f}  {err:>10.1f}")

# Feature importance from XGBoost
print("\nXGBoost Feature importance:")
xgb_imp = xgb_model.feature_importances_
for feat, imp in sorted(zip(FEATURES, xgb_imp),
                          key=lambda x: -x[1]):
    bar = "█" * int(imp * 40)
    print(f"  {feat:<22} {imp:.4f}  {bar}")

# Save XGBoost model
joblib.dump(xgb_model, "backend/models/xgb_model.pkl")
print("\nXGBoost saved → backend/models/xgb_model.pkl")


# ─────────────────────────────────────────────────────────────
# STEP 6: Test the Ensemble logic
#
# In production (main.py), the ensemble works like this:
#
#   rf_prob    = random forest HIGH probability (0 to 1)
#   xgb_depth  = xgboost predicted depth (cm)
#   formula_score = formula-based score
#
#   final_depth = 0.4 * xgb_depth + 0.6 * formula_depth
#   if rf_prob > 0.7 AND final_depth > 30 → HIGH (RED)
#   elif rf_prob > 0.4 OR final_depth > 10  → MEDIUM (YELLOW)
#   else                                   → LOW (GREEN)
# ─────────────────────────────────────────────────────────────

print("\n" + "=" * 40)
print("Testing Ensemble (RF + XGBoost combined)...")
print("=" * 40)

# Get RF probabilities for HIGH class on test set
rf_probs  = rf.predict_proba(X_test)[:, 2]  # column 2 = HIGH
xgb_depth = xgb_model.predict(X_test)

# Ensemble classification
ensemble_preds = []
for prob, depth in zip(rf_probs, xgb_depth):
    if prob >= 0.65 or depth >= 30:
        ensemble_preds.append(2)   # HIGH
    elif prob >= 0.35 or depth >= 10:
        ensemble_preds.append(1)   # MEDIUM
    else:
        ensemble_preds.append(0)   # LOW

ensemble_preds = np.array(ensemble_preds)
print("\nEnsemble (RF + XGBoost) — Classification Report:")
print(classification_report(yc_test, ensemble_preds,
      target_names=["LOW", "MEDIUM", "HIGH"]))


# ─────────────────────────────────────────────────────────────
# STEP 7: Final summary
# ─────────────────────────────────────────────────────────────

print("=" * 60)
print("Day 3 Complete — Models Trained and Saved")
print("=" * 60)
print("\nFiles saved:")
print("  backend/models/rf_model.pkl   ← Random Forest")
print("  backend/models/xgb_model.pkl  ← XGBoost")
print("\nThese models cover the entire Tamil Nadu region:")
print(f"  {df['region'].nunique()} regions including coastal + inland")
print(f"  {len(df)} training samples across all flood scenarios")
print("\nNext: Day 4 → python -m uvicorn backend.main:app --reload")