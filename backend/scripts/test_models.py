# scripts/test_models.py
# Confirms both trained models load and predict correctly
# Run: python scripts/test_models.py

import sys, joblib, numpy as np
sys.path.insert(0, ".")

rf  = joblib.load("backend/models/rf_model.pkl")
xgb = joblib.load("backend/models/xgb_model.pkl")

LABEL = {0: "LOW", 1: "MEDIUM", 2: "HIGH"}
FEATURES = ["rain_score_0_10","river_score_0_10","tide_score_0_10",
            "elevation_factor","twi","dbi","drain_efficiency","slope_deg"]

test_cases = [
    {
        "name": "Chennai Velachery — heavy rain + high river + high tide",
        "x": [8.0, 8.5, 9.0, 6.8, 6.2, 1.8, 0.1, 0.5]
    },
    {
        "name": "Poonamallee — light rain, high elevation",
        "x": [1.0, 2.0, 3.0, 0.7, 4.5, 0.3, 0.9, 1.8]
    },
    {
        "name": "Nagapattinam coast — cyclone scenario",
        "x": [9.5, 9.0, 9.5, 8.2, 7.1, 2.2, 0.05, 0.2]
    },
    {
        "name": "Madurai Vaigai — moderate rain",
        "x": [5.0, 5.5, 4.0, 5.2, 5.8, 1.0, 0.5, 0.6]
    },
]

print("=" * 55)
print("FloodVision AI — Model Prediction Test")
print("=" * 55)

for case in test_cases:
    x = np.array([case["x"]])
    rf_label   = LABEL[rf.predict(x)[0]]
    rf_probs   = rf.predict_proba(x)[0]
    xgb_depth  = xgb.predict(x)[0]

    # Ensemble decision
    high_prob = rf_probs[2]
    if high_prob >= 0.65 or xgb_depth >= 30:
        final = "HIGH (RED)"
    elif high_prob >= 0.35 or xgb_depth >= 10:
        final = "MEDIUM (YELLOW)"
    else:
        final = "LOW (GREEN)"

    print(f"\nTest: {case['name']}")
    print(f"  RF prediction:    {rf_label}")
    print(f"  RF confidence:    LOW={rf_probs[0]:.0%}  "
          f"MED={rf_probs[1]:.0%}  HIGH={rf_probs[2]:.0%}")
    print(f"  XGBoost depth:    {xgb_depth:.1f} cm")
    print(f"  FINAL (ensemble): {final}")