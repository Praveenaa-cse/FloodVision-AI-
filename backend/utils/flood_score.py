# backend/utils/flood_score.py
#
# WHAT THIS DOES:
#   Takes real rainfall, river, tide, elevation data
#   Runs your flood formula
#   Runs Random Forest + XGBoost models
#   Combines them into ONE final risk level
#
# WHY ENSEMBLE (combining models)?
#   Formula alone = good but misses complex patterns
#   RF alone = good classifier but no depth info
#   XGBoost alone = good depth predictor but no probability
#   All three together = much more accurate

import numpy as np
import joblib
import os

MODEL_RF_PATH  = "backend/models/rf_model.pkl"
MODEL_XGB_PATH = "backend/models/xgb_model.pkl"

# Load models once when the file is imported (not on every request)
_rf_model  = None
_xgb_model = None


def _load_models():
    """Loads both ML models from disk. Called once."""
    global _rf_model, _xgb_model
    if _rf_model is None:
        if os.path.exists(MODEL_RF_PATH):
            _rf_model = joblib.load(MODEL_RF_PATH)
            print("RF model loaded")
        else:
            print(f"WARNING: {MODEL_RF_PATH} not found")
    if _xgb_model is None:
        if os.path.exists(MODEL_XGB_PATH):
            _xgb_model = joblib.load(MODEL_XGB_PATH)
            print("XGBoost model loaded")
        else:
            print(f"WARNING: {MODEL_XGB_PATH} not found")


def predict_flood_risk(
    rain_mm_hr:     float,
    river_discharge_m3s: float,
    tide_height_m:  float,
    elevation_m:    float,
    slope_deg:      float,
    danger_m3s:     float = 500.0,
    mhwl_m:         float = 1.40,
    drain_capacity_m3s: float = 0.5,
    area_km2:       float = 1.0,
    surface:        str   = "urban"
) -> dict:
    """
    MAIN FUNCTION — call this from your API endpoint.

    Give it real data → it returns flood risk + depth + warning.

    Parameters:
      rain_mm_hr         = mm per hour from OpenWeather (real)
      river_discharge_m3s= m3/s from Open-Meteo GloFAS (real)
      tide_height_m      = metres from tide calculator (real)
      elevation_m        = metres above sea level from DEM
      slope_deg          = terrain slope in degrees from DEM
      danger_m3s         = river discharge at which flooding starts
      mhwl_m             = Mean High Water Level for this coast
      drain_capacity_m3s = how fast drains remove water
      area_km2           = catchment area size
      surface            = "paved" / "urban" / "suburban" / "green"
    """
    _load_models()

    # ── STEP 1: Compute derived features ─────────────────────

    # Topographic Wetness Index
    # How naturally prone to waterlogging is this spot?
    slope_rad = np.radians(max(slope_deg, 0.01))
    contrib   = max(elevation_m * 15, 1.0)
    twi       = float(np.log(contrib / np.tan(slope_rad)))

    # Drainage Blockage Index
    # Are the drains blocked by high river + high tide?
    river_ratio = river_discharge_m3s / max(danger_m3s, 1)
    tide_ratio  = tide_height_m / max(mhwl_m, 0.1)
    dbi         = min(float(river_ratio + tide_ratio), 3.0)

    # Drain efficiency: drops as DBI rises
    drain_eff = max(0.05, 1.0 - (dbi * 0.6))

    # Runoff using Rational Method: Q = C × i × A
    c_map = {"paved":0.9,"urban":0.75,"suburban":0.55,"green":0.25}
    C     = c_map.get(surface, 0.75)
    runoff_m3s = C * (rain_mm_hr / 1000) * area_km2 * 1e6 / 3600

    # Water depth accumulation on street
    excess    = max(0, runoff_m3s - drain_capacity_m3s * drain_eff)
    depth_formula = min((excess * 3600) / max(area_km2 * 1e6, 1) * 100, 200)

    # Scaled 0-10 scores (for ML model input)
    rain_score  = min(rain_mm_hr, 100) / 10.0
    river_score = min(river_ratio * 10, 10)
    tide_score  = min(tide_ratio * 10, 10)
    elev_factor = max(0, 10 - elevation_m)

    # ── STEP 2: Run ML models ─────────────────────────────────
    features = np.array([[
        rain_score, river_score, tide_score,
        elev_factor, twi, dbi, drain_eff, slope_deg
    ]])

    rf_label    = 1      # fallback MEDIUM
    rf_probs    = [0.2, 0.5, 0.3]
    xgb_depth   = depth_formula

    if _rf_model is not None:
        rf_label  = int(_rf_model.predict(features)[0])
        rf_probs  = _rf_model.predict_proba(features)[0].tolist()

    if _xgb_model is not None:
        xgb_depth = float(_xgb_model.predict(features)[0])

    # ── STEP 3: Ensemble — combine formula + RF + XGBoost ────
    # Weighted average of formula depth and XGBoost depth
    final_depth = 0.4 * depth_formula + 0.6 * xgb_depth

    # Use whichever gives the HIGHER (worse) risk — safety first
    high_prob = rf_probs[2] if len(rf_probs) > 2 else 0

    if high_prob >= 0.60 or final_depth >= 30:
        risk  = "HIGH"
        color = "RED"
        alert_signal = "RED"
        warning = (
            "SEVERE FLOOD RISK — Water logging expected within "
            "2 hours. Move your vehicle immediately."
        )
    elif high_prob >= 0.35 or final_depth >= 10:
        risk  = "MEDIUM"
        color = "YELLOW"
        alert_signal = "YELLOW"
        warning = (
            "MODERATE FLOOD RISK — Water logging possible. "
            "Consider moving vehicle to safer street."
        )
    else:
        risk  = "LOW"
        color = "GREEN"
        alert_signal = "GREEN"
        warning = (
            "LOW FLOOD RISK — Street is currently safe. "
            "Stay alert if rainfall increases."
        )

    return {
        "risk":          risk,
        "color":         color,
        "alert_signal":  alert_signal,
        "warning":       warning,
        "predicted_depth_cm": round(final_depth, 1),
        "formula_depth_cm":   round(depth_formula, 1),
        "xgb_depth_cm":       round(xgb_depth, 1),
        "rf_confidence": {
            "LOW":    round(rf_probs[0] * 100, 1),
            "MEDIUM": round(rf_probs[1] * 100, 1),
            "HIGH":   round(rf_probs[2] * 100 if len(rf_probs) > 2 else 0, 1)
        },
        "features_used": {
            "twi":            round(twi, 3),
            "dbi":            round(dbi, 3),
            "drain_efficiency": round(drain_eff, 3),
            "runoff_m3s":     round(runoff_m3s, 4),
            "rain_score":     round(rain_score, 2),
            "river_score":    round(river_score, 2),
            "tide_score":     round(tide_score, 2)
        }
    }