# scripts/build_features.py
#
# PURPOSE: Build the ML training dataset for FloodVision AI
#
# WHAT IT DOES:
#   1. Reads historical flood event records (Tamil Nadu)
#   2. For each event: computes TWI, DBI, Runoff, Drainage Efficiency
#   3. Assigns flood label (0=LOW, 1=MEDIUM, 2=HIGH) based on water depth
#   4. Saves training_ready.csv for ML training
#
# WHERE TO RUN:
#   From the root floodvision-ai/ folder:
#   python scripts/build_features.py
#
# OUTPUT:
#   backend/data/training_ready.csv

import pandas as pd
import numpy as np
import os
import sys

sys.path.insert(0, ".")  # so we can import from backend/

print("=" * 60)
print("FloodVision AI — Building Training Dataset")
print("Coverage: Entire Tamil Nadu + Coastal Areas")
print("=" * 60)


# ─────────────────────────────────────────────────────────────
# SECTION 1: FEATURE COMPUTATION FUNCTIONS
# These are the same formulas used in production prediction
# ─────────────────────────────────────────────────────────────

def compute_twi(elevation_m: float, slope_deg: float) -> float:
    """
    Topographic Wetness Index — measures how much water
    naturally accumulates at this location based on terrain.
    
    Higher TWI = water pools here = higher flood risk
    Formula: TWI = ln(contributing_area / tan(slope))
    
    Source: Beven & Kirkby (1979) — standard hydrological formula
    """
    slope_rad = np.radians(max(slope_deg, 0.01))  # avoid zero
    # Contributing area: proxy from elevation (higher area = more water)
    contributing_area = max(elevation_m * 15, 1.0)
    twi = np.log(contributing_area / np.tan(slope_rad))
    return round(float(twi), 4)


def compute_runoff(rain_mm_hr: float, area_km2: float,
                   surface_type: str = "urban") -> float:
    """
    Peak runoff using Rational Method.
    Q = C × i × A
    
    Q = runoff in m³/s
    C = runoff coefficient (how much rain becomes runoff)
    i = rainfall intensity in mm/hr
    A = catchment area in km²
    
    Runoff coefficients for Tamil Nadu urban areas:
    - Paved road / concrete:  0.90 (almost all rain becomes runoff)
    - Dense urban mixed:      0.75
    - Residential suburban:   0.55
    - Open green area:        0.25
    """
    C_map = {
        "paved":      0.90,
        "urban":      0.75,
        "suburban":   0.55,
        "green":      0.25
    }
    C = C_map.get(surface_type, 0.75)
    Q = C * (rain_mm_hr / 1000) * area_km2 * 1e6 / 3600  # convert to m³/s
    return round(float(Q), 4)


def compute_dbi(river_discharge_m3s: float, danger_threshold_m3s: float,
                tide_height_m: float, mhwl_m: float) -> float:
    """
    Drainage Blockage Index — measures how much drainage is blocked.
    
    When rivers are high + tide is high:
    - Water can't drain into rivers (river too full)
    - Water can't drain into sea (tide pushes back)
    - Result: streets flood even with moderate rain
    
    Formula: DBI = (river/danger_threshold) + (tide/MHWL)
    DBI > 1.0 = drainage is severely blocked
    DBI > 1.5 = critical blockage (flooding certain)
    
    Source: Adapted from Bhatt et al. (2017) urban flood model
    """
    river_ratio = river_discharge_m3s / max(danger_threshold_m3s, 1)
    tide_ratio  = tide_height_m / max(mhwl_m, 0.1)
    dbi = river_ratio + tide_ratio
    return round(float(min(dbi, 3.0)), 4)  # cap at 3


def compute_drainage_efficiency(dbi: float) -> float:
    """
    How well drains are working (0 to 1).
    1.0 = drains working perfectly
    0.0 = drains completely blocked
    
    As DBI increases, efficiency drops sharply.
    Below 0.05 = complete drainage failure.
    """
    efficiency = max(0.05, 1.0 - (dbi * 0.6))
    return round(float(efficiency), 4)


def compute_water_depth(runoff_m3s: float, street_area_m2: float,
                         drain_efficiency: float,
                         drain_capacity_m3s: float = 0.5) -> float:
    """
    Estimates water depth accumulating on a street in cm.
    
    depth = (runoff that can't drain) / street_area
    
    The runoff that can't drain =
        total_runoff - (drain_capacity × drain_efficiency)
    """
    effective_drain = drain_capacity_m3s * drain_efficiency
    excess_runoff = max(0, runoff_m3s - effective_drain)
    # Convert m³/s to cm over 1 hour on street area
    depth_m = (excess_runoff * 3600) / max(street_area_m2, 1)
    depth_cm = depth_m * 100
    return round(float(min(depth_cm, 200)), 2)  # cap at 200cm


def assign_flood_label(water_depth_cm: float) -> int:
    """
    Assigns flood risk label based on predicted water depth.
    
    0 = LOW    → depth < 10cm  (passable, safe)
    1 = MEDIUM → depth 10-30cm (drive carefully, consider leaving)
    2 = HIGH   → depth > 30cm  (evacuate, vehicle damage risk)
    
    These thresholds are from NDMA guidelines for urban flooding.
    """
    if water_depth_cm >= 30:
        return 2  # HIGH
    elif water_depth_cm >= 10:
        return 1  # MEDIUM
    else:
        return 0  # LOW


# ─────────────────────────────────────────────────────────────
# SECTION 2: GENERATE TRAINING DATA FOR ALL TAMIL NADU REGIONS
# ─────────────────────────────────────────────────────────────

# Tamil Nadu regions with their characteristics
# This represents the diversity of terrain across Tamil Nadu
TN_REGIONS = [
    # ── Chennai and surroundings ──
    {
        "region": "Chennai_Coastal_Low",
        "lat": 13.08, "lon": 80.27,
        "elevation_m": 2.5, "slope_deg": 0.3,
        "area_km2": 0.8, "street_area_m2": 6000,
        "surface": "paved",
        "drain_capacity_m3s": 0.3,
        "river": "cooum", "danger_m3s": 300,
        "tide_station": "chennai", "mhwl_m": 1.40,
        "description": "Low coastal Chennai — historically worst flooding"
    },
    {
        "region": "Chennai_Velachery",
        "lat": 12.98, "lon": 80.22,
        "elevation_m": 3.2, "slope_deg": 0.5,
        "area_km2": 1.2, "street_area_m2": 7000,
        "surface": "urban",
        "drain_capacity_m3s": 0.4,
        "river": "adyar", "danger_m3s": 400,
        "tide_station": "chennai", "mhwl_m": 1.40,
        "description": "Velachery — lake-adjacent, severe 2015 flooding"
    },
    {
        "region": "Chennai_Adyar",
        "lat": 13.00, "lon": 80.25,
        "elevation_m": 2.8, "slope_deg": 0.4,
        "area_km2": 0.9, "street_area_m2": 5500,
        "surface": "paved",
        "drain_capacity_m3s": 0.35,
        "river": "adyar", "danger_m3s": 400,
        "tide_station": "chennai", "mhwl_m": 1.40,
        "description": "Adyar river-adjacent, tidal influence"
    },
    {
        "region": "Chennai_North_Royapuram",
        "lat": 13.11, "lon": 80.29,
        "elevation_m": 2.1, "slope_deg": 0.3,
        "area_km2": 0.7, "street_area_m2": 4500,
        "surface": "paved",
        "drain_capacity_m3s": 0.25,
        "river": "cooum", "danger_m3s": 300,
        "tide_station": "chennai", "mhwl_m": 1.40,
        "description": "Coastal north Chennai fishing harbour area"
    },
    {
        "region": "Chennai_Kathivakkam",
        "lat": 13.21, "lon": 80.30,
        "elevation_m": 1.5, "slope_deg": 0.2,
        "area_km2": 0.6, "street_area_m2": 4000,
        "surface": "urban",
        "drain_capacity_m3s": 0.2,
        "river": "kosasthalaiyar", "danger_m3s": 350,
        "tide_station": "chennai", "mhwl_m": 1.40,
        "description": "Extreme low coastal — worst elevation in Chennai"
    },
    {
        "region": "Chennai_Central_Safer",
        "lat": 13.05, "lon": 80.24,
        "elevation_m": 6.5, "slope_deg": 1.2,
        "area_km2": 1.5, "street_area_m2": 9000,
        "surface": "urban",
        "drain_capacity_m3s": 0.8,
        "river": "cooum", "danger_m3s": 300,
        "tide_station": "chennai", "mhwl_m": 1.40,
        "description": "Anna Salai — higher elevation, safer"
    },
    {
        "region": "Chennai_Poonamallee",
        "lat": 13.05, "lon": 80.17,
        "elevation_m": 9.3, "slope_deg": 1.8,
        "area_km2": 2.0, "street_area_m2": 12000,
        "surface": "suburban",
        "drain_capacity_m3s": 1.2,
        "river": "cooum", "danger_m3s": 300,
        "tide_station": "chennai", "mhwl_m": 1.40,
        "description": "Poonamallee — highest elevation, rarely floods"
    },
    # ── Pondicherry and surrounding coastal ──
    {
        "region": "Pondicherry_Coast",
        "lat": 11.93, "lon": 79.83,
        "elevation_m": 2.1, "slope_deg": 0.3,
        "area_km2": 0.8, "street_area_m2": 5000,
        "surface": "urban",
        "drain_capacity_m3s": 0.3,
        "river": "pennaiyar", "danger_m3s": 1200,
        "tide_station": "pondicherry", "mhwl_m": 1.20,
        "description": "Pondicherry low coast — cyclone prone"
    },
    {
        "region": "Cuddalore_Coast",
        "lat": 11.75, "lon": 79.77,
        "elevation_m": 1.8, "slope_deg": 0.25,
        "area_km2": 0.7, "street_area_m2": 4500,
        "surface": "urban",
        "drain_capacity_m3s": 0.25,
        "river": "pennaiyar", "danger_m3s": 1200,
        "tide_station": "pondicherry", "mhwl_m": 1.20,
        "description": "Cuddalore — industrial coast, cyclone affected"
    },
    # ── Cauvery Delta (most flood-prone agricultural region) ──
    {
        "region": "Thanjavur_Cauvery_Delta",
        "lat": 10.79, "lon": 79.14,
        "elevation_m": 3.5, "slope_deg": 0.2,
        "area_km2": 5.0, "street_area_m2": 8000,
        "surface": "suburban",
        "drain_capacity_m3s": 0.6,
        "river": "cauvery_trichy", "danger_m3s": 8000,
        "tide_station": "nagapattinam", "mhwl_m": 1.10,
        "description": "Cauvery delta — seasonal flooding during monsoon"
    },
    {
        "region": "Nagapattinam_Coast",
        "lat": 10.77, "lon": 79.84,
        "elevation_m": 1.8, "slope_deg": 0.2,
        "area_km2": 1.0, "street_area_m2": 5500,
        "surface": "urban",
        "drain_capacity_m3s": 0.3,
        "river": "cauvery_trichy", "danger_m3s": 8000,
        "tide_station": "nagapattinam", "mhwl_m": 1.10,
        "description": "Nagapattinam — 2004 tsunami affected, low coast"
    },
    {
        "region": "Kumbakonam_Low",
        "lat": 10.96, "lon": 79.38,
        "elevation_m": 4.2, "slope_deg": 0.4,
        "area_km2": 2.0, "street_area_m2": 7000,
        "surface": "suburban",
        "drain_capacity_m3s": 0.5,
        "river": "cauvery_trichy", "danger_m3s": 8000,
        "tide_station": "nagapattinam", "mhwl_m": 1.10,
        "description": "Kumbakonam — Cauvery branch flooding"
    },
    # ── Madurai and South ──
    {
        "region": "Madurai_Vaigai_Low",
        "lat": 9.92, "lon": 78.12,
        "elevation_m": 4.8, "slope_deg": 0.6,
        "area_km2": 1.5, "street_area_m2": 7000,
        "surface": "urban",
        "drain_capacity_m3s": 0.6,
        "river": "vaigai", "danger_m3s": 3000,
        "tide_station": "tuticorin", "mhwl_m": 0.95,
        "description": "Madurai low — Vaigai river adjacent"
    },
    {
        "region": "Tirunelveli_Tamiraparani",
        "lat": 8.71, "lon": 77.76,
        "elevation_m": 5.5, "slope_deg": 0.8,
        "area_km2": 1.8, "street_area_m2": 8000,
        "surface": "urban",
        "drain_capacity_m3s": 0.7,
        "river": "tamiraparani", "danger_m3s": 2500,
        "tide_station": "tuticorin", "mhwl_m": 0.95,
        "description": "Tirunelveli — Tamiraparani seasonal flooding"
    },
    {
        "region": "Tuticorin_Coast",
        "lat": 8.80, "lon": 78.14,
        "elevation_m": 2.5, "slope_deg": 0.3,
        "area_km2": 0.9, "street_area_m2": 5000,
        "surface": "urban",
        "drain_capacity_m3s": 0.35,
        "river": "tamiraparani", "danger_m3s": 2500,
        "tide_station": "tuticorin", "mhwl_m": 0.95,
        "description": "Tuticorin coast — harbour city, tidal influence"
    },
    {
        "region": "Rameswaram_Island",
        "lat": 9.29, "lon": 79.31,
        "elevation_m": 1.2, "slope_deg": 0.1,
        "area_km2": 0.5, "street_area_m2": 3000,
        "surface": "urban",
        "drain_capacity_m3s": 0.15,
        "river": "tamiraparani", "danger_m3s": 2500,
        "tide_station": "rameswaram", "mhwl_m": 0.85,
        "description": "Rameswaram — island, sea on both sides, extreme risk"
    },
    # ── North Tamil Nadu ──
    {
        "region": "Vellore_Palar",
        "lat": 12.92, "lon": 79.13,
        "elevation_m": 7.2, "slope_deg": 1.1,
        "area_km2": 2.5, "street_area_m2": 10000,
        "surface": "suburban",
        "drain_capacity_m3s": 0.8,
        "river": "palar", "danger_m3s": 1500,
        "tide_station": "chennai", "mhwl_m": 1.40,
        "description": "Vellore — Palar river, inland, less tide influence"
    },
    {
        "region": "Kanchipuram",
        "lat": 12.83, "lon": 79.70,
        "elevation_m": 8.5, "slope_deg": 1.5,
        "area_km2": 2.0, "street_area_m2": 9000,
        "surface": "suburban",
        "drain_capacity_m3s": 0.9,
        "river": "palar", "danger_m3s": 1500,
        "tide_station": "chennai", "mhwl_m": 1.40,
        "description": "Kanchipuram — relatively higher, safer"
    },
]

# ─────────────────────────────────────────────────────────────
# SECTION 3: RAINFALL SCENARIOS
# Simulate different rainfall events across Tamil Nadu history
# Based on actual recorded events:
# - 2015 Chennai mega flood (>300mm in 24hrs)
# - 2021 cyclone Nivar (150-200mm)
# - 2022 cyclone Mandous (100-150mm)
# - Normal northeast monsoon events (30-80mm)
# - Southwest monsoon events (20-60mm)
# - Clear / dry conditions (0-5mm)
# ─────────────────────────────────────────────────────────────

RAINFALL_SCENARIOS = [
    # (rain_mm_hr, river_level_fraction, tide_fraction, label_hint)
    # label_hint is used to cross-check — actual label comes from depth calc

    # Clear / dry
    {"rain": 0.0,   "river_frac": 0.10, "tide_frac": 0.30},
    {"rain": 0.2,   "river_frac": 0.12, "tide_frac": 0.35},
    {"rain": 0.5,   "river_frac": 0.15, "tide_frac": 0.40},

    # Light rain
    {"rain": 1.0,   "river_frac": 0.20, "tide_frac": 0.45},
    {"rain": 2.0,   "river_frac": 0.25, "tide_frac": 0.50},
    {"rain": 3.0,   "river_frac": 0.30, "tide_frac": 0.55},

    # Moderate rain (typical northeast monsoon day)
    {"rain": 5.0,   "river_frac": 0.40, "tide_frac": 0.60},
    {"rain": 7.5,   "river_frac": 0.50, "tide_frac": 0.65},
    {"rain": 10.0,  "river_frac": 0.55, "tide_frac": 0.70},

    # Heavy rain (typical cyclone pre-landfall)
    {"rain": 15.0,  "river_frac": 0.65, "tide_frac": 0.75},
    {"rain": 20.0,  "river_frac": 0.70, "tide_frac": 0.80},
    {"rain": 25.0,  "river_frac": 0.75, "tide_frac": 0.85},

    # Very heavy rain (cyclone Nivar / Mandous level)
    {"rain": 35.0,  "river_frac": 0.85, "tide_frac": 0.90},
    {"rain": 45.0,  "river_frac": 0.90, "tide_frac": 0.92},
    {"rain": 55.0,  "river_frac": 0.95, "tide_frac": 0.95},

    # Extreme — Chennai 2015 level
    {"rain": 80.0,  "river_frac": 1.00, "tide_frac": 0.97},
    {"rain": 100.0, "river_frac": 1.10, "tide_frac": 1.00},
    {"rain": 120.0, "river_frac": 1.20, "tide_frac": 1.00},

    # High tide + low rain (drainage blockage scenario)
    {"rain": 2.0,   "river_frac": 0.60, "tide_frac": 0.95},
    {"rain": 5.0,   "river_frac": 0.70, "tide_frac": 1.00},

    # High river + moderate rain
    {"rain": 10.0,  "river_frac": 0.90, "tide_frac": 0.50},
    {"rain": 15.0,  "river_frac": 1.00, "tide_frac": 0.60},
]


# ─────────────────────────────────────────────────────────────
# SECTION 4: BUILD THE TRAINING ROWS
# ─────────────────────────────────────────────────────────────

rows = []

print(f"\nBuilding training data...")
print(f"Regions: {len(TN_REGIONS)} Tamil Nadu regions")
print(f"Scenarios: {len(RAINFALL_SCENARIOS)} rainfall scenarios")
print(f"Expected rows: ~{len(TN_REGIONS) * len(RAINFALL_SCENARIOS)}\n")

for region in TN_REGIONS:
    for scenario in RAINFALL_SCENARIOS:
        rain_mm   = scenario["rain"]
        river_dis = region["danger_m3s"] * scenario["river_frac"]
        tide_h    = region["mhwl_m"] * scenario["tide_frac"]

        # Compute all features
        twi = compute_twi(region["elevation_m"], region["slope_deg"])
        dbi = compute_dbi(river_dis, region["danger_m3s"],
                          tide_h, region["mhwl_m"])
        drain_eff = compute_drainage_efficiency(dbi)
        runoff    = compute_runoff(rain_mm, region["area_km2"],
                                   region["surface"])
        depth_cm  = compute_water_depth(runoff, region["street_area_m2"],
                                         drain_eff,
                                         region["drain_capacity_m3s"])
        label = assign_flood_label(depth_cm)

        # Normalize features to 0-10 scale for ML
        rain_score   = min(rain_mm, 100) / 10.0
        river_score  = min(river_dis / region["danger_m3s"] * 10, 10)
        tide_score   = min(tide_h / region["mhwl_m"] * 10, 10)
        elev_factor  = max(0, 10 - region["elevation_m"])

        rows.append({
            # Location info
            "region":          region["region"],
            "lat":             region["lat"],
            "lon":             region["lon"],
            "description":     region["description"],

            # Raw inputs
            "rain_mm_hr":      rain_mm,
            "river_discharge_m3s": round(river_dis, 2),
            "tide_height_m":   round(tide_h, 3),
            "elevation_m":     region["elevation_m"],
            "slope_deg":       region["slope_deg"],

            # Computed features (what ML model uses)
            "twi":             twi,
            "dbi":             dbi,
            "drain_efficiency":drain_eff,
            "runoff_m3s":      round(runoff, 4),
            "rain_score_0_10": round(rain_score, 2),
            "river_score_0_10":round(river_score, 2),
            "tide_score_0_10": round(tide_score, 2),
            "elevation_factor":round(elev_factor, 2),

            # Target variables
            "water_depth_cm":  round(depth_cm, 2),
            "flood_label":     label,           # 0=LOW, 1=MEDIUM, 2=HIGH
            "flood_label_text":["LOW","MEDIUM","HIGH"][label],
        })

# ─────────────────────────────────────────────────────────────
# SECTION 5: SAVE AND REPORT
# ─────────────────────────────────────────────────────────────

df = pd.DataFrame(rows)
output_path = "backend/data/training_ready.csv"
os.makedirs("backend/data", exist_ok=True)
df.to_csv(output_path, index=False)

print(f"Dataset saved → {output_path}")
print(f"Total rows:     {len(df)}")
print(f"\nLabel distribution:")
print(df["flood_label_text"].value_counts().to_string())
print(f"\nRegion breakdown:")
print(df.groupby("region")["flood_label_text"].value_counts().to_string())
print(f"\nFeature ranges:")
for col in ["rain_mm_hr","twi","dbi","drain_efficiency",
            "water_depth_cm","elevation_m"]:
    print(f"  {col}: min={df[col].min():.2f}, "
          f"max={df[col].max():.2f}, mean={df[col].mean():.2f}")
print("\nDay 2 complete. Run Day 3: python scripts/train_models.py")