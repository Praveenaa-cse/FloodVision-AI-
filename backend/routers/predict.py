# backend/routers/predict.py
#
# WHAT THIS FILE DOES:
#   Defines POST /predict and POST /notify endpoints
#
# HOW /predict WORKS:
#   1. Receive lat/lon from user
#   2. Find nearest street + river + tide station
#   3. Fetch real rainfall (OpenWeather)
#   4. Fetch real river discharge (Open-Meteo)
#   5. Compute real tide (harmonic model)
#   6. Get elevation from DEM
#   7. Run flood_score ensemble
#   8. Find safer nearby streets
#   9. Return full prediction JSON

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import math

from utils.rainfall   import get_current_rainfall, get_forecast_rainfall_3h
from utils.river      import get_river_discharge, find_nearest_river, TN_RIVERS
from utils.tide       import compute_tide_height, find_nearest_coastal_station
from utils.elevation  import get_elevation_from_dem
from utils.flood_score import predict_flood_risk

router = APIRouter()

# ── Tamil Nadu streets database ───────────────────────────────
# This is your street-level data for all of Tamil Nadu
# Each street has: lat, lon, area, typical surface type,
#                  catchment area, drain capacity

TN_STREETS = {
    # ── Chennai ──────────────────────────────────────────────
    "Velachery Main Road": {
        "lat":13.0068,"lon":80.2206,"city":"Chennai",
        "area":"South Chennai","surface":"paved",
        "area_km2":1.2,"drain_capacity_m3s":0.4
    },
    "Adyar Bridge Road": {
        "lat":13.0012,"lon":80.2565,"city":"Chennai",
        "area":"South Chennai","surface":"paved",
        "area_km2":0.9,"drain_capacity_m3s":0.35
    },
    "Kathivakkam High Road": {
        "lat":13.2102,"lon":80.3012,"city":"Chennai",
        "area":"North Chennai","surface":"urban",
        "area_km2":0.6,"drain_capacity_m3s":0.2
    },
    "Royapuram Harbour Road": {
        "lat":13.1121,"lon":80.2954,"city":"Chennai",
        "area":"North Chennai","surface":"paved",
        "area_km2":0.7,"drain_capacity_m3s":0.25
    },
    "Anna Salai": {
        "lat":13.0569,"lon":80.2425,"city":"Chennai",
        "area":"Central Chennai","surface":"urban",
        "area_km2":1.5,"drain_capacity_m3s":0.8
    },
    "T Nagar Pondy Bazaar": {
        "lat":13.0418,"lon":80.2341,"city":"Chennai",
        "area":"Central Chennai","surface":"urban",
        "area_km2":1.0,"drain_capacity_m3s":0.6
    },
    "Poonamallee High Road": {
        "lat":13.0490,"lon":80.1698,"city":"Chennai",
        "area":"West Chennai","surface":"suburban",
        "area_km2":2.0,"drain_capacity_m3s":1.2
    },
    "Sholinganallur Road": {
        "lat":12.9010,"lon":80.2279,"city":"Chennai",
        "area":"South Chennai","surface":"urban",
        "area_km2":1.3,"drain_capacity_m3s":0.55
    },
    # ── Pondicherry ──────────────────────────────────────────
    "Pondicherry Beach Road": {
        "lat":11.9346,"lon":79.8360,"city":"Pondicherry",
        "area":"Coastal Pondicherry","surface":"paved",
        "area_km2":0.8,"drain_capacity_m3s":0.3
    },
    "Romain Rolland Street": {
        "lat":11.9330,"lon":79.8320,"city":"Pondicherry",
        "area":"Pondicherry Town","surface":"urban",
        "area_km2":0.6,"drain_capacity_m3s":0.35
    },
    # ── Nagapattinam ─────────────────────────────────────────
    "Nagapattinam Beach Road": {
        "lat":10.7640,"lon":79.8440,"city":"Nagapattinam",
        "area":"Coastal Nagapattinam","surface":"urban",
        "area_km2":0.7,"drain_capacity_m3s":0.25
    },
    # ── Tuticorin ────────────────────────────────────────────
    "Tuticorin Harbour Road": {
        "lat":8.8006,"lon":78.1460,"city":"Tuticorin",
        "area":"Coastal Tuticorin","surface":"paved",
        "area_km2":0.9,"drain_capacity_m3s":0.35
    },
    # ── Madurai ──────────────────────────────────────────────
    "Vaigai Bridge Road Madurai": {
        "lat":9.9312,"lon":78.1197,"city":"Madurai",
        "area":"Madurai Central","surface":"urban",
        "area_km2":1.1,"drain_capacity_m3s":0.5
    },
    # ── Trichy ───────────────────────────────────────────────
    "Cauvery Bridge Road Trichy": {
        "lat":10.8050,"lon":78.6856,"city":"Trichy",
        "area":"Trichy Central","surface":"urban",
        "area_km2":1.2,"drain_capacity_m3s":0.6
    },
    # ── Rameswaram ───────────────────────────────────────────
    "Rameswaram Main Road": {
        "lat":9.2881,"lon":79.3129,"city":"Rameswaram",
        "area":"Rameswaram Island","surface":"urban",
        "area_km2":0.5,"drain_capacity_m3s":0.15
    },
}


def find_nearest_street(lat: float, lon: float) -> dict:
    """Finds the closest street to given lat/lon."""
    nearest, min_dist = None, float("inf")
    for name, data in TN_STREETS.items():
        dist = math.sqrt((lat-data["lat"])**2 + (lon-data["lon"])**2)
        if dist < min_dist:
            min_dist = dist
            nearest  = {"name": name, **data}
    return nearest


def get_nearby_streets(current_name: str, count: int = 4) -> list:
    """Returns streets other than the current one."""
    return [
        {"name": n, **d}
        for n, d in TN_STREETS.items()
        if n != current_name
    ][:count]


# ── Input model (what the user sends) ────────────────────────
class LocationInput(BaseModel):
    latitude:  float
    longitude: float


# ── Notification input model ──────────────────────────────────
class NotifyInput(BaseModel):
    fcm_token: str     # user's phone FCM token
    risk:      str     # HIGH / MEDIUM / LOW
    street:    str     # street name


# ── POST /predict ─────────────────────────────────────────────
@router.post("/predict")
def predict(location: LocationInput):
    """
    MAIN ENDPOINT.
    Send: { "latitude": 13.08, "longitude": 80.27 }
    Get:  flood risk + warning + safer street suggestion
    """
    lat = location.latitude
    lon = location.longitude

    # Validate coordinates are within Tamil Nadu bounds
    if not (7.5 <= lat <= 14.0 and 76.0 <= lon <= 81.0):
        raise HTTPException(
            status_code=400,
            detail="Coordinates must be within Tamil Nadu / South India"
        )

    # ── Step 1: Find current street ───────────────────────────
    street = find_nearest_street(lat, lon)

    # ── Step 2: Get real rainfall ─────────────────────────────
    rain_data   = get_current_rainfall(lat, lon)
    forecast    = get_forecast_rainfall_3h(lat, lon)
    rain_mm     = rain_data["rain_mm_per_hr"]

    # ── Step 3: Get real river discharge ─────────────────────
    river_name  = find_nearest_river(lat, lon)
    river_info  = TN_RIVERS.get(river_name, {})
    danger_m3s  = river_info.get("danger_m3s", 500)
    river_data  = get_river_discharge(lat, lon, river_name)
    discharge   = river_data["discharge_m3s"]

    # ── Step 4: Get real tide height ─────────────────────────
    tide_station = find_nearest_coastal_station(lat, lon)
    tide_data    = compute_tide_height(tide_station)
    tide_m       = tide_data["tide_height_m"]
    mhwl_m       = tide_data["mhwl_m"]

    # ── Step 5: Get elevation ─────────────────────────────────
    elev_data    = get_elevation_from_dem(lat, lon)
    elevation_m  = elev_data["elevation_m"]
    slope_deg    = elev_data["slope_deg"]

    # ── Step 6: Run ensemble prediction ──────────────────────
    prediction = predict_flood_risk(
        rain_mm_hr          = rain_mm,
        river_discharge_m3s = discharge,
        tide_height_m       = tide_m,
        elevation_m         = elevation_m,
        slope_deg           = slope_deg,
        danger_m3s          = danger_m3s,
        mhwl_m              = mhwl_m,
        drain_capacity_m3s  = street["drain_capacity_m3s"],
        area_km2            = street["area_km2"],
        surface             = street["surface"]
    )

    # ── Step 7: Check nearby streets ─────────────────────────
    nearby = get_nearby_streets(street["name"])
    nearby_risks = []
    safest = None
    safest_depth = float("inf")

    for s in nearby:
        s_river = get_river_discharge(s["lat"], s["lon"],
                                       find_nearest_river(s["lat"],s["lon"]))
        s_tide  = compute_tide_height(
                    find_nearest_coastal_station(s["lat"],s["lon"]))
        s_elev  = get_elevation_from_dem(s["lat"], s["lon"])
        s_pred  = predict_flood_risk(
            rain_mm_hr          = rain_mm,
            river_discharge_m3s = s_river["discharge_m3s"],
            tide_height_m       = s_tide["tide_height_m"],
            elevation_m         = s_elev["elevation_m"],
            slope_deg           = s_elev["slope_deg"],
            danger_m3s          = s_river.get("danger_at_m3s", 500),
            mhwl_m              = s_tide["mhwl_m"],
            drain_capacity_m3s  = s["drain_capacity_m3s"],
            area_km2            = s["area_km2"],
            surface             = s["surface"]
        )
        nearby_risks.append({
            "street":   s["name"],
            "city":     s["city"],
            "area":     s["area"],
            "risk":     s_pred["risk"],
            "color":    s_pred["color"],
            "depth_cm": s_pred["predicted_depth_cm"]
        })
        if s_pred["predicted_depth_cm"] < safest_depth:
            safest_depth = s_pred["predicted_depth_cm"]
            safest = {
                "street": s["name"],
                "city":   s["city"],
                "area":   s["area"],
                "risk":   s_pred["risk"],
                "depth_cm": s_pred["predicted_depth_cm"]
            }

    # ── Step 8: Build evacuation tip ─────────────────────────
    risk = prediction["risk"]
    if risk == "HIGH" and safest:
        tip = (
            f"Move NOW to '{safest['street']}' in {safest['area']}. "
            f"Expected water there: only {safest['depth_cm']} cm. "
            f"Park your vehicle on higher ground immediately."
        )
    elif risk == "MEDIUM" and safest:
        tip = (
            f"Consider moving to '{safest['street']}' in {safest['area']} "
            f"as a precaution. Risk there: {safest['risk']}."
        )
    else:
        tip = "Your current street is safe. No action needed right now."

    # ── Step 9: Return full response ──────────────────────────
    return {
        "your_street": {
            "name":        street["name"],
            "city":        street["city"],
            "area":        street["area"],
            "lat":         lat,
            "lon":         lon
        },
        "flood_prediction": {
            "risk":          prediction["risk"],
            "color":         prediction["color"],
            "alert_signal":  prediction["alert_signal"],
            "warning":       prediction["warning"],
            "depth_cm":      prediction["predicted_depth_cm"],
            "rf_confidence": prediction["rf_confidence"]
        },
        "live_data": {
            "rain_now_mm_hr":      rain_mm,
            "weather":             rain_data["weather_desc"],
            "forecast_3h_mm":      forecast["next_3h_mm"],
            "forecast_6h_mm":      forecast["next_6h_mm"],
            "river_name":          river_name,
            "river_discharge_m3s": discharge,
            "river_status":        river_data["status"],
            "tide_station":        tide_station,
            "tide_height_m":       tide_m,
            "tide_blocked":        tide_data["drainage_blocked"],
            "elevation_m":         elevation_m
        },
        "evacuation_tip": tip,
        "safest_nearby":  safest,
        "nearby_streets": sorted(nearby_risks,
                                  key=lambda x: x["depth_cm"])
    }


# ── POST /notify ──────────────────────────────────────────────
@router.post("/notify")
def send_notification(data: NotifyInput):
    """Sends a push notification to the user's phone."""
    try:
        from utils.notifications import send_flood_alert
        result = send_flood_alert(data.fcm_token, data.risk, data.street)
        return {"success": True, "message_id": result}
    except Exception as e:
        return {"success": False, "error": str(e)}