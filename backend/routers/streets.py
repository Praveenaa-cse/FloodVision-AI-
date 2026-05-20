# backend/routers/streets.py
# GET /streets — returns risk for all Tamil Nadu streets
# Frontend calls this to draw the coloured map overlay

from fastapi import APIRouter
from utils.rainfall    import get_current_rainfall
from utils.river       import get_river_discharge, find_nearest_river, TN_RIVERS
from utils.tide        import compute_tide_height, find_nearest_coastal_station
from utils.elevation   import get_elevation_from_dem
from utils.flood_score import predict_flood_risk
from routers.predict   import TN_STREETS

router = APIRouter()


@router.get("/streets")
def get_all_streets():
    """
    Returns flood prediction for every Tamil Nadu street.
    Used by the frontend to draw the colour-coded map.
    
    Uses Chennai centre coordinates for rainfall
    (rain is roughly uniform within a city at any moment).
    """
    # Get rain once for the region (saves API calls)
    rain_data = get_current_rainfall(13.08, 80.27)
    rain_mm   = rain_data["rain_mm_per_hr"]

    results = []
    for name, data in TN_STREETS.items():
        try:
            river_name = find_nearest_river(data["lat"], data["lon"])
            river_info = TN_RIVERS.get(river_name, {})
            river_d    = get_river_discharge(
                            data["lat"], data["lon"], river_name)
            tide_stn   = find_nearest_coastal_station(
                            data["lat"], data["lon"])
            tide_d     = compute_tide_height(tide_stn)
            elev_d     = get_elevation_from_dem(data["lat"], data["lon"])

            pred = predict_flood_risk(
                rain_mm_hr          = rain_mm,
                river_discharge_m3s = river_d["discharge_m3s"],
                tide_height_m       = tide_d["tide_height_m"],
                elevation_m         = elev_d["elevation_m"],
                slope_deg           = elev_d["slope_deg"],
                danger_m3s          = river_info.get("danger_m3s", 500),
                mhwl_m              = tide_d["mhwl_m"],
                drain_capacity_m3s  = data["drain_capacity_m3s"],
                area_km2            = data["area_km2"],
                surface             = data["surface"]
            )

            results.append({
                "street":   name,
                "city":     data["city"],
                "area":     data["area"],
                "lat":      data["lat"],
                "lon":      data["lon"],
                "risk":     pred["risk"],
                "color":    pred["color"],
                "depth_cm": pred["predicted_depth_cm"],
                "warning":  pred["warning"]
            })
        except Exception as e:
            print(f"Error processing {name}: {e}")
            continue

    return {
        "streets":      sorted(results, key=lambda x: x["depth_cm"],
                                reverse=True),
        "total":        len(results),
        "rain_mm_hr":   rain_mm,
        "weather":      rain_data["weather_desc"]
    }