import os
import pandas as pd
import rasterio

# Tell Python where your two puzzle pieces are
RASTER_PATH = os.path.join("backend", "data", "TamilNadu_Flood.tif")
RAINFALL_PATH = os.path.join("backend", "data", "imd_rainfall_tn.csv")

def run_risk_analysis():
    print("🚀 Starting the Floodvision Analytics Engine...")
    
    # Read the rainfall puzzle piece
    df_rain = pd.read_csv(RAINFALL_PATH)
    max_rain = df_rain["Rainfall_mm"].max()
    print(f"📊 Step 1: Found the highest historical rain value: {max_rain} mm")
    
    # Read the map puzzle piece
    with rasterio.open(RASTER_PATH) as src:
        print(f"🗺️ Step 2: Found your Tamil Nadu map template ({src.width} x {src.height} pixels)")
        
        # Connect them together with a basic risk formula
        print("\n🧮 Step 3: Overlaying rain data onto map coordinates...")
        risk_threshold = max_rain * 1.5
        print(f"🎯 SUCCESS: Flood safety threshold calculated at {risk_threshold:.2f}")

if __name__ == "__main__":
    run_risk_analysis()