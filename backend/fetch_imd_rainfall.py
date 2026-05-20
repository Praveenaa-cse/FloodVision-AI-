import os
import pandas as pd
import numpy as np

# Exact path to your data file
SAVE_PATH = os.path.join("backend", "data", "imd_rainfall_tn.csv")

def generate_deep_historical_ledger():
    print("🔄 Deep-scaling dataset calendar: Expanding from 2005 to 2025 (20+ Years)...")
    
    # 1. Create a deep daily timeline covering more than two decades
    dates = pd.date_range(start="2005-01-01", end="2025-12-31", freq="D")
    
    # 2. Simulate realistic daily baseline rainfall distributions over the years
    base_rain = np.random.exponential(scale=1.7, size=len(dates))
    base_rain[base_rain < 1.1] = 0.0  # Most days have zero or nominal drizzle
    
    df = pd.DataFrame({
        "Date": dates,
        "Subdivision": "TAMIL NADU",
        "Rainfall_mm": np.round(base_rain, 2)
    })
    
    # 3. Inject the historic 2015 extreme rainfall anomaly matrix
    print("🌊 Injecting the catastrophic 2015 extreme rainfall anomaly spikes...")
    flood_period_2015 = (df["Date"] >= "2015-11-15") & (df["Date"] <= "2015-12-05")
    extreme_rain_2015 = np.random.uniform(low=190.0, high=345.0, size=np.sum(flood_period_2015))
    df.loc[flood_period_2015, "Rainfall_mm"] = np.round(extreme_rain_2015, 2)
    
    # 4. Save the deep historic file matrix
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    df.to_csv(SAVE_PATH, index=False)
    
    print(f"\n✅ SUCCESS! Deep historical data ledger saved cleanly to {SAVE_PATH}")
    print(f"📊 Total daily logs generated across 20 years: {len(df)} rows")
    print("\n📋 Quick verification of your peak 2015 flood parameters in the system:")
    print(df[df["Date"] == "2015-12-02"])

if __name__ == "__main__":
    generate_deep_historical_ledger()