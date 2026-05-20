import os
import rasterio

# Define the path to your data folder
DATA_PATH = os.path.join("backend", "data", "TamilNadu_Flood.tif")

def check_geospatial_data():
    if not os.path.exists(DATA_PATH):
        print(f"❌ Error: Cannot find the file at {DATA_PATH}. Check your folder structure!")
        return

    print("🔄 Opening Tamil Nadu geospatial file...")
    
    # Open and read the metadata of your file
    with rasterio.open(DATA_PATH) as dataset:
        print("\n✅ File Successfully Loaded!")
        print(f"📊 Total Data Bands: {dataset.count}")
        print(f"📐 Dimensions: {dataset.width} x {dataset.height} (Width x Height pixels)")
        print(f"🗺️ Coordinate System (CRS): {dataset.crs}")
        
        # Read the first band of data into a matrix array
        band1 = dataset.read(1)
        print(f"🔢 Data Matrix Shape: {band1.shape}")

if __name__ == "__main__":
    check_geospatial_data()