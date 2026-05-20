import ee

# Initialize Earth Engine
ee.Initialize(project='floodvision-data')

# Load Global Flood Database
floods = ee.ImageCollection("GLOBAL_FLOOD_DB/MODIS_EVENTS/V1")

# Load administrative boundaries
admin = ee.FeatureCollection("FAO/GAUL/2015/level1")

# Filter for Tamil Nadu
tamil_nadu = admin.filter(ee.Filter.eq('ADM1_NAME', 'Tamil Nadu'))

# Clip floods to Tamil Nadu
tn_floods = floods.filterDate('2000-01-01', '2018-12-31') \
                  .map(lambda img: img.clip(tamil_nadu))

# Export first flood event to Google Drive
task = ee.batch.Export.image.toDrive(
    image = tn_floods.first().toUint16(),
    description = 'TamilNadu_Flood',
    scale = 250,
    region = tamil_nadu.geometry()
)

task.start()
print("Export started. Check your Google Drive for the GeoTIFF file.")
