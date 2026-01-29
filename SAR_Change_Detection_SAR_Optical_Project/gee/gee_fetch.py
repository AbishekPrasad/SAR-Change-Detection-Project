
import ee

def fetch_sentinel1(lat, lon, start, end, buffer=1500):
    region = ee.Geometry.Point([lon, lat]).buffer(buffer)
    image = (
        ee.ImageCollection("COPERNICUS/S1_GRD")
        .filterBounds(region)
        .filterDate(start, end)
        .filter(ee.Filter.eq("instrumentMode", "IW"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
        .select("VV")
        .median()
        .clip(region)
    )
    return image, region

def fetch_sentinel2_rgb(lat, lon, start, end, buffer=2500):
    region = ee.Geometry.Point([lon, lat]).buffer(buffer)
    image = (
        ee.ImageCollection("COPERNICUS/S2_SR")
        .filterBounds(region)
        .filterDate(start, end)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 10))
        .select(["B4", "B3", "B2"])
        .median()
        .clip(region)
    )
    return image, region
