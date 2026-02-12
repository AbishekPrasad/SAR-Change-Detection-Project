import ee
from config import BUFFER_METERS

def fetch_naip(lat, lon, start, end, buffer=BUFFER_METERS):
    point = ee.Geometry.Point([lon, lat])
    region = point.buffer(buffer)

    col = (
        ee.ImageCollection("USDA/NAIP/DOQQ")
        .filterBounds(region)
        .filterDate(start, end)
        .sort("system:time_start")
    )

    count = col.size().getInfo()
    print(f"[INFO] NAIP images found: {count}")

    if count == 0:
        raise ValueError(
            f"No NAIP found for ({lat},{lon}) between {start} and {end}."
        )

    img = col.median().select(["R","G","B","N"]).clip(region)
    return img, region
