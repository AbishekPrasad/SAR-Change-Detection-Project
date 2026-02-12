# ====================== CONFIG (ALL CAPS) ======================
GEE_PROJECT_ID = "my-first-project-482818"

START_YEAR = 2018
END_YEAR = 2022

OUTPUT_FOLDER = "naip_index"
GEOJSON_PATH = "naip_index/naip_available_zones.geojson"
CSV_PATH = "naip_index/naip_available_centroids.csv"

# Search only over CONUS (where NAIP actually exists)
MIN_LON, MAX_LON = -125, -67
MIN_LAT, MAX_LAT = 25, 49

TILE_SIZE_DEG = 2        # search grid
MAX_WORKERS = 6
SLEEP_SEC = 0.7

MAX_LOCATIONS_PER_YEAR = 30   # <-- YOUR NEW LIMIT
# ===============================================================

import os
import time
import ee
import geopandas as gpd
import pandas as pd
import multiprocessing as mp

def notify(msg):
    print(f"[INFO] {msg}")

# ---------------------------------------------------------
# EACH WORKER MUST INITIALIZE GEE SEPARATELY
# ---------------------------------------------------------
def worker_initialize_gee():
    try:
        ee.Initialize(project=GEE_PROJECT_ID)
    except:
        ee.Initialize()

# ---------------------------------------------------------
# LAND MASK (to remove ocean)
# We use a global land mask dataset
# ---------------------------------------------------------
def get_land_mask():
    return ee.Image("MODIS/006/MOD44W/2015_01_01").select("water_mask")  # 1 = land, 0 = water

# ---------------------------------------------------------
# CREATE TILES OVER CONUS
# ---------------------------------------------------------
def make_naip_tiles():
    tiles = []
    for lon in range(MIN_LON, MAX_LON, TILE_SIZE_DEG):
        for lat in range(MIN_LAT, MAX_LAT, TILE_SIZE_DEG):
            tiles.append(
                ee.Geometry.Rectangle([
                    lon, lat,
                    lon + TILE_SIZE_DEG,
                    lat + TILE_SIZE_DEG
                ])
            )

    notify(f"Created {len(tiles)} search tiles over CONUS.")
    return tiles

# ---------------------------------------------------------
# QUERY NAIP FOR ONE TILE + YEAR (LAND ONLY)
# ---------------------------------------------------------
def get_naip_footprints_for_tile(args):
    year, tile = args

    worker_initialize_gee()
    time.sleep(SLEEP_SEC)

    land_mask = get_land_mask()

    # Keep only areas where land_mask == 1
    land_tile = tile.intersection(land_mask.geometry(), 1)

    col = (
        ee.ImageCollection("USDA/NAIP/DOQQ")
        .filterDate(f"{year}-01-01", f"{year}-12-31")
        .filterBounds(land_tile)
        .select(["R"])  # geometry only
    )

    try:
        count = col.size().getInfo()
    except Exception as e:
        print(f"[WARN] Year {year}: failed count — {e}")
        return []

    if count == 0:
        return []

    fc = col.map(lambda img: ee.Feature(img.geometry()).set({
        "date": img.date().format(),
        "system_id": img.get("system:index"),
        "year": year
    }))

    try:
        geojson = fc.getInfo()
        return geojson["features"]
    except Exception as e:
        print(f"[WARN] Year {year}: error on tile — {e}")
        return []

# ---------------------------------------------------------
# MAIN COLLECTION (PARALLEL, LAND ONLY, 30 PER YEAR)
# ---------------------------------------------------------
def collect_available_zones_parallel():
    notify("Initializing GEE in main process...")
    worker_initialize_gee()

    tiles = make_naip_tiles()
    all_features = []
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    for year in range(START_YEAR, END_YEAR + 1):
        notify(f"\n===== YEAR {year} =====")
        notify(f"Launching {MAX_WORKERS} parallel workers...")

        tasks = [(year, t) for t in tiles]

        with mp.Pool(processes=MAX_WORKERS) as pool:
            results = pool.map(get_naip_footprints_for_tile, tasks)

        year_features = []
        for res in results:
            if res:
                year_features.extend(res)

        # ---- TAKE ONLY FIRST 30 LAND LOCATIONS ----
        year_features = year_features[:MAX_LOCATIONS_PER_YEAR]
        all_features.extend(year_features)

        notify(f"Collected {len(year_features)} LAND footprints for {year} (LIMIT = {MAX_LOCATIONS_PER_YEAR}).")

    if len(all_features) == 0:
        raise ValueError("No NAIP land zones found — check region or dataset.")

    gdf = gpd.GeoDataFrame.from_features({
        "type": "FeatureCollection",
        "features": all_features
    })

    gdf.to_file(GEOJSON_PATH, driver="GeoJSON")
    notify(f"Saved: {GEOJSON_PATH}")

    gdf["centroid_lon"] = gdf.geometry.centroid.x
    gdf["centroid_lat"] = gdf.geometry.centroid.y

    df = gdf[["system_id", "year", "date", "centroid_lat", "centroid_lon"]]
    df.to_csv(CSV_PATH, index=False)

    notify(f"Saved: {CSV_PATH}")
    notify("NAIP LAND zones indexed successfully (30 per year).")

# ------------------- MAIN -------------------
if __name__ == "__main__":
    collect_available_zones_parallel()
