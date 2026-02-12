# ====================== CONFIG (ALL CAPS) ======================
GEE_PROJECT_ID = "my-first-project-482818"

CSV_PATH = "naip_index/naip_available_centroids.csv"
OUTPUT_ROOT = "raw_data"

# EXACT CHIP SIZE
EXPORT_SIZE = 256        # 256 x 256 only
SCALE = 1                # NAIP native resolution
CRS = "EPSG:4326"

# DATE WINDOW AROUND TIMESTAMP
DATE_WINDOW_DAYS = 15
# ===============================================================

import os
import time
import ee
import requests
import pandas as pd

def notify(msg):
    print(f"[INFO] {msg}")

# ---------------------------------------------------------
# GEE INIT
# ---------------------------------------------------------
def initialize_gee():
    notify("Initializing Google Earth Engine...")
    ee.Initialize(project=GEE_PROJECT_ID)
    notify("GEE Initialized.")

# ---------------------------------------------------------
# EXPAND DATE WINDOW
# ---------------------------------------------------------
def expand_date(center_date):
    d = pd.to_datetime(center_date)
    start = (d - pd.Timedelta(days=DATE_WINDOW_DAYS)).strftime("%Y-%m-%d")
    end = (d + pd.Timedelta(days=DATE_WINDOW_DAYS)).strftime("%Y-%m-%d")
    return start, end

# ---------------------------------------------------------
# MAKE TIGHT 256x256 ROI AROUND CENTROID
# ---------------------------------------------------------
def make_256_roi(lat, lon, scale=SCALE, size=EXPORT_SIZE):
    # Convert pixels → degrees (approx)
    half_width_deg = (size * scale) / 2 / 111320.0

    return ee.Geometry.Rectangle([
        lon - half_width_deg,
        lat - half_width_deg,
        lon + half_width_deg,
        lat + half_width_deg
    ])

# ---------------------------------------------------------
# DOWNLOAD ONE 256x256 CHIP
# ---------------------------------------------------------
def download_naip_chip(lat, lon, center_date, out_path):
    notify(f"Fetching NAIP around {center_date} at ({lat}, {lon})")

    start, end = expand_date(center_date)
    roi = make_256_roi(lat, lon, SCALE, EXPORT_SIZE)

    col = (
        ee.ImageCollection("USDA/NAIP/DOQQ")
        .filterBounds(roi)
        .filterDate(start, end)
        .sort("system:time_start")
    )

    count = col.size().getInfo()
    notify(f"NAIP images found in window: {count}")

    if count == 0:
        notify("No NAIP found — skipping.")
        return False

    img = col.median().select(["R", "G", "B", "N"]).clip(roi)

    notify("Requesting 256x256 download URL from GEE...")

    url = img.getDownloadURL({
        "scale": SCALE,
        "crs": CRS,
        "region": roi,
        "filePerBand": False,
        "format": "GEO_TIFF"
    })

    r = requests.get(url, stream=True, timeout=120)
    if r.status_code != 200:
        notify(f"Download failed: HTTP {r.status_code}")
        return False

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    with open(out_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)

    notify(f"Saved: {out_path}")
    return True

# ---------------------------------------------------------
# MAIN PIPELINE
# ---------------------------------------------------------
def main():
    initialize_gee()

    notify(f"Loading centroid index from {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)

    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    for idx, row in df.iterrows():
        lat = row["centroid_lat"]
        lon = row["centroid_lon"]
        loc_id = f"{lat:.4f}_{lon:.4f}"

        loc_dir = os.path.join(OUTPUT_ROOT, loc_id)
        os.makedirs(loc_dir, exist_ok=True)

        print(f"\n===== LOCATION: {loc_id} =====")

        # Take first two timestamps as T1/T2
        subset = df[(df["centroid_lat"] == lat) & (df["centroid_lon"] == lon)]

        dates = subset["date"].tolist()
        if len(dates) < 2:
            notify("Not enough timestamps — skipping.")
            continue

        t1, t2 = dates[0], dates[1]

        print("\n--- Downloading Pair ---")
        print(f"[INFO] T1: {t1}")
        print(f"[INFO] T2: {t2}")

        t1_path = os.path.join(loc_dir, "T1_256.tif")
        t2_path = os.path.join(loc_dir, "T2_256.tif")

        ok1 = download_naip_chip(lat, lon, t1, t1_path)
        ok2 = download_naip_chip(lat, lon, t2, t2_path)

        if not (ok1 and ok2):
            notify("Skipping this location due to failure.")
            continue

        notify(f"Pair downloaded successfully for {loc_id}")

    notify("ALL 256x256 DOWNLOADS COMPLETED.")

# ---------------------------------------------------------
if __name__ == "__main__":
    main()
