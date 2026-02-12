# ====================== CONFIG (ALL CAPS) ======================

GEE_PROJECT_ID = "my-first-project-482818"

# ----- GEOFENCE -----
START_YEAR = 2018
END_YEAR = 2022
OUTPUT_GEOJSON = "naip_geofence_index.geojson"
OUTPUT_CSV = "naip_geofence_index.csv"

# ----- DATA COLLECTION -----
LOCATIONS = [
    {"name": "kansas_city", "lat": 39.0997, "lon": -94.5786},
    {"name": "denver", "lat": 39.7392, "lon": -104.9903},
    {"name": "omaha", "lat": 41.2565, "lon": -95.9345}
]

TIMESTAMP_PAIRS = [
    ("2018-01-01", "2018-12-31", "2022-01-01", "2022-12-31"),
    ("2015-01-01", "2015-12-31", "2020-01-01", "2020-12-31")
]

BUFFER_METERS = 2000
SCALE = 1
CRS = "EPSG:4326"

PATCH_SIZE = 256
STRIDE = 192

RAW_DIR = "raw_downloads"
DATASET_DIR = "dataset"

# ----- PREPROCESSING -----
LEE_WINDOW = 7
EPSILON = 1e-6

# ===============================================================
