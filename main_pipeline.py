import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

import os
from preprocessing.naip_preprocessor import NAIPPreprocessor

RAW_ROOT = "raw_data"
PROCESSED_ROOT = "processed_dataset"

processor = NAIPPreprocessor()

for loc in os.listdir(RAW_ROOT):
    loc_dir = os.path.join(RAW_ROOT, loc)

    t1_path = os.path.join(loc_dir, "T1_256.tif")
    t2_path = os.path.join(loc_dir, "T2_256.tif")

    if not (os.path.exists(t1_path) and os.path.exists(t2_path)):
        print(f"[WARN] Missing T1/T2 for {loc}, skipping.")
        continue

    processor.process_pair(t1_path, t2_path, out_dir=PROCESSED_ROOT)

print("[DONE] All locations processed.")
