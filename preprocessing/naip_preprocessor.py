# ====================== CONFIG (ALL CAPS) ======================
RAW_DATA_ROOT = "raw_data"          # Your downloaded T1/T2 chips
OUTPUT_ROOT = "processed_dataset"   # Where .npy will be saved

PATCH_SIZE = 256
LEE_WINDOW = 7
EPSILON = 1e-6
CHANGE_THRESHOLD = 0.25
# ===================================================

import os
import numpy as np
import rasterio
from scipy.ndimage import uniform_filter

def notify(msg):
    print(f"[INFO] {msg}")

class NAIPPreprocessor:
    def __init__(self, patch_size=PATCH_SIZE, lee_window=LEE_WINDOW):
        self.patch_size = patch_size
        self.lee_window = lee_window

    # ---------- LOAD ----------
    def load_image(self, path):
        notify(f"Loading: {path}")
        with rasterio.open(path) as src:
            img = src.read().astype(np.float32)  # (C, H, W)
        return img

    # ---------- RGB → GRAYSCALE ----------
    def to_grayscale(self, img):
        if img.shape[0] >= 3:
            gray = 0.2989 * img[0] + 0.5870 * img[1] + 0.1140 * img[2]
        else:
            gray = img[0]
        return gray

    # ---------- LEE FILTER (DESPECKLING) ----------
    def lee_sigma_filter(self, img):
        mean = uniform_filter(img, size=self.lee_window)
        mean_sq = uniform_filter(img**2, size=self.lee_window)
        var = mean_sq - mean**2
        noise_var = np.mean(var)

        W = np.maximum(0, (var - noise_var) / (var + EPSILON))
        return mean + W * (img - mean)

    # ---------- LOG TRANSFORM ----------
    def to_db(self, img):
        return 10 * np.log10(np.maximum(img, EPSILON))

    # ---------- Z-SCORE NORMALIZATION ----------
    def normalize(self, img):
        return (img - np.mean(img)) / (np.std(img) + EPSILON)

    # ---------- CHANGE MASK (STRONGER) ----------
    def create_change_mask(self, img1, img2):
        diff = np.abs(img1 - img2)
        diff = (diff - diff.min()) / (diff.max() - diff.min() + EPSILON)

        # Adaptive threshold (better than fixed)
        thresh = np.percentile(diff, 75) * CHANGE_THRESHOLD
        mask = (diff > thresh).astype(np.float32)
        return mask

    # ---------- FULL PIPELINE ----------
    def process_pair(self, t1_path, t2_path, out_dir="processed_dataset"):
        os.makedirs(os.path.join(out_dir, "A"), exist_ok=True)
        os.makedirs(os.path.join(out_dir, "B"), exist_ok=True)
        os.makedirs(os.path.join(out_dir, "MASK"), exist_ok=True)

        img1 = self.load_image(t1_path)
        img2 = self.load_image(t2_path)

        img1 = self.to_grayscale(img1)
        img2 = self.to_grayscale(img2)

        img1 = self.lee_sigma_filter(img1)
        img2 = self.lee_sigma_filter(img2)

        img1 = self.to_db(img1)
        img2 = self.to_db(img2)

        img1 = self.normalize(img1)
        img2 = self.normalize(img2)

        mask = self.create_change_mask(img1, img2)

        # ensure 256×256
        img1 = img1[:256, :256]
        img2 = img2[:256, :256]
        mask = mask[:256, :256]

        base = os.path.basename(os.path.dirname(t1_path))

        np.save(os.path.join(out_dir, "A", f"{base}_T1.npy"), img1)
        np.save(os.path.join(out_dir, "B", f"{base}_T2.npy"), img2)
        np.save(os.path.join(out_dir, "MASK", f"{base}_mask.npy"), mask)

        notify(f"[DONE] Saved processed pair + mask for {base}")

pre = NAIPPreprocessor()

for loc in os.listdir(RAW_DATA_ROOT):
    loc_dir = os.path.join(RAW_DATA_ROOT, loc)

    t1_path = os.path.join(loc_dir, "T1_256.tif")
    t2_path = os.path.join(loc_dir, "T2_256.tif")

    if os.path.exists(t1_path) and os.path.exists(t2_path):
        print(f"\nProcessing location: {loc}")
        pre.process_pair(t1_path, t2_path, out_dir=OUTPUT_ROOT)
    else:
        print(f"[SKIP] Missing T1/T2 for {loc}")