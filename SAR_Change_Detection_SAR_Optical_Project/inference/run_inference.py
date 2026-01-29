
import numpy as np
from scipy.ndimage import binary_opening

def infer(img1, img2, percentile=90, kernel_size=3):
    diff = np.abs(img1 - img2)
    threshold = np.percentile(diff, percentile)
    change_map = (diff > threshold).astype(np.uint8)
    structure = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    return binary_opening(change_map, structure=structure)
