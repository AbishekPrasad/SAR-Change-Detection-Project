
import ee
import matplotlib.pyplot as plt
from gee.gee_fetch import fetch_sentinel1, fetch_sentinel2_rgb
from utils.gee_utils import ee_image_to_numpy, ee_image_to_numpy_rgb
from preprocessing.preprocess import preprocess
from inference.run_inference import infer

ee.Initialize(project="my-first-project-482818")

def run(lat, lon, t1s, t1e, t2s, t2e):
    # SAR
    sar1, region = fetch_sentinel1(lat, lon, t1s, t1e)
    sar2, _ = fetch_sentinel1(lat, lon, t2s, t2e)

    # Optical
    opt1, _ = fetch_sentinel2_rgb(lat, lon, t1s, t1e)
    opt2, _ = fetch_sentinel2_rgb(lat, lon, t2s, t2e)

    # Convert to NumPy
    img1 = preprocess(ee_image_to_numpy(sar1, region))
    img2 = preprocess(ee_image_to_numpy(sar2, region))

    rgb1 = ee_image_to_numpy_rgb(opt1, region)
    rgb2 = ee_image_to_numpy_rgb(opt2, region)

    change_map = infer(img1, img2)

    return img1, img2, rgb1, rgb2, change_map


def display_results(img1, img2, rgb1, rgb2, change_map):
    plt.figure(figsize=(22, 8))

    plt.subplot(2, 3, 1)
    plt.imshow(img1, cmap="gray")
    plt.title("SAR Time 1")
    plt.axis("off")

    plt.subplot(2, 3, 2)
    plt.imshow(img2, cmap="gray")
    plt.title("SAR Time 2")
    plt.axis("off")

    plt.subplot(2, 3, 3)
    plt.imshow(change_map, cmap="gray")
    plt.title("Change Map")
    plt.axis("off")

    plt.subplot(2, 3, 4)
    plt.imshow(rgb1)
    plt.title("Optical Time 1 (Sentinel-2)")
    plt.axis("off")

    plt.subplot(2, 3, 5)
    plt.imshow(rgb2)
    plt.title("Optical Time 2 (Sentinel-2)")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    img1, img2, rgb1, rgb2, result = run(
        lat=10.755143021531744, 
        lon=78.6555948204604,
        t1s="2023-01-01",
        t1e="2023-02-28",
        t2s="2026-01-01",
        t2e="2026-02-28"
    )
    display_results(img1, img2, rgb1, rgb2, result)
