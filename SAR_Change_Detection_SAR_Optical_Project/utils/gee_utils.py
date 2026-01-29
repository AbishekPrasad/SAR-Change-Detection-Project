
import numpy as np
import requests, io

def ee_image_to_numpy(ee_image, region, scale=10):
    url = ee_image.getDownloadURL({"scale": scale, "region": region, "format": "NPY"})
    data = np.load(io.BytesIO(requests.get(url).content))
    if data.dtype.names:
        data = data["VV"]
    return data.astype(np.float32)

def ee_image_to_numpy_rgb(ee_image, region, scale=10):
    url = ee_image.getDownloadURL({"scale": scale, "region": region, "format": "NPY"})
    data = np.load(io.BytesIO(requests.get(url).content))
    r, g, b = data["B4"], data["B3"], data["B2"]
    rgb = np.stack([r, g, b], axis=-1)
    rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-6)
    return rgb
