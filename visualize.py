TIF_PATH = "t1_t2_patches/38.9688_-122.9687/T2_256.tif"

import rasterio
import numpy as np

path = TIF_PATH

with rasterio.open(path) as src:
    print("Number of bands:", src.count)
    print("Width, Height:", src.width, src.height)
    print("Data type:", src.dtypes)
    print("Min pixel value:", np.min(src.read()))
    print("Max pixel value:", np.max(src.read()))

import rasterio
import matplotlib.pyplot as plt
import numpy as np

with rasterio.open(path) as src:
    r = src.read(1)
    g = src.read(2)
    b = src.read(3)

rgb = np.dstack([r, g, b])
rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min())  # normalize to 0–1

plt.imshow(rgb)
plt.axis("off")
plt.show()