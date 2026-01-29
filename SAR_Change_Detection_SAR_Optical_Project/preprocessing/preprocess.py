
import numpy as np

def preprocess(img):
    img = np.maximum(img, 1e-6)
    img = np.log(img)
    return (img - img.mean()) / img.std()
