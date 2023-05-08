from skimage.color import rgba2rgb, rgb2hsv
from skimage.exposure import rescale_intensity
import numpy as np


def preprocess(lineImage: np.ndarray) -> np.ndarray:
    if lineImage.shape[-1] > 3:
        lineImage = rgba2rgb(lineImage)
    greyscale = 1.0 - rgb2hsv(lineImage)[:, :, 2]

    p2, p98 = np.percentile(greyscale, (2, 98))
    return rescale_intensity(greyscale, in_range=(p2, p98))
