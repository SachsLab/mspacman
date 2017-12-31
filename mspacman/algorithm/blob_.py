import numpy as np
from scipy.ndimage.filters import sobel
# Authors : David C.C. Lu <davidlu89@gmail.com>
#
# License : BSD (3-clause)

def detect_blob(comod, threshold=.5, vmin=0, vmax=1):

    if comod.ndim != 2:
        raise ValueError("'comod' must be 2d ndarray.")

    # Normalizing the comod
    _comod = comod[np.newaxis,:,:] if comod.ndim < 3 else comod
    _comod -= vmin
    _comod /= (vmax - vmin)

    d1, d2, d3 = _comod.shape

    tmpx = np.zeros((d1, d2, d3))
    tmpy = np.zeros((d1, d2, d3))

    sobel(_comod, axis=-1, output=tmpx)
    sobel(_comod, axis=1, output=tmpy)

    mag = np.abs(tmpx + 1j*tmpy)

    mag /= mag.max()
    mag[mag<=threshold] = 0
    mag[mag>threshold] = 1

    indices = np.where(mag>0)

    # Calculate dimensional properties of the blob
    width = indices[1].max()-indices[1].min()
    height = indices[-1].max()-indices[-1].min()
    area = mag[mag==1].size

    compact = area / (width * height) if width and height else 0
    center = (int(np.median(indices[1])), int(np.median(indices[-1])))
    center_mass = (int(np.mean(indices[1])), int(np.mean(indices[-1])))

    dimension = {
        'width': int(width),
        'height': int(height),
        'area': int(area)
    }

    properties = {
        'center': center,
        'compact': compact,
        'center_mass': center_mass

    }

    return dimension, properties
