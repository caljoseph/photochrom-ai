import numpy as np
from skimage.color import lab2rgb


def lab_to_rgb(L, ab):
    """
    Convert normalized L and ab tensors to RGB.

    L: [1, H, W] in [0, 100]
    ab: [2, H, W] in [-1, 1]

    Returns: (H, W, 3) RGB image as numpy array
    """
    L = L[0].cpu().numpy()               # [H, W]
    ab = ab.cpu().numpy() * 128.0        # [2, H, W] → denormalize
    ab = ab.transpose(1, 2, 0)           # [H, W, 2]
    lab = np.concatenate([L[:, :, None], ab], axis=2)  # [H, W, 3]
    rgb = lab2rgb(lab)                   # Returns float64 in [0,1]
    return rgb
