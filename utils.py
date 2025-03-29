import cv2
import numpy as np
import math

def DarkChannel(im, sz=15):
    """
    Compute the dark channel of an image.
    :param im: Input image (H, W, C)
    :param sz: Size of the structuring element for erosion
    :return: Dark channel image
    """
    b, g, r = cv2.split(im)
    dc = cv2.min(cv2.min(r, g), b)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (sz, sz))
    dark = cv2.erode(dc, kernel)
    return dark

def AtmLight(im, dark):
    """
    Estimate atmospheric light from the image.
    :param im: Input image (H, W, C)
    :param dark: Dark channel image
    :return: Estimated atmospheric light A (1, 3)
    """
    h, w = im.shape[:2]
    imsz = h * w
    numpx = max(math.floor(imsz / 1000), 1)  # Top 0.1% brightest pixels
    darkvec = dark.reshape(imsz)
    imvec = im.reshape(imsz, 3)

    indices = darkvec.argsort()[imsz - numpx:]  # Indices of brightest pixels
    atmsum = np.zeros([1, 3])

    for ind in indices:
        atmsum += imvec[ind]

    A = atmsum / numpx  # Average atmospheric light
    return A
