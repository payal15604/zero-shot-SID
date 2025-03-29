import numpy as np
from skimage.morphology import square, closing
from airlight import airlight
from cal_transmission import cal_trans
from defog import defog
from skimage.morphology import footprint_rectangle, closing
import cv2

def bounding_function(I, zeta):
    print('entered bounding function')
    I = I.astype(np.float64) / 255.0
    min_I = np.min(I, axis=2)
    MAX = np.max(min_I)
    print('calculating airlight')
    A1 = airlight(I, 3)
    A = np.max(A1)
    print(A)
    delta = zeta / (min_I ** 0.5)
    epsilon = 1e-6  # Small value to avoid division by zero
    est_tr_proposed = 1 / (1 + (MAX * 10 ** (-0.05 * delta)) / (A - min_I + epsilon))
    #est_tr_proposed = np.clip(est_tr_proposed, 0.1, 1.0)  # Ensure valid range

    print('calculating transmission')
    tr1 = min_I >= A
    tr2 = min_I < A
    tr2 = np.abs(tr2 * est_tr_proposed)
    tr4 = np.abs(est_tr_proposed * tr1)
    
    tr3_max = np.max(tr4)
    if tr3_max == 0:
        tr3_max = 1
    
    tr3 = tr4 / tr3_max
    est_tr_proposed = tr2 + tr3
    display_width = 600  # Adjust as needed
    display_height = 400  # Adjust as needed
    resized_image = cv2.resize(est_tr_proposed, (display_width, display_height))
    print(est_tr_proposed)
    
    #cv2.imshow("Dehazed Image", resized_image)
    print("Max:", np.max(est_tr_proposed))
    print("Min:", np.min(est_tr_proposed))
    #cv2.waitKey(0)
    est_tr_proposed = closing(est_tr_proposed, footprint_rectangle((3,3)))
    est_tr_proposed = cal_trans(I, est_tr_proposed, 1, 0.5)
    
    print('defogging')
    r = defog(I, est_tr_proposed, A1, 0.9)
    r = r[..., [2, 1, 0]]  # Swap R and B back

    
    return r, est_tr_proposed, A
