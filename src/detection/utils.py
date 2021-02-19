import numpy as np, cv2
from itertools import zip_longest
# ################################


def sigmoid(value, clip=None):
    if clip is not None:
        value = np.clip(value, -clip, clip)
    return 1 / (1 + np.exp(-value))
# ################################


def score_color(score):
    return tuple(map(int, cv2.applyColorMap(
                np.array([score * 255 / 2], np.uint8), 
                cv2.COLORMAP_RAINBOW).flatten()))
# ################################


def draw_landmarks(frame, landmarks, scores=None, color=None, radius=3, thickness=-1):
    assert (scores is not None) or (color is not None), 'requires scores or color'

    for (x, y), score in zip_longest(
        landmarks.astype(int), 
        scores if scores is not None else []):
        
        color_ = color or score_color(score)
        cv2.circle(frame, (x, y), radius, color_, thickness)
# ################################


def original_roi(size, Mi):
    roi = np.float32([[0, 0], [0, size], [size, size], [size, 0]])
    roi = cv2.perspectiveTransform(roi[None,:,:], Mi)[0]
    return roi
# ################################
