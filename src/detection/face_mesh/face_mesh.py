import numpy as np, cv2
from enum import IntEnum
# ################################
from .. import utils
# ################################

class FaceMeshLandmark(IntEnum):
    eye_left_outer  = 33
    eye_left_inner  = 133

    eye_right_outer = 263
    eye_right_inner = 362
# ################################

class FaceMesh:
    def __init__(self, roi, score, keypoints, bgr_crop, transform):
        ## 4 points roi
        self.roi        = roi
        ## face confidence score
        self.score      = score
        ## 2D keypoints in the original image
        self.keypoints  = keypoints
        ## extracted face crop
        self.bgr_crop   = bgr_crop
        ## 3x3 matrix transform to project image space to crop space
        self.transform  = transform
    # ################################

    def draw(self, frame, color=None):
        color_ = color or utils.score_color(self.score)

        # ROI
        cv2.polylines(frame, [self.roi.astype(np.int32)], True, color_, 2)

        utils.draw_landmarks(frame, self.keypoints, color=color_, radius=2, thickness=-1)

        for lm in FaceMeshLandmark:
            x0, y0 = self.keypoints[lm].astype(int)
            color_ = (255, 0, 0) if lm.name.endswith('outer') else (0, 0, 255)
            cv2.circle(frame, (x0, y0), 4, color_, -1)
    # ################################



    # example to how to draw poly-lines in python
    # l_line = self.keypoints[
    #     min(FaceMeshLandmark.eye_left_0, FaceMeshLandmark.eye_left_1):
    #     max(FaceMeshLandmark.eye_left_0, FaceMeshLandmark.eye_left_1) + 1
    # ].reshape((-1, 1, 2)).astype(np.int32)
    # r_line = self.keypoints[
    #     min(FaceMeshLandmark.eye_right_0, FaceMeshLandmark.eye_right_1):
    #     max(FaceMeshLandmark.eye_right_0, FaceMeshLandmark.eye_right_1) + 1
    # ].reshape((-1, 1, 2)).astype(np.int32)
    # cv2.polylines(frame, [l_line, r_line], False, (0, 0, 0), 1)
