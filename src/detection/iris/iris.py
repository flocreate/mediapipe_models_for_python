import numpy as np, cv2
from enum import IntEnum
# ################################
from .. import utils
# ################################

class IrisLandmark(IntEnum):
    # Iris part
    iris_center = 0
    # Eye part
# ################################



class Iris:
    def __init__(self, roi, eye, iris, bgr_crop, transform):
        ## roi in the original image
        self.roi = roi
        ## eye landmarks
        self.eye = eye
        ## iris landmarks
        self.iris = iris
        ## iris radius
        self.iris_radius = np.linalg.norm(self.iris[1:] - self.iris[0], axis=1).mean()
        ## extracted face crop
        self.bgr_crop = bgr_crop
        ## 3x3 matrix transform to project image space to crop space
        self.transform = transform
    # ################################

    def draw(self, frame, color_box=None, color_eye=None, color_iris=None):
        cv2.polylines(frame, [self.roi.astype(np.int32)], True, color_box or (255, 0, 255), 2)
        # utils.draw_landmarks(frame, self.eye, None, color_eye or (255, 0, 0), 2, -1)
        # utils.draw_landmarks(frame, self.iris, None, color_iris or (0, 0, 255), 2, -1)

        cv2.circle(frame, tuple(self.iris[IrisLandmark.iris_center].astype(int)), 1, (255, 0, 255), -1)
        cv2.circle(frame, tuple(self.iris[IrisLandmark.iris_center].astype(int)), int(self.iris_radius), (255, 0, 255), 1)

        # for eid, (x, y) in enumerate(self.eye.astype(int)):
        #     cv2.putText(frame, str(eid), (x, y), 0, 0.2, (255, 255, 255), 1)
    # ################################
