import numpy as np
import cv2
from enum import IntEnum
# ################################
from .. import utils
# ################################


class FaceLandmark(IntEnum):
    eye_left            = 0
    eye_right           = 1
    nose_tip            = 2
    mouth               = 3
    eye_tragion_left    = 4
    eye_tragion_right   = 5
# ################################


class Face:
    def __init__(self, score, box, keypoints):
        self.score      = score
        self.box        = box
        self.keypoints  = keypoints
    # ################################


    @property
    def x0(self): return self.box[0]
    @property
    def y0(self): return self.box[1]
    @property
    def x1(self): return self.box[2]
    @property
    def y1(self): return self.box[3]
    @property
    def cx(self): return (self.x0 + self.x1) / 2
    @property
    def cy(self): return (self.y0 + self.y1) / 2
    @property
    def center(self): return (self.cx, self.cy)
    @property
    def p0(self): return (self.x0, self.y0)
    @property
    def p1(self): return (self.x1, self.y1)
    @property
    def width(self): return self.x1 - self.x0 + 1
    @property
    def height(self): return self.y1 - self.y0 + 1
    # ################################


    def draw(self, frame, color=None):
        color_ = color or utils.score_color(self.score)

        # draw the box
        x0, y0, x1, y1 = self.box.astype(int)
        cv2.rectangle(frame, (x0, y0), (x1, y1), color_, 2)
        
        # draw landmarks
        utils.draw_landmarks(frame, self.keypoints, color=color_, radius=2)
    # ################################
