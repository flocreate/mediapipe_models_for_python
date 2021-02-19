import numpy as np, cv2
import tensorflow as tf
import os
from threading import Lock
# ################################
from ..face_mesh import FaceMeshLandmark
from .iris import Iris
from .. import utils
# ################################

class IrisDetector:
    """ Implements Mediapipe Iris
        https://drive.google.com/file/d/1bsWbokp9AklH2ANjCfmjqEzzxO1CNbMu/view

        Produces:
            - 71 refined normalized eye & brows contour landmarks
            - 5 normalized iris landmarks

        IrisLandmarkCpu
    """
    MODEL_INPUT  = 64
    CROP_PADDING = 2.3

    INPUT_IDX       = 0    # name:'input_1', index:0, shape:(1, 64, 64, 3)
    OUTPUT_ECAB_IDX = 384  # name:'output_eyes_contours_and_brows', index:384, shape:(1, 213)
    OUTPUT_IRIS_IDX = 385  # name:'output_iris', index:385, shape:(1, 15)

    def __init__(self, model_path):
        assert os.path.exists(model_path)
        ## tflite model path
        self.model_path = model_path
        ## lock to protect model usage
        self.lock = Lock()
        with self.lock:
            ## tflite model
            self.model = tf.lite.Interpreter(model_path=model_path)
            self.model.allocate_tensors()
    # ################################

    def __call__(self, bgr_frame, face_mesh):
        left  = self.process(bgr_frame, face_mesh, False)
        right = self.process(bgr_frame, face_mesh, True)

        return left, right
    # ################################

    def process(self, bgr_frame, face_mesh, right):
        # prepare input
        input, bgr_crop, M = self._prepare(bgr_frame, face_mesh, right)

        # process
        with self.lock:
            # set input tensors
            self.model.set_tensor(self.INPUT_IDX, input)
            # start processing
            self.model.invoke()
            # get result tensors
            ecab_kp = self.model.get_tensor(self.OUTPUT_ECAB_IDX)[0]
            iris_kp = self.model.get_tensor(self.OUTPUT_IRIS_IDX)[0]
        
        # post process
        data = self._pprocess(ecab_kp, iris_kp, M)
        # format
        iris = Iris(*data, bgr_crop, M)

        return iris
    # ################################

    def _prepare(self, bgr_frame, face_mesh, right):
        """ extract face ROI aligning eyes """
        # align & crop eyes
        bgr_crop, M = self._crop(bgr_frame, face_mesh, right)

        # input = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2RGB)
        input = bgr_crop

        # uint8 [0;255] to float32 [0;1]
        input = input.astype(np.float32) / 255.0
        # set as batch of 1
        input = np.expand_dims(input, 0)

        return input, bgr_crop, M
    # ################################

    def _crop(self, bgr_frame, face_mesh, right=False):
        # get landmarks depending on the eye
        if not right:
            lx, ly = face_mesh.keypoints[FaceMeshLandmark.eye_left_outer]
            rx, ry = face_mesh.keypoints[FaceMeshLandmark.eye_left_inner]
        else:
            lx, ly = face_mesh.keypoints[FaceMeshLandmark.eye_right_inner]
            rx, ry = face_mesh.keypoints[FaceMeshLandmark.eye_right_outer]
        
        # get angle that align eye corners
        angle = 180 * np.arctan2(ry-ly, rx-lx) / np.pi
        # get eye center
        cx    = (lx + rx) / 2
        cy    = (ly + ry) / 2
        # compute scale
        width = np.linalg.norm((rx-lx+1, ry-ly+1))
        scale = self.MODEL_INPUT / (width * self.CROP_PADDING)
        # build rotation & scaling transform (preserves rotation center)
        R = cv2.getRotationMatrix2D((cx, cy), angle, scale)
        R = np.vstack((R, [0, 0, 1])) # 2x3 -> 3x3 matrix        
        # build translation transform
        if not right:
            # for left eye, no hflip, only need a simple translation
            ty = (self.MODEL_INPUT / 2) - cy
            tx = (self.MODEL_INPUT / 2) - cx
            T  = np.float32([[1, 0, tx], [0, 1, ty], [0, 0, 1]])
        else:
            # for right eye, need to chain transforms
            # 1) translate so the center becomes 0
            T0 = np.float32([[1, 0, -cx], [0, 1, -cy], [0, 0, 1]])
            # 2) hflip & translate back to crop center
            c  = self.MODEL_INPUT / 2
            T1 = np.float32([[-1, 0, c], [0, 1, c], [0, 0, 1]])
            # 3) combine T1 & T2
            T = np.dot(T1, T0)

        # combine transforms
        M = np.dot(T, R)
        # apply transform
        crop = cv2.warpPerspective(
            bgr_frame, M, (self.MODEL_INPUT, self.MODEL_INPUT),
            cv2.INTER_LINEAR)

        return crop, M
    # ################################

    def _pprocess(self, ecab_kp, iris_kp, M):
        Mi = np.linalg.inv(M)

        ecab_kp     = ecab_kp.reshape((-1, 3))
        eye_kp      = ecab_kp[:,:2]
        eye_kp      = cv2.perspectiveTransform(eye_kp[None,:,:], Mi)[0]

        iris_kp     = iris_kp.reshape((-1, 3))
        iris_kp     = iris_kp[:,:2]
        iris_kp     = cv2.perspectiveTransform(iris_kp[None,:,:], Mi)[0]

        roi = utils.original_roi(self.MODEL_INPUT, Mi)

        return roi, eye_kp, iris_kp
    # ################################
