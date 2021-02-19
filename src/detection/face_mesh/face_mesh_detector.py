import numpy as np, cv2
import tensorflow as tf
import os
from threading import Lock
# ################################
from .face_mesh import FaceMesh
from ..face import FaceLandmark
from .. import utils
# ################################


class FaceMeshDetector:
    NB_KP        = 468
    CROP_SIZE  = 192
    CROP_PADDING = 1.5

    INPUT_IDX       = 0    # name:'input_1', index:0, shape:(1, 192, 192, 3)
    OUTPUT_FLAG_IDX = 210  # name:'conv2d_31', index:210, shape:(1, 1, 1, 1)
    OUTPUT_KP_IDX   = 213  # name:'conv2d_21', index:213, shape:(1, 1, 1, 1404)

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

    def __call__(self, bgr_frame, face):
        # prepare input
        input, M, bgr_crop = self._prepare(bgr_frame, face)
        # process
        with self.lock:
            # set input tensors
            self.model.set_tensor(self.INPUT_IDX, input)
            # start processing
            self.model.invoke()
            # get result tensors
            flag      = self.model.get_tensor(self.OUTPUT_FLAG_IDX)[0]
            keypoints = self.model.get_tensor(self.OUTPUT_KP_IDX)[0]

        # post process
        roi, flag, keypoints = self._pprocess(flag, keypoints, M)

        return FaceMesh(roi, flag, keypoints, bgr_crop, M)
    # ################################


    def _prepare(self, bgr_frame, face):
        """ extract face ROI aligning eyes """
        # align & crop face
        bgr_crop, M = self._crop(bgr_frame, face)

        # input = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2RGB)
        input = bgr_crop
        
        # uint8 [0;255] to float32 [0;1]
        input = input.astype(np.float32) / 255.0
        # set as batch of 1
        input = np.expand_dims(input, 0)

        return input, M, bgr_crop
    # ################################

    def _crop(self, bgr_frame, face):
        # get angle that align eyes
        le_x, le_y = face.keypoints[FaceLandmark.eye_left]
        re_x, re_y = face.keypoints[FaceLandmark.eye_right]
        angle      = 180 * np.arctan2(re_y-le_y, re_x-le_x) / np.pi
        # compute scale
        scale      = self.CROP_SIZE / (max(face.width, face.height) * self.CROP_PADDING)
        # build rotation & scaling transform (preserves rotation center)
        R = cv2.getRotationMatrix2D((face.cx, face.cy), angle, scale)
        R = np.vstack((R, [0, 0, 1])) # 2x3 -> 3x3 matrix
        # build translation transform
        tx = (self.CROP_SIZE / 2) - face.cx
        ty = (self.CROP_SIZE / 2) - face.cy
        T  = np.float32([[1, 0, tx], [0, 1, ty], [0, 0, 1]])
        # combine transforms
        M = np.dot(T, R)
        # apply transform
        crop = cv2.warpPerspective(
            bgr_frame, M, (self.CROP_SIZE, self.CROP_SIZE),
            cv2.INTER_LINEAR)

        return crop, M
    # ################################

    def _pprocess(self, flag, keypoints, M):
        # apply sigmoid activation function
        flag = utils.sigmoid(float(flag.flatten()), 100)

        # invert transform
        Mi = np.linalg.inv(M)

        # reshape & drop Z coordinate
        keypoints = keypoints.reshape((self.NB_KP, 3))[:,:2]
        # correct points (project back in original image)
        keypoints = cv2.perspectiveTransform(keypoints[None,:,:], Mi)[0]

        # build the ROI 
        roi = utils.original_roi(self.CROP_SIZE, Mi)

        return roi, flag, keypoints
    # ################################
