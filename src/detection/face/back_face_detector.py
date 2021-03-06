import numpy as np, cv2
import tensorflow as tf
import os
from threading import Lock
from itertools import starmap
# ################################
from .face import Face
from ..ssd_anchors_calculator import ssd_anchors_calculator
from ..tensors_to_detections_calculator import tensors_to_detections_calculator
from ..faster_nms import faster_nms
# ################################


class BackFaceDetector:
    """
        MediaPipe Face Detector
    """
    INPUT_IDX    = 0
    OUTPUT_C_IDX = 283
    OUTPUT_R_IDX = 284
    
    CROP_SIZE = 256

    ANCHOR_PARAMS = dict(
        num_layers          = 4,
        min_scale           = 0.15625,
        max_scale           = 0.75,
        input_size_height   = 256,
        input_size_width    = 256,
        anchor_offset_x     = 0.5,
        anchor_offset_y     = 0.5,
        strides             = [16, 32, 32, 32],
        aspect_ratios       = [1.0],
        fixed_anchor_size   = True
    )

    DECODE_PARAMS = dict(
        num_classes             = 1,
        num_boxes               = 896,
        num_coords              = 16,
        box_coord_offset        = 0,
        keypoint_coord_offset   = 4,
        num_keypoints           = 6,
        num_values_per_keypoint = 2,
        sigmoid_score           = True,
        score_clipping_thresh   = 100.0,
        reverse_output_order    = True,
        x_scale                 = 256.0,
        y_scale                 = 256.0,
        h_scale                 = 256.0,
        w_scale                 = 256.0
    )

    def __init__(self, model_path: str, score_th: float=0.65, nms_iou_th: float=0.3):
        assert os.path.exists(model_path)
        ## tflite model path
        self.model_path = model_path
        ## lock to protect model usage
        self.lock = Lock()
        with self.lock:
            ## tflite model
            self.model = tf.lite.Interpreter(model_path=model_path)
            self.model.allocate_tensors()

        ## ssd anchors to decode predictions
        self.anchors = ssd_anchors_calculator(**self.ANCHOR_PARAMS)
        ## min score th
        self.score_th = score_th
        # min nms iou th
        self.nms_iou_th = nms_iou_th
    # ################################

    def __call__(self, bgr_frame):
        # prepare input
        input, scale = self._prepare(bgr_frame)
        # process
        with self.lock:
            # set input tensors
            self.model.set_tensor(self.INPUT_IDX, input)
            # start processing
            self.model.invoke()
            # get result tensors
            classificators  = self.model.get_tensor(self.OUTPUT_C_IDX)[0]
            regressors      = self.model.get_tensor(self.OUTPUT_R_IDX)[0]
        # post process
        faces = self._pprocess(classificators, regressors, scale)

        return faces
    # ################################

    def _prepare(self, bgr_frame):
        """ proportional scalin to 256x256 pxl with black padding 
            cast to float32 on [-1;+1]
        """
        h, w  = bgr_frame.shape[:2]
        scale = self.CROP_SIZE / max(h, w)
        input = cv2.resize(bgr_frame, (0, 0), 
            fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        h, w  = input.shape[:2]
        # input = cv2.cvtColor(input, cv2.COLOR_BGR2RGB)
        input = cv2.copyMakeBorder(input, 
            0, self.CROP_SIZE-h, 0, self.CROP_SIZE-w, 
            cv2.BORDER_CONSTANT, 0)
        # bgr -> rgb
        input = cv2.cvtColor(input, cv2.COLOR_BGR2RGB)
        # uint8 [0;255] to float32 [-1;+1]
        input = input.astype(np.float32) / 128.0 - 1.0
        # set as batch of 1
        input = np.expand_dims(input, 0)

        return input, scale
    # ################################

    def _pprocess(self, classificators, regressors, scale):
        # decode predictions
        _, scores, boxes, keypoints = tensors_to_detections_calculator(
            classificators, regressors, 
            self.anchors, **self.DECODE_PARAMS, 
            min_score_thresh = self.score_th)
        
        # NMS
        if len(scores):
            idx         = np.argsort(scores)
            keep_idx    = faster_nms(boxes, idx, self.nms_iou_th)
            scores      = scores[keep_idx]
            boxes       = boxes[keep_idx]
            keypoints   = keypoints[keep_idx]

            # denormalization
            boxes       = boxes * self.CROP_SIZE / scale
            keypoints   = keypoints * self.CROP_SIZE / scale

            # format to objects
            faces = list(starmap(Face, zip(scores, boxes, keypoints)))
        else:
            faces = []

        return faces
    # ################################
