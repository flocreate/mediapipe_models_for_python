import numpy as np
from . import utils
# ################################


def tensors_to_detections_calculator(
    classificators, regressors, anchors,
    num_classes: int, # not used
    num_boxes: int, # not used
    num_coords: int, # not used
    keypoint_coord_offset: int=None, # not used
    num_keypoints: int=0,   # not used
    num_values_per_keypoint: int = 2,
    box_coord_offset: int=0, # not implemented
    x_scale: float=0.0,
    y_scale: float=0.0,
    w_scale: float=0.0,
    h_scale: float=0.0,
    apply_exponential_on_box_size: bool=False,
    reverse_output_order: bool=False,
    ignore_classes: list=[],    # not implemented
    sigmoid_score: bool=False,
    score_clipping_thresh: float=None,
    flip_vertically: bool=False, # not implemented
    min_score_thresh: float=None
    ):
    """ Porting MediaPipe Cpp Code

        https://github.com/google/mediapipe/blob/master/mediapipe/calculators/tflite/tflite_tensors_to_detections_calculator.proto

        https://github.com/google/mediapipe/blob/master/mediapipe/calculators/tflite/tflite_tensors_to_detections_calculator.cc

        returns 
            - normalized boxes      <x0, y0, x1, y1>
            - normalized landmarks  <x0, y0> * 6
    """
    # find top score & label
    labels = np.argmax(classificators, axis=1).flatten()
    scores = np.amax(classificators, axis=1).flatten()

    if sigmoid_score:
        scores = utils.sigmoid(scores, score_clipping_thresh)

    # select only pertinent items
    keep        = (scores >= min_score_thresh)
    labels      = labels[keep]
    scores      = scores[keep]
    regressors  = regressors[keep]
    anchors     = anchors[keep]

    if len(labels):
        # decode boxes
        boxes = decode_boxes(regressors, anchors,
            reverse_output_order, apply_exponential_on_box_size,
            x_scale, y_scale, w_scale, h_scale, flip_vertically)
        # decode key-points
        keypoints = decode_keypoints(regressors, anchors, x_scale, y_scale)

    else:
        boxes = []
        keypoints = []

    return labels, scores, boxes, keypoints
# ################################

def decode_boxes(regressors, anchors, 
    reverse_output_order, apply_exponential_on_box_size,
    x_scale, y_scale, w_scale, h_scale, flip_vertically):

    # TODO: implement flip_vertically

    if not reverse_output_order:
        bcy  = regressors[:,0]
        bcx  = regressors[:,1]
        bh   = regressors[:,2]
        bw   = regressors[:,3]
    else:
        bcx  = regressors[:,0]
        bcy  = regressors[:,1]
        bw   = regressors[:,2]
        bh   = regressors[:,3]
    
    acy = anchors[:,0]
    acx = anchors[:,1]
    ah  = anchors[:,2]
    aw  = anchors[:,3]

    bcx = bcx / x_scale * aw + acx
    bcy = bcy / y_scale * ah + acy

    if apply_exponential_on_box_size:
        bh = np.exp(bh / h_scale) * ah
        bw = np.exp(bw / w_scale) * aw
    else:
        bh = bh / h_scale * ah
        bw = bw / w_scale * aw

    x0 = bcx - bw / 2
    y0 = bcy - bh / 2
    x1 = bcx + bw / 2
    y1 = bcy + bh / 2

    boxes = np.stack((x0, y0, x1, y1), 1)

    return boxes
# ################################


def decode_keypoints(regressors, anchors, x_scale, y_scale):
    acy = anchors[:,0]
    acx = anchors[:,1]
    ah  = anchors[:,2]
    aw  = anchors[:,3]

    keypoints = regressors[:,4:]
    keypoints = keypoints.reshape(len(keypoints), -1, 2)
    keypoints = keypoints / (x_scale, y_scale)
    keypoints = keypoints * np.expand_dims(np.stack((aw, ah), 1), 1)
    keypoints = keypoints + np.expand_dims(np.stack((acx, acy), 1), 1)

    return keypoints
# ################################
