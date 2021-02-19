import numpy as np
# #################################


def faster_nms(boxes, idx=None, iou_th=0.5):
    """ return idx of boxes to keep
        inputs:
            boxes:      [(x0, y0, x1, y1), ...]
            idx:        index of boxes to use
        returns:
            keep_idx:   index of boxes to keep

        inputs (boxes & idx) are not modified
        if idx is not provided, all boxes will be used
        boxes (or idx) must be sorted BEST LAST
    """

    # if no idx provided use all
    if idx is None:
        idx = np.arange(len(boxes))
    else:
        idx = idx.copy()

    keep_idx = []
    while len(idx):
        # keep last box
        keep_idx.append(idx[-1])
        # compute IOU for all remaining boxes
        scores = iou(boxes[idx[-1]], boxes[idx])
        # keep only remaning idx that has low iou
        idx    = idx[scores < iou_th]

    return keep_idx
# #################################


def iou(box, boxes):
    # compute intersection area
    i_tl = np.maximum(box[:2], boxes[:,:2])
    i_br = np.minimum(box[2:], boxes[:,2:])
    i_sz = np.clip(i_br - i_tl + 1, 0, None)
    i_a  = np.prod(i_sz, axis=1)

    # compute union area
    u_tl = np.minimum(box[:2], boxes[:,:2])
    u_br = np.maximum(box[2:], boxes[:,2:])
    u_sz = np.clip(u_br - u_tl + 1, 0, None)
    u_a  = np.prod(u_sz, axis=1)

    # compute iou
    iou = i_a / u_a

    return iou
# #################################


