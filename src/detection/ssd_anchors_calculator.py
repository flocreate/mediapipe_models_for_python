import numpy as np
# ################################

def ssd_anchors_calculator(
        num_layers: int,
        input_size_width: int,
        input_size_height: int,
        min_scale: float, 
        max_scale: float,
        anchor_offset_x: float = 0.5,
        anchor_offset_y: float = 0.5,
        feature_map_width: list = [],
        feature_map_height: list = [],
        strides: list = [],
        aspect_ratios: list = [],
        reduce_boxes_in_lowest_layer: bool = False,
        interpolated_scale_aspect_ratio: float = 1.0,
        fixed_anchor_size: bool = False
    ):
    """ Porting MediaPipe Cpp Code
        
        returns list of anchors [(cy, cx, h, w), ...]

        https://github.com/google/mediapipe/blob/master/mediapipe/calculators/tflite/ssd_anchors_calculator.proto
        
        https://github.com/google/mediapipe/blob/master/mediapipe/calculators/tflite/ssd_anchors_calculator.cc
    """
    anchors = []
    layer_id = 0
    while (layer_id < num_layers):
        anchor_height   = []
        anchor_width    = []
        aspect_ratios_  = []
        scales          = []

        # For same strides, we merge the anchors in the same order.
        last_same_stride_layer = layer_id
        while (last_same_stride_layer < len(strides)) and \
            (strides[last_same_stride_layer] == strides[layer_id]):

            scale = calculate_scale(min_scale, max_scale, 
                last_same_stride_layer, len(strides))

            if (last_same_stride_layer == 0) and reduce_boxes_in_lowest_layer:
                # For first layer, it can be specified to use predefined anchors.
                aspect_ratios_.append(1.0)
                aspect_ratios_.append(2.0)
                aspect_ratios_.append(0.5)
                scales.append(0.1)
                scales.append(scale)
                scales.append(scale)
            else:
                
                aspect_ratios_.extend(aspect_ratios)
                scales.extend([scale] * len(aspect_ratios))
                
                if (interpolated_scale_aspect_ratio > 0.0):
                    if last_same_stride_layer == (len(strides) - 1):
                        scale_next = 1.0
                    else:
                        scale_next = calculate_scale(min_scale, max_scale,
                            last_same_stride_layer + 1, len(strides))

                    scales.append(np.sqrt(scale * scale_next))
                    aspect_ratios_.append(interpolated_scale_aspect_ratio)

            last_same_stride_layer += 1

        anchor_height = np.array(scales) / np.sqrt(aspect_ratios_)
        anchor_width  = np.array(scales) * np.sqrt(aspect_ratios_)

        feature_map_height_ = 0
        feature_map_width_ = 0
        if len(feature_map_height):            
            feature_map_height_ = feature_map_height[layer_id]
            feature_map_width_  = feature_map_width[layer_id]
        else:
            stride = strides[layer_id]
            feature_map_height_ = int(np.ceil(input_size_height / stride))
            feature_map_width_  = int(np.ceil(input_size_width / stride))

        for y in range(feature_map_height_):
            y_center = (y + anchor_offset_y) / feature_map_height_

            for x in range(feature_map_width_):
                x_center = (x + anchor_offset_x) / feature_map_width_

                for ah, aw in zip(anchor_height, anchor_width):
                    if fixed_anchor_size:
                        new_anchor = (y_center, x_center, 1.0, 1.0)
                    else:
                        new_anchor = (y_center, x_center, ah, aw)

                    anchors.append(new_anchor)

        layer_id = last_same_stride_layer

    return np.array(anchors)
# ################################


def calculate_scale(min_scale, max_scale, stride_index, num_strides):
    if (num_strides == 1):
        return (min_scale + max_scale) / 2
    else:
        return min_scale + (max_scale - min_scale) * 1.0 * stride_index / (num_strides - 1)
# ################################
