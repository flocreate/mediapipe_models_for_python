import numpy as np, cv2
import os
import argparse as ap
# ################################
from src.chronometer import Chronometer
from src.detection.palm import PalmDetector
# ################################


# get stream id from command line
parser = ap.ArgumentParser()
parser.add_argument('-i', '--input_stream', default='0')
parser.add_argument('-m', '--model_path', help='tflite model file',
    default=os.path.join('data', 'models', 'palm_detection.tflite'))
args = parser.parse_args()
# ################################

# load face detector
print('Prepare models')
palm_detector = PalmDetector(args.model_path)
# ################################


# open webcam
print('Open Camera Stream')
try:
    # check if stream is an integer (camera id)
    stream_path = int(args.input_stream)
except:
    # was not an integer, use it as it
    stream_path = args.input_path

stream = cv2.VideoCapture()
assert stream.open(stream_path)

chrono = Chronometer()

while True:
    with chrono:
        # get frame
        ok, frame = stream.read()
        if not ok: break

        # detect palms
        palms = palm_detector(frame)

        # render
        for palm in palms:
            palm.draw(frame)

    cv2.imshow('palms', frame)
    if cv2.waitKey(1) != -1: break

print('Close Camera Stream')
stream.release()

