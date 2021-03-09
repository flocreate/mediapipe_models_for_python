import numpy as np, cv2
import os
import json
from glob import glob
import argparse as ap
import requests
# ################################
from src.chronometer import Chronometer
from src.detection.palm import PalmDetector
# ################################


parser = ap.ArgumentParser()
parser.add_argument('input_img_path', help='image file to process')
parser.add_argument('output_img_path', help='image file to generate')
parser.add_argument('-m', '--model_path', help='tflite model file',
    default=os.path.join('data', 'models', 'palm_detection.tflite'))
args = parser.parse_args()
# ################################


chrono = Chronometer().start()

with chrono['prepare model']:
    assert os.path.exists(args.model_path), 'model not found: %s' % args.model_path
    palm_detector = PalmDetector(args.model_path)

with chrono['load image']:
    if not args.input_img_path.startswith('http'):
        # load image from file
        assert os.path.exists(args.input_img_path), 'input image not found: %s' % args.input_img_path
        frame = cv2.imread(args.input_img_path)
        assert frame is not None, 'invalid image file'
    else:
        # load image from url
        try:
            resp = requests.get(args.input_img_path, stream=True)
            assert resp.status_code == 200, '(%d) %r' % (resp.status_code, resp.json())
        except Exception as what:
            raise Exception('error downloading image: %s' % what)
        frame = np.asarray(bytearray(resp.raw.read()), dtype=np.uint8)
        frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)

with chrono['detect palms']:
    palms = palm_detector(frame)
    print('nb palms: %d' % len(palms))

with chrono['render']:
    for palm in palms:
        palm.draw(frame)

with chrono['save result']:
    cv2.imwrite(args.output_img_path, frame)

chrono.stop()
print(json.dumps(chrono.as_dict(), indent=4))
