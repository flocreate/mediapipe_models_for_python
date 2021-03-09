import numpy as np, cv2
import os
import json
import argparse as ap
import requests
# ################################
from src.chronometer import Chronometer
from src.detection.face import FaceDetector
from src.detection.face_mesh import FaceMeshDetector
from src.detection.iris import IrisDetector
# ################################

parser = ap.ArgumentParser()
parser.add_argument('input_img_path', help='image file to process')
parser.add_argument('output_img_path', help='image file to generate')
parser.add_argument('-m', '--model_folder', help='tflite model file',
    default=os.path.join('data', 'models'))
args = parser.parse_args()
# ################################


chrono = Chronometer().start()

with chrono['prepare models']:
    # load face detector
    face_detector = FaceDetector(os.path.join(args.model_folder, 'face_detection_front.tflite'))
    face_mesh_detector = FaceMeshDetector(os.path.join(args.model_folder, 'face_landmark.tflite'))
    iris_detector = IrisDetector(os.path.join(args.model_folder, 'iris_landmark.tflite'))

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

with chrono['detect faces']:
    # detect faces
    faces = face_detector(frame)
    print('nb faces: %d' % len(faces))

with chrono['detect mesh faces']:
    # detect face meshes
    face_meshes = [
        face_mesh_detector(frame, face)
        for face in faces
    ]

with chrono['detect iris']:
    # detect face meshes
    irises = [
        iris_detector(frame, face_mesh)
        for face_mesh in face_meshes
    ]

# for iid, (left, right) in enumerate(irises):
#     eyes = np.hstack((left.bgr_crop, right.bgr_crop))
#     cv2.imwrite('eyes_%d.png' % iid, eyes)

with chrono['render']:
    f0 = frame.copy()
    f1 = frame.copy()
    f2 = frame.copy()

    for face, face_mesh, (iris_l, iris_r) in zip(faces, face_meshes, irises):
        face.draw(f0)
        face_mesh.draw(f1)
        iris_l.draw(f2)
        iris_r.draw(f2)

    result = np.hstack((f0, f1, f2))

with chrono['save result']:
    cv2.imwrite(args.output_img_path, result)

chrono.stop()
print(json.dumps(chrono.as_dict(), indent=4))
