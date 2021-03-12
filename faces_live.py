import numpy as np, cv2
import os
import argparse as ap
from itertools import chain
# ################################
from src.chronometer import Chronometer
from src.detection.face import FrontFaceDetector
from src.detection.face_mesh import FaceMeshDetector
from src.detection.iris import IrisDetector
# ################################


# load face detector
print('Prepare models')
face_detector = FrontFaceDetector(os.path.join(
    'data', 'models', 'face_detection_front.tflite'))
face_mesh_detector = FaceMeshDetector(os.path.join(
    'data', 'models', 'face_landmark.tflite'))
iris_detector = IrisDetector(os.path.join(
    'data', 'models', 'iris_landmark.tflite'))
# ################################

# get stream id from command line
parser = ap.ArgumentParser()
parser.add_argument('-i', '--input_stream', default='0')
args = parser.parse_args()
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

        # detect faces
        faces  = face_detector(frame)
        # detect meshes per valid face
        meshes = [
            face_mesh_detector(frame, face) if (face.score > 0.5) else None
            for face in faces
        ]
        # detect irises per valid mesh
        irises = [
            iris_detector(frame, mesh) if (mesh is not None) and (mesh.score > 0.5) else None
            for mesh in meshes
        ]

        # render
        f0 = frame.copy()
        f1 = frame.copy()
        f2 = frame.copy()

        for face in faces:
            face.draw(f0)
        
        for mesh in meshes:
            if mesh is not None:
                mesh.draw(f1)

        for iris in irises:
            if iris is not None:
                iris[0].draw(f2)
                iris[1].draw(f2)

        frame = np.hstack((f0, f1, f2))

        raw = [iris.bgr_crop for iris in chain(*[
            i for i in irises if i is not None])
            if iris is not None]
        if len(raw):
            raw   = np.hstack(raw)
            w     = max(frame.shape[1], raw.shape[1])
            frame = cv2.copyMakeBorder(frame, 0, 0, 0, w-frame.shape[1], cv2.BORDER_CONSTANT, 0)
            raw   = cv2.copyMakeBorder(raw, 0, 0, 0, w-raw.shape[1], cv2.BORDER_CONSTANT, 0)
            frame = np.vstack((frame, raw))

    cv2.imshow('shadow', frame)
    if cv2.waitKey(1) != -1: break

print('Close Camera Stream')
stream.release()

