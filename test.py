import numpy as np, cv2
import os
import json
from glob import glob
import random
# ################################
from src.chronometer import Chronometer
from src.detection.face import FaceDetector
from src.detection.face_mesh import FaceMeshDetector
from src.detection.iris import IrisDetector
# ################################

chrono = Chronometer().start()

with chrono['prepare models']:
    # load face detector
    face_detector = FaceDetector(os.path.join(
        'data', 'models', 'face_detection_front.tflite'))
    face_mesh_detector = FaceMeshDetector(os.path.join(
        'data', 'models', 'face_landmark.tflite'))
    iris_detector = IrisDetector(os.path.join(
        'data', 'models', 'iris_landmark.tflite'))

with chrono['load image']:
    # load frame to process
    files       = glob('/home/ftaralle/Images/celeba_1024x1024/*.jpg')
    file        = random.choice(files)
    print('Processing:', os.path.basename(file))
    frame       = cv2.imread(file)

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

for iid, (left, right) in enumerate(irises):
    eyes = np.hstack((left.bgr_crop, right.bgr_crop))
    cv2.imwrite('eyes_%d.png' % iid, eyes)

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
    cv2.imwrite('tst.png', result)

chrono.stop()
print(json.dumps(chrono.as_dict(), indent=4))
