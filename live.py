import numpy as np, cv2
import os
# ################################
from src.chronometer import Chronometer
from src.detector.face_detector import FaceDetector
from src.detector.face_mesh_detector import FaceMeshDetector
from src.detector.iris_detector import IrisDetector
# ################################


# load face detector
print('Prepare models')
face_detector = FaceDetector(os.path.join(
    'data', 'models', 'face_detection_front.tflite'))
face_mesh_detector = FaceMeshDetector(os.path.join(
    'data', 'models', 'face_landmark.tflite'))
iris_detector = IrisDetector(os.path.join(
    'data', 'models', 'iris_landmark.tflite'))
# ################################


# open webcam
print('Open Camera Stream')
stream = cv2.VideoCapture()
assert stream.open(0)

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


    cv2.imshow('shadow', frame)
    if cv2.waitKey(1) != -1: break

print('Close Camera Stream')
stream.release()

