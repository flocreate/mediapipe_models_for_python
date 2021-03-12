# Mediapipe models for python

![result preview](https://i.ibb.co/N1d6FRD/test-result.png)

This repository implements some of the [Mediapipe](https://mediapipe.dev/) models.

Even though some of those incredible models are already [available in python](https://github.com/google/mediapipe) and other languages, my goal here is:

- propose a simple-to-use full-stack python implementation
- reduce dependancies as much as possible
- implement something cool cause I love it.

Model files need to be downloaded in `data/models` (see the dedicated readme).

## Supported Models

At the moment, this code supports 3 models:

- [Face Detection (front & back)](https://drive.google.com/file/d/1f39lSzU5Oq-j_OXgS67KfN5wNsoeAZ4V/view): detect faces and few landmarks.
- [Face Mesh](https://drive.google.com/file/d/1QvwWNfFoweGVjsXF3DXzcrCnz-mx-Lha/view): localize 468 landmarks on a detected face.
    - *Z* coordinate of landmark is not managed yet.
- [Iris](https://drive.google.com/file/d/1bsWbokp9AklH2ANjCfmjqEzzxO1CNbMu/view): localize 5+71 lanmarks on a detected eye.
- [Palm Detection](https://drive.google.com/file/d/1yiPfkhb4hSbXJZaSq9vDmhz24XVZmxpL/view)
