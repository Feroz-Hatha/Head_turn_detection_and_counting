# Head Left/Right Turn Identification and Counting
This project involves identifying and counting left/right head turns of a human in a video stream using DLib library's pre-trained facial landmark detector. 

Facial landmarks are used to identify the following salient regions of the face:
- Eyes
- Eyebrows
- Nose
- Mouth
- Jawline

Detecting facial landmarks is a two step process:
1. Localize the face in the image
2. Detect the key facial structures on the face ROI

The pre-trained facial landmark detector inside the dlib library is used to estimate the location of 68 (x, y)-coordinates that map to facial structures on the face.

The indexes of the 68 coordinates can be visualized on the image below:

![This is an image]()
