# Import the required packages
from scipy.spatial import distance as dist
from imutils.video import FileVideoStream
from imutils.video import VideoStream
from imutils import face_utils
import imutils
import numpy as np
import argparse
import time
import dlib
import cv2

# Construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument('-p', '--shape-predictor', required = True, help = 'path to facial landmark predictor')
ap.add_argument('-v', '--video', type = str, default = '', help = 'path to input video file')
args = vars(ap.parse_args())

"""
Define two constants, one for the threshold for the (max) jaw distance ratio below which we identify as a tilt
and another one for the number of consecutive frames where the ratio must be below the threshold to count as a tilt
"""
jaw_dist_ratio_thresh = 0.7
consec_frame_thresh = 5

# Initialize the consecutive frame counters and the total number of left/right tilts
lt_consec_frames = 0
lt_total = 0
rt_consec_frames = 0
rt_total = 0

# Initialize dlib's face detector and then create the facial landmark predictor
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

# Start the video stream thread
print("[INFO] starting video stream thread...")
vs = FileVideoStream(args["video"]).start()
fileStream = True
# vs = VideoStream(src=0).start()
# vs = VideoStream(usePiCamera=True).start()
# fileStream = False
time.sleep(1.0)

# Loop over frames from the video stream
while True:
    # If this is a file video stream, then we need to check if
    # there are any more frames left in the buffer to process
    if fileStream and not vs.more():
        break
    # Grab the frame from the threaded video file stream, resize
    # it, and convert it to grayscale
    frame = vs.read()
    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Detect faces in the grayscale frame
    rects = detector(gray, 0)

    # Loop over the face detections
    for rect in rects:
        # Determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy array
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        # Extract the leftmost, central, and rightmost coordinates of the jaw
        jaw_coords = [list(shape[0]), list(shape[8]), list(shape[16])]

        # Compute the horizontal distance b/w the centre and the leftmost/rightmost points
        ldist = abs(jaw_coords[0][0] - jaw_coords[1][0])
        rdist = abs(jaw_coords[2][0] - jaw_coords[1][0])

        # Check to see if the rdist/ldist ratio is below the jaw_dist_ratio
        # threshold, and if so, increment the left tilt frame counter
        if rdist/ldist < jaw_dist_ratio_thresh:
            lt_consec_frames += 1
        else:
            # If the left tilt has been observed over sufficient no. of consec. frames,
            # increment the total number left tilts
            if lt_consec_frames >= consec_frame_thresh:
                lt_total += 1
            # Reset the consec. left tilt frame counter
            lt_consec_frames = 0

        # Check to see if the ldist/rdist ratio is below the jaw_dist_ratio
        # threshold, and if so, increment the reft tilt frame counter
        if ldist/rdist < jaw_dist_ratio_thresh:
            rt_consec_frames += 1
        else:
            # If the right tilt has been observed over sufficient no. of consec. frames,
            # increment the total number right tilts
            if rt_consec_frames >= consec_frame_thresh:
                rt_total += 1
            # Reset the consec. right tilt frame counter
            rt_consec_frames = 0

        # Display the total number of left/right tilts on the o/p frame
        cv2.putText(frame, "Left-tilts: {}".format(lt_total), (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "Right-tilts: {:.2f}".format(rt_total), (300, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Show the frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
 
    # If the `q` key was pressed, break from the loop
    if key == ord("q"):
        break
# Do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()