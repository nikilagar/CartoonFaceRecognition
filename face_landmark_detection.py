import sys
import os
import dlib
import glob
from skimage import io
import cv2
import numpy as np

if len(sys.argv) != 3:
    print(
        "Give the path to the trained shape predictor model as the first "
        "argument and then the directory containing the facial images.\n"
        "For example, if you are in the python_examples folder then "
        "execute this program by running:\n"
        "    ./face_landmark_detection.py shape_predictor_68_face_landmarks.dat ../examples/faces\n"
        "You can download a trained facial shape predictor from:\n"
        "    http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
    exit()

predictor_path = sys.argv[1]
faces_folder_path = sys.argv[2]

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)
win = dlib.image_window()

for f in glob.glob(os.path.join(faces_folder_path, "*.jp*g")):
    print("Processing file: {}".format(f))
    img = io.imread(f)

    win.clear_overlay()
    win.set_image(img)

    # Ask the detector to find the bounding boxes of each face. The 1 in the
    # second argument indicates that we should upsample the image 1 time. This
    # will make everything bigger and allow us to detect more faces.
    dets = detector(img, 1)
    print("Number of faces detected: {}".format(len(dets)))
    for k, d in enumerate(dets):
        print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
            k, d.left(), d.top(), d.right(), d.bottom()))
        # Get the landmarks/parts for the face in box d.
        shape = predictor(img, d)
        print("Part 0: {}, Part 1: {} ...".format(shape.part(0),
                                                  shape.part(1)))
        # Draw the face landmarks on the screen.
        win.add_overlay(shape)
        #cv2.polylines(img,shape.)
        arr = np.array(shape.parts())
        finalFeaturePoints = []
        differentFeatures = [17, 22, 27, 36, 42, 48, 61, 68]
        pre = 0
        for i in range(8):
            listOfPoints = []
            for j in range(pre, differentFeatures[i] ):
                listOfPoints.append( [arr[j].x, arr[j].y ])
            pre = j + 1
            finalFeaturePoints.append(listOfPoints)
        for i in range(8):
            arr = np.array(finalFeaturePoints[i])
            cv2.polylines(img,np.int32([arr]),0,(0,255,0))
        # cv2.imshow("Face",img)
        # cv2.waitKey(0)

    win.add_overlay(dets)
    dlib.hit_enter_to_continue()