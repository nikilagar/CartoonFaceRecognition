import cv2
import numpy as np
import dlib
from skimage import io

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks (1).dat")

def saveAndReturnLandMarkPoints(srcFile,destFile=None):
    points = []
    img = io.imread(srcFile)
    dets = detector(img, 1)
    outFile = None
    if(destFile is not None):
        outFile = open(destFile, "w")
    print("Number of faces detected: {}".format(len(dets)))
    for k, d in enumerate(dets):
        print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
            k, d.left(), d.top(), d.right(), d.bottom()))
        # Get the landmarks/parts for the face in box d.
        shape = predictor(img, d)
        for i in shape.parts():
            if(outFile is not None):
                outFile.write("" + str(i.x) + " " + str(i.y) + "\n")
            points.append([i.x,i.y])
    if(outFile is not None):
        outFile.close()
    return points




def writeLandMarkPointsToImage(srcFileNdestFile):
    for srcFile,destFile in srcFileNdestFile:
        img = io.imread(srcFile)
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
            # win.add_overlay(shape)
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
        # cv2.imwrite(destFile, img)
