import pyrenn
import LandMarkPointsExtractor.extractAndSaveLandmarkConfigurations as LM
import sys
import glob
import os
import numpy as np

def getDiff(x,y):
    xx = [ [] for i in range(len(x))]
    for i in range(len(x)):
        for j in range(len(x[0])):
            xx[i].append(x[i][j] - y[j])
    return xx

faces_folder_path = sys.argv[1]
caricaturesFolderPath = sys.argv[2]

XFaces = []
XCaricatures = []
YFaces = []
YCaricatures = []

for f in glob.glob(os.path.join(faces_folder_path, "*.jp*g")):
    LandMarkPoints = LM.saveAndReturnLandMarkPoints(f);
    xCoordis = []
    yCoordis = []
    for point in LandMarkPoints:
        xCoordis.append(point[0])
        yCoordis.append(point[1])
    XFaces.append(xCoordis)
    YFaces.append(yCoordis)

for f in glob.glob(os.path.join(caricaturesFolderPath, "*.jp*g")):
    LandMarkPoints = LM.saveAndReturnLandMarkPoints(f);
    xCoordis = []
    yCoordis = []
    for point in LandMarkPoints:
        xCoordis.append(point[0])
        yCoordis.append(point[1])
    XCaricatures.append(xCoordis)
    YCaricatures.append(yCoordis)

meanImageCoordinates = LM.saveAndReturnLandMarkPoints("Dataset/averageFace.jpg")

XFacesMeanDiff = getDiff(XFaces, [ meanImageCoordinates[i][0] for i in range(len(meanImageCoordinates)) ])
YFacesMeanDiff = getDiff(YFaces, [ meanImageCoordinates[i][1] for i in range(len(meanImageCoordinates)) ])

for i in range(len(XCaricatures)):
    for j in range(len(XCaricatures[0])):
        XCaricatures[i][j] = XCaricatures[i][j] - XFaces[i][j]
        YCaricatures[i][j] = YCaricatures[i][j] - YFaces[i][j]

print("XFacesMeand DIff:\n")
for i in XFacesMeanDiff:
    print(i)

print("YFacesMeand DIff:\n")
for i in YFacesMeanDiff:
    print(i)

print("XFacesCaricature DIff:\n")
for i in XCaricatures:
    print(i)

print("YFacesCaricature DIff:\n")
for i in YCaricatures:
    print(i)

net = pyrenn.CreateNN([68,68,68])
import pdb;
pdb.set_trace()
trainX = np.array(XFacesMeanDiff)
trainX = np.transpose(trainX)
outY = np.array(XCaricatures)
outY = np.transpose(outY)

pyrenn.train_LM(trainX, outY, net, verbose=True, k_max=1, E_stop=100 )

Y = pyrenn.NNOut(trainX, net)
print("Output OF NN")
for i in Y:
    print(i)