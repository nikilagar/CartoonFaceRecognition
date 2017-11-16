import LandMarkPointsExtractor.extractAndSaveLandmarkConfigurations as LM
import sys
import glob
import os
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
import displayResults

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
imageArray = []

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

trainX = np.array(XFacesMeanDiff)
outX = np.array(XCaricatures)

model1 = Sequential()
model1.add(Dense(68, input_dim=68, activation='linear'))
model1.add(Dense(68, activation='sigmoid'))
model1.add(Dense(68, activation='linear'))
model1.compile(loss='mean_squared_error', optimizer='adam')
print("Model Created")

model1.fit(trainX, outX, epochs=1000)

trainY = np.array(YFacesMeanDiff)
outY = np.array(YCaricatures)

model2 = Sequential()
model2.add(Dense(68, input_dim=68, activation='linear'))
model2.add(Dense(68, activation='sigmoid'))
model2.add(Dense(68, activation='linear'))
model2.compile(loss='mean_squared_error', optimizer='adam')
print("Model2 Created")

model2.fit(trainY, outY, epochs=1000)


displayResults.showCaricature(model1, model2, meanImageCoordinates)

